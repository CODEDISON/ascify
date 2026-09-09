/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include "AscifyAction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#if LLVM_VERSION_MAJOR >= 13
#include "llvm/Support/SHA256.h"
#endif
#if LLVM_VERSION_MAJOR < 17
#include "clang/Basic/TargetInfo.h"
#endif
#include "LLVMCompat.h"
#include "ArgParse.h"
#include "LocalHeader.h"
#include "ImplicitCudaHeaders.h"
#include "CUDA2DPP.h"
#include "StringUtils.h"

using namespace ascify;

const std::string sDPP = "DPP";
const std::string s_string_literal = "[string literal]";
// Matchers' names
const StringRef sCudaLaunchKernel = "cudaLaunchKernel";
const StringRef sCudaGlobalScalarDoubleParam = "cudaGlobalScalarDoubleParam";
const StringRef sCudaDefaultDim3 = "cudaDefaultDim3";
const StringRef sProvenGlobalAtomicCall = "provenGlobalAtomicCall";
const StringRef sCanonicalWarpAddReduction = "canonicalWarpAddReduction";
const StringRef sCanonicalBinaryReducer = "canonicalBinaryReducer";

namespace {

bool isRecognizedNvidiaSampleHelperCuda(llvm::StringRef path) {
  if (path.empty() || llvm::sys::path::filename(path) != "helper_cuda.h")
    return false;
  const auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError)
    return false;
  const llvm::StringRef contents = (*bufferOrError)->getBuffer();
  // These markers jointly identify NVIDIA cuda-samples' helper_cuda surface
  // and the exact two macro contracts admitted below. A basename match alone
  // is deliberately insufficient: a project-local same-name header must not
  // acquire sample-helper semantics.
  return contents.contains("#ifndef COMMON_HELPER_CUDA_H_") &&
         contents.contains("#define COMMON_HELPER_CUDA_H_") &&
         contents.contains(
             "#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)") &&
         contents.contains(
             "#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)") &&
         contents.contains("template <typename T>\nvoid check(T result") &&
         contents.contains("inline void __getLastCudaError(");
}

enum FrozenNvidiaSampleHelperFileRole : unsigned {
  FrozenNvidiaSampleHelperNone = 0,
  FrozenOfficialNvidiaSampleHelperCuda = 1U << 0,
  FrozenOfficialNvidiaSampleHelperString = 1U << 1,
  FrozenOfficialNvidiaSampleHelperFunctions = 1U << 2,
  FrozenOfficialNvidiaSampleHelperImage = 1U << 3,
  FrozenOfficialNvidiaSampleHelperTimer = 1U << 4,
};

bool sha256Equals(llvm::StringRef contents, llvm::StringRef expectedHex) {
#if LLVM_VERSION_MAJOR >= 13
  if (expectedHex.size() != 64)
    return false;
  llvm::SHA256 digest;
  digest.update(contents);
  const auto bytes = digest.final();
  if (bytes.size() != 32)
    return false;
  const char digits[] = "0123456789abcdef";
  for (size_t index = 0; index < bytes.size(); ++index) {
    const uint8_t byte =
        static_cast<uint8_t>(static_cast<unsigned char>(bytes[index]));
    if (expectedHex[index * 2] != digits[byte >> 4] ||
        expectedHex[index * 2 + 1] != digits[byte & 0x0f])
      return false;
  }
  return true;
#else
  // Support/SHA256.h is unavailable before LLVM 13. Keep the rest of Ascify
  // buildable, but disable this compatibility proof rather than weakening the
  // frozen-source identity check.
  (void)contents;
  (void)expectedHex;
  return false;
#endif
}

bool isFrozenOfficialNvidiaSampleHelperFunctionsPath(llvm::StringRef path) {
  if (path.empty() ||
      llvm::sys::path::filename(path) != "helper_functions.h")
    return false;
  const auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError)
    return false;
  const llvm::StringRef contents = (*bufferOrError)->getBuffer();
  return contents.size() == 2358 &&
         sha256Equals(
             contents,
             "3fdcd18e41ffc2a9c88ade3595384e9cd05a2d84f80b86a2d5982035ca79c426");
}

unsigned frozenNvidiaSampleHelperFileRole(
    clang::SourceManager &sourceManager,
    clang::SourceLocation location) {
  const clang::SourceLocation spelling = sourceManager.getSpellingLoc(location);
  if (spelling.isInvalid())
    return FrozenNvidiaSampleHelperNone;
  const clang::FileID file = sourceManager.getFileID(spelling);
  if (file.isInvalid())
    return FrozenNvidiaSampleHelperNone;
  const llvm::StringRef filename = llvm::sys::path::filename(
      sourceManager.getFilename(spelling));
  if (filename != "helper_cuda.h" && filename != "helper_string.h" &&
      filename != "helper_functions.h" && filename != "helper_image.h" &&
      filename != "helper_timer.h")
    return FrozenNvidiaSampleHelperNone;
  bool invalidBuffer = false;
  const llvm::StringRef contents =
      sourceManager.getBufferData(file, &invalidBuffer);
  if (invalidBuffer)
    return FrozenNvidiaSampleHelperNone;
  if (filename == "helper_cuda.h") {
    // NVIDIA cuda-samples commit b7c5481c556c3fe98db060207ecaa41a4b9a9abc
    if (contents.size() == 28177 &&
        sha256Equals(
            contents,
            "997f9ac1f8e5f8e5f45f8b11eebab5b89305dee7430b90654bafe62283cffee1"))
      return FrozenOfficialNvidiaSampleHelperCuda;
  } else if (filename == "helper_string.h") {
    if (contents.size() == 15079 &&
        sha256Equals(
            contents,
            "26e988c97fb3d77d498e384c685177ed7966e41d5d58ebc9b7d3d696859f5e57"))
      return FrozenOfficialNvidiaSampleHelperString;
  } else if (filename == "helper_functions.h") {
    if (contents.size() == 2358 &&
        sha256Equals(
            contents,
            "3fdcd18e41ffc2a9c88ade3595384e9cd05a2d84f80b86a2d5982035ca79c426"))
      return FrozenOfficialNvidiaSampleHelperFunctions;
  } else if (filename == "helper_image.h") {
    if (contents.size() == 28739 &&
        sha256Equals(
            contents,
            "bc1fe7921bafad278ffa2e4bc8a99c18825208b9f5f47842a7cf7e86cae8b3f1"))
      return FrozenOfficialNvidiaSampleHelperImage;
  } else if (filename == "helper_timer.h") {
    if (contents.size() == 16060 &&
        sha256Equals(
            contents,
            "c48552a7c7b7a5840fcfbc176bfb5a19b501fdc56796b64a63e78e39ab547078"))
      return FrozenOfficialNvidiaSampleHelperTimer;
  }
  return FrozenNvidiaSampleHelperNone;
}

bool isFrozenHelperFunctionsProviderPolicyMacro(llvm::StringRef name) {
  return name == "EXIT_WAIVED" || name == "MAX";
}

const char *const NvidiaSampleHelperObservableMacros[] = {
    "COMMON_HELPER_CUDA_H_", "EXIT_WAIVED", "checkCudaErrors",
    "getLastCudaError", "printLastCudaError", "MAX"};

bool isNvidiaSampleHelperObservableMacro(llvm::StringRef name) {
  return std::find_if(
             std::begin(NvidiaSampleHelperObservableMacros),
             std::end(NvidiaSampleHelperObservableMacros),
             [&](const char *observable) { return name == observable; }) !=
         std::end(NvidiaSampleHelperObservableMacros);
}

bool directPragmaMacroStackTarget(
    llvm::StringRef line, size_t operationPosition,
    llvm::StringRef operation, std::string &target) {
  size_t cursor = operationPosition + operation.size();
  const auto skipHorizontalWhitespace = [&]() {
    while (cursor < line.size() &&
           (line[cursor] == ' ' || line[cursor] == '\t' ||
            line[cursor] == '\f' || line[cursor] == '\v'))
      ++cursor;
  };
  skipHorizontalWhitespace();
  if (cursor >= line.size() || line[cursor] != '(')
    return false;
  ++cursor;
  skipHorizontalWhitespace();
  if (cursor >= line.size() || line[cursor] != '"')
    return false;
  ++cursor;
  const size_t targetBegin = cursor;
  while (cursor < line.size() && line[cursor] != '"') {
    if (line[cursor] == '\\' ||
        !(line[cursor] == '_' ||
          (line[cursor] >= 'a' && line[cursor] <= 'z') ||
          (line[cursor] >= 'A' && line[cursor] <= 'Z') ||
          (cursor != targetBegin && line[cursor] >= '0' &&
           line[cursor] <= '9')))
      return false;
    ++cursor;
  }
  if (cursor == targetBegin || cursor >= line.size())
    return false;
  target = line.slice(targetBegin, cursor).str();
  ++cursor;
  skipHorizontalWhitespace();
  return cursor < line.size() && line[cursor] == ')';
}

bool isAscifyCudaCompatReservedMacro(llvm::StringRef name);

bool pragmaLineHasUnsafeMacroStackOperation(
    llvm::StringRef line, std::string &operation,
    std::string &target) {
  bool inString = false;
  bool inCharacter = false;
  bool escaped = false;
  for (size_t cursor = 0; cursor < line.size();) {
    const char current = line[cursor];
    if (inString || inCharacter) {
      if (escaped) {
        escaped = false;
      } else if (current == '\\') {
        escaped = true;
      } else if ((inString && current == '"') ||
                 (inCharacter && current == '\'')) {
        inString = false;
        inCharacter = false;
      }
      ++cursor;
      continue;
    }
    if (current == '"') {
      inString = true;
      ++cursor;
      continue;
    }
    if (current == '\'') {
      inCharacter = true;
      ++cursor;
      continue;
    }
    if (current == '/' && cursor + 1 < line.size() &&
        line[cursor + 1] == '/')
      return false;
    const bool identifierStart =
        current == '_' || (current >= 'a' && current <= 'z') ||
        (current >= 'A' && current <= 'Z');
    if (!identifierStart) {
      ++cursor;
      continue;
    }
    const size_t begin = cursor++;
    while (cursor < line.size() &&
           (line[cursor] == '_' ||
            (line[cursor] >= 'a' && line[cursor] <= 'z') ||
            (line[cursor] >= 'A' && line[cursor] <= 'Z') ||
            (line[cursor] >= '0' && line[cursor] <= '9')))
      ++cursor;
    const llvm::StringRef identifier = line.slice(begin, cursor);
    if (identifier != "push_macro" && identifier != "pop_macro")
      continue;
    operation = identifier.str();
    std::string directTarget;
    if (!directPragmaMacroStackTarget(
            line, begin, identifier, directTarget)) {
      target.clear();
      return true;
    }
    if (isNvidiaSampleHelperObservableMacro(directTarget) ||
        isAscifyCudaCompatReservedMacro(directTarget)) {
      target = directTarget;
      return true;
    }
  }
  return false;
}

bool pragmaLineContainsObservableIdentifier(
    llvm::StringRef line, std::string &observed) {
  bool inString = false;
  bool inCharacter = false;
  bool escaped = false;
  for (size_t cursor = 0; cursor < line.size();) {
    const char current = line[cursor];
    if (inString || inCharacter) {
      if (escaped) {
        escaped = false;
      } else if (current == '\\') {
        escaped = true;
      } else if ((inString && current == '"') ||
                 (inCharacter && current == '\'')) {
        inString = false;
        inCharacter = false;
      }
      ++cursor;
      continue;
    }
    if (current == '"') {
      inString = true;
      ++cursor;
      continue;
    }
    if (current == '\'') {
      inCharacter = true;
      ++cursor;
      continue;
    }
    if (current == '/' && cursor + 1 < line.size() &&
        line[cursor + 1] == '/')
      return false;
    const bool identifierStart =
        current == '_' || (current >= 'a' && current <= 'z') ||
        (current >= 'A' && current <= 'Z');
    if (!identifierStart) {
      ++cursor;
      continue;
    }
    const size_t begin = cursor++;
    while (cursor < line.size() &&
           (line[cursor] == '_' ||
            (line[cursor] >= 'a' && line[cursor] <= 'z') ||
            (line[cursor] >= 'A' && line[cursor] <= 'Z') ||
            (line[cursor] >= '0' && line[cursor] <= '9')))
      ++cursor;
    const llvm::StringRef identifier = line.slice(begin, cursor);
    if (isNvidiaSampleHelperObservableMacro(identifier)) {
      observed = identifier.str();
      return true;
    }
  }
  return false;
}

struct DirectPragmaToken {
  enum class Kind { Identifier, Number, String, Punctuation };
  Kind kind;
  std::string spelling;
};

std::vector<DirectPragmaToken> directPragmaTokens(llvm::StringRef line) {
  std::vector<DirectPragmaToken> tokens;
  for (size_t cursor = 0; cursor < line.size();) {
    const char current = line[cursor];
    if (current == ' ' || current == '\t' || current == '\f' ||
        current == '\v') {
      ++cursor;
      continue;
    }
    if (current == '/' && cursor + 1 < line.size() &&
        line[cursor + 1] == '/')
      break;
    const bool identifierStart =
        current == '_' || (current >= 'a' && current <= 'z') ||
        (current >= 'A' && current <= 'Z');
    if (identifierStart) {
      const size_t begin = cursor++;
      while (cursor < line.size() &&
             (line[cursor] == '_' ||
              (line[cursor] >= 'a' && line[cursor] <= 'z') ||
              (line[cursor] >= 'A' && line[cursor] <= 'Z') ||
              (line[cursor] >= '0' && line[cursor] <= '9')))
        ++cursor;
      tokens.push_back(
          {DirectPragmaToken::Kind::Identifier,
           line.slice(begin, cursor).str()});
      continue;
    }
    if (current >= '0' && current <= '9') {
      const size_t begin = cursor++;
      while (cursor < line.size() && line[cursor] >= '0' &&
             line[cursor] <= '9')
        ++cursor;
      tokens.push_back(
          {DirectPragmaToken::Kind::Number,
           line.slice(begin, cursor).str()});
      continue;
    }
    if (current == '"') {
      const size_t begin = cursor++;
      bool escaped = false;
      while (cursor < line.size()) {
        const char stringCharacter = line[cursor++];
        if (escaped) {
          escaped = false;
          continue;
        }
        if (stringCharacter == '\\') {
          escaped = true;
          continue;
        }
        if (stringCharacter == '"')
          break;
      }
      tokens.push_back(
          {DirectPragmaToken::Kind::String,
           line.slice(begin, cursor).str()});
      continue;
    }
    tokens.push_back(
        {DirectPragmaToken::Kind::Punctuation,
         std::string(1, current)});
    ++cursor;
  }
  return tokens;
}

bool isDirectlyAllowedHashPragma(
    llvm::StringRef line, std::string &operation,
    bool initiallyTrustedSystemFile) {
  const std::vector<DirectPragmaToken> tokens = directPragmaTokens(line);
  size_t pragma = 0;
  while (pragma < tokens.size() &&
         !(tokens[pragma].kind == DirectPragmaToken::Kind::Identifier &&
           tokens[pragma].spelling == "pragma"))
    ++pragma;
  if (pragma == tokens.size() || ++pragma == tokens.size()) {
    operation = "<unparsed>";
    return false;
  }
  const auto identifierAt = [&](size_t index, llvm::StringRef spelling) {
    return index < tokens.size() &&
           tokens[index].kind == DirectPragmaToken::Kind::Identifier &&
           tokens[index].spelling == spelling;
  };
  const auto punctuationAt = [&](size_t index, llvm::StringRef spelling) {
    return index < tokens.size() &&
           tokens[index].kind == DirectPragmaToken::Kind::Punctuation &&
           tokens[index].spelling == spelling;
  };
  operation = tokens[pragma].spelling;
  const size_t remaining = tokens.size() - pragma;
  if (remaining == 1 &&
      (identifierAt(pragma, "once") ||
       identifierAt(pragma, "nounroll") ||
       identifierAt(pragma, "nv_exec_check_disable") ||
       identifierAt(pragma, "hd_warning_disable")))
    return true;
  if (identifierAt(pragma, "unroll") &&
      (remaining == 1 ||
       (remaining == 2 &&
        tokens[pragma + 1].kind == DirectPragmaToken::Kind::Number)))
    return true;
  if (remaining == 2 &&
      ((identifierAt(pragma, "GCC") ||
        identifierAt(pragma, "clang")) &&
       identifierAt(pragma + 1, "system_header")))
    return true;
  if (remaining >= 3 &&
      (identifierAt(pragma, "GCC") ||
       identifierAt(pragma, "clang")) &&
      identifierAt(pragma + 1, "diagnostic")) {
    if (remaining == 3 &&
        (identifierAt(pragma + 2, "push") ||
         identifierAt(pragma + 2, "pop")))
      return true;
    if (remaining == 4 &&
        (identifierAt(pragma + 2, "ignored") ||
         identifierAt(pragma + 2, "warning") ||
         identifierAt(pragma + 2, "error") ||
         identifierAt(pragma + 2, "fatal")) &&
        tokens[pragma + 3].kind == DirectPragmaToken::Kind::String)
      return true;
  }
  // Clang's CUDA texture header uses a trailing semicolon on its literal
  // macro-stack pragmas. This does not change the target-name proof performed
  // before this syntax admission. Accept exactly one optional semicolon.
  if ((remaining == 4 ||
       (remaining == 5 && punctuationAt(pragma + 4, ";"))) &&
      (identifierAt(pragma, "push_macro") ||
       identifierAt(pragma, "pop_macro")) &&
      punctuationAt(pragma + 1, "(") &&
      tokens[pragma + 2].kind == DirectPragmaToken::Kind::String &&
      punctuationAt(pragma + 3, ")"))
    return true;
  // libstdc++ brackets portable standard declarations with default symbol
  // visibility. Only headers trusted on their first entry may use this form;
  // a user pragma or a later system_header promotion cannot change the ABI of
  // the compatibility declarations inserted by the helper transaction.
  if (initiallyTrustedSystemFile &&
      identifierAt(pragma, "GCC") && identifierAt(pragma + 1, "visibility") &&
      ((remaining == 3 && identifierAt(pragma + 2, "pop")) ||
       (remaining == 6 && identifierAt(pragma + 2, "push") &&
        punctuationAt(pragma + 3, "(") &&
        identifierAt(pragma + 4, "default") &&
        punctuationAt(pragma + 5, ")"))))
    return true;
  return false;
}

bool isAscifyCudaCompatReservedMacro(llvm::StringRef name) {
  static const char *const reserved[] = {
      "ASCIFY_ASCIFY_CUDA_COMPAT_HPP",
      "ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85",
      "ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3",
      "ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS",
      "ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS",
      "ASCIFY_ALIGN",
      "ASCIFY_FORCEINLINE",
      "ASCIFY_GLOBAL",
      "ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY",
      "ASCIFY_DEFINE_GLOBAL_ATOMIC_CAS",
      "ASCIFY_DEFINE_GLOBAL_ATOMIC_UNSIGNED",
      "ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_BINARY",
      "ASCIFY_DEFINE_UNAVAILABLE_GLOBAL_ATOMIC_CAS",
      "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS",
      "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR",
      "ASCIFY_TEST_CONTROLLABLE_EXIT_REGISTRATION",
      "ascify",
      "sampleCheckCudaErrors",
      "sampleGetLastCudaError",
      "sampleFindCudaDevice",
  };
  return std::find_if(
             std::begin(reserved), std::end(reserved),
             [&](const char *candidate) { return name == candidate; }) !=
         std::end(reserved);
}

bool isAscifyCudaCompatPublishedMacro(llvm::StringRef name) {
  static const char *const published[] = {
      "ASCIFY_ASCIFY_CUDA_COMPAT_HPP",
      "ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85",
      "ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3",
      "ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS",
      "ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS",
      "ASCIFY_ALIGN",
      "ASCIFY_FORCEINLINE",
      "ASCIFY_GLOBAL",
      "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS",
      "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR",
  };
  return std::find_if(
             std::begin(published), std::end(published),
             [&](const char *candidate) { return name == candidate; }) !=
         std::end(published);
}

const char *const AscifyCudaCompatIdentifiers[] = {
#include "AscifyCudaCompatIdentifiers.inc"
};

bool isAscifyCudaCompatIdentifier(llvm::StringRef name) {
  return std::find_if(
             std::begin(AscifyCudaCompatIdentifiers),
             std::end(AscifyCudaCompatIdentifiers),
             [&](const char *candidate) { return name == candidate; }) !=
         std::end(AscifyCudaCompatIdentifiers);
}

bool activeFrozenHelperFunctionsProviderMacroBodyMatches(
    llvm::StringRef name, const clang::MacroInfo &macroInfo,
    clang::Preprocessor &preprocessor) {
  if (name == "EXIT_WAIVED") {
    if (macroInfo.isFunctionLike() || macroInfo.getNumTokens() != 1)
      return false;
    const clang::Token &token = *macroInfo.tokens_begin();
    return token.is(clang::tok::numeric_constant) &&
           preprocessor.getSpelling(token) == "2";
  }
  if (name != "MAX" || !macroInfo.isFunctionLike() ||
      macroInfo.getNumParams() != 2)
    return false;
  auto parameter = macroInfo.param_begin();
  if (parameter == macroInfo.param_end() || *parameter == nullptr)
    return false;
  const std::string left = (*parameter++)->getName().str();
  if (parameter == macroInfo.param_end() || *parameter == nullptr)
    return false;
  const std::string right = (*parameter)->getName().str();
  std::vector<std::string> actual;
  actual.reserve(macroInfo.getNumTokens());
  for (const clang::Token &token : macroInfo.tokens())
    actual.push_back(preprocessor.getSpelling(token));
  const std::vector<std::string> expected = {
      "(", "(", left, ">", right, ")", "?", left, ":", right, ")"};
  return actual == expected;
}

bool activeFrozenHelperCudaMaxBodyMatches(
    const clang::MacroInfo &macroInfo,
    clang::Preprocessor &preprocessor) {
  if (!macroInfo.isFunctionLike() || macroInfo.getNumParams() != 2)
    return false;
  auto parameter = macroInfo.param_begin();
  if (parameter == macroInfo.param_end() || *parameter == nullptr)
    return false;
  const std::string left = (*parameter++)->getName().str();
  if (parameter == macroInfo.param_end() || *parameter == nullptr)
    return false;
  const std::string right = (*parameter)->getName().str();
  std::vector<std::string> actual;
  actual.reserve(macroInfo.getNumTokens());
  for (const clang::Token &token : macroInfo.tokens())
    actual.push_back(preprocessor.getSpelling(token));
  const std::vector<std::string> expected = {
      "(", left, ">", right, "?", left, ":", right, ")"};
  return actual == expected;
}

bool locationComesFromFrozenNvidiaSampleHelperFile(
    clang::SourceManager &sourceManager,
    clang::SourceLocation location,
    unsigned admittedRoles) {
  const clang::SourceLocation spelling = sourceManager.getSpellingLoc(location);
  if (spelling.isInvalid())
    return false;
  return (frozenNvidiaSampleHelperFileRole(sourceManager, spelling) &
          admittedRoles) != 0;
}

bool locationComesFromRecognizedNvidiaSampleHelper(
    clang::SourceManager &sourceManager, clang::SourceLocation location) {
  const clang::SourceLocation spelling =
      sourceManager.getSpellingLoc(location);
  if (spelling.isInvalid())
    return false;
  return isRecognizedNvidiaSampleHelperCuda(
      sourceManager.getFilename(spelling));
}

bool isExactAscifyCudaCompatPath(llvm::StringRef path) {
  if (path.empty() ||
      llvm::sys::path::filename(path) != "ascify_cuda_compat.hpp")
    return false;
  const auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError)
    return false;
  const llvm::StringRef contents = (*bufferOrError)->getBuffer();
  if (contents.size() != 43500)
    return false;
#if LLVM_VERSION_MAJOR >= 13
  return sha256Equals(
      contents,
      "c1f87bd416aa4f389171593bd5e47894999d05407e5935d3151fbcd2aa6438c9");
#else
  return contents.contains("#ifndef ASCIFY_ASCIFY_CUDA_COMPAT_HPP") &&
         contents.contains("inline void sampleCheckCudaErrors(") &&
         contents.contains("inline int sampleFindCudaDevice(");
#endif
}

bool locationComesFromAscifyCudaCompat(
    clang::SourceManager &sourceManager, clang::SourceLocation location) {
  const clang::SourceLocation spelling =
      sourceManager.getSpellingLoc(location);
  if (spelling.isInvalid())
    return false;
  const clang::FileID file = sourceManager.getFileID(spelling);
  if (file.isInvalid())
    return false;
  const clang::FileEntry *entry = sourceManager.getFileEntryForID(file);
  if (entry == nullptr || sourceManager.isFileOverridden(entry))
    return false;
  const llvm::StringRef path = sourceManager.getFilename(spelling);
  if (path.empty() ||
      llvm::sys::path::filename(path) != "ascify_cuda_compat.hpp")
    return false;
  bool invalidBuffer = false;
  const llvm::StringRef contents =
      sourceManager.getBufferData(file, &invalidBuffer);
  if (invalidBuffer || contents.size() != 43500)
    return false;
#if LLVM_VERSION_MAJOR >= 13
  return sha256Equals(
      contents,
      "c1f87bd416aa4f389171593bd5e47894999d05407e5935d3151fbcd2aa6438c9");
#else
  return contents.contains("#ifndef ASCIFY_ASCIFY_CUDA_COMPAT_HPP") &&
         contents.contains("inline void sampleCheckCudaErrors(") &&
         contents.contains("inline int sampleFindCudaDevice(");
#endif
}

bool activeNvidiaSampleHelperMacroBodyMatches(
    llvm::StringRef name, const clang::MacroInfo &macroInfo,
    clang::Preprocessor &preprocessor) {
  if (!macroInfo.isFunctionLike() || macroInfo.getNumParams() != 1)
    return false;
  const clang::IdentifierInfo *parameter = *macroInfo.param_begin();
  if (parameter == nullptr)
    return false;
  const std::string parameterName = parameter->getName().str();
  std::vector<std::string> actual;
  actual.reserve(macroInfo.getNumTokens());
  for (const clang::Token &token : macroInfo.tokens())
    actual.push_back(preprocessor.getSpelling(token));

  std::vector<std::string> expected;
  if (name == "checkCudaErrors") {
    expected = {"check", "(", "(", parameterName, ")", ",", "#",
                parameterName, ",", "__FILE__", ",", "__LINE__", ")"};
  } else if (name == "getLastCudaError") {
    expected = {"__getLastCudaError", "(", parameterName, ",",
                "__FILE__", ",", "__LINE__", ")"};
  } else {
    return false;
  }
  return actual == expected;
}

static bool allFunctionRedeclarationsComeFromInitiallyTrustedSystemFiles(
    const clang::FunctionDecl *callee,
    clang::SourceManager &sourceManager,
    const std::set<unsigned> &trustedSystemFileIds);

bool isAdmittedCudaRuntimeStatusCall(
    clang::SourceManager &sourceManager,
    const clang::Expr *expression,
    const std::set<unsigned> &trustedSystemFileIds) {
  if (expression == nullptr)
    return false;
  expression = expression->IgnoreParenImpCasts();
  const auto *call = llvm::dyn_cast<clang::CallExpr>(expression);
  if (call == nullptr)
    return false;
  const clang::FunctionDecl *callee = call->getDirectCallee();
  if (callee == nullptr || callee->getIdentifier() == nullptr)
    return false;
  const llvm::StringRef name = callee->getName();
  // This is a return-domain allowlist, not a general CUDA API allowlist.
  // Every admitted name is implemented by ascify_cuda_compat.hpp with an
  // aclError result. Occupancy, Driver APIs, library statuses, and arbitrary
  // integer-valued calls stay outside the set even when wrapped by NVIDIA's
  // generic checkCudaErrors macro.
  static const char *const admittedNames[] = {
      "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaMemcpyAsync",
      "cudaMemset", "cudaMemsetAsync", "cudaMallocHost",
      "cudaFreeHost", "cudaGetDevice", "cudaSetDevice",
      "cudaGetDeviceCount", "cudaDeviceGetAttribute",
      "cudaDeviceSynchronize", "cudaDeviceReset",
      "cudaFuncGetAttributes", "cudaFuncSetAttribute",
      "cudaStreamCreate", "cudaStreamCreateWithFlags",
      "cudaStreamDestroy", "cudaStreamSynchronize",
      "cudaEventCreate", "cudaEventDestroy", "cudaEventRecord",
      "cudaEventSynchronize", "cudaEventElapsedTime",
      "cudaGetLastError", "cudaPeekAtLastError",
  };
  if (std::find_if(std::begin(admittedNames), std::end(admittedNames),
                   [&](const char *admitted) { return name == admitted; }) ==
      std::end(admittedNames))
    return false;
  const auto mapped = CUDA_RENAMES_MAP().find(name);
  if (mapped == CUDA_RENAMES_MAP().end() ||
      mapped->second.apiType != API_RUNTIME ||
      !mapped->second.dppName.starts_with("ascify::"))
    return false;
  const clang::SourceLocation declarationLocation = callee->getLocation();
  return declarationLocation.isValid() &&
         !sourceManager.isWrittenInMainFile(declarationLocation) &&
         allFunctionRedeclarationsComeFromInitiallyTrustedSystemFiles(
             callee, sourceManager, trustedSystemFileIds) &&
         !locationComesFromRecognizedNvidiaSampleHelper(
             sourceManager, declarationLocation);
}

const char *const RowwiseSimdReservedMacros[] = {
    "ascify",
    "target",
    "dav_c310",
    "rowwise_simd_v1",
    "SimdTryResult",
    "HybridTryResult",
    "NotHandled",
    "RowwiseSimdFacadeV1",
    "RowwiseHybridFacadeV1",
    "HybridRecipeDescriptor",
    "HybridStageDescriptor",
    "DispatchRegisteredHybrid",
    "TrySoftmaxSimd",
    "TryRmsNormSimd",
    "TryLayerNormSimd",
    "TrySoftmaxHybrid",
    "TryRmsNormHybrid",
    "TryLayerNormHybrid",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_HYBRID_REGISTRY_V1_HPP_",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_SELECTORS_V1_HPP_",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_H_",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_TYPES_READY",
    "ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION",
    "ASCIFY_ASCIFY_CUDA_COMPAT_HPP",
    "ASCIFY_ALIGN",
    "ASCIFY_FORCEINLINE",
    "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS",
    "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR",
    "ascify950_softmax_reg_recompute_launch_v1",
    "ascify950_rmsnorm_reg_cached_launch_v1",
    "ascify950_rmsnorm_reg_plain_rowbatch_launch_v1",
    "ascify950_layernorm_reg_cached_launch_v1",
    "handled",
    "status",
    "alignas",
    "alignof",
    "asm",
    "auto",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "char16_t",
    "char32_t",
    "class",
    "const",
    "constexpr",
    "const_cast",
    "continue",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "dynamic_cast",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "public",
    "nullptr",
    "operator",
    "private",
    "protected",
    "register",
    "reinterpret_cast",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "static_cast",
    "struct",
    "switch",
    "template",
    "this",
    "thread_local",
    "throw",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "wchar_t",
    "while",
    "exp",
    "expf",
    "__expf",
    "rsqrt",
    "rsqrtf",
    "__frsqrt_rn",
    "__fdividef",
    "ascify_target_direct_load_tag",
    "ascify_target_direct_store_tag",
    "ascify_target_adapter_owner_type",
    "ascify_target_storage_type",
    "ascify_target_compute_type",
    "ascify_target_store_is_affine",
    "ascify_target_data",
    "ascify_target_row_stride",
    "ascify_target_weight",
};

bool isRowwiseSimdReservedMacro(llvm::StringRef name) {
  return std::find_if(
             std::begin(RowwiseSimdReservedMacros),
             std::end(RowwiseSimdReservedMacros),
             [&](const char *reserved) { return name == reserved; }) !=
         std::end(RowwiseSimdReservedMacros);
}

bool isRowwiseSimdPublishedHeaderMacro(llvm::StringRef name) {
  return name == "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_" ||
         name == "ASCIFY_TARGET_DAV_C310_ROWWISE_HYBRID_REGISTRY_V1_HPP_" ||
         name == "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_SELECTORS_V1_HPP_" ||
         name == "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_H_" ||
         name == "ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION" ||
         name == "ASCIFY_ASCIFY_CUDA_COMPAT_HPP" ||
         name == "ASCIFY_ALIGN" || name == "ASCIFY_FORCEINLINE";
}

bool isRowwiseSimdProtectedDeclarationName(llvm::StringRef name) {
  static const char *const names[] = {
      "ascify",
      "SimdTryResult",
      "HybridTryResult",
      "NotHandled",
      "RowwiseSimdFacadeV1",
      "RowwiseHybridFacadeV1",
      "HybridRecipeDescriptor",
      "HybridStageDescriptor",
      "DispatchRegisteredHybrid",
      "TrySoftmaxSimd",
      "TryRmsNormSimd",
      "TryLayerNormSimd",
      "TrySoftmaxHybrid",
      "TryRmsNormHybrid",
      "TryLayerNormHybrid",
      "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_",
      "ASCIFY_TARGET_DAV_C310_ROWWISE_HYBRID_REGISTRY_V1_HPP_",
      "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_SELECTORS_V1_HPP_",
      "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_H_",
      "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_TYPES_READY",
      "ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION",
      "ASCIFY_ASCIFY_CUDA_COMPAT_HPP",
      "ASCIFY_ALIGN",
      "ASCIFY_FORCEINLINE",
      "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS",
      "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR",
      "ascify_target_direct_load_tag",
      "ascify_target_direct_store_tag",
      "ascify_target_adapter_owner_type",
      "ascify_target_storage_type",
      "ascify_target_compute_type",
      "ascify_target_store_is_affine",
      "ascify_target_data",
      "ascify_target_row_stride",
      "ascify_target_weight",
      "ascify950_softmax_reg_recompute_launch_v1",
      "ascify950_rmsnorm_reg_cached_launch_v1",
      "ascify950_rmsnorm_reg_plain_rowbatch_launch_v1",
      "ascify950_layernorm_reg_cached_launch_v1",
  };
  return std::find_if(
             std::begin(names), std::end(names),
             [&](const char *protectedName) {
               return name == protectedName;
             }) != std::end(names);
}

struct RowwiseSimdConflictingDeclaration {
  clang::SourceLocation location;
  std::string name;

  explicit operator bool() const { return !name.empty(); }
};

RowwiseSimdConflictingDeclaration
findRowwiseSimdConflictingDeclaration(clang::ASTContext &context) {
  class Visitor : public clang::RecursiveASTVisitor<Visitor> {
  public:
    explicit Visitor(clang::SourceManager &sourceManager)
        : sourceManager(sourceManager) {}

    bool VisitFileScopeAsmDecl(clang::FileScopeAsmDecl *declaration) {
      if (conflict || declaration == nullptr)
        return !conflict;
      return rejectUserAsm(declaration->getLocation(), "GNU asm");
    }

    bool VisitGCCAsmStmt(clang::GCCAsmStmt *statement) {
      if (conflict || statement == nullptr)
        return !conflict;
      return rejectUserAsm(statement->getAsmLoc(), "GNU asm");
    }

    bool VisitMSAsmStmt(clang::MSAsmStmt *statement) {
      if (conflict || statement == nullptr)
        return !conflict;
      return rejectUserAsm(statement->getAsmLoc(), "Microsoft asm");
    }

    bool VisitNamespaceDecl(clang::NamespaceDecl *declaration) {
      if (conflict || declaration == nullptr || declaration->isImplicit())
        return !conflict;
      const std::string qualified =
          declaration->getQualifiedNameAsString();
      const llvm::StringRef qualifiedRef(qualified);
      if (qualifiedRef == "ascify::target::dav_c310" ||
          qualifiedRef.starts_with("ascify::target::dav_c310::")) {
        conflict = {
            declaration->getLocation(), qualified};
        return false;
      }
      return true;
    }

    bool VisitNamedDecl(clang::NamedDecl *declaration) {
      if (conflict || declaration == nullptr ||
          declaration->getIdentifier() == nullptr)
        return !conflict;
      const llvm::StringRef name = declaration->getName();
      if (const auto *assemblerLabel =
              declaration->getAttr<clang::AsmLabelAttr>()) {
        llvm::StringRef label = assemblerLabel->getLabel();
        if (!label.empty() && label.front() == '\1')
          label = label.drop_front();
        if (isRowwiseSimdProtectedDeclarationName(label)) {
          conflict = {
              declaration->getLocation(), label.str()};
          return false;
        }
      }
      if (isRowwiseSimdProtectedDeclarationName(name)) {
        conflict = {
            declaration->getLocation(),
            declaration->getQualifiedNameAsString()};
        return false;
      }
      if (name != "RowwiseSimdFacadeV1" &&
          name != "RowwiseHybridFacadeV1" &&
          name != "TrySoftmaxSimd" && name != "TryRmsNormSimd" &&
          name != "TryLayerNormSimd" &&
          name != "TrySoftmaxHybrid" && name != "TryRmsNormHybrid" &&
          name != "TryLayerNormHybrid")
        return true;
      const auto *owner = llvm::dyn_cast_or_null<clang::NamedDecl>(
          declaration->getDeclContext());
      if (owner == nullptr)
        return true;
      const std::string qualifiedOwner =
          owner->getQualifiedNameAsString();
      const llvm::StringRef qualifiedOwnerRef(qualifiedOwner);
      if (qualifiedOwnerRef == "ascify::target::dav_c310" ||
          qualifiedOwnerRef.starts_with("ascify::target::dav_c310::")) {
        conflict = {
            declaration->getLocation(),
            declaration->getQualifiedNameAsString()};
        return false;
      }
      return true;
    }

  private:
    bool rejectUserAsm(
        clang::SourceLocation location, llvm::StringRef spelling) {
      const clang::SourceLocation expansion =
          sourceManager.getExpansionLoc(location);
      if (expansion.isInvalid() ||
          sourceManager.isInSystemHeader(expansion))
        return true;
      conflict = {location, spelling.str()};
      return false;
    }

  public:

    RowwiseSimdConflictingDeclaration conflict;
    clang::SourceManager &sourceManager;
  } visitor(context.getSourceManager());

  visitor.TraverseDecl(context.getTranslationUnitDecl());
  return visitor.conflict;
}

struct RawRowwiseSimdConflict {
  std::string macro;
  std::string declaration;
};

RawRowwiseSimdConflict rawFileRowwiseSimdConflict(
    clang::CompilerInstance &compiler, clang::FileID file,
    bool inspectAsmTokens = false) {
  clang::SourceManager &sourceManager =
      compiler.getSourceManager();
  clang::Preprocessor &preprocessor =
      compiler.getPreprocessor();
  bool invalid = false;
  const llvm::StringRef buffer =
      sourceManager.getBufferData(file, &invalid);
  if (invalid)
    return {};
  clang::Lexer lexer(
      sourceManager.getLocForStartOfFile(file),
      preprocessor.getLangOpts(), buffer.begin(), buffer.begin(),
      buffer.end());

  enum class DirectiveState { None, Hash, Define };
  DirectiveState state = DirectiveState::None;
  RawRowwiseSimdConflict conflict;
  clang::Token token;
  const auto tokenName = [](const clang::Token &candidate) {
    if (candidate.is(clang::tok::raw_identifier))
      return candidate.getRawIdentifier();
    return candidate.getIdentifierInfo() == nullptr
               ? llvm::StringRef()
               : candidate.getIdentifierInfo()->getName();
  };
  lexer.LexFromRawLexer(token);
  while (token.isNot(clang::tok::eof)) {
    if (token.isAtStartOfLine())
      state = DirectiveState::None;
    if (state == DirectiveState::None &&
        token.is(clang::tok::hash) &&
        token.isAtStartOfLine()) {
      state = DirectiveState::Hash;
    } else if (state == DirectiveState::Hash &&
               tokenName(token) == "define") {
      state = DirectiveState::Define;
    } else if (state == DirectiveState::Define &&
               !tokenName(token).empty()) {
      const llvm::StringRef name = tokenName(token);
      if (isRowwiseSimdReservedMacro(name))
        conflict.macro = name.str();
      state = DirectiveState::None;
    } else if (state != DirectiveState::None) {
      state = DirectiveState::None;
    }
    const llvm::StringRef name = tokenName(token);
    if (inspectAsmTokens && conflict.declaration.empty() &&
        (name == "asm" || name == "__asm" || name == "__asm__"))
      conflict.declaration = "GNU asm";
    if (conflict.declaration.empty() &&
        !name.empty() && isRowwiseSimdProtectedDeclarationName(name)) {
      conflict.declaration = name.str();
    }
    if (!conflict.macro.empty() && !conflict.declaration.empty())
      break;
    lexer.LexFromRawLexer(token);
  }
  return conflict;
}

std::string findConventionalMainOuterGuard(
    clang::CompilerInstance &compiler) {
  clang::SourceManager &sourceManager = compiler.getSourceManager();
  bool invalid = false;
  const llvm::StringRef buffer = sourceManager.getBufferData(
      sourceManager.getMainFileID(), &invalid);
  if (invalid)
    return {};

  clang::Lexer lexer(
      sourceManager.getLocForStartOfFile(sourceManager.getMainFileID()),
      compiler.getPreprocessor().getLangOpts(), buffer.begin(),
      buffer.begin(), buffer.end());
  std::vector<std::string> spellings;
  std::vector<bool> startsOfLine;
  clang::Token token;
  lexer.LexFromRawLexer(token);
  while (token.isNot(clang::tok::eof) && spellings.size() < 6) {
    if (token.is(clang::tok::comment)) {
      lexer.LexFromRawLexer(token);
      continue;
    }
    bool spellingInvalid = false;
    const std::string spelling = clang::Lexer::getSpelling(
        token, sourceManager, compiler.getPreprocessor().getLangOpts(),
        &spellingInvalid);
    if (spellingInvalid)
      return {};
    spellings.push_back(spelling);
    startsOfLine.push_back(token.isAtStartOfLine());
    lexer.LexFromRawLexer(token);
  }
  if (spellings.size() != 6 || spellings[0] != "#" ||
      spellings[1] != "ifndef" || spellings[2].empty() ||
      spellings[3] != "#" || spellings[4] != "define" ||
      spellings[5] != spellings[2] || !startsOfLine[0] ||
      !startsOfLine[3]) {
    return {};
  }
  return spellings[2];
}

} // namespace

void AscifyAction::RewriteString(StringRef s, clang::SourceLocation start) {
  auto &SM = getCompilerInstance().getSourceManager();
  size_t begin = 0;
  while ((begin = s.find("cu", begin)) != StringRef::npos) {
    const size_t end = s.find_first_of(" ", begin + 4);
    StringRef name = s.slice(begin, end);
    const auto found = CUDA_RENAMES_MAP().find(name);
    if (found != CUDA_RENAMES_MAP().end()) {
      StringRef repName = found->second.dppName;
      dppCounter counter = {s_string_literal, ConvTypes::CONV_LITERAL, ApiTypes::API_RUNTIME, found->second.supportDegree};
      Statistics::current().incrementCounter(counter, name.str());
      if (!Statistics::isUnsupported(counter)) {
        clang::SourceLocation sl = start.getLocWithOffset(begin + 1);
        ct::Replacement Rep(SM, sl, name.size(), repName.str());
        clang::FullSourceLoc fullSL(sl, SM);
        insertReplacement(Rep, fullSL);
      }
    }
    if (end == StringRef::npos) break;
    begin = end + 1;
  }
}

// TODO
clang::SourceLocation AscifyAction::GetSubstrLocation(const std::string &str, const clang::SourceRange &sr) {
  clang::SourceLocation sl(sr.getBegin());
  return sl;
}

void AscifyAction::FindAndReplace(StringRef name,
                                  clang::SourceLocation sl,
                                  const std::map<StringRef, dppCounter> &repMap,
                                  bool bReplace) {
  const auto found = repMap.find(name);
  if (found == repMap.end()) {
    // So it's an identifier, but not CUDA? Boring.
    return;
  }
  Statistics::current().incrementCounter(found->second, name.str());
  clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();

  // Warn about the deprecated identifier in CUDA but hipify it.
  if (Statistics::isCudaDeprecated(found->second)) {
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "'%0' is deprecated in CUDA.");
    DE.Report(sl, ID) << found->first;
  }

  // TODO: Similar to hipify, add statistics analysis

    // Warn about the unsupported identifier.
  if (Statistics::isUnsupported(found->second)) {
    std::string sWarn = sDPP;
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "'%0' is unsupported in '%1'.");
    DE.Report(sl, ID) << found->first << sWarn;
    return;
  }
  if (!bReplace) {
    return;
  }
  StringRef repName = found->second.dppName;
  auto &SM = getCompilerInstance().getSourceManager();
  ct::Replacement Rep(SM, sl, name.size(), repName.str());
  clang::FullSourceLoc fullSL(sl, SM);
  insertReplacement(Rep, fullSL);
}

namespace {

clang::SourceRange getReadRange(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  clang::SourceLocation begin = exprRange.getBegin();
  clang::SourceLocation end = exprRange.getEnd();
  bool beginSafe = !SM.isMacroBodyExpansion(begin) || clang::Lexer::isAtStartOfMacroExpansion(begin, SM, clang::LangOptions{});
  bool endSafe = !SM.isMacroBodyExpansion(end) || clang::Lexer::isAtEndOfMacroExpansion(end, SM, clang::LangOptions{});
  if (beginSafe && endSafe) {
    return {SM.getFileLoc(begin), SM.getFileLoc(end)};
  } else {
    return {SM.getSpellingLoc(begin), SM.getSpellingLoc(end)};
  }
}

clang::SourceRange getWriteRange(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  clang::SourceLocation begin = exprRange.getBegin();
  clang::SourceLocation end = exprRange.getEnd();
  // If the range is contained within a macro, update the macro definition.
  // Otherwise, use the file location and hope for the best.
  if (!SM.isMacroBodyExpansion(begin) || !SM.isMacroBodyExpansion(end)) {
    return {SM.getFileLoc(begin), SM.getFileLoc(end)};
  }
  return {SM.getSpellingLoc(begin), SM.getSpellingLoc(end)};
}

StringRef readSourceText(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  return clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(getReadRange(SM, exprRange)), SM, clang::LangOptions(), nullptr);
}

static const clang::Token *previousSignificantToken(
    const std::deque<clang::Token> &tokens) {
  if (tokens.size() < 2)
    return nullptr;
  for (std::size_t i = tokens.size() - 1; i > 0;) {
    --i;
    if (tokens[i].getKind() != 0)
      return &tokens[i];
  }
  return nullptr;
}

static bool isUnqualifiedFloatMathName(llvm::StringRef name) {
  return name == "exp" || name == "log" || name == "sqrt" || name == "rsqrt";
}

static bool requiresCudaCompatHeader(llvm::StringRef name) {
  return name == "__global__" ||
         name == "cudaMalloc" ||
         name == "cudaFree" ||
         name == "cudaMemcpy" ||
         name == "cudaMemcpyAsync" ||
         name == "cudaMemset" ||
         name == "cudaMemsetAsync" ||
         name == "cudaMallocHost" ||
         name == "cudaFreeHost" ||
         name == "cudaGetDevice" ||
         name == "cudaSetDevice" ||
         name == "cudaGetDeviceCount" ||
         name == "cudaDeviceSynchronize" ||
         name == "cudaDeviceReset" ||
         name == "cudaGetLastError" ||
         name == "cudaGetErrorString" ||
         name == "cudaDeviceGetAttribute" ||
         name == "cudaStreamCreate" ||
         name == "cudaStreamCreateWithFlags" ||
         name == "cudaStreamDefault" ||
         name == "cudaStreamNonBlocking" ||
         name == "cudaStreamDestroy" ||
         name == "cudaStreamSynchronize" ||
         name == "cudaEventCreate" ||
         name == "cudaEventDestroy" ||
         name == "cudaEventRecord" ||
         name == "cudaEventSynchronize" ||
         name == "cudaEventElapsedTime" ||
         name == "cudaDevAttrWarpSize" ||
         name == "cudaDevAttrMaxThreadsPerMultiProcessor" ||
         name == "cudaDevAttrMaxThreadsPerBlock" ||
         name == "cudaDevAttrMaxSharedMemoryPerBlock" ||
         name == "cudaDevAttrMaxSharedMemoryPerBlockOptin" ||
         name == "cudaPeekAtLastError" ||
         name == "cudaOccupancyMaxActiveBlocksPerMultiprocessor" ||
         name == "__shfl_sync" ||
         name == "__shfl_up_sync" ||
         name == "__shfl_down_sync" ||
         name == "__shfl_xor_sync" ||
         name == "__syncthreads" ||
         name == "__syncthreads_and" ||
         name == "__syncthreads_or" ||
         name == "__syncthreads_count" ||
         name == "__syncwarp" ||
         name == "__threadfence" ||
         name == "__threadfence_block" ||
         name == "__threadfence_system" ||
         name == "__expf" ||
         name == "__logf" ||
         name == "__log2f" ||
         name == "__log10f" ||
         name == "__fdividef" ||
         name == "__frsqrt_rn" ||
         name == "__fsqrt_rn" ||
         name == "exp" ||
         name == "log" ||
         name == "sqrt" ||
         name == "rsqrt" ||
         name == "__forceinline__" ||
         name == "__align__" ||
         name == "nv_bfloat16" ||
         name == "nv_bfloat16_2" ||
         name == "cudaFuncAttributes" ||
         name == "cudaFuncAttribute" ||
         name == "cudaFuncAttributeMaxDynamicSharedMemorySize" ||
         name == "cudaFuncGetAttributes" ||
         name == "cudaFuncSetAttribute";
}

static const clang::Expr *stripParenAndImplicitCasts(const clang::Expr *expr) {
  if (expr == nullptr)
    return nullptr;
  if (const auto *defaultArgument =
          llvm::dyn_cast<clang::CXXDefaultArgExpr>(expr))
    expr = defaultArgument->getExpr();
  return expr->IgnoreParenImpCasts();
}

static const clang::VarDecl *referencedVariable(const clang::Expr *expr) {
  expr = stripParenAndImplicitCasts(expr);
  const auto *reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr);
  return reference == nullptr
             ? nullptr
             : llvm::dyn_cast<clang::VarDecl>(reference->getDecl());
}

static bool evaluatesToNonNegative(const clang::Expr *expr,
                                   uint64_t expected,
                                   clang::ASTContext &context) {
  if (expr == nullptr)
    return false;
  expr = stripParenAndImplicitCasts(expr);
  clang::Expr::EvalResult result;
  if (!expr->EvaluateAsInt(result, context) || !result.Val.isInt())
    return false;
  const llvm::APSInt &value = result.Val.getInt();
  if ((value.isSigned() && value.isNegative()) ||
      value.getBitWidth() > 64)
    return false;
  return value.getZExtValue() == expected;
}

static bool evaluatesToFullWarpMask(const clang::Expr *expr,
                                    clang::ASTContext &context) {
  if (expr == nullptr || !expr->getType()->isIntegerType() ||
      context.getTypeSize(expr->getType()) != 32)
    return false;

  expr = stripParenAndImplicitCasts(expr);
  clang::Expr::EvalResult result;
  if (!expr->EvaluateAsInt(result, context) || !result.Val.isInt())
    return false;
  const llvm::APSInt &value = result.Val.getInt();
  const llvm::APInt normalized = llvm::APInt(value).zextOrTrunc(32);
  return normalized == llvm::APInt(32, UINT32_MAX);
}

static bool isSupportedWarpAddType(clang::QualType type,
                                   clang::ASTContext &context) {
  if (type.isNull() || type->isReferenceType() ||
      type.isVolatileQualified())
    return false;

  type = type.getCanonicalType().getUnqualifiedType();
  const auto *builtin = type->getAs<clang::BuiltinType>();
  if (builtin == nullptr || context.getTypeSize(type) != 32)
    return false;
  return builtin->getKind() == clang::BuiltinType::Float ||
         builtin->getKind() == clang::BuiltinType::Int ||
         builtin->getKind() == clang::BuiltinType::UInt;
}

static const clang::Stmt *singleLoopUpdate(const clang::Stmt *body) {
  const auto *compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(body);
  if (compound == nullptr)
    return body;
  if (compound->size() != 1)
    return nullptr;
  return *compound->body_begin();
}

static const clang::Stmt *directFunctionBodyStatement(
    const clang::ForStmt *loop,
    const clang::FunctionDecl *function) {
  const auto *body =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(function->getBody());
  if (body == nullptr)
    return nullptr;
  for (const clang::Stmt *statement : body->body()) {
    const clang::Stmt *unwrapped = statement;
    while (const auto *attributed =
               llvm::dyn_cast<clang::AttributedStmt>(unwrapped))
      unwrapped = attributed->getSubStmt();
    if (unwrapped == loop)
      return statement;
  }
  return nullptr;
}

static bool isOnlyUnrollPragmasAndWhitespace(llvm::StringRef source) {
  bool sawUnrollPragma = false;
  while (!source.empty()) {
    const std::pair<llvm::StringRef, llvm::StringRef> line =
        source.split('\n');
    source = line.second;
    const llvm::StringRef trimmed = line.first.trim();
    if (trimmed.empty())
      continue;
    if (trimmed != "#pragma unroll")
      return false;
    sawUnrollPragma = true;
  }
  return sawUnrollPragma;
}

// When Clang wraps a loop in AttributedStmt for `#pragma unroll`, own the
// complete physical pragma line as well as the loop.  Starting at column one
// prevents leaving a dangling `#pragma` fragment if an LLVM version reports
// the attribute location at a token after '#'.
static clang::SourceLocation canonicalWarpReductionReplacementBegin(
    const clang::Stmt *ownedStatement,
    const clang::ForStmt *loop,
    clang::SourceManager &sourceManager,
    const clang::LangOptions &langOptions) {
  const clang::SourceLocation loopBegin =
      sourceManager.getFileLoc(loop->getBeginLoc());
  if (ownedStatement == loop)
    return loopBegin;

  const clang::SourceLocation ownedBegin =
      sourceManager.getFileLoc(ownedStatement->getBeginLoc());
  if (ownedBegin.isInvalid() || loopBegin.isInvalid())
    return clang::SourceLocation();
  const std::pair<clang::FileID, unsigned> decomposedOwned =
      sourceManager.getDecomposedLoc(ownedBegin);
  const std::pair<clang::FileID, unsigned> decomposedLoop =
      sourceManager.getDecomposedLoc(loopBegin);
  if (decomposedOwned.first != decomposedLoop.first ||
      decomposedOwned.second >= decomposedLoop.second)
    return clang::SourceLocation();

  const unsigned line = sourceManager.getSpellingLineNumber(ownedBegin);
  const clang::SourceLocation lineBegin =
      sourceManager.translateLineCol(decomposedOwned.first, line, 1);
  if (lineBegin.isInvalid())
    return clang::SourceLocation();
  const llvm::StringRef prefix = clang::Lexer::getSourceText(
      clang::CharSourceRange::getCharRange(lineBegin, loopBegin),
      sourceManager, langOptions);
  if (!isOnlyUnrollPragmasAndWhitespace(prefix))
    return clang::SourceLocation();
  return lineBegin;
}

static bool matchCanonicalWarpAddReduction(
    const clang::ForStmt *loop,
    clang::ASTContext &context,
    clang::SourceManager &sourceManager,
    const clang::VarDecl *&accumulator) {
  accumulator = nullptr;
  if (loop == nullptr || loop->getInit() == nullptr ||
      loop->getCond() == nullptr || loop->getInc() == nullptr ||
      loop->getBody() == nullptr)
    return false;

  const clang::SourceLocation begin = loop->getBeginLoc();
  const clang::SourceLocation end = loop->getEndLoc();
  if (begin.isInvalid() || end.isInvalid() || begin.isMacroID() ||
      end.isMacroID() || !sourceManager.isWrittenInMainFile(begin) ||
      !sourceManager.isWrittenInMainFile(end))
    return false;

  const auto *declaration =
      llvm::dyn_cast<clang::DeclStmt>(loop->getInit());
  if (declaration == nullptr || !declaration->isSingleDecl())
    return false;
  const auto *offset =
      llvm::dyn_cast<clang::VarDecl>(declaration->getSingleDecl());
  if (offset == nullptr || !offset->hasInit() ||
      offset->getStorageDuration() != clang::SD_Automatic ||
      !offset->getType()->isIntegerType() ||
      offset->getType().isVolatileQualified() ||
      context.getTypeSize(offset->getType()) != 32 ||
      !evaluatesToNonNegative(offset->getInit(), 16, context))
    return false;

  const auto *condition = llvm::dyn_cast<clang::BinaryOperator>(
      stripParenAndImplicitCasts(loop->getCond()));
  if (condition == nullptr || condition->getOpcode() != clang::BO_GT ||
      referencedVariable(condition->getLHS()) != offset ||
      !evaluatesToNonNegative(condition->getRHS(), 0, context))
    return false;

  const auto *increment = llvm::dyn_cast<clang::CompoundAssignOperator>(
      stripParenAndImplicitCasts(loop->getInc()));
  if (increment == nullptr || referencedVariable(increment->getLHS()) != offset)
    return false;
  if (increment->getOpcode() == clang::BO_DivAssign) {
    if (!evaluatesToNonNegative(increment->getRHS(), 2, context))
      return false;
  } else if (increment->getOpcode() == clang::BO_ShrAssign) {
    if (!evaluatesToNonNegative(increment->getRHS(), 1, context))
      return false;
  } else {
    return false;
  }

  const clang::Stmt *updateStatement = singleLoopUpdate(loop->getBody());
  const auto *updateExpression =
      llvm::dyn_cast_or_null<clang::Expr>(updateStatement);
  if (updateExpression == nullptr)
    return false;
  updateExpression = stripParenAndImplicitCasts(updateExpression);

  const clang::CallExpr *shuffle = nullptr;
  if (const auto *compoundAdd =
          llvm::dyn_cast<clang::CompoundAssignOperator>(updateExpression)) {
    if (compoundAdd->getOpcode() != clang::BO_AddAssign)
      return false;
    accumulator = referencedVariable(compoundAdd->getLHS());
    shuffle = llvm::dyn_cast_or_null<clang::CallExpr>(
        stripParenAndImplicitCasts(compoundAdd->getRHS()));
  } else if (const auto *assignment =
                 llvm::dyn_cast<clang::BinaryOperator>(updateExpression)) {
    if (assignment->getOpcode() != clang::BO_Assign)
      return false;
    accumulator = referencedVariable(assignment->getLHS());
    const auto *addition = llvm::dyn_cast_or_null<clang::BinaryOperator>(
        stripParenAndImplicitCasts(assignment->getRHS()));
    if (addition == nullptr || addition->getOpcode() != clang::BO_Add)
      return false;
    const clang::Expr *shuffleExpression = nullptr;
    if (referencedVariable(addition->getLHS()) == accumulator)
      shuffleExpression = addition->getRHS();
    else if (referencedVariable(addition->getRHS()) == accumulator)
      shuffleExpression = addition->getLHS();
    else
      return false;
    shuffle = llvm::dyn_cast_or_null<clang::CallExpr>(
        stripParenAndImplicitCasts(shuffleExpression));
  } else {
    return false;
  }

  if (accumulator == nullptr || accumulator == offset || shuffle == nullptr ||
      !isSupportedWarpAddType(accumulator->getType(), context))
    return false;

  const auto *function =
      llvm::dyn_cast<clang::FunctionDecl>(accumulator->getDeclContext());
  if (function == nullptr ||
      (!function->hasAttr<clang::CUDADeviceAttr>() &&
      !function->hasAttr<clang::CUDAGlobalAttr>()) ||
      function->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      function->getDeclContext()->isDependentContext())
    return false;

  const clang::Stmt *ownedStatement =
      directFunctionBodyStatement(loop, function);
  if (ownedStatement == nullptr)
    return false;

  // The transformation removes the entire for-init declaration.  Prove the
  // accumulator is a different declaration that precedes, and therefore
  // remains visible at, the replacement site.
  const clang::SourceLocation accumulatorLoc =
      sourceManager.getFileLoc(accumulator->getLocation());
  const clang::SourceLocation ownedBegin =
      sourceManager.getFileLoc(ownedStatement->getBeginLoc());
  if (accumulator->getLocation().isInvalid() ||
      accumulator->getLocation().isMacroID() ||
      accumulatorLoc.isInvalid() || ownedBegin.isInvalid())
    return false;
  const std::pair<clang::FileID, unsigned> decomposedAccumulator =
      sourceManager.getDecomposedLoc(accumulatorLoc);
  const std::pair<clang::FileID, unsigned> decomposedOwned =
      sourceManager.getDecomposedLoc(ownedBegin);
  if (decomposedAccumulator.first != decomposedOwned.first ||
      decomposedAccumulator.second >= decomposedOwned.second)
    return false;

  const clang::FunctionDecl *callee = shuffle->getDirectCallee();
  if (callee == nullptr ||
      callee->getQualifiedNameAsString() != "__shfl_xor_sync" ||
      callee->getLocation().isInvalid() ||
      !sourceManager.isInSystemHeader(callee->getLocation()) ||
      shuffle->getNumArgs() < 3 || shuffle->getNumArgs() > 4 ||
      !evaluatesToFullWarpMask(shuffle->getArg(0), context) ||
      referencedVariable(shuffle->getArg(1)) != accumulator ||
      referencedVariable(shuffle->getArg(2)) != offset)
    return false;

  const clang::Expr *width = nullptr;
  if (shuffle->getNumArgs() == 4) {
    width = shuffle->getArg(3);
  } else {
    if (callee->getNumParams() != 4 ||
        !callee->getParamDecl(3)->hasDefaultArg())
      return false;
    width = callee->getParamDecl(3)->getDefaultArg();
  }
  if (!evaluatesToNonNegative(width, 32, context))
    return false;
  return true;
}

enum class CanonicalReducerKind {
  None,
  Sum,
  Max,
  Min,
};

static const clang::ParmVarDecl *referencedParameter(
    const clang::Expr *expr) {
  expr = stripParenAndImplicitCasts(expr);
  const auto *reference =
      llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr);
  return reference == nullptr
             ? nullptr
             : llvm::dyn_cast<clang::ParmVarDecl>(reference->getDecl());
}

static clang::QualType canonicalReducerValueType(clang::QualType type) {
  if (type.isNull())
    return type;
  if (type->isReferenceType())
    type = type.getNonReferenceType();
  return type.getCanonicalType().getUnqualifiedType();
}

static bool referencesBothParameters(
    const clang::Expr *lhs,
    const clang::Expr *rhs,
    const clang::ParmVarDecl *first,
    const clang::ParmVarDecl *second) {
  const clang::ParmVarDecl *lhsParameter = referencedParameter(lhs);
  const clang::ParmVarDecl *rhsParameter = referencedParameter(rhs);
  return (lhsParameter == first && rhsParameter == second) ||
         (lhsParameter == second && rhsParameter == first);
}

static CanonicalReducerKind classifyCanonicalReducerReturn(
    const clang::Expr *expression,
    const clang::ParmVarDecl *first,
    const clang::ParmVarDecl *second) {
  expression = stripParenAndImplicitCasts(expression);
  if (const auto *addition =
          llvm::dyn_cast_or_null<clang::BinaryOperator>(expression)) {
    if (addition->getOpcode() == clang::BO_Add &&
        referencesBothParameters(
            addition->getLHS(), addition->getRHS(), first, second))
      return CanonicalReducerKind::Sum;
    return CanonicalReducerKind::None;
  }

  const auto *conditional =
      llvm::dyn_cast_or_null<clang::ConditionalOperator>(expression);
  if (conditional == nullptr)
    return CanonicalReducerKind::None;
  const auto *comparison = llvm::dyn_cast<clang::BinaryOperator>(
      stripParenAndImplicitCasts(conditional->getCond()));
  if (comparison == nullptr ||
      (comparison->getOpcode() != clang::BO_GT &&
       comparison->getOpcode() != clang::BO_LT) ||
      !referencesBothParameters(
          comparison->getLHS(), comparison->getRHS(), first, second))
    return CanonicalReducerKind::None;

  const clang::ParmVarDecl *conditionLhs =
      referencedParameter(comparison->getLHS());
  const clang::ParmVarDecl *conditionRhs =
      referencedParameter(comparison->getRHS());
  const clang::ParmVarDecl *trueValue =
      referencedParameter(conditional->getTrueExpr());
  const clang::ParmVarDecl *falseValue =
      referencedParameter(conditional->getFalseExpr());
  if (trueValue == nullptr || falseValue == nullptr)
    return CanonicalReducerKind::None;

  const bool selectsConditionLhs =
      trueValue == conditionLhs && falseValue == conditionRhs;
  const bool selectsConditionRhs =
      trueValue == conditionRhs && falseValue == conditionLhs;
  if (!selectsConditionLhs && !selectsConditionRhs)
    return CanonicalReducerKind::None;

  if (comparison->getOpcode() == clang::BO_GT)
    return selectsConditionLhs ? CanonicalReducerKind::Max
                               : CanonicalReducerKind::Min;
  return selectsConditionLhs ? CanonicalReducerKind::Min
                             : CanonicalReducerKind::Max;
}

static CanonicalReducerKind classifyCanonicalBinaryReducer(
    const clang::CXXRecordDecl *record,
    clang::ASTContext &context,
    clang::SourceManager &sourceManager) {
  if (record == nullptr || !record->isThisDeclarationADefinition() ||
      record->isImplicit() || record->isUnion() ||
      record->getIdentifier() == nullptr ||
      !record->getDeclContext()->isFileContext() ||
      record->getNumBases() != 0 ||
      record->getLocation().isInvalid() ||
      record->getLocation().isMacroID() ||
      !sourceManager.isWrittenInMainFile(record->getLocation()))
    return CanonicalReducerKind::None;

  const clang::ClassTemplateDecl *classTemplate =
      record->getDescribedClassTemplate();
  if (classTemplate == nullptr ||
      classTemplate->getTemplateParameters()->size() != 1)
    return CanonicalReducerKind::None;
  const auto *valueTypeParameter =
      llvm::dyn_cast<clang::TemplateTypeParmDecl>(
          classTemplate->getTemplateParameters()->getParam(0));
  if (valueTypeParameter == nullptr ||
      valueTypeParameter->isParameterPack() ||
      valueTypeParameter->getIdentifier() == nullptr)
    return CanonicalReducerKind::None;

  if (!record->lookup(
          &context.Idents.get("ascify_reduction_tag")).empty() ||
      !record->lookup(
          &context.Idents.get("ascify_reduction_value_type")).empty() ||
      !record->lookup(
          &context.Idents.get("ascify_reduction_owner_type")).empty())
    return CanonicalReducerKind::None;

  const clang::CXXMethodDecl *callOperator = nullptr;
  for (const clang::CXXMethodDecl *method : record->methods()) {
    if (method->isImplicit() ||
        method->getOverloadedOperator() != clang::OO_Call)
      continue;
    if (callOperator != nullptr)
      return CanonicalReducerKind::None;
    callOperator = method;
  }
  if (callOperator == nullptr || callOperator->isDeleted() ||
      callOperator->isVariadic() || !callOperator->isConst() ||
      callOperator->getNumParams() != 2 ||
      callOperator->getTemplatedKind() !=
          clang::FunctionDecl::TK_NonTemplate ||
      !callOperator->hasAttr<clang::CUDADeviceAttr>() ||
      !callOperator->hasBody())
    return CanonicalReducerKind::None;

  const clang::ParmVarDecl *first = callOperator->getParamDecl(0);
  const clang::ParmVarDecl *second = callOperator->getParamDecl(1);
  clang::QualType firstType =
      canonicalReducerValueType(first->getType());
  clang::QualType secondType =
      canonicalReducerValueType(second->getType());
  clang::QualType returnType =
      canonicalReducerValueType(callOperator->getReturnType());
  const clang::QualType templateValueType =
      context.getTypeDeclType(valueTypeParameter)
          .getCanonicalType().getUnqualifiedType();
  if (firstType.isNull() || secondType.isNull() || returnType.isNull() ||
      templateValueType.isNull() ||
      first->getType().getNonReferenceType().isVolatileQualified() ||
      second->getType().getNonReferenceType().isVolatileQualified() ||
      callOperator->getReturnType()->isReferenceType() ||
      !context.hasSameType(firstType, secondType) ||
      !context.hasSameType(firstType, returnType) ||
      !context.hasSameType(firstType, templateValueType))
    return CanonicalReducerKind::None;

  const auto *body =
      llvm::dyn_cast<clang::CompoundStmt>(callOperator->getBody());
  if (body == nullptr || body->size() != 1 ||
      body->getBeginLoc().isInvalid() || body->getEndLoc().isInvalid() ||
      body->getBeginLoc().isMacroID() || body->getEndLoc().isMacroID())
    return CanonicalReducerKind::None;
  const auto *returnStatement =
      llvm::dyn_cast<clang::ReturnStmt>(*body->body_begin());
  if (returnStatement == nullptr || returnStatement->getRetValue() == nullptr)
    return CanonicalReducerKind::None;

  return classifyCanonicalReducerReturn(
      returnStatement->getRetValue(), first, second);
}

static llvm::StringRef canonicalReducerTagName(CanonicalReducerKind kind) {
  switch (kind) {
  case CanonicalReducerKind::Sum:
    return "Sum";
  case CanonicalReducerKind::Max:
    return "Max";
  case CanonicalReducerKind::Min:
    return "Min";
  case CanonicalReducerKind::None:
    break;
  }
  return "";
}

enum class GlobalAtomicScalarKind {
  Unsupported,
  Int32,
  UInt32,
};

struct GlobalAtomicSpec {
  const char *cudaName;
  const char *wrapperName;
  unsigned argumentCount;
  bool unsignedOnly;
};

static const GlobalAtomicSpec GlobalAtomicSpecs[] = {
    {"atomicAdd", "atomic_add_global", 2, false},
    {"atomicSub", "atomic_sub_global", 2, false},
    {"atomicExch", "atomic_exch_global", 2, false},
    {"atomicMax", "atomic_max_global", 2, false},
    {"atomicMin", "atomic_min_global", 2, false},
    {"atomicInc", "atomic_inc_global", 2, true},
    {"atomicDec", "atomic_dec_global", 2, true},
    {"atomicCAS", "atomic_cas_global", 3, false},
    {"atomicAnd", "atomic_and_global", 2, false},
    {"atomicOr", "atomic_or_global", 2, false},
    {"atomicXor", "atomic_xor_global", 2, false},
};

static const GlobalAtomicSpec *findGlobalAtomicSpec(
    llvm::StringRef name) {
  for (const GlobalAtomicSpec &spec : GlobalAtomicSpecs) {
    if (name == spec.cudaName)
      return &spec;
  }
  return nullptr;
}

static GlobalAtomicScalarKind classifyGlobalAtomicScalar(
    clang::QualType type,
    clang::ASTContext &context) {
  if (type.isNull() || type->isReferenceType() ||
      type.isConstQualified() || type.isVolatileQualified())
    return GlobalAtomicScalarKind::Unsupported;
  type = type.getCanonicalType().getUnqualifiedType();
  if (!type->isIntegerType() || context.getTypeSize(type) != 32)
    return GlobalAtomicScalarKind::Unsupported;
  if (type->isSignedIntegerType())
    return GlobalAtomicScalarKind::Int32;
  if (type->isUnsignedIntegerType())
    return GlobalAtomicScalarKind::UInt32;
  return GlobalAtomicScalarKind::Unsupported;
}

static GlobalAtomicScalarKind classifyGlobalAtomicPointer(
    clang::QualType type,
    clang::ASTContext &context) {
  if (type.isNull())
    return GlobalAtomicScalarKind::Unsupported;
  type = type.getCanonicalType();
  if (!type->isPointerType())
    return GlobalAtomicScalarKind::Unsupported;
  return classifyGlobalAtomicScalar(type->getPointeeType(), context);
}

static const clang::FunctionDecl *enclosingNonLambdaFunction(
    const clang::Stmt *statement,
    clang::ASTContext &context) {
  const clang::Stmt *current = statement;
  std::set<const clang::Stmt *> active;
  while (current != nullptr && active.insert(current).second) {
    const auto parents = context.getParents(*current);
    const clang::Stmt *next = nullptr;
    for (const clang::DynTypedNode &parent : parents) {
      if (parent.get<clang::LambdaExpr>() != nullptr)
        return nullptr;
      if (const auto *function = parent.get<clang::FunctionDecl>())
        return function;
      if (next == nullptr)
        next = parent.get<clang::Stmt>();
    }
    current = next;
  }
  return nullptr;
}

static bool referencesCanonicalValue(const clang::Expr *expression,
                                     const clang::ValueDecl *expected) {
  expression = stripParenAndImplicitCasts(expression);
  const auto *reference =
      llvm::dyn_cast_or_null<clang::DeclRefExpr>(expression);
  return reference != nullptr && expected != nullptr &&
         reference->getDecl()->getCanonicalDecl() ==
             expected->getCanonicalDecl();
}

static bool isExactStringLiteral(const clang::Expr *expression,
                                 llvm::StringRef expected) {
  expression = stripParenAndImplicitCasts(expression);
  const auto *literal = llvm::dyn_cast_or_null<clang::StringLiteral>(expression);
  return literal != nullptr && literal->getString() == expected;
}

static bool hasExactNvidiaFindCudaDeviceSignature(
    const clang::FunctionDecl *function,
    clang::ASTContext &context) {
  if (function == nullptr || function->isVariadic() ||
      function->getNumParams() != 2 ||
      !context.hasSameType(function->getReturnType().getCanonicalType(),
                           context.IntTy) ||
      !context.hasSameType(
          function->getParamDecl(0)->getType().getCanonicalType(),
          context.IntTy))
    return false;

  clang::QualType outer =
      function->getParamDecl(1)->getType().getCanonicalType();
  if (outer.getLocalCVRQualifiers() != 0)
    return false;
  const auto *outerPointer = outer->getAs<clang::PointerType>();
  if (outerPointer == nullptr)
    return false;
  clang::QualType inner = outerPointer->getPointeeType();
  if (inner.getLocalCVRQualifiers() != 0)
    return false;
  const auto *innerPointer = inner->getAs<clang::PointerType>();
  if (innerPointer == nullptr)
    return false;
  const clang::QualType character = innerPointer->getPointeeType();
  return character.isConstQualified() && !character.isVolatileQualified() &&
         context.hasSameType(character.getUnqualifiedType(), context.CharTy);
}

static const clang::CallExpr *directCallNamed(
    const clang::Expr *expression,
    llvm::StringRef expectedName) {
  expression = stripParenAndImplicitCasts(expression);
  const auto *call = llvm::dyn_cast_or_null<clang::CallExpr>(expression);
  const clang::FunctionDecl *callee =
      call == nullptr ? nullptr : call->getDirectCallee();
  return callee != nullptr && callee->getIdentifier() != nullptr &&
                 callee->getName() == expectedName
             ? call
             : nullptr;
}

static const clang::CallExpr *directUnqualifiedSourceCallNamed(
    const clang::Expr *expression,
    llvm::StringRef expectedName,
    clang::SourceManager &sourceManager,
    const clang::LangOptions &languageOptions) {
  const clang::CallExpr *call = directCallNamed(expression, expectedName);
  if (call == nullptr)
    return nullptr;
  const auto *reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
      stripParenAndImplicitCasts(call->getCallee()));
  const clang::FunctionDecl *callee = call->getDirectCallee();
  if (reference == nullptr || reference->hasQualifier() || callee == nullptr ||
      reference->getDecl()->getCanonicalDecl() !=
          callee->getCanonicalDecl())
    return nullptr;
  const clang::SourceLocation tokenLocation = reference->getLocation();
  if (tokenLocation.isInvalid() || tokenLocation.isMacroID())
    return nullptr;
  bool invalidSourceToken = false;
  const llvm::StringRef sourceToken = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(tokenLocation, tokenLocation),
      sourceManager, languageOptions, &invalidSourceToken);
  return !invalidSourceToken && sourceToken == expectedName ? call : nullptr;
}

static bool locationComesFromInitiallyTrustedSystemFile(
    clang::SourceManager &sourceManager,
    clang::SourceLocation location,
    const std::set<unsigned> &trustedSystemFileIds) {
  // Trust the semantic application site, not a token's macro spelling. A
  // project declaration generated by invoking a trusted system macro still
  // belongs to the project file and must not inherit the macro's provenance.
  const clang::SourceLocation expansion =
      sourceManager.getExpansionLoc(location);
  if (expansion.isInvalid())
    return false;
  const clang::FileID file = sourceManager.getFileID(expansion);
  return file.isValid() &&
         trustedSystemFileIds.count(file.getHashValue()) != 0;
}

static bool functionDeclarationComesFromInitiallyTrustedSystemFile(
    const clang::FunctionDecl *declaration,
    clang::SourceManager &sourceManager,
    const std::set<unsigned> &trustedSystemFileIds) {
  if (declaration == nullptr)
    return false;
  const clang::SourceLocation location = declaration->getLocation();
  // Clang may materialize an intrinsic declaration without a source
  // location. User declarations cannot acquire that identity, and any
  // source-backed redeclaration is checked below.
  const bool trustedDeclaration =
      (location.isInvalid() && declaration->getBuiltinID() != 0) ||
      locationComesFromInitiallyTrustedSystemFile(
          sourceManager, location, trustedSystemFileIds);
  if (!trustedDeclaration)
    return false;
  for (const clang::Attr *attribute : declaration->attrs()) {
    if (attribute == nullptr)
      return false;
    const clang::SourceLocation attributeLocation =
        attribute->getLocation();
    if (attributeLocation.isInvalid()) {
      if (attribute->isImplicit())
        continue;
      return false;
    }
    if (!locationComesFromInitiallyTrustedSystemFile(
            sourceManager, sourceManager.getExpansionLoc(attributeLocation),
            trustedSystemFileIds))
      return false;
  }
  return true;
}

static bool allFunctionRedeclarationsComeFromInitiallyTrustedSystemFiles(
    const clang::FunctionDecl *callee,
    clang::SourceManager &sourceManager,
    const std::set<unsigned> &trustedSystemFileIds) {
  if (callee == nullptr)
    return false;
  const clang::FunctionDecl *canonical = callee->getCanonicalDecl();
  if (canonical == nullptr)
    return false;
  for (const clang::FunctionDecl *redecl : canonical->redecls()) {
    if (!functionDeclarationComesFromInitiallyTrustedSystemFile(
            redecl, sourceManager, trustedSystemFileIds))
      return false;
  }
  return true;
}

static bool hasSystemFunctionDeclarationProvenance(
    const clang::CallExpr *call,
    llvm::StringRef expectedName,
    clang::SourceManager &sourceManager,
    const std::set<unsigned> &trustedSystemFileIds) {
  const clang::FunctionDecl *callee =
      call == nullptr ? nullptr : call->getDirectCallee();
  if (callee == nullptr ||
      callee->getQualifiedNameAsString() != expectedName)
    return false;
  const clang::FunctionDecl *canonical = callee->getCanonicalDecl();
  if (canonical == nullptr)
    return false;
  const clang::SourceLocation calleeLocation = callee->getLocation();
  const clang::SourceLocation canonicalLocation = canonical->getLocation();
  return allFunctionRedeclarationsComeFromInitiallyTrustedSystemFiles(
             callee, sourceManager, trustedSystemFileIds) &&
         !locationComesFromRecognizedNvidiaSampleHelper(
             sourceManager, calleeLocation) &&
         !locationComesFromRecognizedNvidiaSampleHelper(
             sourceManager, canonicalLocation);
}

static const clang::CallExpr *directCallStatementNamed(
    const clang::Stmt *statement,
    llvm::StringRef expectedName) {
  return directCallNamed(
      llvm::dyn_cast_or_null<clang::Expr>(statement), expectedName);
}

static const clang::CompoundStmt *exactCompound(
    const clang::Stmt *statement,
    unsigned expectedSize) {
  const auto *compound =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(statement);
  return compound != nullptr && compound->size() == expectedSize
             ? compound
             : nullptr;
}

static bool isDeviceLessThanZero(const clang::Expr *expression,
                                 const clang::VarDecl *deviceVariable,
                                 clang::ASTContext &context) {
  expression = stripParenAndImplicitCasts(expression);
  const auto *comparison =
      llvm::dyn_cast_or_null<clang::BinaryOperator>(expression);
  return comparison != nullptr && comparison->getOpcode() == clang::BO_LT &&
         referencesCanonicalValue(comparison->getLHS(), deviceVariable) &&
         evaluatesToNonNegative(comparison->getRHS(), 0, context);
}

static bool isExitFailureStatement(const clang::Stmt *statement,
                                   clang::ASTContext &context,
                                   clang::SourceManager &sourceManager,
                                   const std::set<unsigned> &trustedSystemFileIds) {
  const clang::CallExpr *call =
      directUnqualifiedSourceCallNamed(
          llvm::dyn_cast_or_null<clang::Expr>(statement), "exit",
          sourceManager, context.getLangOpts());
  return call != nullptr && call->getNumArgs() == 1 &&
         hasSystemFunctionDeclarationProvenance(
             call, "exit", sourceManager, trustedSystemFileIds) &&
         evaluatesToNonNegative(call->getArg(0), EXIT_FAILURE, context);
}

static bool isExactSingleStringCall(const clang::Stmt *statement,
                                    llvm::StringRef calleeName,
                                    llvm::StringRef message,
                                    clang::ASTContext &context,
                                    clang::SourceManager &sourceManager,
                                    const std::set<unsigned> &trustedSystemFileIds) {
  const clang::CallExpr *call =
      directUnqualifiedSourceCallNamed(
          llvm::dyn_cast_or_null<clang::Expr>(statement), calleeName,
          sourceManager, context.getLangOpts());
  return call != nullptr && call->getNumArgs() == 1 &&
         hasSystemFunctionDeclarationProvenance(
             call, calleeName, sourceManager, trustedSystemFileIds) &&
         isExactStringLiteral(call->getArg(0), message);
}

static bool isOfficialFailureBlock(const clang::Stmt *statement,
                                   llvm::StringRef message,
                                   clang::ASTContext &context,
                                   clang::SourceManager &sourceManager,
                                   const std::set<unsigned> &trustedSystemFileIds) {
  const clang::CompoundStmt *block = exactCompound(statement, 2);
  if (block == nullptr)
    return false;
  auto child = block->body_begin();
  return isExactSingleStringCall(
             *child++, "printf", message, context, sourceManager,
             trustedSystemFileIds) &&
         isExitFailureStatement(
             *child, context, sourceManager, trustedSystemFileIds);
}

static const clang::CallExpr *assignmentRhsCallNamed(
    const clang::Stmt *statement,
    const clang::VarDecl *destination,
    llvm::StringRef calleeName,
    clang::SourceManager &sourceManager,
    const clang::LangOptions &languageOptions) {
  const auto *expression =
      llvm::dyn_cast_or_null<clang::Expr>(statement);
  const auto *assignment = llvm::dyn_cast_or_null<clang::BinaryOperator>(
      stripParenAndImplicitCasts(expression));
  if (assignment == nullptr || assignment->getOpcode() != clang::BO_Assign ||
      !referencesCanonicalValue(assignment->getLHS(), destination))
    return nullptr;
  return directUnqualifiedSourceCallNamed(
      assignment->getRHS(), calleeName, sourceManager, languageOptions);
}

static bool callDefinitionComesFromFrozenHelperProfile(
    const clang::CallExpr *call,
    clang::SourceManager &sourceManager,
    unsigned admittedRoles) {
  const clang::FunctionDecl *callee =
      call == nullptr ? nullptr : call->getDirectCallee();
  const clang::FunctionDecl *definition = nullptr;
  return callee != nullptr && callee->hasBody(definition) &&
         definition != nullptr &&
         locationComesFromFrozenNvidiaSampleHelperFile(
             sourceManager, definition->getLocation(), admittedRoles);
}

static bool allFunctionRedeclarationsComeFromFrozenHelperProfile(
    const clang::FunctionDecl *function,
    clang::SourceManager &sourceManager,
    unsigned admittedRoles,
    const std::set<unsigned> &trustedSystemFileIds) {
  if (function == nullptr || function->getCanonicalDecl() == nullptr)
    return false;
  for (const clang::FunctionDecl *redecl :
       function->getCanonicalDecl()->redecls()) {
    if (!locationComesFromFrozenNvidiaSampleHelperFile(
            sourceManager, redecl->getLocation(), admittedRoles))
      return false;
    for (const clang::Attr *attribute : redecl->attrs()) {
      if (attribute == nullptr)
        return false;
      const clang::SourceLocation attributeLocation =
          attribute->getLocation();
      if (attributeLocation.isInvalid()) {
        if (attribute->isImplicit())
          continue;
        return false;
      }
      const clang::SourceLocation expansion =
          sourceManager.getExpansionLoc(attributeLocation);
      if (!locationComesFromFrozenNvidiaSampleHelperFile(
              sourceManager, expansion, admittedRoles) &&
          !locationComesFromInitiallyTrustedSystemFile(
              sourceManager, expansion, trustedSystemFileIds))
        return false;
    }
  }
  return true;
}

// Prove the resolved direct-call graph rooted at the frozen findCudaDevice
// definition. This complements the byte profile: preprocessing or a normal
// C/C++ redeclaration can otherwise make an unchanged helper body bind to a
// project function with telemetry or another side effect.
class FrozenNvidiaSampleHelperCallClosure
    : public clang::RecursiveASTVisitor<
          FrozenNvidiaSampleHelperCallClosure> {
public:
  FrozenNvidiaSampleHelperCallClosure(
      clang::SourceManager &sourceManager,
      const std::set<unsigned> &trustedSystemFileIds,
      unsigned admittedRoles)
      : sourceManager(sourceManager),
        trustedSystemFileIds(trustedSystemFileIds),
        admittedRoles(admittedRoles) {}

  bool prove(const clang::FunctionDecl *root) {
    failed = false;
    visited.clear();
    return proveFrozenDefinition(root) && !failed;
  }

  bool VisitCallExpr(clang::CallExpr *call) {
    if (failed || call == nullptr)
      return false;
    const clang::FunctionDecl *callee = call->getDirectCallee();
    if (callee == nullptr) {
      failed = true;
      return false;
    }
    if (allFunctionRedeclarationsComeFromInitiallyTrustedSystemFiles(
            callee, sourceManager, trustedSystemFileIds))
      return true;
    if (!proveFrozenDefinition(callee)) {
      failed = true;
      return false;
    }
    return true;
  }

private:
  bool proveFrozenDefinition(const clang::FunctionDecl *function) {
    const clang::FunctionDecl *definition = nullptr;
    if (function == nullptr || !function->hasBody(definition) ||
        definition == nullptr ||
        !locationComesFromFrozenNvidiaSampleHelperFile(
            sourceManager, definition->getLocation(), admittedRoles) ||
        !allFunctionRedeclarationsComeFromFrozenHelperProfile(
            function, sourceManager, admittedRoles,
            trustedSystemFileIds))
      return false;
    const clang::FunctionDecl *canonical = definition->getCanonicalDecl();
    if (canonical == nullptr || !visited.insert(canonical).second)
      return canonical != nullptr;
    return TraverseStmt(
        const_cast<clang::Stmt *>(definition->getBody()));
  }

  clang::SourceManager &sourceManager;
  const std::set<unsigned> &trustedSystemFileIds;
  unsigned admittedRoles;
  std::set<const clang::FunctionDecl *> visited;
  bool failed = false;
};

static bool isOfficialFindCondition(
    const clang::Expr *condition,
    const clang::ParmVarDecl *argcParameter,
    const clang::ParmVarDecl *argvParameter,
    clang::SourceManager &sourceManager,
    clang::ASTContext &context,
    unsigned admittedStringRole) {
  const clang::CallExpr *call =
      directUnqualifiedSourceCallNamed(
          condition, "checkCmdLineFlag", sourceManager,
          context.getLangOpts());
  return call != nullptr && call->getNumArgs() == 3 &&
         referencesCanonicalValue(call->getArg(0), argcParameter) &&
         referencesCanonicalValue(call->getArg(1), argvParameter) &&
         isExactStringLiteral(call->getArg(2), "device") &&
         callDefinitionComesFromFrozenHelperProfile(
             call, sourceManager, admittedStringRole);
}

static bool isGetDeviceArgumentAssignment(
    const clang::Stmt *statement,
    const clang::VarDecl *deviceVariable,
    const clang::ParmVarDecl *argcParameter,
    const clang::ParmVarDecl *argvParameter,
    clang::SourceManager &sourceManager,
    clang::ASTContext &context,
    unsigned admittedStringRole) {
  const clang::CallExpr *call = assignmentRhsCallNamed(
      statement, deviceVariable, "getCmdLineArgumentInt", sourceManager,
      context.getLangOpts());
  return call != nullptr && call->getNumArgs() == 3 &&
         referencesCanonicalValue(call->getArg(0), argcParameter) &&
         referencesCanonicalValue(call->getArg(1), argvParameter) &&
         isExactStringLiteral(call->getArg(2), "device=") &&
         callDefinitionComesFromFrozenHelperProfile(
             call, sourceManager, admittedStringRole);
}

static bool isDeviceInitAssignment(
    const clang::Stmt *statement,
    const clang::VarDecl *deviceVariable,
    clang::SourceManager &sourceManager,
    clang::ASTContext &context,
    unsigned admittedCudaRole) {
  const clang::CallExpr *call = assignmentRhsCallNamed(
      statement, deviceVariable, "gpuDeviceInit", sourceManager,
      context.getLangOpts());
  return call != nullptr && call->getNumArgs() == 1 &&
         referencesCanonicalValue(call->getArg(0), deviceVariable) &&
         callDefinitionComesFromFrozenHelperProfile(
             call, sourceManager, admittedCudaRole);
}

static bool isMaxGflopsAssignment(
    const clang::Stmt *statement,
    const clang::VarDecl *deviceVariable,
    clang::SourceManager &sourceManager,
    clang::ASTContext &context,
    unsigned admittedCudaRole) {
  const clang::CallExpr *call = assignmentRhsCallNamed(
      statement, deviceVariable, "gpuGetMaxGflopsDeviceId", sourceManager,
      context.getLangOpts());
  return call != nullptr && call->getNumArgs() == 0 &&
         callDefinitionComesFromFrozenHelperProfile(
             call, sourceManager, admittedCudaRole);
}

static const clang::CallExpr *checkedRuntimeCallNamed(
    const clang::Stmt *statement,
    llvm::StringRef runtimeName,
    clang::SourceManager &sourceManager,
    clang::ASTContext &context,
    unsigned admittedCudaRole,
    const std::set<unsigned> &trustedSystemFileIds) {
  const clang::CallExpr *check =
      directCallStatementNamed(statement, "check");
  if (check == nullptr || check->getNumArgs() != 4 ||
      !callDefinitionComesFromFrozenHelperProfile(
          check, sourceManager, admittedCudaRole) ||
      !isAdmittedCudaRuntimeStatusCall(
          sourceManager, check->getArg(0), trustedSystemFileIds) ||
      llvm::dyn_cast_or_null<clang::StringLiteral>(
          stripParenAndImplicitCasts(check->getArg(1))) == nullptr ||
      llvm::dyn_cast_or_null<clang::StringLiteral>(
          stripParenAndImplicitCasts(check->getArg(2))) == nullptr ||
      !check->getArg(3)->isIntegerConstantExpr(context))
    return nullptr;
  return directCallNamed(check->getArg(0), runtimeName);
}

static bool isCheckedSetDeviceStatement(
    const clang::Stmt *statement,
    const clang::VarDecl *deviceVariable,
    clang::SourceManager &sourceManager,
    clang::ASTContext &context,
    unsigned admittedCudaRole,
    const std::set<unsigned> &trustedSystemFileIds) {
  const clang::CallExpr *call = checkedRuntimeCallNamed(
      statement, "cudaSetDevice", sourceManager, context,
      admittedCudaRole, trustedSystemFileIds);
  return call != nullptr && call->getNumArgs() == 1 &&
         referencesCanonicalValue(call->getArg(0), deviceVariable);
}

static bool referencesNamedDeclaration(const clang::Expr *expression,
                                       llvm::StringRef name) {
  expression = stripParenAndImplicitCasts(expression);
  const auto *reference =
      llvm::dyn_cast_or_null<clang::DeclRefExpr>(expression);
  return reference != nullptr && reference->getDecl() != nullptr &&
         reference->getDecl()->getName() == name;
}

static bool isAddressOfValue(const clang::Expr *expression,
                             const clang::ValueDecl *value) {
  expression = stripParenAndImplicitCasts(expression);
  const auto *address =
      llvm::dyn_cast_or_null<clang::UnaryOperator>(expression);
  return address != nullptr && address->getOpcode() == clang::UO_AddrOf &&
         referencesCanonicalValue(address->getSubExpr(), value);
}

static bool isCheckedDeviceAttributeStatement(
    const clang::Stmt *statement,
    const clang::VarDecl *output,
    llvm::StringRef attributeName,
    const clang::VarDecl *deviceVariable,
    clang::SourceManager &sourceManager,
    clang::ASTContext &context,
    unsigned admittedCudaRole,
    const std::set<unsigned> &trustedSystemFileIds) {
  const clang::CallExpr *call = checkedRuntimeCallNamed(
      statement, "cudaDeviceGetAttribute", sourceManager, context,
      admittedCudaRole, trustedSystemFileIds);
  return call != nullptr && call->getNumArgs() == 3 &&
         isAddressOfValue(call->getArg(0), output) &&
         referencesNamedDeclaration(call->getArg(1), attributeName) &&
         referencesCanonicalValue(call->getArg(2), deviceVariable);
}

static bool isIntZeroVariable(const clang::VarDecl *variable,
                              llvm::StringRef name,
                              clang::ASTContext &context) {
  return variable != nullptr && variable->getName() == name &&
         context.hasSameType(variable->getType().getCanonicalType(),
                             context.IntTy) &&
         evaluatesToNonNegative(variable->getInit(), 0, context);
}

static bool getMajorMinorDeclarations(const clang::Stmt *statement,
                                      const clang::VarDecl *&major,
                                      const clang::VarDecl *&minor,
                                      clang::ASTContext &context) {
  const auto *declarations =
      llvm::dyn_cast_or_null<clang::DeclStmt>(statement);
  if (declarations == nullptr)
    return false;
  std::vector<const clang::VarDecl *> variables;
  for (const clang::Decl *declaration : declarations->decls()) {
    const auto *variable = llvm::dyn_cast<clang::VarDecl>(declaration);
    if (variable == nullptr)
      return false;
    variables.push_back(variable);
  }
  if (variables.size() != 2 ||
      !isIntZeroVariable(variables[0], "major", context) ||
      !isIntZeroVariable(variables[1], "minor", context))
    return false;
  major = variables[0];
  minor = variables[1];
  return true;
}

static bool isOfficialDeviceSummaryPrint(
    const clang::Stmt *statement,
    const clang::VarDecl *deviceVariable,
    const clang::VarDecl *major,
    const clang::VarDecl *minor,
    clang::SourceManager &sourceManager,
    clang::ASTContext &context,
    unsigned admittedCudaRole,
    const std::set<unsigned> &trustedSystemFileIds) {
  const clang::CallExpr *print =
      directUnqualifiedSourceCallNamed(
          llvm::dyn_cast_or_null<clang::Expr>(statement), "printf",
          sourceManager, context.getLangOpts());
  if (print == nullptr || print->getNumArgs() != 5 ||
      !hasSystemFunctionDeclarationProvenance(
          print, "printf", sourceManager, trustedSystemFileIds) ||
      !isExactStringLiteral(
          print->getArg(0),
          "GPU Device %d: \"%s\" with compute capability %d.%d\n\n") ||
      !referencesCanonicalValue(print->getArg(1), deviceVariable) ||
      !referencesCanonicalValue(print->getArg(3), major) ||
      !referencesCanonicalValue(print->getArg(4), minor))
    return false;
  const clang::CallExpr *architecture =
      directUnqualifiedSourceCallNamed(
          print->getArg(2), "_ConvertSMVer2ArchName", sourceManager,
          context.getLangOpts());
  return architecture != nullptr && architecture->getNumArgs() == 2 &&
         referencesCanonicalValue(architecture->getArg(0), major) &&
         referencesCanonicalValue(architecture->getArg(1), minor) &&
         callDefinitionComesFromFrozenHelperProfile(
             architecture, sourceManager, admittedCudaRole);
}

static bool isOfficialExplicitDeviceBranch(
    const clang::Stmt *statement,
    const clang::ParmVarDecl *argcParameter,
    const clang::ParmVarDecl *argvParameter,
    const clang::VarDecl *deviceVariable,
    clang::SourceManager &sourceManager,
    clang::ASTContext &context,
    unsigned admittedCudaRole,
    unsigned admittedStringRole,
    const std::set<unsigned> &trustedSystemFileIds) {
  const clang::CompoundStmt *branch = exactCompound(statement, 2);
  if (branch == nullptr)
    return false;
  auto child = branch->body_begin();
  if (!isGetDeviceArgumentAssignment(
          *child++, deviceVariable, argcParameter, argvParameter,
          sourceManager, context, admittedStringRole))
    return false;
  const auto *validation = llvm::dyn_cast_or_null<clang::IfStmt>(*child);
  if (validation == nullptr || validation->getElse() == nullptr ||
      !isDeviceLessThanZero(
          validation->getCond(), deviceVariable, context) ||
      !isOfficialFailureBlock(
          validation->getThen(), "Invalid command line parameter\n ",
          context, sourceManager, trustedSystemFileIds))
    return false;
  const clang::CompoundStmt *valid =
      exactCompound(validation->getElse(), 2);
  if (valid == nullptr)
    return false;
  auto validChild = valid->body_begin();
  if (!isDeviceInitAssignment(
          *validChild++, deviceVariable, sourceManager, context,
          admittedCudaRole))
    return false;
  const auto *initialized =
      llvm::dyn_cast_or_null<clang::IfStmt>(*validChild);
  return initialized != nullptr && initialized->getElse() == nullptr &&
         isDeviceLessThanZero(
             initialized->getCond(), deviceVariable, context) &&
         isOfficialFailureBlock(
             initialized->getThen(), "exiting...\n", context,
             sourceManager, trustedSystemFileIds);
}

static bool isOfficialAutomaticDeviceBranch(
    const clang::Stmt *statement,
    const clang::VarDecl *deviceVariable,
    clang::SourceManager &sourceManager,
    clang::ASTContext &context,
    unsigned admittedCudaRole,
    const std::set<unsigned> &trustedSystemFileIds) {
  const clang::CompoundStmt *branch = exactCompound(statement, 6);
  if (branch == nullptr)
    return false;
  auto child = branch->body_begin();
  if (!isMaxGflopsAssignment(
          *child++, deviceVariable, sourceManager, context,
          admittedCudaRole) ||
      !isCheckedSetDeviceStatement(
          *child++, deviceVariable, sourceManager, context,
          admittedCudaRole, trustedSystemFileIds))
    return false;
  const clang::VarDecl *major = nullptr;
  const clang::VarDecl *minor = nullptr;
  if (!getMajorMinorDeclarations(*child++, major, minor, context) ||
      !isCheckedDeviceAttributeStatement(
          *child++, major, "cudaDevAttrComputeCapabilityMajor",
          deviceVariable, sourceManager, context, admittedCudaRole,
          trustedSystemFileIds) ||
      !isCheckedDeviceAttributeStatement(
          *child++, minor, "cudaDevAttrComputeCapabilityMinor",
          deviceVariable, sourceManager, context, admittedCudaRole,
          trustedSystemFileIds))
    return false;
  return isOfficialDeviceSummaryPrint(
      *child, deviceVariable, major, minor, sourceManager, context,
      admittedCudaRole, trustedSystemFileIds);
}

static bool hasProvenNvidiaFindCudaDeviceDefinition(
    const clang::FunctionDecl *callee,
    clang::ASTContext &context,
    clang::SourceManager &sourceManager,
    const std::set<unsigned> &trustedSystemFileIds) {
  if (!hasExactNvidiaFindCudaDeviceSignature(callee, context))
    return false;
  const clang::FunctionDecl *definition = nullptr;
  if (!callee->hasBody(definition) || definition == nullptr ||
      !definition->isInlineSpecified() ||
      definition->getTemplatedKind() !=
          clang::FunctionDecl::TK_NonTemplate ||
      definition->getCanonicalDecl() != callee->getCanonicalDecl() ||
      !locationComesFromRecognizedNvidiaSampleHelper(
          sourceManager, definition->getLocation()))
    return false;

  const clang::SourceLocation definitionSpelling =
      sourceManager.getSpellingLoc(definition->getLocation());
  if (definitionSpelling.isInvalid())
    return false;
  const unsigned admittedCudaRole = frozenNvidiaSampleHelperFileRole(
      sourceManager, definitionSpelling);
  unsigned admittedStringRole = FrozenNvidiaSampleHelperNone;
  if (admittedCudaRole == FrozenOfficialNvidiaSampleHelperCuda)
    admittedStringRole = FrozenOfficialNvidiaSampleHelperString;
  else
    return false;

  const auto *body =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(definition->getBody());
  if (body == nullptr || body->size() != 3)
    return false;
  auto statement = body->body_begin();
  const auto *declarationStatement =
      llvm::dyn_cast_or_null<clang::DeclStmt>(*statement++);
  const auto *selection = llvm::dyn_cast_or_null<clang::IfStmt>(*statement++);
  const auto *returnStatement =
      llvm::dyn_cast_or_null<clang::ReturnStmt>(*statement);
  if (declarationStatement == nullptr ||
      !declarationStatement->isSingleDecl() || selection == nullptr ||
      selection->getElse() == nullptr || returnStatement == nullptr)
    return false;
  const auto *deviceVariable = llvm::dyn_cast<clang::VarDecl>(
      declarationStatement->getSingleDecl());
  if (deviceVariable == nullptr || deviceVariable->getName() != "devID" ||
      !context.hasSameType(deviceVariable->getType().getCanonicalType(),
                           context.IntTy) ||
      !evaluatesToNonNegative(deviceVariable->getInit(), 0, context) ||
      !referencesCanonicalValue(returnStatement->getRetValue(),
                                deviceVariable))
    return false;

  const bool exactOfficialBody = isOfficialFindCondition(
             selection->getCond(), definition->getParamDecl(0),
             definition->getParamDecl(1), sourceManager, context,
             admittedStringRole) &&
         isOfficialExplicitDeviceBranch(
             selection->getThen(), definition->getParamDecl(0),
             definition->getParamDecl(1), deviceVariable,
             sourceManager, context, admittedCudaRole,
             admittedStringRole, trustedSystemFileIds) &&
         isOfficialAutomaticDeviceBranch(
             selection->getElse(), deviceVariable,
             sourceManager, context, admittedCudaRole,
             trustedSystemFileIds);
  if (!exactOfficialBody)
    return false;
  FrozenNvidiaSampleHelperCallClosure closure(
      sourceManager, trustedSystemFileIds,
      FrozenOfficialNvidiaSampleHelperCuda |
          FrozenOfficialNvidiaSampleHelperString);
  return closure.prove(definition);
}

static bool proveNvidiaFindCudaDeviceCall(
    const clang::CallExpr *expression,
    clang::ASTContext &context,
    clang::SourceManager &sourceManager,
    const clang::LangOptions &languageOptions,
    const std::set<unsigned> &trustedSystemFileIds,
    clang::SourceLocation &nameLocation,
    clang::FileID &rewriteFile,
    unsigned &rewriteBegin,
    unsigned &rewriteEnd) {
  if (expression == nullptr || expression->getNumArgs() != 2)
    return false;
  const clang::FunctionDecl *callee = expression->getDirectCallee();
  if (callee == nullptr || callee->getIdentifier() == nullptr ||
      callee->getName() != "findCudaDevice" ||
      !hasProvenNvidiaFindCudaDeviceDefinition(
          callee, context, sourceManager, trustedSystemFileIds))
    return false;

  const clang::FunctionDecl *enclosing =
      enclosingNonLambdaFunction(expression, context);
  if (enclosing == nullptr ||
      enclosing->hasAttr<clang::CUDADeviceAttr>() ||
      enclosing->hasAttr<clang::CUDAGlobalAttr>())
    return false;

  const clang::Expr *calleeExpression =
      stripParenAndImplicitCasts(expression->getCallee());
  const auto *reference =
      llvm::dyn_cast_or_null<clang::DeclRefExpr>(calleeExpression);
  if (reference == nullptr || reference->hasQualifier() ||
      reference->getDecl()->getCanonicalDecl() !=
          callee->getCanonicalDecl())
    return false;
  nameLocation = reference->getLocation();
  const clang::SourceLocation fileLocation =
      sourceManager.getFileLoc(nameLocation);
  if (nameLocation.isInvalid() || nameLocation.isMacroID() ||
      fileLocation.isInvalid() ||
      !sourceManager.isWrittenInMainFile(fileLocation))
    return false;
  bool invalidSourceToken = false;
  const llvm::StringRef sourceToken = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(nameLocation, nameLocation),
      sourceManager, languageOptions, &invalidSourceToken);
  if (invalidSourceToken || sourceToken != "findCudaDevice")
    return false;
  const clang::SourceLocation end = clang::Lexer::getLocForEndOfToken(
      fileLocation, 0, sourceManager, languageOptions);
  if (end.isInvalid())
    return false;
  const std::pair<clang::FileID, unsigned> beginDecomposed =
      sourceManager.getDecomposedLoc(fileLocation);
  const std::pair<clang::FileID, unsigned> endDecomposed =
      sourceManager.getDecomposedLoc(end);
  if (beginDecomposed.first != endDecomposed.first ||
      endDecomposed.second <= beginDecomposed.second)
    return false;
  rewriteFile = beginDecomposed.first;
  rewriteBegin = beginDecomposed.second;
  rewriteEnd = endDecomposed.second;
  return true;
}

// This is deliberately a provenance proof, not a pointer-type guess.  Only a
// direct parameter of the current CUDA global function establishes global
// memory.  Local aliases, globals, fields, shared variables, helper parameters,
// and conditional pointer selection all fail closed.
static bool isAddressRootedAtCurrentGlobalParameter(
    const clang::Expr *expression,
    const clang::FunctionDecl *kernel,
    clang::ASTContext &context) {
  if (expression == nullptr || kernel == nullptr)
    return false;
  expression = stripParenAndImplicitCasts(expression);

  if (const auto *reference =
          llvm::dyn_cast_or_null<clang::DeclRefExpr>(expression)) {
    const auto *parameter =
        llvm::dyn_cast<clang::ParmVarDecl>(reference->getDecl());
    const auto *owner =
        parameter == nullptr
            ? nullptr
            : llvm::dyn_cast<clang::FunctionDecl>(
                  parameter->getDeclContext());
    return parameter != nullptr && owner != nullptr &&
           owner->getCanonicalDecl() == kernel->getCanonicalDecl() &&
           classifyGlobalAtomicPointer(parameter->getType(), context) !=
               GlobalAtomicScalarKind::Unsupported;
  }

  if (const auto *cast =
          llvm::dyn_cast_or_null<clang::CastExpr>(expression)) {
    if (cast->getCastKind() != clang::CK_BitCast &&
        cast->getCastKind() != clang::CK_NoOp)
      return false;
    if (classifyGlobalAtomicPointer(cast->getType(), context) ==
            GlobalAtomicScalarKind::Unsupported ||
        classifyGlobalAtomicPointer(cast->getSubExpr()->getType(), context) ==
            GlobalAtomicScalarKind::Unsupported)
      return false;
    return isAddressRootedAtCurrentGlobalParameter(
        cast->getSubExpr(), kernel, context);
  }

  if (const auto *subscript =
          llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(expression))
    return isAddressRootedAtCurrentGlobalParameter(
        subscript->getBase(), kernel, context);

  if (const auto *unary =
          llvm::dyn_cast_or_null<clang::UnaryOperator>(expression)) {
    if (unary->getOpcode() != clang::UO_AddrOf &&
        unary->getOpcode() != clang::UO_Deref)
      return false;
    return isAddressRootedAtCurrentGlobalParameter(
        unary->getSubExpr(), kernel, context);
  }

  const auto *binary =
      llvm::dyn_cast_or_null<clang::BinaryOperator>(expression);
  if (binary == nullptr ||
      (binary->getOpcode() != clang::BO_Add &&
       binary->getOpcode() != clang::BO_Sub))
    return false;
  const bool lhsPointer =
      classifyGlobalAtomicPointer(binary->getLHS()->getType(), context) !=
      GlobalAtomicScalarKind::Unsupported;
  const bool rhsPointer =
      classifyGlobalAtomicPointer(binary->getRHS()->getType(), context) !=
      GlobalAtomicScalarKind::Unsupported;
  if (binary->getOpcode() == clang::BO_Sub) {
    return lhsPointer && !rhsPointer &&
           binary->getRHS()->getType()->isIntegerType() &&
           isAddressRootedAtCurrentGlobalParameter(
               binary->getLHS(), kernel, context);
  }
  if (lhsPointer == rhsPointer)
    return false;
  const clang::Expr *pointer =
      lhsPointer ? binary->getLHS() : binary->getRHS();
  const clang::Expr *offset =
      lhsPointer ? binary->getRHS() : binary->getLHS();
  return offset->getType()->isIntegerType() &&
         isAddressRootedAtCurrentGlobalParameter(
             pointer, kernel, context);
}

static bool hasExactGlobalAtomicSignature(
    const clang::FunctionDecl *callee,
    const GlobalAtomicSpec &spec,
    GlobalAtomicScalarKind kind,
    clang::ASTContext &context) {
  if (callee == nullptr || callee->isVariadic() ||
      callee->getNumParams() != spec.argumentCount ||
      classifyGlobalAtomicPointer(
          callee->getParamDecl(0)->getType(), context) != kind ||
      classifyGlobalAtomicScalar(callee->getReturnType(), context) != kind)
    return false;
  for (unsigned index = 1; index < spec.argumentCount; ++index) {
    if (classifyGlobalAtomicScalar(
            callee->getParamDecl(index)->getType(), context) != kind)
      return false;
  }
  return true;
}

/**
  * Get a string representation of the expression `arg`, unless it's a defaulting function
  * call argument, in which case get a 0. Used for building argument lists to kernel calls.
  */
std::string stringifyZeroDefaultedArg(clang::SourceManager &SM, const clang::Expr *arg) {
  if (clang::isa<clang::CXXDefaultArgExpr>(arg)) return "0";
  else return std::string(readSourceText(SM, arg->getSourceRange()));
}

} // anonymous namespace

/**
  * Look at, and consider altering, a given token.
  *
  * If it's not a CUDA identifier, nothing happens.
  * If it's an unsupported CUDA identifier, a warning is emitted.
  * Otherwise, the source file is updated with the corresponding ascification.
  *
  * Returns true when the raw lexer was advanced past rewritten text.
  */
bool AscifyAction::isInSemanticRewriteRange(clang::SourceLocation loc) {
  if (loc.isInvalid())
    return false;
  auto &SM = getCompilerInstance().getSourceManager();
  const clang::SourceLocation fileLoc = SM.getFileLoc(loc);
  if (fileLoc.isInvalid())
    return false;
  const std::pair<clang::FileID, unsigned> decomposed =
      SM.getDecomposedLoc(fileLoc);
  for (const SemanticRewriteRange &range : SemanticRewriteRanges) {
    if (range.file == decomposed.first &&
        decomposed.second >= range.beginOffset &&
        decomposed.second < range.endOffset)
      return true;
  }
  return false;
}

bool AscifyAction::isInNvidiaSampleHelperMacroRange(
    clang::SourceLocation loc) {
  if (loc.isInvalid())
    return false;
  auto &sourceManager = getCompilerInstance().getSourceManager();
  const clang::SourceLocation expansion =
      sourceManager.getExpansionLoc(loc);
  const clang::SourceLocation fileLoc = sourceManager.getFileLoc(expansion);
  if (fileLoc.isInvalid())
    return false;
  const std::pair<clang::FileID, unsigned> decomposed =
      sourceManager.getDecomposedLoc(fileLoc);
  for (const SemanticRewriteRange &range :
       nvidiaSampleHelperMacroRanges) {
    if (range.file == decomposed.first &&
        decomposed.second >= range.beginOffset &&
        decomposed.second < range.endOffset)
      return true;
  }
  return false;
}

bool hasInputTopLevelAscifyDeclaration(
    clang::ASTContext &context, clang::SourceManager &sourceManager) {
  class Visitor : public clang::RecursiveASTVisitor<Visitor> {
  public:
    explicit Visitor(clang::SourceManager &sourceManager)
        : sourceManager(sourceManager) {}

    bool VisitNamedDecl(clang::NamedDecl *declaration) {
      if (conflict || declaration == nullptr || declaration->isImplicit() ||
          declaration->getIdentifier() == nullptr ||
          declaration->getName() != "ascify" ||
          declaration->getDeclContext() == nullptr ||
          !declaration->getDeclContext()->getRedeclContext()
               ->isTranslationUnit() ||
          locationComesFromAscifyCudaCompat(
              sourceManager, declaration->getLocation()))
        return !conflict;
      conflict = true;
      return false;
    }

    bool hasConflict() const { return conflict; }

  private:
    clang::SourceManager &sourceManager;
    bool conflict = false;
  } visitor(sourceManager);
  visitor.TraverseDecl(context.getTranslationUnitDecl());
  return visitor.hasConflict();
}

bool AscifyAction::hasUnsupportedNvidiaSampleHelperDeclarationUse() {
  class Visitor : public clang::RecursiveASTVisitor<Visitor> {
  public:
    Visitor(clang::SourceManager &sourceManager,
            clang::ASTContext &astContext,
            const clang::LangOptions &languageOptions,
            const std::set<unsigned> &trustedSystemFileIds,
            const std::vector<SemanticRewriteRange> &supportedRanges,
            std::vector<NvidiaSampleHelperMacroCandidate> &candidates,
            std::vector<NvidiaSampleFindDeviceCandidate> &findCandidates)
        : sourceManager(sourceManager), astContext(astContext),
          languageOptions(languageOptions),
          trustedSystemFileIds(trustedSystemFileIds),
          supportedRanges(supportedRanges),
          candidates(candidates), findCandidates(findCandidates) {}

    bool VisitCallExpr(clang::CallExpr *expression) {
      if (expression == nullptr)
        return true;
      const clang::FunctionDecl *callee = expression->getDirectCallee();
      if (callee != nullptr && callee->getIdentifier() != nullptr &&
          callee->getName() == "findCudaDevice" &&
          locationComesFromRecognizedNvidiaSampleHelper(
              sourceManager, callee->getLocation())) {
        clang::SourceLocation nameLocation;
        clang::FileID rewriteFile;
        unsigned rewriteBegin = 0;
        unsigned rewriteEnd = 0;
        if (proveNvidiaFindCudaDeviceCall(
                expression, astContext, sourceManager, languageOptions,
                trustedSystemFileIds,
                nameLocation, rewriteFile, rewriteBegin, rewriteEnd)) {
          findCandidates.push_back(
              {nameLocation,
               {rewriteFile, rewriteBegin, rewriteEnd},
               false});
        }
        return true;
      }
      if (callee == nullptr || callee->getIdentifier() == nullptr ||
          callee->getName() != "check" ||
          !locationComesFromRecognizedNvidiaSampleHelper(
              sourceManager, callee->getLocation()))
        return true;
      NvidiaSampleHelperMacroCandidate *candidate =
          candidateAt(expression->getExprLoc(),
                      NvidiaSampleHelperMacroKind::CheckCudaErrors);
      if (candidate == nullptr || expression->getNumArgs() == 0)
        return true;
      candidate->statusDomainProven = isAdmittedCudaRuntimeStatusCall(
          sourceManager, expression->getArg(0), trustedSystemFileIds);
      return true;
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr *expression) {
      return expression == nullptr
                 ? true
                 : inspect(expression->getExprLoc(),
                           expression->getDecl());
    }

    bool VisitUnresolvedLookupExpr(
        clang::UnresolvedLookupExpr *expression) {
      if (expression == nullptr)
        return true;
      for (const clang::NamedDecl *declaration : expression->decls()) {
        if (!inspect(expression->getNameLoc(), declaration))
          return false;
      }
      return true;
    }

    bool VisitMemberExpr(clang::MemberExpr *expression) {
      return expression == nullptr
                 ? true
                 : inspect(expression->getMemberLoc(),
                           expression->getMemberDecl());
    }

    bool VisitCXXConstructExpr(clang::CXXConstructExpr *expression) {
      return expression == nullptr
                 ? true
                 : inspect(expression->getExprLoc(),
                           expression->getConstructor());
    }

    bool VisitTypedefTypeLoc(clang::TypedefTypeLoc location) {
      const clang::TypedefType *type = location.getTypePtr();
      return inspect(location.getNameLoc(),
                     type == nullptr ? nullptr : type->getDecl());
    }

    bool VisitRecordTypeLoc(clang::RecordTypeLoc location) {
      return inspect(location.getNameLoc(), location.getDecl());
    }

    bool VisitEnumTypeLoc(clang::EnumTypeLoc location) {
      return inspect(location.getNameLoc(), location.getDecl());
    }

    bool VisitUsingShadowDecl(clang::UsingShadowDecl *declaration) {
      return declaration == nullptr
                 ? true
                 : inspect(declaration->getLocation(),
                           declaration->getTargetDecl());
    }

    bool hasConflict() const { return conflict; }
    llvm::StringRef conflictingName() const { return name; }

  private:
    NvidiaSampleHelperMacroCandidate *candidateAt(
        clang::SourceLocation location,
        NvidiaSampleHelperMacroKind kind) {
      const clang::SourceLocation expansion =
          sourceManager.getExpansionLoc(location);
      const clang::SourceLocation fileLoc =
          sourceManager.getFileLoc(expansion);
      if (fileLoc.isInvalid())
        return nullptr;
      const std::pair<clang::FileID, unsigned> decomposed =
          sourceManager.getDecomposedLoc(fileLoc);
      for (NvidiaSampleHelperMacroCandidate &candidate : candidates) {
        const SemanticRewriteRange &range = candidate.invocationRange;
        if (candidate.kind == kind && range.file == decomposed.first &&
            decomposed.second >= range.beginOffset &&
            decomposed.second < range.endOffset)
          return &candidate;
      }
      return nullptr;
    }

    bool isSupportedHelperRewrite(clang::SourceLocation location) const {
      const clang::SourceLocation expansion =
          sourceManager.getExpansionLoc(location);
      const clang::SourceLocation fileLoc =
          sourceManager.getFileLoc(expansion);
      if (fileLoc.isInvalid())
        return false;
      const std::pair<clang::FileID, unsigned> decomposed =
          sourceManager.getDecomposedLoc(fileLoc);
      for (const SemanticRewriteRange &range : supportedRanges) {
        if (range.file == decomposed.first &&
            decomposed.second >= range.beginOffset &&
            decomposed.second < range.endOffset)
          return true;
      }
      for (const NvidiaSampleFindDeviceCandidate &candidate :
           findCandidates) {
        const SemanticRewriteRange &range = candidate.rewriteRange;
        if (range.file == decomposed.first &&
            decomposed.second >= range.beginOffset &&
            decomposed.second < range.endOffset)
          return true;
      }
      return false;
    }

    bool inspect(clang::SourceLocation useLocation,
                 const clang::NamedDecl *referenced) {
      if (conflict || referenced == nullptr)
        return !conflict;
      const clang::SourceLocation expansion =
          sourceManager.getExpansionLoc(useLocation);
      if (expansion.isInvalid() ||
          !locationComesFromRecognizedNvidiaSampleHelper(
              sourceManager, referenced->getLocation()) ||
          isSupportedHelperRewrite(useLocation) ||
          (!sourceManager.isWrittenInMainFile(expansion) &&
           locationComesFromRecognizedNvidiaSampleHelper(
               sourceManager, useLocation)))
        return true;
      conflict = true;
      name = referenced->getQualifiedNameAsString();
      return false;
    }

    clang::SourceManager &sourceManager;
    clang::ASTContext &astContext;
    const clang::LangOptions &languageOptions;
    const std::set<unsigned> &trustedSystemFileIds;
    const std::vector<SemanticRewriteRange> &supportedRanges;
    std::vector<NvidiaSampleHelperMacroCandidate> &candidates;
    std::vector<NvidiaSampleFindDeviceCandidate> &findCandidates;
    bool conflict = false;
    std::string name;
  } visitor(getCompilerInstance().getSourceManager(),
            getCompilerInstance().getASTContext(),
            getCompilerInstance().getLangOpts(),
            trustedSystemFileIds,
            nvidiaSampleHelperMacroRanges,
            nvidiaSampleHelperMacroCandidates,
            nvidiaSampleFindDeviceCandidates);

  visitor.TraverseDecl(
      getCompilerInstance().getASTContext().getTranslationUnitDecl());
  for (const NvidiaSampleFindDeviceCandidate &candidate :
       nvidiaSampleFindDeviceCandidates) {
    nvidiaSampleFindDeviceNormalOffsets.insert(
        getCompilerInstance().getSourceManager().getFileOffset(
            getCompilerInstance().getSourceManager().getFileLoc(
                candidate.nameLocation)));
  }
  const bool helperNeedsCompat =
      !nvidiaSampleHelperMacroCandidates.empty() ||
      !nvidiaSampleFindDeviceCandidates.empty();
  if (helperNeedsCompat && ascifyCudaCompatActiveMacroConflict)
    nvidiaSampleHelperUnsupportedMacroUse = true;
  const bool compatNamespaceConflict =
      helperNeedsCompat && !hasCudaCompatHeaderBeforeNvidiaHelper &&
      ascifyCudaCompatDeclarationConflict;
  if (visitor.hasConflict()) {
    llvm::errs() << "Ascify NVIDIA sample-helper closure: residual helper "
                 << "declaration '" << visitor.conflictingName()
                 << "' keeps helper_cuda.h\n";
  }
  if (compatNamespaceConflict) {
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: top-level declaration "
        << "'ascify' conflicts with the required compat namespace; "
        << "helper_cuda.h kept\n";
  }
  for (const NvidiaSampleHelperMacroCandidate &candidate :
       nvidiaSampleHelperMacroCandidates) {
    if (candidate.kind ==
            NvidiaSampleHelperMacroKind::CheckCudaErrors &&
        !candidate.statusDomainProven) {
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: checkCudaErrors status "
          << "domain not proven; macro and include kept\n";
    }
  }
  return visitor.hasConflict() || compatNamespaceConflict;
}

bool AscifyAction::RewriteToken(clang::Lexer &, clang::Token &tok) {
  auditRawNvidiaSampleHelperToken(tok);
  // AST semantic transformations own their complete source span.  The raw
  // identifier pass runs later and must not add overlapping token replacements
  // inside that span.
  if (isInSemanticRewriteRange(tok.getLocation()))
    return false;

  if (!AscifyAMAP) {
    clang::SourceRange sr(tok.getLocation());
    for (const auto &skipped : SkippedSourceRanges) {
      if (skipped.fullyContains(sr))
        return false;
    }
  }

  if (tok.is(clang::tok::string_literal)) {
    llvm::StringRef s(tok.getLiteralData(), tok.getLength());
    RewriteString(unquoteStr(s), tok.getLocation());
    return false;
  }
  if (!tok.isAnyIdentifier())
    return false;

  llvm::StringRef name = tok.getRawIdentifier();

  // Do not turn std::exp (or another explicitly-qualified function) into an
  // invalid std::expf spelling. Unqualified CUDA device math remains mapped.
  if (isUnqualifiedFloatMathName(name)) {
    const clang::Token *previous = previousSignificantToken(rawTokenWindow);
    if (previous != nullptr && previous->is(clang::tok::coloncolon))
      return false;
  }

  if (requiresCudaCompatHeader(name))
    needsCudaCompatHeader = true;

  FindAndReplace(name, tok.getLocation(), CUDA_RENAMES_MAP());
  return false;
}

bool AscifyAction::Exclude(const dppCounter &hipToken) {
  return false;
}

void AscifyAction::FileChanged(
    clang::SourceLocation location,
    clang::PPCallbacks::FileChangeReason reason,
    clang::SrcMgr::CharacteristicKind fileType) {
  if (reason == clang::PPCallbacks::ExitFile) {
    if (!nvidiaSampleHelperCudaFileStack.empty())
      nvidiaSampleHelperCudaFileStack.pop_back();
    return;
  }
  if (reason != clang::PPCallbacks::EnterFile)
    return;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::SourceLocation spelling =
      sourceManager.getSpellingLoc(location);
  if (spelling.isInvalid()) {
    nvidiaSampleHelperCudaFileStack.push_back(
        !nvidiaSampleHelperCudaFileStack.empty() &&
        nvidiaSampleHelperCudaFileStack.back());
    return;
  }
  const clang::FileID file = sourceManager.getFileID(spelling);
  if (file.isInvalid()) {
    nvidiaSampleHelperCudaFileStack.push_back(
        !nvidiaSampleHelperCudaFileStack.empty() &&
        nvidiaSampleHelperCudaFileStack.back());
    return;
  }
  const unsigned frozenRole =
      frozenNvidiaSampleHelperFileRole(sourceManager, spelling);
  const bool parentInsideRemovedHelper =
      !nvidiaSampleHelperCudaFileStack.empty() &&
      nvidiaSampleHelperCudaFileStack.back();
  const bool insideRemovedHelper =
      parentInsideRemovedHelper ||
      frozenRole == FrozenOfficialNvidiaSampleHelperCuda;
  nvidiaSampleHelperCudaFileStack.push_back(insideRemovedHelper);

  const llvm::StringRef enteredFilename =
      llvm::sys::path::filename(sourceManager.getFilename(spelling));
  const bool exactCompat =
      locationComesFromAscifyCudaCompat(sourceManager, spelling);
  if (enteredFilename == "ascify_cuda_compat.hpp" && !exactCompat) {
    ascifyCudaCompatIncludeConflict = true;
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify CUDA compat closure: parsed compat header identity is "
        << "unrecognized; helper edits kept\n";
  }
  if (frozenRole != FrozenOfficialNvidiaSampleHelperCuda && !exactCompat &&
      !sourceManager.isWrittenInMainFile(spelling) &&
      nvidiaSampleRetainedFileIdHashes.insert(file.getHashValue()).second)
    nvidiaSampleRetainedFileIds.push_back(file);

  const clang::FileEntry *entry = sourceManager.getFileEntryForID(file);
  // Imaginary/remapped buffers have no stable include-search identity. A
  // content override must not inherit the physical file's trusted identity.
  if (entry == nullptr || sourceManager.isFileOverridden(entry))
    return;
  if (exactCompat) {
    hasCudaCompatHeader = true;
    if (nvidiaSampleHelperIncludes.empty())
      hasCudaCompatHeaderBeforeNvidiaHelper = true;
  }
  const llvm::sys::fs::UniqueID &uniqueId = entry->getUniqueID();
  if (uniqueId.getDevice() == 0 && uniqueId.getFile() == 0)
    return;
  const std::pair<std::uint64_t, std::uint64_t> identity(
      uniqueId.getDevice(), uniqueId.getFile());
  const bool wasTrusted =
      initiallyTrustedSystemFileIdentities.count(identity) != 0;
  const bool wasUntrusted =
      initiallyUntrustedSystemFileIdentities.count(identity) != 0;
  if (!wasTrusted && !wasUntrusted) {
    if (frozenRole == FrozenOfficialNvidiaSampleHelperFunctions &&
        frozenNvidiaSampleDirectFunctionsRootIdentities.count(identity) != 0) {
      frozenOfficialNvidiaSampleHelperFunctionsSeen = true;
      frozenNvidiaSampleDirectFunctionsRootFileIds.insert(
          file.getHashValue());
    }
    if (frozenRole == FrozenOfficialNvidiaSampleHelperImage)
      frozenNvidiaSampleMacroProviderIdentities.insert(identity);
    if (fileType == clang::SrcMgr::C_System ||
        fileType == clang::SrcMgr::C_ExternCSystem)
      initiallyTrustedSystemFileIdentities.insert(identity);
    else
      initiallyUntrustedSystemFileIdentities.insert(identity);
  }
  if (wasTrusted ||
      initiallyTrustedSystemFileIdentities.count(identity) != 0)
    trustedSystemFileIds.insert(file.getHashValue());
}

void AscifyAction::InclusionDirective(clang::SourceLocation hash_loc,
                                      const clang::Token&,
                                      StringRef file_name,
                                      bool is_angled,
                                      clang::CharSourceRange filename_range,
                                      StringRef resolved_file_name, StringRef,
                                      StringRef, const clang::Module*) {
  std::string frontendCompatibilityError;
  if (!ascify::ValidateFrontendCompatibilityInclude(
          frontendCompatibility, file_name.str(), resolved_file_name.str(),
          frontendCompatibilityError)) {
    clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error, "%0");
    DE.Report(filename_range.getBegin(), ID) << frontendCompatibilityError;
    return;
  }
  outs() << "File included: " << file_name << "\n";
  auto &SM = getCompilerInstance().getSourceManager();
  const bool recognizedNvidiaHelper =
      isRecognizedNvidiaSampleHelperCuda(resolved_file_name);
  if (!SM.isWrittenInMainFile(hash_loc)) {
    if (recognizedNvidiaHelper) {
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: recognized helper was "
          << "included transitively; all helper edits kept\n";
    }
    return;
  }
  const clang::SourceLocation filenameBegin = filename_range.getBegin();
  const clang::SourceLocation filenameEnd = filename_range.getEnd();
  const clang::SourceLocation hashFile = SM.getFileLoc(hash_loc);
  const clang::SourceLocation beginFile = SM.getFileLoc(filenameBegin);
  const clang::SourceLocation endFile = SM.getFileLoc(filenameEnd);
  bool invalidSpelling = false;
  const llvm::StringRef spelling = clang::Lexer::getSourceText(
      filename_range, SM, getCompilerInstance().getLangOpts(),
      &invalidSpelling);
  const std::string filename = file_name.str();
  const std::string quoted = "\"" + filename + "\"";
  const std::string angled = "<" + filename + ">";
  const std::string quotedWithoutEnd = "\"" + filename;
  const std::string angledWithoutEnd = "<" + filename;
  const bool directSpelling =
      !invalidSpelling &&
      (spelling == file_name || spelling == quoted ||
       spelling == angled || spelling == quotedWithoutEnd ||
       spelling == angledWithoutEnd);
  const bool directRange =
      hash_loc.isFileID() && filenameBegin.isFileID() &&
      filenameEnd.isFileID() && hashFile.isValid() &&
      beginFile.isValid() && endFile.isValid() &&
      SM.isWrittenInMainFile(beginFile) &&
      SM.isWrittenInMainFile(endFile) &&
      SM.getFileID(hashFile) == SM.getFileID(beginFile) &&
      SM.getFileID(beginFile) == SM.getFileID(endFile);
  if (file_name == "helper_functions.h" && directRange && directSpelling &&
      preprocessorConditionalDepth == 0 &&
      isFrozenOfficialNvidiaSampleHelperFunctionsPath(resolved_file_name)) {
    llvm::sys::fs::UniqueID directIdentity;
    if (!llvm::sys::fs::getUniqueID(resolved_file_name, directIdentity) &&
        !(directIdentity.getDevice() == 0 && directIdentity.getFile() == 0)) {
      frozenNvidiaSampleDirectFunctionsRootIdentities.insert(
          {directIdentity.getDevice(), directIdentity.getFile()});
    }
  }
  if (recognizedNvidiaHelper) {
    if (file_name == "helper_cuda.h" && directRange && directSpelling) {
      nvidiaSampleHelperIncludes.push_back({hash_loc, filenameEnd});
      recognizedNvidiaSampleHelperPaths.insert(resolved_file_name.str());
      auditActiveAscifyCudaCompatMacrosAtInsertion();
    } else {
      // A TU may contain both a removable direct include and another include
      // that reaches the same recognized helper through a macro. The latter
      // cannot be edited by a direct source range, so it invalidates the
      // entire helper transaction instead of being ignored in cardinality.
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: helper include is not a "
          << "direct main-file literal; include and calls kept\n";
    }
  }
  if (file_name == "ascify/ascify_cuda_compat.hpp" &&
      !isExactAscifyCudaCompatPath(resolved_file_name)) {
    ascifyCudaCompatIncludeConflict = true;
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify CUDA compat closure: include spelling resolves to an "
        << "unrecognized header; helper edits kept\n";
  }
  if (file_name == ascify::DavC310TargetRecipe::TargetHeader)
    hasDavC310TargetHeader = true;
  if (file_name == ascify::DavC310TargetRecipe::SimdTargetHeader)
    hasDavC310SimdTargetHeader = true;
  if (file_name == "cub/cub.cuh" ||
      file_name == "acl_cub/aclcub.hpp") {
    hasCubCompatHeader = true;
    if (firstCubCompatHeaderLoc.isInvalid())
      firstCubCompatHeaderLoc = hash_loc;
  }
  if (!firstHeader) {
    firstHeader = true;
    firstHeaderLoc = hash_loc;
  }
  const auto found = CUDA_INCLUDE_MAP.find(file_name);
  if (found != CUDA_INCLUDE_MAP.end()) {
    bool exclude = Exclude(found->second);
    Statistics::current().incrementCounter(found->second, file_name.str());
    clang::SourceLocation sl = filename_range.getBegin();

    if (Statistics::isUnsupported(found->second)) {
      clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
      std::string sWarn = sDPP;
      const auto ID = DE.getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "'%0' is unsupported header in '%1'.");
      DE.Report(sl, ID) << found->first << sWarn;
      return;
    }
    clang::StringRef newInclude;
    // Keep the same include type that the user gave.
    if (!exclude) {
      clang::SmallString<128> includeBuffer;
      llvm::StringRef name = found->second.dppName;
      if (is_angled)
        newInclude = llvm::Twine("<" + name + ">").toStringRef(includeBuffer);
      else
        newInclude =
            llvm::Twine("\"" + name + "\"").toStringRef(includeBuffer);
    } else {
      // hashLoc is location of the '#', thus replacing the whole include
      // directive by empty newInclude starting with '#'.
      sl = hash_loc;
    }
    const char *B = SM.getCharacterData(sl);
    const char *E = SM.getCharacterData(filename_range.getEnd());
    ct::Replacement Rep(SM, sl, E - B, newInclude.str());
    const clang::SourceLocation replacementEnd =
        sl.getLocWithOffset(static_cast<int>(E - B));
    // The raw identifier pass also sees tokens inside include spellings (for
    // example both `cub` tokens in <cub/cub.cuh>).  Make the preprocessor's
    // include rewrite the sole owner of that span.
    insertSemanticReplacement(
        Rep, clang::FullSourceLoc{sl, SM}, sl, replacementEnd);
    return;
  }

  if (localHeaderContext == nullptr)
    return;

  const clang::SourceLocation fileLoc =
      SM.getFileLoc(filename_range.getBegin());
  const bool isLiteral = !filename_range.getBegin().isMacroID();
  const unsigned sourceOffset = fileLoc.isValid() ? SM.getFileOffset(fileLoc)
                                                   : 0;
  const std::string resolvedPath = resolved_file_name.str();
  const LocalHeaderIncludeDecision decision = localHeaderContext->observe(
      sourceOffset, file_name.str(), resolvedPath, is_angled, isLiteral);
  if (decision.fatal) {
    clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                       "local-header closure failed: %0");
    DE.Report(hash_loc, ID) << decision.diagnostic;
    return;
  }
  if (!decision.redirect)
    return;

  clang::SourceLocation sl = filename_range.getBegin();
  const char *B = SM.getCharacterData(sl);
  const char *E = SM.getCharacterData(filename_range.getEnd());
  ct::Replacement Rep(SM, sl, E - B, decision.emittedSpelling);
  const clang::SourceLocation replacementEnd =
      sl.getLocWithOffset(static_cast<int>(E - B));
  insertSemanticReplacement(
      Rep, clang::FullSourceLoc{sl, SM}, sl, replacementEnd);
}

void AscifyAction::PragmaDirective(
    clang::SourceLocation Loc,
    clang::PragmaIntroducerKind Introducer) {
  finalizePendingNvidiaSampleHelperPragma();
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::SourceLocation expansion =
      sourceManager.getExpansionLoc(Loc);
  if (Introducer != clang::PIK_HashPragma) {
    // _Pragma/__pragma tokens are lexed from a scratch buffer.  Admit only a
    // diagnostic pragma applied inside a file that was a system header on its
    // first EnterFile callback.  The matching specialized callback below
    // clears this pending bit; every other macro-generated pragma fails closed
    // at the next pragma or at EndSourceFileAction.
    if (!locationComesFromInitiallyTrustedSystemFile(
            sourceManager, expansion, trustedSystemFileIds)) {
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: untrusted "
          << "macro-generated pragma keeps all helper edits\n";
      return;
    }
    nvidiaSampleHelperPendingMacroGeneratedPragma = true;
    return;
  }
  const clang::SourceLocation fileLocation =
      sourceManager.getFileLoc(expansion);
  if (fileLocation.isInvalid()) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    return;
  }
  bool invalidBuffer = false;
  const clang::FileID file = sourceManager.getFileID(fileLocation);
  const llvm::StringRef buffer =
      sourceManager.getBufferData(file, &invalidBuffer);
  if (invalidBuffer) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    return;
  }
  const unsigned offset = sourceManager.getFileOffset(fileLocation);
  if (offset >= buffer.size()) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    return;
  }
  std::string logicalLine;
  for (size_t index = offset; index < buffer.size();) {
    const char current = buffer[index];
    if (current == '?' && index + 2 < buffer.size() &&
        buffer[index + 1] == '?' && buffer[index + 2] == '/') {
      // With -trigraphs, the question-question-slash trigraph becomes a
      // backslash before line splicing.  The SourceManager buffer retains the
      // phase-1 spelling, so do not reconstruct this pragma.
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: trigraph pragma spelling "
          << "keeps all helper edits\n";
      return;
    }
    if (current == '/' && index + 1 < buffer.size() &&
        buffer[index + 1] == '*') {
      // A block comment may span a physical newline while the surrounding
      // preprocessing directive continues after comment replacement.  Keep
      // the transaction instead of trying to duplicate Clang's comment lexer.
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: block-comment pragma "
          << "spelling keeps all helper edits\n";
      return;
    }
    if (current == '\\') {
      size_t newline = index + 1;
      while (newline < buffer.size() &&
             (buffer[newline] == ' ' || buffer[newline] == '\t' ||
              buffer[newline] == '\f' || buffer[newline] == '\v'))
        ++newline;
      if (newline < buffer.size() && buffer[newline] == '\n') {
        index = newline + 1;
        continue;
      }
      if (newline < buffer.size() && buffer[newline] == '\r') {
        index = newline + 1;
        if (index < buffer.size() && buffer[index] == '\n')
          ++index;
        continue;
      }
    }
    if (current == '\r' || current == '\n')
      break;
    logicalLine.push_back(current);
    ++index;
  }
  const llvm::StringRef line(logicalLine);
  std::string macroStackOperation;
  std::string macroStackTarget;
  if (pragmaLineHasUnsafeMacroStackOperation(
          line, macroStackOperation, macroStackTarget)) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: pragma macro-stack "
        << "operation keeps all helper edits: " << macroStackOperation;
    if (!macroStackTarget.empty())
      llvm::errs() << " of observable '" << macroStackTarget << "'";
    else
      llvm::errs() << " has an unresolved target";
    llvm::errs() << "\n";
    return;
  }
  std::string observedPragmaMacro;
  if (pragmaLineContainsObservableIdentifier(line, observedPragmaMacro)) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: pragma observation of '"
        << observedPragmaMacro << "' keeps all helper edits\n";
    return;
  }
  std::string pragmaOperation;
  if (!isDirectlyAllowedHashPragma(
          line, pragmaOperation,
          locationComesFromInitiallyTrustedSystemFile(
              sourceManager, fileLocation, trustedSystemFileIds))) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: unproven hash pragma '"
        << pragmaOperation << "' keeps all helper edits at "
        << fileLocation.printToString(sourceManager) << ": " << line << "\n";
  }
}

void AscifyAction::PragmaDiagnostic() {
  // PPCallbacks emits this immediately after PragmaDirective for a parsed
  // diagnostic push, pop, or mapping.  It is the only non-hash pragma class
  // admitted by the helper transaction.
  nvidiaSampleHelperPendingMacroGeneratedPragma = false;
}

void AscifyAction::finalizePendingNvidiaSampleHelperPragma() {
  if (!nvidiaSampleHelperPendingMacroGeneratedPragma)
    return;
  nvidiaSampleHelperPendingMacroGeneratedPragma = false;
  nvidiaSampleHelperUnsupportedMacroUse = true;
  llvm::errs()
      << "Ascify NVIDIA sample-helper closure: unresolved trusted-system "
      << "macro-generated pragma keeps all helper edits\n";
}

bool AscifyAction::cudaLaunchKernel(const mat::MatchFinder::MatchResult &Result) {
  return true;
}

bool AscifyAction::lowerCudaGlobalScalarDoubleParam(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *param =
      Result.Nodes.getNodeAs<clang::ParmVarDecl>(sCudaGlobalScalarDoubleParam);
  if (param == nullptr)
    return false;

  const auto *function =
      llvm::dyn_cast<clang::FunctionDecl>(param->getDeclContext());
  if (function == nullptr || !function->hasAttr<clang::CUDAGlobalAttr>())
    return false;

  const clang::TypeSourceInfo *typeSource = param->getTypeSourceInfo();
  if (typeSource == nullptr)
    return false;

  const clang::BuiltinTypeLoc builtinTypeLoc =
      typeSource->getTypeLoc().getUnqualifiedLoc().getAs<clang::BuiltinTypeLoc>();
  if (!builtinTypeLoc ||
      builtinTypeLoc.getTypePtr()->getKind() != clang::BuiltinType::Double)
    return false;

  auto &SM = getCompilerInstance().getSourceManager();
  const clang::LangOptions &LO = getCompilerInstance().getLangOpts();
  const clang::SourceLocation doubleLoc = builtinTypeLoc.getBuiltinLoc();
  if (doubleLoc.isInvalid() || doubleLoc.isMacroID() ||
      !SM.isWrittenInMainFile(doubleLoc))
    return false;

  const llvm::StringRef spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(
          clang::SourceRange(doubleLoc, doubleLoc)),
      SM, LO);
  if (spelling != "double")
    return false;

  const unsigned offset = SM.getFileOffset(doubleLoc);
  if (!loweredDeviceDoubleParamOffsets.insert(offset).second)
    return true;

  static const dppCounter counter = {
      "float", CONV_DEVICE_TYPE, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(
      counter, "CUDA __global__ by-value scalar double parameter");

  clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
  const auto ID = DE.getCustomDiagID(
      clang::DiagnosticsEngine::Warning,
      "lowering by-value scalar double parameter '%0' of CUDA __global__ "
      "function '%1' to float for Ascend SIMT; use "
      "'-no-lower-device-double-params' to disable");
  DE.Report(doubleLoc, ID)
      << param->getName() << function->getQualifiedNameAsString();

  ct::Replacement Rep(SM, doubleLoc, spelling.size(), "float");
  const clang::SourceLocation doubleEnd =
      clang::Lexer::getLocForEndOfToken(doubleLoc, 0, SM, LO);
  if (!insertSemanticReplacement(
          Rep, clang::FullSourceLoc(doubleLoc, SM), doubleLoc, doubleEnd)) {
    loweredDeviceDoubleParamOffsets.erase(offset);
    return false;
  }
  return true;
}

bool AscifyAction::rewriteCudaDefaultDim3(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *variable =
      Result.Nodes.getNodeAs<clang::VarDecl>(sCudaDefaultDim3);
  if (variable == nullptr || Result.SourceManager == nullptr ||
      variable->isImplicit() || llvm::isa<clang::ParmVarDecl>(variable) ||
      variable->getIdentifier() == nullptr || !variable->isLocalVarDecl() ||
      variable->hasExternalStorage())
    return false;

  // This lowering repairs host launch-shape declarations only. Global,
  // device, kernel, shared, and extern declarations retain their source
  // semantics and fail closed if the target cannot compile them.
  const auto *function =
      llvm::dyn_cast<clang::FunctionDecl>(variable->getDeclContext());
  if (function == nullptr ||
      function->hasAttr<clang::CUDADeviceAttr>() ||
      function->hasAttr<clang::CUDAGlobalAttr>())
    return false;

  auto &sourceManager = *Result.SourceManager;
  const clang::CXXRecordDecl *record =
      variable->getType().getUnqualifiedType()->getAsCXXRecordDecl();
  if (record == nullptr || record->getQualifiedNameAsString() != "dim3" ||
      record->getLocation().isInvalid() ||
      !sourceManager.isInSystemHeader(record->getLocation()))
    return false;

  const clang::SourceLocation nameLocation = variable->getLocation();
  if (nameLocation.isInvalid() || nameLocation.isMacroID() ||
      !sourceManager.isWrittenInMainFile(nameLocation))
    return false;

  const clang::LangOptions &langOptions =
      getCompilerInstance().getLangOpts();
  const clang::SourceLocation nameEnd =
      clang::Lexer::getLocForEndOfToken(
          nameLocation, 0, sourceManager, langOptions);
  if (nameEnd.isInvalid())
    return false;

  // A CUDA default-initialized declaration has no source initializer. Match
  // that syntax directly instead of depending on whether a Clang release
  // materializes the implicit CXXConstructExpr in the AST. Explicit (), {},
  // assignment, macro, attribute, array, and user-defined dim3 forms remain
  // untouched.
  clang::Token nextToken;
  if (clang::Lexer::getRawToken(
          nameEnd, nextToken, sourceManager, langOptions,
          /*IgnoreWhiteSpace=*/true) ||
      (nextToken.getKind() != clang::tok::comma &&
       nextToken.getKind() != clang::tok::semi))
    return false;

  const llvm::StringRef sourceName = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(
          clang::SourceRange(nameLocation, nameLocation)),
      sourceManager, langOptions);
  if (sourceName != variable->getName())
    return false;

  const unsigned offset = sourceManager.getFileOffset(nameLocation);
  if (!rewrittenCudaDefaultDim3Offsets.insert(offset).second)
    return true;

  static const dppCounter counter = {
      "dim3", CONV_TYPE, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(
      counter, "CUDA host-local default dim3 construction");

  const std::string replacement =
      sourceName.str() + " = dim3(1U, 1U, 1U)";
  ct::Replacement Rep(
      sourceManager, nameLocation, sourceName.size(), replacement);
  if (!insertSemanticReplacement(
          Rep, clang::FullSourceLoc(nameLocation, sourceManager),
          nameLocation, nameEnd)) {
    rewrittenCudaDefaultDim3Offsets.erase(offset);
    return false;
  }
  return true;
}

bool AscifyAction::rewriteProvenGlobalAtomicCall(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *call =
      Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr)
    return false;

  const clang::FunctionDecl *callee = call->getDirectCallee();
  if (callee == nullptr || callee->getIdentifier() == nullptr)
    return false;
  const GlobalAtomicSpec *spec =
      findGlobalAtomicSpec(callee->getName());
  if (spec == nullptr || call->getNumArgs() != spec->argumentCount ||
      callee->getQualifiedNameAsString() != spec->cudaName ||
      !callee->hasAttr<clang::CUDADeviceAttr>())
    return false;

  auto &context = *Result.Context;
  auto &sourceManager = *Result.SourceManager;
  if (callee->getLocation().isInvalid() ||
      !sourceManager.isInSystemHeader(callee->getLocation()))
    return false;
  const clang::FunctionDecl *kernel =
      enclosingNonLambdaFunction(call, context);
  if (kernel == nullptr || !kernel->hasAttr<clang::CUDAGlobalAttr>() ||
      call->getArg(0) == nullptr)
    return false;

  const GlobalAtomicScalarKind kind = classifyGlobalAtomicPointer(
      call->getArg(0)->getType(), context);
  if (kind == GlobalAtomicScalarKind::Unsupported ||
      (spec->unsignedOnly && kind != GlobalAtomicScalarKind::UInt32) ||
      !hasExactGlobalAtomicSignature(callee, *spec, kind, context))
    return false;
  for (unsigned index = 1; index < spec->argumentCount; ++index) {
    if (classifyGlobalAtomicScalar(
            call->getArg(index)->getType(), context) != kind)
      return false;
  }
  if (!isAddressRootedAtCurrentGlobalParameter(
          call->getArg(0), kernel, context))
    return false;

  const clang::Expr *calleeExpression =
      stripParenAndImplicitCasts(call->getCallee());
  const auto *reference =
      llvm::dyn_cast_or_null<clang::DeclRefExpr>(calleeExpression);
  if (reference == nullptr || reference->getDecl()->getCanonicalDecl() !=
                                  callee->getCanonicalDecl())
    return false;
  const clang::SourceLocation identifierLocation =
      reference->getLocation();
  if (identifierLocation.isInvalid() || identifierLocation.isMacroID() ||
      !sourceManager.isWrittenInMainFile(identifierLocation))
    return false;

  const clang::LangOptions &languageOptions =
      getCompilerInstance().getLangOpts();
  const llvm::StringRef spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(
          clang::SourceRange(identifierLocation, identifierLocation)),
      sourceManager, languageOptions);
  if (spelling != spec->cudaName)
    return false;
  const unsigned offset = sourceManager.getFileOffset(identifierLocation);
  if (!rewrittenGlobalAtomicOffsets.insert(offset).second)
    return true;

  const clang::SourceLocation identifierEnd =
      clang::Lexer::getLocForEndOfToken(
          identifierLocation, 0, sourceManager, languageOptions);
  if (identifierEnd.isInvalid()) {
    rewrittenGlobalAtomicOffsets.erase(offset);
    return false;
  }
  const std::string replacement =
      std::string("ascify::") + spec->wrapperName;
  ct::Replacement edit(
      sourceManager, identifierLocation, spelling.size(), replacement);
  if (!insertSemanticReplacement(
          edit,
          clang::FullSourceLoc(identifierLocation, sourceManager),
          identifierLocation,
          identifierEnd)) {
    rewrittenGlobalAtomicOffsets.erase(offset);
    return false;
  }

  static const dppCounter counter = {
      "ascify global atomic wrapper",
      CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(
      counter, std::string("AST-proven ") + spec->cudaName);
  needsCudaCompatHeader = true;
  return true;
}

bool AscifyAction::rewriteCanonicalWarpAddReduction(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *loop =
      Result.Nodes.getNodeAs<clang::ForStmt>(sCanonicalWarpAddReduction);
  if (loop == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr)
    return false;

  const clang::VarDecl *accumulator = nullptr;
  if (!matchCanonicalWarpAddReduction(
          loop, *Result.Context, *Result.SourceManager, accumulator) ||
      accumulator == nullptr || accumulator->getName().empty())
    return false;

  auto &SM = *Result.SourceManager;
  const clang::LangOptions &LO = getCompilerInstance().getLangOpts();
  const auto *function =
      llvm::dyn_cast<clang::FunctionDecl>(accumulator->getDeclContext());
  if (function == nullptr)
    return false;
  const clang::Stmt *ownedStatement =
      directFunctionBodyStatement(loop, function);
  if (ownedStatement == nullptr)
    return false;
  const clang::SourceLocation begin =
      canonicalWarpReductionReplacementBegin(
          ownedStatement, loop, SM, LO);
  const clang::SourceLocation endToken =
      SM.getFileLoc(ownedStatement->getEndLoc());
  const clang::SourceLocation end =
      clang::Lexer::getLocForEndOfToken(endToken, 0, SM, LO);
  if (begin.isInvalid() || end.isInvalid())
    return false;

  const unsigned offset = SM.getFileOffset(begin);
  if (!rewrittenWarpAddReductionOffsets.insert(offset).second)
    return true;

  const unsigned endOffset = SM.getFileOffset(end);
  if (endOffset <= offset) {
    rewrittenWarpAddReductionOffsets.erase(offset);
    return false;
  }

  const std::string name = accumulator->getNameAsString();
  std::string indentation;
  if (begin != SM.getFileLoc(loop->getBeginLoc())) {
    const unsigned loopColumn =
        SM.getSpellingColumnNumber(loop->getBeginLoc());
    if (loopColumn > 1)
      indentation.assign(loopColumn - 1, ' ');
  }
  const std::string replacementText =
      indentation + name + " = ascify::warp_reduce_add(" + name + ");";
  ct::Replacement Rep(SM, begin, endOffset - offset, replacementText);
  if (!insertSemanticReplacement(
          Rep, clang::FullSourceLoc(begin, SM), begin, end)) {
    rewrittenWarpAddReductionOffsets.erase(offset);
    return false;
  }

  static const dppCounter counter = {
      "ascify::warp_reduce_add", CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(
      counter, "canonical full-warp add reduction");
  needsCudaCompatHeader = true;
  return true;
}

bool AscifyAction::tagCanonicalBinaryReducer(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *record =
      Result.Nodes.getNodeAs<clang::CXXRecordDecl>(sCanonicalBinaryReducer);
  if (!hasCubCompatHeader || record == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr)
    return false;

  const CanonicalReducerKind kind = classifyCanonicalBinaryReducer(
      record, *Result.Context, *Result.SourceManager);
  const llvm::StringRef tag = canonicalReducerTagName(kind);
  if (tag.empty())
    return false;

  auto &SM = *Result.SourceManager;
  const clang::SourceLocation includeLoc =
      SM.getFileLoc(firstCubCompatHeaderLoc);
  const clang::SourceLocation recordLoc =
      SM.getFileLoc(record->getBeginLoc());
  const clang::SourceLocation closeBrace =
      SM.getFileLoc(record->getBraceRange().getEnd());
  if (includeLoc.isInvalid() || recordLoc.isInvalid() ||
      closeBrace.isInvalid() ||
      SM.getFileID(includeLoc) != SM.getFileID(recordLoc) ||
      SM.getFileOffset(includeLoc) >= SM.getFileOffset(recordLoc) ||
      !SM.isWrittenInMainFile(closeBrace))
    return false;
  const unsigned offset = SM.getFileOffset(closeBrace);
  if (!taggedCanonicalReducerOffsets.insert(offset).second)
    return true;

  std::string annotation =
      "\n public:\n"
      "  using ascify_reduction_tag = ::aclcub::";
  annotation += tag.str();
  annotation += ";\n"
                "  using ascify_reduction_value_type = ";
  annotation += record->getDescribedClassTemplate()
                    ->getTemplateParameters()
                    ->getParam(0)
                    ->getNameAsString();
  annotation += ";\n"
                "  using ascify_reduction_owner_type = ";
  annotation += record->getNameAsString();
  annotation += ";\n";
  ct::Replacement Rep(SM, closeBrace, 0, annotation);
  if (!insertReplacement(
          Rep, clang::FullSourceLoc(closeBrace, SM))) {
    taggedCanonicalReducerOffsets.erase(offset);
    return false;
  }

  static const dppCounter counter = {
      "aclcub semantic reduction tag",
      CONV_DEVICE_FUNC, API_CUB, 0, FULL};
  Statistics::current().incrementCounter(
      counter, "canonical pure binary reduction functor");
  return true;
}

bool AscifyAction::cudaDeviceFuncCall(const mat::MatchFinder::MatchResult &Result) {
  return true;
}

bool AscifyAction::cubNamespacePrefix(const mat::MatchFinder::MatchResult &Result) {
  return true;
}

bool AscifyAction::cubUsingNamespaceDecl(const mat::MatchFinder::MatchResult &Result) {
  return false;
}

bool AscifyAction::cubFunctionTemplateDecl(const mat::MatchFinder::MatchResult &Result) {
  return false;
}

bool AscifyAction::cudaHostFuncCall(const mat::MatchFinder::MatchResult &Result) {
  return false;
}

bool AscifyAction::cudaOverloadedHostFuncCall(const mat::MatchFinder::MatchResult& Result) {
  return false;
}

bool AscifyAction::half2Member(const mat::MatchFinder::MatchResult &Result) {
  return false;
}

bool AscifyAction::dataTypeSelection(const mat::MatchFinder::MatchResult &Result) {
  return false;
}

bool AscifyAction::insertReplacement(const ct::Replacement &rep,
                                     const clang::FullSourceLoc &fullSL) {
  std::string errorMessage;
  if (!llcompat::insertReplacement(*replacements, rep, &errorMessage)) {
    clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "cannot apply Ascify source replacement: %0");
    DE.Report(fullSL, ID) << errorMessage;
    return false;
  }
  if (PrintStats || PrintStatsCSV) {
    rep.getLength();
    Statistics::current().lineTouched(fullSL.getExpansionLineNumber());
    Statistics::current().bytesChanged(rep.getLength());
  }
  return true;
}

bool AscifyAction::insertSemanticReplacement(
    const ct::Replacement &rep,
    const clang::FullSourceLoc &fullSL,
    clang::SourceLocation begin,
    clang::SourceLocation end) {
  auto &SM = getCompilerInstance().getSourceManager();
  begin = SM.getFileLoc(begin);
  end = SM.getFileLoc(end);
  if (begin.isInvalid() || end.isInvalid())
    return false;
  const std::pair<clang::FileID, unsigned> decomposedBegin =
      SM.getDecomposedLoc(begin);
  const std::pair<clang::FileID, unsigned> decomposedEnd =
      SM.getDecomposedLoc(end);
  if (decomposedBegin.first != decomposedEnd.first ||
      decomposedEnd.second <= decomposedBegin.second)
    return false;
  for (const SemanticRewriteRange &range : SemanticRewriteRanges) {
    if (range.file == decomposedBegin.first &&
        decomposedBegin.second < range.endOffset &&
        range.beginOffset < decomposedEnd.second) {
      clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
      const auto ID = DE.getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "skipping overlapping Ascify semantic rewrite");
      DE.Report(fullSL, ID);
      return false;
    }
  }
  if (!insertReplacement(rep, fullSL))
    return false;
  SemanticRewriteRanges.push_back(
      {decomposedBegin.first, decomposedBegin.second, decomposedEnd.second});
  return true;
}
std::unique_ptr<clang::ASTConsumer> AscifyAction::CreateASTConsumer(clang::CompilerInstance &CI, StringRef) {
  Finder.reset(new mat::MatchFinder);
  davC310TargetRecipe.reset();
  // Replace the <<<...>>> language extension with a hip kernel launch
  Finder->addMatcher(mat::cudaKernelCallExpr(mat::isExpansionInMainFile()).bind(sCudaLaunchKernel), this);
  if (!NoLowerDeviceDoubleParams) {
    Finder->addMatcher(
        mat::parmVarDecl(mat::isExpansionInMainFile())
            .bind(sCudaGlobalScalarDoubleParam),
        this);
  }
  Finder->addMatcher(
      mat::varDecl(mat::isExpansionInMainFile())
          .bind(sCudaDefaultDim3),
      this);
  // Keep this matcher independent of API spelling and perform the CUDA API,
  // scalar type, enclosing-kernel, and address-provenance proof in the
  // callback. A failed proof makes no edit, so shared/local/helper pointers
  // cannot acquire a forged global-memory address-space cast.
  Finder->addMatcher(
      mat::callExpr(mat::callee(mat::functionDecl()),
                    mat::isExpansionInMainFile())
          .bind(sProvenGlobalAtomicCall),
      this);
  if (TargetPolicy == "dav-c310-vec" && SimtMathMode == "fast") {
    // Keep the matcher broad and perform the semantic proof in the callback.
    // This prevents source-name coupling and makes every failed proof a
    // no-op that falls back to the translated shuffle loop.
    Finder->addMatcher(
        mat::forStmt(mat::isExpansionInMainFile())
            .bind(sCanonicalWarpAddReduction),
        this);
    Finder->addMatcher(
        mat::cxxRecordDecl(mat::isDefinition(),
                           mat::isExpansionInMainFile())
            .bind(sCanonicalBinaryReducer),
        this);
    davC310TargetRecipe.registerMatchers(*Finder, this);
  }
  return Finder->newASTConsumer();
}

void AscifyAction::Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD) {
  ConditionalDirectiveEntered();
  auditExternalNvidiaSampleHelperPreprocessorUse(
      Loc, MacroNameTok, "#ifndef");
  auditFrozenNvidiaSampleHelperMacroDependency(
      Loc, MacroNameTok, MD, "#ifndef");
}

void AscifyAction::Ifdef(clang::SourceLocation Loc,
                         const clang::Token &MacroNameTok,
                         const clang::MacroDefinition &MD) {
  ConditionalDirectiveEntered();
  auditExternalNvidiaSampleHelperPreprocessorUse(
      Loc, MacroNameTok, "#ifdef");
  auditFrozenNvidiaSampleHelperMacroDependency(
      Loc, MacroNameTok, MD, "#ifdef");
}

void AscifyAction::Elifdef(clang::SourceLocation Loc,
                           const clang::Token &MacroNameTok,
                           const clang::MacroDefinition &MD,
                           llvm::StringRef directive) {
  auditExternalNvidiaSampleHelperPreprocessorUse(
      Loc, MacroNameTok, directive);
  auditFrozenNvidiaSampleHelperMacroDependency(
      Loc, MacroNameTok, MD, directive);
}

void AscifyAction::ElifdefSkipped(
    clang::SourceLocation Loc,
    clang::SourceRange ConditionRange,
    llvm::StringRef directive) {
  auditSkippedNvidiaSampleHelperElifDirective(
      Loc, ConditionRange, directive);
}

void AscifyAction::ConditionalDirectiveEntered() {
  ++preprocessorConditionalDepth;
}

void AscifyAction::ConditionalDirectiveEnded() {
  if (preprocessorConditionalDepth != 0)
    --preprocessorConditionalDepth;
}

void AscifyAction::Defined(const clang::Token &MacroNameTok,
                           const clang::MacroDefinition &MD,
                           clang::SourceRange Range) {
  auditExternalNvidiaSampleHelperPreprocessorUse(
      Range.getBegin(), MacroNameTok, "defined");
  auditFrozenNvidiaSampleHelperMacroDependency(
      Range.getBegin(), MacroNameTok, MD, "defined");
}

void AscifyAction::MacroUndefined(
    const clang::Token &MacroNameTok,
    const clang::MacroDefinition &MD) {
  auditExternalNvidiaSampleHelperPreprocessorUse(
      MacroNameTok.getLocation(), MacroNameTok, "#undef");
  auditFrozenNvidiaSampleHelperMacroDependency(
      MacroNameTok.getLocation(), MacroNameTok, MD, "#undef");
}

void AscifyAction::MacroDefined(const clang::Token &MacroNameTok) {
  if (MacroNameTok.getIdentifierInfo() == nullptr)
    return;
  const llvm::StringRef name =
      MacroNameTok.getIdentifierInfo()->getName();
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const bool fromCompat = locationComesFromAscifyCudaCompat(
      sourceManager, MacroNameTok.getLocation());
  if (isAscifyCudaCompatReservedMacro(name) && !fromCompat) {
      nvidiaSampleHelperOutputMacroEverDefined = true;
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: reserved output macro '"
          << name << "' was already defined outside Ascify compat; all "
          << "helper edits kept\n";
  }
  const unsigned frozenMacroGraph =
      FrozenOfficialNvidiaSampleHelperCuda |
      FrozenOfficialNvidiaSampleHelperString |
      FrozenOfficialNvidiaSampleHelperFunctions |
      FrozenOfficialNvidiaSampleHelperImage |
      FrozenOfficialNvidiaSampleHelperTimer;
  if (isNvidiaSampleHelperObservableMacro(name) && !fromCompat &&
      !locationComesFromFrozenNvidiaSampleHelperFile(
          sourceManager, MacroNameTok.getLocation(), frozenMacroGraph)) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: external definition of "
        << "observable helper macro '" << name
        << "' keeps all helper edits\n";
  }
  if (!fromCompat && !isAscifyCudaCompatReservedMacro(name) &&
      isAscifyCudaCompatIdentifier(name)) {
    const clang::SourceLocation definitionLocation =
        sourceManager.getSpellingLoc(MacroNameTok.getLocation());
    const unsigned frozenMacroGraph =
        FrozenOfficialNvidiaSampleHelperCuda |
        FrozenOfficialNvidiaSampleHelperString |
        FrozenOfficialNvidiaSampleHelperFunctions |
        FrozenOfficialNvidiaSampleHelperImage |
        FrozenOfficialNvidiaSampleHelperTimer;
    const clang::FileID definitionFile = definitionLocation.isValid()
        ? sourceManager.getFileID(definitionLocation)
        : clang::FileID();
#if LLVM_VERSION_MAJOR >= 8
    const clang::FileID predefinesFile =
        getCompilerInstance().getPreprocessor().getPredefinesFileID();
    const bool compilerPredefinition =
        definitionFile.isValid() && predefinesFile.isValid() &&
        definitionFile == predefinesFile &&
        !sourceManager.isWrittenInCommandLineFile(definitionLocation);
#else
    (void)definitionFile;
    const bool compilerPredefinition = false;
#endif
    const bool trustedDefinition =
        compilerPredefinition ||
        (definitionLocation.isValid() &&
         (locationComesFromInitiallyTrustedSystemFile(
              sourceManager, definitionLocation, trustedSystemFileIds) ||
          locationComesFromFrozenNvidiaSampleHelperFile(
              sourceManager, definitionLocation, frozenMacroGraph)));
    if (!trustedDefinition && !ascifyCudaCompatActiveMacroConflict) {
      ascifyCudaCompatActiveMacroConflict = true;
      ascifyCudaCompatActiveMacroConflictName = name.str();
      llvm::errs()
          << "Ascify CUDA compat closure: untrusted input macro '" << name
          << "' collides with the frozen compat identifier surface\n";
    }
  }
  if (TargetRecipe != ascify::DavC310TargetRecipe::SimdRecipeName)
    return;
  if (isRowwiseSimdReservedMacro(name))
    rowwiseSimdMacrosEverDefined.insert(name.str());
}

void AscifyAction::auditActiveAscifyCudaCompatMacrosAtInsertion() {
  if (hasCudaCompatHeaderBeforeNvidiaHelper ||
      ascifyCudaCompatActiveMacroConflict)
    return;
  clang::Preprocessor &preprocessor =
      getCompilerInstance().getPreprocessor();
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const unsigned frozenMacroGraph =
      FrozenOfficialNvidiaSampleHelperCuda |
      FrozenOfficialNvidiaSampleHelperString |
      FrozenOfficialNvidiaSampleHelperFunctions |
      FrozenOfficialNvidiaSampleHelperImage |
      FrozenOfficialNvidiaSampleHelperTimer;
  for (const char *name : AscifyCudaCompatIdentifiers) {
    const clang::IdentifierInfo *identifier =
        preprocessor.getIdentifierInfo(name);
    if (identifier == nullptr)
      continue;
    const clang::MacroInfo *macroInfo =
        preprocessor.getMacroDefinition(identifier).getMacroInfo();
    if (macroInfo == nullptr || macroInfo->isBuiltinMacro())
      continue;
    const clang::SourceLocation definitionLocation =
        sourceManager.getSpellingLoc(macroInfo->getDefinitionLoc());
    const clang::FileID definitionFile = definitionLocation.isValid()
        ? sourceManager.getFileID(definitionLocation)
        : clang::FileID();
#if LLVM_VERSION_MAJOR >= 8
    const clang::FileID predefinesFile = preprocessor.getPredefinesFileID();
    const bool compilerPredefinition =
        definitionFile.isValid() && predefinesFile.isValid() &&
        definitionFile == predefinesFile &&
        !sourceManager.isWrittenInCommandLineFile(definitionLocation);
#else
    (void)definitionFile;
    const bool compilerPredefinition = false;
#endif
    const bool trustedSystemDefinition =
        definitionLocation.isValid() &&
        locationComesFromInitiallyTrustedSystemFile(
            sourceManager, definitionLocation, trustedSystemFileIds);
    const bool frozenHelperDefinition =
        definitionLocation.isValid() &&
        locationComesFromFrozenNvidiaSampleHelperFile(
            sourceManager, definitionLocation, frozenMacroGraph);
    if (compilerPredefinition || trustedSystemDefinition ||
        frozenHelperDefinition ||
        locationComesFromAscifyCudaCompat(
            sourceManager, definitionLocation))
      continue;
    ascifyCudaCompatActiveMacroConflict = true;
    ascifyCudaCompatActiveMacroConflictName = name;
    llvm::errs()
        << "Ascify CUDA compat closure: active untrusted macro '" << name
        << "' collides with the frozen compat header at the helper include\n";
    return;
  }
}

void AscifyAction::MacroExpands(const clang::Token &MacroNameTok,
                                const clang::MacroDefinition &MD,
                                clang::SourceRange Range) {
  if (MacroNameTok.getIdentifierInfo() == nullptr)
    return;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::MacroInfo *macroInfo = MD.getMacroInfo();
  if (isAscifyCudaCompatPublishedMacro(
          MacroNameTok.getIdentifierInfo()->getName()))
    auditExternalNvidiaSampleHelperPreprocessorUse(
        MacroNameTok.getLocation(), MacroNameTok, "expansion");
  auditFrozenNvidiaSampleHelperMacroDependency(
      MacroNameTok.getLocation(), MacroNameTok, MD, "expansion");
  if (macroInfo == nullptr ||
      !locationComesFromRecognizedNvidiaSampleHelper(
          sourceManager, macroInfo->getDefinitionLoc()))
    return;
  const llvm::StringRef name =
      MacroNameTok.getIdentifierInfo()->getName();
  const clang::SourceLocation macroLocation =
      sourceManager.getExpansionLoc(MacroNameTok.getLocation());
  if (macroLocation.isInvalid()) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    return;
  }
  if (!sourceManager.isWrittenInMainFile(macroLocation)) {
    // NVIDIA cuda-samples v13.3 also has a reversed direct-include order:
    // helper_cuda.h followed by helper_functions.h. In that exact graph,
    // helper_cuda's MAX is active while the frozen helper_image body expands
    // MAX internally. Removing helper_cuda makes the same exact image file
    // publish its own equivalent MAX before that use. Admit only this one
    // order-dependent internal expansion: exact producer and consumer roles,
    // exact active producer body, an observed exact helper_functions root,
    // and the first physical identity of the non-overridden image provider.
    bool admittedFrozenInternalMaxExpansion = false;
    if (name == "MAX" &&
        frozenOfficialNvidiaSampleHelperFunctionsSeen &&
        frozenNvidiaSampleHelperFileRole(sourceManager, macroLocation) ==
            FrozenOfficialNvidiaSampleHelperImage &&
        frozenNvidiaSampleHelperFileRole(
            sourceManager, macroInfo->getDefinitionLoc()) ==
            FrozenOfficialNvidiaSampleHelperCuda &&
        activeFrozenHelperCudaMaxBodyMatches(
            *macroInfo, getCompilerInstance().getPreprocessor())) {
      admittedFrozenInternalMaxExpansion =
          isDirectFrozenFunctionsImageProviderInstance(macroLocation);
    }
    if (admittedFrozenInternalMaxExpansion)
      return;
    // Ignore implementation-internal expansions from the recognized helper,
    // but never remove the helper when an arbitrary local/user header relies
    // on one of its macros: this prototype does not rewrite that header.
    if (!locationComesFromRecognizedNvidiaSampleHelper(
            sourceManager, macroLocation)) {
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: helper macro expansion "
          << "outside the main file keeps all helper edits\n";
    }
    return;
  }
  if (name != "checkCudaErrors" && name != "getLastCudaError") {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    return;
  }
  // Only rewrite a macro name token that is spelled directly in the main
  // file. For example, in `#define CHECK checkCudaErrors`, the nested helper
  // expansion points back to the source token `CHECK`; replacing that offset
  // with the longer helper name would corrupt the invocation. Macro-ID and
  // source-spelling checks make aliases and forwarding fail closed.
  const clang::SourceLocation macroNameLocation =
      MacroNameTok.getLocation();
  bool invalidSourceToken = false;
  const llvm::StringRef sourceToken = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(macroNameLocation,
                                             macroNameLocation),
      sourceManager, getCompilerInstance().getLangOpts(),
      &invalidSourceToken);
  if (macroNameLocation.isMacroID() || invalidSourceToken ||
      sourceToken != name) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: macro '" << name
        << "' is not a direct source-token invocation; macro and include "
        << "kept\n";
    return;
  }
  if (!macroInfo->isFunctionLike() || macroInfo->getNumParams() != 1) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    nvidiaSampleHelperRewriteFailed = true;
    return;
  }
  if (!activeNvidiaSampleHelperMacroBodyMatches(
          name, *macroInfo, getCompilerInstance().getPreprocessor())) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: active macro body for '"
        << name << "' is not admitted; macro and include kept\n";
    return;
  }

  const clang::SourceLocation rangeBegin =
      sourceManager.getExpansionLoc(Range.getBegin());
  clang::SourceLocation rangeEnd =
      sourceManager.getExpansionLoc(Range.getEnd());
  rangeEnd = clang::Lexer::getLocForEndOfToken(
      rangeEnd, 0, sourceManager,
      getCompilerInstance().getLangOpts());
  if (rangeBegin.isValid() && rangeEnd.isValid()) {
    const std::pair<clang::FileID, unsigned> begin =
        sourceManager.getDecomposedLoc(
            sourceManager.getFileLoc(rangeBegin));
    const std::pair<clang::FileID, unsigned> end =
        sourceManager.getDecomposedLoc(
            sourceManager.getFileLoc(rangeEnd));
    if (begin.first == end.first && end.second > begin.second) {
      const SemanticRewriteRange invocation =
          {begin.first, begin.second, end.second};
      nvidiaSampleHelperMacroRanges.push_back(invocation);
      nvidiaSampleHelperMacroCandidates.push_back(
          {name == "checkCudaErrors"
               ? NvidiaSampleHelperMacroKind::CheckCudaErrors
               : NvidiaSampleHelperMacroKind::GetLastCudaError,
           macroLocation, invocation,
           name == "getLastCudaError", false});
      nvidiaSampleHelperNormalMacroOffsets.insert(
          sourceManager.getFileOffset(
              sourceManager.getFileLoc(macroLocation)));
    } else {
      nvidiaSampleHelperRewriteFailed = true;
    }
  } else {
    nvidiaSampleHelperRewriteFailed = true;
  }

}

void AscifyAction::rewriteProvenNvidiaSampleHelperMacros() {
  // Replacements are accumulated without rollback support. Restrict closure
  // to one direct recognized include so include removal is atomic: duplicate
  // includes keep both directives and every original helper macro call.
  if (nvidiaSampleHelperIncludes.size() != 1) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: expected exactly one "
        << "recognized helper_cuda.h include; all helper edits kept\n";
    return;
  }
  // Replacements are not rollback-capable. Complete every provenance,
  // preprocessor, status-domain, declaration, and range proof before the
  // first helper edit. One unsafe helper use therefore keeps every helper
  // macro invocation and the include in their original form.
  if (nvidiaSampleHelperUnsupportedMacroUse ||
      nvidiaSampleHelperUnsupportedDeclarationUse ||
      nvidiaSampleHelperRewriteFailed ||
      nvidiaSampleHelperOutputMacroEverDefined) {
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: global proof failed before "
        << "replacement; all helper edits kept\n";
    return;
  }
  for (const NvidiaSampleHelperMacroCandidate &candidate :
       nvidiaSampleHelperMacroCandidates) {
    if (!candidate.statusDomainProven) {
      nvidiaSampleHelperRewriteFailed = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: candidate proof was "
          << "incomplete; all helper edits kept\n";
      return;
    }
  }
  // The raw-token audit runs later in ExecuteAction. Do not touch the global
  // replacement set here. EndSourceFileAction commits macro and include edits
  // together only after that audit and every other rewrite have completed.
  nvidiaSampleHelperReadyToCommit = true;
}

void AscifyAction::auditRawNvidiaSampleHelperToken(
    const clang::Token &token) {
  if (nvidiaSampleHelperIncludes.empty() || !token.isAnyIdentifier())
    return;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  bool invalidSpelling = false;
  const std::string name = clang::Lexer::getSpelling(
      token, sourceManager, getCompilerInstance().getLangOpts(),
      &invalidSpelling);
  if (invalidSpelling) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: unreadable raw identifier "
        << "keeps helper_cuda.h\n";
    return;
  }
  const bool publishedCompatMacro =
      !hasCudaCompatHeaderBeforeNvidiaHelper &&
      isAscifyCudaCompatPublishedMacro(name);
  if (!isNvidiaSampleHelperObservableMacro(name) &&
      name != "findCudaDevice" && !publishedCompatMacro)
    return;
  const clang::SourceLocation fileLoc =
      sourceManager.getFileLoc(token.getLocation());
  if (fileLoc.isInvalid() || !sourceManager.isWrittenInMainFile(fileLoc))
    return;
  const unsigned offset = sourceManager.getFileOffset(fileLoc);
  if (nvidiaSampleHelperNormalMacroOffsets.count(offset) != 0 ||
      nvidiaSampleFindDeviceNormalOffsets.count(offset) != 0)
    return;
  if (!nvidiaSampleHelperUnsupportedMacroUse) {
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: residual raw PP use of '"
        << name << "' keeps helper_cuda.h\n";
  }
  nvidiaSampleHelperUnsupportedMacroUse = true;
}

void AscifyAction::auditRawPublishedCompatTokensInFile(
    clang::FileID file) {
  if (file.isInvalid() || hasCudaCompatHeaderBeforeNvidiaHelper)
    return;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  bool invalidBuffer = false;
  const llvm::StringRef buffer =
      sourceManager.getBufferData(file, &invalidBuffer);
  if (invalidBuffer) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify CUDA compat closure: retained header could not be "
        << "audited for published macros; helper edits kept\n";
    return;
  }
  const clang::SourceLocation start =
      sourceManager.getLocForStartOfFile(file);
  clang::Lexer lexer(
      start, getCompilerInstance().getLangOpts(), buffer.begin(),
      buffer.begin(), buffer.end());
  clang::Token token;
  lexer.LexFromRawLexer(token);
  while (token.isNot(clang::tok::eof)) {
    if (token.isAnyIdentifier()) {
      bool invalidSpelling = false;
      const std::string name = clang::Lexer::getSpelling(
          token, sourceManager, getCompilerInstance().getLangOpts(),
          &invalidSpelling);
      if (invalidSpelling) {
        nvidiaSampleHelperUnsupportedMacroUse = true;
        llvm::errs()
            << "Ascify CUDA compat closure: retained header contains an "
            << "unreadable raw identifier; helper edits kept\n";
        return;
      }
      if (isAscifyCudaCompatPublishedMacro(name)) {
        nvidiaSampleHelperUnsupportedMacroUse = true;
        llvm::errs()
            << "Ascify CUDA compat closure: retained header observes "
            << "published macro token '" << name
            << "'; helper edits kept\n";
        return;
      }
    }
    lexer.LexFromRawLexer(token);
  }
}

void AscifyAction::auditExternalNvidiaSampleHelperPreprocessorUse(
    clang::SourceLocation location,
    const clang::Token &macroNameToken,
    llvm::StringRef directive) {
  if (macroNameToken.getIdentifierInfo() == nullptr)
    return;
  const llvm::StringRef name =
      macroNameToken.getIdentifierInfo()->getName();
  const bool observableHelper =
      isNvidiaSampleHelperObservableMacro(name);
  const bool downstreamCompatPublication =
      !nvidiaSampleHelperIncludes.empty() &&
      !hasCudaCompatHeaderBeforeNvidiaHelper &&
      isAscifyCudaCompatPublishedMacro(name);
  if (!observableHelper && !downstreamCompatPublication)
    return;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::SourceLocation expansion =
      sourceManager.getExpansionLoc(location);
  if (expansion.isInvalid()) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: invalid observable "
        << directive << " location keeps all helper edits\n";
    return;
  }
  const unsigned frozenRole =
      frozenNvidiaSampleHelperFileRole(sourceManager, expansion);
  const unsigned frozenMacroGraph =
      FrozenOfficialNvidiaSampleHelperCuda |
      FrozenOfficialNvidiaSampleHelperString |
      FrozenOfficialNvidiaSampleHelperFunctions |
      FrozenOfficialNvidiaSampleHelperImage |
      FrozenOfficialNvidiaSampleHelperTimer;
  if (locationComesFromRecognizedNvidiaSampleHelper(
          sourceManager, expansion) ||
      (frozenRole & frozenMacroGraph) != 0 ||
      locationComesFromAscifyCudaCompat(sourceManager, expansion))
    return;
  nvidiaSampleHelperUnsupportedMacroUse = true;
  llvm::errs()
      << "Ascify NVIDIA sample-helper closure: observable " << directive
      << " use of '" << name << "' keeps all helper edits\n";
}

void AscifyAction::auditSkippedNvidiaSampleHelperElifDirective(
    clang::SourceLocation location,
    clang::SourceRange conditionRange,
    llvm::StringRef directive) {
  (void)conditionRange;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::SourceLocation expansion =
      sourceManager.getExpansionLoc(location);
  const unsigned frozenRole = expansion.isValid()
      ? frozenNvidiaSampleHelperFileRole(sourceManager, expansion)
      : FrozenNvidiaSampleHelperNone;
  const unsigned frozenMacroGraph =
      FrozenOfficialNvidiaSampleHelperCuda |
      FrozenOfficialNvidiaSampleHelperString |
      FrozenOfficialNvidiaSampleHelperFunctions |
      FrozenOfficialNvidiaSampleHelperImage |
      FrozenOfficialNvidiaSampleHelperTimer;
  if (expansion.isValid() &&
      (locationComesFromRecognizedNvidiaSampleHelper(
           sourceManager, expansion) ||
       (frozenRole & frozenMacroGraph) != 0))
    return;
  // Clang reports only a raw ConditionRange for a skipped #elifdef/ifndef.
  // That spelling is not phase-2 cleaned, so line splices can hide an
  // observable helper name.  These directives are rare in CUDA Samples; keep
  // the transaction fail-closed instead of maintaining a second lexer.
  nvidiaSampleHelperUnsupportedMacroUse = true;
  llvm::errs()
      << "Ascify NVIDIA sample-helper closure: unproven skipped "
      << directive << " keeps all helper edits\n";
}

bool AscifyAction::isDirectFrozenFunctionsImageProviderInstance(
    clang::SourceLocation imageLocation) {
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::SourceLocation imageSpelling =
      sourceManager.getSpellingLoc(imageLocation);
  if (imageSpelling.isInvalid() ||
      frozenNvidiaSampleHelperFileRole(sourceManager, imageSpelling) !=
          FrozenOfficialNvidiaSampleHelperImage)
    return false;
  const clang::FileID imageFile = sourceManager.getFileID(imageSpelling);
  const clang::FileEntry *imageEntry = imageFile.isValid()
      ? sourceManager.getFileEntryForID(imageFile)
      : nullptr;
  if (imageEntry == nullptr || sourceManager.isFileOverridden(imageEntry))
    return false;
  const llvm::sys::fs::UniqueID &imageUniqueId = imageEntry->getUniqueID();
  const std::pair<std::uint64_t, std::uint64_t> imageIdentity(
      imageUniqueId.getDevice(), imageUniqueId.getFile());
  if ((imageIdentity.first == 0 && imageIdentity.second == 0) ||
      frozenNvidiaSampleMacroProviderIdentities.count(imageIdentity) == 0)
    return false;
  const clang::SourceLocation includeLocation =
      sourceManager.getIncludeLoc(imageFile);
  const clang::SourceLocation includerSpelling =
      sourceManager.getSpellingLoc(includeLocation);
  if (includerSpelling.isInvalid() ||
      frozenNvidiaSampleHelperFileRole(sourceManager, includerSpelling) !=
          FrozenOfficialNvidiaSampleHelperFunctions)
    return false;
  const clang::FileID includerFile =
      sourceManager.getFileID(includerSpelling);
  return includerFile.isValid() &&
      frozenNvidiaSampleDirectFunctionsRootFileIds.count(
          includerFile.getHashValue()) != 0;
}

void AscifyAction::auditFrozenNvidiaSampleHelperMacroDependency(
    clang::SourceLocation useLocation,
    const clang::Token &macroNameToken,
    const clang::MacroDefinition &definition,
    llvm::StringRef useKind) {
  if (macroNameToken.getIdentifierInfo() == nullptr)
    return;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::SourceLocation expansion =
      sourceManager.getExpansionLoc(useLocation);
  const unsigned frozenProfile =
      FrozenOfficialNvidiaSampleHelperCuda |
      FrozenOfficialNvidiaSampleHelperString;
  const unsigned frozenConsumerRole =
      expansion.isValid()
          ? frozenNvidiaSampleHelperFileRole(sourceManager, expansion)
          : FrozenNvidiaSampleHelperNone;
  if (expansion.isInvalid() ||
      (frozenConsumerRole & frozenProfile) == 0)
    return;

  const clang::MacroInfo *macroInfo = definition.getMacroInfo();
  // An undefined #if/#ifdef operand supplies no replacement tokens and does
  // not import project semantics into the frozen helper profile.
  if (macroInfo == nullptr)
    return;
  const clang::SourceLocation definitionLocation =
      sourceManager.getSpellingLoc(macroInfo->getDefinitionLoc());
  const clang::FileID definitionFile = definitionLocation.isValid()
      ? sourceManager.getFileID(definitionLocation)
      : clang::FileID();
#if LLVM_VERSION_MAJOR >= 8
  const clang::FileID predefinesFile =
      getCompilerInstance().getPreprocessor().getPredefinesFileID();
  const bool compilerPredefinition =
      definitionFile.isValid() && predefinesFile.isValid() &&
      definitionFile == predefinesFile &&
      !sourceManager.isWrittenInCommandLineFile(definitionLocation);
#else
  // isWrittenInCommandLineFile was added in LLVM 8. The frozen SHA profile is
  // already unavailable before LLVM 13, so conservatively admit no
  // non-builtin predefinition on older builds instead of weakening the
  // command-line distinction.
  (void)definitionFile;
  const bool compilerPredefinition = false;
#endif
  // CUDA Samples' exact b7c helper_functions.h closure enters helper_image.h
  // before helper_cuda.h. The exact image helper provides the two active
  // policy macros observed by the frozen consumers: EXIT_WAIVED=2 is tested in
  // helper_string/helper_cuda, while MAX(a,b)=((a>b)?a:b) is tested only in
  // helper_cuda. Admit exactly those contracts and no alternative provider.
  // Both the exact root and exact image provider must have been observed at
  // their first physical-file entry. A copied name, -D, remapped buffer,
  // later system_header promotion, #line spelling, or redefinition therefore
  // cannot borrow this exception.
  bool frozenOfficialProviderMacroDependency = false;
  const llvm::StringRef macroName =
      macroNameToken.getIdentifierInfo()->getName();
  const bool admittedConsumer =
      (macroName == "EXIT_WAIVED" &&
       (frozenConsumerRole & frozenProfile) != 0) ||
      (macroName == "MAX" &&
       frozenConsumerRole == FrozenOfficialNvidiaSampleHelperCuda);
  if (admittedConsumer && useKind == "#ifndef" &&
      frozenOfficialNvidiaSampleHelperFunctionsSeen &&
      definitionLocation.isValid() &&
      activeFrozenHelperFunctionsProviderMacroBodyMatches(
          macroName, *macroInfo,
          getCompilerInstance().getPreprocessor())) {
    frozenOfficialProviderMacroDependency =
        isDirectFrozenFunctionsImageProviderInstance(definitionLocation);
  }
  const bool frozenCoreDefinition =
      definitionLocation.isValid() &&
      locationComesFromFrozenNvidiaSampleHelperFile(
          sourceManager, definitionLocation, frozenProfile);
  // These names are CUDA Samples policy values, not compiler/platform
  // facilities. Do not let an arbitrary -isystem header, builtin spelling,
  // or compiler predefinition provide them. The frozen core or the exact
  // helper_functions closure above are the only admitted origins.
  const bool genericTrustedDefinition =
      !isFrozenHelperFunctionsProviderPolicyMacro(macroName) &&
      (macroInfo->isBuiltinMacro() || compilerPredefinition ||
       (definitionLocation.isValid() &&
        locationComesFromInitiallyTrustedSystemFile(
            sourceManager, definitionLocation, trustedSystemFileIds)));
  const bool trustedDefinition =
      frozenOfficialProviderMacroDependency || frozenCoreDefinition ||
      genericTrustedDefinition;
  if (trustedDefinition)
    return;

  nvidiaSampleHelperUnsupportedMacroUse = true;
  if (nvidiaSampleFrozenProfileMacroConflict)
    return;
  nvidiaSampleFrozenProfileMacroConflict = true;
  llvm::errs()
      << "Ascify NVIDIA sample-helper closure: untrusted " << useKind
      << " macro dependency '"
      << macroNameToken.getIdentifierInfo()->getName()
      << "' changed a frozen helper profile; all helper edits kept\n";
}

void AscifyAction::finalizeNvidiaSampleHelperClosure() {
  if (nvidiaSampleHelperIncludes.empty())
    return;
  const bool safeToRemove =
      nvidiaSampleHelperIncludes.size() == 1 &&
      nvidiaSampleHelperReadyToCommit &&
      nvidiaSampleHelperRawAuditCompleted &&
      !nvidiaSampleHelperUnsupportedMacroUse &&
      !nvidiaSampleHelperUnsupportedDeclarationUse &&
      !nvidiaSampleHelperRewriteFailed &&
      !nvidiaSampleHelperOutputMacroEverDefined;
  if (!safeToRemove) {
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: include kept"
        << ", check_rewrites=" << nvidiaSampleCheckCudaErrorsRewrites
        << ", get_last_rewrites="
        << nvidiaSampleGetLastCudaErrorRewrites
        << ", find_device_rewrites="
        << nvidiaSampleFindCudaDeviceRewrites
        << ", unsupported_macro="
        << (nvidiaSampleHelperUnsupportedMacroUse ? 1 : 0)
        << ", unsupported_declaration="
        << (nvidiaSampleHelperUnsupportedDeclarationUse ? 1 : 0)
        << ", rewrite_failed="
        << (nvidiaSampleHelperRewriteFailed ? 1 : 0)
        << ", ready="
        << (nvidiaSampleHelperReadyToCommit ? 1 : 0)
        << ", raw_audit_completed="
        << (nvidiaSampleHelperRawAuditCompleted ? 1 : 0) << "\n";
    return;
  }

  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  ct::Replacements staged(*replacements);
  std::vector<SemanticRewriteRange> stagedSemanticRanges;
  std::vector<ct::Replacement> committedMacroReplacements;
  std::vector<clang::SourceLocation> committedMacroLocations;
  std::string stagingError;
  auto overlaps = [](const SemanticRewriteRange &lhs,
                     const SemanticRewriteRange &rhs) {
    return lhs.file == rhs.file && lhs.beginOffset < rhs.endOffset &&
           rhs.beginOffset < lhs.endOffset;
  };
  for (const NvidiaSampleHelperMacroCandidate &candidate :
       nvidiaSampleHelperMacroCandidates) {
    for (const SemanticRewriteRange &existing : SemanticRewriteRanges) {
      if (overlaps(candidate.invocationRange, existing)) {
        stagingError = "overlap with an existing semantic rewrite";
        break;
      }
    }
    for (const SemanticRewriteRange &pending : stagedSemanticRanges) {
      if (overlaps(candidate.invocationRange, pending)) {
        stagingError = "overlap between helper macro rewrites";
        break;
      }
    }
    if (!stagingError.empty())
      break;
    const llvm::StringRef originalName =
        candidate.kind == NvidiaSampleHelperMacroKind::CheckCudaErrors
            ? "checkCudaErrors"
            : "getLastCudaError";
    const llvm::StringRef replacementName =
        candidate.kind == NvidiaSampleHelperMacroKind::CheckCudaErrors
            ? "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS"
            : "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR";
    ct::Replacement replacement(sourceManager, candidate.nameLocation,
                                originalName.size(),
                                replacementName.str());
    if (!llcompat::insertReplacement(staged, replacement, &stagingError))
      break;
    committedMacroReplacements.push_back(replacement);
    committedMacroLocations.push_back(candidate.nameLocation);
    stagedSemanticRanges.push_back(candidate.invocationRange);
  }
  for (const NvidiaSampleFindDeviceCandidate &candidate :
       nvidiaSampleFindDeviceCandidates) {
    for (const SemanticRewriteRange &existing : SemanticRewriteRanges) {
      if (overlaps(candidate.rewriteRange, existing)) {
        stagingError =
            "findCudaDevice overlap with an existing semantic rewrite";
        break;
      }
    }
    for (const SemanticRewriteRange &pending : stagedSemanticRanges) {
      if (overlaps(candidate.rewriteRange, pending)) {
        stagingError = "findCudaDevice overlap within helper transaction";
        break;
      }
    }
    if (!stagingError.empty())
      break;
    ct::Replacement replacement(
        sourceManager, candidate.nameLocation, 14U,
        "::ascify::sampleFindCudaDevice");
    if (!llcompat::insertReplacement(staged, replacement, &stagingError))
      break;
    committedMacroReplacements.push_back(replacement);
    committedMacroLocations.push_back(candidate.nameLocation);
    stagedSemanticRanges.push_back(candidate.rewriteRange);
  }

  bool compatInsertedByClosure =
      hasCudaCompatHeaderBeforeNvidiaHelper;
  const NvidiaSampleHelperInclude &include =
      nvidiaSampleHelperIncludes.front();
  const char *begin = sourceManager.getCharacterData(include.hashLocation);
  const char *end = sourceManager.getCharacterData(include.filenameEnd);
  if (stagingError.empty() &&
      (begin == nullptr || end == nullptr || end < begin)) {
    stagingError = "invalid direct include range";
  }
  // helper_cuda's direct portable include surface is part of the source
  // contract even when no downstream AST node points back to it.  Keep those
  // declarations and helper_string macros at the original include position;
  // replace only the CUDA-specific helper layer.
  std::string includeReplacementText =
      "#include <stdint.h>\n"
      "#include <stdio.h>\n"
      "#include <stdlib.h>\n"
      "#include <string.h>\n"
      "#include <helper_string.h>";
  const bool helperNeedsCompat =
      !nvidiaSampleHelperMacroCandidates.empty() ||
      !nvidiaSampleFindDeviceCandidates.empty();
  if (stagingError.empty() &&
      (needsCudaCompatHeader || helperNeedsCompat) &&
      !compatInsertedByClosure) {
    includeReplacementText +=
        "\n#include <ascify/ascify_cuda_compat.hpp>";
    compatInsertedByClosure = true;
  }
  std::unique_ptr<ct::Replacement> includeReplacement;
  if (stagingError.empty()) {
    includeReplacement.reset(new ct::Replacement(
        sourceManager, include.hashLocation,
        static_cast<unsigned>(end - begin), includeReplacementText));
    if (!llcompat::insertReplacement(
            staged, *includeReplacement, &stagingError)) {
      // Keep stagingError for the fail-closed diagnostic below.
    }
  }
  if (!stagingError.empty()) {
    nvidiaSampleHelperRewriteFailed = true;
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: staged transaction failed; "
        << "all helper edits kept: " << stagingError << "\n";
    return;
  }

  // No shared state was changed before this point. Commit the complete set in
  // one assignment, then publish ranges, counters, and header state.
  *replacements = std::move(staged);
  SemanticRewriteRanges.insert(SemanticRewriteRanges.end(),
                               stagedSemanticRanges.begin(),
                               stagedSemanticRanges.end());
  for (NvidiaSampleHelperMacroCandidate &candidate :
       nvidiaSampleHelperMacroCandidates) {
    candidate.rewritten = true;
    if (candidate.kind == NvidiaSampleHelperMacroKind::CheckCudaErrors)
      ++nvidiaSampleCheckCudaErrorsRewrites;
    else
      ++nvidiaSampleGetLastCudaErrorRewrites;
  }
  for (NvidiaSampleFindDeviceCandidate &candidate :
       nvidiaSampleFindDeviceCandidates) {
    candidate.rewritten = true;
    ++nvidiaSampleFindCudaDeviceRewrites;
  }
  needsCudaCompatHeader = needsCudaCompatHeader || helperNeedsCompat;
  if (compatInsertedByClosure)
    hasCudaCompatHeader = true;
  if (compatInsertedByClosure)
    hasCudaCompatHeaderBeforeNvidiaHelper = true;
  if (PrintStats || PrintStatsCSV) {
    for (size_t index = 0; index < committedMacroReplacements.size();
         ++index) {
      Statistics::current().lineTouched(
          clang::FullSourceLoc(committedMacroLocations[index], sourceManager)
              .getExpansionLineNumber());
      Statistics::current().bytesChanged(
          committedMacroReplacements[index].getLength());
    }
    Statistics::current().lineTouched(
        clang::FullSourceLoc(include.hashLocation, sourceManager)
            .getExpansionLineNumber());
    Statistics::current().bytesChanged(includeReplacement->getLength());
  }
  llvm::errs()
      << "Ascify NVIDIA sample-helper closure: include removed; portable "
      << "surface retained"
      << ", includes=1"
      << ", check_rewrites=" << nvidiaSampleCheckCudaErrorsRewrites
      << ", get_last_rewrites="
      << nvidiaSampleGetLastCudaErrorRewrites
      << ", find_device_rewrites="
      << nvidiaSampleFindCudaDeviceRewrites << "\n";
}

void AscifyAction::EndSourceFileAction() {
  finalizePendingNvidiaSampleHelperPragma();
  if (needsCudaCompatHeader &&
      nvidiaSampleHelperOutputMacroEverDefined) {
    const auto diagnostic =
        getCompilerInstance().getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "Ascify cannot publish CUDA compatibility output because a "
            "reserved compat macro was defined by the input");
    getCompilerInstance().getDiagnostics().Report(diagnostic);
    return;
  }
  if (needsCudaCompatHeader && ascifyCudaCompatIncludeConflict) {
    const auto diagnostic =
        getCompilerInstance().getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "Ascify cannot publish CUDA compatibility output because the "
            "compat include spelling resolves to an unrecognized header");
    getCompilerInstance().getDiagnostics().Report(diagnostic);
    return;
  }
  if (needsCudaCompatHeader && ascifyCudaCompatActiveMacroConflict) {
    const auto diagnostic =
        getCompilerInstance().getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "Ascify cannot publish CUDA compatibility output because active "
            "input macro '%0' collides with the frozen compat header");
    getCompilerInstance().getDiagnostics().Report(diagnostic)
        << ascifyCudaCompatActiveMacroConflictName;
    return;
  }
  if (needsCudaCompatHeader &&
      !hasCudaCompatHeaderBeforeNvidiaHelper &&
      ascifyCudaCompatDeclarationConflict) {
    const auto diagnostic =
        getCompilerInstance().getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "Ascify cannot publish CUDA compatibility output because the "
            "input owns the top-level name 'ascify'");
    getCompilerInstance().getDiagnostics().Report(diagnostic);
    return;
  }
  finalizeNvidiaSampleHelperClosure();
  std::string includes;
  if (needsCudaCompatHeader && !hasCudaCompatHeader)
    includes += "#include <ascify/ascify_cuda_compat.hpp>\n";
  const bool useSimdTargetRecipe =
      TargetRecipe == ascify::DavC310TargetRecipe::SimdRecipeName;
  const bool hasSelectedTargetHeader =
      useSimdTargetRecipe ? hasDavC310SimdTargetHeader
                          : hasDavC310TargetHeader;
  if (needsDavC310TargetHeader && !hasSelectedTargetHeader) {
    includes += "#include <";
    includes += useSimdTargetRecipe
                    ? ascify::DavC310TargetRecipe::SimdTargetHeader
                    : ascify::DavC310TargetRecipe::TargetHeader;
    includes += ">\n";
  }
  if (includes.empty())
    return;

  if (useSimdTargetRecipe) {
    std::string shielded;
    std::vector<const char *> headerShieldNames;
    for (const char *name : RowwiseSimdReservedMacros) {
      if (isRowwiseSimdPublishedHeaderMacro(name))
        continue;
      headerShieldNames.push_back(name);
      shielded += "#pragma push_macro(\"";
      shielded += name;
      shielded += "\")\n#undef ";
      shielded += name;
      shielded += "\n";
    }
    shielded += includes;
    for (auto name = headerShieldNames.rbegin();
         name != headerShieldNames.rend(); ++name) {
      shielded += "#pragma pop_macro(\"";
      shielded += *name;
      shielded += "\")\n";
    }
    includes = std::move(shielded);
    const std::string mainOuterGuardName =
        findConventionalMainOuterGuard(getCompilerInstance());
    if (!mainOuterGuardName.empty()) {
      includes =
          "#pragma push_macro(\"" + mainOuterGuardName + "\")\n#undef " +
          mainOuterGuardName + "\n" + includes +
          "#pragma pop_macro(\"" + mainOuterGuardName + "\")\n";
    }
  }

  auto &SM = getCompilerInstance().getSourceManager();
  const clang::FileID mainFile = SM.getMainFileID();
  const clang::SourceLocation insertLoc =
      useSimdTargetRecipe
          ? SM.getLocForStartOfFile(mainFile)
          : (firstHeader ? firstHeaderLoc
                         : SM.getLocForStartOfFile(mainFile));
  ct::Replacement Rep(SM, insertLoc, 0, includes);
  insertReplacement(Rep, clang::FullSourceLoc(insertLoc, SM));
}

namespace {

/**
  * A silly little class to proxy PPCallbacks back to the AscifyAction class.
  */
class PPCallbackProxy : public clang::PPCallbacks {
  AscifyAction &ascifyAction;

public:
  explicit PPCallbackProxy(AscifyAction &action): ascifyAction(action) {}

  void FileChanged(
      clang::SourceLocation location,
      clang::PPCallbacks::FileChangeReason reason,
      clang::SrcMgr::CharacteristicKind fileType,
      clang::FileID) override {
    ascifyAction.FileChanged(location, reason, fileType);
  }

  void InclusionDirective(clang::SourceLocation hash_loc, const clang::Token &include_token,
                          StringRef file_name, bool is_angled, clang::CharSourceRange filename_range,
#if LLVM_VERSION_MAJOR < 15
                          const clang::FileEntry *file,
#elif LLVM_VERSION_MAJOR == 15
                          Optional<clang::FileEntryRef> file,
#else
                          clang::OptionalFileEntryRef file,
#endif
                          StringRef search_path, StringRef relative_path,
#if LLVM_VERSION_MAJOR < 19
                          const clang::Module *SuggestedModule
#else
                          const clang::Module *SuggestedModule,
                          bool ModuleImported
#endif
#if LLVM_VERSION_MAJOR > 6
                        , clang::SrcMgr::CharacteristicKind FileType
#endif
                         ) override {
#if LLVM_VERSION_MAJOR < 15
    const StringRef resolvedFileName =
        file == nullptr ? StringRef() : file->getName();
#else
    // LLVM 15+ carries the selected path on FileEntryRef.  In LLVM 23 the
    // underlying FileEntry no longer exposes getName(), so keep the resolved
    // FileEntryRef as the source of truth instead of guessing include search.
    const StringRef resolvedFileName =
        file ? file->getName() : StringRef();
#endif
    ascifyAction.InclusionDirective(
        hash_loc, include_token, file_name, is_angled, filename_range,
        resolvedFileName, search_path, relative_path, SuggestedModule);
  }

  void PragmaDirective(clang::SourceLocation Loc, clang::PragmaIntroducerKind Introducer) override {
    ascifyAction.PragmaDirective(Loc, Introducer);
  }

#if LLVM_VERSION_MAJOR >= 13
  void PragmaDiagnosticPush(
      clang::SourceLocation, StringRef) override {
    ascifyAction.PragmaDiagnostic();
  }

  void PragmaDiagnosticPop(
      clang::SourceLocation, StringRef) override {
    ascifyAction.PragmaDiagnostic();
  }

  void PragmaDiagnostic(
      clang::SourceLocation, StringRef, clang::diag::Severity,
      StringRef) override {
    ascifyAction.PragmaDiagnostic();
  }
#endif

  void Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD) override {
    ascifyAction.Ifndef(Loc, MacroNameTok, MD);
  }

  void If(clang::SourceLocation,
          clang::SourceRange,
          clang::PPCallbacks::ConditionValueKind) override {
    ascifyAction.ConditionalDirectiveEntered();
  }

  void Ifdef(clang::SourceLocation Loc,
             const clang::Token &MacroNameTok,
             const clang::MacroDefinition &MD) override {
    ascifyAction.Ifdef(Loc, MacroNameTok, MD);
  }

#if LLVM_VERSION_MAJOR >= 13
  void Elifdef(clang::SourceLocation Loc,
               const clang::Token &MacroNameTok,
               const clang::MacroDefinition &MD) override {
    ascifyAction.Elifdef(Loc, MacroNameTok, MD, "#elifdef");
  }

  void Elifdef(clang::SourceLocation Loc,
               clang::SourceRange ConditionRange,
               clang::SourceLocation) override {
    ascifyAction.ElifdefSkipped(Loc, ConditionRange, "#elifdef");
  }

  void Elifndef(clang::SourceLocation Loc,
                const clang::Token &MacroNameTok,
                const clang::MacroDefinition &MD) override {
    ascifyAction.Elifdef(Loc, MacroNameTok, MD, "#elifndef");
  }

  void Elifndef(clang::SourceLocation Loc,
                clang::SourceRange ConditionRange,
                clang::SourceLocation) override {
    ascifyAction.ElifdefSkipped(Loc, ConditionRange, "#elifndef");
  }
#endif

  void Endif(clang::SourceLocation, clang::SourceLocation) override {
    ascifyAction.ConditionalDirectiveEnded();
  }

  void Defined(const clang::Token &MacroNameTok,
               const clang::MacroDefinition &MD,
               clang::SourceRange Range) override {
    ascifyAction.Defined(MacroNameTok, MD, Range);
  }

  void MacroUndefined(
      const clang::Token &MacroNameTok,
      const clang::MacroDefinition &MD,
      const clang::MacroDirective *) override {
    ascifyAction.MacroUndefined(MacroNameTok, MD);
  }

  void MacroDefined(
      const clang::Token &MacroNameTok,
      const clang::MacroDirective *) override {
    ascifyAction.MacroDefined(MacroNameTok);
  }

  void MacroExpands(const clang::Token &MacroNameTok,
                    const clang::MacroDefinition &MD,
                    clang::SourceRange Range,
                    const clang::MacroArgs *) override {
    ascifyAction.MacroExpands(MacroNameTok, MD, Range);
  }

  void SourceRangeSkipped(clang::SourceRange Range, clang::SourceLocation EndifLoc) override {
    ascifyAction.AddSkippedSourceRange(Range);
  }
};
}

bool AscifyAction::BeginInvocation(clang::CompilerInstance &CI) {
  clang::DiagnosticsEngine &DE = CI.getDiagnostics();
  if (TargetPolicy != "portable" && TargetPolicy != "dav-c310-vec") {
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "unsupported --target-policy value '%0'; expected 'portable' or "
        "'dav-c310-vec'");
    DE.Report(ID) << TargetPolicy;
    return false;
  }
  if (SimtMathMode != "precise" && SimtMathMode != "fast") {
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "unsupported --simt-math value '%0'; expected 'precise' or 'fast'");
    DE.Report(ID) << SimtMathMode;
    return false;
  }
  if (TargetRecipe != "none" &&
      TargetRecipe != ascify::DavC310TargetRecipe::SimdRecipeName) {
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "unsupported --target-recipe value '%0'; expected 'none' or "
        "'dav-3510-rowwise-simd-v1'");
    DE.Report(ID) << TargetRecipe;
    return false;
  }
  if (TargetRecipe == ascify::DavC310TargetRecipe::SimdRecipeName &&
      (TargetPolicy != "dav-c310-vec" || SimtMathMode != "fast")) {
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "--target-recipe=dav-3510-rowwise-simd-v1 requires "
        "--target-policy=dav-c310-vec and --simt-math=fast");
    DE.Report(ID);
    return false;
  }
  llcompat::RetainExcludedConditionalBlocks(CI);
  return true;
}

void AscifyAction::ExecuteAction() {
  clang::Preprocessor &PP = getCompilerInstance().getPreprocessor();
  // Register yourself as the preprocessor callback, by proxy.
  PP.addPPCallbacks(std::unique_ptr<PPCallbackProxy>(new PPCallbackProxy(*this)));
#if LLVM_VERSION_MAJOR > 3
  Statistics::cudaVersionUsedByClang = Statistics::convertCudaToolkitVersion(clang::ToCudaVersion(PP.getTargetInfo().getSDKVersion()));
  llvm::errs() << " !!!!!!! CUDA SDK version detected: " << int(clang::ToCudaVersion(PP.getTargetInfo().getSDKVersion())) << "\n";
#endif
  // Now we're done futzing with the lexer, have the subclass proceeed with Sema and AST matching.
  clang::ASTFrontendAction::ExecuteAction();
  auto &SM = getCompilerInstance().getSourceManager();
  ascifyCudaCompatDeclarationConflict =
      hasInputTopLevelAscifyDeclaration(
          getCompilerInstance().getASTContext(), SM);
  if (!nvidiaSampleHelperIncludes.empty()) {
    nvidiaSampleHelperUnsupportedDeclarationUse =
        hasUnsupportedNvidiaSampleHelperDeclarationUse();
    rewriteProvenNvidiaSampleHelperMacros();
  }
  if (TargetPolicy == "dav-c310-vec" && SimtMathMode == "fast") {
    if (TargetRecipe == ascify::DavC310TargetRecipe::SimdRecipeName) {
      const RawRowwiseSimdConflict rawMainConflict =
          rawFileRowwiseSimdConflict(
              getCompilerInstance(), SM.getMainFileID(), false);
      std::string conflictingMacro = rawMainConflict.macro;
      if (rowwiseSimdRawConflictingDeclaration.empty())
        rowwiseSimdRawConflictingDeclaration =
            rawMainConflict.declaration;
      for (const char *name : RowwiseSimdReservedMacros) {
        if (!conflictingMacro.empty())
          break;
        const clang::IdentifierInfo *identifier =
            PP.getIdentifierInfo(name);
        if (rowwiseSimdMacrosEverDefined.count(name) != 0 ||
            (identifier != nullptr && PP.isMacroDefined(identifier))) {
          conflictingMacro = name;
          break;
        }
      }
      if (!conflictingMacro.empty()) {
        const auto diagnostic =
            getCompilerInstance().getDiagnostics().getCustomDiagID(
                clang::DiagnosticsEngine::Error,
                "--target-recipe=dav-3510-rowwise-simd-v1 rejects reserved "
                "macro '%0'");
        getCompilerInstance().getDiagnostics().Report(diagnostic)
            << conflictingMacro;
        return;
      }
      if (!rowwiseSimdRawConflictingDeclaration.empty()) {
        const auto diagnostic =
            getCompilerInstance().getDiagnostics().getCustomDiagID(
                clang::DiagnosticsEngine::Error,
                "--target-recipe=dav-3510-rowwise-simd-v1 rejects raw "
                "input token '%0' reserved by the row-wise SIMD recipe");
        getCompilerInstance().getDiagnostics().Report(diagnostic)
            << rowwiseSimdRawConflictingDeclaration;
        return;
      }
      const RowwiseSimdConflictingDeclaration conflictingDeclaration =
          findRowwiseSimdConflictingDeclaration(
              getCompilerInstance().getASTContext());
      if (conflictingDeclaration) {
        const auto diagnostic =
            getCompilerInstance().getDiagnostics().getCustomDiagID(
                clang::DiagnosticsEngine::Error,
                "--target-recipe=dav-3510-rowwise-simd-v1 rejects "
                "input declaration or asm '%0' reserved by the row-wise "
                "SIMD recipe");
        getCompilerInstance().getDiagnostics()
            .Report(conflictingDeclaration.location, diagnostic)
            << conflictingDeclaration.name;
        return;
      }
    }
    const ascify::DavC310TargetRecipe::FinalizedEdits recipeEdits =
        davC310TargetRecipe.finalize(
            getCompilerInstance().getASTContext(), SM, PP.getLangOpts(),
            TargetRecipe == ascify::DavC310TargetRecipe::SimdRecipeName);
    unsigned insertedRecipeEdits = 0;
    for (const ascify::DavC310TargetRecipe::Edit &edit :
         recipeEdits.edits) {
      ct::Replacement replacement(SM, edit.location, 0, edit.text);
      if (insertReplacement(
              replacement, clang::FullSourceLoc(edit.location, SM)))
        ++insertedRecipeEdits;
    }
    needsDavC310TargetHeader =
        recipeEdits.needsTargetHeader && insertedRecipeEdits != 0;
    if (insertedRecipeEdits != 0) {
      llvm::errs()
          << "Ascify dav-c310 recipe: adapters(load="
          << recipeEdits.directLoadAdapters << ",store="
          << recipeEdits.directStoreAdapters
          << "), direct_wrappers(softmax="
          << recipeEdits.softmaxDirectWrappers << ",rmsnorm="
          << recipeEdits.rmsNormDirectWrappers << ",layernorm="
          << recipeEdits.layerNormDirectWrappers << ")\n";
    }
  }
  // Start lexing the specified input file.
  llcompat::Memory_Buffer FromFile = llcompat::getMemoryBuffer(SM);
  clang::Lexer RawLex(SM.getMainFileID(), FromFile, SM, PP.getLangOpts());
  RawLex.SetKeepWhitespaceMode(true);
  // Perform a token-level rewrite of CUDA identifiers to hip ones. The raw-mode lexer gives us enough
  // information to tell the difference between identifiers, string literals, and "other stuff". It also
  // ignores preprocessor directives, so this transformation will operate inside preprocessor-deleted code.
  rawTokenWindow.clear();
  clang::Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  while (RawTok.isNot(clang::tok::eof)) {
    while (rawTokenWindow.size() >= kRawTokenWindowCap)
      rawTokenWindow.pop_front();
    rawTokenWindow.push_back(RawTok);
    if (RewriteToken(RawLex, RawTok))
      continue;
    RawLex.LexFromRawLexer(RawTok);
  }
  if (!nvidiaSampleHelperIncludes.empty()) {
    for (clang::FileID file : nvidiaSampleRetainedFileIds)
      auditRawPublishedCompatTokensInFile(file);
  }
  // Every helper edit remains staged until the raw pass reaches EOF. Any
  // earlier target-recipe return leaves this false, so EndSourceFileAction
  // cannot commit a closure that skipped raw preprocessor auditing.
  nvidiaSampleHelperRawAuditCompleted = true;
}

void AscifyAction::AddSkippedSourceRange(clang::SourceRange Range) {
  SkippedSourceRanges.push_back(Range);
  if (TargetRecipe != ascify::DavC310TargetRecipe::SimdRecipeName)
    return;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::SourceLocation begin =
      sourceManager.getFileLoc(Range.getBegin());
  if (begin.isInvalid() || sourceManager.isInSystemHeader(begin))
    return;
  const clang::FileID file = sourceManager.getFileID(begin);
  if (file.isInvalid() ||
      !rowwiseSimdRawScannedFiles.insert(file.getHashValue()).second)
    return;
  const RawRowwiseSimdConflict conflict =
      rawFileRowwiseSimdConflict(getCompilerInstance(), file, true);
  if (!conflict.macro.empty())
    rowwiseSimdMacrosEverDefined.insert(conflict.macro);
  if (rowwiseSimdRawConflictingDeclaration.empty())
    rowwiseSimdRawConflictingDeclaration = conflict.declaration;
}

void AscifyAction::run(const mat::MatchFinder::MatchResult &Result) {
  if (davC310TargetRecipe.collect(Result))
    return;
  if (Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>(sCudaLaunchKernel) != nullptr) {
    (void)cudaLaunchKernel(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::ParmVarDecl>(
          sCudaGlobalScalarDoubleParam) != nullptr) {
    (void)lowerCudaGlobalScalarDoubleParam(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::VarDecl>(sCudaDefaultDim3) != nullptr) {
    (void)rewriteCudaDefaultDim3(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::CallExpr>(
          sProvenGlobalAtomicCall) != nullptr) {
    (void)rewriteProvenGlobalAtomicCall(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::ForStmt>(
          sCanonicalWarpAddReduction) != nullptr) {
    (void)rewriteCanonicalWarpAddReduction(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::CXXRecordDecl>(
          sCanonicalBinaryReducer) != nullptr) {
    (void)tagCanonicalBinaryReducer(Result);
    return;
  }
}
