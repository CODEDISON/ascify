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
#include <functional>
#include <initializer_list>
#include <iterator>
#include <string>
#include <utility>
#include "AscifyAction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/DirectoryLookup.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
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
#include "HostCxxRootInput.h"
#include "ImplicitCudaHeaders.h"
#include "CUDA2DPP.h"
#include "StringUtils.h"
#include "DeviceMemoryLowering.h"
#include "UniformBlockReduction.h"
#include "CudaVec3AbiBoundary.h"
#include "CudaTextureBoundary.h"
#include "BoundedDeviceTemplateInlining.h"
#include "DeviceAggregateInlining.h"
#include "StandaloneDeviceTarget.h"
#include "Scalar64Lowering.h"
#include "NttpVoidCall.h"
#include "FullWarpGroups.h"

using namespace ascify;

const std::string sTargetPlatform = "Ascend";
const std::string s_string_literal = "[string literal]";
// Matchers' names
const StringRef sCudaGlobalScalarDoubleParam = "cudaGlobalScalarDoubleParam";
const StringRef sCudaDefaultDim3 = "cudaDefaultDim3";
const StringRef sCudaHalf2DirectInit = "cudaHalf2DirectInit";
const StringRef sCudaDeviceSymbol = "cudaDeviceSymbol";
const StringRef sCudaHalf2Operator = "cudaHalf2Operator";
const StringRef sFrontendHostMath = "frontendHostMath";
const StringRef sProvenGlobalAtomicCall = "provenGlobalAtomicCall";
const StringRef sCudaLaneIdAsm = "cudaLaneIdAsm";
const StringRef sCudaAttributeGuard = "cudaAttributeGuard";
const StringRef sGridConstantParameterGuard = "gridConstantParameterGuard";
const StringRef sCanonicalWarpAddReduction = "canonicalWarpAddReduction";
const StringRef sCanonicalBinaryReducer = "canonicalBinaryReducer";

namespace {

bool isRecognizedNvidiaSampleHelperDrvapi(llvm::StringRef path) {
  if (path.empty() ||
      llvm::sys::path::filename(path) != "helper_cuda_drvapi.h")
    return false;
  const auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError)
    return false;
  const llvm::StringRef contents = (*bufferOrError)->getBuffer();
  // NVIDIA cuda-samples v13.0: the exact frozen Driver-API helper bytes.
  return contents.size() == 12664 &&
         contents.contains("inline CUdevice findCudaDeviceDRV(") &&
         contents.contains("bool inline findFatbinPath(") &&
         contents.contains("__checkCudaErrors");
}

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
  FrozenOfficialNvidiaSampleHelperDrvapi = 1U << 5,
  FrozenOfficialNvidiaSampleHelperException = 1U << 6,
};

std::string contextualInputSha256(llvm::StringRef contents) {
#if LLVM_VERSION_MAJOR >= 13
  llvm::SHA256 digest;
  digest.update(contents);
  const auto bytes = digest.final();
  static const char hex[] = "0123456789abcdef";
  std::string result;
  for (const auto byte : bytes) {
    const unsigned value = static_cast<unsigned char>(byte);
    result += hex[value >> 4];
    result += hex[value & 15];
  }
  return result;
#else
  return {};
#endif
}

std::string contextualCanonicalPath(llvm::StringRef path) {
  llvm::SmallString<256> resolved;
  if (path.empty() || llvm::sys::fs::real_path(path, resolved))
    return {};
  return resolved.str().str();
}

std::string contextualRecordPathBinding(LocalHeaderInputEvidence &evidence,
                                        llvm::StringRef path) {
  const auto canonical = contextualCanonicalPath(path);
  if (canonical.empty())
    return {};  // Synthetic predefines and scratch buffers have no file identity.
  const auto inserted = evidence.pathBindings.emplace(path.str(), canonical);
  if (!inserted.second && inserted.first->second != canonical)
    evidence.changedPathBinding = path.str();
  return canonical;
}

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
      filename != "helper_timer.h" && filename != "helper_cuda_drvapi.h" &&
      filename != "exception.h")
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
    // Reviewed v13.4 commit 5443602d89ed99aede2e4b7bf329daddeadb320e:
    // only the SM 0xa7 cores/name table entries differ; helper contracts and
    // all four frozen dependencies are unchanged. Admit these exact parsed
    // bytes while retaining every active-definition and call-closure proof.
    if (contents.size() == 28219 &&
        sha256Equals(
            contents,
            "0494246768a4e3dbfcd688d79f4abad1fa39fc9f2d9dc3264a894f1c96d3785d"))
      return FrozenOfficialNvidiaSampleHelperCuda;
  } else if (filename == "exception.h") {
    // NVIDIA cuda-samples v13.0 Common/exception.h: one of the four frozen
    // helper-closure dependencies (helper_cuda.h includes it and the closure
    // retains it in the translated TU). Same provenance class as the other
    // frozen roles for the dynamic-warp conditional audit.
    if (contents.size() == 6168 &&
        sha256Equals(
            contents,
            "ad043023a652bf3610671aad88237383c7bf2e17c66f8da5830d1d668fe7bbe7"))
      return FrozenOfficialNvidiaSampleHelperException;
  } else if (filename == "helper_cuda_drvapi.h") {
    // NVIDIA cuda-samples v13.0 commit b7c5481c556c3fe98db060207ecaa41a4b9a9abc:
    // the Driver-API helper surface (findCudaDeviceDRV, findFatbinPath and the
    // __checkCudaErrors macro guarded by __cuda_cuda_h__).
    if (contents.size() == 12664 &&
        sha256Equals(
            contents,
            "016f4d7ef6ee9493835e7a142e87125fe64181ac1f5ce549dca8b919df088e05"))
      return FrozenOfficialNvidiaSampleHelperDrvapi;
  } else if (filename == "helper_string.h") {
    // Reviewed v13.0 commit 3f1c50965017932fc81e6d94a3fc9e04c105b312:
    // sdkFindFilePath retains the original Samples/ search paths. This header
    // stays included in the generated TU; only its exact provenance is added.
    if (contents.size() == 14875 &&
        sha256Equals(
            contents,
            "616ac8b2588a6e73aed207f85faf0da5e617404390df1790f6637b1042dbcc94"))
      return FrozenOfficialNvidiaSampleHelperString;
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
      "ASCIFY_REGISTER_DEVICE_SYMBOL",
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
      "ASCIFY_REGISTER_DEVICE_SYMBOL",
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
const char *const DynamicWarpCompatIdentifiers[] = {
#include "DynamicWarpCompatIdentifiers.inc"
};
bool isDynamicWarpCompatIdentifier(llvm::StringRef name) {
  return std::find_if(std::begin(DynamicWarpCompatIdentifiers),
      std::end(DynamicWarpCompatIdentifiers),
      [&](const char* candidate) { return name==candidate; })!=
      std::end(DynamicWarpCompatIdentifiers);
}
bool isDynamicWarpControlledToken(llvm::StringRef name) {
  return isDynamicWarpCompatIdentifier(name)||name.starts_with("_CG_")||
      name.starts_with("_COOPERATIVE_GROUPS_");
}
bool isNewDynamicProjectionName(llvm::StringRef name) {
  return name=="cooperative_groups"||name=="grid_group"||name=="coalesced_group"||name=="this_grid"||
      name=="coalesced_threads"||name=="binary_partition"||
      name=="labeled_partition"||name=="greater";
}

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

bool activeFrozenHelperImageMinBodyMatches(
    const clang::MacroInfo &macroInfo,
    clang::Preprocessor &preprocessor) {
  if (!macroInfo.isFunctionLike() || macroInfo.isVariadic() ||
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
  for (const clang::Token &token : macroInfo.tokens())
    actual.push_back(preprocessor.getSpelling(token));
  const std::vector<std::string> expected = {
      "(", "(", left, "<", right, ")", "?", left, ":", right, ")"};
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
  if (isRecognizedNvidiaSampleHelperCuda(sourceManager.getFilename(spelling)))
    return true;
  // The Driver-API helper plays the same closure role for vectorAddDrv-style
  // samples that include it instead of helper_cuda.h.
  const llvm::StringRef path = sourceManager.getFilename(spelling);
  return !path.empty() &&
         llvm::sys::path::filename(path) == "helper_cuda_drvapi.h" &&
         frozenNvidiaSampleHelperFileRole(sourceManager, spelling) ==
             FrozenOfficialNvidiaSampleHelperDrvapi;
}

bool isExactAscifyCudaCompatPath(llvm::StringRef path) {
  if (path.empty() ||
      llvm::sys::path::filename(path) != "ascify_cuda_compat.hpp")
    return false;
  const auto bufferOrError = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrError)
    return false;
  const llvm::StringRef contents = (*bufferOrError)->getBuffer();
  if (contents.size() != 136614)
    return false;
#if LLVM_VERSION_MAJOR >= 13
  return sha256Equals(
      contents,
      "0f01fd41b19087943d79bcce6019a2e93a47d7e17df8abda4ec0a4020b9ac12e");
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
  if (invalidBuffer || contents.size() != 136614)
    return false;
#if LLVM_VERSION_MAJOR >= 13
  return sha256Equals(
      contents,
      "0f01fd41b19087943d79bcce6019a2e93a47d7e17df8abda4ec0a4020b9ac12e");
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
    // helper_cuda_drvapi.h defines the same macro name with a Driver-API
    // body: __checkCudaErrors(err, __FILE__, __LINE__). Both bodies carry the
    // value plus file/line and exit on failure; admit the exact spelling.
    if (actual != expected) {
      expected = {"__checkCudaErrors", "(", parameterName, ",",
                  "__FILE__", ",", "__LINE__", ")"};
    }
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

static bool sdkCudaPriorityDeclaration(
    const clang::FunctionDecl *function, clang::SourceManager &source,
    const std::set<unsigned> &trusted);

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
  // Every admitted name is implemented by ascify_cuda_compat.hpp or its
  // symbol adapter with an aclError result. Errors (including unavailable
  // device symbol registration) remain errors; domain proof is not a claim
  // that every runtime operation succeeds. Driver APIs, library statuses,
  // and arbitrary integer-valued calls stay outside the set even when
  // wrapped by NVIDIA's generic checkCudaErrors macro. The documented
  // occupancy estimate is admitted because it is implemented with an
  // aclError result like every other name here.
  static const char *const admittedNames[] = {
      "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaMemcpyAsync",
      "cudaMallocPitch", "cudaMallocArray", "cudaMalloc3DArray", "cudaFreeArray",
      "cudaMemcpy2D", "cudaMemset2D", "cudaMemcpyToArray", "cudaMemcpy2DToArray", "cudaMemcpy3D",
      "cudaCreateTextureObject", "cudaDestroyTextureObject",
      "cudaMemset", "cudaMemsetAsync", "cudaMallocHost",
      "cudaHostAlloc", "cudaHostRegister", "cudaHostUnregister",
      "cudaHostGetDevicePointer", "cudaSetDeviceFlags",
      "cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetName",
      "cuDeviceGetAttribute", "cuGetErrorString",
      "cuCtxCreate", "cuCtxDestroy", "cuCtxSynchronize",
      "cuModuleLoadData", "cuModuleUnload", "cuModuleGetFunction",
      "cuMemAlloc", "cuMemFree", "cuMemcpyHtoD", "cuMemcpyDtoH",
      "cuLaunchKernel",
      "cuCtxCreate_v4", "cuMemAlloc_v2", "cuMemFree_v2", "cuCtxDestroy_v2",
      "cuMemcpyHtoD_v2", "cuMemcpyDtoH_v2",
      "cuMemcpyHtoD_v2_ptds", "cuMemcpyDtoH_v2_ptds",
      "cudaProfilerStart", "cudaProfilerStop",
      "cudaLaunchHostFunc", "cudaStreamAddCallback",
      "cudaFreeHost", "cudaGetDevice", "cudaSetDevice",
      "cudaGetDeviceCount", "cudaDeviceGetAttribute",
      "cudaDeviceCanAccessPeer", "cudaDeviceEnablePeerAccess",
      "cudaDeviceDisablePeerAccess",
      "cudaGetDeviceProperties", "cudaGetDeviceProperties_v2",
      "cudaMemcpyToSymbol", "cudaMemcpyFromSymbol",
      "cudaDeviceSynchronize", "cudaDeviceReset",
      "cudaFuncGetAttributes", "cudaFuncSetAttribute",
      "cudaStreamCreate", "cudaStreamCreateWithFlags",
      "cudaDeviceGetStreamPriorityRange", "cudaStreamCreateWithPriority",
      "cudaStreamDestroy", "cudaStreamSynchronize",
      "cudaStreamQuery", "cudaStreamWaitEvent",
      "cudaEventCreate", "cudaEventCreateWithFlags", "cudaEventDestroy", "cudaEventRecord",
      "cudaEventSynchronize", "cudaEventElapsedTime", "cudaEventQuery",
      "cudaGetLastError", "cudaPeekAtLastError",
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
      "cudaOccupancyMaxPotentialBlockSize",
  };
  if (std::find_if(std::begin(admittedNames), std::end(admittedNames),
                   [&](const char *admitted) { return name == admitted; }) ==
      std::end(admittedNames))
    return false;
  const auto mapped = CUDA_RENAMES_MAP().find(name);
  if (mapped == CUDA_RENAMES_MAP().end() ||
      (mapped->second.apiType != API_RUNTIME &&
       mapped->second.apiType != API_DRIVER) ||
      !mapped->second.dppName.starts_with("ascify::"))
    return false;
  // API_DRIVER entries are the reviewed ascify::cu* Driver line: each returns
  // a real aclError exactly like the Runtime names above, so the checkCudaErrors
  // status domain is proven for them too. Anything else Driver-shaped that is
  // not in the map keeps failing at the find() above.
  if ((name == "cudaDeviceGetStreamPriorityRange" ||
       name == "cudaStreamCreateWithPriority") &&
      !sdkCudaPriorityDeclaration(callee, sourceManager, trustedSystemFileIds))
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

std::string findConventionalOuterGuardInFile(
    clang::CompilerInstance &compiler, clang::FileID file) {
  clang::SourceManager &sourceManager = compiler.getSourceManager();
  bool invalid = false;
  const llvm::StringRef buffer = sourceManager.getBufferData(
      file, &invalid);
  if (invalid)
    return {};

  clang::Lexer lexer(
      sourceManager.getLocForStartOfFile(file),
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

std::string findConventionalMainOuterGuard(clang::CompilerInstance &compiler) {
  return findConventionalOuterGuardInFile(compiler,compiler.getSourceManager().getMainFileID());
}

// The prefix recognizer above is also used for output-header shielding; it
// does not prove that a conditional encloses the whole input. Runtime-header
// admission needs that stronger property before treating one guard level as
// an unconditional part of a selected local header. Keep the original guard
// and all its source tokens: this is not a guessed parent macro environment.
bool hasCompleteOuterGuardInFile(clang::CompilerInstance &compiler, clang::FileID file) {
  const std::string guard = findConventionalOuterGuardInFile(compiler,file);
  if (guard.empty() || isAscifyCudaCompatPublishedMacro(guard) ||
      isNvidiaSampleHelperObservableMacro(guard))
    return false;
  auto &source = compiler.getSourceManager();
  bool invalid = false;
  const auto buffer = source.getBufferData(file, &invalid);
  if (invalid) return false;
  clang::Lexer lexer(source.getLocForStartOfFile(file),
                     compiler.getLangOpts(), buffer.begin(), buffer.begin(),
                     buffer.end());
  struct RawToken { std::string spelling; bool startsLine; bool isHash; };
  std::vector<RawToken> tokens;
  clang::Token token;
  lexer.LexFromRawLexer(token);
  while (token.isNot(clang::tok::eof)) {
    if (token.isNot(clang::tok::comment)) {
      bool bad = false;
      const auto spelling = clang::Lexer::getSpelling(
          token, source, compiler.getLangOpts(), &bad);
      if (bad || token.is(clang::tok::unknown)) return false;
      // Pragma operators can restore a macro without a textual #undef. Do
      // not interpret strings or future replacement-list expansions here.
      if (token.isAnyIdentifier() &&
          (spelling == "_Pragma" || spelling == "__pragma"))
        return false;
      tokens.push_back({spelling, token.isAtStartOfLine(),
                        token.is(clang::tok::hash)});
    }
    lexer.LexFromRawLexer(token);
  }
  unsigned depth = 0;
  bool defined = false;
  for (std::size_t i = 0; i < tokens.size();) {
    // Directive identity comes from the token kind: the C++ digraph %:
    // also denotes # and must not hide guard mutations or macro-stack ops.
    if (!tokens[i].isHash || !tokens[i].startsLine) {
      if (depth == 0 || !defined) return false;
      ++i;
      continue;
    }
    std::size_t end = i + 1;
    while (end < tokens.size() && !tokens[end].startsLine) ++end;
    if (end < i + 2) return false;
    const auto &directive = tokens[i + 1].spelling;
    const auto size = end - i;
    if (i == 0) {
      if (size != 3 || directive != "ifndef" ||
          tokens[i + 2].spelling != guard) return false;
      depth = 1;
    } else if (!defined) {
      if (i != 3 || size != 3 || directive != "define" ||
          tokens[i + 2].spelling != guard) return false;
      defined = true;
    } else if (directive == "if" || directive == "ifdef" ||
               directive == "ifndef") {
      ++depth;
    } else if (directive == "endif") {
      if (depth == 0 || size != 2) return false;
      --depth;
      if (depth == 0) return end == tokens.size();
    } else if (directive == "else" || directive == "elif" ||
               directive == "elifdef" || directive == "elifndef") {
      if (depth <= 1) return false;
    } else if ((directive == "define" || directive == "undef") &&
               size >= 3 && tokens[i + 2].spelling == guard) {
      return false;
    } else if (directive == "pragma" && size >= 3 &&
               (tokens[i + 2].spelling == "push_macro" ||
                tokens[i + 2].spelling == "pop_macro")) {
      // Fail closed for every macro-stack operand, including escaped or
      // macro-generated strings, rather than guessing which name it denotes.
      return false;
    }
    i = end;
  }
  return false;
}

bool hasCompleteMainOuterGuard(clang::CompilerInstance &compiler) {
  return hasCompleteOuterGuardInFile(compiler,compiler.getSourceManager().getMainFileID());
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

  // Warn about CUDA deprecation before applying the target mapping.
  if (Statistics::isCudaDeprecated(found->second)) {
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "'%0' is deprecated in CUDA.");
    DE.Report(sl, ID) << found->first;
  }

  // Warn about the unsupported identifier.
  if (Statistics::isUnsupported(found->second)) {
    std::string sWarn = sTargetPlatform;
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "'%0' is unsupported in '%1'.");
    DE.Report(sl, ID) << found->first << sWarn;
    return;
  }
  if (!bReplace) {
    return;
  }
  StringRef repName = found->second.dppName;
  // The AST proof owns this exact original qualifier, not every __shared__
  // spelling. Merge the object alignment into its ordinary token replacement.
  if (name == "__shared__" &&
      sharedWordAlignmentPlan.qualifiers.count(sl.getRawEncoding()))
    repName = "__ubuf__ ASCIFY_ALIGN(4)";
  // The final texture boundary still authenticates the original SDK call.
  // Only an explicit fast policy selects unquantized FP32 interpolation;
  // precise output retains the bounded 1/256-weight sampler.
  if (TargetPolicy == "dav-c310-vec" && SimtMathMode == "fast") {
    if (name == "tex2D" && repName == "ascify::tex2D")
      repName = "ascify::tex2DFast";
    else if (name == "tex2DLayered" && repName == "ascify::tex2DLayered")
      repName = "ascify::tex2DLayeredFast";
    else if (name == "texCubemap" && repName == "ascify::texCubemap")
      repName = "ascify::texCubemapFast";
  }
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

// A new header projection must use the configured physical SDK, never a
// same-named user header or a virtual override. This also anchors which macro
// definitions and record declarations belong to the removed SDK surface.
static bool pathIsWithinPhysicalDirectory(llvm::StringRef path,
                                          llvm::StringRef directory) {
  if (directory.empty() || path.empty())
    return false;
  llvm::SmallString<256> physicalRoot, physicalPath;
  if (llvm::sys::fs::real_path(directory, physicalRoot) ||
      llvm::sys::fs::real_path(path, physicalPath))
    return false;
  physicalRoot += "/";
  return llvm::StringRef(physicalPath).starts_with(physicalRoot);
}

static bool pathIsWithinConfiguredCudaInclude(llvm::StringRef path) {
  if (CudaPath.empty())
    return false;
  llvm::SmallString<256> root(CudaPath.getValue());
  llvm::sys::path::append(root, "include");
  return pathIsWithinPhysicalDirectory(path, root);
}

// These are the Runtime and forced-wrapper dependency roles verified with the
// real SDK. Merely adding a native-header shadow (or another caller file) under
// --cuda-path does not make it part of the removed SDK surface. Unknown SDK
// dependencies remain retained audit inputs until their role is established.
static bool isConfiguredCudaSdkDependencyRole(llvm::StringRef path) {
  if (!pathIsWithinConfiguredCudaInclude(path))
    return false;
  llvm::SmallString<256> root(CudaPath.getValue()), physicalRoot, physicalFile;
  llvm::sys::path::append(root, "include");
  if (llvm::sys::fs::real_path(root, physicalRoot) ||
      llvm::sys::fs::real_path(path, physicalFile))
    return false;
  const auto relative = llvm::StringRef(physicalFile).drop_front(physicalRoot.size() + 1);
  static const std::set<std::string> dependencies = {
      "builtin_types.h", "channel_descriptor.h", "cuda.h",
      "cuda_device_runtime_api.h", "cuda_runtime.h", "cuda_runtime_api.h",
      "cuda_profiler_api.h",
      "cudart_platform.h", "curand.h", "curand_mtgp32.h", "curand_mtgp32_kernel.h",
      "device_atomic_functions.h", "device_atomic_functions.hpp",
      "device_launch_parameters.h", "device_types.h", "driver_functions.h",
      "driver_types.h", "host_config.h", "host_defines.h", "library_types.h",
      "math_constants.h", "surface_indirect_functions.h", "surface_types.h",
      "texture_indirect_functions.h", "texture_types.h", "vector_functions.h",
      "vector_functions.hpp", "vector_types.h", "sm_20_atomic_functions.hpp",
      "sm_20_intrinsics.hpp", "sm_32_atomic_functions.hpp", "sm_60_atomic_functions.hpp",
      "sm_61_intrinsics.hpp", "crt/common_functions.h", "crt/device_functions.h",
      "crt/device_functions.hpp", "crt/device_double_functions.hpp", "crt/func_macro.h",
      "crt/host_config.h", "crt/host_defines.h", "crt/host_runtime.h",
      "crt/math_functions.h", "crt/math_functions.hpp", "crt/sm_70_rt.hpp",
      "crt/storage_class.h", "nv/target", "nv/detail/__target_macros",
      "nv/detail/__preprocessor", "cccl/nv/target",
      "cccl/nv/detail/__target_macros", "cccl/nv/detail/__preprocessor"};
  return dependencies.count(relative.str()) != 0;
}

// CUDA installs these three nv headers both at the legacy include root and
// below cccl. An explicit -I<SDK>/include/cccl may select the latter inode.
// Admit that finite alias only when its physical bytes match the canonical
// dependency. FileEntered independently checks the selected parsed buffer.
static bool physicalCudaSdkNvAliasMatches(llvm::StringRef expected,
                                          llvm::StringRef resolved) {
  if (!pathIsWithinConfiguredCudaInclude(expected) ||
      !pathIsWithinConfiguredCudaInclude(resolved))
    return false;
  llvm::SmallString<256> root(CudaPath.getValue()), physicalRoot;
  llvm::sys::path::append(root, "include");
  if (llvm::sys::fs::real_path(root, physicalRoot))
    return false;
  const auto relative = expected.drop_front(physicalRoot.size() + 1);
  if (relative != "nv/target" && relative != "nv/detail/__target_macros" &&
      relative != "nv/detail/__preprocessor")
    return false;
  const auto resolvedRelative = resolved.drop_front(physicalRoot.size() + 1);
  if (resolvedRelative != "cccl/" + relative.str())
    return false;
  const auto canonicalBytes = llvm::MemoryBuffer::getFile(expected);
  const auto aliasBytes = llvm::MemoryBuffer::getFile(resolved);
  return canonicalBytes && aliasBytes &&
         (*canonicalBytes)->getBuffer() == (*aliasBytes)->getBuffer();
}

static bool physicalCudaSdkNvRoleMatches(llvm::StringRef path) {
  llvm::SmallString<256> root(CudaPath.getValue()), physicalRoot, physicalFile;
  llvm::sys::path::append(root, "include");
  if (llvm::sys::fs::real_path(root, physicalRoot) ||
      llvm::sys::fs::real_path(path, physicalFile))
    return false;
  const auto relative = llvm::StringRef(physicalFile).drop_front(physicalRoot.size() + 1);
  if (relative != "cccl/nv/target" &&
      relative != "cccl/nv/detail/__target_macros" &&
      relative != "cccl/nv/detail/__preprocessor")
    return true;
  llvm::SmallString<256> canonical(physicalRoot);
  llvm::sys::path::append(canonical, relative.drop_front(5));
  return physicalCudaSdkNvAliasMatches(canonical, physicalFile);
}

static bool physicalFileIdentityMatches(llvm::StringRef path,
                                        const clang::FileEntry &entry) {
  llvm::sys::fs::UniqueID physical;
  return !llvm::sys::fs::getUniqueID(path, physical) &&
         physical == entry.getUniqueID();
}

static bool physicalSourceFileMatches(llvm::StringRef path,
                                      const clang::FileEntry &entry,
                                      clang::SourceManager &source) {
  if (source.isFileOverridden(&entry) ||
      !physicalFileIdentityMatches(path, entry))
    return false;
  const auto physical = llvm::MemoryBuffer::getFile(path);
  if (!physical)
    return false;
  const auto file = source.translateFile(&entry);
  if (file.isValid()) {
    bool invalid = false;
    const auto contents = source.getBufferData(file, &invalid);
    return !invalid && contents == (*physical)->getBuffer();
  }
  const auto contents = source.getFileManager().getBufferForFile(path);
  return contents && (*contents)->getBuffer() == (*physical)->getBuffer();
}

static unsigned physicalLineBegin(llvm::StringRef contents, unsigned offset) {
  while (offset > 0 && contents[offset - 1] != '\n' &&
         contents[offset - 1] != '\r')
    --offset;
  return offset;
}

static bool hasExactDirectiveTokens(
    llvm::StringRef line, std::initializer_list<llvm::StringRef> expected) {
  const auto tokens = directPragmaTokens(line);
  if (tokens.size() != expected.size())
    return false;
  size_t index = 0;
  for (const auto spelling : expected)
    if (tokens[index++].spelling != spelling)
      return false;
  return true;
}

// An include-search path or system_header pragma cannot own an OpenMP header.
// This optional role is the actual header of the toolchain used to build this
// converter, with an independently reviewed complete LLVM23 byte identity.
// Installed builds that no longer have that toolchain fail closed here.
static bool isFrozenNativeOpenMPHeader(clang::SourceManager &source,
                                      clang::FileID file) {
#ifdef ASCIFY_CLANG_RESOURCE_BUILD_DIR
  if (file.isInvalid())
    return false;
  const auto start = source.getLocForStartOfFile(file);
  const auto path = source.getFilename(start);
  const auto *entry = source.getFileEntryForID(file);
  llvm::SmallString<256> owned(ASCIFY_CLANG_RESOURCE_BUILD_DIR);
  llvm::sys::path::append(owned, "include", "omp.h");
  if (entry == nullptr || llvm::sys::path::filename(path) != "omp.h" ||
      !physicalFileIdentityMatches(owned, *entry) ||
      !physicalSourceFileMatches(path, *entry, source))
    return false;
  bool invalid = false;
  const auto contents = source.getBufferData(file, &invalid);
  if (invalid || contents.size() != 27483 ||
      contextualInputSha256(contents) !=
          "4066c7a384a8c9a8a241287db0e2625216746da831c61083124d11ae203ab66f")
    return false;
  const auto include = source.getIncludeLoc(file);
  if (include.isInvalid() || include.isMacroID() ||
      !source.isWrittenInMainFile(include))
    return false;
  const auto main = source.getBufferData(source.getMainFileID(), &invalid);
  if (invalid)
    return false;
  const unsigned begin = physicalLineBegin(main, source.getFileOffset(include));
  const auto end = main.find_first_of("\r\n", begin);
  return hasExactDirectiveTokens(main.slice(begin, end),
                                {"#", "include", "<", "omp", ".", "h", ">"});
#else
  return false;
#endif
}

static bool isFrozenOpenMPVariantPragma(llvm::StringRef line) {
  return hasExactDirectiveTokens(line, {"#", "pragma", "omp", "end", "declare", "variant"}) ||
      hasExactDirectiveTokens(line, {"#", "pragma", "omp", "begin", "declare", "variant", "match",
          "(", "device", "=", "{", "kind", "(", "host", ")", "}", ")"}) ||
      hasExactDirectiveTokens(line, {"#", "pragma", "omp", "begin", "declare", "variant", "match",
          "(", "device", "=", "{", "kind", "(", "nohost", ")", "}", ")"});
}

// A caller-selected ResourceDir is an include-search setting, not an owner
// identity. Only compiler-bootstrap inputs with matching product resources
// may be excluded from the retained SDK-observation audit.
static std::string ownedClangResourceDirectory() {
  static int executableAnchor;
  const auto executable = llvm::sys::fs::getMainExecutable(nullptr,
                                                         &executableAnchor);
  const auto directory = llvm::sys::path::parent_path(executable);
#ifdef ASCIFY_CLANG_RESOURCE_INSTALL_FROM_BINDIR
  llvm::SmallString<256> installed(directory);
  llvm::sys::path::append(installed, ASCIFY_CLANG_RESOURCE_INSTALL_FROM_BINDIR);
  llvm::SmallString<256> wrapper(installed);
  llvm::sys::path::append(wrapper, "include", "__clang_cuda_runtime_wrapper.h");
  if (llvm::sys::fs::exists(wrapper))
    return installed.str().str();
#endif
#if defined(ASCIFY_CLANG_RESOURCE_BUILD_DIR) && defined(ASCIFY_EXECUTABLE_BUILD_DIR)
  if (llvm::sys::fs::equivalent(directory, ASCIFY_EXECUTABLE_BUILD_DIR))
    return ASCIFY_CLANG_RESOURCE_BUILD_DIR;
#endif
  return {};
}

static bool isOwnedCudaCompilerInput(clang::FileID file,
                                     clang::CompilerInstance &compiler,
                                     bool allowFrozenSampleAncestry = false) {
  auto &source = compiler.getSourceManager();
  const auto &resourceRoot = compiler.getHeaderSearchOpts().ResourceDir;
  const auto ownedRoot = ownedClangResourceDirectory();
  if (ownedRoot.empty() || file.isInvalid() ||
      !pathIsWithinPhysicalDirectory(
          source.getFilename(source.getLocForStartOfFile(file)), resourceRoot))
    return false;
  llvm::SmallString<256> physicalRoot;
  if (llvm::sys::fs::real_path(resourceRoot, physicalRoot))
    return false;
  std::set<unsigned> visited;
  while (file.isValid() && visited.insert(file.getHashValue()).second) {
    const auto begin = source.getLocForStartOfFile(file);
    const auto filename = source.getFilename(begin);
    if (pathIsWithinPhysicalDirectory(filename, resourceRoot)) {
      const auto *entry = source.getFileEntryForID(file);
      if (entry == nullptr || !physicalSourceFileMatches(filename, *entry, source))
        return false;
      llvm::SmallString<256> physicalFile;
      if (llvm::sys::fs::real_path(filename, physicalFile))
        return false;
      const auto relative = llvm::StringRef(physicalFile).drop_front(
          physicalRoot.size() + 1);
      llvm::SmallString<256> reference(ownedRoot);
      llvm::sys::path::append(reference, relative);
      const auto expected = llvm::MemoryBuffer::getFile(reference);
      bool invalid = false;
      const auto contents = source.getBufferData(file, &invalid);
      if (!expected || invalid || contents != (*expected)->getBuffer())
        return false;
      const auto include = source.getSpellingLoc(source.getIncludeLoc(file));
      if (llvm::sys::path::filename(filename) == "__clang_cuda_runtime_wrapper.h" &&
          include.isValid() && source.getFileID(include) ==
              compiler.getPreprocessor().getPredefinesFileID())
        return true;
    } else {
      // Profiler projection also owns byte-frozen Samples helpers. Their
      // standard-library children may enter product resources independently
      // of Clang's forced CUDA wrapper (notably cuda_wrappers/new). Keep this
      // additional root opt-in so runtime-header projection is unchanged.
      if (allowFrozenSampleAncestry &&
          frozenNvidiaSampleHelperFileRole(source, begin) != 0)
        return true;
      // A resource child cannot acquire compiler ownership through a caller
      // header that merely intercepts an include in the forced wrapper.
      const auto *entry = source.getFileEntryForID(file);
      const bool sdkParent = isConfiguredCudaSdkDependencyRole(filename);
      const bool nativeParent = source.isInSystemHeader(begin) &&
          (pathIsWithinPhysicalDirectory(filename, "/usr/include") ||
           pathIsWithinPhysicalDirectory(filename, "/usr/lib/gcc"));
      if (entry == nullptr || (!sdkParent && !nativeParent) ||
          !physicalSourceFileMatches(filename, *entry, source))
        return false;
    }
    const auto include = source.getSpellingLoc(source.getIncludeLoc(file));
    if (include.isInvalid() || source.isWrittenInMainFile(include))
      return false;
    file = source.getFileID(include);
  }
  return false;
}

// This additional role is local to the full-warp source proof. Native C/C++
// headers can enter private compiler resources outside the forced CUDA wrapper.
// Include-search settings, copied resources and system_header pragmas cannot
// establish this role: require the actual product inode and parsed bytes.
static bool physicalOwnedClangInput(clang::FileID file,
                                    clang::CompilerInstance &compiler) {
  auto &source = compiler.getSourceManager();
  if (file.isInvalid()) return false;
  const auto owned = ownedClangResourceDirectory();
  const auto path = source.getFilename(source.getLocForStartOfFile(file));
  const auto *entry = source.getFileEntryForID(file);
  llvm::SmallString<256> root, selected, physical;
  if (owned.empty() || entry == nullptr ||
      llvm::sys::fs::real_path(owned, root) ||
      llvm::sys::fs::real_path(compiler.getHeaderSearchOpts().ResourceDir, selected) ||
      root != selected || !pathIsWithinPhysicalDirectory(path, root) ||
      llvm::sys::fs::real_path(path, physical)) return false;
  llvm::SmallString<256> reference(root);
  llvm::sys::path::append(reference, llvm::StringRef(physical).drop_front(root.size() + 1));
  if (!physicalFileIdentityMatches(reference, *entry) ||
      !physicalSourceFileMatches(path, *entry, source)) return false;
  bool invalid = false;
  const auto parsed = source.getBufferData(file, &invalid);
  const auto bytes = llvm::MemoryBuffer::getFile(reference);
  return !invalid && bytes && parsed == (*bytes)->getBuffer() &&
      contextualInputSha256(parsed) == contextualInputSha256((*bytes)->getBuffer());
}

static bool fullWarpCommandMacroMention(clang::FileID file,
                                        clang::CompilerInstance &compiler) {
  std::set<std::string> names;
  for (const auto &option : compiler.getPreprocessorOpts().Macros) {
    const auto name = llvm::StringRef(option.first).split('=').first.split('(').first;
    if (name.empty() || name.contains('\\') || name.contains('\n') || name.contains('\r')) return true;
    names.insert(name.str());
  }
  auto &source = compiler.getSourceManager();
  bool invalid = false;
  const auto bytes = source.getBufferData(file, &invalid);
  if (invalid) return true;
  clang::Lexer lexer(source.getLocForStartOfFile(file), compiler.getLangOpts(),
                     bytes.begin(), bytes.begin(), bytes.end());
  for (;;) {
    clang::Token token; lexer.LexFromRawLexer(token);
    if (token.is(clang::tok::eof)) return false;
    if (token.isAnyIdentifier() && names.count(clang::Lexer::getSpelling(
            token, source, compiler.getLangOpts()))) return true;
  }
}

static bool fullWarpNativeCompilerInput(clang::FileID file,
    clang::CompilerInstance &compiler, const std::set<unsigned> &initialSystemFiles) {
  auto &source = compiler.getSourceManager();
  if (!physicalOwnedClangInput(file, compiler)) return false;
  std::set<unsigned> visited;
  while (file.isValid() && visited.insert(file.getHashValue()).second) {
    if (fullWarpCommandMacroMention(file, compiler)) return false;
    if (!physicalOwnedClangInput(file, compiler)) {
      const auto at = source.getLocForStartOfFile(file);
      const auto path = source.getFilename(at);
      // The configured --cuda-path tree is an audited producer-side SDK
      // identity (separate dependency checks own its provenance); an owned
      // wrapper chain may legitimately root there, same as the host system
      // headers below.
      if (pathIsWithinConfiguredCudaInclude(path))
        return true;
      const auto *entry = source.getFileEntryForID(file);
      return entry && initialSystemFiles.count(file.getHashValue()) &&
          source.isInSystemHeader(at) && physicalSourceFileMatches(path, *entry, source) &&
          (pathIsWithinPhysicalDirectory(path, "/usr/include") ||
           pathIsWithinPhysicalDirectory(path, "/usr/lib/gcc"));
    }
    const auto include = source.getIncludeLoc(file);
    if (!include.isFileID() || source.isWrittenInMainFile(include)) return false;
    file = source.getFileID(include);
  }
  return false;
}

// This establishes a physical parent-system provider, not general ownership
// of system declarations. The existing context transaction still proves the
// original include position, complete token stream, macros and file bytes.
static bool physicalParentHostSystemMacro(
    clang::FileID definitionFile, clang::FileID useFile,
    clang::CompilerInstance &compiler,
    const std::set<unsigned> &initialSystemFiles) {
  auto &source = compiler.getSourceManager();
  if (definitionFile.isInvalid() || useFile.isInvalid() ||
      definitionFile == useFile ||
      initialSystemFiles.count(definitionFile.getHashValue()) == 0)
    return false;
  const auto begin = source.getLocForStartOfFile(definitionFile);
  const auto path = source.getFilename(begin);
  const auto *entry = source.getFileEntryForID(definitionFile);
  if (entry == nullptr || !source.isInSystemHeader(begin) ||
      isConfiguredCudaSdkDependencyRole(path) ||
      !physicalSourceFileMatches(path, *entry, source))
    return false;
  const auto &resourceRoot = compiler.getHeaderSearchOpts().ResourceDir;
  if (pathIsWithinPhysicalDirectory(path, resourceRoot)) {
    const auto ownedRoot = ownedClangResourceDirectory();
    llvm::SmallString<256> physicalRoot, physicalFile;
    if (ownedRoot.empty() ||
        llvm::sys::fs::real_path(resourceRoot, physicalRoot) ||
        llvm::sys::fs::real_path(path, physicalFile))
      return false;
    const auto relative = llvm::StringRef(physicalFile).drop_front(
        physicalRoot.size() + 1);
    llvm::SmallString<256> reference(ownedRoot);
    llvm::sys::path::append(reference, relative);
    const auto expected = llvm::MemoryBuffer::getFile(reference);
    bool invalid = false;
    const auto contents = source.getBufferData(definitionFile, &invalid);
    if (!expected || invalid || contents != (*expected)->getBuffer())
      return false;
  } else if (!compiler.getTarget().getTriple().isOSLinux() ||
             (!pathIsWithinPhysicalDirectory(path, "/usr/include") &&
              !pathIsWithinPhysicalDirectory(path, "/usr/lib/gcc"))) {
    return false;
  }
  std::set<unsigned> visited;
  auto file = definitionFile;
  while (file.isValid() && visited.insert(file.getHashValue()).second) {
    if (file == useFile)
      return false;
    // The original root can intentionally be a verified virtual proof input;
    // LocalHeader.cpp binds its saved bytes separately. Other ancestors must
    // still be actual physical files, not a remapped source of the provider.
    if (file == source.getMainFileID())
      return true;
    const auto *ancestor = source.getFileEntryForID(file);
    const auto name = source.getFilename(source.getLocForStartOfFile(file));
    if (ancestor == nullptr || !physicalSourceFileMatches(name, *ancestor, source))
      return false;
    const auto include = source.getSpellingLoc(source.getIncludeLoc(file));
    if (!include.isFileID())
      return false;
    file = source.getFileID(include);
  }
  return false;
}

static bool isConfiguredCudaRuntimeApiPath(llvm::StringRef path,
                                           clang::SourceManager &source) {
  if (!pathIsWithinConfiguredCudaInclude(path))
    return false;
  llvm::SmallString<256> expected(CudaPath.getValue()), physicalExpected,
      physicalActual;
  llvm::sys::path::append(expected, "include", "cuda_runtime_api.h");
  if (llvm::sys::fs::real_path(expected, physicalExpected) ||
      llvm::sys::fs::real_path(path, physicalActual) ||
      physicalExpected != physicalActual)
    return false;
#if LLVM_VERSION_MAJOR >= 10
  auto file = source.getFileManager().getFileRef(path);
  if (!file) {
    llvm::consumeError(file.takeError());
    return false;
  }
  return physicalSourceFileMatches(physicalExpected, file->getFileEntry(), source);
#else
  const auto file = source.getFileManager().getFile(path);
  return file && physicalSourceFileMatches(physicalExpected, **file, source);
#endif
}

// Host C++ has no forced CUDA wrapper. Authenticate this additional direct
// Runtime umbrella root before its early builtin/channel headers are entered.
static bool isConfiguredCudaRuntimeHeaderPath(llvm::StringRef path,
                                           clang::SourceManager &source) {
  if (!pathIsWithinConfiguredCudaInclude(path))
    return false;
  llvm::SmallString<256> expected(CudaPath.getValue()), physicalExpected,
      physicalActual;
  llvm::sys::path::append(expected, "include", "cuda_runtime.h");
  if (llvm::sys::fs::real_path(expected, physicalExpected) ||
      llvm::sys::fs::real_path(path, physicalActual) ||
      physicalExpected != physicalActual)
    return false;
#if LLVM_VERSION_MAJOR >= 10
  auto file = source.getFileManager().getFileRef(path);
  if (!file) {
    llvm::consumeError(file.takeError());
    return false;
  }
  return physicalSourceFileMatches(physicalExpected, file->getFileEntry(), source);
#else
  const auto file = source.getFileManager().getFile(path);
  return file && physicalSourceFileMatches(physicalExpected, **file, source);
#endif
}

static bool verifiedFrontendCgFile(
    clang::FileID file, clang::CompilerInstance &compiler,
    const ascify::FrontendCompatibilityConfig &profile) {
  auto &source = compiler.getSourceManager();
  if (file.isInvalid()) return false;
  const auto *entry = source.getFileEntryForID(file);
  const auto path = source.getFilename(source.getLocForStartOfFile(file));
  bool invalid = false;
  const auto bytes = source.getBufferData(file, &invalid);
  return !invalid && entry != nullptr &&
      ascify::IsVerifiedFrontendCooperativeGroupsInput(
          profile, path.str(), bytes.str()) &&
      physicalSourceFileMatches(path, *entry, source);
}

static bool verifiedForcedFrontendHostMathFile(
    clang::FileID file, clang::CompilerInstance &compiler,
    const ascify::FrontendCompatibilityConfig &profile) {
  auto &source = compiler.getSourceManager();
  if (file.isInvalid()) return false;
  const auto *entry = source.getFileEntryForID(file);
  const auto begin = source.getLocForStartOfFile(file);
  const auto include = source.getIncludeLoc(file);
  bool invalid = false;
  const auto bytes = source.getBufferData(file, &invalid);
  return !invalid && entry != nullptr && include.isValid() &&
      source.getFileID(source.getSpellingLoc(include)) ==
          compiler.getPreprocessor().getPredefinesFileID() &&
      ascify::IsVerifiedFrontendHostMathInput(
          profile, source.getFilename(begin).str(), bytes.str()) &&
      physicalSourceFileMatches(source.getFilename(begin), *entry, source);
}

static bool ordinaryRuntimeProfileInput(clang::CompilerInstance &compiler) {
  const auto &pp = compiler.getPreprocessorOpts();
  const auto &lang = compiler.getLangOpts();
  for (const auto &macro : pp.Macros) {
    const llvm::StringRef definition(macro.first);
    if (definition.split('=').first ==
            "ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_" ||
        definition.contains('\n') || definition.contains('\r'))
      return false;
  }
  return lang.CUDA && !pp.RetainExcludedConditionalBlocks &&
      !lang.Modules && !lang.CPlusPlusModules &&
      pp.ImplicitPCHInclude.empty() && pp.PCHThroughHeader.empty() &&
      pp.PrecompiledPreambleBytes.first == 0 &&
      !pp.PrecompiledPreambleBytes.second &&
      compiler.getFrontendOpts().ModuleFiles.empty() &&
      compiler.getASTContext().getExternalSource() == nullptr;
}

static bool frontendCgMacroTokens(
    const clang::MacroInfo &macro, clang::Preprocessor &preprocessor,
    std::initializer_list<llvm::StringRef> expected, bool parameter = false) {
  if (macro.isFunctionLike() != parameter || macro.isC99Varargs() ||
      macro.isGNUVarargs() || macro.getNumTokens() != expected.size())
    return false;
  if (parameter && (macro.getNumParams() != 1 ||
                    (*macro.param_begin())->getName() != "a"))
    return false;
  auto token = macro.tokens_begin();
  for (const auto spelling : expected)
    if (preprocessor.getSpelling(*token++) != spelling) return false;
  return true;
}

static bool trustedFrontendCgCudaControl(
    llvm::StringRef name, const clang::MacroInfo *macro,
    clang::CompilerInstance &compiler) {
  if (macro == nullptr ||
      !frontendCgMacroTokens(*macro, compiler.getPreprocessor(), {"1"}))
    return false;
  auto &source = compiler.getSourceManager();
  const auto definition = source.getSpellingLoc(macro->getDefinitionLoc());
  const auto file = source.getFileID(definition);
  const bool builtin = definition.isValid() &&
      file == compiler.getPreprocessor().getPredefinesFileID() &&
      source.isWrittenInBuiltinFile(definition) &&
      !source.isWrittenInCommandLineFile(definition);
  return builtin || (name == "__CUDACC__" &&
                     isOwnedCudaCompilerInput(file, compiler));
}

static bool frontendCgControlName(llvm::StringRef name) {
  return name == "__CUDA__" || name == "__CUDACC__" ||
      name == "ASCIFY_FRONTEND_COMPAT_DEVICE_" ||
      name == "ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_" ||
      name == "ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_" ||
      name == "ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_REDUCE_H_";
}

// Original CUDA 13.4 CG controls, reviewed against the complete registered
// top/reduce/info/driver_abi/helpers/memory inputs. No SDK MacroInfo authority
// follows from this list. The supported projection requires caller silence.
static const std::set<std::string> &originalCgControls() {
  static const std::set<std::string> names = {
      "CG_DEBUG", "LOAD_ENVREG", "_CG_ABI_EXPERIMENTAL", "_CG_ABI_VERSION",
      "_CG_ABORT", "_CG_ASM_PTR_CONSTRAINT", "_CG_ASSERT", "_CG_BEGIN_NAMESPACE",
      "_CG_CLUSTER_INTRINSICS_AVAILABLE", "_CG_CONCAT_INNER", "_CG_CONCAT_OUTER",
      "_CG_CONSTEXPR_QUALIFIER", "_CG_CONST_DECL", "_CG_CPP11_FEATURES",
      "_CG_CUDA_ARCH", "_CG_DEBUG", "_CG_DEPRECATED", "_CG_DRIVER_API_H",
      "_CG_END_NAMESPACE", "_CG_HAS_CLUSTER_GROUP", "_CG_HAS_FP16_COLLECTIVE",
      "_CG_HAS_GRID_GROUP", "_CG_HAS_INSTR_ELECT", "_CG_HAS_MATCH_COLLECTIVE",
      "_CG_HAS_MULTI_GRID_GROUP", "_CG_HAS_RESERVED_SHARED", "_CG_HAS_STL_ATOMICS",
      "_CG_INFO_H_", "_CG_LIMIT_INCLUDED_DEPENDENCIES", "_CG_NAMESPACE",
      "_CG_QUALIFIER", "_CG_SKIP_BARRIER_INIT_TARGET", "_CG_STATIC_CONSTEXPR_QUALIFIER",
      "_CG_STATIC_CONST_DECL", "_CG_STATIC_QUALIFIER", "_CG_STL_NAMESPACE",
      "_CG_THREAD_SCOPE", "_CG_USER_PROVIDED_SHARED_MEMORY", "_CG_USE_CUDA_STL",
      "_CG_USE_OWN_TRAITS", "_CG_VERSION", "_COOPERATIVE_GROUPS_HELPERS_H_",
      "_COOPERATIVE_GROUPS_H_", "_COOPERATIVE_GROUPS_MEMORY_H_",
      "_COOPERATIVE_GROUPS_REDUCE_H"};
  return names;
}

static bool originalCgControlName(llvm::StringRef name) {
  return originalCgControls().count(name.str()) != 0;
}

static bool originalCgControlText(llvm::StringRef text) {
  for (const auto &name : originalCgControls())
    if (text.contains(name)) return true;
  return false;
}

static bool trustedFrontendCgControlOwner(
    llvm::StringRef name, clang::SourceLocation location,
    clang::CompilerInstance &compiler,
    const ascify::FrontendCompatibilityConfig &profile) {
  auto &source = compiler.getSourceManager();
  const auto written = source.getSpellingLoc(location);
  if (written.isInvalid()) return false;
  const auto file = source.getFileID(written);
  if (name == "__CUDA__" || name == "__CUDACC__")
    return (file == compiler.getPreprocessor().getPredefinesFileID() &&
            source.isWrittenInBuiltinFile(written) &&
            !source.isWrittenInCommandLineFile(written)) ||
        (name == "__CUDACC__" && isOwnedCudaCompilerInput(file, compiler));
  return verifiedFrontendCgFile(file, compiler, profile);
}

static bool frontendCgMacroExpansionMatches(
    llvm::StringRef name, const clang::MacroInfo *macro,
    clang::FileID useFile, clang::CompilerInstance &compiler,
    const ascify::FrontendCompatibilityConfig &profile) {
  if (macro == nullptr) return false;
  auto &source = compiler.getSourceManager();
  auto &preprocessor = compiler.getPreprocessor();
  const auto definition = source.getSpellingLoc(macro->getDefinitionLoc());
  if (definition.isInvalid()) return false;
  const auto file = source.getFileID(definition);
  if (name == "ASCIFY_FRONTEND_COMPAT_DEVICE_" ||
      name == "ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_")
    return file == useFile && verifiedFrontendCgFile(file, compiler, profile) &&
        frontendCgMacroTokens(*macro, preprocessor, {"__device__"});
  // The Linux CUDA SDK expands device through three exact macros. Do not
  // trust their names alone, copied definitions, or a caller redefinition.
  const auto path = source.getFilename(definition);
  const auto *entry = source.getFileEntryForID(file);
  llvm::SmallString<256> expected(CudaPath.getValue()), actualPath, expectedPath;
  llvm::sys::path::append(expected, "include", "crt", "host_defines.h");
  if (entry == nullptr || !compiler.getTarget().getTriple().isOSLinux() ||
      !isConfiguredCudaSdkDependencyRole(path) ||
      llvm::sys::fs::real_path(path, actualPath) ||
      llvm::sys::fs::real_path(expected, expectedPath) ||
      actualPath != expectedPath ||
      !physicalSourceFileMatches(path, *entry, source))
    return false;
  if (name == "__device__")
    return frontendCgMacroTokens(*macro, preprocessor,
                                 {"__location__", "(", "device", ")"});
  if (name == "__location__")
    return frontendCgMacroTokens(*macro, preprocessor,
                                 {"__annotate__", "(", "a", ")"}, true);
  if (name == "__annotate__")
    return frontendCgMacroTokens(*macro, preprocessor,
                                 {"__attribute__", "(", "(", "a", ")", ")"}, true);
  return false;
}

// The Linux stdio entry may use the product's stdarg header even though it is
// outside the forced CUDA wrapper ancestry. This proves only that exact file,
// not ownership of its resource directory or any other inquiry in the file.
static bool productStdargFromNativeStdio(clang::FileID file,
                                        clang::CompilerInstance &compiler) {
  auto &source = compiler.getSourceManager();
  const auto begin = source.getLocForStartOfFile(file);
  const auto path = source.getFilename(begin);
  const auto &resources = compiler.getHeaderSearchOpts().ResourceDir;
  const auto owned = ownedClangResourceDirectory();
  llvm::SmallString<256> selected(resources), reference(owned), physicalSelected,
      physicalPath;
  llvm::sys::path::append(selected, "include", "stdarg.h");
  llvm::sys::path::append(reference, "include", "stdarg.h");
  const auto *entry = source.getFileEntryForID(file);
  if (owned.empty() || entry == nullptr ||
      llvm::sys::fs::real_path(selected, physicalSelected) ||
      llvm::sys::fs::real_path(path, physicalPath) ||
      physicalSelected != physicalPath ||
      !physicalSourceFileMatches(path, *entry, source))
    return false;
  bool invalid = false;
  const auto expected = llvm::MemoryBuffer::getFile(reference);
  const auto contents = source.getBufferData(file, &invalid);
  if (!expected || invalid || contents != (*expected)->getBuffer())
    return false;
  const auto include = source.getSpellingLoc(source.getIncludeLoc(file));
  if (include.isInvalid())
    return false;
  const auto parentFile = source.getFileID(include);
  if (parentFile.isInvalid())
    return false;
  const auto parentBegin = source.getLocForStartOfFile(parentFile);
  const auto parentPath = source.getFilename(parentBegin);
  const auto *parentEntry = source.getFileEntryForID(parentFile);
  llvm::SmallString<256> physicalParent;
  return parentEntry != nullptr && source.isInSystemHeader(parentBegin) &&
      !llvm::sys::fs::real_path(parentPath, physicalParent) &&
      physicalParent == "/usr/include/stdio.h" &&
      physicalSourceFileMatches(parentPath, *parentEntry, source);
}

// Member-pointer class access changed with the value NestedNameSpecifier API.
// Select the available primary API rather than guessing a version boundary.
template <class Member>
static auto runtimeHeaderMemberClass(const Member *member, int)
    -> decltype(member->getClass(), clang::QualType()) {
  return clang::QualType(member->getClass(), 0);
}
template <class Member>
static auto runtimeHeaderMemberClass(const Member *member, long)
    -> decltype(member->getQualifier().getAsType(), clang::QualType()) {
  const auto qualifier = member->getQualifier();
  using Qualifier = decltype(member->getQualifier());
  if (qualifier.getKind() != Qualifier::Kind::Type)
    return {};
  return clang::QualType(qualifier.getAsType(), 0);
}

template <typename Visitor>
static auto traverseRuntimeIdentityTypeLoc(Visitor &visitor,
    clang::TemplateSpecializationTypeLoc type, bool qualifier, int)
    -> decltype(visitor.TraverseTemplateSpecializationTypeLoc(type, qualifier)) {
  return visitor.TraverseTemplateSpecializationTypeLoc(type, qualifier);
}
template <typename Visitor>
static auto traverseRuntimeIdentityTypeLoc(Visitor &visitor,
    clang::TemplateSpecializationTypeLoc type, bool, long)
    -> decltype(visitor.TraverseTemplateSpecializationTypeLoc(type)) {
  return visitor.TraverseTemplateSpecializationTypeLoc(type);
}

static bool requiresCudaCompatHeader(llvm::StringRef name) {
  return ascify::IsCudaTextureName(name) || name == "__global__" ||
         name == "cudaMalloc" ||
         name == "cudaFree" ||
         name == "cudaMemcpy" ||
         name == "cudaMemcpyToSymbol" || name == "cudaMemcpyFromSymbol" ||
         name == "cudaMemcpyAsync" ||
         name == "cudaMemset" ||
         name == "cudaMemsetAsync" ||
         name == "cudaMallocHost" ||
         name == "cudaProfilerStart" ||
         name == "cudaProfilerStop" ||
         name == "cudaLaunchHostFunc" ||
         name == "cudaHostFn_t" ||
         name == "cudaFreeHost" ||
         name == "cudaGetDevice" ||
         name == "cudaSetDevice" ||
         name == "cudaGetDeviceCount" ||
         name == "cudaDeviceCanAccessPeer" ||
         name == "cudaDeviceEnablePeerAccess" ||
         name == "cudaDeviceDisablePeerAccess" ||
         name == "cudaDeviceSynchronize" ||
         name == "cudaDeviceReset" ||
         name == "cudaGetLastError" ||
         name == "cudaGetErrorString" ||
         name == "cudaDeviceGetAttribute" ||
         name == "cudaDeviceProp" ||
         name == "cudaGetDeviceProperties" ||
         name == "cudaGetDeviceProperties_v2" ||
         name == "__any_sync" ||
         name == "__all_sync" ||
         name == "cudaStreamCreate" ||
         name == "cudaStreamCreateWithFlags" ||
         name == "cudaDeviceGetStreamPriorityRange" ||
         name == "cudaStreamCreateWithPriority" ||
         name == "cudaStreamDefault" ||
         name == "cudaStreamNonBlocking" ||
         name == "cudaStreamLegacy" ||
         name == "cudaStreamPerThread" ||
         name == "cudaStreamDestroy" ||
         name == "cudaStreamSynchronize" ||
         name == "cudaStreamQuery" ||
         name == "cudaStreamWaitEvent" ||
         name == "cudaEventWaitDefault" ||
         name == "cudaEventWaitExternal" ||
         name == "cudaEventCreate" ||
         name == "cudaEventDestroy" ||
         name == "cudaEventRecord" ||
         name == "cudaEventQuery" ||
         name == "cudaEventSynchronize" ||
         name == "cudaEventElapsedTime" ||
         name == "cudaDevAttrWarpSize" ||
         name == "cudaDevAttrMaxThreadsPerMultiProcessor" ||
         name == "cudaDevAttrMaxThreadsPerBlock" ||
         name == "cudaDevAttrMaxSharedMemoryPerBlock" ||
         name == "cudaDevAttrMaxSharedMemoryPerBlockOptin" ||
         name == "cudaPeekAtLastError" ||
         name == "cudaOccupancyMaxActiveBlocksPerMultiprocessor" ||
         name == "cudaOccupancyMaxPotentialBlockSize" ||
         name == "cudaErrorAssert" ||
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

// findCudaDevice is also used to initialize ordinary automatic locals in
// NVIDIA's complete Samples (for example, template's devID). The shared
// statement-only walker intentionally does not cross declarations: retain
// that policy for its other semantic callers and prove this narrow path here.
static const clang::FunctionDecl *enclosingNvidiaFindDeviceFunction(
    const clang::Stmt *statement,
    clang::ASTContext &context) {
  const clang::Stmt *current = statement;
  std::set<const clang::Stmt *> active;
  while (current != nullptr && active.insert(current).second) {
    const auto parents = context.getParents(*current);
    if (parents.size() != 1)
      return nullptr;
    const clang::DynTypedNode &parent = parents[0];
    if (parent.get<clang::LambdaExpr>() != nullptr)
      return nullptr;
    if (const auto *function = parent.get<clang::FunctionDecl>()) {
      const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(function);
      if (method != nullptr && method->getParent()->isLambda())
        return nullptr;
      return function;
    }
    if (const auto *variable = parent.get<clang::VarDecl>()) {
      // Namespace/global, static-local, thread-local, parameters and lambda
      // captures must not acquire an ordinary host-call execution context.
      if (!variable->isLocalVarDecl() || !variable->hasLocalStorage() ||
          variable->getTLSKind() != clang::VarDecl::TLS_None)
        return nullptr;
      const auto declarationParents = context.getParents(*variable);
      if (declarationParents.size() != 1)
        return nullptr;
      current = declarationParents[0].get<clang::DeclStmt>();
    } else {
      current = parent.get<clang::Stmt>();
    }
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

// Exact SDK status/prototype proof for the two added priority entrypoints.
// This does not broaden Runtime-header projection or any caller-source seal.
static bool sdkCudaPriorityDeclaration(
    const clang::FunctionDecl *function, clang::SourceManager &source,
    const std::set<unsigned> &trusted) {
  if (!function || !function->getIdentifier() ||
      !function->getDeclContext()->getRedeclContext()->isTranslationUnit() ||
      function->getQualifiedNameAsString() != function->getNameAsString() ||
      function->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      !function->isExternC() || function->isVariadic() ||
      !allFunctionRedeclarationsComeFromInitiallyTrustedSystemFiles(function,source,trusted))
    return false;
  const bool range=function->getName()=="cudaDeviceGetStreamPriorityRange";
  if (!range && function->getName()!="cudaStreamCreateWithPriority") return false;
  auto &context=function->getASTContext();
  const auto physicalSdk=[&](clang::SourceLocation at) {
    if (at.isInvalid() || !at.isFileID()) return false;
    const auto *entry=source.getFileEntryForID(source.getFileID(at));
    const auto path=source.getFilename(at);
    return entry && trusted.count(source.getFileID(at).getHashValue()) &&
        pathIsWithinConfiguredCudaInclude(path) && physicalSourceFileMatches(path,*entry,source);
  };
  for (const auto *declaration:function->getCanonicalDecl()->redecls()) {
    if (declaration->doesThisDeclarationHaveABody() || declaration->isVariadic() ||
        declaration->getNumParams()!=(range?2U:3U) ||
        !physicalSdk(declaration->getLocation()) ||
        !isConfiguredCudaRuntimeApiPath(source.getFilename(declaration->getLocation()),source))
      return false;
    const auto *type=declaration->getReturnType()->getAs<clang::EnumType>();
    const auto *status=type?type->getDecl():nullptr;
    if (!status || !status->getIdentifier() || status->getName()!="cudaError" ||
        !status->getDeclContext()->getRedeclContext()->isTranslationUnit()) return false;
    for (const auto *redecl:status->redecls())
      if (!physicalSdk(redecl->getLocation())) return false;
    if (range) {
      for (const auto *parameter:declaration->parameters()) {
        auto pointer=parameter->getType();
        if (pointer.isNull() || pointer.getCVRQualifiers() || !pointer->isPointerType() ||
            !context.hasSameType(pointer->getPointeeType(),context.IntTy)) return false;
      }
    } else {
      auto pointer=declaration->getParamDecl(0)->getType();
      for (unsigned level=0;level!=2;++level) {
        if (pointer.isNull() || pointer.getCVRQualifiers() || !pointer->isPointerType()) return false;
        pointer=pointer->getPointeeType();
      }
      const auto *type=pointer.isNull()?nullptr:pointer->getAs<clang::RecordType>();
      const auto *stream=type?type->getDecl():nullptr;
      if (!stream || pointer.getCVRQualifiers() || !stream->getIdentifier() ||
          stream->getName()!="CUstream_st" || stream->getDefinition() ||
          !stream->getDeclContext()->getRedeclContext()->isTranslationUnit() ||
          !context.hasSameType(declaration->getParamDecl(1)->getType(),context.UnsignedIntTy) ||
          !context.hasSameType(declaration->getParamDecl(2)->getType(),context.IntTy)) return false;
      for (const auto *redecl:stream->redecls())
        if (!physicalSdk(redecl->getLocation())) return false;
    }
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
      enclosingNvidiaFindDeviceFunction(expression, context);
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

// The two authenticated helper versions differ observably: SM0xa7 falls back
// with printf in v13.0, but is a silent table hit in v13.4.
static unsigned frozenNvidiaSMCoresVersion(clang::SourceManager &source,
                                         clang::SourceLocation location) {
  if (!location.isFileID() ||
      frozenNvidiaSampleHelperFileRole(source, location) !=
          FrozenOfficialNvidiaSampleHelperCuda)
    return 0;
  bool invalid = false;
  const auto bytes = source.getBufferData(source.getFileID(location), &invalid);
  if (invalid) return 0;
  if (bytes.size() == 28177 && sha256Equals(bytes,
      "997f9ac1f8e5f8e5f45f8b11eebab5b89305dee7430b90654bafe62283cffee1"))
    return 130;
  if (bytes.size() == 28219 && sha256Equals(bytes,
      "0494246768a4e3dbfcd688d79f4abad1fa39fc9f2d9dc3264a894f1c96d3785d"))
    return 134;
  return 0;
}

static bool frozenSMCoresDefinitionHasNoMacroTokens(
    const clang::FunctionDecl *definition, clang::Preprocessor &pp) {
  auto &source = pp.getSourceManager();
  const auto begin = definition->getBeginLoc(), end = definition->getEndLoc();
  if (!begin.isFileID() || !end.isFileID() ||
      source.getFileID(begin) != source.getFileID(end)) return false;
  const auto file = source.getFileID(begin);
  bool invalid = false;
  const auto bytes = source.getBufferData(file, &invalid);
  if (invalid) return false;
  const auto stop = clang::Lexer::getLocForEndOfToken(end, 0, source, pp.getLangOpts());
  if (!stop.isFileID() || source.getFileID(stop) != file) return false;
  const unsigned last = source.getFileOffset(stop);
  clang::Lexer lexer(source.getLocForStartOfFile(file), pp.getLangOpts(),
      bytes.begin(), bytes.begin() + source.getFileOffset(begin), bytes.end());
  clang::Token token;
  while (true) {
    lexer.LexFromRawLexer(token);
    if (token.is(clang::tok::eof) || source.getFileOffset(token.getLocation()) >= last) break;
    if (token.is(clang::tok::raw_identifier) &&
        pp.getMacroDefinitionAtLoc(pp.getIdentifierInfo(token.getRawIdentifier()),
                                   token.getLocation())) return false;
  }
  return true;
}

static bool isOrdinaryEvaluatedSMCoresCall(const clang::CallExpr *call,
                                          clang::ASTContext &context) {
  auto node = clang::DynTypedNode::create(*call);
  for (unsigned depth = 0; depth != 96; ++depth) {
    const auto parents = context.getParents(node);
    if (parents.size() != 1) return false;
    node = parents[0];
    if (const auto *function = node.get<clang::FunctionDecl>())
      return function->getDeclContext()->isFileContext() &&
          function->getTemplatedKind() == clang::FunctionDecl::TK_NonTemplate &&
          !function->hasAttr<clang::CUDADeviceAttr>() &&
          !function->hasAttr<clang::CUDAGlobalAttr>();
    if (const auto *variable = node.get<clang::VarDecl>()) {
      if (!variable->hasLocalStorage() || variable->isConstexpr()) return false;
      continue;
    }
    const auto *statement = node.get<clang::Stmt>();
    // Closed parent domain: sizeof/decltype/noexcept/traits, lambdas,
    // constructors and unproved future template bodies do not qualify.
    if (statement == nullptr || llvm::isa<clang::CUDAKernelCallExpr>(statement) ||
        !llvm::isa<clang::CallExpr, clang::ParenExpr, clang::ImplicitCastExpr,
                   clang::CastExpr, clang::BinaryOperator, clang::UnaryOperator,
                   clang::ConditionalOperator, clang::ReturnStmt,
                   clang::CompoundStmt, clang::DeclStmt, clang::IfStmt,
                   clang::ForStmt, clang::WhileStmt, clang::DoStmt,
                   clang::SwitchStmt, clang::CaseStmt, clang::DefaultStmt,
                   clang::LabelStmt>(statement)) return false;
  }
  return false;
}

static bool proveNvidiaSMCoresCall(
    const clang::CallExpr *call, clang::ASTContext &context,
    clang::Preprocessor &pp, const std::set<unsigned> &trusted,
    clang::SourceLocation &nameLocation, clang::FileID &file,
    unsigned &begin, unsigned &end, std::string &target) {
  if (call == nullptr || call->getNumArgs() != 2 ||
      !isOrdinaryEvaluatedSMCoresCall(call, context)) return false;
  auto &source = context.getSourceManager();
  const auto *callee = call->getDirectCallee();
  const clang::FunctionDecl *definition = nullptr;
  if (callee == nullptr || callee->getIdentifier() == nullptr ||
      callee->getName() != "_ConvertSMVer2Cores" ||
      !callee->getDeclContext()->isTranslationUnit() ||
      callee->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      !callee->hasBody(definition) || definition == nullptr ||
      !callee->isInlineSpecified() || callee->isVariadic() ||
      callee->getNumParams() != 2 ||
      !context.hasSameType(callee->getReturnType(), context.IntTy) ||
      !context.hasSameType(callee->getParamDecl(0)->getType(), context.IntTy) ||
      !context.hasSameType(callee->getParamDecl(1)->getType(), context.IntTy)) return false;
  const unsigned version = frozenNvidiaSMCoresVersion(source, definition->getLocation());
  if (!version || !allFunctionRedeclarationsComeFromFrozenHelperProfile(
          callee, source, FrozenOfficialNvidiaSampleHelperCuda, trusted)) return false;
  for (const auto *redecl : callee->getCanonicalDecl()->redecls())
    if (redecl->hasAttrs() ||
        frozenNvidiaSMCoresVersion(source, redecl->getLocation()) != version) return false;
  if (!frozenSMCoresDefinitionHasNoMacroTokens(definition, pp)) return false;
  FrozenNvidiaSampleHelperCallClosure closure(source, trusted,
                                              FrozenOfficialNvidiaSampleHelperCuda);
  if (!closure.prove(callee)) return false;
  class PrintProof : public clang::RecursiveASTVisitor<PrintProof> {
  public:
    PrintProof(clang::ASTContext &context, const std::set<unsigned> &trusted)
        : context(context), trusted(trusted) {}
    bool VisitCallExpr(clang::CallExpr *expression) {
      const auto *function = expression->getDirectCallee();
      if (function == nullptr || function->getIdentifier() == nullptr ||
          function->getName() != "printf" || function->getBuiltinID() == 0 ||
          !function->getDeclContext()->getRedeclContext()->isTranslationUnit() ||
          !function->isVariadic() || function->getNumParams() != 1 ||
          expression->getNumArgs() != 4 ||
          !context.hasSameType(function->getReturnType(), context.IntTy) ||
          !allFunctionRedeclarationsComeFromInitiallyTrustedSystemFiles(
              function, context.getSourceManager(), trusted)) { failed = true; return true; }
      // A first -isystem file is not proof of libc behaviour. Admit only the
      // ordinary bodyless external-C printf shape observed in the real SDK.
      // Source definitions, fortify wrappers, asm aliases and semantic
      // attributes (pure/const/noreturn/weak/etc.) must never be hidden by a
      // system-header marker. Runtime symbol/provider binding stays separate.
      for (const auto *redecl : function->getCanonicalDecl()->redecls()) {
        if (!redecl->isExternC() || !redecl->hasExternalFormalLinkage() ||
            redecl->hasBody()) { failed = true; return true; }
        for (const auto *attribute : redecl->attrs())
          if (!attribute->isImplicit() ||
              !llvm::isa<clang::BuiltinAttr, clang::FormatAttr>(attribute)) {
            failed = true; return true;
          }
      }
      const auto type = function->getParamDecl(0)->getType();
      if (!type->isPointerType() || !type->getPointeeType().isConstQualified() ||
          !context.hasSameType(type->getPointeeType().getUnqualifiedType(), context.CharTy))
        failed = true;
      ++calls; return true;
    }
    bool proved() const { return !failed && calls == 1; }
  private:
    clang::ASTContext &context;
    const std::set<unsigned> &trusted;
    unsigned calls = 0;
    bool failed = false;
  } print(context, trusted);
  print.TraverseStmt(const_cast<clang::Stmt *>(definition->getBody()));
  if (!print.proved()) return false;
  const auto *reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
      stripParenAndImplicitCasts(call->getCallee()));
  if (reference == nullptr || reference->hasQualifier() ||
      reference->getDecl()->getCanonicalDecl() != callee->getCanonicalDecl()) return false;
  nameLocation = reference->getLocation();
  if (!nameLocation.isFileID() || !source.isWrittenInMainFile(nameLocation)) return false;
  class MacroArgumentProof : public clang::RecursiveASTVisitor<MacroArgumentProof> {
  public:
    bool VisitExpr(clang::Expr *expression) {
      if (expression->getBeginLoc().isMacroID() || expression->getEndLoc().isMacroID()) failed = true;
      return true;
    }
    bool failed = false;
  } arguments;
  arguments.TraverseStmt(const_cast<clang::CallExpr *>(call));
  if (arguments.failed) return false;
  bool invalid = false;
  if (clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(nameLocation, nameLocation),
          source, pp.getLangOpts(), &invalid) != "_ConvertSMVer2Cores" || invalid) return false;
  const auto after = clang::Lexer::getLocForEndOfToken(nameLocation, 0, source, pp.getLangOpts());
  if (!after.isFileID() || source.getFileID(after) != source.getFileID(nameLocation)) return false;
  file = source.getFileID(nameLocation); begin = source.getFileOffset(nameLocation);
  end = source.getFileOffset(after);
  target = version == 130 ? "::ascify::sampleSMVersionToCoresV130" :
                            "::ascify::sampleSMVersionToCoresV134";
  return end > begin;
}

// NVIDIA sample-helper checkCudaCapabilities(major, minor) gate. The ported
// target keeps the original informational printfs and compares against the
// same reviewed epoch surface exposed by cudaGetDeviceProperties; it never
// fabricates an NVIDIA SM readout. Prove: TU-scope inline non-template bool
// definition from the frozen helper profile, two integer-literal arguments,
// and a plain file-spelled name token.
static llvm::StringRef nvidiaSampleHelperRewriteTarget(llvm::StringRef name) {
  return llvm::StringSwitch<llvm::StringRef>(name)
      .Case("checkCudaCapabilities", "::ascify::sampleCheckCudaCapabilities")
      .Case("findCudaDeviceDRV", "::ascify::sampleFindCudaDeviceDRV")
      .Case("findFatbinPath", "::ascify::sampleFindFatbinPath")
      .Case("gpuDeviceInit", "::ascify::sampleGpuDeviceInit")
      .Case("gpuGetMaxGflopsDeviceId", "::ascify::sampleGpuGetMaxGflopsDeviceId")
      .Default(llvm::StringRef());
}

// Shared proof for the fully-replaced NVIDIA sample-helper functions: the
// definition is TU-scope inline non-template from a frozen helper profile,
// the invocation is in the main file, and the name token is spelled plainly.
// The whole body is replaced by the compat implementation, so argument
// expression forms are unconstrained by this proof.
static bool proveNvidiaSampleHelperFunctionCall(
    const clang::CallExpr *call, clang::ASTContext &context,
    clang::Preprocessor &pp, const std::set<unsigned> &trusted,
    unsigned expectedArguments, clang::SourceLocation &nameLocation,
    clang::FileID &file, unsigned &begin, unsigned &end) {
  if (call == nullptr || call->getNumArgs() != expectedArguments) return false;
  auto &source = context.getSourceManager();
  const auto *callee = call->getDirectCallee();
  const clang::FunctionDecl *definition = nullptr;
  if (callee == nullptr || callee->getIdentifier() == nullptr ||
      nvidiaSampleHelperRewriteTarget(callee->getName()).empty() ||
      !callee->getDeclContext()->isTranslationUnit() ||
      callee->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      !callee->hasBody(definition) || definition == nullptr ||
      !callee->isInlineSpecified() || callee->isVariadic() ||
      callee->getNumParams() != static_cast<unsigned>(expectedArguments))
    return false;
  // checkCudaCapabilities lives in helper_cuda.h; the Driver helpers live in
  // helper_cuda_drvapi.h. Both are frozen profiles, and the function must
  // come entirely from whichever one defines it.
  if (!allFunctionRedeclarationsComeFromFrozenHelperProfile(
          callee, source, FrozenOfficialNvidiaSampleHelperCuda, trusted) &&
      !allFunctionRedeclarationsComeFromFrozenHelperProfile(
          callee, source, FrozenOfficialNvidiaSampleHelperDrvapi, trusted))
    return false;
  nameLocation = source.getFileLoc(call->getExprLoc());
  if (nameLocation.isInvalid() || nameLocation.isMacroID()) return false;
  bool invalid = false;
  if (clang::Lexer::getSourceText(
          clang::CharSourceRange::getTokenRange(nameLocation, nameLocation),
          source, pp.getLangOpts(), &invalid) != callee->getName() || invalid)
    return false;
  const auto after =
      clang::Lexer::getLocForEndOfToken(nameLocation, 0, source, pp.getLangOpts());
  if (!after.isFileID() || source.getFileID(after) != source.getFileID(nameLocation))
    return false;
  file = source.getFileID(nameLocation);
  begin = source.getFileOffset(nameLocation);
  end = source.getFileOffset(after);
  return end > begin;
}

// Ignored leading attributes need not belong to Decl::getSourceRange(). Find
// a conservative physical prefix from lexical declaration boundaries instead.
// A directive is skippable only before any pending declaration tokens; an
// interleaved directive rejects this proof rather than hiding an ATTR prefix
// or allowing an inactive branch's punctuation to manufacture a boundary.
static clang::SourceLocation physicalDeclarationPrefixBegin(
    clang::SourceLocation declaredBegin, clang::Preprocessor &preprocessor) {
  if (declaredBegin.isInvalid() || declaredBegin.isMacroID()) return {};
  auto &source = preprocessor.getSourceManager();
  const auto file = source.getFileID(declaredBegin);
  const auto first = source.getLocForStartOfFile(file);
  bool invalid = false;
  const auto contents = source.getBufferData(file, &invalid);
  if (invalid) return {};
  clang::Lexer lexer(first, preprocessor.getLangOpts(), contents.begin(),
                     contents.begin(), contents.end());
  auto boundary = first;
  bool directive = false, pending = false;
  unsigned parentheses = 0, brackets = 0;
  const auto target = source.getFileOffset(declaredBegin);
  clang::Token token;
  for (;;) {
    lexer.LexFromRawLexer(token);
    if (token.is(clang::tok::eof)) return {};
    if (directive && token.isAtStartOfLine()) {
      directive = false;
      boundary = token.getLocation();
    }
    if (source.getFileOffset(token.getLocation()) >= target)
      return !directive && parentheses == 0 && brackets == 0 ? boundary : clang::SourceLocation{};
    if (directive) continue;
    if (token.is(clang::tok::hash) && token.isAtStartOfLine()) {
      if (pending || parentheses != 0 || brackets != 0) return {};
      directive = true;
      continue;
    }
    if (parentheses == 0 && brackets == 0 &&
        (token.is(clang::tok::semi) || token.is(clang::tok::l_brace) ||
         token.is(clang::tok::r_brace))) {
      boundary = clang::Lexer::getLocForEndOfToken(
          token.getLocation(), 0, source, preprocessor.getLangOpts());
      if (boundary.isInvalid()) return {};
      pending = false;
      continue;
    }
    pending = true;
    if (token.is(clang::tok::l_paren)) ++parentheses;
    else if (token.is(clang::tok::r_paren)) {
      if (parentheses == 0) return {};
      --parentheses;
    } else if (token.is(clang::tok::l_square)) ++brackets;
    else if (token.is(clang::tok::r_square)) {
      if (brackets == 0) return {};
      --brackets;
    }
  }
}

// Prove the exact physical interval, including preprocessing syntax absent
// from the AST. The one optional qualifier token has its own SDK-chain proof.
static bool hasPhysicalMacroFreeRange(clang::SourceRange range,
                                     clang::Preprocessor &preprocessor,
                                     clang::SourceLocation allowedQualifier = {}) {
  if (preprocessor.getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
  auto &source = preprocessor.getSourceManager();
  const auto begin = range.getBegin(), last = range.getEnd();
  if (begin.isInvalid() || last.isInvalid() || begin.isMacroID() || last.isMacroID() ||
      source.getFileID(begin) != source.getFileID(last)) return false;
  const auto file = source.getFileID(begin);
  const auto *entry = source.getFileEntryForID(file);
  if (entry == nullptr || !physicalSourceFileMatches(
          source.getFilename(begin), *entry, source)) return false;
  bool invalid = false;
  const auto physical = llvm::MemoryBuffer::getFile(source.getFilename(begin));
  const auto contents = source.getBufferData(file, &invalid);
  if (!physical || invalid || contents != (*physical)->getBuffer()) return false;
  const auto text = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(range), source,
      preprocessor.getLangOpts(), &invalid);
  if (invalid || text.empty()) return false;
  const std::string owned = text.str();
  clang::Lexer lexer(begin, preprocessor.getLangOpts(), owned.data(),
                     owned.data(), owned.data() + owned.size());
  clang::Token token;
  bool previousLeftSquare = false;
  for (;;) {
    lexer.LexFromRawLexer(token);
    if (token.is(clang::tok::eof)) return true;
    if (token.needsCleaning() || token.is(clang::tok::hash) ||
        token.is(clang::tok::hashhash)) return false;
    if (previousLeftSquare && token.is(clang::tok::l_square)) return false;
    previousLeftSquare = token.is(clang::tok::l_square);
    const auto name = clang::Lexer::getSpelling(token, source, preprocessor.getLangOpts());
    if (name == "_Pragma" || name == "__pragma" || name == "__attribute__" ||
        name == "__attribute" || name == "__declspec" || name == "alignas") return false;
    if (!token.isAnyIdentifier()) continue;
    if (token.getLocation() == allowedQualifier && name == "__device__") continue;
    const auto *identifier = preprocessor.getIdentifierInfo(name);
    if (preprocessor.getMacroDefinitionAtLoc(identifier, token.getLocation()).getMacroInfo())
      return false;
  }
}

// A visible one-return host function forwards the Runtime result unchanged.
// Its declared enum type alone is not a proof, and this never follows another
// user wrapper or changes the API's arguments, return value or error handling.
static bool isAdmittedHostRuntimeStatusForwarder(
    const clang::Expr *expression, clang::ASTContext &context,
    clang::Preprocessor &preprocessor,
    const std::set<unsigned> &trustedSystemFileIds,
    const LocalHeaderRewriteContext *localHeaders) {
  const auto *call = llvm::dyn_cast_or_null<clang::CallExpr>(
      stripParenAndImplicitCasts(expression));
  const auto *function = call == nullptr ? nullptr : call->getDirectCallee();
  if (function == nullptr || function->getIdentifier() == nullptr ||
      !function->getDeclContext()->isFileContext() ||
      function->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      function->isVariadic() || function->hasAttr<clang::CUDADeviceAttr>() ||
      function->hasAttr<clang::CUDAGlobalAttr>()) return false;
  const auto *reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
      stripParenAndImplicitCasts(call->getCallee()));
  if (reference == nullptr || reference->hasQualifier() ||
      reference->hasExplicitTemplateArgs() || reference->getLocation().isMacroID() ||
      reference->getDecl()->getCanonicalDecl() != function->getCanonicalDecl()) return false;
  auto &source = context.getSourceManager();
  const auto result = function->getReturnType();
  const auto *enumeration = result->getAs<clang::EnumType>();
  if (enumeration == nullptr || enumeration->getDecl()->getName() != "cudaError" ||
      !enumeration->getDecl()->getDeclContext()->isTranslationUnit() ||
      !locationComesFromInitiallyTrustedSystemFile(
          source, enumeration->getDecl()->getLocation(), trustedSystemFileIds)) return false;
  const clang::FunctionDecl *definition = nullptr;
  if (!function->hasBody(definition) || definition == nullptr) return false;
  for (const auto *declaration : function->redecls()) {
    const auto prefix = physicalDeclarationPrefixBegin(
        declaration->getBeginLoc(), preprocessor);
    if (declaration->isImplicit() || declaration->getLocation().isInvalid() ||
        locationComesFromInitiallyTrustedSystemFile(
            source, declaration->getLocation(), trustedSystemFileIds) ||
        prefix.isInvalid() || !hasPhysicalMacroFreeRange(
            {prefix, declaration->getEndLoc()}, preprocessor)) return false;
    // Merely parsing an external header does not promise its translation.
    // A selected root header is published only with the closure transaction.
    if (!source.isWrittenInMainFile(declaration->getLocation()) &&
        (localHeaders == nullptr || localHeaders->plan == nullptr ||
         localHeaders->plan->failed() ||
         !localHeaders->plan->isSelectedRootHeader(
             source.getFilename(declaration->getLocation()).str()))) return false;
    for (const auto *attribute : declaration->attrs())
      if (!attribute->isImplicit() ||
          !llvm::isa<clang::CUDAHostAttr>(attribute)) return false;
    for (const auto *parameter : declaration->parameters())
      if (parameter->hasDefaultArg() || parameter->isParameterPack()) return false;
  }
  const auto *body = llvm::dyn_cast<clang::CompoundStmt>(definition->getBody());
  if (body == nullptr || body->size() != 1) return false;
  const auto *returned = llvm::dyn_cast<clang::ReturnStmt>(*body->body_begin());
  return returned != nullptr && returned->getRetValue() != nullptr &&
      context.hasSameType(result, returned->getRetValue()->getType()) &&
      isAdmittedCudaRuntimeStatusCall(source, returned->getRetValue(), trustedSystemFileIds);
}

// The original sortingNetworks host stores each Runtime result immediately
// before checking it. Prove that one definition-to-use edge; the variable's
// spelling or enum type alone does not establish an ACL return domain.
// A CUDA stream-callback body checks its own cudaError_t status parameter:
// `void CUDART_CB myStreamCallback(cudaStream_t, cudaError_t, void*)` with
// `checkCudaErrors(status)`. The value domain of that parameter is exactly
// what our rewritten cudaStreamAddCallback trampoline passes (a last-error
// snapshot), so admitting precisely the (stream, cudaError, data) callback
// shape's second parameter is an honest status-domain proof.
static bool isAdmittedStreamCallbackStatusParameter(
    const clang::CallExpr *check, clang::ASTContext &context,
    clang::SourceManager &sourceManager,
    const std::set<unsigned> &trustedSystemFileIds) {
  if (check == nullptr || check->getNumArgs() == 0)
    return false;
  const auto *reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
      stripParenAndImplicitCasts(check->getArg(0)));
  const auto *parameter = reference == nullptr ? nullptr :
      llvm::dyn_cast_or_null<clang::ParmVarDecl>(reference->getDecl());
  if (parameter == nullptr || parameter->hasAttrs() ||
      !sourceManager.isWrittenInMainFile(parameter->getLocation()))
    return false;
  const auto *enumeration = parameter->getType()->getAs<clang::EnumType>();
  if (enumeration == nullptr ||
      enumeration->getDecl()->getName() != "cudaError" ||
      !locationComesFromInitiallyTrustedSystemFile(
          sourceManager, enumeration->getDecl()->getLocation(),
          trustedSystemFileIds))
    return false;
  const clang::FunctionDecl *owner = llvm::dyn_cast_or_null<clang::FunctionDecl>(
      parameter->getDeclContext());
  if (owner == nullptr || !owner->hasBody() ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      owner->hasAttr<clang::CUDADeviceAttr>() ||
      owner->hasAttr<clang::CUDAGlobalAttr>() ||
      owner->getNumParams() != 3 ||
      parameter != owner->getParamDecl(1) ||
      !sourceManager.isWrittenInMainFile(owner->getLocation()))
    return false;
  // (cudaStream_t -> pointer, cudaError_t, void*) shape after rewrites.
  const clang::Type *streamParameter =
      owner->getParamDecl(0)->getType().getTypePtrOrNull();
  const clang::Type *dataParameter =
      owner->getParamDecl(2)->getType().getTypePtrOrNull();
  return streamParameter != nullptr && streamParameter->isPointerType() &&
         dataParameter != nullptr && dataParameter->isVoidPointerType();
}

static bool isAdmittedNvidiaLocalStatusUse(
    const clang::CallExpr *check, clang::ASTContext &context,
    clang::SourceManager &sourceManager,
    const std::set<unsigned> &trustedSystemFileIds,
    clang::Preprocessor &preprocessor,
    const LocalHeaderRewriteContext *localHeaders) {
  if (check == nullptr || check->getNumArgs() == 0)
    return false;
  const auto *reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
      stripParenAndImplicitCasts(check->getArg(0)));
  const auto *variable = reference == nullptr ? nullptr :
      llvm::dyn_cast<clang::VarDecl>(reference->getDecl());
  if (variable == nullptr || !variable->isLocalVarDecl() ||
      !variable->hasLocalStorage() || variable->hasAttrs() ||
      variable->getTLSKind() != clang::VarDecl::TLS_None ||
      variable->getType().hasQualifiers())
    return false;
  const auto *enumeration = variable->getType()->getAs<clang::EnumType>();
  if (enumeration == nullptr || enumeration->getDecl()->getName() != "cudaError" ||
      !locationComesFromInitiallyTrustedSystemFile(
          sourceManager, enumeration->getDecl()->getLocation(),
          trustedSystemFileIds))
    return false;
  const auto *owner = enclosingNvidiaFindDeviceFunction(check, context);
  if (owner == nullptr || !owner->hasBody() ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      owner->hasAttr<clang::CUDADeviceAttr>() ||
      owner->hasAttr<clang::CUDAGlobalAttr>() ||
      variable->getDeclContext()->getRedeclContext() !=
          owner->getRedeclContext() ||
      !sourceManager.isWrittenInMainFile(variable->getLocation()))
    return false;
  const auto parents = context.getParents(*check);
  if (parents.size() != 1)
    return false;
  const auto *block = parents[0].get<clang::CompoundStmt>();
  if (block == nullptr)
    return false;
  const clang::Stmt *previous = nullptr;
  bool found = false;
  for (const clang::Stmt *statement : block->body()) {
    if (statement == check) {
      found = true;
      break;
    }
    previous = statement;
  }
  const auto *assignment =
      llvm::dyn_cast_or_null<clang::BinaryOperator>(previous);
  if (!found || assignment == nullptr ||
      assignment->getOpcode() != clang::BO_Assign ||
      assignment->getBeginLoc().isMacroID() ||
      assignment->getEndLoc().isMacroID() ||
      assignment->getOperatorLoc().isMacroID() ||
      !sourceManager.isWrittenInMainFile(assignment->getBeginLoc()) ||
      !referencesCanonicalValue(assignment->getLHS(), variable) ||
      !context.hasSameType(variable->getType(), assignment->getRHS()->getType()) ||
      (!isAdmittedCudaRuntimeStatusCall(
          sourceManager, assignment->getRHS(), trustedSystemFileIds) &&
       !isAdmittedHostRuntimeStatusForwarder(
          assignment->getRHS(), context, preprocessor, trustedSystemFileIds, localHeaders)))
    return false;

  // Audit the entire owning body, including later statements: an address or
  // reference established on another iteration could otherwise bypass the
  // apparently adjacent assignment. Only scalar value reads and direct
  // assignments are permitted; captures and other lvalue uses fail closed.
  class Uses : public clang::RecursiveASTVisitor<Uses> {
  public:
    Uses(const clang::VarDecl *variable, clang::ASTContext &context)
        : variable(variable->getCanonicalDecl()), context(context) {}

    bool VisitLambdaExpr(clang::LambdaExpr *lambda) {
      for (const auto &capture : lambda->captures()) {
        if (capture.capturesVariable() &&
            capture.getCapturedVar()->getCanonicalDecl() == variable)
          return false;
      }
      return true;
    }

    bool VisitDeclRefExpr(clang::DeclRefExpr *use) {
      if (use->getDecl()->getCanonicalDecl() != variable)
        return true;
      const clang::Expr *current = use;
      while (true) {
        const auto parents = context.getParents(*current);
        if (parents.size() != 1)
          return false;
        if (const auto *parent = parents[0].get<clang::ParenExpr>()) {
          current = parent;
          continue;
        }
        if (const auto *cast = parents[0].get<clang::ImplicitCastExpr>())
          return cast->getCastKind() == clang::CK_LValueToRValue;
        if (const auto *binary = parents[0].get<clang::BinaryOperator>()) {
          if (binary->getOpcode() != clang::BO_Assign ||
              binary->getLHS() != current)
            return false;
          // Assignment itself yields an lvalue: accepting its LHS alone
          // would miss aliases such as `auto &alias = (error = success)`.
          // All admitted writes discard that result as a block statement.
          const auto assignmentParents = context.getParents(*binary);
          return assignmentParents.size() == 1 &&
                 assignmentParents[0].get<clang::CompoundStmt>() != nullptr;
        }
        return false;
      }
    }

  private:
    const clang::VarDecl *variable;
    clang::ASTContext &context;
  } uses(variable, context);
  return uses.TraverseStmt(owner->getBody());
}

// Prove a direct parameter root, independent of the enclosing function kind.
// The global-atomic caller separately requires a CUDA kernel, which establishes
// GM. The generic-add caller requires a pure device helper and preserves its
// actual address space. Local aliases, globals, fields, shared variables and
// conditional pointer selection fail this expression proof.
static bool isAddressRootedAtCurrentFunctionParameter(
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
    return isAddressRootedAtCurrentFunctionParameter(
        cast->getSubExpr(), kernel, context);
  }

  if (const auto *subscript =
          llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(expression))
    return isAddressRootedAtCurrentFunctionParameter(
        subscript->getBase(), kernel, context);

  if (const auto *unary =
          llvm::dyn_cast_or_null<clang::UnaryOperator>(expression)) {
    if (unary->getOpcode() != clang::UO_AddrOf &&
        unary->getOpcode() != clang::UO_Deref)
      return false;
    return isAddressRootedAtCurrentFunctionParameter(
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
           isAddressRootedAtCurrentFunctionParameter(
               binary->getLHS(), kernel, context);
  }
  if (lhsPointer == rhsPointer)
    return false;
  const clang::Expr *pointer =
      lhsPointer ? binary->getLHS() : binary->getRHS();
  const clang::Expr *offset =
      lhsPointer ? binary->getRHS() : binary->getLHS();
  return offset->getType()->isIntegerType() &&
         isAddressRootedAtCurrentFunctionParameter(
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
            std::vector<NvidiaSampleFindDeviceCandidate> &findCandidates,
            std::vector<NvidiaSampleSMCoresCandidate> &smCandidates,
            std::vector<NvidiaSampleSMCoresCandidate> &checkCapabilitiesCandidates,
            std::vector<NvidiaSampleTemplateGuard> &templateGuards,
            clang::Preprocessor &preprocessor,
            const LocalHeaderRewriteContext *localHeaders)
        : sourceManager(sourceManager), astContext(astContext),
          languageOptions(languageOptions),
          trustedSystemFileIds(trustedSystemFileIds),
          supportedRanges(supportedRanges),
          candidates(candidates), findCandidates(findCandidates), smCandidates(smCandidates),
          checkCapabilitiesCandidates(checkCapabilitiesCandidates),
          templateGuards(templateGuards), preprocessor(preprocessor), localHeaders(localHeaders) {}

    bool VisitCallExpr(clang::CallExpr *expression) {
      if (expression == nullptr)
        return true;
      const clang::FunctionDecl *callee = expression->getDirectCallee();
      if (callee != nullptr && callee->getIdentifier() != nullptr &&
          callee->getName() == "_ConvertSMVer2Cores" &&
          sourceManager.isWrittenInMainFile(
              sourceManager.getExpansionLoc(expression->getExprLoc()))) {
        clang::SourceLocation location;
        clang::FileID file;
        unsigned begin = 0, end = 0;
        std::string target;
        if (proveNvidiaSMCoresCall(expression, astContext, preprocessor,
                                  trustedSystemFileIds, location, file,
                                  begin, end, target))
          smCandidates.push_back({location, {file, begin, end}, target});
        return true;
      }
      if (callee != nullptr && callee->getIdentifier() != nullptr &&
          !nvidiaSampleHelperRewriteTarget(callee->getName()).empty() &&
          sourceManager.isWrittenInMainFile(
              sourceManager.getExpansionLoc(expression->getExprLoc()))) {
        clang::SourceLocation location;
        clang::FileID file;
        unsigned begin = 0, end = 0;
        if (proveNvidiaSampleHelperFunctionCall(
                expression, astContext, preprocessor, trustedSystemFileIds,
                callee->getNumParams(), location, file, begin, end))
          checkCapabilitiesCandidates.push_back(
              {location, {file, begin, end},
               nvidiaSampleHelperRewriteTarget(callee->getName()).str()});
        return true;
      }
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
      if (!isFrozenCheckCallee(expression))
        return true;
      NvidiaSampleHelperMacroCandidate *candidate =
          candidateAt(expression->getExprLoc(),
                      NvidiaSampleHelperMacroKind::CheckCudaErrors);
      if (candidate == nullptr || expression->getNumArgs() == 0)
        return true;
      // A type-dependent observation (an uninstantiated template body whose
      // argument list depends on template parameters) has no direct callee
      // and cannot decide the domain. Never let it overwrite the proof of a
      // concrete instantiation of the same candidate range; an uninstantiated
      // candidate simply keeps its unproven default and is refused.
      if (!expression->isTypeDependent())
        candidate->statusDomainProven = isAdmittedCudaRuntimeStatusCall(
              sourceManager, expression->getArg(0), trustedSystemFileIds) ||
            isAdmittedNvidiaLocalStatusUse(
              expression, astContext, sourceManager, trustedSystemFileIds, preprocessor, localHeaders) ||
            isAdmittedStreamCallbackStatusParameter(
              expression, astContext, sourceManager, trustedSystemFileIds);
      // A dependent check/cudaMemcpy/cudaFree lookup cannot be proved by its
      // spelling. Remember the pattern and audit its instantiated ASTs below;
      // any accepted template receives an explicit output type-domain guard.
      if (const auto *owner =
              enclosingNonLambdaFunction(expression, astContext)) {
        if (const auto *functionTemplate = owner->getDescribedFunctionTemplate()) {
          templatePatterns[functionTemplate][candidate] = expression;
        }
      }
      return true;
    }

    void proveGuardedTemplateStatusDomains() {
      for (const auto &entry : templatePatterns) {
        bool needsGuard = false;
        for (const auto &pattern : entry.second)
          needsGuard |= !pattern.first->statusDomainProven;
        if (!needsGuard)
          continue;
        // Template-independent status domain: every recorded pattern's check
        // argument is a call whose callee is directly resolved and admitted
        // by the runtime return-domain allowlist. The domain then does not
        // depend on the template parameter, so the int/float/double type
        // guard is unnecessary; each instantiated copy is still verified
        // below, mirroring the guarded path's InstantiationVisitor.
        bool independent = true;
        for (const auto &pattern : entry.second) {
          const clang::Expr *argument =
              pattern.second->getArg(0)->IgnoreParenImpCasts();
          const auto *call = llvm::dyn_cast<clang::CallExpr>(argument);
          if (call == nullptr || call->getDirectCallee() == nullptr ||
              !isAdmittedCudaRuntimeStatusCall(
                  sourceManager, argument, trustedSystemFileIds)) {
            independent = false;
            break;
          }
        }
        if (independent) {
          class IndependentInstantiationVisitor
              : public clang::RecursiveASTVisitor<
                    IndependentInstantiationVisitor> {
          public:
            IndependentInstantiationVisitor(
                const Visitor &owner, const TemplatePatterns &patterns)
                : owner(owner), patterns(patterns) {}
            bool VisitCallExpr(clang::CallExpr *call) {
              for (const auto &pattern : patterns) {
                if (call->getExprLoc() != pattern.second->getExprLoc())
                  continue;
                ++counts[pattern.first];
                if (!owner.isFrozenCheckCallee(call) ||
                    call->getNumArgs() == 0 ||
                    !isAdmittedCudaRuntimeStatusCall(
                        owner.sourceManager, call->getArg(0),
                        owner.trustedSystemFileIds))
                  failed = true;
              }
              return true;
            }
            bool proved() const {
              if (failed)
                return false;
              for (const auto &pattern : patterns) {
                const auto count = counts.find(pattern.first);
                if (count == counts.end() || count->second == 0)
                  return false;
              }
              return true;
            }
          private:
            const Visitor &owner;
            const TemplatePatterns &patterns;
            std::map<NvidiaSampleHelperMacroCandidate *, unsigned> counts;
            bool failed = false;
          } verifier(*this, entry.second);
          for (const clang::FunctionDecl *specialization :
               entry.first->specializations()) {
            const clang::TemplateSpecializationKind kind =
                specialization->getTemplateSpecializationKind();
            if ((kind != clang::TSK_ImplicitInstantiation &&
                 kind != clang::TSK_ExplicitInstantiationDefinition) ||
                !specialization->hasBody())
              continue;
            verifier.TraverseStmt(
                const_cast<clang::Stmt *>(specialization->getBody()));
            if (verifier.proved())
              break;
          }
          if (verifier.proved()) {
            for (const auto &pattern : entry.second)
              pattern.first->statusDomainProven = true;
            llvm::errs()
                << "Ascify NVIDIA sample-helper closure: proved template-"
                   "independent status domain for '"
                << entry.first->getQualifiedNameAsString()
                << "' via admitted runtime callees\n";
            continue;
          }
        }
        NvidiaSampleTemplateGuard guard;
        if (!proveTemplate(entry.first, entry.second, guard)) {
          llvm::errs() << "Ascify NVIDIA sample-helper closure: dependent "
                          "template status domain not proven for '"
                       << entry.first->getQualifiedNameAsString()
                       << "'; all helper edits kept\n";
          continue;
        }
        for (const auto &pattern : entry.second)
          pattern.first->statusDomainProven = true;
        llvm::errs() << "Ascify NVIDIA sample-helper closure: proved dependent "
                        "template status domain for '"
                     << entry.first->getQualifiedNameAsString()
                     << "'; guard pending helper transaction: "
                     << llvm::StringRef(guard.text).trim() << "\n";
        templateGuards.push_back(std::move(guard));
      }
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
    bool hasUnprovedSMCoresUse() const { return unprovedSMCoresUse; }

  private:
    using TemplatePatterns =
        std::map<NvidiaSampleHelperMacroCandidate *, const clang::CallExpr *>;

    bool isFrozenCheckCallee(const clang::CallExpr *call) const {
      const auto frozenCheck = [&](const clang::NamedDecl *declaration) {
        if (const auto *functionTemplate =
                llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(declaration))
          declaration = functionTemplate->getTemplatedDecl();
        const auto *function =
            llvm::dyn_cast_or_null<clang::FunctionDecl>(declaration);
        return function != nullptr && function->getIdentifier() != nullptr &&
               (function->getName() == "check" ||
                // helper_cuda_drvapi.h spells the same contract __checkCudaErrors.
                function->getName() == "__checkCudaErrors") &&
               locationComesFromRecognizedNvidiaSampleHelper(
                   sourceManager, function->getLocation());
      };
      if (call->getDirectCallee() != nullptr)
        return frozenCheck(call->getDirectCallee());
      const auto *lookup = llvm::dyn_cast<clang::UnresolvedLookupExpr>(
          call->getCallee()->IgnoreParenImpCasts());
      if (lookup == nullptr || lookup->decls_begin() == lookup->decls_end())
        return false;
      for (const clang::NamedDecl *declaration : lookup->decls()) {
        if (!frozenCheck(declaration))
          return false;
      }
      return true;
    }

    bool proveTemplate(const clang::FunctionTemplateDecl *functionTemplate,
                       const TemplatePatterns &patterns,
                       NvidiaSampleTemplateGuard &guard) const {
      const clang::FunctionDecl *function = functionTemplate->getTemplatedDecl();
      const clang::TemplateParameterList *parameters =
          functionTemplate->getTemplateParameters();
      if (function == nullptr || parameters == nullptr || parameters->size() != 1 ||
          !function->getDeclContext()->isTranslationUnit() ||
          function->hasAttr<clang::CUDADeviceAttr>() ||
          function->hasAttr<clang::CUDAGlobalAttr>())
        return false;
      const auto *parameter =
          llvm::dyn_cast<clang::TemplateTypeParmDecl>(parameters->getParam(0));
      const auto *body =
          llvm::dyn_cast_or_null<clang::CompoundStmt>(function->getBody());
      if (parameter == nullptr || parameter->isParameterPack() ||
          parameter->getIdentifier() == nullptr || parameter->getDepth() != 0 ||
          body == nullptr || body->getLBracLoc().isInvalid() ||
          body->getLBracLoc().isMacroID() || body->getRBracLoc().isMacroID() ||
          parameter->getLocation().isMacroID() ||
          !sourceManager.isWrittenInMainFile(body->getLBracLoc()) ||
          !sourceManager.isWrittenInMainFile(body->getRBracLoc()))
        return false;
      const std::string parameterName = parameter->getNameAsString();
      // The emitted trait and parameter tokens must retain their meaning at
      // the exact insertion point, even if macros are undefined later.
      for (const std::string &token :
           {std::string("static_assert"), std::string("__is_same"),
            parameterName, std::string("int"), std::string("float"),
            std::string("double")}) {
        if (preprocessor.getMacroDefinitionAtLoc(
                preprocessor.getIdentifierInfo(token), body->getLBracLoc()))
          return false;
      }
      bool invalidToken = false;
      const llvm::StringRef parameterToken = clang::Lexer::getSourceText(
          clang::CharSourceRange::getTokenRange(
              parameter->getLocation(), parameter->getLocation()),
          sourceManager, languageOptions, &invalidToken);
      if (invalidToken || parameterToken != parameterName)
        return false;

      std::set<std::string> types;
      for (const clang::FunctionDecl *specialization :
           functionTemplate->specializations()) {
        const clang::TemplateSpecializationKind kind =
            specialization->getTemplateSpecializationKind();
        const clang::TemplateArgumentList *arguments =
            specialization->getTemplateSpecializationArgs();
        if ((kind != clang::TSK_ImplicitInstantiation &&
             kind != clang::TSK_ExplicitInstantiationDefinition) ||
            !specialization->hasBody() || arguments == nullptr ||
            arguments->size() != 1 ||
            arguments->get(0).getKind() != clang::TemplateArgument::Type)
          return false;
        const clang::QualType type = arguments->get(0).getAsType().getCanonicalType();
        if (type.isNull() || type.getLocalCVRQualifiers() != 0)
          return false;
        if (astContext.hasSameType(type, astContext.IntTy))
          types.insert("int");
        else if (astContext.hasSameType(type, astContext.FloatTy))
          types.insert("float");
        else if (astContext.hasSameType(type, astContext.DoubleTy))
          types.insert("double");
        else
          return false;

        class InstantiationVisitor
            : public clang::RecursiveASTVisitor<InstantiationVisitor> {
        public:
          InstantiationVisitor(const Visitor &owner,
                               const TemplatePatterns &patterns)
              : owner(owner), patterns(patterns) {}
          bool VisitCallExpr(clang::CallExpr *call) {
            for (const auto &pattern : patterns) {
              if (call->getExprLoc() != pattern.second->getExprLoc())
                continue;
              ++counts[pattern.first];
              const clang::FunctionDecl *callee = call->getDirectCallee();
              if (callee == nullptr || !owner.isFrozenCheckCallee(call) ||
                  !allFunctionRedeclarationsComeFromFrozenHelperProfile(
                      callee, owner.sourceManager,
                      FrozenOfficialNvidiaSampleHelperCuda,
                      owner.trustedSystemFileIds) ||
                  call->getNumArgs() != 4 ||
                  !isAdmittedCudaRuntimeStatusCall(
                      owner.sourceManager, call->getArg(0),
                      owner.trustedSystemFileIds))
                failed = true;
            }
            return true;
          }
          bool proved() const {
            if (failed)
              return false;
            for (const auto &pattern : patterns) {
              const auto count = counts.find(pattern.first);
              if (count == counts.end() || count->second != 1)
                return false;
            }
            return true;
          }
        private:
          const Visitor &owner;
          const TemplatePatterns &patterns;
          std::map<NvidiaSampleHelperMacroCandidate *, unsigned> counts;
          bool failed = false;
        } instantiated(*this, patterns);
        instantiated.TraverseStmt(
            const_cast<clang::Stmt *>(specialization->getBody()));
        if (!instantiated.proved())
          return false;
      }
      if (types.empty())
        return false;

      guard.insertionLocation = clang::Lexer::getLocForEndOfToken(
          body->getLBracLoc(), 0, sourceManager, languageOptions);
      if (guard.insertionLocation.isInvalid())
        return false;
      const auto anchor = sourceManager.getDecomposedLoc(body->getLBracLoc());
      guard.anchorRange = {anchor.first, anchor.second, anchor.second + 1};
      guard.text = "\nstatic_assert(";
      for (const std::string &type : types) {
        if (guard.text != "\nstatic_assert(")
          guard.text += " || ";
        guard.text += "__is_same(" + parameterName + ", " + type + ")";
      }
      guard.text += ", \"Ascify: dependent helper status is only proven for "
                    "the emitted template type domain\");\n";
      return true;
    }

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
      for (const auto &candidate : smCandidates) {
        const auto &range = candidate.rewriteRange;
        if (range.file == decomposed.first && decomposed.second >= range.beginOffset &&
            decomposed.second < range.endOffset) return true;
      }
      for (const auto &candidate : checkCapabilitiesCandidates) {
        const auto &range = candidate.rewriteRange;
        if (range.file == decomposed.first && decomposed.second >= range.beginOffset &&
            decomposed.second < range.endOffset) return true;
      }
      return false;
    }

    bool inspect(clang::SourceLocation useLocation,
                 const clang::NamedDecl *referenced) {
      // This visitor also collects status-domain proofs. A residual helper
      // prevents the complete helper transaction, but must not stop traversal
      // before later, independent Runtime checks can be classified. Preserve
      // the first residual for its diagnostic and keep walking the AST.
      if (referenced == nullptr)
        return true;
      const clang::SourceLocation expansion =
          sourceManager.getExpansionLoc(useLocation);
      if (referenced->getIdentifier() != nullptr &&
          referenced->getName() == "_ConvertSMVer2Cores" &&
          expansion.isValid() && sourceManager.isWrittenInMainFile(expansion) &&
          !isSupportedHelperRewrite(useLocation)) {
        unprovedSMCoresUse = true;
        if (!conflict) { conflict = true; name = referenced->getQualifiedNameAsString(); }
        return true;
      }
      if (conflict) return true;
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
      return true;
    }

    clang::SourceManager &sourceManager;
    clang::ASTContext &astContext;
    const clang::LangOptions &languageOptions;
    const std::set<unsigned> &trustedSystemFileIds;
    const std::vector<SemanticRewriteRange> &supportedRanges;
    std::vector<NvidiaSampleHelperMacroCandidate> &candidates;
    std::vector<NvidiaSampleFindDeviceCandidate> &findCandidates;
    std::vector<NvidiaSampleSMCoresCandidate> &smCandidates;
    std::vector<NvidiaSampleSMCoresCandidate> &checkCapabilitiesCandidates;
    std::vector<NvidiaSampleTemplateGuard> &templateGuards;
    clang::Preprocessor &preprocessor;
    const LocalHeaderRewriteContext *localHeaders;
    std::map<const clang::FunctionTemplateDecl *, TemplatePatterns> templatePatterns;
    bool conflict = false;
    bool unprovedSMCoresUse = false;
    std::string name;
  } visitor(getCompilerInstance().getSourceManager(),
            getCompilerInstance().getASTContext(),
            getCompilerInstance().getLangOpts(),
            trustedSystemFileIds,
            nvidiaSampleHelperMacroRanges,
            nvidiaSampleHelperMacroCandidates,
            nvidiaSampleFindDeviceCandidates,
            nvidiaSampleSMCoresCandidates,
            nvidiaSampleCheckCapabilitiesCandidates,
            nvidiaSampleTemplateGuards,
            getCompilerInstance().getPreprocessor(), localHeaderContext);

  visitor.TraverseDecl(
      getCompilerInstance().getASTContext().getTranslationUnitDecl());
  visitor.proveGuardedTemplateStatusDomains();
  for (const NvidiaSampleFindDeviceCandidate &candidate :
       nvidiaSampleFindDeviceCandidates) {
    nvidiaSampleFindDeviceNormalOffsets.insert(
        getCompilerInstance().getSourceManager().getFileOffset(
            getCompilerInstance().getSourceManager().getFileLoc(
                candidate.nameLocation)));
  }
  for (const auto &candidate : nvidiaSampleSMCoresCandidates)
    nvidiaSampleSMCoresNormalOffsets.insert(
        getCompilerInstance().getSourceManager().getFileOffset(candidate.nameLocation));
  for (const auto &candidate : nvidiaSampleCheckCapabilitiesCandidates)
    nvidiaSampleCheckCapabilitiesNormalOffsets.insert(
        getCompilerInstance().getSourceManager().getFileOffset(candidate.nameLocation));
  if (visitor.hasUnprovedSMCoresUse()) {
    nvidiaSampleSMCoresUnsupportedUse = true;
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "CUDA sample-helper SM table boundary: unproved function use");
    getCompilerInstance().getDiagnostics().Report(diagnostic);
    cudaRuntimeApiBoundaryFailed = true;
  }
  const bool helperNeedsCompat =
      !nvidiaSampleHelperMacroCandidates.empty() ||
      !nvidiaSampleFindDeviceCandidates.empty() ||
      !nvidiaSampleSMCoresCandidates.empty() ||
      !nvidiaSampleCheckCapabilitiesCandidates.empty();
  if (helperNeedsCompat && ascifyCudaCompatActiveMacroConflict) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    // A proved SM callee is still conditional on the complete helper
    // transaction. Do not publish a partially rewritten root when a compat
    // macro collision prevents that transaction. Unrelated residual helpers
    // keep their existing retain-only policy when this conflict is absent.
    if (!nvidiaSampleSMCoresCandidates.empty()) {
      nvidiaSampleSMCoresUnsupportedUse = true;
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "CUDA sample-helper SM table boundary: input macro '%0' prevents the helper transaction");
      getCompilerInstance().getDiagnostics().Report(diagnostic)
          << ascifyCudaCompatActiveMacroConflictName;
      cudaRuntimeApiBoundaryFailed = true;
    }
  }
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
          << "domain not proven; macro and include kept";
      const auto &sourceManager = getCompilerInstance().getSourceManager();
      const auto location = sourceManager.getPresumedLoc(candidate.nameLocation);
      if (location.isValid())
        llvm::errs() << " at " << location.getFilename() << ":"
                     << location.getLine() << ":" << location.getColumn();
      llvm::errs() << "\n";
      // A root-only publication that keeps an unproved checkCudaErrors
      // invocation still depends on the original helper status machinery, so
      // it is a partial status success. Refuse the whole conversion instead
      // of publishing a retained-helper output; getLastCudaError candidates
      // are proven by construction at record time.
      const auto diagnostic =
          getCompilerInstance().getDiagnostics().getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "CUDA sample-helper boundary: checkCudaErrors status domain "
              "not proven; conversion refused");
      getCompilerInstance().getDiagnostics().Report(candidate.nameLocation,
                                                    diagnostic);
      cudaRuntimeApiBoundaryFailed = true;
    }
  }
  return visitor.hasConflict() || compatNamespaceConflict;
}

void AscifyAction::proveNvidiaSampleOpenMPParallelPragmas() {
  if (nvidiaSampleOpenMPParallelPragmaLines.empty())
    return;
  auto &context = getCompilerInstance().getASTContext();
  auto &source = getCompilerInstance().getSourceManager();
  auto pending = nvidiaSampleOpenMPParallelPragmaLines;
  class Visitor : public clang::RecursiveASTVisitor<Visitor> {
  public:
    Visitor(clang::ASTContext &context, clang::SourceManager &source,
            std::set<unsigned> &pending)
        : context(context), source(source), pending(pending) {}
    bool VisitOMPParallelDirective(clang::OMPParallelDirective *directive) {
      const auto begin = directive->getBeginLoc();
      const auto end = directive->getEndLoc();
      if (directive->getNumClauses() != 0 || begin.isInvalid() ||
          begin.isMacroID() || end.isInvalid() || end.isMacroID() ||
          !source.isWrittenInMainFile(begin) ||
          !source.isWrittenInMainFile(end))
        return true;
      const auto *owner = enclosingNvidiaFindDeviceFunction(directive, context);
      if (owner == nullptr || !owner->hasBody() ||
          owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
          owner->isDependentContext() ||
          owner->hasAttr<clang::CUDADeviceAttr>() ||
          owner->hasAttr<clang::CUDAGlobalAttr>())
        return true;
      bool invalid = false;
      const auto contents = source.getBufferData(source.getMainFileID(), &invalid);
      if (!invalid)
        pending.erase(physicalLineBegin(contents, source.getFileOffset(begin)));
      return true;
    }
    clang::ASTContext &context;
    clang::SourceManager &source;
    std::set<unsigned> &pending;
  } visitor(context, source, pending);
  visitor.TraverseDecl(context.getTranslationUnitDecl());
  if (!pending.empty()) {
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs() << "Ascify NVIDIA sample-helper closure: unproved OpenMP "
                    "parallel owner keeps all helper edits\n";
  }
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

  const auto &boundSource = getCompilerInstance().getSourceManager();
  if (boundSource.isWrittenInMainFile(tok.getLocation())) {
    const auto offset = boundSource.getFileOffset(tok.getLocation());
    const auto policy = kernelThreadBoundPlan.policyQualifiers.find(offset);
    const auto occupancy = kernelThreadBoundPlan.occupancyCallees.find(offset);
    std::string target;
    if (name == "__global__" && policy != kernelThreadBoundPlan.policyQualifiers.end())
      target = "ASCIFY_GLOBAL __launch_bounds__(" + std::to_string(policy->second) + ")";
    else if (name == "cudaOccupancyMaxActiveBlocksPerMultiprocessor" ||
             name == "cudaOccupancyMaxPotentialBlockSize") {
      if (occupancy == kernelThreadBoundPlan.occupancyCallees.end()) {
        const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error, "occupancy kernel-bound boundary: raw API has no proven kernel bound");
        getCompilerInstance().getDiagnostics().Report(tok.getLocation(), diagnostic);
        cudaRuntimeApiBoundaryFailed = true;
        return false;
      }
      target = occupancy->second;
    }
    if (!target.empty()) {
      const auto mapped = CUDA_RENAMES_MAP().find(name);
      if (mapped != CUDA_RENAMES_MAP().end())
        Statistics::current().incrementCounter(mapped->second, name.str());
      const auto end = clang::Lexer::getLocForEndOfToken(tok.getLocation(), 0, boundSource,
                                                         getCompilerInstance().getLangOpts());
      const ct::Replacement edit(boundSource, tok.getLocation(), tok.getLength(), target);
      if (!insertSemanticReplacement(edit, clang::FullSourceLoc(tok.getLocation(), boundSource),
                                     tok.getLocation(), end)) {
        cudaRuntimeApiBoundaryFailed = true;
      } else {
        // Like the FuncAttributes proof, only this successfully rewritten,
        // bound-proven callee token may nest in a certified status macro.
        // Its arguments and the enclosing expression gain no exemption.
        if (occupancy != kernelThreadBoundPlan.occupancyCallees.end())
          provenKernelAttributeCalleeRanges.push_back(
              {boundSource.getFileID(tok.getLocation()), offset, boundSource.getFileOffset(end)});
        const auto required = kernelThreadBoundHeaderPolicies.find(offset);
        if (required != kernelThreadBoundHeaderPolicies.end()) {
          const auto &proof = required->second;
          if (!localHeaderContext || localHeaderContext->depth != 1 ||
              !localHeaderContext->appliedKernelThreadBoundPolicies.emplace(
                  LocalKernelThreadBoundPolicyKey{proof.sourcePath, proof.qualifierOffset}, proof).second)
            cudaRuntimeApiBoundaryFailed = true;
        }
      }
      needsCudaCompatHeader = true;
      return false;
    }
  }

  // Do not turn std::exp (or another explicitly-qualified function) into an
  // invalid std::expf spelling. Unqualified CUDA device math remains mapped.
  if (isUnqualifiedFloatMathName(name)) {
    const clang::Token *previous = previousSignificantToken(rawTokenWindow);
    if (previous != nullptr && previous->is(clang::tok::coloncolon))
      return false;
    const auto &source = getCompilerInstance().getSourceManager();
    if (source.isWrittenInMainFile(tok.getLocation()) &&
        preservedHostMathCalleeOffsets.count(
            source.getFileOffset(tok.getLocation())))
      return false;
  }

  if (ascify::IsCudaTextureName(name)) {
    const auto &source = getCompilerInstance().getSourceManager();
    if (!source.isWrittenInMainFile(tok.getLocation()) ||
        !cudaTextureLiteralOffsets.count(source.getFileOffset(tok.getLocation()))) {
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "CUDA texture boundary: raw texture name '%0' has no literal SDK proof");
      getCompilerInstance().getDiagnostics().Report(tok.getLocation(),diagnostic) << name;
      cudaRuntimeApiBoundaryFailed = true;
      return false;
    }
    needsCudaTextureHeader = true;
  }
  if (requiresCudaCompatHeader(name))
    needsCudaCompatHeader = true;

  FindAndReplace(name, tok.getLocation(), CUDA_RENAMES_MAP());
  return false;
}

void AscifyAction::recordContextPragmaSource(clang::FileID file) {
  if (localHeaderContext == nullptr || localHeaderContext->inputEvidence == nullptr)
    return;
  auto &source = getCompilerInstance().getSourceManager();
  if (file.isInvalid() || source.getFileEntryForID(file) == nullptr)
    return;
  bool invalid = false;
  const auto bytes = source.getBufferData(file, &invalid);
  if (invalid) return;
  auto &evidence = *localHeaderContext->inputEvidence;
  const auto start = source.getLocForStartOfFile(file);
  const auto path = contextualCanonicalPath(source.getFilename(start));
  bool safe = true, afterHash = false;
  evidence.literalContextPragmaFiles.erase(path);
  evidence.contextOnceLineOffsets.erase(path);
  const auto ownedPackLines = packedRecords.ownedPragmaLines();
  unsigned hashOffset = 0;
  std::vector<unsigned> onceLines;
  clang::Lexer raw(start, getCompilerInstance().getLangOpts(), bytes.begin(),
                   bytes.begin(), bytes.end());
  clang::Token token;
  for (;;) {
    raw.LexFromRawLexer(token);
    if (token.is(clang::tok::eof)) break;
    const auto word = clang::Lexer::getSpelling(
        token, source, getCompilerInstance().getLangOpts());
    if (token.isAnyIdentifier() && (word == "_Pragma" || word == "__pragma"))
      safe = false;
    if (afterHash && (word == "line" || token.is(clang::tok::numeric_constant)))
      evidence.hasPhysicalLineDirective = true;
    if (afterHash && word == "pragma") {
      const auto begin = physicalLineBegin(bytes, hashOffset);
      const auto end = bytes.find_first_of("\r\n", begin);
      const auto line = bytes.slice(begin, end);
      const auto tokens = directPragmaTokens(line);
      const bool once = hasExactDirectiveTokens(line, {"#", "pragma", "once"});
      const bool unroll = hasExactDirectiveTokens(line, {"#", "pragma", "unroll"}) ||
          (tokens.size() == 4 && tokens[0].spelling == "#" &&
           tokens[1].spelling == "pragma" && tokens[2].spelling == "unroll" &&
           !tokens[3].spelling.empty() && tokens[3].spelling.size() <= 4 &&
           std::all_of(tokens[3].spelling.begin(), tokens[3].spelling.end(),
                       [](char c) { return c >= '0' && c <= '9'; }));
      const bool ownedPack = std::find(ownedPackLines.begin(), ownedPackLines.end(),
          std::make_pair(file, begin)) != ownedPackLines.end();
      if (token.needsCleaning() || line.contains('\\') || line.contains("??") ||
          line.contains("/*") || line.contains("*/") || (!once && !unroll && !ownedPack))
        safe = false;
      if (once) onceLines.push_back(begin);
    }
    afterHash = token.is(clang::tok::hash) && token.isAtStartOfLine();
    if (afterHash) hashOffset = source.getFileOffset(token.getLocation());
  }
  if (safe && onceLines.size() <= 1 && !path.empty()) {
    evidence.literalContextPragmaFiles.insert(path);
    evidence.contextOnceLineOffsets[path] = std::move(onceLines);
  }
}

void AscifyAction::FileChanged(
    clang::SourceLocation location,
    clang::PPCallbacks::FileChangeReason reason,
    clang::SrcMgr::CharacteristicKind fileType) {
  if (reason == clang::PPCallbacks::EnterFile || reason == clang::PPCallbacks::ExitFile)
    packedRecords.includeOrFileTransition();
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
  enteredSourceFileIds.insert(file);
  recordContextPragmaSource(file);
  if (frontendCompatibility.enabled() &&
      verifiedForcedFrontendHostMathFile(
          file, getCompilerInstance(), frontendCompatibility)) {
    auto &compiler = getCompilerInstance();
    auto &preprocessor = compiler.getPreprocessor();
    if (compiler.getLangOpts().CUDA &&
        !compiler.getPreprocessorOpts().RetainExcludedConditionalBlocks &&
        trustedFrontendCgCudaControl("__CUDA__", preprocessor.getMacroInfo(
            preprocessor.getIdentifierInfo("__CUDA__")), compiler) &&
        trustedFrontendCgCudaControl("__CUDACC__", preprocessor.getMacroInfo(
            preprocessor.getIdentifierInfo("__CUDACC__")), compiler))
      verifiedFrontendHostMathEntryFileIds.insert(file.getHashValue());
  }
  if (frontendCompatibility.enabled() &&
      verifiedFrontendCgFile(file, getCompilerInstance(), frontendCompatibility)) {
    auto &compiler = getCompilerInstance();
    auto &preprocessor = compiler.getPreprocessor();
    if (compiler.getLangOpts().CUDA &&
        !compiler.getPreprocessorOpts().RetainExcludedConditionalBlocks &&
        trustedFrontendCgCudaControl("__CUDA__", preprocessor.getMacroInfo(
            preprocessor.getIdentifierInfo("__CUDA__")), compiler) &&
        trustedFrontendCgCudaControl("__CUDACC__", preprocessor.getMacroInfo(
            preprocessor.getIdentifierInfo("__CUDACC__")), compiler))
      verifiedFrontendCgEntryFileIds.insert(file.getHashValue());
  }
  if (localHeaderContext != nullptr &&
      localHeaderContext->inputEvidence != nullptr) {
    auto &evidence = *localHeaderContext->inputEvidence;
    const auto path = contextualRecordPathBinding(evidence, sourceManager.getFilename(spelling));
    if (!path.empty()) {
      ++evidence.fileEntries[path];
      // The root may be an intentional virtual input. Inventory its real
      // source separately; the virtual input is covered by the token digest.
      if (sourceManager.isWrittenInMainFile(spelling)) {
        if (!evidence.parsedRootSourcePath.empty() &&
            evidence.parsedRootSourcePath != path)
          evidence.changedPathBinding = path;
        evidence.parsedRootSourcePath = path;
        bool invalid = false;
        const auto parsed = sourceManager.getBufferData(file, &invalid);
        evidence.parsedRootSha256 = invalid ? std::string() :
            contextualInputSha256(parsed);
        if (auto physical = llvm::MemoryBuffer::getFile(path))
          evidence.fileSha256[path] = contextualInputSha256((*physical)->getBuffer());
      } else {
        bool invalid = false;
        const auto contents = sourceManager.getBufferData(file, &invalid);
        if (!invalid) {
          evidence.fileSha256[path] = contextualInputSha256(contents);
          // Inlining changes include depth. A pragma hidden in an inactive
          // depth-dependent branch must not become active without appearing
          // in the token/macro digests. Reject raw pragma introducers too.
          if (localHeaderContext->plan != nullptr &&
              localHeaderContext->plan->isSelectedRootHeader(path)) {
            clang::Lexer raw(sourceManager.getLocForStartOfFile(file),
                getCompilerInstance().getLangOpts(), contents.begin(),
                contents.begin(), contents.end());
            clang::Token token;
            bool afterHash = false;
            for (;;) {
              raw.LexFromRawLexer(token);
              if (token.is(clang::tok::eof)) break;
              const auto word = clang::Lexer::getSpelling(
                  token, sourceManager, getCompilerInstance().getLangOpts());
              if (token.isAnyIdentifier() &&
                  (word == "_Pragma" || word == "__pragma" ||
                   (afterHash && word == "pragma")))
                evidence.pragmaFiles.insert(path);
              afterHash = token.is(clang::tok::hash);
            }
          }
        }
      }
    }
  }
  const auto *sdkEntry = sourceManager.getFileEntryForID(file);
  const auto include = sourceManager.getSpellingLoc(sourceManager.getIncludeLoc(file));
  const auto parentFile = include.isValid() ? sourceManager.getFileID(include) : clang::FileID();
  unsigned hostRuntimeIncludeLine = 0;
  if (include.isFileID() && sourceManager.isWrittenInMainFile(include)) {
    bool invalid = false;
    const auto parentBytes = sourceManager.getBufferData(parentFile, &invalid);
    if (!invalid)
      hostRuntimeIncludeLine = sourceManager.getLocForStartOfFile(parentFile)
          .getLocWithOffset(physicalLineBegin(parentBytes,
              sourceManager.getFileOffset(include))).getRawEncoding();
  }
  const bool directHostRuntimeRoot =
      !getCompilerInstance().getLangOpts().CUDA && hostRuntimeIncludeLine != 0 &&
      hostCudaRuntimeLiteralRootLines.count(hostRuntimeIncludeLine) != 0 &&
      isConfiguredCudaRuntimeHeaderPath(sourceManager.getFilename(spelling), sourceManager);
  const bool sdkRole = !sourceManager.isWrittenInMainFile(spelling) &&
      isConfiguredCudaSdkDependencyRole(sourceManager.getFilename(spelling)) &&
      (directHostRuntimeRoot ||
       isConfiguredCudaRuntimeApiPath(sourceManager.getFilename(spelling), sourceManager) ||
       (cudaProfilerApiHeaderAdmitted &&
        llvm::sys::path::filename(sourceManager.getFilename(spelling)) ==
            "cuda_profiler_api.h" && include.isValid() &&
        sourceManager.isWrittenInMainFile(include)) ||
       (parentFile.isValid() &&
        (cudaRuntimeApiSdkFileIds.count(parentFile.getHashValue()) ||
         isOwnedCudaCompilerInput(parentFile, getCompilerInstance()))));
  if (sdkRole) {
    if (sdkEntry != nullptr && physicalSourceFileMatches(
            sourceManager.getFilename(spelling), *sdkEntry, sourceManager) &&
        physicalCudaSdkNvRoleMatches(sourceManager.getFilename(spelling)))
      cudaRuntimeApiSdkFileIds.insert(file.getHashValue());
    else
      cudaRuntimeApiSdkIdentityFailure = spelling;
  }
  for (auto &candidate : profilerAvailabilityCandidates) {
    if (candidate.includeObserved && include.isFileID() &&
        parentFile == candidate.file &&
        candidate.lookupEntry != nullptr && sdkEntry != nullptr &&
        candidate.lookupEntry->getUniqueID() == sdkEntry->getUniqueID() &&
        isPhysicalCudaProfilerHeader(sourceManager.getFilename(spelling), sdkEntry) &&
        cudaRuntimeApiSdkFileIds.count(file.getHashValue())) {
      const auto parentBytes = sourceManager.getBufferData(parentFile);
      if (physicalLineBegin(parentBytes, sourceManager.getFileOffset(include)) ==
          physicalLineBegin(parentBytes, candidate.includeBegin))
        candidate.sdkEntered = true;
    }
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
  observeDirectImageEntry(file, !wasTrusted && !wasUntrusted);
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

namespace {
struct ProfilerPhysicalToken {
  std::string text;
  unsigned begin = 0;
  unsigned end = 0;
  bool identifier = false;
};

// Read actual physical tokens, including inactive source. Never reconstruct a
// directive by joining macro expansions or by trusting its evaluated value.
static bool profilerPhysicalTokens(clang::CompilerInstance &compiler,
                                  clang::FileID file, unsigned begin,
                                  unsigned end,
                                  std::vector<ProfilerPhysicalToken> &tokens,
                                  bool rejectLineSplices = true) {
  auto &source = compiler.getSourceManager();
  bool invalid = false;
  const auto bytes = source.getBufferData(file, &invalid);
  if (invalid || begin > end || end > bytes.size() ||
      (rejectLineSplices && (bytes.slice(begin, end).contains('\\') ||
                            bytes.slice(begin, end).contains("??")))) return false;
  clang::Lexer lexer(source.getLocForStartOfFile(file), compiler.getLangOpts(),
                     bytes.data(), bytes.data(), bytes.data() + bytes.size());
  clang::Token token;
  while (true) {
    lexer.LexFromRawLexer(token);
    if (token.is(clang::tok::eof)) break;
    const unsigned offset = source.getFileOffset(token.getLocation());
    if (offset >= end) break;
    if (offset < begin || token.is(clang::tok::comment)) continue;
    const auto after = clang::Lexer::getLocForEndOfToken(
        token.getLocation(), 0, source, compiler.getLangOpts());
    if (after.isInvalid() || source.getFileOffset(after) > end) return false;
    tokens.push_back({clang::Lexer::getSpelling(token, source,
                                              compiler.getLangOpts()),
                      offset, source.getFileOffset(after), token.isAnyIdentifier()});
  }
  return true;
}

static unsigned profilerPhysicalLineEnd(llvm::StringRef bytes, unsigned begin) {
  const auto end = bytes.find_first_of("\r\n", begin);
  return end == llvm::StringRef::npos ? bytes.size() : end;
}

static unsigned profilerNextPhysicalLine(llvm::StringRef bytes, unsigned end) {
  if (end < bytes.size() && bytes[end] == '\r') ++end;
  if (end < bytes.size() && bytes[end] == '\n') ++end;
  return end;
}

static bool profilerLiteralOperand(
    llvm::StringRef bytes, const std::vector<ProfilerPhysicalToken> &tokens,
    size_t &cursor, unsigned &begin, unsigned &end) {
  static constexpr llvm::StringLiteral literal("<cuda_profiler_api.h>");
  if (cursor >= tokens.size()) return false;
  begin = tokens[cursor].begin;
  end = begin + literal.size();
  if (bytes.substr(begin, literal.size()) != literal) return false;
  while (cursor < tokens.size() && tokens[cursor].begin < end) {
    if (tokens[cursor].end > end) return false;
    ++cursor;
  }
  return cursor != 0 && tokens[cursor - 1].end == end;
}
} // namespace

bool AscifyAction::isPhysicalCudaProfilerHeader(
    llvm::StringRef path, const clang::FileEntry *entry) {
#if LLVM_VERSION_MAJOR >= 13
  if (CudaPath.empty() || path.empty()) return false;
  auto &source = getCompilerInstance().getSourceManager();
  llvm::SmallString<256> expected(CudaPath.getValue()), physicalExpected,
      physicalActual;
  llvm::sys::path::append(expected, "include", "cuda_profiler_api.h");
  const auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer || (*buffer)->getBufferSize() != 4566 ||
      !sha256Equals((*buffer)->getBuffer(),
          "bb8e03c217df1309ccfe3249c4fac1bcd577bdcff5c65944217548195204118e") ||
      llvm::sys::fs::real_path(expected, physicalExpected) ||
      llvm::sys::fs::real_path(path, physicalActual) ||
      physicalExpected != physicalActual) return false;
  if (entry != nullptr) {
    llvm::sys::fs::UniqueID identity;
    if (source.isFileOverridden(entry) || entry->getSize() != 4566 ||
        llvm::sys::fs::getUniqueID(path, identity) ||
        identity != entry->getUniqueID()) return false;
  }
  return true;
#else
  return false;
#endif
}

void AscifyAction::HasInclude(
    clang::SourceLocation location, llvm::StringRef filename, bool angled,
    const clang::FileEntry *entry, llvm::StringRef resolvedPath) {
  if (frontendCompatibility.enabled() &&
      (llvm::sys::path::filename(filename) == "cooperative_groups.h" ||
       filename.starts_with("cooperative_groups/"))) originalCgInquiry = true;
  if (filename != "cuda_profiler_api.h") return;
  auto &source = getCompilerInstance().getSourceManager();
  if (!source.isWrittenInMainFile(source.getExpansionLoc(location))) return;
  const bool physical = location.isFileID() && angled && entry != nullptr &&
      isPhysicalCudaProfilerHeader(resolvedPath, entry);
  profilerHasIncludeObservations.push_back(
      {location, resolvedPath.str(), entry, physical});
}

bool AscifyAction::isWitnessedProfilerHostRoot(clang::FileID file) {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  const auto &pp = compiler.getPreprocessorOpts();
  const auto &lang = compiler.getLangOpts();
  if (hostRootInput == nullptr || lang.CUDA || file.isInvalid() ||
      file != source.getMainFileID() || lang.Modules || lang.CPlusPlusModules ||
      !compiler.getHeaderSearchOpts().VFSOverlayFiles.empty() ||
      !pp.RemappedFiles.empty() || !pp.RemappedFileBuffers.empty() ||
      !pp.ImplicitPCHInclude.empty() || !pp.PCHThroughHeader.empty() ||
      pp.PrecompiledPreambleBytes.first != 0 || pp.PrecompiledPreambleBytes.second ||
      !compiler.getFrontendOpts().ModuleFiles.empty() ||
      compiler.getASTContext().getExternalSource() != nullptr)
    return false;
  if (localHeaderContext != nullptr &&
      (localHeaderContext->depth != 0 || localHeaderContext->virtualInput != nullptr ||
       localHeaderContext->plan == nullptr ||
       localHeaderContext->sourcePath != localHeaderContext->plan->rootSourcePath() ||
       localHeaderContext->sourcePath != hostRootInput->path()))
    return false;
  const auto *entry = source.getFileEntryForID(file);
  const auto origin = source.getLocForStartOfFile(file);
#if LLVM_VERSION_MAJOR >= 15
  const auto entryRef = source.getFileEntryRefForID(file);
  if (!entryRef || &entryRef->getFileEntry() != entry ||
      entryRef->getName() != hostRootInput->path()) return false;
#else
  if (entry == nullptr || entry->getName() != hostRootInput->path()) return false;
#endif
  bool invalid = false;
  const auto bytes = source.getBufferData(file, &invalid);
  // The VFS main entry was created by main.cpp, so its UID need not equal the
  // real file's UID. Bind that entry's name/size and exact parsed bytes to our
  // explicit witness; the witness still rechecks the physical UID and SHA.
  return entry != nullptr && !source.isFileOverridden(entry) && !invalid &&
      hostRootInput->matchesParsed(source.getFilename(origin), bytes,
                                   entry->getSize());
}

void AscifyAction::If(clang::SourceLocation location,
                     clang::SourceRange condition,
                     clang::PPCallbacks::ConditionValueKind value) {
  recordDirectImageUtilityControl("", "if", location, nullptr);
  const unsigned entryDepth = preprocessorConditionalDepth;
  ConditionalDirectiveEntered();
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  if (!location.isFileID() || !source.isWrittenInMainFile(location)) return;
  const auto file = source.getFileID(location);
  bool invalid = false;
  const auto bytes = source.getBufferData(file, &invalid);
  if (invalid) return;
  const unsigned begin = physicalLineBegin(bytes, source.getFileOffset(location));
  const unsigned end = profilerPhysicalLineEnd(bytes, begin);
  const auto line = bytes.slice(begin, end);
  if (!line.contains("cuda_profiler_api.h") ||
      !line.contains("__has_include")) return;
  std::vector<ProfilerPhysicalToken> tokens;
  if (!profilerPhysicalTokens(compiler, file, begin, end, tokens, false)) {
    profilerAvailabilityProofFailed = true; return;
  }
  bool probeToken = false, headerToken = false;
  for (const auto &token : tokens) {
    probeToken |= token.text == "__has_include" || token.text == "__has_include_next";
    headerToken |= token.text == "cuda_profiler_api" ||
        token.text.find("cuda_profiler_api.h") != std::string::npos;
  }
  // Mentions in comments do not make an unrelated #if a profiler rule.
  if (!probeToken || !headerToken) return;
  auto fail = [&]() { profilerAvailabilityProofFailed = true; };
  const auto &ppOptions = compiler.getPreprocessorOpts();
  const auto &lang = compiler.getLangOpts();
  const auto *mainEntry = source.getFileEntryForID(file);
  if (line.contains('\\') || line.contains("??") ||
      !profilerAvailabilityCandidates.empty() || entryDepth != 0 ||
      TargetPolicy != "dav-c310-vec" || TargetRecipe != "none" ||
      ppOptions.RetainExcludedConditionalBlocks || lang.Modules ||
      lang.CPlusPlusModules || !ppOptions.ImplicitPCHInclude.empty() ||
      !ppOptions.PCHThroughHeader.empty() ||
      ppOptions.PrecompiledPreambleBytes.first != 0 ||
      ppOptions.PrecompiledPreambleBytes.second ||
      !compiler.getFrontendOpts().ModuleFiles.empty() ||
      compiler.getASTContext().getExternalSource() != nullptr ||
      mainEntry == nullptr || source.isFileOverridden(mainEntry) ||
      (!physicalSourceFileMatches(source.getFilename(location), *mainEntry, source) &&
       !isWitnessedProfilerHostRoot(file)) ||
      (localHeaderContext != nullptr &&
       (localHeaderContext->depth != 0 || localHeaderContext->virtualInput != nullptr))) {
    fail(); return;
  }
  if (tokens.size() < 5 || tokens[0].text != "#" || tokens[1].text != "if") {
    fail(); return;
  }
  ProfilerAvailabilityCandidate candidate;
  candidate.file = file;
  candidate.ifLocation = location;
  size_t cursor = 2;
  if (tokens[cursor].text == "defined") {
    if (tokens.size() < cursor + 6 || tokens[cursor + 1].text != "(" ||
        tokens[cursor + 2].text != "__has_include" ||
        tokens[cursor + 3].text != ")" || tokens[cursor + 4].text != "&&") {
      fail(); return;
    }
    candidate.builtinOffsets.insert(tokens[cursor + 2].begin);
    cursor += 5;
  }
  if (cursor + 2 >= tokens.size() || tokens[cursor].text != "__has_include" ||
      tokens[cursor + 1].text != "(") { fail(); return; }
  const unsigned lookupOffset = tokens[cursor].begin;
  candidate.builtinOffsets.insert(lookupOffset);
  cursor += 2;
  if (condition.getBegin().isInvalid() || condition.getEnd().isInvalid()) {
    fail(); return;
  }
  // A genuine builtin expands to a scratch numeric token. Bind its complete
  // expansion range back to the already-proved physical directive; requiring
  // the evaluated numeric token itself to be FileID would reject every probe.
  const auto physicalCondition = source.getExpansionRange(condition);
  const auto conditionBegin = physicalCondition.getBegin();
  const auto conditionEnd = physicalCondition.isTokenRange()
      ? clang::Lexer::getLocForEndOfToken(physicalCondition.getEnd(), 0,
                                        source, compiler.getLangOpts())
      : physicalCondition.getEnd();
  if (!profilerLiteralOperand(bytes, tokens, cursor, candidate.operandBegin,
                              candidate.operandEnd) ||
      cursor + 1 != tokens.size() || tokens[cursor].text != ")" ||
      !conditionBegin.isFileID() || !conditionEnd.isFileID() ||
      source.getFileID(conditionBegin) != file ||
      source.getFileID(conditionEnd) != file ||
      source.getFileOffset(conditionBegin) != tokens[2].begin ||
      source.getFileOffset(conditionEnd) != tokens.back().end ||
      value != clang::PPCallbacks::CVK_True) { fail(); return; }
  const auto *builtin = compiler.getPreprocessor().getMacroInfo(
      compiler.getPreprocessor().getIdentifierInfo("__has_include"));
  if (builtin == nullptr || !builtin->isBuiltinMacro()) { fail(); return; }
  unsigned lookups = 0;
  for (const auto &observation : profilerHasIncludeObservations) {
    if (!observation.location.isFileID() ||
        source.getFileID(observation.location) != file ||
        // LLVM23 reports FilenameLoc, not the __has_include identifier.
        source.getFileOffset(observation.location) != candidate.operandBegin) continue;
    ++lookups;
    candidate.lookupProved = observation.physicalSdk;
    candidate.sdkPath = observation.resolvedPath;
    candidate.lookupEntry = observation.entry;
  }
  if (lookups != 1 || !candidate.lookupProved) { fail(); return; }
  // Parse the entire bounded physical block now, before its contents are
  // processed. Callbacks must subsequently confirm the exact include/endif.
  unsigned position = profilerNextPhysicalLine(bytes, end);
  unsigned nonempty = 0;
  bool closed = false;
  while (position < bytes.size()) {
    const unsigned lineEnd = profilerPhysicalLineEnd(bytes, position);
    std::vector<ProfilerPhysicalToken> body;
    if (!profilerPhysicalTokens(compiler, file, position, lineEnd, body)) {
      fail(); return;
    }
    position = profilerNextPhysicalLine(bytes, lineEnd);
    if (body.empty()) continue;
    if (body.size() == 2 && body[0].text == "#" && body[1].text == "endif") {
      if (nonempty < 1) { fail(); return; }
      candidate.endifBegin = body[0].begin;
      candidate.resumeOffset = position;
      if (position <= lineEnd || position >= bytes.size()) { fail(); return; }
      closed = true; break;
    }
    if (nonempty == 0 && body.size() >= 3 && body[0].text == "#" &&
        body[1].text == "include") {
      size_t operand = 2;
      unsigned first = 0, last = 0;
      if (!profilerLiteralOperand(bytes, body, operand, first, last) ||
          operand != body.size()) { fail(); return; }
      candidate.includeBegin = body[0].begin;
    } else if (nonempty == 1 && body.size() == 4 && body[0].text == "#" &&
               body[1].text == "define" && body[2].identifier &&
               body[3].text == "1") {
      const llvm::StringRef name(body[2].text);
      if (name.starts_with("__") || name.starts_with("ASCIFY_") ||
          name.starts_with("ACL_") || name.starts_with("MSVP_") ||
          name.starts_with("CUDA_") || name.starts_with("CUDART_") ||
          profilerMacroDefinitions[name.str()] != 0 ||
          profilerMacroUndefinitions[name.str()] != 0 ||
          compiler.getPreprocessor().getMacroInfo(
              compiler.getPreprocessor().getIdentifierInfo(name)) != nullptr) {
        fail(); return;
      }
      candidate.flag = name.str();
      candidate.flagBegin = body[2].begin;
    } else { fail(); return; }
    ++nonempty;
  }
  if (!closed) { fail(); return; }
  profilerAvailabilityCandidates.push_back(std::move(candidate));
}

void AscifyAction::Else(clang::SourceLocation location,
                       clang::SourceLocation ifLocation) {
  recordDirectImageUtilityControl("", "else/elif", location, nullptr);
  for (const auto &candidate : profilerAvailabilityCandidates)
    if (candidate.ifLocation == ifLocation) profilerAvailabilityProofFailed = true;
}

void AscifyAction::Endif(clang::SourceLocation location,
                        clang::SourceLocation ifLocation) {
  auto &source = getCompilerInstance().getSourceManager();
  for (auto &candidate : profilerAvailabilityCandidates) {
    if (candidate.ifLocation != ifLocation) continue;
    candidate.endifObserved = location.isFileID() &&
        source.getFileID(location) == candidate.file &&
        physicalLineBegin(source.getBufferData(candidate.file),
                          source.getFileOffset(location)) <= candidate.endifBegin &&
        source.getFileOffset(location) >= candidate.endifBegin;
    candidate.endifLocation = location;
  }
  ConditionalDirectiveEnded();
}

bool AscifyAction::isProvedProfilerAvailabilityToken(
    clang::SourceLocation location) const {
  if (!location.isFileID()) return false;
  const auto &source = getCompilerInstance().getSourceManager();
  for (const auto &candidate : profilerAvailabilityCandidates)
    if (candidate.sealed && source.getFileID(location) == candidate.file &&
        candidate.builtinOffsets.count(source.getFileOffset(location))) return true;
  return false;
}

bool AscifyAction::sealProfilerAvailabilityProof() {
  if (profilerAvailabilityCandidates.empty() && !profilerAvailabilityProofFailed)
    return true;
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto reject = [&]() {
    const auto diagnostic = compiler.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "CUDA profiler header boundary: cannot prove positive physical profiler availability block");
    compiler.getDiagnostics().Report(diagnostic);
    return false;
  };
  if (profilerAvailabilityProofFailed || profilerAvailabilityCandidates.size() != 1 ||
      profilerAvailabilityControlModified || profilerAvailabilityPositionObserved)
    return reject();
  auto &candidate = profilerAvailabilityCandidates.front();
  if (!candidate.lookupProved || !candidate.includeObserved ||
      candidate.lookupEntry == nullptr ||
      !candidate.includeMapped || !candidate.sdkEntered || !candidate.endifObserved ||
      !isPhysicalCudaProfilerHeader(candidate.sdkPath) ||
      (!candidate.flag.empty() &&
       (profilerMacroDefinitions[candidate.flag] != 1 ||
        profilerMacroUndefinitions[candidate.flag] != 0))) return reject();
  const auto *builtin = compiler.getPreprocessor().getMacroInfo(
      compiler.getPreprocessor().getIdentifierInfo("__has_include"));
  if (builtin == nullptr || !builtin->isBuiltinMacro()) return reject();
  if (!candidate.flag.empty()) {
    const auto *flag = compiler.getPreprocessor().getMacroInfo(
        compiler.getPreprocessor().getIdentifierInfo(candidate.flag));
    if (flag == nullptr || flag->isFunctionLike() || flag->getNumTokens() != 1 ||
        !flag->getDefinitionLoc().isFileID() ||
        source.getFileID(flag->getDefinitionLoc()) != candidate.file ||
        source.getFileOffset(flag->getDefinitionLoc()) != candidate.flagBegin ||
        compiler.getPreprocessor().getSpelling(*flag->tokens_begin()) != "1")
      return reject();
  }
  bool invalid = false;
  const auto bytes = source.getBufferData(candidate.file, &invalid);
  if (invalid) return reject();
  std::vector<ProfilerPhysicalToken> all;
  if (!profilerPhysicalTokens(compiler, candidate.file, 0, bytes.size(), all, false))
    return reject();
  for (size_t index = 0; index < all.size(); ++index) {
    const auto &token = all[index];
    // Do not reinterpret a caller #line or GNU line marker as a proved
    // position. The resume below comes only from this real SourceManager.
    if (token.text == "#" && index + 1 < all.size() &&
        (all[index + 1].text == "line" ||
         (!all[index + 1].text.empty() &&
          all[index + 1].text[0] >= '0' && all[index + 1].text[0] <= '9')))
      return reject();
    if (!candidate.flag.empty() && token.text == candidate.flag &&
        token.begin != candidate.flagBegin) {
      if (token.begin < source.getFileOffset(candidate.ifLocation)) return reject();
      const unsigned lineBegin = physicalLineBegin(bytes, token.begin);
      std::vector<ProfilerPhysicalToken> use;
      if (!profilerPhysicalTokens(compiler, candidate.file, lineBegin,
              profilerPhysicalLineEnd(bytes, lineBegin), use)) return reject();
      const bool directConditional = use.size() == 3 && use[0].text == "#" &&
          (use[1].text == "ifdef" || use[1].text == "ifndef") &&
          use[2].text == candidate.flag;
      const bool definedConditional = use.size() == 6 && use[0].text == "#" &&
          use[1].text == "if" && use[2].text == "defined" &&
          use[3].text == "(" && use[4].text == candidate.flag && use[5].text == ")";
      if (!directConditional && !definedConditional) return reject();
    }
    // Keep the explicit position-macro boundary conservative. The same
    // transaction also restores the original presumed line for implicit
    // __builtin_LINE consumers; frozen diagnostics keep their old contract.
    if (token.text == "__LINE__") return reject();
  }
  const auto origin = source.getLocForStartOfFile(candidate.file);
  const auto *entry = source.getFileEntryForID(candidate.file);
  if (entry == nullptr || source.isFileOverridden(entry) ||
      (!physicalSourceFileMatches(source.getFilename(origin), *entry, source) &&
       !isWitnessedProfilerHostRoot(candidate.file)))
    return reject();
  const auto operand = origin.getLocWithOffset(candidate.operandBegin);
  const auto endif = origin.getLocWithOffset(candidate.endifBegin);
  const auto resume = origin.getLocWithOffset(candidate.resumeOffset);
  const auto presumed = source.getPresumedLoc(resume);
  if (presumed.isInvalid() || presumed.getLine() == 0 ||
      presumed.getLine() > 2147483647u) return reject();
  ct::Replacements staged(*replacements);
  const ct::Replacement probe(source, operand,
      candidate.operandEnd - candidate.operandBegin, "<acl/acl_prof.h>");
  const ct::Replacement missing(source, endif, 0,
      "#else\n#error \"Ascify admitted profiler branch requires the target ACL profiler header\"\n");
  const ct::Replacement position(source, resume, 0,
      "#line " + std::to_string(presumed.getLine()) + "\n");
  for (const auto &range : SemanticRewriteRanges) {
    if (range.file != candidate.file) continue;
    if ((candidate.operandBegin < range.endOffset &&
         range.beginOffset < candidate.operandEnd) ||
        (range.beginOffset <= candidate.endifBegin &&
         candidate.endifBegin < range.endOffset) ||
        (range.beginOffset <= candidate.resumeOffset &&
         candidate.resumeOffset < range.endOffset)) return reject();
  }
  if (!llcompat::insertReplacement(staged, probe) ||
      !llcompat::insertReplacement(staged, missing) ||
      !llcompat::insertReplacement(staged, position)) return reject();
  *replacements = std::move(staged);
  SemanticRewriteRanges.push_back(
      {candidate.file, candidate.operandBegin, candidate.operandEnd});
  candidate.sealed = true;
  Statistics::current().cudaProfilerAvailabilityEvidence.push_back(
      "source=" + source.getFilename(origin).str() +
      ";source_sha256=" + contextualInputSha256(bytes) +
      ";if_offset=" + std::to_string(source.getFileOffset(candidate.ifLocation)) +
      ";include_offset=" + std::to_string(candidate.includeBegin) +
      ";sdk=" + candidate.sdkPath +
      ";lookup_device=" + std::to_string(candidate.lookupEntry->getUniqueID().getDevice()) +
      ";lookup_file=" + std::to_string(candidate.lookupEntry->getUniqueID().getFile()) +
      ";sdk_sha256=bb8e03c217df1309ccfe3249c4fac1bcd577bdcff5c65944217548195204118e" +
      ";source_flag=" + candidate.flag +
      ";resume_line=" + std::to_string(presumed.getLine()));
  return true;
}

// This is one parsed compiler definition, not authority borrowed from the
// unparsed original CG inputs. Caller mentions remain forbidden even at 1.
bool AscifyAction::proveOriginalCgCompilerDefault(const clang::MacroInfo *macro,
    clang::SourceLocation identifier, bool capture) {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  if (!frontendCompatibility.enabled() || macro == nullptr ||
      !identifier.isValid() || !identifier.isFileID() ||
      macro->getDefinitionLoc() != identifier ||
      !ordinaryRuntimeProfileInput(compiler) ||
      !frontendCgMacroTokens(*macro, compiler.getPreprocessor(), {"1"}))
    return false;
  const auto file = source.getFileID(identifier);
  const auto token = macro->tokens_begin();
  if (file.isInvalid() || !token->is(clang::tok::numeric_constant) ||
      !token->getLocation().isValid() || !token->getLocation().isFileID() ||
      source.getFileID(token->getLocation()) != file ||
      !isOwnedCudaCompilerInput(file, compiler))
    return false;
  llvm::SmallString<256> selectedRoot, ownedRoot, path;
  if (llvm::sys::fs::real_path(compiler.getHeaderSearchOpts().ResourceDir, selectedRoot) ||
      llvm::sys::fs::real_path(ownedClangResourceDirectory(), ownedRoot) ||
      selectedRoot != ownedRoot)
    return false;
  llvm::SmallString<256> expected(ownedRoot);
  llvm::sys::path::append(expected, "include", "__clang_cuda_runtime_wrapper.h");
  const auto *entry = source.getFileEntryForID(file);
  const auto filename = source.getFilename(identifier);
  bool invalid = false;
  const auto bytes = source.getBufferData(file, &invalid);
  const char *hash = "6caeb75ad8b888bc93141b700608cf3581f334fc5cc0971e96e3f699e86ceb45";
  const llvm::StringRef directive = "#define _CG_CLUSTER_INTRINSICS_AVAILABLE 1\n";
  const auto offset = bytes.find(directive);
  if (entry == nullptr || entry->getSize() != 18786 || invalid || bytes.size() != 18786 ||
      llvm::sys::fs::real_path(filename, path) || path != expected ||
      !physicalSourceFileMatches(filename, *entry, source) ||
      contextualInputSha256(bytes) != hash || offset == llvm::StringRef::npos ||
      bytes.find(directive, offset + 1) != llvm::StringRef::npos ||
      source.getFileOffset(identifier) != offset + llvm::StringRef("#define ").size() ||
      source.getFileOffset(token->getLocation()) != offset + directive.size() - 2)
    return false;
  OriginalCgInput input{path.str().str(), hash, entry->getUniqueID().getDevice(),
      entry->getUniqueID().getFile(), 18786};
  if (capture) {
    if (originalCgCompilerDefaultMacro != nullptr || originalCgControlChanged)
      return false;
    originalCgCompilerDefaultMacro = macro;
    originalCgCompilerDefaultDefinition = identifier;
    originalCgCompilerDefaultInput = input;
    return true;
  }
  return macro == originalCgCompilerDefaultMacro &&
      identifier == originalCgCompilerDefaultDefinition &&
      input.path == originalCgCompilerDefaultInput.path &&
      input.sha256 == originalCgCompilerDefaultInput.sha256 &&
      input.device == originalCgCompilerDefaultInput.device &&
      input.file == originalCgCompilerDefaultInput.file &&
      input.size == originalCgCompilerDefaultInput.size;
}

bool AscifyAction::proveOriginalCgInclude(clang::SourceLocation location,
    const clang::Token &includeToken, llvm::StringRef name, bool angled,
    clang::CharSourceRange filenameRange, llvm::StringRef searchPath,
    llvm::StringRef resolvedPath,
    std::string &error) {
  if (!frontendCompatibility.enabled()) return true;
  const auto reject = [&](const char *reason) { error = reason; return false; };
  llvm::SmallString<256> actual;
  if (llvm::sys::fs::real_path(resolvedPath, actual)) {
    if (name == "cooperative_groups.h" || name == "cooperative_groups/reduce.h")
      return reject("selected CG provider cannot be resolved");
    return true;
  }
  const std::string projectedTop = frontendCompatibility.canonicalRoot + "/cooperative_groups.h";
  const std::string projectedReduce = frontendCompatibility.canonicalRoot + "/cooperative_groups/reduce.h";
  const bool actualTop = actual == projectedTop, actualReduce = actual == projectedReduce;
  if (!actualTop && !actualReduce) return true;
  if ((actualTop && name != "cooperative_groups.h") ||
      (actualReduce && name != "cooperative_groups/reduce.h"))
    return reject("an alias to a product CG parser header has no admitted original provider");
#if LLVM_VERSION_MAJOR != 23
  return reject("original CG provider proof requires the reviewed LLVM23 interface");
#else
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto &pp = compiler.getPreprocessor();
  auto &headers = pp.getHeaderSearchInfo();
  const auto &options = compiler.getPreprocessorOpts();
  if (!location.isFileID() || !filenameRange.getBegin().isFileID() ||
      !filenameRange.getEnd().isFileID() || includeToken.getIdentifierInfo() == nullptr ||
      includeToken.getIdentifierInfo()->getPPKeywordID() != clang::tok::pp_include ||
      compiler.getLangOpts().Modules || compiler.getLangOpts().CPlusPlusModules ||
      !compiler.getHeaderSearchOpts().VFSOverlayFiles.empty() ||
      !options.RemappedFiles.empty() || !options.RemappedFileBuffers.empty() ||
      !options.ImplicitPCHInclude.empty() || !options.PCHThroughHeader.empty() ||
      options.PrecompiledPreambleBytes.first || options.PrecompiledPreambleBytes.second ||
      !compiler.getFrontendOpts().ModuleFiles.empty() ||
      compiler.getASTContext().getExternalSource() || headers.getExternalLookup() ||
      headers.HasIncludeAliasMap())
    return reject("original CG provider has an unsupported directive, remap, module or PCH context");
  llvm::SmallString<256> profile, selectedSearch, sdk;
  if (llvm::sys::fs::real_path(frontendCompatibility.canonicalRoot, profile) ||
      llvm::sys::fs::real_path(searchPath, selectedSearch) || selectedSearch != profile ||
      CudaPath.empty()) return reject("original CG provider has no exact profile search binding");
  sdk = CudaPath.getValue();
  llvm::sys::path::append(sdk, "include");
  llvm::SmallString<256> physicalSdk;
  if (llvm::sys::fs::real_path(sdk, physicalSdk)) return reject("original CG SDK root is not physical");
  if (!originalCgSdkRoot.empty() && originalCgSdkRoot != physicalSdk)
    return reject("original CG SDK root changed");
  unsigned optionCount = 0, directoryCount = 0;
  for (const auto &entry : compiler.getHeaderSearchOpts().UserEntries) {
    llvm::SmallString<256> path;
    if (!llvm::sys::fs::real_path(entry.Path, path) && path == profile) ++optionCount;
  }
  clang::ConstSearchDirIterator profileDirectory = nullptr;
  for (auto iterator = headers.search_dir_begin(); iterator != headers.search_dir_end(); ++iterator) {
    if (!iterator->isNormalDir()) return reject("original CG lookup does not admit frameworks or header maps");
    llvm::SmallString<256> path;
    if (!llvm::sys::fs::real_path(iterator->getName(), path) && path == profile) {
      ++directoryCount;
      profileDirectory = iterator;
    }
  }
  if (optionCount != 1 || directoryCount != 1)
    return reject("original CG lookup requires one unambiguous injected profile directory");
  // The actual lookup selected this profile, so every eligible prefix already
  // missed. Query its normal suffix in Clang's order without LookupFileCache,
  // SearchDirs mutation, PP CurDir changes or entering any candidate file.
  clang::OptionalFileEntryRef original;
  for (auto iterator = std::next(profileDirectory); iterator != headers.search_dir_end(); ++iterator) {
    llvm::StringRef query = name;
    llvm::SmallString<256> foundSearch, relative, mapped;
    bool userFramework = false, framework = false, headerMap = false;
    original = iterator->LookupFile(query, headers, location, &foundSearch, &relative,
        nullptr, nullptr, userFramework, framework, headerMap, mapped, true);
    if (userFramework || framework || headerMap || !mapped.empty() || query != name)
      return reject("original CG lookup was mapped");
    if (original) break;
  }
  if (!original) return reject("original CG SDK provider was not found without the injected profile");
  llvm::SmallString<256> expected(physicalSdk), candidate;
  llvm::sys::path::append(expected, name);
  if (original->getName() != expected ||
      llvm::sys::fs::real_path(original->getName(), candidate) || candidate != expected)
    return reject("original CG provider is a caller shadow or another SDK root");
  llvm::SmallString<256> runtime(physicalSdk);
  llvm::sys::path::append(runtime, "cuda_runtime_api.h");
  auto runtimeEntry = source.getFileManager().getFileRef(runtime);
  if (!runtimeEntry) {
    llvm::consumeError(runtimeEntry.takeError());
    return reject("original CG runtime provider is unavailable");
  }
  const auto runtimeFile = source.translateFile(&runtimeEntry->getFileEntry());
  if (runtimeFile.isInvalid() || !cudaRuntimeApiSdkFileIds.count(runtimeFile.getHashValue()) ||
      !isConfiguredCudaRuntimeApiPath(runtime, source))
    return reject("original CG provider has no authenticated parsed runtime-api owner");
  struct Role { const char *path; std::uint64_t size; const char *sha256; };
  static const Role roles[] = {
      {"cuda_runtime_api.h", 669895, "2c6732345e9662d91733f5e45fa9c76c955640ef02c4bf17ceb7f1bcc209d3de"},
      {"cooperative_groups.h", 55405, "64f650713034bfb355c12c5d64ce12efbd03a01efcfd67a76f2e2c8fa37376da"},
      {"cooperative_groups/reduce.h", 2949, "0748600e4a8cfbab9ea934e06f76f7e00d111f8bc6cfc9817f97b68d3d5d275a"},
      {"cooperative_groups/details/info.h", 12465, "14eae9dcbb6de0f71b2b67c0339517f63b2c159b63fcba954a1ccb15c2a249a6"},
      {"cooperative_groups/details/driver_abi.h", 3964, "bfe6546f852018a264e8d4765825870f7c7fe36cbeb68808d6eac59fbd068be8"},
      {"cooperative_groups/details/helpers.h", 22242, "063f2d5718bfd6d741949e65fb7a08044919a81f73675ca875acf8cf3890466a"},
      {"cooperative_groups/details/memory.h", 5606, "8444b749f817201b23d8c16b0bf33908e5e53a2ad206eba13cc0099d6a08f76c"}};
  for (const auto &role : roles) {
    llvm::SmallString<256> path(physicalSdk), canonical;
    llvm::sys::path::append(path, role.path);
    if (llvm::sys::fs::real_path(path, canonical) || canonical != path)
      return reject("original CG role path changed or escaped its SDK");
    auto entry = source.getFileManager().getFileRef(path);
    if (!entry) {
      llvm::consumeError(entry.takeError());
      return reject("original CG registered role is missing");
    }
    const auto bytes = llvm::MemoryBuffer::getFile(path);
    const auto &file = entry->getFileEntry();
    if (!bytes || file.getSize() != role.size || (*bytes)->getBufferSize() != role.size ||
        contextualInputSha256((*bytes)->getBuffer()) != role.sha256 ||
        !physicalSourceFileMatches(path, file, source))
      return reject("original CG role has no reviewed physical byte identity");
    OriginalCgInput input{path.str().str(), role.sha256,
        file.getUniqueID().getDevice(), file.getUniqueID().getFile(), role.size};
    const auto found = std::find_if(originalCgInputs.begin(), originalCgInputs.end(),
        [&](const OriginalCgInput &old) { return old.path == input.path; });
    if (found != originalCgInputs.end()) {
      if (found->sha256 != input.sha256 || found->device != input.device ||
          found->file != input.file || found->size != input.size)
        return reject("original CG role changed between includes");
    } else originalCgInputs.push_back(input);
  }
  for (const auto &control : originalCgControls()) {
    const auto definition = pp.getMacroDefinitionAtLoc(pp.getIdentifierInfo(control), location);
    if (control == "_CG_CLUSTER_INTRINSICS_AVAILABLE") {
      if (definition.isAmbiguous() || !definition.getModuleMacros().empty() ||
          !proveOriginalCgCompilerDefault(definition.getMacroInfo(),
              originalCgCompilerDefaultDefinition, false))
        return reject("original CG projection has no exact compiler-owned default control");
    } else if (definition)
      return reject("original CG projection requires an unmodified default control state");
  }
  originalCgSdkRoot = physicalSdk.str().str();
  originalCgIncludes.push_back(name.str() + ":" + candidate.str().str());
  return true;
#endif
}

bool AscifyAction::sealDynamicWarpGroupInputs() {
  if (!dynamicWarpGroups.used()) return true;
  auto &compiler=getCompilerInstance();
  auto &source=compiler.getSourceManager();
  auto &pp=compiler.getPreprocessor();
  const auto reject=[&](llvm::StringRef reason) {
    const auto id=compiler.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,"dynamic warp group boundary: %0");
    compiler.getDiagnostics().Report(id)<<reason;
    cudaRuntimeApiBoundaryFailed=true;return false;
  };
  auto &ast=compiler.getASTContext();
  const auto *aux=ast.getAuxTargetInfo();
  if (!frontendCompatibility.sourceCudaHostDriver ||
      frontendCompatibility.callerSdkMetadataOverride ||
      !ordinaryRuntimeProfileInput(compiler) || compiler.getLangOpts().CUDAIsDevice ||
      ast.getTargetInfo().getTriple().str()!="aarch64-conda-linux-gnu" || !aux ||
      aux->getTriple().str()!="nvptx64-nvidia-cuda" ||
      ast.getTargetInfo().getSDKVersion()!=llvm::VersionTuple(13,2) ||
      originalCgSdkRoot.empty() || originalCgIncludes.empty() ||
      cudaAttributeControlModified || deviceTemplateAttributeMacroConflict || frontendCgControlModified)
    return reject("current host-driver SDK tuple or physical CG projection is unproved");
  // Unparsed SDK providers are byte/UID witnesses only, never AST owners.
  struct Role { const char *path; std::uint64_t size; const char *sha256; };
  static const Role roles[]={
    {"cooperative_groups/details/partitioning.h",7153,"010cfe4e17aa5f7a27a97d91988522a73614541dbbdf384b947270fe45fd0f65"},
    {"cooperative_groups/details/functional.h",8905,"d8157c8bc062773d24831b989090806f0171388651cb31e01be73fad52a14737"},
    {"cooperative_groups/details/reduce.h",23296,"323a8c0f04f44f2599938256705dd61f0f31b7032aca2cc0e02df3cbb7fc79ed"},
    {"cooperative_groups/details/coalesced_reduce.h",4257,"a41420158ee2eb857ced73404e0d5410840754d60e22d42d1e34b9078ca7f297"},
    {"cooperative_groups/details/coalesced_scan.h",7314,"0df66fe5de56d1726ffad655860ac87632e3b3a682c7fbb4a32d650c8a63a354"}};
  const auto remember=[&](const std::string &path,std::uint64_t size,llvm::StringRef sha) {
    llvm::SmallString<256> canonical;
    auto entry=source.getFileManager().getFileRef(path);
    if (!entry) { llvm::consumeError(entry.takeError()); return false; }
    const auto bytes=llvm::MemoryBuffer::getFile(path);
    const auto &file=entry->getFileEntry();
    if (llvm::sys::fs::real_path(path,canonical)||canonical!=path||!bytes||
        file.getSize()!=size||(*bytes)->getBufferSize()!=size||
        contextualInputSha256((*bytes)->getBuffer())!=sha||
        !physicalSourceFileMatches(path,file,source)) return false;
    OriginalCgInput input{path,sha.str(),file.getUniqueID().getDevice(),file.getUniqueID().getFile(),size};
    const auto old=std::find_if(originalCgInputs.begin(),originalCgInputs.end(),
        [&](const OriginalCgInput &value) { return value.path==path; });
    if (old==originalCgInputs.end()) originalCgInputs.push_back(input);
    else if (old->sha256!=input.sha256||old->size!=input.size||
             old->device!=input.device||old->file!=input.file) return false;
    return true;
  };
  for (const auto &role:roles)
    if (!remember(originalCgSdkRoot+"/"+role.path,role.size,role.sha256))
      return reject("dynamic SDK provider changed");
  std::string backend,error;
  if (!ascify::ResolveDynamicWarpGroupsBackend(frontendCompatibility,backend,error)||
      !remember(backend,8916,"a26e6f1a40deede2d5571e46e2c59e7053c4a8c254de6b197eb12fbda5cbbc98"))
    return reject("owned dynamic backend has no complete physical identity");
  const auto native=[&](clang::SourceLocation location) {
    if (!location.isFileID()) return false;
    const auto file=source.getFileID(location);
    if (cudaRuntimeApiSdkFileIds.count(file.getHashValue())||isOwnedCudaCompilerInput(file,compiler))return true;
    if (verifiedFrontendCgEntryFileIds.count(file.getHashValue())&&
        !macroAlteredFrontendCgFileIds.count(file.getHashValue())&&
        verifiedFrontendCgFile(file,compiler,frontendCompatibility))return true;
    if (ordinaryRuntimeProfileInput(compiler)&&!frontendHostMathGuardModified&&
        !frontendCgControlModified&&!cudaAttributeControlModified&&
        verifiedFrontendHostMathEntryFileIds.count(file.getHashValue())&&
        !macroAlteredFrontendHostMathFileIds.count(file.getHashValue())&&
        verifiedForcedFrontendHostMathFile(file,compiler,frontendCompatibility)) return true;
    const auto *entry=source.getFileEntryForID(file);
    return entry&&trustedSystemFileIds.count(file.getHashValue())&&source.isInSystemHeader(location)&&
        !source.isFileOverridden(entry)&&physicalSourceFileMatches(source.getFilename(location),*entry,source)&&
        (pathIsWithinPhysicalDirectory(source.getFilename(location),"/usr/include")||
         pathIsWithinPhysicalDirectory(source.getFilename(location),"/usr/lib/gcc"));
  };
  for (const auto &event:dynamicWarpMacroEvents)
    if (!native(event.second)) return reject("caller define/undef history captures a dynamic backend token");
  for (const auto &macro:compiler.getPreprocessorOpts().Macros) {
    const llvm::StringRef definition(macro.first);
    if (isDynamicWarpControlledToken(definition.split('=').first.split('(').first.trim())||
        definition.contains('\\')||definition.contains('\n')||definition.contains('\r'))
      return reject("command-line macro history is outside the dynamic contract");
  }
  for (const auto &entry:pp.macros(false)) {
    if(!isDynamicWarpControlledToken(entry.first->getName()))continue;
    const auto *macro=pp.getMacroInfo(entry.first);
    if (macro&&!macro->isBuiltinMacro()&&!native(macro->getDefinitionLoc()))
      return reject("incoming dynamic backend macro has no native owner");
  }
  // This raw-token role exempts only compiler cacheline stack directives. It
  // does not classify the predefines buffer as native or grant AST ownership.
  const auto compilerCachelineStackToken=[&](clang::SourceLocation location,llvm::StringRef tokenName) {
    const auto file=pp.getPredefinesFileID();
    if (!location.isValid()||!location.isFileID()||file.isInvalid()||
        source.getFileID(location)!=file||source.getFileEntryForID(file)) return false;
    bool invalid=false;const auto bytes=source.getBufferData(file,&invalid);
    if (invalid||bytes!=pp.getPredefines()) return false;
    const auto offset=source.getFileOffset(location);if(offset>=bytes.size())return false;
    auto begin=offset;while(begin&&bytes[begin-1]!='\n'&&bytes[begin-1]!='\r')--begin;
    auto end=offset;while(end<bytes.size()&&bytes[end]!='\n'&&bytes[end]!='\r')++end;
    if(tokenName!="push_macro"||offset!=begin+llvm::StringRef("#pragma ").size())return false;
    const char *const keys[]={"__GCC_DESTRUCTIVE_SIZE","__GCC_CONSTRUCTIVE_SIZE"};
    const std::string expected[]={
      "#pragma push_macro(\"__GCC_DESTRUCTIVE_SIZE\")",
      "#pragma push_macro(\"__GCC_CONSTRUCTIVE_SIZE\")"};
    unsigned occurrences[2]={0,0};llvm::StringRef remaining=bytes;
    while(!remaining.empty()) {
      const auto split=remaining.split('\n');
      for(unsigned i=0;i<2;++i)if(split.first==expected[i])++occurrences[i];
      remaining=split.second;
    }
    // The observed compiler emits each of these two directives twice. No
    // extra occurrence or unobserved pop directive acquires this token role.
    if(occurrences[0]!=2||occurrences[1]!=2)return false;
    const auto line=bytes.slice(begin,end);
    for(unsigned i=0;i<2;++i)
      if(line==expected[i]&&!isDynamicWarpControlledToken(keys[i])&&
          !isAscifyCudaCompatIdentifier(keys[i]))return true;
    return false;
  };
  // Inspect all retained caller file instances, including inactive directives.
  // A later undef or a macro stack restore cannot erase the source history.
  for (const auto file:enteredSourceFileIds) {
    const auto begin=source.getLocForStartOfFile(file);
    // Reuse the existing physical product-resource/native-parent proof for
    // this raw-token role only; it never grants CUDA SDK AST ownership.
    if (native(begin)||fullWarpNativeCompilerInput(file,compiler,trustedSystemFileIds)) continue;
    // The configured --cuda-path is an already-audited SDK dependency tree
    // (its include tuple is pinned by the SDK identity checks elsewhere).
    // Its headers are producer-side input, not caller retentions.
    if (pathIsWithinConfiguredCudaInclude(source.getFilename(begin)))
      continue;
    // Same producer-side class for the toolchain's own CUDA wrappers: a file
    // that byte- and hash-matches the product's owned clang resource tree
    // (physicalOwnedClangInput pins resourceRoot == the owned install and
    // re-reads the reference file) cannot be a caller retention. The command
    // -D names those wrappers legitimately spell do not void that identity.
    if (physicalOwnedClangInput(file,compiler)) continue;
    bool invalid=false;const auto bytes=source.getBufferData(file,&invalid);
    if (invalid) return reject("cannot inventory dynamic caller macro history");
    const bool frozenHelper=frozenNvidiaSampleHelperFileRole(source,begin)!=0;
    clang::Lexer lexer(begin,compiler.getLangOpts(),bytes.begin(),bytes.begin(),bytes.end());
    unsigned directive=0;
    // Target-equivalent __CUDA_ARCH__ witness state: the conversion parses
    // under the exact --cuda-gpu-arch value and the target compile pins
    // __CUDA_ARCH__ to that same value, so a directive line whose only
    // identifier is the compiler builtin evaluates identically on the
    // target. A caller-defined __CUDA_ARCH__ voids the witness.
    unsigned archConditionalDepth=0;
    unsigned long long archDriverPinnedValue=0;
    int archCollectRemaining=0;
    bool archCollectIsOpener=false;
    bool archSawForeignIdentifier=false;
    bool archBuiltinClean=true;
    bool archCommandPinned=false;
    for (const auto &macro:compiler.getPreprocessorOpts().Macros) {
      // The conversion driver pins __CUDA_ARCH__ from --cuda-gpu-arch; that
      // plain numeric define is exactly the target-equivalent witness and
      // stays clean. Anything else (undef, spliced, multi-token) voids it.
      llvm::StringRef spelling(macro.first);
      if (!spelling.starts_with("__CUDA_ARCH__")) continue;
      if (macro.second||!spelling.consume_front("__CUDA_ARCH__=")||
          spelling.empty()||spelling.getAsInteger(10,archDriverPinnedValue))
        archBuiltinClean=false;
      else
        archCommandPinned=true;
    }
    (void)archCommandPinned;
    // No definition-location audit is needed here: any caller-source
    // #define/#undef of __CUDA_ARCH__ is rejected outright by the raw
    // define/undef audit below, and every non-caller file class (system
    // headers, owned resources, the configured SDK tree) is skipped before
    // this loop body runs. The offload layer remains free to own the
    // builtin's final definition.
    for (;;) {
      clang::Token token;lexer.LexFromRawLexer(token);
      if (token.is(clang::tok::eof)) {
        if (archCollectRemaining>0) {
          if (!archBuiltinClean||archSawForeignIdentifier)
            return reject("retained caller conditional has no target-equivalent source witness");
          if (archCollectIsOpener) ++archConditionalDepth;
          archCollectRemaining=0;
        }
        break;
      }
      if (token.isAtStartOfLine()) {
        if (archCollectRemaining>0) {
          if (!archBuiltinClean||archSawForeignIdentifier)
            return reject("retained caller conditional has no target-equivalent source witness");
          if (archCollectIsOpener) ++archConditionalDepth;
          archCollectRemaining=0;
        }
        directive=token.is(clang::tok::hash)?1:0;
      }
      if (!token.isAnyIdentifier())continue;
      const auto name=clang::Lexer::getSpelling(token,source,compiler.getLangOpts());
      if(llvm::StringRef(name).starts_with("_CG_")||llvm::StringRef(name).starts_with("_COOPERATIVE_GROUPS_"))
        return reject("retained caller token observes a private original CG control");
      if ((name=="_Pragma"||name=="__pragma"||name=="push_macro"||name=="pop_macro")&&
          !compilerCachelineStackToken(token.getLocation(),name))
        return reject("dynamic caller macro-stack history is unproved");
      // Current-CI declaration witnesses cannot see a target-only branch.
      // Frozen official helpers grant only this conditional-token role: their
      // exact bytes cannot inject the owned group namespace. All macro/name
      // checks here and their existing independent closure proof still apply.
      if (name=="ascify"||name=="ascify_cg"||name=="asc_shfl"||
          name=="asc_ballot"||name=="asc_activemask")
        return reject("retained caller token can inject an owned backend name");
      if (directive==1&&!frozenHelper&&
          (name=="if"||name=="ifdef"||name=="ifndef"||name=="elif"||
           name=="elifdef"||name=="elifndef"||name=="else"||name=="endif")) {
        if (name=="else"||name=="endif") {
          if (archConditionalDepth==0)
            return reject("retained caller conditional has no target-equivalent source witness");
          if (name=="endif") --archConditionalDepth;
          directive=0;
          continue;
        } else if (llvm::StringRef(name).starts_with("elif")) {
          if (archConditionalDepth==0)
            return reject("retained caller conditional has no target-equivalent source witness");
          archCollectRemaining=64;
          archSawForeignIdentifier=false;
          archCollectIsOpener=false;
          directive=2;
        } else {
          archCollectRemaining=64;
          archSawForeignIdentifier=false;
          archCollectIsOpener=true;
          directive=2;
        }
        continue;
      } else if (archCollectRemaining>0) {
        if (llvm::StringRef(name)!="__CUDA_ARCH__")
          archSawForeignIdentifier=true;
        --archCollectRemaining;
        continue;
      }
      if (directive==1)directive=(name=="define"||name=="undef")?2:0;
      else if (directive==2) {
        if (name=="__CUDA_ARCH__") {
          // The command-line pin physically lands as a #define in the
          // nameless predefines buffer; that is the validated driver pin
          // itself. Only a define/undef inside a named caller file can
          // forge the builtin.
          if (!llvm::StringRef(source.getFilename(begin)).empty())
            return reject("retained define/undef captures the pinned architecture builtin");
          directive=0;
          continue;
        }
        if (token.needsCleaning()||isDynamicWarpControlledToken(name))
          return reject("retained define/undef captures a dynamic backend token");
        directive=0;
      }
    }
  }
  if (localHeaderContext&&localHeaderContext->inputEvidence)
    localHeaderContext->inputEvidence->dynamicWarpSdkTuple=
        ast.getTargetInfo().getTriple().str()+";"+ast.getTargetInfo().getSDKVersion().getAsString()+
        ";"+aux->getTriple().str();
  return true;
}

bool AscifyAction::sealOriginalCgProviders() {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  const auto fail = [&]() {
    const auto diagnostic = compiler.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "frontend compatibility original CG provider boundary: identity or default-control history is unproved");
    compiler.getDiagnostics().Report(diagnostic);
    cudaRuntimeApiBoundaryFailed = true;
    return false;
  };
  if (originalCgInquiry) return fail(); // Unmodeled __has_include semantics cannot borrow the injected file.
  for (const auto file : enteredSourceFileIds) {
    if (!verifiedFrontendCgEntryFileIds.count(file.getHashValue())) continue;
    const auto location = source.getLocForStartOfFile(file);
    llvm::SmallString<256> path;
    if (llvm::sys::fs::real_path(source.getFilename(location), path)) return fail();
    const std::string role = path == frontendCompatibility.canonicalRoot + "/cooperative_groups.h"
        ? "cooperative_groups.h" : path == frontendCompatibility.canonicalRoot + "/cooperative_groups/reduce.h"
        ? "cooperative_groups/reduce.h" : std::string();
    if (role.empty() || std::none_of(originalCgIncludes.begin(), originalCgIncludes.end(),
          [&](const std::string &include) { return llvm::StringRef(include).starts_with(role + ":"); }))
      return fail();
  }
  if (originalCgIncludes.empty()) return true;
  if (!proveOriginalCgCompilerDefault(compiler.getPreprocessor().getMacroInfo(
          compiler.getPreprocessor().getIdentifierInfo("_CG_CLUSTER_INTRINSICS_AVAILABLE")),
          originalCgCompilerDefaultDefinition, false)) return fail();
  llvm::SmallString<256> root(CudaPath.getValue()), canonical;
  llvm::sys::path::append(root, "include");
  if (llvm::sys::fs::real_path(root, canonical) || canonical != originalCgSdkRoot) return fail();
  for (const auto &macro : compiler.getPreprocessorOpts().Macros)
    if (originalCgControlText(macro.first) || llvm::StringRef(macro.first).contains('\\'))
      originalCgControlChanged = true; // Unmodeled escaped CLI replacements fail closed.
  for (const auto file : enteredSourceFileIds) {
    bool invalid = false;
    const auto bytes = source.getBufferData(file, &invalid);
    if (invalid) return fail();
    clang::Lexer lexer(source.getLocForStartOfFile(file), compiler.getLangOpts(),
        bytes.begin(), bytes.begin(), bytes.end());
    for (;;) {
      clang::Token token;
      lexer.LexFromRawLexer(token);
      if (token.is(clang::tok::eof)) break;
      if (token.is(clang::tok::raw_identifier)) {
        const auto name = clang::Lexer::getSpelling(token, source, compiler.getLangOpts());
        if (originalCgControlName(name) &&
            !(name == "_CG_CLUSTER_INTRINSICS_AVAILABLE" &&
              proveOriginalCgCompilerDefault(originalCgCompilerDefaultMacro,
                  token.getLocation(), false)))
          originalCgControlChanged = true;
      }
      // Decoded strings also close escaped pragma push/pop targets. This
      // bounded default mode does not admit caller mentions of private CG controls.
      if (token.is(clang::tok::string_literal)) {
        clang::StringLiteralParser literal(llvm::ArrayRef<clang::Token>(&token, 1),
            source, compiler.getLangOpts(), compiler.getTarget(), nullptr);
        if (literal.hadError || originalCgControlText(literal.GetString()))
          originalCgControlChanged = true;
      }
    }
  }
  if (originalCgControlChanged) return fail();
  auto providerInputs = originalCgInputs;
  providerInputs.push_back(originalCgCompilerDefaultInput);
  for (const auto &input : providerInputs) {
    llvm::SmallString<256> physical;
    auto entry = source.getFileManager().getFileRef(input.path);
    if (!entry) { llvm::consumeError(entry.takeError()); return fail(); }
    const auto bytes = llvm::MemoryBuffer::getFile(input.path);
    const auto &file = entry->getFileEntry();
    if (llvm::sys::fs::real_path(input.path, physical) || physical != input.path ||
        !bytes || (*bytes)->getBufferSize() != input.size ||
        contextualInputSha256((*bytes)->getBuffer()) != input.sha256 ||
        file.getUniqueID().getDevice() != input.device || file.getUniqueID().getFile() != input.file ||
        !physicalSourceFileMatches(input.path, file, source)) return fail();
    if (localHeaderContext && localHeaderContext->inputEvidence) {
      auto &evidence = *localHeaderContext->inputEvidence;
      const auto path = contextualRecordPathBinding(evidence, input.path);
      if (path.empty()) return fail();
      evidence.fileSha256[path] = input.sha256;
      evidence.providerIdentities[path] = {input.device, input.file};
      evidence.providerSizes[path] = input.size;
    }
  }
  return true;
}

void AscifyAction::InclusionDirective(clang::SourceLocation hash_loc,
                                      const clang::Token& includeToken,
                                      StringRef file_name,
                                      bool is_angled,
                                      clang::CharSourceRange filename_range,
                                      StringRef resolved_file_name, StringRef search_path,
                                      StringRef, const clang::Module* module) {
  packedRecords.includeOrFileTransition();
  if (StandaloneDeviceExecutable && !resolved_file_name.empty()) {
    deviceAggregateResolvedIncludes.insert(contextualCanonicalPath(resolved_file_name));
    bool bad=false;
    const auto text=clang::Lexer::getSourceText(filename_range,getCompilerInstance().getSourceManager(),getCompilerInstance().getLangOpts(),&bad);
    const std::string expected=(is_angled?"<":"\"")+file_name.str()+(is_angled?">":"\"");
    deviceAggregateIncludeEdges.push_back({hash_loc,filename_range,file_name.str(),resolved_file_name.str(),search_path.str(),is_angled,
        !bad && hash_loc.isFileID() && filename_range.getBegin().isFileID() && filename_range.getEnd().isFileID() && text==expected &&
        includeToken.getIdentifierInfo() && includeToken.getIdentifierInfo()->getPPKeywordID()==clang::tok::pp_include});
  }
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
  if (!proveOriginalCgInclude(hash_loc, includeToken, file_name, is_angled,
                              filename_range, search_path, resolved_file_name, frontendCompatibilityError)) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "frontend compatibility original CG provider boundary: %0");
    getCompilerInstance().getDiagnostics().Report(filename_range.getBegin(), diagnostic)
        << frontendCompatibilityError;
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  outs() << "File included: " << file_name << "\n";
  auto &SM = getCompilerInstance().getSourceManager();
  if (file_name=="stdio.h" && is_angled && !module && hash_loc.isFileID() &&
      SM.isWrittenInMainFile(hash_loc) && filename_range.getBegin().isFileID() &&
      filename_range.getEnd().isFileID() &&
      includeToken.getIdentifierInfo() && includeToken.getIdentifierInfo()->getName()=="include") {
    bool invalid=false;
    const auto literal=clang::Lexer::getSourceText(filename_range,SM,
        getCompilerInstance().getLangOpts(),&invalid);
    if (!invalid && literal=="<stdio.h>")
      uniformNativeStdioIncludes.emplace(SM.getFileOffset(hash_loc),resolved_file_name.str());
  }
  // Preserve every real include lookup, including guarded/skipped children.
  // An input with a resource spelling cannot substitute a caller-owned file
  // and disappear from the later FileID inventory merely by being empty.
  if (includeToken.getIdentifierInfo() &&
      includeToken.getIdentifierInfo()->getName() == "include") {
    const auto owned = ownedClangResourceDirectory();
    const auto parent = SM.getFileID(SM.getExpansionLoc(hash_loc));
    llvm::SmallString<256> expected;
    if (!is_angled && physicalOwnedClangInput(parent, getCompilerInstance())) {
      expected = llvm::sys::path::parent_path(SM.getFilename(SM.getLocForStartOfFile(parent)));
      llvm::sys::path::append(expected, file_name);
    }
    if (expected.empty() || !llvm::sys::fs::exists(expected)) {
      expected = owned;
      llvm::sys::path::append(expected, "include", file_name);
    }
    if (!owned.empty() && llvm::sys::fs::exists(expected) &&
        pathIsWithinPhysicalDirectory(expected, owned)) {
      llvm::sys::fs::UniqueID wanted, actual;
      llvm::SmallString<256> wantedPath, actualPath;
      bool invalid = false;
      const auto literal = clang::Lexer::getSourceText(filename_range, SM,
          getCompilerInstance().getLangOpts(), &invalid);
      const auto at = SM.getLocForStartOfFile(parent);
      const auto *entry = SM.getFileEntryForID(parent);
      const auto path = SM.getFilename(at);
      const bool nativeParent = entry && trustedSystemFileIds.count(parent.getHashValue()) &&
          SM.isInSystemHeader(at) && physicalSourceFileMatches(path, *entry, SM) &&
          (pathIsWithinPhysicalDirectory(path, "/usr/include") ||
           pathIsWithinPhysicalDirectory(path, "/usr/lib/gcc"));
      const bool forcedWrapper = parent == getCompilerInstance().getPreprocessor().getPredefinesFileID() &&
          llvm::sys::path::filename(expected) == "__clang_cuda_runtime_wrapper.h";
      const bool ownedParent = forcedWrapper || physicalOwnedClangInput(parent, getCompilerInstance()) ||
          cudaRuntimeApiSdkFileIds.count(parent.getHashValue()) || nativeParent;
      if (!ownedParent || module || !hash_loc.isFileID() || invalid ||
          !filename_range.getBegin().isFileID() || !filename_range.getEnd().isFileID() ||
          literal != (is_angled ? "<" + file_name.str() + ">" : "\"" + file_name.str() + "\"") ||
          llvm::sys::fs::getUniqueID(expected, wanted) ||
          llvm::sys::fs::getUniqueID(resolved_file_name, actual) || wanted != actual ||
          llvm::sys::fs::real_path(expected, wantedPath) ||
          llvm::sys::fs::real_path(resolved_file_name, actualPath) || wantedPath != actualPath)
        fullWarpCompilerIncludeFailure = hash_loc;
    }
  }
  if (localHeaderContext != nullptr && localHeaderContext->inputEvidence != nullptr) {
    auto &evidence = *localHeaderContext->inputEvidence;
    const auto parent = contextualRecordPathBinding(evidence, SM.getFilename(SM.getExpansionLoc(hash_loc)));
    const auto child = contextualRecordPathBinding(evidence, resolved_file_name);
    if (!parent.empty()) {
      const auto begin = filename_range.getBegin();
      const auto end = filename_range.getEnd();
      bool invalid = false;
      const auto literal = clang::Lexer::getSourceText(
          filename_range, SM, getCompilerInstance().getLangOpts(), &invalid);
      const bool direct = !invalid && hash_loc.isFileID() &&
          begin.isFileID() && end.isFileID() &&
          SM.getFileID(hash_loc) == SM.getFileID(begin) &&
          SM.getFileID(begin) == SM.getFileID(end) &&
          !is_angled && literal == ("\"" + file_name.str() + "\"") &&
          module == nullptr && !getCompilerInstance().getLangOpts().Modules &&
          includeToken.getIdentifierInfo() != nullptr &&
          includeToken.getIdentifierInfo()->getName() == "include";
      localHeaderContext->inputEvidence->originalEdges.push_back(
          {parent, file_name.str(), child, {},
           SM.getFileOffset(SM.getExpansionLoc(hash_loc)), 0, {}, is_angled,
           direct, direct ? SM.getFileOffset(begin) : 0, false});
    }
  }
  if (!is_angled && localHeaderContext != nullptr &&
      localHeaderContext->inputEvidence != nullptr) {
    const auto parent = contextualCanonicalPath(SM.getFilename(
        SM.getExpansionLoc(hash_loc)));
    if (!parent.empty())
      localHeaderContext->inputEvidence->quotedIncludeParents.insert(parent);
  }

  // A removed SDK or owned compiler header may not silently obtain a
  // same-named dependency from a caller directory. Retained SDK parents keep
  // their normal lookup and remain subject to the retained observation audit.
  // Account for quoted relative lookup
  // first, then the configured include root; native-library dependencies
  // without a corresponding SDK file remain subject to the retained audit.
  const auto includeParent = SM.getFilename(SM.getExpansionLoc(hash_loc));
  const auto includeParentFile = SM.getFileID(SM.getExpansionLoc(hash_loc));
  const bool removedSdkParent = includeParentFile.isValid() &&
      (cudaRuntimeApiSdkFileIds.count(includeParentFile.getHashValue()) ||
       isOwnedCudaCompilerInput(includeParentFile, getCompilerInstance()));
  if (removedSdkParent) {
    llvm::SmallString<256> expected;
    if (!is_angled) {
      expected = llvm::sys::path::parent_path(includeParent);
      llvm::sys::path::append(expected, file_name);
    }
    if (expected.empty() || !llvm::sys::fs::exists(expected)) {
      expected = CudaPath.getValue();
      llvm::sys::path::append(expected, "include", file_name);
    }
    if (llvm::sys::fs::exists(expected) &&
        pathIsWithinConfiguredCudaInclude(expected)) {
      llvm::SmallString<256> expectedPhysical, resolvedPhysical;
      if (llvm::sys::fs::real_path(expected, expectedPhysical) ||
          llvm::sys::fs::real_path(resolved_file_name, resolvedPhysical) ||
          (expectedPhysical != resolvedPhysical &&
           !physicalCudaSdkNvAliasMatches(expectedPhysical, resolvedPhysical)))
        cudaRuntimeApiSdkIdentityFailure = hash_loc;
    }
  }

  const bool recognizedNvidiaHelper =
      isRecognizedNvidiaSampleHelperCuda(resolved_file_name) ||
      isRecognizedNvidiaSampleHelperDrvapi(resolved_file_name);
  observeDirectImageInclude(hash_loc, includeToken, file_name, filename_range,
                            resolved_file_name, module);
  if (!SM.isWrittenInMainFile(hash_loc)) {
    if (recognizedNvidiaHelper) {
      // This is a request for a second, original-position proof, not an
      // exception to helper provenance. The current transaction still fails.
      // Only a selected direct local provider can become a contextual input;
      // the joint parse must then prove the usual direct helper transaction.
      if (localHeaderContext != nullptr && localHeaderContext->depth == 0 &&
          localHeaderContext->plan != nullptr)
        localHeaderContext->plan->requireInheritedHelperContext(
            SM.getFilename(SM.getExpansionLoc(hash_loc)).str());
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
  // FileSkipped, not a header spelling or a pre-existing guard macro, will
  // prove that this directive contributes no source tokens. Keep the exact
  // directive and selected path to bind that later callback to this include.
  if (file_name == "cub/util_type.cuh" && directRange && directSpelling &&
      preprocessorConditionalDepth == 0 && module == nullptr &&
      !getCompilerInstance().getLangOpts().Modules &&
      includeToken.getIdentifierInfo() != nullptr &&
      includeToken.getIdentifierInfo()->getName() == "include")
    redundantCubIncludeCandidates.push_back(
        {hash_loc, filenameBegin, filenameEnd, resolved_file_name.str()});
  // A selected physical header is parsed as its own main file. Treat only a
  // proved complete outer guard as structural, for both Runtime entry points.
  // A caller main, virtual contextual input or extra condition is unchanged.
  const bool guardedLocalRuntimeRoot =
      (file_name == "cuda_runtime.h" || file_name == "cuda_runtime_api.h") &&
      preprocessorConditionalDepth == 1 && localHeaderContext != nullptr &&
      localHeaderContext->plan != nullptr && localHeaderContext->depth != 0 &&
      localHeaderContext->virtualInput == nullptr &&
      hasCompleteMainOuterGuard(getCompilerInstance());
  // This is only an additional physical SDK provenance root. It does not
  // enable the broader Runtime-header projection or exempt any AST/layout,
  // macro-observation, transitive dependency or publication check.
  if (file_name == "cuda_runtime.h" && !getCompilerInstance().getLangOpts().CUDA &&
      !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks &&
      directRange && directSpelling &&
      (preprocessorConditionalDepth == 0 || guardedLocalRuntimeRoot) &&
      module == nullptr && !getCompilerInstance().getLangOpts().Modules &&
      includeToken.getIdentifierInfo() != nullptr &&
      includeToken.getIdentifierInfo()->getName() == "include" &&
      isConfiguredCudaRuntimeHeaderPath(resolved_file_name, SM)) {
    bool invalid = false;
    const auto mainBytes = SM.getBufferData(SM.getMainFileID(), &invalid);
    if (!invalid)
      hostCudaRuntimeLiteralRootLines.insert(
          SM.getLocForStartOfFile(SM.getMainFileID()).getLocWithOffset(
              physicalLineBegin(mainBytes, SM.getFileOffset(hash_loc))).getRawEncoding());
  }
  if (file_name == "cuda_runtime_api.h") {
    // A recursive closure translates a selected header as its own main file.
    // Its ordinary whole-file guard is preserved in the emitted header. No
    // arbitrary main-TU condition or nested local condition is admitted here.
    if (!directRange || !directSpelling ||
        (preprocessorConditionalDepth != 0 && !guardedLocalRuntimeRoot) ||
        !isConfiguredCudaRuntimeApiPath(resolved_file_name, SM)) {
      cudaRuntimeApiBoundaryFailed = true;
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "CUDA runtime-API header boundary: requires an unconditional direct "
          "literal include of the physical --cuda-path/include/cuda_runtime_api.h");
      getCompilerInstance().getDiagnostics().Report(hash_loc, diagnostic);
      return;
    }
    cudaRuntimeApiHeaderAdmitted = true;
    needsCudaCompatHeader = true;
    auditActiveAscifyCudaCompatMacrosAtInsertion();
  }
  if (file_name == "cuda_profiler_api.h") {
    bool provedConditional = false;
    for (auto &candidate : profilerAvailabilityCandidates) {
      if (directRange && directSpelling && is_angled &&
          preprocessorConditionalDepth == 1 && candidate.lookupProved &&
          SM.getFileID(hash_loc) == candidate.file &&
          SM.getFileOffset(hash_loc) == candidate.includeBegin &&
          includeToken.getIdentifierInfo() != nullptr &&
          includeToken.getIdentifierInfo()->getName() == "include" &&
          !candidate.includeObserved) {
        candidate.includeObserved = true;
        candidate.includeLocation = hash_loc;
        provedConditional = true;
      }
    }
    if (!directRange || !directSpelling ||
        (preprocessorConditionalDepth != 0 && !provedConditional) ||
        !isPhysicalCudaProfilerHeader(resolved_file_name)) {
      cudaRuntimeApiBoundaryFailed = true;
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "CUDA profiler header boundary: requires an unconditional direct "
          "literal include of the verified physical CUDA profiler header");
      getCompilerInstance().getDiagnostics().Report(hash_loc, diagnostic);
      return;
    }
    cudaProfilerApiHeaderAdmitted = true;
    needsCudaCompatHeader = true;
    auditActiveAscifyCudaCompatMacrosAtInsertion();
  }
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
    if ((file_name == "helper_cuda.h" || file_name == "helper_cuda_drvapi.h") &&
        directRange && directSpelling) {
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
    Statistics::current().incrementCounter(found->second, file_name.str());
    clang::SourceLocation sl = filename_range.getBegin();

    if (Statistics::isUnsupported(found->second)) {
      clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
      std::string sWarn = sTargetPlatform;
      const auto ID = DE.getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "'%0' is unsupported header in '%1'.");
      DE.Report(sl, ID) << found->first << sWarn;
      return;
    }
    // Preserve the source delimiter style and own the replacement text until
    // Replacement has copied it. A StringRef into a block-local SmallString
    // would dangle after that block ends.
    const std::string name = found->second.dppName.str();
    const std::string newInclude =
        is_angled ? "<" + name + ">" : "\"" + name + "\"";
    const char *B = SM.getCharacterData(sl);
    const char *E = SM.getCharacterData(filename_range.getEnd());
    ct::Replacement Rep(SM, sl, E - B, newInclude);
    const clang::SourceLocation replacementEnd =
        sl.getLocWithOffset(static_cast<int>(E - B));
    // The raw identifier pass also sees tokens inside include spellings (for
    // example both `cub` tokens in <cub/cub.cuh>).  Make the preprocessor's
    // include rewrite the sole owner of that span.
    const bool inserted = insertSemanticReplacement(
        Rep, clang::FullSourceLoc{sl, SM}, sl, replacementEnd);
    if (file_name == "cuda_profiler_api.h")
      for (auto &candidate : profilerAvailabilityCandidates)
        if (candidate.includeLocation == hash_loc)
          candidate.includeMapped = inserted && newInclude == "<acl/acl_prof.h>";
    if (inserted && file_name == "cub/cub.cuh" &&
        name == "acl_cub/aclcub.hpp" && directRange && directSpelling)
      mappedCubIncludes.push_back(
          {hash_loc, filenameEnd, resolved_file_name.str(),
           preprocessorConditionalDepth == 0});
    return;
  }

  if (localHeaderContext == nullptr)
    return;

  // A recognized direct helper include belongs to the all-or-nothing helper
  // transaction. Quoting it must not independently redirect its filename into
  // a local-header artifact: that would overlap the later whole-directive
  // replacement, or change a retained helper when the proof fails. The same
  // frozen-profile and literal-range proof above also handles angled includes.
  if (recognizedNvidiaHelper && file_name == "helper_cuda.h" &&
      directRange && directSpelling)
    return;

  const clang::SourceLocation fileLoc =
      SM.getFileLoc(filename_range.getBegin());
  const bool isLiteral = !filename_range.getBegin().isMacroID();
  const unsigned sourceOffset = fileLoc.isValid() ? SM.getFileOffset(fileLoc)
                                                   : 0;
  const std::string resolvedPath = resolved_file_name.str();
  const auto presumedInclude = SM.getPresumedLoc(hash_loc);
  const LocalHeaderIncludeDecision decision = localHeaderContext->observe(
      sourceOffset, file_name.str(), resolvedPath, is_angled, isLiteral,
      presumedInclude.isValid() ? presumedInclude.getLine() + 1 : 0,
      presumedInclude.isValid() ? presumedInclude.getFilename() : "");
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

void AscifyAction::FileSkipped(const clang::FileEntry& skippedFile,
                               const clang::Token& filenameToken,
                               StringRef resolvedPath) {
  observeDirectImageSkip(skippedFile, filenameToken, resolvedPath);
  packedRecords.includeOrFileTransition();
  const auto location = filenameToken.getLocation();
  if (!location.isFileID())
    return;
  auto& source = getCompilerInstance().getSourceManager();
  for (const auto &candidate : profilerAvailabilityCandidates) {
    if (source.getFileID(location) != candidate.file) continue;
    const auto bytes = source.getBufferData(candidate.file);
    if (physicalLineBegin(bytes, source.getFileOffset(location)) ==
        physicalLineBegin(bytes, candidate.includeBegin))
      profilerAvailabilityProofFailed = true;
  }
  if (localHeaderContext != nullptr && localHeaderContext->inputEvidence != nullptr) {
    auto &evidence = *localHeaderContext->inputEvidence;
    const auto parent = contextualRecordPathBinding(evidence, source.getFilename(location));
    const auto child = contextualRecordPathBinding(evidence, resolvedPath);
    const auto previous = evidence.fileSha256.find(child);
    const auto file = source.translateFile(&skippedFile);
    bool invalid = false;
    const auto contents = file.isValid()
        ? source.getBufferData(file, &invalid) : llvm::StringRef();
    // A pre-defined guard is not evidence of an earlier physical include.
    // Bind the actual FileSkipped callback to a literal edge and the exact
    // already-entered, non-overridden file bytes and physical identity.
    if (!parent.empty() && !child.empty() && file.isValid() && !invalid &&
        enteredSourceFileIds.count(file) != 0 &&
        evidence.fileEntries[child] == 1 && previous != evidence.fileSha256.end() &&
        !previous->second.empty() && previous->second == contextualInputSha256(contents) &&
        physicalSourceFileMatches(resolvedPath, skippedFile, source)) {
      for (auto &edge : evidence.originalEdges)
        if (edge.parentSourcePath == parent && edge.childSourcePath == child &&
            edge.directLiteral && edge.filenameOffset == source.getFileOffset(location))
          edge.physicallySkipped = true;
    }
  }
  for (const auto& candidate : redundantCubIncludeCandidates) {
    if (candidate.filenameBegin != location ||
        candidate.resolvedPath != resolvedPath ||
        !ascify::IsRedundantFrozenCubUtilInclude(
            source, CudaPath.getValue(), mappedCubIncludes, skippedFile,
            candidate.hashLocation))
      continue;
    // Removing just a filename must not turn ignored directive-tail tokens
    // into live C++. Admit one physical line with a whitespace/comment tail;
    // retain continuations, multiline block comments, and extra tokens.
    bool invalid = false;
    const auto bytes = source.getBufferData(
        source.getFileID(candidate.hashLocation), &invalid);
    const unsigned begin = source.getFileOffset(candidate.hashLocation);
    const unsigned end = source.getFileOffset(candidate.filenameEnd);
    if (invalid || end < begin || end > bytes.size())
      return;
    const auto prefix = bytes.slice(begin, end);
    if (prefix.find_first_of("\r\n\\") != StringRef::npos ||
        prefix.contains("?" "?/"))
      return;
    size_t cursor = end;
    while (cursor < bytes.size() && bytes[cursor] != '\n' &&
           bytes[cursor] != '\r') {
      if (bytes[cursor] == ' ' || bytes[cursor] == '\t' ||
          bytes[cursor] == '\v' || bytes[cursor] == '\f') {
        ++cursor;
        continue;
      }
      if (bytes.substr(cursor, 2) == "//")
        break;
      if (bytes.substr(cursor, 2) == "/*") {
        const size_t close = bytes.find("*/", cursor + 2);
        const size_t newline = bytes.find_first_of("\r\n", cursor + 2);
        if (close == StringRef::npos ||
            (newline != StringRef::npos && newline < close))
          return;
        cursor = close + 2;
        continue;
      }
      return;
    }
    const unsigned length = source.getFileOffset(candidate.filenameEnd) -
                            source.getFileOffset(candidate.hashLocation);
    ct::Replacement replacement(source, candidate.hashLocation, length, "");
    if (insertSemanticReplacement(
            replacement, clang::FullSourceLoc(candidate.hashLocation, source),
            candidate.hashLocation, candidate.filenameEnd))
      llvm::errs() << "Ascify CUB redundant include: removed frozen "
                      "cub/util_type.cuh after mapped umbrella\n";
    return;
  }
}

void AscifyAction::PragmaDirective(
    clang::SourceLocation Loc,
    clang::PragmaIntroducerKind Introducer) {
  if (StandaloneDeviceExecutable) deviceAggregateMacroEvidence.pragmas.push_back(Loc);
  recordDirectImageUtilityControl("", "pragma", Loc, nullptr);
  if (localHeaderContext != nullptr &&
      localHeaderContext->inputEvidence != nullptr) {
    auto &sourceManager = getCompilerInstance().getSourceManager();
    const auto path = contextualCanonicalPath(sourceManager.getFilename(
        sourceManager.getExpansionLoc(Loc)));
    if (!path.empty())
      localHeaderContext->inputEvidence->pragmaFiles.insert(path);
  }

  finalizePendingNvidiaSampleHelperPragma();
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::SourceLocation expansion =
      sourceManager.getExpansionLoc(Loc);
  if (Introducer != clang::PIK_HashPragma && expansion.isValid()) {
    const auto spelling = sourceManager.getSpellingLoc(Loc);
    const auto file = sourceManager.getFileID(spelling);
    if (!locationComesFromInitiallyTrustedSystemFile(
            sourceManager, spelling, trustedSystemFileIds) &&
        !cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
        frozenNvidiaSampleHelperFileRole(sourceManager, spelling) == 0 &&
        !isOwnedCudaCompilerInput(file, getCompilerInstance()))
      profilerAvailabilityControlModified = true;
  }
  if (Introducer != clang::PIK_HashPragma) {
    if (frontendCompatibility.enabled()) {
      auto &compiler = getCompilerInstance();
      const auto file = expansion.isValid() ? sourceManager.getFileID(expansion)
                                            : clang::FileID();
      const auto *entry = file.isValid() ? sourceManager.getFileEntryForID(file)
                                         : nullptr;
      const auto path = expansion.isValid() ? sourceManager.getFilename(expansion)
                                            : llvm::StringRef();
      const bool physicalNative = expansion.isValid() && entry != nullptr &&
          sourceManager.isInSystemHeader(expansion) &&
          physicalSourceFileMatches(path, *entry, sourceManager) &&
          (pathIsWithinPhysicalDirectory(path, "/usr/include") ||
           pathIsWithinPhysicalDirectory(path, "/usr/lib/gcc"));
      // An imported generic pragma macro can conceal a guard-stack string
      // from caller raw tokens. Observe its actual expansion as well. Native
      // headers retain their original ownership; caller -isystem is not proof.
      if (!physicalNative &&
          !cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
          (!file.isValid() || !isOwnedCudaCompilerInput(file, compiler))) {
        frontendHostMathGuardModified = true;
        originalCgControlChanged = true;
      }
    }
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
  const unsigned lineBegin = physicalLineBegin(buffer, offset);
  const auto lineEnd = buffer.find_first_of("\r\n", lineBegin);
  const auto physicalLine = buffer.slice(lineBegin, lineEnd);
  if (localHeaderContext != nullptr && localHeaderContext->inputEvidence != nullptr &&
      !hasExactDirectiveTokens(physicalLine, {"#", "pragma", "once"})) {
    // User line directives are excluded before this evidence can authorize
    // contextual pragma replay. Only governed flattening supplies #line maps.
    const auto presumed = sourceManager.getPresumedLoc(fileLocation);
    if (presumed.isInvalid()) {
      nvidiaSampleHelperUnsupportedMacroUse = true;
      return;
    }
    const std::string path = presumed.getFilename();
    contextualPragmaEvents += std::to_string(path.size()) + ":" + path + ":" +
        std::to_string(presumed.getLine()) + ":" +
        std::to_string(physicalLine.size()) + ":" + physicalLine.str() + "\n";
  }
  if (getCompilerInstance().getLangOpts().OpenMP && !Loc.isMacroID()) {
    if (sourceManager.isWrittenInMainFile(fileLocation) &&
        hasExactDirectiveTokens(physicalLine,
                                {"#", "pragma", "omp", "parallel"})) {
      nvidiaSampleOpenMPParallelPragmaLines.insert(lineBegin);
      return;
    }
    if (isFrozenOpenMPVariantPragma(physicalLine) &&
        isFrozenNativeOpenMPHeader(sourceManager, file))
      return;
  }
  const bool packOpen = hasExactDirectiveTokens(physicalLine, {"#", "pragma", "pack", "(", "1", ")"});
  const bool packReset = hasExactDirectiveTokens(physicalLine, {"#", "pragma", "pack", "(", ")"});
  if (packOpen || packReset) {
    const bool selected = sourceManager.isWrittenInMainFile(fileLocation) ||
        (localHeaderContext != nullptr && localHeaderContext->plan != nullptr &&
         localHeaderContext->plan->isSelectedRootHeader(sourceManager.getFilename(fileLocation).str()));
    const bool governedVirtual = localHeaderContext != nullptr &&
        localHeaderContext->virtualInput != nullptr && sourceManager.isWrittenInMainFile(fileLocation) &&
        buffer == *localHeaderContext->virtualInput;
    if (packedRecords.pragma(getCompilerInstance(), fileLocation, packOpen, packReset,
                             selected, governedVirtual, unmodifiedHostInput)) return;
  }
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

bool AscifyAction::rewriteFrontendHostMath(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *reference =
      Result.Nodes.getNodeAs<clang::DeclRefExpr>(sFrontendHostMath);
  if (!frontendCompatibility.enabled() || reference == nullptr ||
      Result.Context == nullptr || Result.SourceManager == nullptr)
    return false;
  const auto *callee = llvm::dyn_cast<clang::FunctionDecl>(reference->getDecl());
  if (callee == nullptr ||
      (callee->getQualifiedNameAsString() != "max" &&
       callee->getQualifiedNameAsString() != "min"))
    return false;
  const bool isMaximum = callee->getName() == "max";
  const llvm::StringRef sourceName = isMaximum ? "max" : "min";
  const bool isFloatMaximum = isMaximum &&
      callee->getReturnType()->isSpecificBuiltinType(clang::BuiltinType::Float);
  const auto hasParameterType = [&](unsigned index, clang::BuiltinType::Kind type) {
    return callee->getParamDecl(index)->getType()->isSpecificBuiltinType(type);
  };
  const bool isMixedUnsignedMinimum = !isMaximum &&
      callee->getNumParams() == 2 &&
      callee->getReturnType()->isSpecificBuiltinType(clang::BuiltinType::UInt) &&
      ((hasParameterType(0, clang::BuiltinType::Int) &&
        hasParameterType(1, clang::BuiltinType::UInt)) ||
       (hasParameterType(0, clang::BuiltinType::UInt) &&
        hasParameterType(1, clang::BuiltinType::Int)));
  const auto argumentType = isFloatMaximum ? clang::BuiltinType::Float
                                           : clang::BuiltinType::Int;
  const llvm::StringRef targetName = isMixedUnsignedMinimum
      ? "ascify::cuda_host_min_unsigned" : isFloatMaximum ? "__builtin_fmaxf"
      : isMaximum ? "__builtin_elementwise_max" : "__builtin_elementwise_min";
  auto &sourceManager = *Result.SourceManager;
  const std::string expectedHeader =
      frontendCompatibility.canonicalRoot + "/host_math.h";
  if (sourceManager.getFilename(callee->getLocation()) != expectedHeader)
    return false;  // User and CUDA device overloads retain their identities.

  auto reject = [&]() {
    auto &diagnostics = getCompilerInstance().getDiagnostics();
    const unsigned id = diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Ascify frontend host %0 requires a direct, non-macro %1 "
        "call with no qualifier or global :: in a non-template host function; "
        "this use is not admitted");
    const llvm::StringRef domain = isFloatMaximum ? "float/float" :
        isMixedUnsignedMinimum ? "int/unsigned-int or unsigned-int/int" : "int/int";
    diagnostics.Report(reference->getLocation(), id) << sourceName << domain;
    return false;
  };
  // A using-declaration can expose the same product function as ns::max.
  // Replacing only its name would leave ns::__builtin_fmaxf, which is not a
  // builtin. Admit only unqualified max and the literal global ::max form.
  if (reference->hasQualifier()) {
    const auto qualifierRange = reference->getQualifierLoc().getSourceRange();
    if (qualifierRange.getBegin().isMacroID() ||
        qualifierRange.getEnd().isMacroID() ||
        clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(qualifierRange),
            sourceManager, getCompilerInstance().getLangOpts()) != "::")
      return reject();
  }
  auto &context = *Result.Context;
  // A local initializer has VarDecl between its expression and the owning
  // function. The general statement-only walker intentionally does not cross
  // that declaration boundary. Prove this host context without widening the
  // atomic/half2 rules that use that walker.
  const clang::FunctionDecl *function = nullptr;
  const clang::Stmt *ancestor = reference;
  std::set<const clang::Stmt *> visited;
  while (ancestor != nullptr && visited.insert(ancestor).second) {
    const auto parents = context.getParents(*ancestor);
    if (parents.size() != 1 || parents[0].get<clang::LambdaExpr>() != nullptr)
      break;
    if (const auto *owner = parents[0].get<clang::FunctionDecl>()) {
      function = owner;
      break;
    }
    if (const auto *variable = parents[0].get<clang::VarDecl>()) {
      if (variable->isLocalVarDecl())
        function = llvm::dyn_cast<clang::FunctionDecl>(variable->getDeclContext());
      break;
    }
    ancestor = parents[0].get<clang::Stmt>();
  }
  const auto *method = llvm::dyn_cast_or_null<clang::CXXMethodDecl>(function);
  if (frontendHostMathBuiltinMacroDefined || function == nullptr ||
      (method != nullptr && method->getParent()->isLambda()) ||
      function->hasAttr<clang::CUDADeviceAttr>() ||
      function->hasAttr<clang::CUDAGlobalAttr>() ||
      function->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      reference->hasExplicitTemplateArgs() ||
      callee->getNumParams() != 2 ||
      (!isMixedUnsignedMinimum &&
       !callee->getReturnType()->isSpecificBuiltinType(argumentType)))
    return reject();
  for (const auto *parameter : callee->parameters()) {
    if (!isMixedUnsignedMinimum &&
        !parameter->getType()->isSpecificBuiltinType(argumentType))
      return reject();
  }

  const clang::Expr *current = reference;
  const clang::CallExpr *call = nullptr;
  for (unsigned depth = 0; depth != 16; ++depth) {
    const auto parents = context.getParents(*current);
    if (parents.size() != 1)
      break;
    if (const auto *parentCall = parents[0].get<clang::CallExpr>()) {
      if (parentCall->getCallee()->IgnoreParenImpCasts() == reference &&
          parentCall->getDirectCallee() == callee)
        call = parentCall;
      break;
    }
    const auto *parentExpression = parents[0].get<clang::Expr>();
    if (parentExpression == nullptr ||
        (!llvm::isa<clang::ParenExpr>(parentExpression) &&
         !llvm::isa<clang::ImplicitCastExpr>(parentExpression)))
      break;
    current = parentExpression;
  }
  const clang::SourceLocation location = reference->getLocation();
  if (call == nullptr || call->getNumArgs() != 2 || location.isInvalid() ||
      location.isMacroID() || !sourceManager.isWrittenInMainFile(location))
    return reject();
  const auto &languageOptions = getCompilerInstance().getLangOpts();
  if (clang::Lexer::getSourceText(
          clang::CharSourceRange::getTokenRange(location, location),
          sourceManager, languageOptions) != sourceName)
    return reject();
  const unsigned offset = sourceManager.getFileOffset(location);
  if (!rewrittenFrontendHostMathOffsets.insert(offset).second)
    return true;
  const clang::SourceLocation end = clang::Lexer::getLocForEndOfToken(
      location, 0, sourceManager, languageOptions);
  // Builtins preserve CUDA value semantics without binding to user overloads.
  // Both substitutions leave the argument expressions evaluated exactly once.
  // The mixed minimum's finite unsigned/unsigned target converts the signed
  // value before comparison, matching CUDA's unsigned-returning overloads.
  const std::string replacementName = reference->hasQualifier()
      ? targetName.str() : "::" + targetName.str();
  const ct::Replacement replacement(sourceManager, location, 3,
                                     replacementName);
  if (!insertSemanticReplacement(replacement,
          clang::FullSourceLoc(location, sourceManager), location, end))
    return false;
  if (isMixedUnsignedMinimum)
    needsCudaCompatHeader = true;
  return true;
}

bool AscifyAction::rewriteCudaHalf2Operator(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *matched =
      Result.Nodes.getNodeAs<clang::CXXOperatorCallExpr>(sCudaHalf2Operator);
  if (matched == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr)
    return false;
  auto &sourceManager = *Result.SourceManager;
  const auto &languageOptions = getCompilerInstance().getLangOpts();
  auto proven = [&](const clang::CXXOperatorCallExpr *call) {
    if (call == nullptr || call->getNumArgs() != 2 ||
        (call->getOperator() != clang::OO_Plus &&
         call->getOperator() != clang::OO_Star))
      return false;
    const auto *callee = call->getDirectCallee();
    const auto *record = call->getType()->getAsCXXRecordDecl();
    if (callee == nullptr || record == nullptr ||
        (record->getQualifiedNameAsString() != "half2" &&
         record->getQualifiedNameAsString() != "__half2"))
      return false;
    for (const auto location : {callee->getLocation(), record->getLocation()}) {
      const auto header =
          llvm::sys::path::filename(sourceManager.getFilename(location));
      if (!sourceManager.isInSystemHeader(location) ||
          (header != "cuda_fp16.h" && header != "cuda_fp16.hpp"))
        return false;
    }
    return true;
  };
  if (!proven(matched))
    return false;
  const auto *function = enclosingNonLambdaFunction(matched, *Result.Context);
  if (function == nullptr ||
      (!function->hasAttr<clang::CUDADeviceAttr>() &&
       !function->hasAttr<clang::CUDAGlobalAttr>()))
    return false;
  if (function->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate) {
    auto &diagnostics = getCompilerInstance().getDiagnostics();
    const unsigned id = diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Ascify half2 operator lowering does not admit template functions; "
        "other instantiations must retain their arithmetic");
    diagnostics.Report(matched->getExprLoc(), id);
    return false;
  }

  // Rewrite the outermost admitted expression once, irrespective of matcher
  // visitation order. Native CUDA half2 operators round after each operation;
  // keeping explicit CANN x2 intrinsics prevents multiply/add contraction.
  const clang::CXXOperatorCallExpr *root = matched;
  const clang::Expr *current = matched;
  for (;;) {
    const auto parents = Result.Context->getParents(*current);
    if (parents.size() != 1)
      break;
    const auto *parent = parents[0].get<clang::Expr>();
    if (const auto *operation =
            llvm::dyn_cast_or_null<clang::CXXOperatorCallExpr>(parent)) {
      if (!proven(operation))
        break;
      root = operation;
    } else if (parent == nullptr ||
               (!llvm::isa<clang::ParenExpr>(parent) &&
                !llvm::isa<clang::ImplicitCastExpr>(parent) &&
                !llvm::isa<clang::MaterializeTemporaryExpr>(parent) &&
                !llvm::isa<clang::CXXBindTemporaryExpr>(parent) &&
                !(llvm::isa<clang::CXXConstructExpr>(parent) &&
                  llvm::cast<clang::CXXConstructExpr>(parent)
                      ->getConstructor()->isCopyOrMoveConstructor()))) {
      break;
    }
    current = parent;
  }

  class LeafProof : public clang::RecursiveASTVisitor<LeafProof> {
   public:
    explicit LeafProof(clang::ASTContext &context) : context_(context) {}
    bool TraversePseudoObjectExpr(clang::PseudoObjectExpr *expression) {
      // CUDA execution coordinates use compiler-owned property getter calls.
      // Prove the complete builtin/property identity before bypassing those
      // implicit calls; explicit user calls and other properties stay rejected.
      if (expression == nullptr || expression->getBeginLoc().isMacroID() ||
          expression->getEndLoc().isMacroID())
        return false;
      const auto *call = llvm::dyn_cast<clang::CallExpr>(
          expression->getResultExpr()->IgnoreParenImpCasts());
      const auto *method = call == nullptr ? nullptr :
          llvm::dyn_cast_or_null<clang::CXXMethodDecl>(call->getDirectCallee());
      if (method == nullptr || call->getNumArgs() != 0 ||
          !call->getType()->isSpecificBuiltinType(clang::BuiltinType::UInt) ||
          (method->getName() != "__fetch_builtin_x" &&
           method->getName() != "__fetch_builtin_y" &&
           method->getName() != "__fetch_builtin_z"))
        return false;
      const auto *property = llvm::dyn_cast<clang::MSPropertyRefExpr>(
          expression->getSyntacticForm()->IgnoreParenImpCasts());
      if (property == nullptr || property->isArrow() ||
          method->getName() != "__fetch_builtin_" +
                                   property->getPropertyDecl()->getName().str())
        return false;
      // Sema captures the property base in an OpaqueValueExpr. Only a direct
      // builtin variable (optionally parenthesized) is admitted, so a call or
      // macro hidden in a comma/property base cannot bypass the leaf checks.
      const clang::Expr *base = property->getBaseExpr();
      while (base != nullptr) {
        if (base->getBeginLoc().isMacroID() || base->getEndLoc().isMacroID())
          return false;
        if (const auto *opaque = llvm::dyn_cast<clang::OpaqueValueExpr>(base))
          base = opaque->getSourceExpr();
        else if (const auto *paren = llvm::dyn_cast<clang::ParenExpr>(base))
          base = paren->getSubExpr();
        else if (const auto *cast = llvm::dyn_cast<clang::ImplicitCastExpr>(base))
          base = cast->getSubExpr();
        else
          break;
      }
      const auto *baseReference =
          llvm::dyn_cast_or_null<clang::DeclRefExpr>(base);
      if (baseReference == nullptr)
        return false;
      class References : public clang::RecursiveASTVisitor<References> {
       public:
        std::set<const clang::VarDecl *> variables;
        bool VisitDeclRefExpr(clang::DeclRefExpr *reference) {
          if (const auto *variable =
                  llvm::dyn_cast<clang::VarDecl>(reference->getDecl()))
            variables.insert(variable->getCanonicalDecl());
          return true;
        }
      } references;
      references.TraverseStmt(expression);
      if (references.variables.size() != 1)
        return false;
      const auto *variable = *references.variables.begin();
      if (baseReference->getDecl()->getCanonicalDecl() != variable)
        return false;
      const auto name = variable->getName();
      if (name != "threadIdx" && name != "blockIdx" &&
          name != "blockDim" && name != "gridDim")
        return false;
      const auto *record = variable->getType()->getAsCXXRecordDecl();
      if (record == nullptr || record->getQualifiedNameAsString() !=
              "__cuda_builtin_" + name.str() + "_t" ||
          method->getParent()->getCanonicalDecl() != record->getCanonicalDecl())
        return false;
      auto &sourceManager = context_.getSourceManager();
      for (auto location : {variable->getLocation(), record->getLocation(),
                            method->getLocation(),
                            property->getPropertyDecl()->getLocation()}) {
        location = sourceManager.getExpansionLoc(location);
        if (!sourceManager.isInSystemHeader(location) ||
            llvm::sys::path::filename(sourceManager.getFilename(location)) !=
                "__clang_cuda_builtin_vars.h")
          return false;
      }
      return true;
    }
    bool VisitExpr(clang::Expr *expression) {
      return !expression->getBeginLoc().isMacroID() &&
             !expression->getEndLoc().isMacroID();
    }
    bool VisitCallExpr(clang::CallExpr *) { return false; }
    bool VisitCXXConstructExpr(clang::CXXConstructExpr *) { return false; }
    bool VisitDeclRefExpr(clang::DeclRefExpr *reference) {
      const auto name = reference->getDecl()->getName();
      const auto found = CUDA_RENAMES_MAP().find(name);
      return found == CUDA_RENAMES_MAP().end() || found->second.dppName == name;
    }
   private:
    clang::ASTContext &context_;
  };
  auto render = [&](auto &&self, const clang::Expr *expression,
                    std::string &result) -> bool {
    expression = expression->IgnoreParenImpCasts();
    if (const auto *temporary =
            llvm::dyn_cast<clang::MaterializeTemporaryExpr>(expression))
      return self(self, temporary->getSubExpr(), result);
    if (const auto *temporary =
            llvm::dyn_cast<clang::CXXBindTemporaryExpr>(expression))
      return self(self, temporary->getSubExpr(), result);
    if (const auto *copy = llvm::dyn_cast<clang::CXXConstructExpr>(expression)) {
      if (copy->getNumArgs() == 1 &&
          copy->getConstructor()->isCopyOrMoveConstructor())
        return self(self, copy->getArg(0), result);
      return false;
    }
    if (const auto *operation =
            llvm::dyn_cast<clang::CXXOperatorCallExpr>(expression)) {
      if (!proven(operation))
        return false;
      std::string left, right;
      if (!self(self, operation->getArg(0), left) ||
          !self(self, operation->getArg(1), right))
        return false;
      result = (operation->getOperator() == clang::OO_Plus ? "__haddx2(" :
                                                               "__hmulx2(") +
               left + ", " + right + ")";
      return true;
    }
    LeafProof proof(*Result.Context);
    if (!proof.TraverseStmt(const_cast<clang::Expr *>(expression)))
      return false;
    result = clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(expression->getSourceRange()),
        sourceManager, languageOptions).str();
    if (result.empty())
      return false;
    result = "(" + result + ")";
    return true;
  };
  const auto begin = root->getBeginLoc();
  const auto last = root->getEndLoc();
  auto reject = [&]() {
    auto &diagnostics = getCompilerInstance().getDiagnostics();
    const unsigned id = diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Ascify half2 arithmetic requires a direct SDK operator expression "
        "with non-macro operands and no additional calls or CUDA renames; "
        "separate half rounding could not be preserved");
    diagnostics.Report(root->getExprLoc(), id);
    return false;
  };
  if (begin.isInvalid() || last.isInvalid() || begin.isMacroID() ||
      last.isMacroID() || !sourceManager.isWrittenInMainFile(begin))
    return reject();
  std::string replacementText;
  if (!render(render, root, replacementText))
    return reject();
  const unsigned offset = sourceManager.getFileOffset(begin);
  if (!rewrittenCudaHalf2OperatorOffsets.insert(offset).second)
    return true;
  const auto end = clang::Lexer::getLocForEndOfToken(
      last, 0, sourceManager, languageOptions);
  const ct::Replacement replacement(sourceManager,
      clang::CharSourceRange::getCharRange(begin, end), replacementText);
  return insertSemanticReplacement(replacement,
      clang::FullSourceLoc(begin, sourceManager), begin, end);
}

// Only complete, non-const, trivially copyable file-scope CUDA device objects
// receive a resolver. This does not rewrite their declaration or consumer code.
// Reject references, templates, inline definitions and C++ const subobjects:
// those require linkage/folding proofs beyond this same-module address recipe.
static bool admitsDeviceSymbolStorage(clang::QualType type,
                                     clang::ASTContext &context,
                                     unsigned depth = 0) {
  if (depth > 32 || type.isNull() || type->isDependentType() ||
      type->isIncompleteType() || type.isConstQualified() ||
      type.isVolatileQualified() || type->isReferenceType()) return false;
  if (const auto *array = context.getAsConstantArrayType(type))
    return !array->getSize().isZero() &&
           admitsDeviceSymbolStorage(array->getElementType(), context, depth + 1);
  if (type->isArrayType()) return false;
  if (type->isArithmeticType() || type->isEnumeralType() || type->isPointerType())
    return true;
  const auto *recordType = type->getAs<clang::RecordType>();
  if (recordType == nullptr || !type.isTriviallyCopyableType(context) ||
      !type->isStandardLayoutType()) return false;
  const auto *record = recordType->getDecl()->getDefinition();
  if (record == nullptr || record->isUnion()) return false;
  if (const auto *cppRecord = llvm::dyn_cast<clang::CXXRecordDecl>(record))
    if (cppRecord->getNumBases() != 0) return false;
  for (const auto *field : record->fields())
    if (!admitsDeviceSymbolStorage(field->getType(), context, depth + 1)) return false;
  return true;
}

bool AscifyAction::registerCudaDeviceSymbol(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *variable = Result.Nodes.getNodeAs<clang::VarDecl>(sCudaDeviceSymbol);
  if (variable == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr || variable->isImplicit() ||
      variable->getIdentifier() == nullptr || !variable->hasGlobalStorage() ||
      !variable->getDeclContext()->isFileContext() || variable->isStaticDataMember() ||
      variable->getDescribedVarTemplate() != nullptr || variable->isInline() ||
      llvm::isa<clang::VarTemplateSpecializationDecl>(variable) ||
      variable->isThisDeclarationADefinition() == clang::VarDecl::DeclarationOnly ||
      (!variable->hasAttr<clang::CUDAConstantAttr>() &&
       !variable->hasAttr<clang::CUDADeviceAttr>()) ||
      variable->hasAttr<clang::CUDASharedAttr>() ||
      !admitsDeviceSymbolStorage(variable->getType(), *Result.Context)) return false;
  auto &source = *Result.SourceManager;
  // Qualified references prevent the generated parameter/local names from
  // shadowing the original object. Anonymous namespace lookup can become
  // ambiguous under absolute qualification, so retain that unproved case.
  for (const auto *owner = variable->getDeclContext(); owner != nullptr;
       owner = owner->getParent()) {
    if (const auto *space = llvm::dyn_cast<clang::NamespaceDecl>(owner))
      if (space->isAnonymousNamespace()) return false;
  }
  const auto name = variable->getLocation();
  const auto last = variable->getEndLoc();
  if (name.isInvalid() || last.isInvalid() || name.isMacroID() || last.isMacroID() ||
      !source.isWrittenInMainFile(name) || !source.isWrittenInMainFile(last)) return false;
  const auto &options = getCompilerInstance().getLangOpts();
  const auto next = clang::Lexer::findNextToken(last, source, options);
  if (!next || !next->is(clang::tok::semi) || next->getLocation().isMacroID()) return false;
  const auto insertion = clang::Lexer::getLocForEndOfToken(next->getLocation(), 0, source, options);
  if (insertion.isInvalid()) return false;
  const unsigned offset = source.getFileOffset(name);
  if (!registeredCudaSymbolOffsets.insert(offset).second) return true;
  const std::string tag = "ascify_generated_symbol_" + std::to_string(offset);
  // Generated identifiers must not bind to project declarations or macros,
  // including declarations produced by token pasting in a local header.
  auto &preprocessor = getCompilerInstance().getPreprocessor();
  for (const char *suffix : {"_address", "_resolve", "_registration"}) {
    auto *identifier = &Result.Context->Idents.get(tag + suffix);
    if (!variable->getDeclContext()->lookup(identifier).empty() ||
        preprocessor.getMacroDefinitionAtLoc(identifier, name).getMacroInfo() != nullptr) {
      needsCudaCompatHeader = true;
      cudaDeviceSymbolGeneratedNameConflict = true;
      registeredCudaSymbolOffsets.erase(offset);
      return false;
    }
  }
  auto *addressBuiltin = &Result.Context->Idents.get("__builtin_addressof");
  if (preprocessor.getMacroDefinitionAtLoc(addressBuiltin, name).getMacroInfo() != nullptr) {
    needsCudaCompatHeader = true;
    cudaDeviceSymbolGeneratedNameConflict = true;
    registeredCudaSymbolOffsets.erase(offset);
    return false;
  }
  const std::string text = "\nASCIFY_REGISTER_DEVICE_SYMBOL(::" +
      variable->getQualifiedNameAsString() + ", " + tag + ");\n";
  if (!insertReplacement(ct::Replacement(source, insertion, 0, text),
                         clang::FullSourceLoc(insertion, source))) {
    registeredCudaSymbolOffsets.erase(offset);
    return false;
  }
  needsCudaCompatHeader = true;
  static const dppCounter counter = {
      "CUDA device symbol resolver", CONV_MEMORY, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(counter, "CUDA device symbol resolver");
  return true;
}

bool AscifyAction::rewriteCudaHalf2DirectInit(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *variable =
      Result.Nodes.getNodeAs<clang::VarDecl>(sCudaHalf2DirectInit);
  if (variable == nullptr || Result.SourceManager == nullptr ||
      variable->isImplicit() || !variable->isLocalVarDecl() ||
      variable->getInitStyle() != clang::VarDecl::CallInit ||
      variable->getInit() == nullptr)
    return false;
  const auto *function =
      llvm::dyn_cast<clang::FunctionDecl>(variable->getDeclContext());
  if (function == nullptr ||
      (!function->hasAttr<clang::CUDADeviceAttr>() &&
       !function->hasAttr<clang::CUDAGlobalAttr>()))
    return false;

  const auto *construction = llvm::dyn_cast<clang::CXXConstructExpr>(
      variable->getInit()->IgnoreParenImpCasts());
  const auto *record = variable->getType()->getAsCXXRecordDecl();
  if (construction == nullptr || construction->getNumArgs() != 2 ||
      record == nullptr ||
      (record->getQualifiedNameAsString() != "half2" &&
       record->getQualifiedNameAsString() != "__half2"))
    return false;

  auto &sourceManager = *Result.SourceManager;
  const auto declarationLocation = record->getLocation();
  const llvm::StringRef header =
      llvm::sys::path::filename(sourceManager.getFilename(declarationLocation));
  if (!sourceManager.isInSystemHeader(declarationLocation) ||
      (header != "cuda_fp16.h" && header != "cuda_fp16.hpp"))
    return false;
  if (function->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate) {
    auto &diagnostics = getCompilerInstance().getDiagnostics();
    const unsigned id = diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Ascify half2 constructor lowering does not admit template functions");
    diagnostics.Report(variable->getLocation(), id);
    return false;
  }
  const clang::SourceLocation name = variable->getLocation();
  if (name.isInvalid() || name.isMacroID() ||
      !sourceManager.isWrittenInMainFile(name) ||
      construction->getBeginLoc().isMacroID() ||
      construction->getEndLoc().isMacroID())
    return false;
  for (const clang::Expr *argument : construction->arguments()) {
    if (argument->getBeginLoc().isMacroID() ||
        argument->getEndLoc().isMacroID())
      return false;
  }

  const auto &languageOptions = getCompilerInstance().getLangOpts();
  const auto nameEnd = clang::Lexer::getLocForEndOfToken(
      name, 0, sourceManager, languageOptions);
  clang::Token opening;
  if (nameEnd.isInvalid() ||
      clang::Lexer::getRawToken(nameEnd, opening, sourceManager,
                               languageOptions, true) ||
      opening.getKind() != clang::tok::l_paren)
    return false;
  const auto location = opening.getLocation();
  const unsigned offset = sourceManager.getFileOffset(location);
  if (!rewrittenCudaHalf2DirectInitOffsets.insert(offset).second)
    return true;
  // CANN's native half2 is a vector, not CUDA's class with a two-half
  // constructor. The half/half facade preserves component conversion and
  // copying. Replacing only '(' leaves argument evaluation and rewrites intact.
  const auto end = clang::Lexer::getLocForEndOfToken(
      location, 0, sourceManager, languageOptions);
  const ct::Replacement replacement(sourceManager, location, 1,
                                     " = ::ascify::make_half2_from_halves(");
  return insertSemanticReplacement(replacement,
      clang::FullSourceLoc(location, sourceManager), location, end);
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

bool AscifyAction::rewriteCudaLaneIdAsm(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *statement = Result.Nodes.getNodeAs<clang::GCCAsmStmt>(sCudaLaneIdAsm);
  if (statement == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr || statement->isVolatile() ||
      statement->isSimple() ||
      statement->getNumOutputs() != 1 || statement->getNumInputs() != 0 ||
      statement->getNumClobbers() != 0 ||
      !statement->getOutputName(0).empty() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
#if LLVM_VERSION_MAJOR >= 9
  if (statement->isAsmGoto())
    return false;
#endif
#if LLVM_VERSION_MAJOR >= 21
  // Clang 21 added constexpr asm expressions. This proof accepts only
  // ordinary string literals for both the instruction and its constraint.
  const auto *asmLiteral = llvm::dyn_cast_or_null<clang::StringLiteral>(
      statement->getAsmStringExpr());
  if (!llvm::isa_and_nonnull<clang::StringLiteral>(
          statement->getOutputConstraintExpr(0)))
    return false;
#else
  const auto *asmLiteral = statement->getAsmString();
#endif
  if (asmLiteral == nullptr ||
      asmLiteral->getString() != "mov.u32 %0, %%laneid;" ||
      statement->getOutputConstraint(0) != "=r")
    return false;
  auto &context = *Result.Context;
  auto &source = *Result.SourceManager;
  const auto *owner = enclosingNonLambdaFunction(statement, context);
  if (owner == nullptr || owner->hasAttr<clang::CUDAHostAttr>() ||
      (!owner->hasAttr<clang::CUDADeviceAttr>() &&
       !owner->hasAttr<clang::CUDAGlobalAttr>()) ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      owner->isDependentContext())
    return false;
  if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(owner))
    if (method->getParent()->isLambda())
      return false;
  const auto *output = statement->getOutputExpr(0);
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(output->IgnoreParens());
  const auto *variable = reference == nullptr ? nullptr :
      llvm::dyn_cast<clang::VarDecl>(reference->getDecl());
  if (variable == nullptr || !variable->isLocalVarDecl() ||
      !variable->hasLocalStorage() || variable->getDeclContext() != owner ||
      variable->getTLSKind() != clang::VarDecl::TLS_None ||
      variable->hasAttr<clang::CUDASharedAttr>() ||
      variable->hasAttr<clang::CUDAConstantAttr>() ||
      variable->hasAttr<clang::CUDADeviceAttr>() ||
      variable->getType().isVolatileQualified() ||
      variable->getType().isConstQualified() ||
      !context.hasSameType(variable->getType(), context.UnsignedIntTy) ||
      context.getTypeSize(variable->getType()) != 32 ||
      output->isTypeDependent() || output->isValueDependent())
    return false;
  const auto begin = statement->getBeginLoc();
  const auto last = statement->getEndLoc();
  if (begin.isInvalid() || last.isInvalid() || begin.isMacroID() ||
      last.isMacroID() || !source.isWrittenInMainFile(begin) ||
      !source.isWrittenInMainFile(last))
    return false;
  const auto &options = getCompilerInstance().getLangOpts();
  const auto end = clang::Lexer::getLocForEndOfToken(last, 0, source, options);
  if (end.isInvalid())
    return false;
  const unsigned firstOffset = source.getFileOffset(begin);
  const unsigned endOffset = source.getFileOffset(end);
  const auto file = source.getFileID(begin);
  // AST endpoints do not expose macro-expanded punctuation or a concatenated
  // string token. Retain any statement containing any macro invocation.
  for (const auto &macro : cudaLaneIdMacroRanges)
    if (macro.file == file && firstOffset < macro.endOffset &&
        macro.beginOffset < endOffset)
      return false;
  bool invalid = false;
  const auto spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(output->getSourceRange()),
      source, options, &invalid);
  if (invalid || spelling.empty())
    return false;
  if (!rewrittenCudaLaneIdOffsets.insert(firstOffset).second)
    return true;
  ct::Replacement edit(source, begin, endOffset - firstOffset,
                       "(" + spelling.str() + ") = ::ascify::cuda_lane_id()");
  if (!insertSemanticReplacement(edit, clang::FullSourceLoc(begin, source),
                                 begin, end)) {
    rewrittenCudaLaneIdOffsets.erase(firstOffset);
    return false;
  }
  needsCudaCompatHeader = true;
  static const dppCounter counter = {
      "CUDA lane ID", CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(counter, "CUDA lane ID instruction");
  return true;
}

bool AscifyAction::rewriteCudaPackedSadAsm(
    const mat::MatchFinder::MatchResult &Result) {
  // The existing matcher observes all main-buffer asm statements. This rule
  // recognizes the instruction and operand proof, never a wrapper's name.
  const auto *statement = Result.Nodes.getNodeAs<clang::GCCAsmStmt>(sCudaLaneIdAsm);
  if (statement == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr || statement->isVolatile() ||
      statement->isSimple() || statement->getNumOutputs() != 1 ||
      statement->getNumInputs() != 3 || statement->getNumClobbers() != 0 ||
      !statement->getOutputName(0).empty() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
#if LLVM_VERSION_MAJOR >= 9
  if (statement->isAsmGoto()) return false;
#endif
#if LLVM_VERSION_MAJOR >= 21
  const auto *literal = llvm::dyn_cast_or_null<clang::StringLiteral>(
      statement->getAsmStringExpr());
  if (!llvm::isa_and_nonnull<clang::StringLiteral>(
          statement->getOutputConstraintExpr(0))) return false;
#else
  const auto *literal = statement->getAsmString();
#endif
  if (literal == nullptr || literal->getString() !=
      "vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;" ||
      statement->getOutputConstraint(0) != "=r") return false;
  for (unsigned index = 0; index < 3; ++index) {
#if LLVM_VERSION_MAJOR >= 21
    if (!llvm::isa_and_nonnull<clang::StringLiteral>(
            statement->getInputConstraintExpr(index))) return false;
#endif
    if (!statement->getInputName(index).empty() ||
        statement->getInputConstraint(index) != "r") return false;
  }
  auto &context = *Result.Context;
  auto &source = *Result.SourceManager;
  const auto *owner = enclosingNonLambdaFunction(statement, context);
  if (owner == nullptr || owner->hasAttr<clang::CUDAHostAttr>() ||
      (!owner->hasAttr<clang::CUDADeviceAttr>() &&
       !owner->hasAttr<clang::CUDAGlobalAttr>()) ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      owner->isDependentContext() || context.getTypeSize(context.UnsignedIntTy) != 32)
    return false;
  if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(owner))
    if (method->getParent()->isLambda()) return false;
  const auto privateValue = [&](const clang::VarDecl *variable, bool input) {
    return variable != nullptr &&
        (variable->isLocalVarDecl() ||
         (input && llvm::isa<clang::ParmVarDecl>(variable))) &&
        variable->hasLocalStorage() && variable->getDeclContext() == owner &&
        variable->getStorageClass() == clang::SC_None && !variable->hasAttrs() &&
        variable->getTLSKind() == clang::VarDecl::TLS_None &&
        !variable->hasAttr<clang::CUDASharedAttr>() &&
        !variable->hasAttr<clang::CUDAConstantAttr>() &&
        !variable->hasAttr<clang::CUDADeviceAttr>() &&
        !variable->getType().isVolatileQualified() &&
        (input || !variable->getType().isConstQualified()) &&
        context.hasSameType(variable->getType().getUnqualifiedType(),
                            context.UnsignedIntTy);
  };
  const auto *output = statement->getOutputExpr(0);
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(output->IgnoreParens());
  if (reference == nullptr ||
      !privateValue(llvm::dyn_cast<clang::VarDecl>(reference->getDecl()), false) ||
      output->isTypeDependent() || output->isValueDependent()) return false;
  for (unsigned index = 0; index < 3; ++index) {
    const auto *input = statement->getInputExpr(index);
    if (input->isTypeDependent() || input->isValueDependent() ||
        !context.hasSameType(input->getType().getUnqualifiedType(),
                             context.UnsignedIntTy)) return false;
    input = input->IgnoreParenImpCasts();
    if (llvm::isa<clang::IntegerLiteral>(input)) continue;
    const auto *value = llvm::dyn_cast<clang::DeclRefExpr>(input);
    if (value == nullptr ||
        !privateValue(llvm::dyn_cast<clang::VarDecl>(value->getDecl()), true))
      return false;
  }
  const auto begin = statement->getBeginLoc();
  const auto last = statement->getEndLoc();
  if (begin.isInvalid() || last.isInvalid() || begin.isMacroID() ||
      last.isMacroID() || !source.isWrittenInMainFile(begin) ||
      !source.isWrittenInMainFile(last)) return false;
  const auto &options = getCompilerInstance().getLangOpts();
  const auto end = clang::Lexer::getLocForEndOfToken(last, 0, source, options);
  if (end.isInvalid()) return false;
  const unsigned firstOffset = source.getFileOffset(begin);
  const unsigned endOffset = source.getFileOffset(end);
  const auto file = source.getFileID(begin);
  for (const auto &macro : cudaLaneIdMacroRanges)
    if (macro.file == file && firstOffset < macro.endOffset &&
        macro.beginOffset < endOffset) return false;
  // Active AST literals omit inactive branches and directive side effects.
  // Replacing the whole statement must not erase either. Inspect all physical
  // tokens, including digraph directives and inactive pragma operators.
  bool invalidBuffer = false;
  const auto buffer = source.getBufferData(file, &invalidBuffer);
  if (invalidBuffer || firstOffset >= endOffset || endOffset > buffer.size())
    return false;
  clang::Lexer lexer(source.getLocForStartOfFile(file), options,
                     buffer.begin(), buffer.begin() + firstOffset, buffer.end());
  clang::Token token;
  for (;;) {
    lexer.LexFromRawLexer(token);
    if (source.getFileOffset(token.getLocation()) >= endOffset) break;
    if (token.is(clang::tok::eof) || token.is(clang::tok::hash) ||
        token.is(clang::tok::hashhash)) return false;
    if (token.is(clang::tok::raw_identifier)) {
      // Escaped newlines can spell a pragma operator without its literal
      // raw bytes. The admitted grammar needs no cleaned identifiers.
      if (token.needsCleaning() || token.getRawIdentifier() == "_Pragma" ||
          token.getRawIdentifier() == "__pragma") return false;
      // Also reject escaped/unicode spellings instead of guessing their
      // preprocessor identifier identity from physical bytes.
      for (const char c : token.getRawIdentifier())
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_')) return false;
    }
  }
  std::vector<std::string> operands;
  for (const auto *expression : {output, statement->getInputExpr(0),
                                statement->getInputExpr(1), statement->getInputExpr(2)}) {
    bool invalid = false;
    const auto spelling = clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(expression->getSourceRange()),
        source, options, &invalid);
    if (invalid || spelling.empty()) return false;
    operands.push_back(spelling.str());
  }
  if (!rewrittenCudaPackedSadOffsets.insert(firstOffset).second) return true;
  const std::string replacement = "(" + operands[0] +
      ") = ::ascify::packed_u8_sad4_add((" + operands[1] + "), (" +
      operands[2] + "), (" + operands[3] + "))";
  ct::Replacement edit(source, begin, endOffset - firstOffset, replacement);
  if (!insertSemanticReplacement(edit, clang::FullSourceLoc(begin, source), begin, end)) {
    rewrittenCudaPackedSadOffsets.erase(firstOffset);
    return false;
  }
  needsCudaCompatHeader = true;
  static const dppCounter counter = {
      "CUDA packed byte SAD", CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(counter, "CUDA packed byte SAD instruction");
  return true;
}

bool AscifyAction::rewriteCudaSmDiagnostic(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *call =
      Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr || call->getNumArgs() < 3 ||
      call->isTypeDependent() || call->isValueDependent() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
  auto &context = *Result.Context;
  auto &source = *Result.SourceManager;
  const auto *callee = call->getDirectCallee();
  if (callee == nullptr || callee->getQualifiedNameAsString() != "printf" ||
      !callee->isVariadic() || callee->getNumParams() != 1 ||
      !context.hasSameType(callee->getReturnType(), context.IntTy) ||
      callee->hasBody())
    return false;
  // A caller's -isystem annotation or printf spelling is not libc ownership.
  // This bounded rule admits the physical Linux libc declaration only.
  for (const auto *declaration : callee->redecls()) {
    const auto location = source.getSpellingLoc(declaration->getLocation());
    if (location.isInvalid() || !source.isInSystemHeader(location))
      return false;
    const auto file = source.getFileID(location);
    const auto *entry = source.getFileEntryForID(file);
    if (entry == nullptr || source.isFileOverridden(entry) ||
        contextualCanonicalPath(source.getFilename(location)) !=
            "/usr/include/stdio.h")
      return false;
  }
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(
      call->getCallee()->IgnoreParenImpCasts());
  if (reference == nullptr || reference->hasQualifier())
    return false;
  const auto *owner = enclosingNonLambdaFunction(call, context);
  if (owner == nullptr || owner->hasAttr<clang::CUDADeviceAttr>() ||
      owner->hasAttr<clang::CUDAGlobalAttr>() || owner->isDependentContext() ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate)
    return false;
  const auto parents = context.getParents(*call);
  // The new diagnostic has a different byte count. Never rewrite a used
  // printf result, a conditional expression or another nested call.
  if (parents.size() != 1 || parents[0].get<clang::CompoundStmt>() == nullptr)
    return false;
  const auto *literal = llvm::dyn_cast<clang::StringLiteral>(
      call->getArg(0)->IgnoreParenImpCasts());
  if (literal == nullptr || literal->getCharByteWidth() != 1)
    return false;
  const auto format = literal->getString();
  if (format.contains('\0') ||
      (!format.contains("SM ") && !format.contains("Compute ") &&
       !format.contains("compute capability")))
    return false;

  struct Conversion { size_t offset; char kind; };
  std::vector<Conversion> conversions;
  for (size_t index = 0; index < format.size(); ++index) {
    if (format[index] != '%')
      continue;
    if (++index == format.size())
      return false;
    if (format[index] == '%')
      continue;
    // Deliberately exclude positional, width, precision, length, star and
    // write-through %n formats. Each supported conversion takes one argument.
    if (format[index] != 'd' && format[index] != 's')
      return false;
    conversions.push_back({index - 1, format[index]});
  }
  if (conversions.size() + 1 != call->getNumArgs())
    return false;

  auto localMember = [&](const clang::Expr *expression,
                         llvm::StringRef name) -> const clang::VarDecl * {
    const auto *member = llvm::dyn_cast<clang::MemberExpr>(
        expression->IgnoreParenImpCasts());
    if (member == nullptr || member->isArrow() ||
        member->getMemberDecl()->getName() != name ||
        member->getType().isVolatileQualified() ||
        !context.hasSameType(member->getType().getUnqualifiedType(), context.IntTy))
      return nullptr;
    const auto *field = llvm::dyn_cast<clang::FieldDecl>(member->getMemberDecl());
    if (field == nullptr || field->getParent()->getQualifiedNameAsString() !=
                                "cudaDeviceProp")
      return nullptr;
    const auto location = source.getSpellingLoc(field->getLocation());
    if (location.isInvalid() ||
        !cudaRuntimeApiSdkFileIds.count(source.getFileID(location).getHashValue()))
      return nullptr;
    const auto *base = llvm::dyn_cast<clang::DeclRefExpr>(
        member->getBase()->IgnoreParenImpCasts());
    const auto *variable = base == nullptr ? nullptr :
        llvm::dyn_cast<clang::VarDecl>(base->getDecl());
    if (variable == nullptr || !variable->isLocalVarDecl() ||
        !variable->hasLocalStorage() || variable->getDeclContext() != owner ||
        variable->getTLSKind() != clang::VarDecl::TLS_None ||
        variable->getType()->isReferenceType() ||
        variable->getType().isVolatileQualified())
      return nullptr;
    return variable;
  };
  // Attribute-query diagnostics use independent integer locals. Require the
  // complete use set of each local to be one owned query and this one display;
  // neither names nor initial zero values establish a capability value.
  auto localAttribute = [&](const clang::Expr *expression, llvm::StringRef name)
      -> const clang::Expr * {
    const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(
        expression->IgnoreParenImpCasts());
    const auto *variable = reference == nullptr ? nullptr :
        llvm::dyn_cast<clang::VarDecl>(reference->getDecl());
    if (variable == nullptr || !variable->isLocalVarDecl() ||
        !variable->hasLocalStorage() || variable->getDeclContext() != owner ||
        variable->getTLSKind() != clang::VarDecl::TLS_None ||
        !context.hasSameType(variable->getType(), context.IntTy))
      return nullptr;
    const clang::Expr *device = nullptr;
    unsigned reads = 0, queries = 0;
    class Uses : public clang::RecursiveASTVisitor<Uses> {
    public:
      std::function<bool(clang::DeclRefExpr *)> check;
      bool VisitDeclRefExpr(clang::DeclRefExpr *use) { return check(use); }
      const clang::VarDecl *variable = nullptr;
      bool VisitLambdaExpr(clang::LambdaExpr *lambda) {
        for (const auto &capture : lambda->captures())
          if (capture.capturesVariable() && capture.getCapturedVar() == variable)
            return false;
        return true;
      }
    } uses;
    uses.variable = variable;
    uses.check = [&](clang::DeclRefExpr *use) {
      if (use->getDecl() != variable) return true;
      const clang::Expr *current = use;
      auto parents = context.getParents(*current);
      while (parents.size() == 1) {
        if (const auto *paren = parents[0].get<clang::ParenExpr>()) current = paren;
        else if (const auto *cast = parents[0].get<clang::ImplicitCastExpr>()) {
          if (cast->getCastKind() != clang::CK_LValueToRValue) return false;
          current = cast;
        } else break;
        parents = context.getParents(*current);
      }
      if (parents.size() != 1) return false;
      if (parents[0].get<clang::CallExpr>() == call) {
        ++reads;
        return true;
      }
      const auto *address = parents[0].get<clang::UnaryOperator>();
      if (address == nullptr || address->getOpcode() != clang::UO_AddrOf)
        return false;
      const auto addressParents = context.getParents(*address);
      const auto *query = addressParents.size() == 1 ?
          addressParents[0].get<clang::CallExpr>() : nullptr;
      const auto *function = query == nullptr ? nullptr : query->getDirectCallee();
      if (function == nullptr || query->getNumArgs() != 3 ||
          query->getArg(0) != address || function->hasBody() ||
          function->getQualifiedNameAsString() != "cudaDeviceGetAttribute")
        return false;
      for (const auto *declaration : function->redecls()) {
        const auto location = source.getSpellingLoc(declaration->getLocation());
        if (location.isInvalid() || !cudaRuntimeApiSdkFileIds.count(
                source.getFileID(location).getHashValue())) return false;
      }
      const auto *attribute = llvm::dyn_cast<clang::DeclRefExpr>(
          query->getArg(1)->IgnoreParenImpCasts());
      const auto *enumerator = attribute == nullptr ? nullptr :
          llvm::dyn_cast<clang::EnumConstantDecl>(attribute->getDecl());
      if (enumerator == nullptr || enumerator->getName() != name ||
          !cudaRuntimeApiSdkFileIds.count(source.getFileID(
              source.getSpellingLoc(enumerator->getLocation())).getHashValue()))
        return false;
      const auto queryParents = context.getParents(*query);
      // Both queries and the display must be direct statements in the same
      // block, with the query text preceding the display.
      if (queryParents.size() != 1 ||
          queryParents[0].get<clang::CompoundStmt>() == nullptr ||
          queryParents[0].get<clang::CompoundStmt>() !=
              context.getParents(*call)[0].get<clang::CompoundStmt>())
        return false;
      if (query->getBeginLoc().isMacroID() || query->getEndLoc().isMacroID() ||
          !source.isWrittenInMainFile(query->getBeginLoc()) ||
          !source.isBeforeInTranslationUnit(query->getEndLoc(), call->getBeginLoc()))
        return false;
      device = query->getArg(2)->IgnoreParenImpCasts();
      if (device->getType().isVolatileQualified() ||
          (!llvm::isa<clang::IntegerLiteral>(device) &&
           !llvm::isa<clang::DeclRefExpr>(device))) return false;
      ++queries;
      return true;
    };
    if (!uses.TraverseStmt(owner->getBody()) || reads != 1 || queries != 1)
      return nullptr;
    return device;
  };
  size_t pair = conversions.size();
  const clang::VarDecl *property = nullptr;
  for (size_t index = 0; index + 1 < conversions.size(); ++index) {
    if (conversions[index].kind != 'd' || conversions[index + 1].kind != 'd' ||
        conversions[index + 1].offset != conversions[index].offset + 3 ||
        format[conversions[index].offset + 2] != '.')
      continue;
    const auto *major = localMember(call->getArg(index + 1), "major");
    const auto *minor = localMember(call->getArg(index + 2), "minor");
    if (major == nullptr || major != minor) {
      const auto *majorDevice = localAttribute(call->getArg(index + 1),
          "cudaDevAttrComputeCapabilityMajor");
      const auto *minorDevice = localAttribute(call->getArg(index + 2),
          "cudaDevAttrComputeCapabilityMinor");
      if (majorDevice == nullptr || minorDevice == nullptr) continue;
      const auto *lhsRef = llvm::dyn_cast<clang::DeclRefExpr>(majorDevice);
      const auto *rhsRef = llvm::dyn_cast<clang::DeclRefExpr>(minorDevice);
      const auto *lhsInt = llvm::dyn_cast<clang::IntegerLiteral>(majorDevice);
      const auto *rhsInt = llvm::dyn_cast<clang::IntegerLiteral>(minorDevice);
      if (!((lhsRef && rhsRef && lhsRef->getDecl() == rhsRef->getDecl()) ||
            (lhsInt && rhsInt && lhsInt->getValue() == rhsInt->getValue())))
        continue;
    }
    if (pair != conversions.size())
      return false;
    pair = index;
    property = major;
  }
  if (pair == conversions.size())
    return false;
  // Remaining expressions must not contain nested API rewrites or effects.
  for (unsigned index = 1; index < call->getNumArgs(); ++index) {
    if (index == pair + 1 || index == pair + 2)
      continue;
    const auto *expression = call->getArg(index)->IgnoreParenImpCasts();
    if (expression->HasSideEffects(context) ||
        expression->getType().isVolatileQualified())
      return false;
    if (const auto *member = llvm::dyn_cast<clang::MemberExpr>(expression)) {
      if (member->isArrow() ||
          !llvm::isa<clang::DeclRefExpr>(member->getBase()->IgnoreParenImpCasts()) ||
          member->getBase()->getType().isVolatileQualified())
        return false;
    } else if (!llvm::isa<clang::DeclRefExpr>(expression) &&
               !llvm::isa<clang::IntegerLiteral>(expression) &&
               !llvm::isa<clang::StringLiteral>(expression)) {
      return false;
    }
  }
  const auto begin = call->getBeginLoc();
  const auto last = call->getEndLoc();
  if (begin.isInvalid() || last.isInvalid() || begin.isMacroID() ||
      last.isMacroID() || !source.isWrittenInMainFile(begin) ||
      !source.isWrittenInMainFile(last))
    return false;
  const auto &options = getCompilerInstance().getLangOpts();
  const auto end = clang::Lexer::getLocForEndOfToken(last, 0, source, options);
  if (end.isInvalid())
    return false;
  const auto firstOffset = source.getFileOffset(begin);
  const auto endOffset = source.getFileOffset(end);
  const auto file = source.getFileID(begin);
  for (const auto &macro : cudaLaneIdMacroRanges)
    if (macro.file == file && firstOffset < macro.endOffset &&
        macro.beginOffset < endOffset)
      return false;

  std::string diagnostic = format.str();
  diagnostic.replace(conversions[pair].offset, 5, "N/A (Ascend)");
  // Octal escapes are exactly three digits, so adjacent decimal/hexadecimal
  // text cannot extend an escape. Preserve all other bytes, including UTF-8.
  std::string replacement = "printf(\"";
  for (unsigned char byte : diagnostic) {
    if (byte == '\\' || byte == '"') {
      replacement += '\\'; replacement += static_cast<char>(byte);
    } else if (byte >= 32 && byte <= 126) {
      replacement += static_cast<char>(byte);
    } else {
      replacement += '\\';
      replacement += static_cast<char>('0' + ((byte >> 6) & 7));
      replacement += static_cast<char>('0' + ((byte >> 3) & 7));
      replacement += static_cast<char>('0' + (byte & 7));
    }
  }
  replacement += '"';
  for (unsigned index = 1; index < call->getNumArgs(); ++index) {
    if (index == pair + 1 || index == pair + 2)
      continue;
    bool invalid = false;
    const auto spelling = clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(call->getArg(index)->getSourceRange()),
        source, options, &invalid);
    if (invalid || spelling.empty())
      return false;
    replacement += ", " + spelling.str();
  }
  replacement += ')';
  if (rewrittenCudaSmDiagnosticOffsets.count(firstOffset))
    return true;

  // Some callers initialize these two fields before querying the device.
  // Remove only dead literal-zero stores when every use of the entire local
  // object is this display, a store, or an owned output-only property query.
  // In particular, passing even another member's address can expose the whole
  // record; copies, aliases, captures and any other observations retain stores.
  std::vector<const clang::BinaryOperator *> deadStores;
  if (property != nullptr) {
    class Uses : public clang::RecursiveASTVisitor<Uses> {
    public:
      std::function<bool(clang::DeclRefExpr *)> check;
      const clang::VarDecl *variable = nullptr;
      bool VisitDeclRefExpr(clang::DeclRefExpr *use) { return check(use); }
      bool VisitLambdaExpr(clang::LambdaExpr *lambda) {
        for (const auto &capture : lambda->captures())
          if (capture.capturesVariable() && capture.getCapturedVar() == variable)
            return false;
        return true;
      }
    } uses;
    uses.variable = property;
    uses.check = [&](clang::DeclRefExpr *use) {
      if (use->getDecl() != property) return true;
      const clang::Expr *current = use;
      auto parents = context.getParents(*current);
      while (parents.size() == 1 && parents[0].get<clang::ParenExpr>()) {
        current = parents[0].get<clang::ParenExpr>();
        parents = context.getParents(*current);
      }
      if (parents.size() != 1) return false;
      if (const auto *address = parents[0].get<clang::UnaryOperator>()) {
        if (address->getOpcode() != clang::UO_AddrOf) return false;
        const auto addressParents = context.getParents(*address);
        const auto *query = addressParents.size() == 1 ?
            addressParents[0].get<clang::CallExpr>() : nullptr;
        const auto *function = query == nullptr ? nullptr : query->getDirectCallee();
        if (function == nullptr || function->hasBody() ||
            function->getQualifiedNameAsString() != "cudaGetDeviceProperties" ||
            query->getNumArgs() != 2 || query->getArg(0) != address)
          return false;
        for (const auto *declaration : function->redecls()) {
          const auto location = source.getSpellingLoc(declaration->getLocation());
          if (location.isInvalid() || !cudaRuntimeApiSdkFileIds.count(
                  source.getFileID(location).getHashValue())) return false;
        }
        return true;
      }
      const auto *member = parents[0].get<clang::MemberExpr>();
      if (member == nullptr || member->isArrow()) return false;
      const auto field = member->getMemberDecl()->getName();
      current = member;
      parents = context.getParents(*current);
      while (parents.size() == 1) {
        if (const auto *paren = parents[0].get<clang::ParenExpr>()) current = paren;
        else if (const auto *cast = parents[0].get<clang::ImplicitCastExpr>()) {
          if (cast->getCastKind() != clang::CK_LValueToRValue &&
              cast->getCastKind() != clang::CK_ArrayToPointerDecay) return false;
          current = cast;
        } else break;
        parents = context.getParents(*current);
      }
      if (parents.size() != 1) return false;
      if (parents[0].get<clang::CallExpr>() == call) return true;
      const auto *assignment = parents[0].get<clang::BinaryOperator>();
      if (assignment == nullptr || assignment->getOpcode() != clang::BO_Assign ||
          assignment->getLHS() != current || (field != "major" && field != "minor"))
        return false;
      const auto *zero = llvm::dyn_cast<clang::IntegerLiteral>(
          assignment->getRHS()->IgnoreParenImpCasts());
      if (zero == nullptr || !zero->getValue().isZero()) return false;
      const auto assignmentParents = context.getParents(*assignment);
      if (assignmentParents.size() != 1 ||
          assignmentParents[0].get<clang::CompoundStmt>() == nullptr) return false;
      deadStores.push_back(assignment);
      return true;
    };
    if (!uses.TraverseStmt(owner->getBody())) deadStores.clear();
  }

  // Publish the display and optional dead stores as one replacement transaction.
  ct::Replacements staged(*replacements);
  std::vector<SemanticRewriteRange> ranges;
  std::vector<ct::Replacement> edits;
  auto stage = [&](clang::SourceLocation first, clang::SourceLocation lastToken,
                   const std::string &text) {
    if (first.isInvalid() || lastToken.isInvalid() || first.isMacroID() ||
        lastToken.isMacroID() || !source.isWrittenInMainFile(first) ||
        !source.isWrittenInMainFile(lastToken)) return false;
    const auto after = clang::Lexer::getLocForEndOfToken(lastToken, 0, source, options);
    if (after.isInvalid() || source.getFileID(first) != source.getFileID(after)) return false;
    SemanticRewriteRange range{source.getFileID(first), source.getFileOffset(first),
                               source.getFileOffset(after)};
    if (range.endOffset <= range.beginOffset) return false;
    auto overlaps = [&](const SemanticRewriteRange &other) {
      return range.file == other.file && range.beginOffset < other.endOffset &&
             other.beginOffset < range.endOffset;
    };
    for (const auto &macro : cudaLaneIdMacroRanges) if (overlaps(macro)) return false;
    for (const auto &existing : SemanticRewriteRanges) if (overlaps(existing)) return false;
    for (const auto &pending : ranges) if (overlaps(pending)) return false;
    ct::Replacement edit(source, first, range.endOffset - range.beginOffset, text);
    if (!llcompat::insertReplacement(staged, edit)) return false;
    ranges.push_back(range);
    edits.push_back(edit);
    return true;
  };
  if (!stage(begin, last, replacement)) return false;
  for (const auto *assignment : deadStores)
    if (!stage(assignment->getBeginLoc(), assignment->getEndLoc(), "(void)0"))
      return false;
  *replacements = std::move(staged);
  SemanticRewriteRanges.insert(SemanticRewriteRanges.end(), ranges.begin(), ranges.end());
  rewrittenCudaSmDiagnosticOffsets.insert(firstOffset);
  if (PrintStats || PrintStatsCSV) {
    for (const auto &edit : edits) Statistics::current().bytesChanged(edit.getLength());
    Statistics::current().lineTouched(clang::FullSourceLoc(begin, source).getExpansionLineNumber());
  }
  static const dppCounter counter = {
      "CUDA SM diagnostic", CONV_DEVICE, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(counter, "CUDA SM display unavailable on Ascend");
  return true;
}

bool AscifyAction::rewriteCudaFloatFrexp(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *call = Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr || Result.SourceManager == nullptr ||
      call->getNumArgs() != 2 || call->isTypeDependent() || call->isValueDependent() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
  auto &context = *Result.Context;
  auto &source = *Result.SourceManager;
  const auto *callee = call->getDirectCallee();
  if (callee == nullptr || callee->isVariadic() || callee->getQualifiedNameAsString() != "frexp" ||
      callee->getNumParams() != 2 || !callee->hasAttr<clang::CUDADeviceAttr>() ||
      !context.hasSameType(callee->getReturnType(), context.FloatTy) ||
      !context.hasSameType(callee->getParamDecl(0)->getType().getUnqualifiedType(), context.FloatTy) ||
      !context.hasSameType(callee->getParamDecl(1)->getType().getUnqualifiedType(),
                           context.getPointerType(context.IntTy)) ||
      context.getTypeSize(context.FloatTy) != 32 || context.getTypeSize(context.IntTy) != 32)
    return false;
  for (const auto *declaration : callee->redecls()) {
    const auto location = source.getSpellingLoc(declaration->getLocation());
    if (location.isInvalid() || !source.isInSystemHeader(location)) return false;
    const auto file = source.getFileID(location);
    if (!cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
        !isOwnedCudaCompilerInput(file, getCompilerInstance())) return false;
  }
  const auto *owner = enclosingNvidiaFindDeviceFunction(call, context);
  if (owner == nullptr || owner->hasAttr<clang::CUDAHostAttr>() ||
      (!owner->hasAttr<clang::CUDADeviceAttr>() && !owner->hasAttr<clang::CUDAGlobalAttr>()) ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      owner->isDependentContext())
    return false;
  if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(owner))
    if (method->getParent()->isLambda()) return false;
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(call->getCallee()->IgnoreParenImpCasts());
  if (reference == nullptr || reference->hasQualifier() ||
      reference->getDecl()->getCanonicalDecl() != callee->getCanonicalDecl()) return false;
  const auto begin = call->getBeginLoc(), last = call->getEndLoc();
  const auto name = reference->getLocation();
  if (begin.isInvalid() || last.isInvalid() || name.isInvalid() ||
      begin.isMacroID() || last.isMacroID() || name.isMacroID() ||
      !source.isWrittenInMainFile(begin) || !source.isWrittenInMainFile(last) ||
      !source.isWrittenInMainFile(name)) return false;
  const auto &options = getCompilerInstance().getLangOpts();
  const auto end = clang::Lexer::getLocForEndOfToken(last, 0, source, options);
  const auto nameEnd = clang::Lexer::getLocForEndOfToken(name, 0, source, options);
  if (end.isInvalid() || nameEnd.isInvalid()) return false;
  const auto file = source.getFileID(begin);
  const unsigned first = source.getFileOffset(begin), limit = source.getFileOffset(end);
  for (const auto &macro : cudaLaneIdMacroRanges)
    if (macro.file == file && first < macro.endOffset && macro.beginOffset < limit)
      return false;
  const auto spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(name, name), source, options);
  if (spelling != "frexp") return false;
  const auto offset = source.getFileOffset(name);
  if (!rewrittenCudaFloatFrexpOffsets.insert(offset).second) return true;
  if (!insertSemanticReplacement(ct::Replacement(source, name, spelling.size(),
          "::ascify::cuda_frexp_float"), clang::FullSourceLoc(name, source), name, nameEnd)) {
    rewrittenCudaFloatFrexpOffsets.erase(offset);
    return false;
  }
  needsCudaCompatHeader = true;
  static const dppCounter counter = {
      "CUDA float frexp overload", CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(counter, "SDK float frexp(float, int*)");
  return true;
}

bool AscifyAction::preserveHostStandardMathCall(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *call = Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr || Result.SourceManager == nullptr ||
      call->getNumArgs() != 1 || call->isTypeDependent() || call->isValueDependent() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
  const auto *callee = call->getDirectCallee();
  if (callee == nullptr || callee->getIdentifier() == nullptr)
    return false;
  const auto name = callee->getName();
  // These are standard host overload sets. rsqrt is a CUDA extension and
  // deliberately keeps its separate device-API mapping.
  if (name != "exp" && name != "log" && name != "sqrt")
    return false;
  const auto qualified = callee->getQualifiedNameAsString();
  auto &source = *Result.SourceManager;
  if ((qualified != name && qualified != "std::" + name.str()) ||
      !allFunctionRedeclarationsComeFromInitiallyTrustedSystemFiles(
          callee, source, trustedSystemFileIds))
    return false;
  // The existing unique-parent proof crosses ordinary automatic initializers,
  // but rejects lambdas, static/TLS locals, globals and ambiguous parentage.
  // A host call must not suppress a shared host/device spelling's mapping.
  const auto *owner = enclosingNvidiaFindDeviceFunction(call, *Result.Context);
  if (owner == nullptr || owner->hasAttr<clang::CUDADeviceAttr>() ||
      owner->hasAttr<clang::CUDAGlobalAttr>() || owner->isDependentContext() ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate)
    return false;
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(
      call->getCallee()->IgnoreParenImpCasts());
  if (reference == nullptr || reference->hasQualifier() ||
      reference->getDecl()->getCanonicalDecl() != callee->getCanonicalDecl())
    return false;
  const auto location = reference->getLocation();
  if (location.isInvalid() || location.isMacroID() ||
      !source.isWrittenInMainFile(location))
    return false;
  const auto spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(location, location), source,
      getCompilerInstance().getLangOpts());
  if (spelling != name)
    return false;
  // No expression is replaced: preserve the exact source spelling, overload
  // resolution, argument conversions and evaluation count of this host call.
  preservedHostMathCalleeOffsets.insert(source.getFileOffset(location));
  return true;
}

bool AscifyAction::rewriteUnsupportedCudaSmAttribute(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *call = Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr || Result.SourceManager == nullptr ||
      call->getNumArgs() != 3 || call->isTypeDependent() || call->isValueDependent() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
  auto &source = *Result.SourceManager;
  const auto *callee = call->getDirectCallee();
  if (callee == nullptr || callee->getQualifiedNameAsString() != "cudaDeviceGetAttribute" ||
      !allFunctionRedeclarationsComeFromInitiallyTrustedSystemFiles(
          callee, source, trustedSystemFileIds))
    return false;
  const auto *owner = enclosingNonLambdaFunction(call, *Result.Context);
  if (owner == nullptr || owner->hasAttr<clang::CUDADeviceAttr>() ||
      owner->hasAttr<clang::CUDAGlobalAttr>() || owner->isDependentContext() ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate)
    return false;
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(
      call->getArg(1)->IgnoreParenImpCasts());
  const auto *selector = reference == nullptr ? nullptr :
      llvm::dyn_cast<clang::EnumConstantDecl>(reference->getDecl());
  if (selector == nullptr || reference->hasQualifier())
    return false;
  const auto name = selector->getName();
  if (name != "cudaDevAttrComputeCapabilityMajor" &&
      name != "cudaDevAttrComputeCapabilityMinor")
    return false;
  const auto declaration = source.getSpellingLoc(selector->getLocation());
  if (declaration.isInvalid() ||
      !cudaRuntimeApiSdkFileIds.count(source.getFileID(declaration).getHashValue()))
    return false;
  const auto begin = reference->getLocation();
  if (begin.isInvalid() || begin.isMacroID() || !source.isWrittenInMainFile(begin))
    return false;
  const auto &options = getCompilerInstance().getLangOpts();
  const auto end = clang::Lexer::getLocForEndOfToken(begin, 0, source, options);
  const auto spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(begin, begin), source, options);
  if (end.isInvalid() || spelling != name)
    return false;
  // Only this verified API operand becomes a nonnumeric selector. All enum
  // reflection/arithmetic/assignments elsewhere retain their CUDA spelling.
  ct::Replacement edit(source, begin, spelling.size(), "::ascify::" + name.str());
  if (!insertSemanticReplacement(edit, clang::FullSourceLoc(begin, source), begin, end))
    return false;
  needsCudaCompatHeader = true;
  return true;
}

// A launch-bound proof describes one concrete kernel and one nonescaping
// output object. It does not turn the partial adapter into full CUDA metadata.
bool AscifyAction::trustedKernelBoundQualifier(llvm::StringRef name,
                                                clang::SourceLocation location) {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto &preprocessor = compiler.getPreprocessor();
  std::set<const clang::IdentifierInfo *> seen;
  std::vector<const clang::IdentifierInfo *> pending = {preprocessor.getIdentifierInfo(name)};
  while (!pending.empty()) {
    const auto *identifier = pending.back(); pending.pop_back();
    if (!seen.insert(identifier).second) continue;
    if (seen.size() > 16) return false;
    const auto *info = preprocessor.getMacroDefinitionAtLoc(identifier, location).getMacroInfo();
    if (!info) return false;
    const auto file = source.getFileID(source.getSpellingLoc(info->getDefinitionLoc()));
    if (file.isInvalid() || (!cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
                            !isOwnedCudaCompilerInput(file, compiler))) return false;
    for (const auto &token : info->tokens()) {
      if (token.is(clang::tok::hash) || token.is(clang::tok::hashhash)) return false;
      const auto *dependency = token.getIdentifierInfo();
      if (!dependency) continue;
      bool parameter = false;
      if (info->isFunctionLike())
        for (const auto *argument : info->params()) parameter |= argument == dependency;
      if (!parameter && preprocessor.getMacroDefinitionAtLoc(dependency, location))
        pending.push_back(dependency);
    }
  }
  return true;
}

bool AscifyAction::stageKernelThreadBoundFact(const ascify::PureSimtKernelThreadBound &fact) {
  if (!fact.policyQualifier.isValid()) return true;
  auto &source = getCompilerInstance().getSourceManager();
  if (source.isWrittenInMainFile(fact.policyQualifier)) return true;
  if (!localHeaderContext || localHeaderContext->depth != 0 || !localHeaderContext->plan ||
      localHeaderContext->plan->failed() ||
      localHeaderContext->sourcePath != localHeaderContext->plan->rootSourcePath() ||
      localHeaderContext->artifactPath != localHeaderContext->plan->rootArtifactPath() ||
      localHeaderContext->sourceOnlyJointInput ||
      localHeaderContext->requireHelperTransaction || !fact.kernel ||
      !fact.policyQualifier.isFileID() || !fact.kernel->getLocation().isFileID() ||
      source.getFileID(fact.kernel->getLocation()) != source.getFileID(fact.policyQualifier)) return false;
  const auto file = source.getFileID(fact.policyQualifier);
  const auto path = contextualCanonicalPath(source.getFilename(fact.policyQualifier));
  if (path.empty() || !localHeaderContext->plan->isSelectedRootHeader(path)) return false;
  const auto *entry = source.getFileEntryForID(file);
  bool invalid = false;
  const auto bytes = source.getBufferData(file, &invalid);
  if (!entry || invalid || !physicalSourceFileMatches(path, *entry, source)) return false;
  std::string artifact;
  for (const auto &node : localHeaderContext->plan->nodes())
    if (node.sourcePath == path && node.depth == 1) artifact = node.artifactPath;
  LocalKernelThreadBoundPolicy policy;
  policy.sourcePath = path; policy.sourceSha256 = contextualInputSha256(bytes);
  policy.sourceBytes = bytes.size(); policy.kernelName = fact.kernel->getQualifiedNameAsString();
  policy.declarationOffset = source.getFileOffset(fact.kernel->getLocation());
  policy.qualifierOffset = source.getFileOffset(fact.policyQualifier);
  policy.maximum = fact.maximum; policy.artifactPath = artifact;
  return localHeaderContext->plan->requireKernelThreadBoundPolicy(policy);
}

bool AscifyAction::prepareKernelThreadBoundHeaderPolicies() {
  if (!localHeaderContext || !localHeaderContext->plan || localHeaderContext->depth != 1) return true;
  auto &compiler = getCompilerInstance(); auto &source = compiler.getSourceManager();
  const auto reject = [&](const char *message) {
    const auto diagnostic = compiler.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "kernel-bound header policy boundary: %0");
    compiler.getDiagnostics().Report(diagnostic) << message;
    cudaRuntimeApiBoundaryFailed = true;
    return false;
  };
  const auto path = contextualCanonicalPath(localHeaderContext->sourcePath);
  for (const auto &entry : localHeaderContext->plan->kernelThreadBoundPolicies()) {
    const auto &policy = entry.second;
    if (policy.sourcePath != path) continue;
    const auto main = source.getMainFileID();
    const auto begin = source.getLocForStartOfFile(main);
    const auto *file = source.getFileEntryForID(main);
    bool invalid = false; const auto bytes = source.getBufferData(main, &invalid);
    if (TargetPolicy != "dav-c310-vec" || TargetRecipe != "none" ||
        cudaAttributeControlModified || deviceTemplateAttributeMacroConflict ||
        compiler.getPreprocessorOpts().RetainExcludedConditionalBlocks ||
        localHeaderContext->virtualInput || policy.artifactPath != localHeaderContext->artifactPath ||
        path != contextualCanonicalPath(source.getFilename(begin)) || !file || invalid ||
        !physicalSourceFileMatches(path, *file, source) || bytes.size() != policy.sourceBytes ||
        contextualInputSha256(bytes) != policy.sourceSha256)
      return reject("selected-header parse input/target identity differs from the root fact");
    struct Find : clang::RecursiveASTVisitor<Find> {
      clang::SourceManager &source; const LocalKernelThreadBoundPolicy &policy;
      std::vector<const clang::FunctionDecl *> matches;
      Find(clang::SourceManager &s, const LocalKernelThreadBoundPolicy &p) : source(s), policy(p) {}
      bool VisitFunctionDecl(clang::FunctionDecl *function) {
        const auto loc = function->getLocation();
        if (loc.isFileID() && source.isWrittenInMainFile(loc) &&
            source.getFileOffset(loc) == policy.declarationOffset &&
            function->getQualifiedNameAsString() == policy.kernelName) matches.push_back(function);
        return true;
      }
    } find(source, policy);
    find.TraverseDecl(compiler.getASTContext().getTranslationUnitDecl());
    if (find.matches.size() != 1) return reject("kernel identity is not unique at its bound physical offset");
    ascify::PureSimtKernelThreadBound fact; clang::SourceLocation bad; std::string error;
    if (!ascify::ProvePureSimtKernelThreadBound(compiler.getASTContext(), find.matches.front(),
        [&](clang::SourceLocation loc) { return ownedNttpVoidCallLocation(loc); },
        [&](llvm::StringRef name, clang::SourceLocation loc) { return trustedKernelBoundQualifier(name, loc); },
        [&](llvm::StringRef name, clang::SourceLocation loc) {
          return !compiler.getPreprocessor().getMacroDefinitionAtLoc(
              compiler.getPreprocessor().getIdentifierInfo(name), loc);
        }, fact, bad, error) || !fact.policyQualifier.isFileID() ||
        !source.isWrittenInMainFile(fact.policyQualifier) || fact.maximum != policy.maximum ||
        source.getFileOffset(fact.policyQualifier) != policy.qualifierOffset)
      return reject("standalone header cannot rebuild the exact source kernel-bound policy");
    if (!kernelThreadBoundHeaderPolicies.emplace(policy.qualifierOffset, policy).second)
      return reject("two policies alias one physical header qualifier");
    kernelThreadBoundPlan.kernels[fact.kernel] = fact;
    kernelThreadBoundPlan.policyQualifiers[policy.qualifierOffset] = policy.maximum;
  }
  return true;
}

bool AscifyAction::rewriteProvenKernelAttributeCall(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *call = Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr || Result.SourceManager == nullptr ||
      call->getNumArgs() != 2 || call->isTypeDependent() || call->isValueDependent() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks ||
      TargetPolicy != "dav-c310-vec") return false;
  auto &context = *Result.Context;
  auto &source = *Result.SourceManager;
  const auto *callee = call->getDirectCallee();
  if (callee == nullptr || callee->getQualifiedNameAsString() != "cudaFuncGetAttributes" ||
      callee->isVariadic() || callee->getNumParams() != 2 ||
      !isAdmittedCudaRuntimeStatusCall(source, call, trustedSystemFileIds)) return false;
  for (const auto *decl : callee->redecls()) {
    const auto loc = source.getSpellingLoc(decl->getLocation());
    if (loc.isInvalid() || !cudaRuntimeApiSdkFileIds.count(source.getFileID(loc).getHashValue()))
      return false;
  }
  const auto *owner = enclosingNvidiaFindDeviceFunction(call, context);
  if (owner == nullptr || !owner->hasBody() ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      owner->hasAttr<clang::CUDADeviceAttr>() || owner->hasAttr<clang::CUDAGlobalAttr>())
    return false;
  // Prove the complete physical spelling at its original preprocessing point.
  // This also sees empty-expanding tokens omitted from AST child ranges.
  auto stablePhysicalRange = [&](clang::SourceLocation begin,
                                 clang::SourceLocation end,
                                 bool sdkQualifiers, unsigned boundary) {
    if (begin.isInvalid() || end.isInvalid() || begin.isMacroID() || end.isMacroID() ||
        source.getFileID(begin) != source.getFileID(end)) return false;
    const auto file = source.getFileID(begin);
    bool invalid = false;
    const auto buffer = source.getBufferData(file, &invalid);
    if (invalid) return false;
    const auto finish = clang::Lexer::getLocForEndOfToken(
        end, 0, source, getCompilerInstance().getLangOpts());
    if (finish.isInvalid()) return false;
    const unsigned first = source.getFileOffset(begin), last = source.getFileOffset(finish);
    clang::Lexer lexer(source.getLocForStartOfFile(file),
        getCompilerInstance().getLangOpts(), buffer.data(), buffer.data(),
        buffer.data() + buffer.size());
    auto &preprocessor = getCompilerInstance().getPreprocessor();
    clang::Token token;
    unsigned conditionalDepth = 0;
    bool directiveLine = false, entered = false, hasPrevious = false;
    clang::tok::TokenKind previous = clang::tok::unknown;
    while (true) {
      lexer.LexFromRawLexer(token);
      if (token.is(clang::tok::eof)) break;
      const unsigned position = source.getFileOffset(token.getLocation());
      if (position >= last) break;
      if (token.isAtStartOfLine()) directiveLine = false;
      if (token.is(clang::tok::hash) && token.isAtStartOfLine()) {
        if (position >= first) return false;
        directiveLine = true;
        lexer.LexFromRawLexer(token);
        if (token.is(clang::tok::eof)) break;
        if (token.is(clang::tok::raw_identifier)) {
          const auto name = token.getRawIdentifier();
          if (name == "if" || name == "ifdef" || name == "ifndef") ++conditionalDepth;
          else if (name == "endif") {
            if (conditionalDepth == 0) return false;
            --conditionalDepth;
          }
        }
        continue;
      }
      if (directiveLine) continue;
      if (position < first) {
        previous = token.getKind(); hasPrevious = true; continue;
      }
      if (conditionalDepth != 0) return false;
      if (!entered) {
        if (boundary != 0 && hasPrevious && previous != clang::tok::semi &&
            previous != clang::tok::r_brace &&
            !(boundary == 1 && previous == clang::tok::l_brace)) return false;
        entered = true;
      }
      if (!token.is(clang::tok::raw_identifier)) continue;
      const auto name = token.getRawIdentifier();
      const auto macro = preprocessor.getMacroDefinitionAtLoc(
          preprocessor.getIdentifierInfo(name), token.getLocation());
      if (!macro) continue;
      if (!sdkQualifiers || (name != "__global__" && name != "__launch_bounds__"))
        return false;
      const auto *info = macro.getMacroInfo();
      if (info == nullptr) return false;
      const auto location = source.getSpellingLoc(info->getDefinitionLoc());
      if (location.isInvalid() ||
          !cudaRuntimeApiSdkFileIds.count(source.getFileID(location).getHashValue()))
        return false;
    }
    return entered;
  };
  const auto *address = llvm::dyn_cast<clang::UnaryOperator>(call->getArg(0)->IgnoreParenImpCasts());
  if (address == nullptr || address->getOpcode() != clang::UO_AddrOf) return false;
  const auto *output = llvm::dyn_cast<clang::DeclRefExpr>(address->getSubExpr()->IgnoreParenImpCasts());
  const auto *object = output == nullptr ? nullptr : llvm::dyn_cast<clang::VarDecl>(output->getDecl());
  if (object == nullptr || !object->isLocalVarDecl() || !object->hasLocalStorage() ||
      object->getType().hasQualifiers() || object->getType()->isReferenceType() ||
      object->getDeclContext()->getRedeclContext() != owner->getRedeclContext()) return false;
  const auto *record = object->getType()->getAs<clang::RecordType>();
  if (record == nullptr || record->getDecl()->getQualifiedNameAsString() != "cudaFuncAttributes")
    return false;
  const auto recordLoc = source.getSpellingLoc(record->getDecl()->getLocation());
  if (recordLoc.isInvalid() || !cudaRuntimeApiSdkFileIds.count(source.getFileID(recordLoc).getHashValue()))
    return false;
  const auto objectParents = context.getParents(*object);
  const auto *declaration = objectParents.size() == 1
      ? objectParents[0].get<clang::DeclStmt>() : nullptr;
  if (declaration == nullptr || !declaration->isSingleDecl() ||
      !stablePhysicalRange(declaration->getBeginLoc(), declaration->getEndLoc(), false, 1))
    return false;
  // A source alias may itself vary under target preprocessing. Keep the direct
  // SDK record spelling, with an optional elaborated `struct` keyword only.
  if (object->getTypeSourceInfo() == nullptr) return false;
  bool invalidTypeText = false;
  auto typeText = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(
          object->getTypeSourceInfo()->getTypeLoc().getSourceRange()), source,
      getCompilerInstance().getLangOpts(), &invalidTypeText).trim();
  if (typeText.consume_front("struct")) {
    if (typeText.ltrim().size() == typeText.size()) return false;
    typeText = typeText.trim();
  }
  if (invalidTypeText || typeText != "cudaFuncAttributes") return false;
  // CUDA and the partial target record have different aggregate layouts.
  // Only an absent source initializer or syntactically empty braces preserve
  // the admitted max field if the query fails without modifying the output.
  if (const auto *initializer = object->getInit()) {
    if (const auto *list = llvm::dyn_cast<clang::InitListExpr>(initializer)) {
      if (const auto *syntax = list->getSyntacticForm()) list = syntax;
      if (list->getNumInits() != 0 || list->getBeginLoc().isMacroID() ||
          list->getEndLoc().isMacroID()) return false;
      // Even an empty-expanding macro between physical braces can change under
      // target preprocessing. Require the actual brace contents to be empty.
      bool invalid = false;
      auto spelling = clang::Lexer::getSourceText(
          clang::CharSourceRange::getTokenRange(list->getSourceRange()), source,
          getCompilerInstance().getLangOpts(), &invalid).trim();
      if (invalid || !spelling.consume_front("{") || !spelling.consume_back("}") ||
          !spelling.trim().empty()) return false;
    } else {
      // Clang represents `cudaFuncAttributes a;` with an implicit trivial
      // default construction at the declarator, despite no written initializer.
      const auto *construction = llvm::dyn_cast<clang::CXXConstructExpr>(initializer);
      if (construction == nullptr || construction->getNumArgs() != 0 ||
          !construction->getConstructor()->isImplicit() ||
          !construction->getConstructor()->isDefaultConstructor() ||
          !construction->getConstructor()->isTrivial() ||
          construction->getParenOrBraceRange().isValid() ||
          construction->getBeginLoc() != object->getLocation() ||
          construction->getEndLoc() != object->getLocation()) return false;
    }
  }
  struct Uses : clang::RecursiveASTVisitor<Uses> {
    clang::ASTContext &context;
    const clang::VarDecl *object;
    const clang::DeclRefExpr *output;
    unsigned reads = 0;
    Uses(clang::ASTContext &c, const clang::VarDecl *v, const clang::DeclRefExpr *o)
        : context(c), object(v), output(o) {}
    bool VisitLambdaExpr(clang::LambdaExpr *lambda) {
      for (const auto &capture : lambda->captures())
        if (capture.capturesVariable() &&
            capture.getCapturedVar()->getCanonicalDecl() == object->getCanonicalDecl())
          return false;
      return true;
    }
    bool VisitDeclRefExpr(clang::DeclRefExpr *ref) {
      if (ref->getDecl()->getCanonicalDecl() != object->getCanonicalDecl() || ref == output)
        return true;
      const clang::Stmt *node = ref;
      auto parents = context.getParents(*node);
      while (parents.size() == 1 && parents[0].get<clang::ParenExpr>() != nullptr) {
        node = parents[0].get<clang::ParenExpr>(); parents = context.getParents(*node);
      }
      const auto *member = parents.size() == 1 ? parents[0].get<clang::MemberExpr>() : nullptr;
      if (member == nullptr || member->isArrow() ||
          member->getMemberDecl()->getName() != "maxThreadsPerBlock") return false;
      // Parentheses and an lvalue conditional's selected operand preserve
      // the same field location. Admit them only on a path ending in a value
      // read; reference binding, writes and taking the result's address fail.
      node = member;
      parents = context.getParents(*node);
      while (parents.size() == 1) {
        if (const auto *paren = parents[0].get<clang::ParenExpr>()) {
          node = paren;
        } else if (const auto *conditional = parents[0].get<clang::ConditionalOperator>()) {
          if (!conditional->isLValue() ||
              (conditional->getTrueExpr() != node && conditional->getFalseExpr() != node))
            return false;
          node = conditional;
        } else break;
        parents = context.getParents(*node);
      }
      const auto *read = parents.size() == 1 ? parents[0].get<clang::ImplicitCastExpr>() : nullptr;
      if (read == nullptr || read->getCastKind() != clang::CK_LValueToRValue) return false;
      ++reads;
      return true;
    }
  } uses(context, object, output);
  if (!uses.TraverseStmt(owner->getBody()) || uses.reads == 0) return false;

  const clang::Expr *kernelExpr = call->getArg(1)->IgnoreParenImpCasts();
  if (const auto *unary = llvm::dyn_cast<clang::UnaryOperator>(kernelExpr)) {
    if (unary->getOpcode() != clang::UO_AddrOf) return false;
    kernelExpr = unary->getSubExpr()->IgnoreParenImpCasts();
  }
  const auto *kernelRef = llvm::dyn_cast<clang::DeclRefExpr>(kernelExpr);
  const auto *kernel = kernelRef == nullptr ? nullptr : llvm::dyn_cast<clang::FunctionDecl>(kernelRef->getDecl());
  // A macro body can select a different kernel under target preprocessing.
  // Keep only the original physical argument token through argument expansion.
  if (kernelRef == nullptr) return false;
  auto physicalArgumentLocation = [&](clang::SourceLocation location) {
    while (location.isMacroID()) {
      if (!source.isMacroArgExpansion(location)) return clang::SourceLocation();
      location = source.getImmediateSpellingLoc(location);
    }
    return location;
  };
  // Scan the whole physical call as well: an empty-expanding macro before or
  // after the kernel argument does not appear in that argument's AST range.
  const auto callBegin = physicalArgumentLocation(call->getBeginLoc());
  const auto callEnd = physicalArgumentLocation(call->getEndLoc());
  if (callBegin.isInvalid() || callEnd.isInvalid() ||
      source.getFileID(callBegin) != source.getFileID(callEnd)) return false;
  if (!stablePhysicalRange(callBegin, callEnd, false, 0)) return false;
  auto &preprocessor = getCompilerInstance().getPreprocessor();
  const auto kernelBegin = physicalArgumentLocation(call->getArg(1)->getBeginLoc());
  const auto kernelEnd = physicalArgumentLocation(call->getArg(1)->getEndLoc());
  if (kernelBegin.isInvalid() || kernelEnd.isInvalid() ||
      source.getFileID(kernelBegin) != source.getFileID(kernelEnd)) return false;
  bool invalidKernelText = false;
  const auto kernelText = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(kernelBegin, kernelEnd), source,
      getCompilerInstance().getLangOpts(), &invalidKernelText);
  if (invalidKernelText || kernelText.empty()) return false;
  // Namespace qualifier macros and empty-expanding tokens can occur between
  // physical endpoints. Prove the complete source reference has no such macro.
  // Unlike a complete SourceManager buffer, this source fragment is followed
  // by punctuation. Give the raw lexer its required NUL-terminated buffer.
  const std::string ownedKernelText = kernelText.str();
  clang::Lexer kernelLexer(clang::SourceLocation(),
      getCompilerInstance().getLangOpts(), ownedKernelText.data(), ownedKernelText.data(),
      ownedKernelText.data() + ownedKernelText.size());
  clang::Token token;
  std::string kernelSpelling;
  while (true) {
    // LexFromRawLexer reports buffer exhaustion even for its final valid token.
    // Test token kind so a final identifier without whitespace is still proved.
    kernelLexer.LexFromRawLexer(token);
    if (token.is(clang::tok::eof)) break;
    if (token.is(clang::tok::raw_identifier)) {
      if (preprocessor.getMacroDefinitionAtLoc(
              preprocessor.getIdentifierInfo(token.getRawIdentifier()), kernelBegin))
        return false;
      kernelSpelling += token.getRawIdentifier().str();
    } else if (token.is(clang::tok::coloncolon)) {
      kernelSpelling += "::";
    } else if (!token.isOneOf(clang::tok::l_paren, clang::tok::r_paren,
                              clang::tok::amp)) {
      return false;
    }
  }
  llvm::StringRef qualifiedSpelling(kernelSpelling);
  qualifiedSpelling.consume_front("::");
  // Aliases or unqualified lookup through another namespace can select a
  // different function after preprocessing. Keep the actual canonical name.
  if (kernel == nullptr || qualifiedSpelling != kernel->getQualifiedNameAsString())
    return false;
  ascify::PureSimtKernelThreadBound boundFact;
  clang::SourceLocation boundLocation;
  std::string boundError;
  if (!ascify::ProvePureSimtKernelThreadBound(context, kernel,
          [&](clang::SourceLocation location) { return ownedNttpVoidCallLocation(location); },
          [&](llvm::StringRef name, clang::SourceLocation location) {
            return trustedKernelBoundQualifier(name, location);
          }, [&](llvm::StringRef name, clang::SourceLocation location) {
            return !getCompilerInstance().getPreprocessor().getMacroDefinitionAtLoc(
                getCompilerInstance().getPreprocessor().getIdentifierInfo(name), location);
          }, boundFact, boundLocation, boundError) ||
      !stageKernelThreadBoundFact(boundFact)) return false;
  const unsigned bound = boundFact.maximum;
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(call->getCallee()->IgnoreParenImpCasts());
  if (reference == nullptr || reference->hasQualifier() || reference->hasExplicitTemplateArgs())
    return false;
  auto name = reference->getLocation();
  if (name.isMacroID() && !source.isMacroArgExpansion(name)) return false;
  name = source.getSpellingLoc(name);
  if (name.isInvalid() || !source.isWrittenInMainFile(name)) return false;
  const auto &options = getCompilerInstance().getLangOpts();
  const auto end = clang::Lexer::getLocForEndOfToken(name, 0, source, options);
  const auto spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(name, name), source, options);
  if (end.isInvalid() || spelling != "cudaFuncGetAttributes") return false;
  const unsigned offset = source.getFileOffset(name);
  if (!rewrittenKernelAttributeOffsets.insert(offset).second) return true;
  const std::string target = "::ascify::cudaFuncGetAttributesMaxThreads<" + std::to_string(bound) + ">";
  ct::Replacement edit(source, name, spelling.size(), target);
  if (!insertSemanticReplacement(edit, clang::FullSourceLoc(name, source), name, end)) {
    rewrittenKernelAttributeOffsets.erase(offset); return false;
  }
  provenKernelAttributeCalleeRanges.push_back(
      {source.getFileID(name), offset, source.getFileOffset(end)});
  provenKernelAttributeObjects.insert(object->getCanonicalDecl());
  kernelThreadBoundPlan.kernels[boundFact.kernel] = boundFact;
  if (boundFact.policyQualifier.isValid() && source.isWrittenInMainFile(boundFact.policyQualifier))
    kernelThreadBoundPlan.policyQualifiers[source.getFileOffset(boundFact.policyQualifier)] = bound;
  needsCudaCompatHeader = true;
  static const dppCounter counter = {"CUDA kernel launch bound", CONV_DEVICE, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(counter, "proven per-kernel maximum threads");
  return true;
}

bool AscifyAction::auditKernelAttributeObservations() {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  struct Audit : clang::RecursiveASTVisitor<Audit> {
    clang::SourceManager &source;
    const std::set<unsigned> &sdk;
    const std::set<const clang::VarDecl *> &proven;
    clang::SourceLocation bad;
    Audit(clang::SourceManager &s, const std::set<unsigned> &f,
          const std::set<const clang::VarDecl *> &v) : source(s), sdk(f), proven(v) {}
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool VisitDeclRefExpr(clang::DeclRefExpr *reference) {
      // A pointer-to-member can observe an unproved object without producing
      // a MemberExpr at its eventual use site. Never publish that escape.
      const auto *field = llvm::dyn_cast<clang::FieldDecl>(reference->getDecl());
      if (field == nullptr || field->getName() != "maxThreadsPerBlock" ||
          field->getParent()->getQualifiedNameAsString() != "cudaFuncAttributes")
        return true;
      const auto use = source.getExpansionLoc(reference->getExprLoc());
      if (use.isValid() && sdk.count(source.getFileID(use).getHashValue())) return true;
      const auto loc = source.getSpellingLoc(field->getLocation());
      if (loc.isInvalid() || !sdk.count(source.getFileID(loc).getHashValue())) return true;
      bad = reference->getExprLoc(); return false;
    }
    bool VisitOffsetOfExpr(clang::OffsetOfExpr *offset) {
      // offsetof stores field components outside the ordinary expression
      // children, so MemberExpr/DeclRefExpr visitors cannot see this layout use.
      const auto use = source.getExpansionLoc(offset->getExprLoc());
      if (use.isValid() && sdk.count(source.getFileID(use).getHashValue())) return true;
      for (unsigned index = 0; index < offset->getNumComponents(); ++index) {
        const auto &component = offset->getComponent(index);
        if (component.getKind() != clang::OffsetOfNode::Field) continue;
        const auto *field = component.getField();
        if (field->getName() != "maxThreadsPerBlock" ||
            field->getParent()->getQualifiedNameAsString() != "cudaFuncAttributes")
          continue;
        const auto loc = source.getSpellingLoc(field->getLocation());
        if (loc.isInvalid() || !sdk.count(source.getFileID(loc).getHashValue())) continue;
        bad = offset->getExprLoc(); return false;
      }
      return true;
    }
    bool VisitMemberExpr(clang::MemberExpr *member) {
      const auto *field = llvm::dyn_cast<clang::FieldDecl>(member->getMemberDecl());
      if (field == nullptr || field->getName() != "maxThreadsPerBlock") return true;
      // Audit retained local, forced and -isystem project headers too. Only
      // physically admitted CUDA SDK implementation files may observe their
      // own metadata without publishing the project compatibility adapter.
      const auto use = source.getExpansionLoc(member->getExprLoc());
      if (use.isValid() && sdk.count(source.getFileID(use).getHashValue())) return true;
      if (field->getParent()->getQualifiedNameAsString() != "cudaFuncAttributes")
        return true;
      const auto loc = source.getSpellingLoc(field->getLocation());
      if (loc.isInvalid() || !sdk.count(source.getFileID(loc).getHashValue())) return true;
      const auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(member->getBase()->IgnoreParenImpCasts());
      const auto *variable = ref == nullptr ? nullptr : llvm::dyn_cast<clang::VarDecl>(ref->getDecl());
      if (variable != nullptr && proven.count(variable->getCanonicalDecl())) return true;
      bad = member->getExprLoc(); return false;
    }
  } audit(source, cudaRuntimeApiSdkFileIds, provenKernelAttributeObjects);
  if (audit.TraverseDecl(compiler.getASTContext().getTranslationUnitDecl())) return true;
  const auto diagnostic = compiler.getDiagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error,
      "Ascify cannot publish unproved per-kernel maxThreadsPerBlock metadata at %0");
  // EndSourceFileAction runs after the diagnostic consumer releases its
  // source-language state: use a location-free diagnostic with textual origin.
  compiler.getDiagnostics().Report(diagnostic) << audit.bad.printToString(source);
  return false;
}

bool AscifyAction::preserveCudaIntegerIntrinsicArgumentConversion(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *call =
      Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr || call->getNumArgs() != 1 ||
      call->isTypeDependent() || call->isValueDependent() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
  const auto *callee = call->getDirectCallee();
  auto &context = *Result.Context;
  auto &source = *Result.SourceManager;
  if (callee == nullptr || callee->getIdentifier() == nullptr ||
      (callee->getQualifiedNameAsString() != "__ffs" &&
       callee->getQualifiedNameAsString() != "__clz") ||
      !callee->hasAttr<clang::CUDADeviceAttr>() || callee->isVariadic() ||
      callee->getNumParams() != 1 ||
      !context.hasSameType(callee->getReturnType(), context.IntTy) ||
      !context.hasSameType(callee->getParamDecl(0)->getType(), context.IntTy))
    return false;
  // A system-header annotation alone cannot make a project overload a CUDA
  // intrinsic. Require every declaration to belong to the physical SDK or
  // the compiler-owned forced CUDA wrapper closure.
  for (const auto *declaration : callee->redecls()) {
    const auto location = source.getSpellingLoc(declaration->getLocation());
    if (location.isInvalid() || !source.isInSystemHeader(location))
      return false;
    const auto file = source.getFileID(location);
    if (!cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
        !isOwnedCudaCompilerInput(file, getCompilerInstance()))
      return false;
  }
  const auto *owner = enclosingNonLambdaFunction(call, context);
  if (owner == nullptr || owner->hasAttr<clang::CUDAHostAttr>() ||
      (!owner->hasAttr<clang::CUDADeviceAttr>() &&
       !owner->hasAttr<clang::CUDAGlobalAttr>()) ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate)
    return false;
  const auto *reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
      call->getCallee()->IgnoreParenImpCasts());
  if (reference == nullptr || reference->getDecl()->getCanonicalDecl() !=
                                  callee->getCanonicalDecl() ||
      reference->getLocation().isMacroID())
    return false;
  const auto *conversion = llvm::dyn_cast<clang::ImplicitCastExpr>(call->getArg(0));
  if (conversion == nullptr || conversion->getCastKind() != clang::CK_IntegralCast ||
      !context.hasSameType(conversion->getType(), context.IntTy) ||
      !conversion->getSubExpr()->getType()->isIntegerType())
    return false;
  const auto *argument = conversion->getSubExpr();
  // Calls can contain independent rewrites, so retain them. Also retain
  // macro-expanded argument-expression node boundaries; an operator-only
  // macro token inside an ordinary node keeps its original spelling below.
  // Templates require a separate source-range proof and are left unchanged.
  struct ArgumentAudit : clang::RecursiveASTVisitor<ArgumentAudit> {
    bool VisitStmt(clang::Stmt *statement) {
      return !statement->getBeginLoc().isMacroID() &&
             !statement->getEndLoc().isMacroID();
    }
    bool VisitCallExpr(clang::CallExpr *) { return false; }
  } audit;
  if (!audit.TraverseStmt(const_cast<clang::Expr *>(argument)))
    return false;
  const auto begin = argument->getBeginLoc();
  const auto last = argument->getEndLoc();
  if (begin.isInvalid() || last.isInvalid() ||
      !source.isWrittenInMainFile(begin) || !source.isWrittenInMainFile(last))
    return false;
  const auto &options = getCompilerInstance().getLangOpts();
  const auto end = clang::Lexer::getLocForEndOfToken(last, 0, source, options);
  bool invalid = false;
  const auto spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(argument->getSourceRange()),
      source, options, &invalid);
  if (invalid || spelling.empty() || end.isInvalid())
    return false;
  const auto offset = source.getFileOffset(begin);
  if (!rewrittenIntegerIntrinsicArgumentOffsets.insert(offset).second)
    return true;
  // Preserve the proven CUDA int(int) overload of __ffs or __clz. Some CUDA
  // frontend/host combinations instead declare __clz as unsigned(unsigned);
  // that signature is deliberately outside this proof. Native SIMT also
  // overloads long and long long, so materialize the resolved conversion.
  ct::Replacement edit(source, begin, source.getFileOffset(end) - offset,
                       "static_cast<int>(" + spelling.str() + ")");
  if (!insertSemanticReplacement(edit, clang::FullSourceLoc(begin, source),
                                 begin, end)) {
    rewrittenIntegerIntrinsicArgumentOffsets.erase(offset);
    return false;
  }
  static const dppCounter ffsCounter = {
      "__ffs int argument", CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  static const dppCounter clzCounter = {
      "__clz int argument", CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  const bool isFfs = callee->getName() == "__ffs";
  Statistics::current().incrementCounter(
      isFfs ? ffsCounter : clzCounter,
      isFfs ? "CUDA __ffs integral conversion" : "CUDA __clz integral conversion");
  return true;
}

bool AscifyAction::rewriteProvenGenericAtomicExchCall(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *call = Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr || Result.SourceManager == nullptr ||
      call->isTypeDependent() || call->isValueDependent() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
  auto &context = *Result.Context;
  auto &source = *Result.SourceManager;
  const auto *callee = call->getDirectCallee();
  const auto *spec = findGlobalAtomicSpec("atomicExch");
  if (callee == nullptr || callee->getQualifiedNameAsString() != "atomicExch" ||
      call->getNumArgs() != 2 || !callee->hasAttr<clang::CUDADeviceAttr>())
    return false;
  const auto kind = classifyGlobalAtomicPointer(call->getArg(0)->getType(), context);
  if (kind == GlobalAtomicScalarKind::Unsupported ||
      !hasExactGlobalAtomicSignature(callee, *spec, kind, context) ||
      classifyGlobalAtomicScalar(call->getArg(1)->getType(), context) != kind)
    return false;
  for (const auto *declaration : callee->redecls()) {
    const auto location = source.getSpellingLoc(declaration->getLocation());
    if (location.isInvalid() || !source.isInSystemHeader(location)) return false;
    const auto file = source.getFileID(location);
    if (!cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
        !isOwnedCudaCompilerInput(file, getCompilerInstance())) return false;
  }
  const auto *owner = enclosingNvidiaFindDeviceFunction(call, context);
  if (owner == nullptr || llvm::isa<clang::CXXMethodDecl>(owner) ||
      !owner->getDeclContext()->getRedeclContext()->isTranslationUnit() ||
      owner->hasAttr<clang::CUDAHostAttr>() ||
      (!owner->hasAttr<clang::CUDADeviceAttr>() && !owner->hasAttr<clang::CUDAGlobalAttr>()) ||
      (owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate &&
       owner->getTemplatedKind() != clang::FunctionDecl::TK_FunctionTemplate))
    return false;
  // The primary template is admitted only for this already-resolved,
  // nondependent call and a fixed int/uint parameter type. No dependent
  // argument or class lookup is inferred from an observed specialization.
  const auto *address = call->getArg(0)->IgnoreParenImpCasts();
  bool takesAddress = false;
  if (const auto *unary = llvm::dyn_cast<clang::UnaryOperator>(address)) {
    if (unary->getOpcode() != clang::UO_AddrOf) return false;
    takesAddress = true;
    address = unary->getSubExpr()->IgnoreParenImpCasts();
  }
  const auto *root = llvm::dyn_cast<clang::DeclRefExpr>(address);
  const auto *variable = root == nullptr ? nullptr :
      llvm::dyn_cast<clang::VarDecl>(root->getDecl());
  if (variable == nullptr || variable->getDeclContext()->getRedeclContext() !=
          owner->getRedeclContext()) return false;
  if (const auto *parameter = llvm::dyn_cast<clang::ParmVarDecl>(variable)) {
    const auto type = parameter->getType();
    if (takesAddress) {
      if (!type->isLValueReferenceType() ||
          classifyGlobalAtomicScalar(type->getPointeeType(), context) != kind)
        return false;
    } else if (classifyGlobalAtomicPointer(type, context) != kind) {
      return false;
    }
  } else if (!takesAddress || !variable->hasAttr<clang::CUDASharedAttr>() ||
             variable->hasAttr<clang::CUDAConstantAttr>() ||
             classifyGlobalAtomicScalar(variable->getType(), context) != kind) {
    return false;
  }
  // Preserve the actual GM/UB address. In particular, a scalar reference
  // parameter does not establish GM, so no explicit address-space cast occurs.
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(call->getCallee()->IgnoreParenImpCasts());
  if (reference == nullptr || reference->hasQualifier() ||
      reference->getDecl()->getCanonicalDecl() != callee->getCanonicalDecl()) return false;
  const auto begin = call->getBeginLoc(), last = call->getEndLoc();
  const auto name = reference->getLocation();
  if (begin.isInvalid() || last.isInvalid() || name.isInvalid() ||
      begin.isMacroID() || last.isMacroID() || name.isMacroID() ||
      !source.isWrittenInMainFile(begin) || !source.isWrittenInMainFile(last) ||
      !source.isWrittenInMainFile(name)) return false;
  const auto &options = getCompilerInstance().getLangOpts();
  const auto end = clang::Lexer::getLocForEndOfToken(last, 0, source, options);
  const auto nameEnd = clang::Lexer::getLocForEndOfToken(name, 0, source, options);
  if (end.isInvalid() || nameEnd.isInvalid()) return false;
  const auto file = source.getFileID(begin);
  const unsigned first = source.getFileOffset(begin), limit = source.getFileOffset(end);
  for (const auto &macro : cudaLaneIdMacroRanges)
    if (macro.file == file && first < macro.endOffset && macro.beginOffset < limit)
      return false;
  const auto spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(name, name), source, options);
  if (spelling != "atomicExch") return false;
  const auto offset = source.getFileOffset(name);
  if (!rewrittenGlobalAtomicOffsets.insert(offset).second) return true;
  if (!insertSemanticReplacement(ct::Replacement(source, name, spelling.size(),
          "::ascify::atomic_exch_generic"), clang::FullSourceLoc(name, source), name, nameEnd)) {
    rewrittenGlobalAtomicOffsets.erase(offset);
    return false;
  }
  needsCudaCompatHeader = true;
  static const dppCounter counter = {
      "CUDA generic exchange", CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(counter, "SDK int/uint generic atomicExch");
  return true;
}

bool AscifyAction::rewriteProvenGenericAtomicMaxCall(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *call = Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr || Result.SourceManager == nullptr ||
      call->isTypeDependent() || call->isValueDependent() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
  const auto *callee = call->getDirectCallee();
  const auto *spec = findGlobalAtomicSpec("atomicMax");
  auto &context = *Result.Context;
  auto &source = *Result.SourceManager;
  if (callee == nullptr || callee->getQualifiedNameAsString() != "atomicMax" ||
      call->getNumArgs() != 2 || !callee->hasAttr<clang::CUDADeviceAttr>())
    return false;
  const auto kind = classifyGlobalAtomicPointer(call->getArg(0)->getType(), context);
  if (kind == GlobalAtomicScalarKind::Unsupported ||
      !hasExactGlobalAtomicSignature(callee, *spec, kind, context) ||
      classifyGlobalAtomicScalar(call->getArg(1)->getType(), context) != kind)
    return false;
  for (const auto *declaration : callee->redecls()) {
    const auto location = source.getSpellingLoc(declaration->getLocation());
    if (location.isInvalid() || !source.isInSystemHeader(location)) return false;
    const auto file = source.getFileID(location);
    if (!cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
        !isOwnedCudaCompilerInput(file, getCompilerInstance())) return false;
  }
  const auto *owner = enclosingNonLambdaFunction(call, context);
  if (owner == nullptr || !owner->hasAttr<clang::CUDADeviceAttr>() ||
      owner->hasAttr<clang::CUDAHostAttr>() || owner->hasAttr<clang::CUDAGlobalAttr>() ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      !isAddressRootedAtCurrentFunctionParameter(call->getArg(0), owner, context))
    return false;
  // Same generic-pointer contract as the admitted helper atomicAdd: the
  // parameter root does not imply GM, so no address-space cast occurs.
  const auto *reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
      call->getCallee()->IgnoreParenImpCasts());
  if (reference == nullptr || reference->hasQualifier() ||
      reference->getDecl()->getCanonicalDecl() != callee->getCanonicalDecl())
    return false;
  const auto name = reference->getLocation();
  if (name.isInvalid() || name.isMacroID() || !source.isWrittenInMainFile(name))
    return false;
  const auto &options = getCompilerInstance().getLangOpts();
  const auto spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(clang::SourceRange(name, name)), source, options);
  if (spelling != "atomicMax") return false;
  const auto end = clang::Lexer::getLocForEndOfToken(name, 0, source, options);
  if (end.isInvalid()) return false;
  const auto offset = source.getFileOffset(name);
  if (!rewrittenGlobalAtomicOffsets.insert(offset).second) return true;
  if (!insertSemanticReplacement(ct::Replacement(source, name, spelling.size(),
        "::ascify::atomic_max_generic"), clang::FullSourceLoc(name, source), name, end)) {
    rewrittenGlobalAtomicOffsets.erase(offset);
    return false;
  }
  needsCudaCompatHeader = true;
  static const dppCounter counter = {
      "CUDA generic helper atomic max", CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(counter, "SDK int/uint helper atomicMax");
  return true;
}

bool AscifyAction::rewriteProvenGenericAtomicAddCall(
    const mat::MatchFinder::MatchResult &Result) {
  const auto *call = Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr || Result.SourceManager == nullptr ||
      call->isTypeDependent() || call->isValueDependent() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
  const auto *callee = call->getDirectCallee();
  const auto *spec = findGlobalAtomicSpec("atomicAdd");
  auto &context = *Result.Context;
  auto &source = *Result.SourceManager;
  if (callee == nullptr || callee->getQualifiedNameAsString() != "atomicAdd" ||
      call->getNumArgs() != 2 || !callee->hasAttr<clang::CUDADeviceAttr>())
    return false;
  const auto kind = classifyGlobalAtomicPointer(call->getArg(0)->getType(), context);
  if (kind == GlobalAtomicScalarKind::Unsupported ||
      !hasExactGlobalAtomicSignature(callee, *spec, kind, context) ||
      classifyGlobalAtomicScalar(call->getArg(1)->getType(), context) != kind)
    return false;
  for (const auto *declaration : callee->redecls()) {
    const auto location = source.getSpellingLoc(declaration->getLocation());
    if (location.isInvalid() || !source.isInSystemHeader(location)) return false;
    const auto file = source.getFileID(location);
    if (!cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
        !isOwnedCudaCompilerInput(file, getCompilerInstance())) return false;
  }
  const auto *owner = enclosingNonLambdaFunction(call, context);
  if (owner == nullptr || !owner->hasAttr<clang::CUDADeviceAttr>() ||
      owner->hasAttr<clang::CUDAHostAttr>() || owner->hasAttr<clang::CUDAGlobalAttr>() ||
      owner->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      !isAddressRootedAtCurrentFunctionParameter(call->getArg(0), owner, context))
    return false;
  // The existing expression verifier proves a direct parameter root only.
  // Here it does not imply GM: a helper can receive either GM or UB, including
  // from another TU. The public generic API preserves both actual spaces.
  const auto *reference = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
      call->getCallee()->IgnoreParenImpCasts());
  if (reference == nullptr || reference->hasQualifier() ||
      reference->getDecl()->getCanonicalDecl() != callee->getCanonicalDecl())
    return false;
  const auto name = reference->getLocation();
  if (name.isInvalid() || name.isMacroID() || !source.isWrittenInMainFile(name))
    return false;
  const auto &options = getCompilerInstance().getLangOpts();
  const auto spelling = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(clang::SourceRange(name, name)), source, options);
  if (spelling != "atomicAdd") return false;
  const auto end = clang::Lexer::getLocForEndOfToken(name, 0, source, options);
  if (end.isInvalid()) return false;
  const auto offset = source.getFileOffset(name);
  if (!rewrittenGlobalAtomicOffsets.insert(offset).second) return true;
  if (!insertSemanticReplacement(ct::Replacement(source, name, spelling.size(),
        "::ascify::atomic_add_generic"), clang::FullSourceLoc(name, source), name, end)) {
    rewrittenGlobalAtomicOffsets.erase(offset);
    return false;
  }
  needsCudaCompatHeader = true;
  static const dppCounter counter = {
      "CUDA generic helper atomic add", CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(counter, "SDK int/uint helper atomicAdd");
  return true;
}

// This is deliberately smaller than the general symbol-copy storage domain:
// one directly spelled uint32 device counter, zero-initialized in this module.
// The ordinary symbol resolver must also commit before publication.
static const clang::VarDecl *provedAtomicIncDeviceCounter(
    const clang::Expr *argument, clang::CompilerInstance &compiler,
    const ascify::FrontendCompatibilityConfig &profile) {
  const auto *address = llvm::dyn_cast_or_null<clang::UnaryOperator>(
      stripParenAndImplicitCasts(argument));
  const auto *reference = address == nullptr ? nullptr :
      llvm::dyn_cast<clang::DeclRefExpr>(address->getSubExpr()->IgnoreParens());
  const auto *variable = reference == nullptr ? nullptr :
      llvm::dyn_cast<clang::VarDecl>(reference->getDecl());
  if (address == nullptr || address->getOpcode() != clang::UO_AddrOf ||
      reference == nullptr || reference->hasQualifier() ||
      reference->hasExplicitTemplateArgs() || variable == nullptr ||
      variable->isImplicit() || variable->getIdentifier() == nullptr ||
      !variable->hasGlobalStorage() || !variable->getDeclContext()->isFileContext() ||
      variable->isStaticDataMember() || variable->getDescribedVarTemplate() != nullptr ||
      variable->isInline() || llvm::isa<clang::VarTemplateSpecializationDecl>(variable) ||
      variable->getTLSKind() != clang::VarDecl::TLS_None ||
      variable->getPreviousDecl() != nullptr || variable->getMostRecentDecl() != variable ||
      variable->isThisDeclarationADefinition() == clang::VarDecl::DeclarationOnly ||
      !variable->hasAttr<clang::CUDADeviceAttr>() ||
      variable->hasAttr<clang::CUDAConstantAttr>() ||
      variable->hasAttr<clang::CUDASharedAttr>() || variable->getType().hasQualifiers())
    return nullptr;
  const auto *builtin = llvm::dyn_cast<clang::BuiltinType>(variable->getType().getTypePtr());
  auto &context = compiler.getASTContext();
  if (builtin == nullptr || builtin->getKind() != clang::BuiltinType::UInt ||
      context.getTypeSize(variable->getType()) != 32 ||
      !admitsDeviceSymbolStorage(variable->getType(), context)) return nullptr;
  for (const auto *attribute : variable->attrs())
    if (!llvm::isa<clang::CUDADeviceAttr>(attribute)) return nullptr;
  for (const auto *owner = variable->getDeclContext(); owner != nullptr; owner = owner->getParent())
    if (const auto *space = llvm::dyn_cast<clang::NamespaceDecl>(owner))
      if (space->isAnonymousNamespace()) return nullptr;
  const auto *zero = llvm::dyn_cast_or_null<clang::IntegerLiteral>(
      variable->getInit() == nullptr ? nullptr : variable->getInit()->IgnoreParenImpCasts());
  if (zero == nullptr || !zero->getValue().isZero() ||
      variable->getInitStyle() != clang::VarDecl::CInit) return nullptr;
  auto &source = compiler.getSourceManager();
  auto &preprocessor = compiler.getPreprocessor();
  const auto qualifier = source.getExpansionLoc(
      variable->getAttr<clang::CUDADeviceAttr>()->getLocation());
  const auto name = variable->getLocation(), last = variable->getEndLoc();
  if (qualifier.isInvalid() || name.isInvalid() || last.isInvalid() ||
      name.isMacroID() || last.isMacroID() ||
      !source.isWrittenInMainFile(qualifier) || !source.isWrittenInMainFile(name) ||
      !source.isWrittenInMainFile(last) ||
      !hasPhysicalMacroFreeRange(argument->getSourceRange(), preprocessor)) return nullptr;
  clang::Token token;
  if (clang::Lexer::getRawToken(qualifier, token, source, compiler.getLangOpts(), true) ||
      token.needsCleaning() || clang::Lexer::getSpelling(token, source, compiler.getLangOpts()) != "__device__")
    return nullptr;
  const auto file = source.getFileID(qualifier);
  for (const auto macro : {"__device__", "__location__", "__annotate__"}) {
    const auto *identifier = preprocessor.getIdentifierInfo(macro);
    if (!frontendCgMacroExpansionMatches(macro,
            preprocessor.getMacroDefinitionAtLoc(identifier, qualifier).getMacroInfo(),
            file, compiler, profile)) return nullptr;
  }
  for (const auto macro : {"__attribute__", "device"})
    if (preprocessor.getMacroDefinitionAtLoc(
            preprocessor.getIdentifierInfo(macro), qualifier).getMacroInfo()) return nullptr;
  const auto prefix = physicalDeclarationPrefixBegin(qualifier, preprocessor);
  if (prefix.isInvalid() ||
      !hasPhysicalMacroFreeRange({prefix, last}, preprocessor, qualifier)) return nullptr;
  return variable;
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
  const clang::VarDecl *deviceCounter = nullptr;
  if (!isAddressRootedAtCurrentFunctionParameter(call->getArg(0), kernel, context)) {
    if (callee->getName() != "atomicInc" || kind != GlobalAtomicScalarKind::UInt32 ||
        cudaAttributeControlModified) return false;
    for (const auto *declaration : callee->redecls()) {
      const auto location = sourceManager.getExpansionLoc(declaration->getLocation());
      if (location.isInvalid() ||
          !cudaRuntimeApiSdkFileIds.count(sourceManager.getFileID(location).getHashValue()))
        return false;
      for (const auto *attribute : declaration->attrs())
        if (!llvm::isa<clang::CUDADeviceAttr>(attribute) &&
            !llvm::isa<clang::AlwaysInlineAttr>(attribute)) return false;
    }
    deviceCounter = provedAtomicIncDeviceCounter(call->getArg(0), getCompilerInstance(), frontendCompatibility);
    if (deviceCounter == nullptr) return false;
  }

  const clang::Expr *calleeExpression =
      stripParenAndImplicitCasts(call->getCallee());
  const auto *reference =
      llvm::dyn_cast_or_null<clang::DeclRefExpr>(calleeExpression);
  if (reference == nullptr || reference->getDecl()->getCanonicalDecl() !=
                                  callee->getCanonicalDecl())
    return false;
  if (deviceCounter != nullptr &&
      (reference->hasQualifier() || reference->hasExplicitTemplateArgs()))
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
  if (deviceCounter != nullptr)
    requiredAtomicSymbolOffsets.insert(sourceManager.getFileOffset(deviceCounter->getLocation()));
  return true;
}

bool AscifyAction::rewriteProvenDeviceMemoryCall(
    const mat::MatchFinder::MatchResult &Result) {
  // Object type and storage proofs describe the actual preprocessed program.
  // Do not infer them from the legacy mode that retains excluded branches.
  if (getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks)
    return false;
  const auto *call =
      Result.Nodes.getNodeAs<clang::CallExpr>(sProvenGlobalAtomicCall);
  if (call == nullptr || Result.Context == nullptr ||
      Result.SourceManager == nullptr)
    return false;
  ascify::DeviceMemoryRewrite plan;
  if (!ascify::planDeviceMemoryCall(
          call, *Result.Context, trustedSystemFileIds, plan))
    return false;
  auto &sourceManager = *Result.SourceManager;
  const auto begin = plan.range.getBegin();
  const auto last = plan.range.getEnd();
  if (begin.isInvalid() || last.isInvalid() || begin.isMacroID() ||
      last.isMacroID() || !sourceManager.isWrittenInMainFile(begin) ||
      !sourceManager.isWrittenInMainFile(last))
    return false;
  const auto end = clang::Lexer::getLocForEndOfToken(
      last, 0, sourceManager, getCompilerInstance().getLangOpts());
  if (end.isInvalid())
    return false;
  const unsigned offset = sourceManager.getFileOffset(begin);
  const unsigned endOffset = sourceManager.getFileOffset(end);
  if (endOffset <= offset)
    return false;
  if (!rewrittenDeviceMemoryOffsets.insert(offset).second)
    return true;
  ct::Replacement edit(sourceManager, begin, endOffset - offset, plan.replacement);
  if (!insertSemanticReplacement(
          edit, clang::FullSourceLoc(begin, sourceManager), begin, end)) {
    rewrittenDeviceMemoryOffsets.erase(offset);
    return false;
  }
  static const dppCounter counter = {
      "ascify private device memcpy", CONV_DEVICE_FUNC, API_RUNTIME, 0, FULL};
  Statistics::current().incrementCounter(counter, plan.apiName);
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
  scalar64Plan = {};
  dynamicWarpGroups = {};
  dynamicWarpMacroEvents.clear();
  capturedDynamicProjectionName.clear();
  nttpVoidCallPlan = {};
  directImageWitness = {};
  directImageUtilityMacroUses.clear();
  directImageUtilityControls.clear();
  directImageUtilityInvalid = false;
  originalCgInputs.clear();
  kernelThreadBoundPlan = {};
  kernelThreadBoundHeaderPolicies.clear();
  originalCgIncludes.clear();
  originalCgSdkRoot.clear();
  originalCgControlChanged = false;
  originalCgInquiry = false;
  originalCgCompilerDefaultMacro = nullptr;
  originalCgCompilerDefaultDefinition = {};
  originalCgCompilerDefaultInput = {};
  davC310TargetRecipe.reset();
  if (!NoLowerDeviceDoubleParams) {
    Finder->addMatcher(
        mat::parmVarDecl(mat::isExpansionInMainFile())
            .bind(sCudaGlobalScalarDoubleParam),
        this);
  }
  Finder->addMatcher(
      mat::varDecl(mat::isExpansionInMainFile()).bind(sCudaDeviceSymbol), this);
  Finder->addMatcher(
      mat::varDecl(mat::isExpansionInMainFile())
          .bind(sCudaDefaultDim3),
      this);
  Finder->addMatcher(
      mat::varDecl(mat::isExpansionInMainFile()).bind(sCudaHalf2DirectInit),
      this);
  Finder->addMatcher(
      mat::cxxOperatorCallExpr(mat::isExpansionInMainFile())
          .bind(sCudaHalf2Operator),
      this);
  if (frontendCompatibility.enabled()) {
    Finder->addMatcher(
        mat::declRefExpr(mat::to(mat::functionDecl(mat::hasAnyName("max", "min"))),
                         mat::isExpansionInMainFile())
            .bind(sFrontendHostMath),
        this);
  }
  // Keep this matcher independent of API spelling and perform the CUDA API,
  // scalar type, enclosing function, and address-provenance proofs in the
  // callbacks. Separate proofs admit global atomics or a private object copy;
  // a failed proof makes no edit and cannot forge a memory address space.
  Finder->addMatcher(
      mat::callExpr(mat::callee(mat::functionDecl()),
                    mat::isExpansionInMainFile())
          .bind(sProvenGlobalAtomicCall),
      this);
  // Test location through Stmt's concrete-kind dispatch: the abstract
  // AsmStmt base does not supply the GCC statement's source location.
  Finder->addMatcher(
      mat::stmt(mat::isExpansionInMainFile(), mat::asmStmt())
          .bind(sCudaLaneIdAsm),
      this);
  Finder->addMatcher(
      mat::functionDecl(mat::isExpansionInMainFile()).bind(sCudaAttributeGuard),
      this);
  Finder->addMatcher(mat::functionDecl(mat::hasAnyParameter(
      mat::parmVarDecl(mat::hasAttr(clang::attr::CUDAGridConstant))))
      .bind(sGridConstantParameterGuard), this);
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

void AscifyAction::recordCudaAttributeGuard(
    clang::SourceLocation location, const clang::Token &name,
    const clang::MacroDefinition &definition) {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto &preprocessor = compiler.getPreprocessor();
  const auto *macro = definition.getMacroInfo();
  const bool compilerDefinition = macro != nullptr &&
      ((source.getFileID(macro->getDefinitionLoc()) == preprocessor.getPredefinesFileID() &&
        source.isWrittenInBuiltinFile(macro->getDefinitionLoc())) ||
       isOwnedCudaCompilerInput(source.getFileID(macro->getDefinitionLoc()), compiler));
  if (TargetPolicy != "dav-c310-vec" || AscifyAMAP ||
      compiler.getPreprocessorOpts().RetainExcludedConditionalBlocks ||
      cudaAttributeControlModified || !location.isFileID() ||
      !source.isWrittenInMainFile(location) ||
      name.getIdentifierInfo() == nullptr ||
      name.getIdentifierInfo()->getName() != "__CUDACC__" || macro == nullptr ||
      !compilerDefinition)
    return;
  bool invalid = false;
  const auto file = source.getFileID(location);
  const auto buffer = source.getBufferData(file, &invalid);
  const unsigned offset = source.getFileOffset(location);
  if (invalid || offset >= buffer.size()) return;
  const auto newline = buffer.rfind('\n', offset);
  const auto lineBegin = newline == llvm::StringRef::npos ? 0 : newline + 1;
  const auto lineEnd = buffer.find('\n', offset);
  if (lineEnd == llvm::StringRef::npos) return;
  auto line = buffer.slice(lineBegin, lineEnd).trim(" \t");
  if (!line.consume_front("#")) return;
  line = line.ltrim(" \t");
  if (!line.consume_front("ifdef") || line.empty() ||
      (line.front() != ' ' && line.front() != '\t') ||
      line.trim(" \t") != "__CUDACC__") return;
  const auto start = source.getLocForStartOfFile(file);
  clang::Lexer lexer(start, compiler.getLangOpts(), buffer.begin(),
                     buffer.begin() + lineEnd + 1, buffer.end());
  clang::Token tokens[4];
  for (auto &token : tokens) lexer.LexFromRawLexer(token);
  if (!tokens[0].isAnyIdentifier() || !tokens[1].isAnyIdentifier() ||
      !tokens[2].is(clang::tok::hash) || !tokens[2].isAtStartOfLine() ||
      !tokens[3].isAnyIdentifier()) return;
  auto spelling = [&](const clang::Token &token) {
    return clang::Lexer::getSpelling(token, source, compiler.getLangOpts());
  };
  if (spelling(tokens[0]) != "__host__" || spelling(tokens[1]) != "__device__" ||
      spelling(tokens[3]) != "endif") return;
  const unsigned closingOffset = source.getFileOffset(tokens[2].getLocation());
  const auto closingEnd = buffer.find('\n', closingOffset);
  if (closingEnd == llvm::StringRef::npos ||
      buffer.slice(closingOffset, closingEnd).trim(" \t") != "#endif") return;
  // Exclude escaped physical lines and alternative directive spellings. The
  // raw lexer otherwise permits ordinary whitespace and comments in the body.
  const auto physical = buffer.slice(lineBegin, closingEnd);
  if (physical.contains('\\') || physical.contains('\r') ||
      physical.contains("??") || physical.contains("%:")) return;
  const unsigned hash = lineBegin + buffer.slice(lineBegin, lineEnd).find('#');
  cudaAttributeGuards.push_back({start.getLocWithOffset(hash),
      start.getLocWithOffset(lineEnd), tokens[0].getLocation(),
      tokens[1].getLocation(), false});
}

void AscifyAction::proveCudaAttributeGuard(
    const mat::MatchFinder::MatchResult &result) {
  const auto *function = result.Nodes.getNodeAs<clang::FunctionDecl>(sCudaAttributeGuard);
  if (function == nullptr || !function->isThisDeclarationADefinition() ||
      llvm::isa<clang::CXXMethodDecl>(function) ||
      function->getLexicalDeclContext()->isRecord() ||
      function->getTemplateSpecializationKind() != clang::TSK_Undeclared ||
      function->hasAttr<clang::CUDAGlobalAttr>()) return;
  // Opening an inactive target guard must not manufacture a device FP64
  // interface. This is a known-signature exclusion, not reachability analysis
  // or a proof that arbitrary function bodies are supported by the target.
  std::set<const clang::Type *> visitedTypes;
  std::function<bool(clang::QualType)> hasKnownFp64 =
      [&](clang::QualType original) -> bool {
    if (original.isNull()) return false;
    const auto type = original.getCanonicalType();
    if (!visitedTypes.insert(type.getTypePtr()).second) return false;
    if (type->isSpecificBuiltinType(clang::BuiltinType::Double) ||
        type->isSpecificBuiltinType(clang::BuiltinType::LongDouble)) return true;
    if (const auto *pointer = type->getAs<clang::PointerType>())
      return hasKnownFp64(pointer->getPointeeType());
    if (const auto *reference = type->getAs<clang::ReferenceType>())
      return hasKnownFp64(reference->getPointeeType());
    if (const auto *array = result.Context->getAsArrayType(type))
      return hasKnownFp64(array->getElementType());
    if (const auto *vector = type->getAs<clang::VectorType>())
      return hasKnownFp64(vector->getElementType());
    if (const auto *complex = type->getAs<clang::ComplexType>())
      return hasKnownFp64(complex->getElementType());
    if (const auto *atomic = type->getAs<clang::AtomicType>())
      return hasKnownFp64(atomic->getValueType());
    if (const auto *member = type->getAs<clang::MemberPointerType>())
      return hasKnownFp64(member->getPointeeType());
    if (const auto *prototype = type->getAs<clang::FunctionProtoType>()) {
      if (hasKnownFp64(prototype->getReturnType())) return true;
      for (const auto parameter : prototype->param_types())
        if (hasKnownFp64(parameter)) return true;
    }
    return false;
  };
  if (hasKnownFp64(function->getReturnType())) return;
  for (const auto *parameter : function->parameters())
    if (hasKnownFp64(parameter->getType())) return;
  const auto *host = function->getAttr<clang::CUDAHostAttr>();
  const auto *device = function->getAttr<clang::CUDADeviceAttr>();
  if (host == nullptr || device == nullptr || host->isInherited() || device->isInherited()) return;
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto &preprocessor = compiler.getPreprocessor();
  auto ownedAttribute = [&](llvm::StringRef name, clang::SourceLocation location) {
    std::set<const clang::IdentifierInfo *> seen;
    std::vector<const clang::IdentifierInfo *> pending = {
        preprocessor.getIdentifierInfo(name)};
    while (!pending.empty()) {
      const auto *identifier = pending.back();
      pending.pop_back();
      if (!seen.insert(identifier).second) continue;
      if (seen.size() > 16) return false;
      const auto *info = preprocessor.getMacroDefinitionAtLoc(
          identifier, location).getMacroInfo();
      if (info == nullptr) return false;
      const auto file = source.getFileID(source.getSpellingLoc(info->getDefinitionLoc()));
      if (!file.isValid() ||
          (!cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
           !isOwnedCudaCompilerInput(file, compiler))) return false;
      for (const auto &token : info->tokens()) {
        if (token.is(clang::tok::hash) || token.is(clang::tok::hashhash)) return false;
        const auto *dependency = token.getIdentifierInfo();
        if (dependency == nullptr) continue;
        bool parameter = false;
        if (info->isFunctionLike())
          for (const auto *argument : info->params()) parameter |= argument == dependency;
        if (!parameter && preprocessor.getMacroDefinitionAtLoc(dependency, location).getMacroInfo())
          pending.push_back(dependency);
      }
    }
    return true;
  };
  for (auto &guard : cudaAttributeGuards) {
    if (source.getExpansionLoc(host->getLocation()) != guard.host ||
        source.getExpansionLoc(device->getLocation()) != guard.device ||
        !ownedAttribute("__host__", guard.host) ||
        !ownedAttribute("__device__", guard.device)) continue;
    guard.functionProven = true;
  }
}

// This is a value-only specialization, not an implementation of CUDA's
// per-grid parameter-object address/lifetime contract. Every use and every
// declaration must be proved before any qualifier belonging to a kernel moves.
void AscifyAction::specializeGridConstantParameters(
    const clang::FunctionDecl *function) {
  if (function == nullptr) return;
  bool hasGridConstant = false;
  for (const auto *p : function->parameters())
    hasGridConstant |= p->hasAttr<clang::CUDAGridConstantAttr>();
  if (!hasGridConstant) return;
  // A declaration with a definition is checked by that definition's callback.
  if (function->getDefinition() != nullptr &&
      function != function->getDefinition()) return;
  auto &compiler = getCompilerInstance();
  auto &context = compiler.getASTContext();
  auto &source = compiler.getSourceManager();
  auto &preprocessor = compiler.getPreprocessor();
  auto reject = [&]() {
    auto &diagnostics = compiler.getDiagnostics();
    const auto id = diagnostics.getCustomDiagID(clang::DiagnosticsEngine::Error,
        "Ascify grid-constant boundary: requires an owned, non-template kernel "
        "definition with const int-array value parameters and direct scalar "
        "reads only; no parameter address, reference, copy or reflection");
    diagnostics.Report(function->getLocation(), id);
  };
  if (TargetPolicy != "dav-c310-vec" ||
      (localHeaderContext != nullptr && localHeaderContext->depth != 0) ||
      !function->hasBody() ||
      !function->hasAttr<clang::CUDAGlobalAttr>() ||
      llvm::isa<clang::CXXMethodDecl>(function) ||
      function->getLexicalDeclContext()->isRecord() ||
      function->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate ||
      function->isDependentContext() || function->isVariadic()) {
    reject(); return;
  }
  auto ownedAttribute = [&](clang::SourceLocation location) {
    std::set<const clang::IdentifierInfo *> seen;
    std::vector<const clang::IdentifierInfo *> pending = {
        preprocessor.getIdentifierInfo("__grid_constant__")};
    while (!pending.empty()) {
      const auto *identifier = pending.back(); pending.pop_back();
      if (!seen.insert(identifier).second) continue;
      if (seen.size() > 16) return false;
      const auto *info = preprocessor.getMacroDefinitionAtLoc(
          identifier, location).getMacroInfo();
      if (info == nullptr) return false;
      const auto file = source.getFileID(source.getSpellingLoc(info->getDefinitionLoc()));
      if (!file.isValid() ||
          (!cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
           !isOwnedCudaCompilerInput(file, compiler))) return false;
      for (const auto &token : info->tokens()) {
        if (token.is(clang::tok::hash) || token.is(clang::tok::hashhash)) return false;
        const auto *dependency = token.getIdentifierInfo();
        if (dependency == nullptr) continue;
        bool parameter = false;
        if (info->isFunctionLike())
          for (const auto *argument : info->params()) parameter |= argument == dependency;
        if (!parameter && preprocessor.getMacroDefinitionAtLoc(dependency, location).getMacroInfo())
          pending.push_back(dependency);
      }
    }
    return true;
  };
  std::map<const clang::ParmVarDecl *, const clang::FieldDecl *> parameters;
  for (const auto *p : function->parameters()) {
    if (!p->hasAttr<clang::CUDAGridConstantAttr>()) continue;
    const auto type = p->getType().getCanonicalType();
    const auto *record = type->getAsCXXRecordDecl();
    if (!type.getQualifiers().hasOnlyConst() ||
        type->isReferenceType() || record == nullptr || !record->isCompleteDefinition() ||
        record->isUnion() || !record->isTrivial() || !record->isStandardLayout() ||
        !record->isAggregate() || record->getNumBases() != 0 || record->hasAttrs() ||
        record->getLocation().isMacroID() || record->getBeginLoc().isMacroID() ||
        !source.isWrittenInMainFile(record->getLocation())) {
      reject(); return;
    }
    const clang::FieldDecl *field = nullptr;
    for (const auto *member : record->decls()) {
      if (member->isImplicit()) continue;
      const auto *candidate = llvm::dyn_cast<clang::FieldDecl>(member);
      if (candidate == nullptr || field != nullptr) { reject(); return; }
      field = candidate;
    }
    const auto *array = field == nullptr ? nullptr : context.getAsConstantArrayType(field->getType());
    if (array == nullptr || array->getSize().isZero() || field->isMutable() ||
        field->getLocation().isMacroID() || field->getBeginLoc().isMacroID() ||
        !source.isWrittenInMainFile(field->getLocation()) ||
        field->getAccess() != clang::AS_public ||
        field->isBitField() || field->hasAttrs() || field->hasInClassInitializer() ||
        field->getType().isVolatileQualified() ||
        array->getElementType().isVolatileQualified() ||
        !array->getElementType()->isSpecificBuiltinType(clang::BuiltinType::Int) ||
        context.getTypeSize(array->getElementType()) != 32 ||
        context.getTypeSize(type) > 32000 * 8 ||
        context.getTypeSize(type) != context.getTypeSize(field->getType()) ||
        context.getTypeAlign(type) != context.getTypeAlign(context.IntTy)) {
      reject(); return;
    }
    parameters.emplace(p, field);
  }
  // Bound the entire signature, not just one record. These shapes have been
  // exercised on the target: one int-array value with up to three 32-bit
  // integral / 64-bit int-pointer arguments. This conservative source-layout
  // envelope is not a model of the complete CCEC kernel calling convention.
  if (parameters.size() != 1 || function->getNumParams() > 4) { reject(); return; }
  uint64_t signatureBytes = 0, signatureAlignment = 1;
  unsigned pointerArguments = 0, integerArguments = 0;
  for (const auto *p : function->parameters()) {
    const auto type = p->getType().getCanonicalType();
    const auto int32 = [&](clang::QualType value) {
      return !value.isNull() &&
          (value.getQualifiers().empty() || value.getQualifiers().hasOnlyConst()) &&
          (value->isSpecificBuiltinType(clang::BuiltinType::Int) ||
           value->isSpecificBuiltinType(clang::BuiltinType::UInt)) &&
          context.getTypeSize(value) == 32;
    };
    if (!parameters.count(p) &&
        !int32(type) &&
        !(type->isPointerType() &&
          (type.getQualifiers().empty() || type.getQualifiers().hasOnlyConst()) &&
          context.getTypeSize(type) == 64 && int32(type->getPointeeType()))) {
      reject(); return;
    }
    if (!parameters.count(p)) {
      if (type->isPointerType()) ++pointerArguments;
      else ++integerArguments;
    }
    const uint64_t alignment = context.getTypeAlignInChars(type).getQuantity();
    const uint64_t bytes = context.getTypeSizeInChars(type).getQuantity();
    if (!alignment || alignment > 8 || bytes > 32000) { reject(); return; }
    signatureBytes = ((signatureBytes + alignment - 1) / alignment) * alignment + bytes;
    signatureAlignment = std::max(signatureAlignment, alignment);
  }
  signatureBytes = ((signatureBytes + signatureAlignment - 1) / signatureAlignment) * signatureAlignment;
  if (signatureBytes > 32024 || pointerArguments > 1 || integerArguments > 2 ||
      (!parameters.count(function->getParamDecl(0)) &&
       !parameters.count(function->getParamDecl(function->getNumParams() - 1)))) {
    reject(); return;
  }
  class Uses : public clang::RecursiveASTVisitor<Uses> {
    clang::ASTContext &context;
    const std::map<const clang::ParmVarDecl *, const clang::FieldDecl *> &parameters;
    const clang::FunctionDecl *function;
    const clang::Stmt *parent(const clang::Stmt *node) {
      auto p = context.getParents(*node);
      return p.size() == 1 ? p[0].get<clang::Stmt>() : nullptr;
    }
    const clang::Stmt *pastParens(const clang::Stmt *node) {
      while (const auto *p = llvm::dyn_cast_or_null<clang::ParenExpr>(parent(node))) node = p;
      return node;
    }
  public:
    bool valid = true;
    Uses(clang::ASTContext &c, decltype(parameters) p, const clang::FunctionDecl *f)
        : context(c), parameters(p), function(f) {}
    bool shouldVisitImplicitCode() const { return true; }
    bool VisitDeclRefExpr(clang::DeclRefExpr *reference) {
      const auto *parameter = llvm::dyn_cast<clang::ParmVarDecl>(reference->getDecl());
      const auto found = parameters.find(parameter);
      if (found == parameters.end()) return true;
      const auto *base = pastParens(reference);
      const auto *member = llvm::dyn_cast_or_null<clang::MemberExpr>(parent(base));
      if (member == nullptr || member->isArrow() || member->getBase() != base ||
          member->getMemberDecl() != found->second) { valid = false; return true; }
      const auto *decay = llvm::dyn_cast_or_null<clang::ImplicitCastExpr>(parent(pastParens(member)));
      if (decay == nullptr || decay->getCastKind() != clang::CK_ArrayToPointerDecay) {
        valid = false; return true;
      }
      const auto *array = llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(parent(pastParens(decay)));
      if (array == nullptr || array->getBase()->IgnoreParens() != decay) {
        valid = false; return true;
      }
      const auto *read = llvm::dyn_cast_or_null<clang::ImplicitCastExpr>(parent(pastParens(array)));
      if (read == nullptr || read->getCastKind() != clang::CK_LValueToRValue) {
        valid = false; return true;
      }
      // A scalar read still cannot hide inside a lambda capture or an
      // unevaluated/type-observation context. Walk declarations as well as
      // statements, since local initializers insert VarDecl parents.
      auto node = clang::DynTypedNode::create(*read);
      for (unsigned depth = 0; depth != 256; ++depth) {
        const auto parents = context.getParents(node);
        if (parents.size() != 1) { valid = false; return true; }
        node = parents[0];
        if (const auto *f = node.get<clang::FunctionDecl>()) {
          valid &= f == function; return true;
        }
        const auto *stmt = node.get<clang::Stmt>();
        if (stmt == nullptr && node.get<clang::VarDecl>() == nullptr) {
          valid = false; return true;
        }
        if (stmt && llvm::isa<clang::LambdaExpr, clang::UnaryExprOrTypeTraitExpr,
                             clang::CXXNoexceptExpr, clang::CXXTypeidExpr>(stmt)) {
          valid = false; return true;
        }
      }
      valid = false; return true;
    }
  } uses(context, parameters, function);
  uses.TraverseStmt(function->getBody());
  if (!uses.valid) { reject(); return; }
  ct::Replacements staged = *replacements;
  std::vector<SemanticRewriteRange> admitted;
  for (const auto *declaration : function->redecls()) {
    if (declaration->getLocation().isMacroID() ||
        !source.isWrittenInMainFile(declaration->getLocation()) ||
        declaration->getNumParams() != function->getNumParams()) { reject(); return; }
    for (unsigned i = 0; i != declaration->getNumParams(); ++i) {
      const auto *p = declaration->getParamDecl(i);
      if (p->hasDefaultArg() || !context.hasSameType(
              p->getType(), function->getParamDecl(i)->getType())) { reject(); return; }
      const bool expected = parameters.count(function->getParamDecl(i));
      if (expected != p->hasAttr<clang::CUDAGridConstantAttr>()) { reject(); return; }
      if (!expected) {
        if (p->hasAttrs()) { reject(); return; }
        continue;
      }
      unsigned attributes = 0;
      for (const auto *attribute : p->attrs()) {
        const auto *grid = llvm::dyn_cast<clang::CUDAGridConstantAttr>(attribute);
        if (grid == nullptr || grid->isInherited() || ++attributes != 1) { reject(); return; }
        const auto location = source.getExpansionLoc(grid->getLocation());
        if (!grid->getLocation().isMacroID() || !location.isFileID() ||
            !source.isWrittenInMainFile(location) || !ownedAttribute(location)) { reject(); return; }
        const auto token = clang::Lexer::getSourceText(
            clang::CharSourceRange::getTokenRange(location, location), source, compiler.getLangOpts());
        if (token != "__grid_constant__") { reject(); return; }
        const auto file = source.getFileID(location);
        const unsigned begin = source.getFileOffset(location), end = begin + token.size();
        for (const auto &range : SemanticRewriteRanges)
          if (range.file == file && begin < range.endOffset && range.beginOffset < end) { reject(); return; }
        std::string error;
        if (!llcompat::insertReplacement(staged, ct::Replacement(source, location, token.size(), ""), &error)) {
          reject(); return;
        }
        admitted.push_back({file, begin, end});
      }
    }
  }
  *replacements = std::move(staged);
  SemanticRewriteRanges.insert(SemanticRewriteRanges.end(), admitted.begin(), admitted.end());
  llvm::errs() << "Ascify grid-constant value-only specialization: " << admitted.size()
               << " qualifier tokens; const parameters, body and launch preserved\n";
}

void AscifyAction::finalizeCudaAttributeGuards() {
  if (cudaAttributeControlModified) return;
  auto &source = getCompilerInstance().getSourceManager();
  ct::Replacements staged = *replacements;
  std::vector<SemanticRewriteRange> admitted;
  for (const auto &guard : cudaAttributeGuards) {
    if (!guard.functionProven) continue;
    const auto file = source.getFileID(guard.begin);
    const unsigned begin = source.getFileOffset(guard.begin);
    const unsigned end = source.getFileOffset(guard.end);
    for (const auto &range : SemanticRewriteRanges)
      if (range.file == file && begin < range.endOffset && range.beginOffset < end)
        return;
    const ct::Replacement replacement(source, guard.begin, end - begin,
        "#if 1 /* Ascify: proven CUDA host/device attributes */");
    std::string error;
    if (!llcompat::insertReplacement(staged, replacement, &error)) return;
    admitted.push_back({file, begin, end});
  }
  if (admitted.empty()) return;
  *replacements = std::move(staged);
  SemanticRewriteRanges.insert(SemanticRewriteRanges.end(), admitted.begin(), admitted.end());
  llvm::errs() << "Ascify attribute-only CUDA guards: " << admitted.size() << "\n";
}

void AscifyAction::Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD) {
  if (MacroNameTok.getIdentifierInfo())
    recordDirectImageUtilityControl(MacroNameTok.getIdentifierInfo()->getName(),
        "ifndef", MacroNameTok.getLocation(), MD.getMacroInfo());
  if (MacroNameTok.getIdentifierInfo() && originalCgControlName(MacroNameTok.getIdentifierInfo()->getName()))
    originalCgControlChanged = true;
  ConditionalDirectiveEntered();
  auditExternalNvidiaSampleHelperPreprocessorUse(
      Loc, MacroNameTok, "#ifndef");
  auditFrozenNvidiaSampleHelperMacroDependency(
      Loc, MacroNameTok, MD, "#ifndef");
}

void AscifyAction::Ifdef(clang::SourceLocation Loc,
                         const clang::Token &MacroNameTok,
                         const clang::MacroDefinition &MD) {
  if (MacroNameTok.getIdentifierInfo())
    recordDirectImageUtilityControl(MacroNameTok.getIdentifierInfo()->getName(),
        "ifdef", MacroNameTok.getLocation(), MD.getMacroInfo());
  if (MacroNameTok.getIdentifierInfo() && originalCgControlName(MacroNameTok.getIdentifierInfo()->getName()))
    originalCgControlChanged = true;
  recordCudaAttributeGuard(Loc, MacroNameTok, MD);
  recordInactiveNvidiaSampleHelperBranch(Loc, MacroNameTok, MD);
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
  if (MacroNameTok.getIdentifierInfo())
    recordDirectImageUtilityControl(MacroNameTok.getIdentifierInfo()->getName(),
        directive, MacroNameTok.getLocation(), MD.getMacroInfo());
  auditExternalNvidiaSampleHelperPreprocessorUse(
      Loc, MacroNameTok, directive);
  auditFrozenNvidiaSampleHelperMacroDependency(
      Loc, MacroNameTok, MD, directive);
}

void AscifyAction::ElifdefSkipped(
    clang::SourceLocation Loc,
    clang::SourceRange ConditionRange,
    llvm::StringRef directive) {
  recordDirectImageUtilityControl("", directive, Loc, nullptr);
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
  if (MacroNameTok.getIdentifierInfo())
    recordDirectImageUtilityControl(MacroNameTok.getIdentifierInfo()->getName(),
        "defined", MacroNameTok.getLocation(), MD.getMacroInfo());
  if (MacroNameTok.getIdentifierInfo() &&
      originalCgControlName(MacroNameTok.getIdentifierInfo()->getName()))
    originalCgControlChanged = true;
  auditExternalNvidiaSampleHelperPreprocessorUse(
      Range.getBegin(), MacroNameTok, "defined");
  auditFrozenNvidiaSampleHelperMacroDependency(
      Range.getBegin(), MacroNameTok, MD, "defined");
}

void AscifyAction::MacroUndefined(
    const clang::Token &MacroNameTok,
    const clang::MacroDefinition &MD) {
  if (StandaloneDeviceExecutable && MacroNameTok.getIdentifierInfo())
    deviceAggregateMacroEvidence.controls.push_back({MacroNameTok.getIdentifierInfo()->getName().str(),MacroNameTok.getLocation()});
  if (MacroNameTok.getIdentifierInfo() &&
      isDynamicWarpControlledToken(MacroNameTok.getIdentifierInfo()->getName()))
    dynamicWarpMacroEvents.emplace_back(MacroNameTok.getIdentifierInfo()->getName().str(),
                                        MacroNameTok.getLocation());
  if (MacroNameTok.getIdentifierInfo())
    recordDirectImageUtilityControl(MacroNameTok.getIdentifierInfo()->getName(),
        "undef", MacroNameTok.getLocation(), MD.getMacroInfo());
  if (MacroNameTok.getIdentifierInfo() != nullptr) {
    const auto name = MacroNameTok.getIdentifierInfo()->getName();
    if (originalCgControlName(name)) originalCgControlChanged = true;
    ++profilerMacroUndefinitions[name.str()];
    if (name == "__has_include") profilerAvailabilityControlModified = true;
  }
  if (frontendCompatibility.enabled() &&
      MacroNameTok.getIdentifierInfo() != nullptr &&
      MacroNameTok.getIdentifierInfo()->getName() ==
          "ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_")
    frontendHostMathGuardModified = true;
  if (frontendCompatibility.enabled() &&
      MacroNameTok.getIdentifierInfo() != nullptr &&
      frontendCgControlName(MacroNameTok.getIdentifierInfo()->getName()) &&
      !trustedFrontendCgControlOwner(
          MacroNameTok.getIdentifierInfo()->getName(), MacroNameTok.getLocation(),
          getCompilerInstance(), frontendCompatibility))
    frontendCgControlModified = true;
  if (MacroNameTok.getIdentifierInfo() != nullptr &&
      MacroNameTok.getIdentifierInfo()->getName() == "__CUDACC__") {
    auto &compiler = getCompilerInstance();
    auto &source = compiler.getSourceManager();
    if (!isOwnedCudaCompilerInput(source.getFileID(
            source.getSpellingLoc(MacroNameTok.getLocation())), compiler))
      cudaAttributeControlModified = true;
  }
  if (MacroNameTok.getIdentifierInfo() != nullptr &&
      MacroNameTok.getIdentifierInfo()->getName() == "__STRICT_ANSI__")
    cudaRuntimeApiStrictControlModified = true;
  if (MacroNameTok.getIdentifierInfo() != nullptr &&
      MacroNameTok.getIdentifierInfo()->getName() == "__MVS__")
    cudaRuntimeApiMvsControlModified = true;
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
  if (StandaloneDeviceExecutable)
    deviceAggregateMacroEvidence.controls.push_back({name.str(),MacroNameTok.getLocation()});
  if (isDynamicWarpControlledToken(name))
    dynamicWarpMacroEvents.emplace_back(name.str(),MacroNameTok.getLocation());
  recordDirectImageUtilityControl(name, "define", MacroNameTok.getLocation(),
      getCompilerInstance().getPreprocessor().getMacroInfo(MacroNameTok.getIdentifierInfo()));
  if (originalCgControlName(name) &&
      !(name == "_CG_CLUSTER_INTRINSICS_AVAILABLE" &&
        proveOriginalCgCompilerDefault(getCompilerInstance().getPreprocessor().getMacroInfo(
            MacroNameTok.getIdentifierInfo()), MacroNameTok.getLocation(), true)))
    originalCgControlChanged = true;
  ++profilerMacroDefinitions[name.str()];
  if (name == "__has_include") {
    const auto *info = getCompilerInstance().getPreprocessor().getMacroInfo(
        MacroNameTok.getIdentifierInfo());
    if (info == nullptr || !info->isBuiltinMacro())
      profilerAvailabilityControlModified = true;
  }
  if (name == "__attribute__" || name == "always_inline")
    deviceTemplateAttributeMacroConflict = true;
  if (name == "inline" || name == "__attribute__" || name == "always_inline" ||
      name == "static_assert" || name == "sizeof" || name == "alignof")
    deviceAggregateTokenMacroConflict = true;
  if (name == "__MVS__")
    cudaRuntimeApiMvsControlModified = true;
  // Rewritten builtin tokens are preprocessed again by the target compiler.
  // Refuse a translation unit that gives that output name macro semantics.
  if (name == "__builtin_fmaxf" || name == "__builtin_elementwise_min" ||
      name == "__builtin_elementwise_max")
    frontendHostMathBuiltinMacroDefined = true;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const auto definitionFile = sourceManager.getFileID(
      sourceManager.getSpellingLoc(MacroNameTok.getLocation()));
  if (frontendCompatibility.enabled() &&
      name == "ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_" &&
      !verifiedForcedFrontendHostMathFile(
          definitionFile, getCompilerInstance(), frontendCompatibility))
    frontendHostMathGuardModified = true;
  if (frontendCompatibility.enabled() && frontendCgControlName(name) &&
      !trustedFrontendCgControlOwner(name, MacroNameTok.getLocation(),
                                    getCompilerInstance(), frontendCompatibility))
    frontendCgControlModified = true;
  if (name == "__CUDACC__" &&
      (definitionFile != getCompilerInstance().getPreprocessor().getPredefinesFileID() ||
       !sourceManager.isWrittenInBuiltinFile(MacroNameTok.getLocation())) &&
      !isOwnedCudaCompilerInput(definitionFile, getCompilerInstance()))
    cudaAttributeControlModified = true;
  if (name == "__STRICT_ANSI__" &&
      (definitionFile != getCompilerInstance().getPreprocessor().getPredefinesFileID() ||
       !sourceManager.isWrittenInBuiltinFile(MacroNameTok.getLocation())))
    cudaRuntimeApiStrictControlModified = true;
  if (definitionFile.isValid() &&
      cudaRuntimeApiSdkFileIds.count(definitionFile.getHashValue()))
    cudaRuntimeApiSdkMacros.insert(name.str());
  else if (const auto *info = getCompilerInstance().getPreprocessor().getMacroInfo(
               MacroNameTok.getIdentifierInfo())) {
    auto &dependencies = cudaRuntimeApiRetainedMacroDependencies[name.str()];
    for (const auto &token : info->tokens()) {
      if (token.is(clang::tok::hashhash))
        cudaRuntimeApiRetainedPasteMacros.insert(name.str());
      if (token.getIdentifierInfo() == nullptr)
        continue;
      bool parameter = false;
      if (info->isFunctionLike())
        for (const auto *argument : info->params())
          parameter |= argument == token.getIdentifierInfo();
      if (!parameter)
        dependencies.insert(token.getIdentifierInfo()->getName().str());
    }
  }
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
      FrozenOfficialNvidiaSampleHelperTimer |
      FrozenOfficialNvidiaSampleHelperDrvapi;
  const bool preservedCallerMax = name == "MAX" &&
      recordPreservedMaxDefinition(MacroNameTok);
  if (isNvidiaSampleHelperObservableMacro(name) && !fromCompat &&
      !preservedCallerMax &&
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
      FrozenOfficialNvidiaSampleHelperTimer |
      FrozenOfficialNvidiaSampleHelperDrvapi;
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

void AscifyAction::observeCudaRuntimeApiMacroUse(
    const clang::Token &token, const clang::MacroDefinition &definition) {
  if (token.getIdentifierInfo() == nullptr)
    return;
  auto &source = getCompilerInstance().getSourceManager();
  const auto location = source.getExpansionLoc(token.getLocation());
  if (location.isInvalid())
    return;
  const auto file = source.getFileID(location);
  // A consumed direct qualifier owns its official internal expansion chain,
  // e.g. __global__ -> __annotate__. This is not an independent SDK macro
  // observation. Conditional and replacement-list uses still fail raw audit.
  if (token.getLocation().isMacroID()) {
    clang::Token outer;
    if (!clang::Lexer::getRawToken(location, outer, source,
                                  getCompilerInstance().getLangOpts(), true)) {
      const auto spelling = clang::Lexer::getSpelling(
          outer, source, getCompilerInstance().getLangOpts());
      if (spelling == "__global__" || spelling == "__device__" ||
          spelling == "__shared__" || spelling == "__forceinline__" ||
          spelling == "__align__")
        return;
    }
  }
  if (!file.isValid() || cudaRuntimeApiSdkFileIds.count(file.getHashValue()))
    return;
  const auto identity = file.getHashValue();
  auto ownership = cudaRuntimeApiCompilerInputAtExpansion.find(identity);
  if (ownership == cudaRuntimeApiCompilerInputAtExpansion.end())
    ownership = cudaRuntimeApiCompilerInputAtExpansion.emplace(
        identity, isOwnedCudaCompilerInput(file, getCompilerInstance())).first;
  if (!ownership->second) {
    const auto name = token.getIdentifierInfo()->getName();
    // Preserve the individual use until the final FileID/control audit. A
    // same-named observation in caller code must never inherit this origin.
    if (frontendCompatibility.enabled() &&
        source.getFilename(location) ==
            frontendCompatibility.canonicalRoot + "/cooperative_groups.h" &&
        verifiedFrontendCgEntryFileIds.count(identity) &&
        frontendCgMacroExpansionMatches(name, definition.getMacroInfo(), file,
            getCompilerInstance(), frontendCompatibility)) {
      cudaRuntimeApiCgMacroObservations.push_back({name.str(), file, location});
    } else {
      cudaRuntimeApiExternalMacroUses.insert(name.str());
    }
  }
}

bool AscifyAction::auditCudaProfilerApiSurface() {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  const auto reject = [&](clang::SourceLocation location) {
    const auto diagnostic = compiler.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "CUDA profiler header boundary: observes removed profiler header surface");
    compiler.getDiagnostics().Report(location, diagnostic);
    return false;
  };
  if (cudaRuntimeApiSdkIdentityFailure.isValid())
    return reject(cudaRuntimeApiSdkIdentityFailure);
  if (cudaRuntimeApiExternalMacroUses.count("__CUDA_PROFILER_API_H__"))
    return reject({});
  for (auto info = source.fileinfo_begin(); info != source.fileinfo_end(); ++info) {
    const auto file = source.translateFile(info->first);
    if (file.isInvalid() || cudaRuntimeApiSdkFileIds.count(file.getHashValue()))
      continue;
    const auto begin = source.getLocForStartOfFile(file);
    if (isOwnedCudaCompilerInput(file, compiler, true) ||
        productStdargFromNativeStdio(file, compiler) ||
        frozenNvidiaSampleHelperFileRole(source, begin) != 0)
      continue;
    const auto *entry = source.getFileEntryForID(file);
    if (source.isInSystemHeader(begin) && entry != nullptr &&
        !source.isFileOverridden(entry) &&
        physicalSourceFileMatches(source.getFilename(begin), *entry, source) &&
        (pathIsWithinPhysicalDirectory(source.getFilename(begin), "/usr/include") ||
         pathIsWithinPhysicalDirectory(source.getFilename(begin), "/usr/lib/gcc")))
      continue;
    bool invalid = false;
    const auto data = source.getBufferData(file, &invalid);
    if (invalid) { return reject(begin); }
    clang::Lexer lexer(begin, compiler.getLangOpts(), data.begin(), data.begin(), data.end());
    clang::Token token;
    bool previousHash = false;
    lexer.LexFromRawLexer(token);
    while (!token.is(clang::tok::eof)) {
      if (token.is(clang::tok::hashhash)) { return reject(token.getLocation()); }
      const bool followsHash = previousHash;
      previousHash = token.is(clang::tok::hash);
      if (!profilerAvailabilityCandidates.empty() && followsHash &&
          token.is(clang::tok::numeric_constant)) return reject(token.getLocation());
      if (token.isAnyIdentifier()) {
        const auto name = clang::Lexer::getSpelling(token, source, compiler.getLangOpts());
        if (!profilerAvailabilityCandidates.empty() && followsHash && name == "line")
          return reject(token.getLocation());
        for (const auto &candidate : profilerAvailabilityCandidates)
          if (!candidate.flag.empty() && name == candidate.flag &&
              file != candidate.file) return reject(token.getLocation());
        if (name == "__has_include" &&
            isProvedProfilerAvailabilityToken(token.getLocation())) {
          lexer.LexFromRawLexer(token);
          continue;
        }
        if (name == "__CUDA_PROFILER_API_H__" || name == "__has_include" ||
            name == "__has_include_next" || name == "_Pragma" ||
            name == "__pragma" || name == "push_macro" || name == "pop_macro" ||
            (!profilerAvailabilityCandidates.empty() && name == "__LINE__" &&
             file == profilerAvailabilityCandidates.front().file))
          return reject(token.getLocation());
      } else if (token.is(clang::tok::string_literal)) {
        const auto literal = clang::Lexer::getSpelling(token, source, compiler.getLangOpts());
        if (literal.find("__CUDA_PROFILER_API_H__") != std::string::npos)
          return reject(token.getLocation());
      }
      lexer.LexFromRawLexer(token);
    }
  }
  return true;
}

bool AscifyAction::auditCudaRuntimeApiSurface(bool includeRecords) {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  const auto reject = [&](clang::SourceLocation location,
                          llvm::StringRef reason, llvm::StringRef name) {
    const auto diagnostic = compiler.getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "CUDA runtime-API header boundary: %0 '%1'");
    compiler.getDiagnostics().Report(location, diagnostic) << reason << name;
    return false;
  };
  if (cudaRuntimeApiSdkIdentityFailure.isValid())
    return reject(cudaRuntimeApiSdkIdentityFailure,
                  "cannot prove physical SDK dependency contents",
                  source.getFilename(cudaRuntimeApiSdkIdentityFailure));
  const auto isSdkMacro = [&](llvm::StringRef name) {
    return cudaRuntimeApiSdkMacros.count(name.str()) != 0;
  };
  const auto isCudaCompilerCondition = [](llvm::StringRef name) {
    return name.starts_with("__CUDA") || name.starts_with("__NVCC");
  };
  const auto admittedCodeMacro = [&](llvm::StringRef name) {
    const auto mapped = CUDA_RENAMES_MAP().find(name);
    return mapped != CUDA_RENAMES_MAP().end() &&
           !Statistics::isUnsupported(mapped->second);
  };
  const auto provedCgFile = [&](clang::FileID file) {
    return file.isValid() && !frontendCgControlModified &&
        !cudaAttributeControlModified &&
        verifiedFrontendCgEntryFileIds.count(file.getHashValue()) &&
        !macroAlteredFrontendCgFileIds.count(file.getHashValue()) &&
        source.getFilename(source.getLocForStartOfFile(file)) ==
            frontendCompatibility.canonicalRoot + "/cooperative_groups.h" &&
        verifiedFrontendCgFile(file, compiler, frontendCompatibility);
  };
  const auto provedHostMathFile = [&](clang::FileID file) {
    return ordinaryRuntimeProfileInput(compiler) && file.isValid() &&
        !frontendHostMathGuardModified && !frontendCgControlModified &&
        !cudaAttributeControlModified &&
        verifiedFrontendHostMathEntryFileIds.count(file.getHashValue()) &&
        !macroAlteredFrontendHostMathFileIds.count(file.getHashValue()) &&
        verifiedForcedFrontendHostMathFile(file, compiler, frontendCompatibility);
  };
  if (frontendCompatibility.enabled() &&
      (!ordinaryRuntimeProfileInput(compiler) || frontendHostMathGuardModified ||
       verifiedFrontendHostMathEntryFileIds.empty()))
    return reject({}, "unproved forced host-math preprocessing", "host_math.h");
  for (const auto &observation : cudaRuntimeApiCgMacroObservations) {
    if (!provedCgFile(observation.file))
      return reject(observation.location,
                    "unproved product CG macro provenance", observation.name);
  }
  std::set<std::string> unprovedRetainedMacros = cudaRuntimeApiRetainedPasteMacros;
  unprovedRetainedMacros.insert(cudaRuntimeApiSdkMacros.begin(),
                               cudaRuntimeApiSdkMacros.end());
  for (const auto &macro : cudaRuntimeApiRetainedMacroDependencies)
    for (const auto &dependency : macro.second)
      if (isCudaCompilerCondition(dependency))
        unprovedRetainedMacros.insert(dependency);
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto &macro : cudaRuntimeApiRetainedMacroDependencies)
      for (const auto &dependency : macro.second)
        if (unprovedRetainedMacros.count(dependency) &&
            unprovedRetainedMacros.insert(macro.first).second)
          changed = true;
  }
  // Expansion callbacks also expose SDK macro names assembled by ##. Raw
  // scanning below covers inactive branches, #if/#ifdef/defined, replacement
  // lists, undefinitions and string operands to pragma macro-stack operations.
  for (const auto &name : cudaRuntimeApiExternalMacroUses) {
    if (isSdkMacro(name) && !admittedCodeMacro(name))
      return reject({}, "observes removed SDK macro", name);
    if (isCudaCompilerCondition(name) && !admittedCodeMacro(name))
      return reject({}, "observes CUDA compiler condition", name);
  }
  auto &preprocessor = compiler.getPreprocessor();
  const auto *strict = preprocessor.getMacroInfo(
      preprocessor.getIdentifierInfo("__STRICT_ANSI__"));
  bool strictModeInvariant = !cudaRuntimeApiStrictControlModified &&
      compiler.getLangOpts().CPlusPlus17 && !compiler.getLangOpts().GNUMode &&
      strict != nullptr && strict->getNumTokens() == 1 &&
      source.getFileID(strict->getDefinitionLoc()) == preprocessor.getPredefinesFileID() &&
      source.isWrittenInBuiltinFile(strict->getDefinitionLoc()) &&
      clang::Lexer::getSpelling(strict->getReplacementToken(0), source,
                               compiler.getLangOpts()) == "1";
  bool nativeMvsAbsent = !cudaRuntimeApiMvsControlModified &&
      compiler.getTarget().getTriple().isOSLinux() &&
      preprocessor.getMacroInfo(preprocessor.getIdentifierInfo("__MVS__")) == nullptr;
  bool nativeMvsControlPolluted = cudaRuntimeApiMvsControlModified;
  for (const auto &macro : compiler.getPreprocessorOpts().Macros) {
    const llvm::StringRef definition(macro.first);
    if (definition.split('=').first == "__STRICT_ANSI__" ||
        definition.contains('\n') || definition.contains('\r'))
      strictModeInvariant = false;
    if (definition.split('=').first == "__MVS__" ||
        definition.contains('\n') || definition.contains('\r'))
      nativeMvsAbsent = false;
    if (definition.split('=').first == "__MVS__")
      nativeMvsControlPolluted = true;
  }
  if (nativeMvsControlPolluted)
    return reject({}, "observes native stdarg control", "__MVS__");
  // This is a boolean proof of one native-library condition, not a blanket
  // exemption for __CUDACC__. Both the CUDA parser and the documented strict
  // C++17 target compile define __STRICT_ANSI__, so the first conjunct is
  // false independently of CUDA and native float128 macro values.
  const auto isFalseNativeStrictCondition = [&](llvm::StringRef data,
                                                unsigned offset) {
    if (!strictModeInvariant || offset >= data.size())
      return false;
    const auto begin = data.rfind('\n', offset);
    const auto end = data.find('\n', offset);
    const auto line = data.slice(begin == llvm::StringRef::npos ? 0 : begin + 1,
                                end == llvm::StringRef::npos ? data.size() : end);
    std::string compact;
    for (const char character : line)
      if (character != ' ' && character != '\t' && character != '\r')
        compact += character;
    return compact == "#if!defined(__STRICT_ANSI__)&&defined(_GLIBCXX_USE_FLOAT128)&&!defined(__CUDACC__)";
  };
  const auto isFalseNativeStdargCondition = [&](llvm::StringRef data,
                                                unsigned offset) {
    if (!nativeMvsAbsent || offset >= data.size())
      return false;
    const auto begin = data.rfind('\n', offset);
    const auto end = data.find('\n', offset);
    const auto line = data.slice(begin == llvm::StringRef::npos ? 0 : begin + 1,
                                end == llvm::StringRef::npos ? data.size() : end);
    std::string compact;
    for (const char character : line)
      if (character != ' ' && character != '\t' && character != '\r')
        compact += character;
    return compact == "#ifdefined(__MVS__)&&__has_include_next(<stdarg.h>)";
  };
  std::set<unsigned> auditedFiles;
  std::set<unsigned> nativeFiles;
  // Preserve the original fileinfo inventory exactly. The only additional
  // inclusion instances needed by A1 are authenticated entries of the exact
  // main product CG header. Adding every entered file would also rescan native
  // and compiler headers under different include ancestry, changing unrelated
  // Runtime rules. Synthetic buffers and all non-CG entries retain their old
  // inventory and ownership rules; macro observations remain checked above.
  std::set<clang::FileID> inputFiles;
  for (auto info = source.fileinfo_begin(); info != source.fileinfo_end(); ++info)
    inputFiles.insert(source.translateFile(info->first));
  for (const auto file : enteredSourceFileIds)
    if (file.isValid() &&
        verifiedFrontendCgEntryFileIds.count(file.getHashValue()) &&
        source.getFilename(source.getLocForStartOfFile(file)) ==
            frontendCompatibility.canonicalRoot + "/cooperative_groups.h" &&
        verifiedFrontendCgFile(file, compiler, frontendCompatibility))
      inputFiles.insert(file);
  // Extend O only by independently authenticated forced host-math entries H.
  // In particular, do not rescan arbitrary native/compiler/synthetic entries.
  for (const auto file : enteredSourceFileIds)
    if (provedHostMathFile(file)) inputFiles.insert(file);
  for (const auto file : inputFiles) {
    if (file.isInvalid() || cudaRuntimeApiSdkFileIds.count(file.getHashValue()))
      continue;
    const auto begin = source.getLocForStartOfFile(file);
    // Product-matching bytes and the actual forced-wrapper include ancestry
    // establish compiler ownership. Extra files under a caller ResourceDir
    // remain audited, including -include and -isystem inputs.
    if (isOwnedCudaCompilerInput(file, compiler))
      continue;
    auditedFiles.insert(file.getHashValue());
    const auto *entry = source.getFileEntryForID(file);
    const bool physicalNativeSystemHeader =
        source.isInSystemHeader(begin) && entry != nullptr &&
        !source.isFileOverridden(entry) &&
        physicalSourceFileMatches(source.getFilename(begin), *entry, source) &&
        (pathIsWithinPhysicalDirectory(source.getFilename(begin), "/usr/include") ||
         pathIsWithinPhysicalDirectory(source.getFilename(begin), "/usr/lib/gcc"));
    if (physicalNativeSystemHeader)
      nativeFiles.insert(file.getHashValue());
    const bool productNativeStdarg = productStdargFromNativeStdio(file, compiler);
    bool invalid = false;
    const auto data = source.getBufferData(file, &invalid);
    if (invalid)
      return reject(begin, "cannot audit retained file", source.getFilename(begin));
    const bool frozenCg = provedCgFile(file);
    const bool frozenHostMath = provedHostMathFile(file);
    const auto provedCgControlToken = [&](llvm::StringRef name,
                                          clang::SourceLocation location) {
      if (!frozenCg) return false;
      // These positions belong to the exact authenticated product bytes.
      // Do not exempt the file: all other tokens retain the ordinary audit.
      if (name == "ASCIFY_FRONTEND_COMPAT_DEVICE_") return true;
      const auto offset = source.getFileOffset(location);
      const auto start = data.rfind('\n', offset);
      const auto end = data.find('\n', offset);
      const auto line = data.slice(start == llvm::StringRef::npos ? 0 : start + 1,
                                  end == llvm::StringRef::npos ? data.size() : end);
      return ((name == "__CUDACC__" || name == "__CUDA__") &&
              line == "#if defined(__CUDACC__) || defined(__CUDA__)") ||
          (name == "__device__" &&
           line == "#define ASCIFY_FRONTEND_COMPAT_DEVICE_ __device__");
    };
    const auto provedHostMathControlToken = [&](llvm::StringRef name,
                                                clang::SourceLocation location) {
      if (!frozenHostMath || (name != "__CUDACC__" && name != "__CUDA__"))
        return false;
      const auto offset = source.getFileOffset(location);
      const auto start = data.rfind('\n', offset), end = data.find('\n', offset);
      return data.slice(start == llvm::StringRef::npos ? 0 : start + 1,
                        end == llvm::StringRef::npos ? data.size() : end) ==
          "#if defined(__CUDACC__) || defined(__CUDA__)";
    };
    clang::Lexer lexer(begin, compiler.getLangOpts(), data.begin(), data.begin(), data.end());
    clang::Token token;
    bool directive = false;
    bool pragmaOperand = false;
    bool pragmaDirective = false;
    lexer.LexFromRawLexer(token);
    while (!token.is(clang::tok::eof)) {
      if (token.isAtStartOfLine()) {
        directive = token.is(clang::tok::hash);
        pragmaDirective = false;
      }
      if (token.is(clang::tok::hashhash) && !physicalNativeSystemHeader)
        return reject(token.getLocation(), "unproved retained token paste", "##");
      if (token.is(clang::tok::r_paren))
        pragmaOperand = false;
      if (token.isAnyIdentifier()) {
        const auto name = clang::Lexer::getSpelling(token, source, compiler.getLangOpts());
        if (frontendCompatibility.enabled() && !frozenHostMath &&
            name == "ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_")
          return reject(token.getLocation(), "observes forced host-math control", name);
        if (frontendCompatibility.enabled() && !physicalNativeSystemHeader &&
            (name == "_Pragma" || name == "__pragma" ||
             name == "push_macro" || name == "pop_macro"))
          return reject(token.getLocation(), "unproved forced host-math macro-stack surface", name);
        if (name == "_Pragma" || name == "__pragma")
          pragmaOperand = true;
        if (directive && name == "pragma")
          pragmaDirective = true;
        const bool provedFalseStdargInquiry = productNativeStdarg && directive &&
            name == "__has_include_next" &&
            isFalseNativeStdargCondition(data, source.getFileOffset(token.getLocation()));
        const bool provedProfilerInquiry = name == "__has_include" &&
            isProvedProfilerAvailabilityToken(token.getLocation());
        if (!physicalNativeSystemHeader && !provedFalseStdargInquiry &&
            !provedProfilerInquiry &&
            (name == "__has_include" || name == "__has_include_next" ||
             name == "__has_feature" || name == "__has_extension" ||
             name == "__has_builtin" || name == "__has_attribute" ||
             name == "__has_cpp_attribute" || name == "__is_identifier" ||
             name == "__is_target_arch" || name == "__is_target_environment"))
          return reject(token.getLocation(), "unproved retained compiler inquiry", name);
        if (!physicalNativeSystemHeader && directive &&
            (name == "__builtin_classify_type" || name == "__builtin_object_size" ||
             name == "__builtin_dynamic_object_size" || name == "__builtin_dump_struct" ||
             name == "__builtin_preserve_access_index"))
          return reject(token.getLocation(), "unproved retained compiler SDK inquiry", name);
        const bool provedFalseNativeCondition = physicalNativeSystemHeader &&
            directive && name == "__CUDACC__" &&
            isFalseNativeStrictCondition(data, source.getFileOffset(token.getLocation()));
        const bool provedCgControl = provedCgControlToken(name, token.getLocation()) ||
            provedHostMathControlToken(name, token.getLocation());
        if (!provedFalseNativeCondition && !provedCgControl &&
            isSdkMacro(name) && (directive || !admittedCodeMacro(name)))
          return reject(token.getLocation(), "observes removed SDK macro", name);
        if (!provedFalseNativeCondition && !provedCgControl && isCudaCompilerCondition(name) &&
            (directive || !admittedCodeMacro(name)))
          return reject(token.getLocation(), "observes CUDA compiler condition", name);
        // A caller need not spell ## itself: e.g. an unexpanded replacement
        // may call glibc __CONCAT through multiple aliases. Native paste
        // implementation files are retained, but their caller-side use has
        // no proof of the removed SDK observation or future expansion domain.
        if (!physicalNativeSystemHeader && !provedCgControl &&
            unprovedRetainedMacros.count(name) &&
            (directive || !admittedCodeMacro(name)))
          return reject(token.getLocation(), "unproved retained macro dependency", name);
      } else if ((directive || pragmaOperand) && token.is(clang::tok::string_literal)) {
        const auto literal = clang::Lexer::getSpelling(token, source, compiler.getLangOpts());
        if (frontendCompatibility.enabled() && !frozenHostMath &&
            literal.find("ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_") !=
                std::string::npos)
          return reject(token.getLocation(), "observes forced host-math control in string", literal);
        if ((pragmaOperand || pragmaDirective) && !physicalNativeSystemHeader &&
            literal.find('\\') != std::string::npos)
          return reject(token.getLocation(), "unproved escaped pragma string", literal);
        for (const auto &name : cudaRuntimeApiSdkMacros) {
          if (literal.find(name) != std::string::npos)
            return reject(token.getLocation(), "observes SDK macro in pragma/replacement string", name);
        }
        if (literal.find("__CUDA") != std::string::npos ||
            literal.find("__NVCC") != std::string::npos)
          return reject(token.getLocation(), "observes CUDA compiler condition in string", literal);
      }
      lexer.LexFromRawLexer(token);
    }
  }
  return !includeRecords || auditCudaRuntimeApiRecords(false);
}

bool AscifyAction::auditCudaRuntimeApiRecords(bool textureOnly) {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  std::set<unsigned> auditedFiles, nativeFiles;
  std::set<unsigned> frozenTextureUtilityFiles;
  std::set<unsigned> frozenTextureCgFiles;
  std::set<unsigned> frozenRuntimePatternFiles;
  // A physical file may be entered more than once with different include
  // parents or macro contexts. translateFile returns only its first FileID;
  // retain that inventory, but audit every observed inclusion independently.
  std::set<clang::FileID> inputFiles = enteredSourceFileIds;
  for (auto info = source.fileinfo_begin(); info != source.fileinfo_end(); ++info) {
    inputFiles.insert(source.translateFile(info->first));
  }
  for (const auto file : inputFiles) {
    if (file.isInvalid() || source.getFileEntryForID(file) == nullptr ||
        cudaRuntimeApiSdkFileIds.count(file.getHashValue()) ||
        isOwnedCudaCompilerInput(file, compiler, textureOnly)) continue;
    auditedFiles.insert(file.getHashValue());
    const auto begin = source.getLocForStartOfFile(file);
    if (!textureOnly && ordinaryRuntimeProfileInput(compiler) &&
        !frontendCgControlModified && !cudaAttributeControlModified) {
      if (!frontendHostMathGuardModified &&
          verifiedFrontendHostMathEntryFileIds.count(file.getHashValue()) &&
          !macroAlteredFrontendHostMathFileIds.count(file.getHashValue()) &&
          verifiedForcedFrontendHostMathFile(file, compiler, frontendCompatibility))
        frozenRuntimePatternFiles.insert(file.getHashValue());
      // Full Runtime admission uses only the main CG header whose macro/raw
      // provenance was proved by auditCudaRuntimeApiSurface. reduce.h remains
      // outside this rule even though the texture-only projection admits it.
      if (verifiedFrontendCgEntryFileIds.count(file.getHashValue()) &&
          !macroAlteredFrontendCgFileIds.count(file.getHashValue()) &&
          source.getFilename(begin) ==
              frontendCompatibility.canonicalRoot + "/cooperative_groups.h" &&
          verifiedFrontendCgFile(file, compiler, frontendCompatibility)) {
        frozenRuntimePatternFiles.insert(file.getHashValue());
        frozenTextureCgFiles.insert(file.getHashValue());
      }
    }
    if (textureOnly) {
      if (frozenNvidiaSampleHelperFileRole(source, begin) != 0)
        frozenTextureUtilityFiles.insert(file.getHashValue());
      // CG SFINAE utilities belong to the explicitly selected product profile.
      // Admit only their physical, byte-identical inclusion with the proved
      // CUDA controls and exact device-attribute expansion chain. This set is
      // used solely for dependent utility types; concrete instances and all
      // texture layout observations retain the ordinary checks below.
      if (!frontendCgControlModified && !cudaAttributeControlModified &&
          verifiedFrontendCgEntryFileIds.count(file.getHashValue()) &&
          !macroAlteredFrontendCgFileIds.count(file.getHashValue()) &&
          verifiedFrontendCgFile(file, compiler, frontendCompatibility)) {
        frozenTextureUtilityFiles.insert(file.getHashValue());
        frozenTextureCgFiles.insert(file.getHashValue());
      }
      // exception.h is the one templated dependency outside the five existing
      // helper roles. Require its exact parsed bytes and the authenticated
      // immediate helper_functions include instance; a basename is not proof.
      const auto include = source.getIncludeLoc(file);
      bool invalid = false;
      const auto bytes = source.getBufferData(file, &invalid);
      const auto *profileEntry = source.getFileEntryForID(file);
      // The explicitly selected parser profile force-includes its own
      // type-constrained host max/min templates. These exact templates do not
      // observe texture layout. Keep the exception restricted to their actual
      // product file and forced include, with unchanged preprocessed tokens.
      // Concrete instantiations still traverse the ordinary record checks.
      if (!invalid && profileEntry != nullptr && include.isValid() &&
          source.getFileID(source.getSpellingLoc(include)) ==
              compiler.getPreprocessor().getPredefinesFileID() &&
          !macroAlteredFrontendHostMathFileIds.count(file.getHashValue()) &&
          physicalSourceFileMatches(source.getFilename(begin), *profileEntry, source) &&
          ascify::IsVerifiedFrontendHostMathInput(
              frontendCompatibility, source.getFilename(begin).str(), bytes.str()))
        frozenTextureUtilityFiles.insert(file.getHashValue());
      if (!invalid && llvm::sys::path::filename(source.getFilename(begin)) == "exception.h" &&
          include.isValid() &&
          frozenNvidiaSampleHelperFileRole(source, include) == FrozenOfficialNvidiaSampleHelperFunctions &&
          bytes.size() == 6168 && sha256Equals(bytes,
              "ad043023a652bf3610671aad88237383c7bf2e17c66f8da5830d1d668fe7bbe7"))
        frozenTextureUtilityFiles.insert(file.getHashValue());
      // The same exact utility is reached from Image without Functions in a
      // separately sealed provider graph. This authorizes only the existing
      // texture-only dependent-type rule; traversal/reflection is unchanged.
      if (isProvedDirectImageUtilityFile(file))
        frozenTextureUtilityFiles.insert(file.getHashValue());
    }
    const auto *entry = source.getFileEntryForID(file);
    if (source.isInSystemHeader(begin) && entry != nullptr &&
        !source.isFileOverridden(entry) &&
        physicalSourceFileMatches(source.getFilename(begin), *entry, source) &&
        (pathIsWithinPhysicalDirectory(source.getFilename(begin), "/usr/include") ||
         pathIsWithinPhysicalDirectory(source.getFilename(begin), "/usr/lib/gcc")))
      nativeFiles.insert(file.getHashValue());
  }
  class RecordAudit : public clang::RecursiveASTVisitor<RecordAudit> {
  public:
    RecordAudit(clang::ASTContext &context, clang::SourceManager &source,
                const std::set<unsigned> &sdk,
                const std::set<unsigned> &audited,
                const std::set<unsigned> &native, const std::set<unsigned> &frozenTextureUtilityFiles,
                const std::set<unsigned> &frozenTextureCgFiles,
                const std::set<unsigned> &frozenRuntimePatternFiles, bool textureOnly,
                const ascify::Scalar64Plan& scalar64,
                const ascify::DynamicWarpGroupStats& dynamicGroups,
                const ascify::NttpVoidCallPlan& nttpVoid,
                const ascify::HostPointerSwapPlan& hostPointerSwap)
        : context(context), source(source), sdk(sdk), audited(audited), native(native), frozenTextureUtilityFiles(frozenTextureUtilityFiles),
          frozenTextureCgFiles(frozenTextureCgFiles),
          frozenRuntimePatternFiles(frozenRuntimePatternFiles),
          textureOnly(textureOnly), scalar64(scalar64), dynamicGroups(dynamicGroups), nttpVoid(nttpVoid), hostPointerSwap(hostPointerSwap) {}
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool shouldVisitImplicitCode() const { return true; }
    bool TraverseDecl(clang::Decl *declaration) {
      const auto savedPatterns = runtimePatternDepth;
      const auto savedInstances = runtimeInstanceDepth;
      const auto savedParameters = runtimePatternParameters.size();
      const auto savedCopiedParts = runtimeCopiedPatternParts.size();
      if (declaration != nullptr && !textureOnly) {
        // A concrete class can still contain an uninstantiated member
        // template. Only its proved parameter declarations and exact primary
        // FunctionDecl enter this scope, never its specialization collection.
        const bool copiedPatternPart =
            std::find(runtimeCopiedPatternParts.begin(),
                      runtimeCopiedPatternParts.end(), declaration) !=
            runtimeCopiedPatternParts.end();
        if (copiedPatternPart) {
          runtimeInstanceDepth = 0;
          ++runtimePatternDepth;
        }
        if (const auto *member = llvm::dyn_cast<clang::FunctionTemplateDecl>(declaration))
          if (provedCopiedRuntimeMemberPattern(member)) {
            runtimeCopiedPatternParts.push_back(member->getTemplatedDecl());
            for (const auto *parameter : *member->getTemplateParameters())
              runtimeCopiedPatternParts.push_back(parameter);
          }
        const auto location = declaration->getLocation();
        const bool writtenPattern = location.isFileID() &&
            frozenRuntimePatternFiles.count(
                source.getFileID(location).getHashValue());
        if (const auto *partial = llvm::dyn_cast<
                clang::ClassTemplatePartialSpecializationDecl>(declaration)) {
          if (writtenPattern && partial->getInstantiatedFromMember() == nullptr)
            ++runtimePatternDepth;
          else
            ++runtimeInstanceDepth;
        } else if (llvm::isa<clang::ClassTemplateSpecializationDecl>(declaration) ||
                   llvm::isa<clang::VarTemplateSpecializationDecl>(declaration)) {
          ++runtimeInstanceDepth;
        } else if (writtenPattern &&
                   (llvm::isa<clang::FunctionTemplateDecl>(declaration) ||
                    llvm::isa<clang::ClassTemplateDecl>(declaration))) {
          ++runtimePatternDepth;
        }
        if (const auto *function = llvm::dyn_cast<clang::FunctionDecl>(declaration))
          if (!copiedPatternPart &&
              (function->getTemplateSpecializationKind() != clang::TSK_Undeclared ||
               function->getInstantiatedFromMemberFunction() != nullptr))
            ++runtimeInstanceDepth;
        const clang::TemplateParameterList *parameters = nullptr;
        if (writtenPattern) {
          if (const auto *partial = llvm::dyn_cast<
                  clang::ClassTemplatePartialSpecializationDecl>(declaration))
            parameters = partial->getTemplateParameters();
          else if (const auto *pattern = llvm::dyn_cast<clang::TemplateDecl>(declaration))
            parameters = pattern->getTemplateParameters();
        }
        if (parameters != nullptr)
          for (const auto *parameter : *parameters)
            if (const auto *type = llvm::dyn_cast<clang::TemplateTypeParmDecl>(parameter))
              runtimePatternParameters.push_back(type);
      }
      // Source locations of instantiated declarations still point into the
      // product header. They must not inherit the written-pattern exception.
      const bool accepted =
          clang::RecursiveASTVisitor<RecordAudit>::TraverseDecl(declaration);
      runtimePatternDepth = savedPatterns;
      runtimeInstanceDepth = savedInstances;
      runtimePatternParameters.resize(savedParameters);
      runtimeCopiedPatternParts.resize(savedCopiedParts);
      return accepted;
    }
    bool TraverseTemplateSpecializationTypeLoc(
        clang::TemplateSpecializationTypeLoc type) {
      return TraverseTemplateSpecializationTypeLoc(type, true);
    }
    bool TraverseTemplateSpecializationTypeLoc(
        clang::TemplateSpecializationTypeLoc type, bool qualifier) {
      const auto savedArguments = runtimeIdentityArguments.size();
      clang::SourceLocation argument;
      const clang::TagDecl *record = nullptr;
      if (provedRuntimeIdentityArgument(type, argument, record))
        runtimeIdentityArguments.emplace_back(argument.getRawEncoding(), record);
      auto &base = static_cast<clang::RecursiveASTVisitor<RecordAudit> &>(*this);
      const bool accepted = traverseRuntimeIdentityTypeLoc(base, type, qualifier, 0);
      runtimeIdentityArguments.resize(savedArguments);
      return accepted;
    }
    bool VisitNamespaceDecl(clang::NamespaceDecl *declaration) {
      const auto location = source.getExpansionLoc(declaration->getLocation());
      if (!textureOnly && location.isValid() &&
          frozenRuntimePatternFiles.count(source.getFileID(location).getHashValue()) &&
          declaration->getName() == "ascify_frontend_compat_detail" &&
          !declaration->getDeclContext()->isTranslationUnit()) {
        badLocation = location;
        badName = "ascify_frontend_compat_detail";
        badReason = "forced host-math header included outside translation-unit namespace";
        return false;
      }
      if (location.isValid() &&
          frozenTextureCgFiles.count(source.getFileID(location).getHashValue()) &&
          declaration->getName() == "cooperative_groups" &&
          !declaration->getDeclContext()->isTranslationUnit()) {
        badLocation = location;
        badName = "cooperative_groups";
        badReason = "product CG header included outside translation-unit namespace";
        return false;
      }
      return true;
    }
    bool TraverseInitListExpr(clang::InitListExpr *expression) {
      // An empty adapter initializer zero-initializes the published fields.
      // Its semantic form also contains SDK-only subobject initializers whose
      // source location is the caller's '{}'. Do not mistake those implicit
      // expressions for separately written SDK type observations. Nonempty
      // lists still expose SDK aggregate order and remain fully audited.
      const auto *syntax = expression->isSemanticForm()
          ? expression->getSyntacticForm() : expression;
      const auto *record = expression->getType()->getAs<clang::RecordType>();
      if (syntax != nullptr && record != nullptr) {
        const auto *declaration = record->getDecl();
        const auto definition = source.getSpellingLoc(declaration->getLocation());
        const auto name = declaration->getName();
        if (definition.isValid() &&
            sdk.count(source.getFileID(definition).getHashValue()) &&
            declaration->getDeclContext()->getRedeclContext()->isTranslationUnit() &&
            ((!textureOnly && (name == "cudaDeviceProp" || name == "cudaFuncAttributes")) ||
             ascify::IsCudaTextureRecord(declaration))) {
          if (ascify::IsCudaTextureRecord(declaration)) {
            if (ascify::IsCudaTextureZeroInitializer(syntax,context,name == "cudaMemcpy3DParms"))
              return true;
          } else if (syntax->getNumInits() == 0) {
            return true;
          }
          const auto location = source.getExpansionLoc(syntax->getBeginLoc());
          if (location.isValid() &&
              audited.count(source.getFileID(location).getHashValue())) {
            badLocation = location;
            badName = name.str();
            badReason = "unproved SDK aggregate initializer";
            return false;
          }
        }
      }
      return clang::RecursiveASTVisitor<RecordAudit>::TraverseInitListExpr(expression);
    }
    bool VisitTypeLoc(clang::TypeLoc type) {
      if (!textureOnly&&dynamicGroups.sourceTypeLocations.count(type.getOpaqueData()))return true;
      if (scalar64.scalarSignatureTypeLocs.count(type.getOpaqueData())) return true;
      // Exact signature/local T occurrences of a three-statement pointer swap.
      // Both audit routes require the same closed plan and emitted guard;
      // children/instances and inquiry operands still traverse normally.
      if (hostPointerSwap.typeLocs.count(type.getOpaqueData())) return true;
      return inspect(type.getType(), type.getBeginLoc());
    }
    bool VisitArrayTypeLoc(clang::ArrayTypeLoc type) {
      return inspectEnumNumericOperand(type.getSizeExpr(), type.getLBracketLoc());
    }
    bool VisitFieldDecl(clang::FieldDecl *field) {
      return !field->isBitField() ||
          inspectEnumNumericOperand(field->getBitWidth(), field->getLocation());
    }
    bool VisitEnumConstantDecl(clang::EnumConstantDecl *constant) {
      return inspectEnumNumericOperand(constant->getInitExpr(), constant->getLocation());
    }
    bool VisitExpr(clang::Expr *expression) {
      // These are exact current-CI nodes. Traversal and all inquiry/template
      // visitors continue; no record type, body or arbitrary callee is exempt.
      if (!textureOnly&&(dynamicGroups.scalarExpressions.count(expression)||
          dynamicGroups.valueExpressions.count(expression)||
          dynamicGroups.sourceReferences.count(expression)||
          dynamicGroups.directCallees.count(expression)||
          dynamicGroups.functorConstructions.count(expression)))return true;
      // Aggregate value-initialization of an admitted runtime adapter also
      // creates zero-initializers for SDK-only subobjects. They are not a
      // separately written SDK type observation. Explicit TypeLocs and all
      // reflection operands remain audited.
      if (llvm::isa<clang::ImplicitValueInitExpr>(expression))
        return true;
      // Exact expression/type facts from the guarded builtin scalar family,
      // never a void-NTTP, class, inquiry or whole-body exemption.
      if (scalar64.scalarExpressions.count(expression)) return true;
      // Only the exact call/lookup pair proved with every caller instance and
      // a transactional global qualification. Never a void/record/body bypass.
      if (nttpVoid.expressions.count(expression)) return true;
      // Inquiry visitors never consult this set; body traversal never stops.
      if (hostPointerSwap.expressions.count(expression)) return true;
      // A fixed scalar NTTP call can retain a dependent Expr type until
      // overload resolution, even though no SDK record can enter its type.
      // Prove only this expression; normal traversal still visits its body,
      // template arguments, reflection operands and concrete instantiations.
      // Static-assert diagnostic strings can have a null QualType.
      if (textureOnly && !expression->getType().isNull() &&
          expression->getType()->isDependentType()) {
        if (!provedNttpScalarType(expression).isNull())
          return true;
        if (const auto *lookup = llvm::dyn_cast<clang::UnresolvedLookupExpr>(expression)) {
          const auto parents = context.getParents(*lookup);
          if (parents.size() == 1)
            if (const auto *call = parents[0].get<clang::CallExpr>())
              if (call->getCallee() == lookup &&
                  !provedNttpScalarType(call).isNull())
                return true;
        }
      }
      return inspect(expression->getType(), expression->getExprLoc());
    }
    bool VisitFunctionDecl(clang::FunctionDecl *function) {
      if (textureOnly) return true;
      const auto location = source.getExpansionLoc(function->getLocation());
      if (location.isInvalid() ||
          !audited.count(source.getFileID(location).getHashValue()) ||
          native.count(source.getFileID(location).getHashValue()))
        return true;
      for (const auto *parameter : function->parameters())
        if (!inspectEnumInterface(parameter->getType(), location))
          return false;
      // Ordinary error-code returns are part of the explicit runtime API
      // contract. A template return instead exposes an unproved future
      // interface/type-selection domain.
      const auto result = function->getReturnType();
      const auto *status = result->getAs<clang::EnumType>();
      if (function->getTemplatedKind() == clang::FunctionDecl::TK_NonTemplate &&
          !llvm::isa<clang::CXXConversionDecl>(function) &&
          status != nullptr && status->getDecl()->getName() == "cudaError" &&
          status->getDecl()->getDeclContext()->isTranslationUnit()) {
        const auto declaration = source.getSpellingLoc(status->getDecl()->getLocation());
        if (declaration.isValid() &&
            sdk.count(source.getFileID(declaration).getHashValue()))
          return true;
      }
      return inspectEnumInterface(result, location);
    }
    bool VisitTypeDecl(clang::TypeDecl *declaration) {
      if (textureOnly) return true;
      const auto location = source.getExpansionLoc(declaration->getLocation());
      if (location.isInvalid() ||
          !audited.count(source.getFileID(location).getHashValue()))
        return true;
      const auto name = declaration->getName();
      const auto mapped = CUDA_RUNTIME_TYPE_NAME_MAP.find(name);
      if (name != "dim3" &&
          (mapped == CUDA_RUNTIME_TYPE_NAME_MAP.end() ||
           Statistics::isUnsupported(mapped->second)))
        return true;
      badLocation = location;
      badName = name.str();
      badReason = "conflicting retained runtime type identity";
      return false;
    }
    bool VisitTemplateSpecializationTypeLoc(clang::TemplateSpecializationTypeLoc type) {
      for (unsigned i = 0; i < type.getNumArgs(); ++i) {
        const auto argument = type.getArgLoc(i);
        if (argument.getArgument().getKind() == clang::TemplateArgument::Type &&
            !inspect(argument.getArgument().getAsType(), argument.getLocation(), false))
          return false;
      }
      return true;
    }
    bool VisitDeclRefExpr(clang::DeclRefExpr *expression) {
      if (!textureOnly || !expression->hasExplicitTemplateArgs()) return true;
      for (const auto &argument : expression->template_arguments())
        if (argument.getArgument().getKind() == clang::TemplateArgument::Type &&
            !inspect(argument.getArgument().getAsType(), argument.getLocation(), false))
          return false;
      return true;
    }
    bool VisitTypeTraitExpr(clang::TypeTraitExpr *expression) {
      for (unsigned i = 0; i < expression->getNumArgs(); ++i)
        if (!inspect(expression->getArg(i)->getType(), expression->getExprLoc(), false))
          return false;
      return true;
    }
    bool VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *expression) {
      if (ascify::IsCudaTextureZeroSize(expression,context,sdk,native)) return true;
      return inspect(expression->getTypeOfArgument(), expression->getExprLoc(), false);
    }
    bool VisitCXXTypeidExpr(clang::CXXTypeidExpr *expression) {
      if (expression->isTypeOperand())
        return inspect(expression->getTypeOperand(context), expression->getExprLoc(), false);
      return inspect(expression->getExprOperand()->getType(), expression->getExprLoc(), false);
    }
    bool VisitOffsetOfExpr(clang::OffsetOfExpr *expression) {
      return inspect(expression->getTypeSourceInfo()->getType(),
                     expression->getExprLoc(), false);
    }
    bool VisitAlignedAttr(clang::AlignedAttr *attribute) {
      if (!attribute->isAlignmentExpr() && attribute->getAlignmentType() != nullptr)
        return inspect(attribute->getAlignmentType()->getType(),
                       attribute->getLocation(), false);
      if (attribute->isAlignmentExpr())
        return inspectEnumNumericOperand(attribute->getAlignmentExpr(), attribute->getLocation());
      return true;
    }
    bool VisitCastExpr(clang::CastExpr *expression) {
      if (textureOnly) return true;
      const auto location = source.getExpansionLoc(expression->getExprLoc());
      if (location.isInvalid() ||
          !audited.count(source.getFileID(location).getHashValue()))
        return true;
      const auto *enumeration = sdkEnumeration(
          expression->getSubExpr()->IgnoreParenImpCasts()->getType());
      const auto destination = expression->getType();
      const auto *constructed = sdkEnumeration(destination);
      if (constructed != nullptr && constructed == enumeration)
        return true;
      if (constructed != nullptr && constructed != enumeration) {
        badLocation = location;
        badName = constructed->getNameAsString();
        badReason = "unproved SDK enum numeric observation";
        return false;
      }
      if (enumeration == nullptr ||
          (!destination->isIntegerType() && !destination->isRealFloatingType() &&
           !destination->isPointerType()))
        return true;
      // Error truth and symbolic equality belong to the error adapter;
      // numeric ordinals do not. In particular ACL's invalid-parameter code
      // is different from CUDA's enum ordinal 1.
      if (destination->isBooleanType() && enumeration->getName() == "cudaError")
        return true;
      if (llvm::isa<clang::ImplicitCastExpr>(expression)) {
        const auto parents = context.getParents(*expression);
        if (parents.size() == 1)
          if (const auto *comparison = parents[0].get<clang::BinaryOperator>())
            if ((comparison->getOpcode() == clang::BO_EQ ||
                 comparison->getOpcode() == clang::BO_NE) &&
                sdkEnumeration(comparison->getLHS()->IgnoreParenImpCasts()->getType()) == enumeration &&
                sdkEnumeration(comparison->getRHS()->IgnoreParenImpCasts()->getType()) == enumeration &&
                provedSdkEnumEquality(comparison, enumeration))
              return true;
      }
      badLocation = location;
      badName = enumeration->getNameAsString();
      badReason = "unproved SDK enum numeric observation";
      return false;
    }
    bool VisitBinaryOperator(clang::BinaryOperator *expression) {
      if (textureOnly) return true;
      const auto location = source.getExpansionLoc(expression->getExprLoc());
      if (location.isInvalid() ||
          !audited.count(source.getFileID(location).getHashValue()) ||
          (expression->getOpcode() != clang::BO_EQ &&
           expression->getOpcode() != clang::BO_NE))
        return true;
      const auto *enumeration = sdkEnumeration(
          expression->getLHS()->IgnoreParenImpCasts()->getType());
      if (enumeration == nullptr ||
          sdkEnumeration(expression->getRHS()->IgnoreParenImpCasts()->getType()) != enumeration ||
          provedSdkEnumEquality(expression, enumeration))
        return true;
      badLocation = location;
      badName = enumeration->getNameAsString();
      badReason = "unproved SDK enum symbolic comparison";
      return false;
    }
    bool VisitCallExpr(clang::CallExpr *expression) {
      const auto location = source.getExpansionLoc(expression->getExprLoc());
      if (location.isInvalid() ||
          !audited.count(source.getFileID(location).getHashValue()) ||
          native.count(source.getFileID(location).getHashValue()))
        return true;
      const auto *callee = expression->getDirectCallee();
      if (callee == nullptr || callee->getBuiltinID() == 0 ||
          expression->getNumArgs() == 0)
        return true;
      const auto name = callee->getName();
      inquiryVariables.clear();
      if (name == "__builtin_classify_type")
        return inspectCompilerInquiry(expression->getArg(0)->IgnoreParenImpCasts(),
                                      expression->getExprLoc(), false);
      if (name == "__builtin_object_size" ||
          name == "__builtin_dynamic_object_size" ||
          name == "__builtin_dump_struct" ||
          name == "__builtin_preserve_access_index")
        return inspectCompilerInquiry(expression->getArg(0),
                                      expression->getExprLoc(), true);
      return true;
    }
    clang::SourceLocation badLocation;
    std::string badName;
    std::string badReason = "unproved SDK record ABI";
  private:
    bool ownedRuntimeCgDeclaration(const clang::Decl *declaration) const {
      if (declaration == nullptr || !declaration->getLocation().isFileID())
        return false;
      const auto file = source.getFileID(declaration->getLocation()).getHashValue();
      return frozenRuntimePatternFiles.count(file) && frozenTextureCgFiles.count(file);
    }
    bool provedCopiedRuntimeMemberPattern(
        const clang::FunctionTemplateDecl *member) const {
      if (textureOnly || !ownedRuntimeCgDeclaration(member) ||
          member->isMemberSpecialization())
        return false;
      const auto *origin = member->getInstantiatedFromMemberTemplate();
      // One class-instantiation step is the admitted domain. Do not infer
      // ownership from a shared spelling location or chase arbitrary chains.
      if (origin == nullptr || origin == member ||
          origin->getInstantiatedFromMemberTemplate() != nullptr ||
          origin->isMemberSpecialization() || !ownedRuntimeCgDeclaration(origin))
        return false;
      const auto *owner = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
          member->getDeclContext());
      const auto *originOwner = llvm::dyn_cast<clang::CXXRecordDecl>(
          origin->getDeclContext());
      if (owner == nullptr || originOwner == nullptr ||
          owner->getSpecializationKind() != clang::TSK_ImplicitInstantiation ||
          !originOwner->isDependentContext() ||
          !ownedRuntimeCgDeclaration(owner) || !ownedRuntimeCgDeclaration(originOwner))
        return false;
      const auto *ownerPattern = owner->getTemplateInstantiationPattern();
      if (ownerPattern == nullptr ||
          ownerPattern->getCanonicalDecl() != originOwner->getCanonicalDecl())
        return false;
      // A partial specialization has a separate record redeclaration chain.
      // Authenticate the primary template as well: a caller redeclaration of
      // that template must not disappear behind an owned partial pattern.
      const auto *primary = owner->getSpecializedTemplate();
      if (!ownedRuntimeCgDeclaration(primary))
        return false;
      for (const auto *redecl : primary->redecls()) {
        if (!ownedRuntimeCgDeclaration(redecl))
          return false;
        const auto *primaryRecord = llvm::dyn_cast<clang::CXXRecordDecl>(
            redecl->getTemplatedDecl());
        if (!ownedRuntimeCgDeclaration(primaryRecord))
          return false;
        for (const auto *record : primaryRecord->redecls())
          if (!ownedRuntimeCgDeclaration(record))
            return false;
      }
      for (const auto *record : {static_cast<const clang::CXXRecordDecl *>(owner),
                                 originOwner})
        for (const auto *redecl : record->redecls())
          if (!ownedRuntimeCgDeclaration(redecl))
            return false;
      for (const auto *pattern : {member, origin}) {
        for (const auto *redecl : pattern->redecls())
          if (!ownedRuntimeCgDeclaration(redecl) || redecl->isMemberSpecialization())
            return false;
        const auto *function = pattern->getTemplatedDecl();
        if (!ownedRuntimeCgDeclaration(function) ||
            function->getDescribedFunctionTemplate() != pattern ||
            function->getTemplateSpecializationKind() != clang::TSK_Undeclared ||
            !function->getType()->isDependentType())
          return false;
        for (const auto *redecl : function->redecls())
          if (!ownedRuntimeCgDeclaration(redecl) ||
              redecl->getTemplateSpecializationKind() != clang::TSK_Undeclared)
            return false;
      }
      const auto *parameters = member->getTemplateParameters();
      const auto *originalParameters = origin->getTemplateParameters();
      if (parameters->size() == 0 || parameters->size() != originalParameters->size() ||
          member->getLocation() != origin->getLocation())
        return false;
      for (unsigned i = 0; i < parameters->size(); ++i)
        if (!ownedRuntimeCgDeclaration(parameters->getParam(i)) ||
            !ownedRuntimeCgDeclaration(originalParameters->getParam(i)) ||
            parameters->getParam(i)->getKind() != originalParameters->getParam(i)->getKind() ||
            parameters->getParam(i)->getLocation() != originalParameters->getParam(i)->getLocation())
          return false;
      return true;
    }
    bool literalRuntimeCgToken(clang::SourceLocation location,
                               llvm::StringRef spelling,
                               clang::FileID file) const {
      if (!location.isFileID() || source.getFileID(location) != file ||
          !frozenRuntimePatternFiles.count(file.getHashValue()) ||
          !frozenTextureCgFiles.count(file.getHashValue()))
        return false;
      clang::Token token;
      if (clang::Lexer::getRawToken(location, token, source,
                                   context.getLangOpts(), true) ||
          token.isNot(clang::tok::raw_identifier) || token.needsCleaning() ||
          token.getRawIdentifier() != spelling)
        return false;
      return token.getLocation() == location;
    }
    bool provedRuntimeIdentityArgument(clang::TemplateSpecializationTypeLoc type,
                                       clang::SourceLocation &argument,
                                       const clang::TagDecl *&record) const {
      // The owned CG pattern names uint4 only as a lane-type equality selector.
      // It does not instantiate a layout. Permit this exact argument during
      // this TypeLoc traversal, never the template body or future instances.
      if (textureOnly || runtimePatternDepth == 0 || runtimeInstanceDepth != 0 ||
          type.getNumArgs() != 2 || !type.getType()->isDependentType() ||
          !type.getTemplateNameLoc().isFileID())
        return false;
      const auto file = source.getFileID(type.getTemplateNameLoc());
      if (!literalRuntimeCgToken(type.getTemplateNameLoc(), "is_same", file))
        return false;
      const auto *identity = llvm::dyn_cast_or_null<clang::ClassTemplateDecl>(
          type.getTypePtr()->getTemplateName().getAsTemplateDecl());
      if (identity == nullptr || identity->getName() != "is_same" ||
          !identity->getDeclContext()->isStdNamespace())
        return false;
      for (const auto *declaration : identity->redecls()) {
        const auto location = declaration->getLocation();
        if (!location.isFileID() ||
            !native.count(source.getFileID(location).getHashValue()))
          return false;
      }
      const auto first = type.getArgLoc(0);
      const auto second = type.getArgLoc(1);
      if (first.getArgument().getKind() != clang::TemplateArgument::Type ||
          second.getArgument().getKind() != clang::TemplateArgument::Type ||
          first.getTypeSourceInfo() == nullptr || second.getTypeSourceInfo() == nullptr)
        return false;
      const auto parameterLoc = first.getTypeSourceInfo()->getTypeLoc().
          getAs<clang::TemplateTypeParmTypeLoc>();
      const auto argumentLoc = second.getTypeSourceInfo()->getTypeLoc();
      const auto recordLoc = argumentLoc.getAs<clang::RecordTypeLoc>();
      const auto aliasLoc = argumentLoc.getAs<clang::TypedefTypeLoc>();
      const clang::RecordDecl *declaration = nullptr;
      clang::SourceLocation recordUse;
      if (!recordLoc.isNull() &&
          llvm::isa<clang::RecordType>(second.getArgument().getAsType().getTypePtr())) {
        declaration = recordLoc.getTypePtr()->getDecl();
        recordUse = recordLoc.getBeginLoc();
      } else if (!aliasLoc.isNull() &&
                 llvm::isa<clang::TypedefType>(second.getArgument().getAsType().getTypePtr())) {
        // CUDA also writes `typedef struct uint4 uint4` in its physical
        // vector_types.h. This is a single authenticated SDK spelling layer,
        // not permission to desugar a caller alias or arbitrary typedef chain.
        const auto *alias = llvm::dyn_cast<clang::TypedefDecl>(
            aliasLoc.getTypePtr()->getDecl());
        if (alias == nullptr || alias->getName() != "uint4" ||
            !alias->getDeclContext()->getRedeclContext()->isTranslationUnit() ||
            alias->getTypeSourceInfo() == nullptr ||
            alias->getUnderlyingType().hasLocalQualifiers() ||
            !llvm::isa<clang::RecordType>(alias->getUnderlyingType().getTypePtr()))
          return false;
        const auto underlying = alias->getTypeSourceInfo()->getTypeLoc().
            getAs<clang::RecordTypeLoc>();
        if (underlying.isNull() || !underlying.getBeginLoc().isFileID() ||
            !sdk.count(source.getFileID(underlying.getBeginLoc()).getHashValue()))
          return false;
        for (const auto *redecl : alias->redecls()) {
          const auto location = redecl->getLocation();
          if (!location.isFileID() ||
              !sdk.count(source.getFileID(location).getHashValue()))
            return false;
        }
        declaration = underlying.getTypePtr()->getDecl();
        recordUse = aliasLoc.getBeginLoc();
      }
      if (parameterLoc.isNull() || declaration == nullptr ||
          first.getArgument().getAsType().hasLocalQualifiers() ||
          second.getArgument().getAsType().hasLocalQualifiers() ||
          !llvm::isa<clang::TemplateTypeParmType>(first.getArgument().getAsType().getTypePtr()))
        return false;
      const auto *parameter = parameterLoc.getTypePtr()->getDecl();
      if (parameter == nullptr || parameter->isParameterPack() ||
          std::find(runtimePatternParameters.begin(), runtimePatternParameters.end(),
                    parameter) == runtimePatternParameters.end() ||
          !literalRuntimeCgToken(parameter->getLocation(), parameter->getName(), file) ||
          !literalRuntimeCgToken(parameterLoc.getBeginLoc(), parameter->getName(), file))
        return false;
      const auto *definition = declaration->getDefinition();
      if (definition == nullptr || declaration->getName() != "uint4" ||
          !declaration->getDeclContext()->getRedeclContext()->isTranslationUnit() ||
          !definition->getLocation().isFileID() ||
          !sdk.count(source.getFileID(definition->getLocation()).getHashValue()) ||
          !literalRuntimeCgToken(recordUse, "uint4", file) ||
          second.getLocation() != recordUse)
        return false;
      for (const auto *redeclaration : declaration->redecls()) {
        const auto location = redeclaration->getLocation();
        if (!location.isFileID()) return false;
        if (sdk.count(source.getFileID(location).getHashValue())) continue;
        const auto *forward = llvm::dyn_cast<clang::RecordDecl>(redeclaration);
        if (forward == nullptr || forward->isCompleteDefinition() || forward->isImplicit() ||
            !forward->getDeclContext()->isTranslationUnit() ||
            !literalRuntimeCgToken(location, "uint4", file))
          return false;
      }
      argument = recordUse;
      record = declaration->getCanonicalDecl();
      return true;
    }
    clang::QualType fixedArithmeticType(clang::QualType type) const {
      if (type.isNull() || type->isDependentType()) return {};
      type = type.getCanonicalType().getUnqualifiedType();
      return type->isBuiltinType() && type->isArithmeticType() ? type : clang::QualType{};
    }
    bool provedIntNttpArgument(const clang::Expr *expression,
                               const clang::TemplateParameterList *parameters,
                               unsigned depth = 0) const {
      if (expression == nullptr || depth > 32 ||
          !context.hasSameType(expression->getType(), context.IntTy)) return false;
      expression = expression->IgnoreParenImpCasts();
      if (llvm::isa<clang::IntegerLiteral>(expression)) return true;
      if (const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expression)) {
        for (const auto *parameter : *parameters)
          if (reference->getDecl() == parameter) return true;
        return false;
      }
      if (const auto *unary = llvm::dyn_cast<clang::UnaryOperator>(expression))
        return (unary->getOpcode() == clang::UO_Plus || unary->getOpcode() == clang::UO_Minus) &&
            provedIntNttpArgument(unary->getSubExpr(), parameters, depth + 1);
      if (const auto *binary = llvm::dyn_cast<clang::BinaryOperator>(expression))
        return (binary->getOpcode() == clang::BO_Add || binary->getOpcode() == clang::BO_Sub) &&
            provedIntNttpArgument(binary->getLHS(), parameters, depth + 1) &&
            provedIntNttpArgument(binary->getRHS(), parameters, depth + 1);
      return false;
    }
    clang::QualType provedNttpScalarType(const clang::Expr *expression,
                                        unsigned depth = 0) const {
      if (expression == nullptr || expression->getType().isNull() || depth > 32) return {};
      if (!expression->getType()->isDependentType())
        return fixedArithmeticType(expression->getType());
      if (const auto *paren = llvm::dyn_cast<clang::ParenExpr>(expression))
        return provedNttpScalarType(paren->getSubExpr(), depth + 1);
      if (const auto *binary = llvm::dyn_cast<clang::BinaryOperator>(expression)) {
        if (binary->getOpcode() != clang::BO_Add && binary->getOpcode() != clang::BO_Sub &&
            binary->getOpcode() != clang::BO_Mul) return {};
        auto left = provedNttpScalarType(binary->getLHS(), depth + 1);
        const auto right = provedNttpScalarType(binary->getRHS(), depth + 1);
        if (left.isNull() || right.isNull() || !context.hasSameType(left, right)) return {};
        if (context.isPromotableIntegerType(left)) left = context.getPromotedIntegerType(left);
        return left;
      }
      // Do not admit operators, members, function pointers or a bare overload
      // address. The unresolved lookup must belong to this exact plain call.
      if (expression->getStmtClass() != clang::Stmt::CallExprClass) return {};
      const auto *call = llvm::cast<clang::CallExpr>(expression);
      const auto *lookup = llvm::dyn_cast<clang::UnresolvedLookupExpr>(call->getCallee());
      if (lookup == nullptr || lookup->getQualifier() ||
          !lookup->hasExplicitTemplateArgs()) return {};
      auto candidate = lookup->decls_begin();
      if (candidate == lookup->decls_end()) return {};
      const auto *functionTemplate = llvm::dyn_cast<clang::FunctionTemplateDecl>(*candidate++);
      if (candidate != lookup->decls_end() || functionTemplate == nullptr) return {};
      const auto *function = functionTemplate->getTemplatedDecl();
      if (llvm::isa<clang::CXXMethodDecl>(function) ||
          !function->getDeclContext()->isFileContext() || function->isVariadic() ||
          function->getDefinition() == nullptr ||
          call->getNumArgs() != function->getNumParams()) return {};
      const auto result = fixedArithmeticType(function->getReturnType());
      if (result.isNull()) return {};
      const auto *parameters = functionTemplate->getTemplateParameters();
      if (parameters->size() == 0 || parameters->size() != lookup->getNumTemplateArgs() ||
          parameters->getRequiresClause() != nullptr || function->getTrailingRequiresClause())
        return {};
      for (const auto *parameter : *parameters) {
        const auto *integer = llvm::dyn_cast<clang::NonTypeTemplateParmDecl>(parameter);
        if (integer == nullptr || integer->isParameterPack() || integer->hasDefaultArgument() ||
            !context.hasSameType(integer->getType(), context.IntTy)) return {};
      }
      for (const auto &argument : lookup->template_arguments()) {
        const auto &value = argument.getArgument();
        if (value.getKind() == clang::TemplateArgument::Integral) {
          if (!context.hasSameType(value.getIntegralType(), context.IntTy)) return {};
        } else if (value.getKind() != clang::TemplateArgument::Expression ||
                   !provedIntNttpArgument(value.getAsExpr(), parameters)) return {};
      }
      for (unsigned i = 0; i < call->getNumArgs(); ++i) {
        const auto actual = fixedArithmeticType(call->getArg(i)->getType());
        const auto formal = fixedArithmeticType(function->getParamDecl(i)->getType());
        if (actual.isNull() || formal.isNull() || !context.hasSameType(actual, formal)) return {};
      }
      // Clang marks this lookup ADL-dependent. Builtin arithmetic actual
      // arguments have no associated classes/namespaces, so ADL adds nothing.
      return result;
    }
    bool provedSdkEnumEquality(const clang::BinaryOperator *comparison,
                              const clang::EnumDecl *enumeration) const {
      // The supported error symbols and memcpy directions have distinct target
      // values. Device attributes do not: two CUDA attributes share one adapter.
      if (enumeration->getName() == "cudaError" ||
          enumeration->getName() == "cudaMemcpyKind")
        return true;
      const auto *left = llvm::dyn_cast<clang::DeclRefExpr>(
          comparison->getLHS()->IgnoreParenImpCasts());
      const auto *right = llvm::dyn_cast<clang::DeclRefExpr>(
          comparison->getRHS()->IgnoreParenImpCasts());
      if (left == nullptr || right == nullptr ||
          left->getDecl() != right->getDecl() ||
          !llvm::isa<clang::EnumConstantDecl>(left->getDecl()))
        return false;
      const auto mapped = CUDA_RUNTIME_TYPE_NAME_MAP.find(left->getDecl()->getName());
      return mapped != CUDA_RUNTIME_TYPE_NAME_MAP.end() &&
          !Statistics::isUnsupported(mapped->second);
    }
    bool inspectEnumNumericOperand(const clang::Expr *operand,
                                   clang::SourceLocation location) {
      if (textureOnly) return true;
      location = source.getExpansionLoc(location);
      if (operand == nullptr || location.isInvalid() ||
          !audited.count(source.getFileID(location).getHashValue()))
        return true;
      operand = operand->IgnoreParenImpCasts();
      while (const auto *constant = llvm::dyn_cast<clang::ConstantExpr>(operand))
        operand = constant->getSubExpr()->IgnoreParenImpCasts();
      const auto *enumeration = sdkEnumeration(operand->getType());
      if (enumeration == nullptr)
        return true;
      badLocation = location;
      badName = enumeration->getNameAsString();
      badReason = "unproved SDK enum numeric observation";
      return false;
    }
    const clang::EnumDecl *sdkEnumeration(clang::QualType type) const {
      if (type.isNull())
        return nullptr;
      const auto *enumeration = type->getAs<clang::EnumType>();
      if (enumeration == nullptr)
        return nullptr;
      const auto *declaration = enumeration->getDecl();
      const auto definition = source.getSpellingLoc(declaration->getLocation());
      return definition.isValid() &&
          sdk.count(source.getFileID(definition).getHashValue()) ? declaration : nullptr;
    }
    bool rejectInquiryOrigin(clang::SourceLocation location) {
      location = source.getExpansionLoc(location);
      if (location.isInvalid() ||
          !audited.count(source.getFileID(location).getHashValue()))
        return true;
      badLocation = location;
      badName = "<pointer origin>";
      badReason = "unproved compiler SDK inquiry";
      return false;
    }
    bool inspectCompilerInquiry(const clang::Expr *operand,
                                clang::SourceLocation location,
                                bool inspectPointerOrigin) {
      // Builtin prototypes can erase the argument to void*. Inspect the
      // original operand, and for object/layout inquiries its pointer origin
      // as well: '&properties.name[0]' otherwise hides the containing SDK ABI.
      if (!inspect(operand->IgnoreParenImpCasts()->getType(), location, false)) {
        if (badReason == "unproved SDK record ABI" ||
            badReason == "unproved SDK enum reflection")
          badReason = "unproved compiler SDK inquiry";
        return false;
      }
      if (inspectPointerOrigin) {
        const auto *original = operand->IgnoreParenImpCasts();
        if (const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(original)) {
          if (const auto *variable = llvm::dyn_cast<clang::VarDecl>(reference->getDecl())) {
            if (variable->getType()->isPointerType()) {
              // Mutable and externally supplied pointers can acquire an SDK
              // object after initialization. Constant aliases must prove the
              // initializer's origin, rather than stop at their erased type.
              if (!variable->getType().isConstQualified() || !variable->hasInit() ||
                  !inquiryVariables.insert(variable).second)
                return rejectInquiryOrigin(location);
              return inspectCompilerInquiry(variable->getInit(), location, true);
            }
          }
        }
        // A pointer-returning accessor/lambda or pointer-valued field has no
        // local origin proof. Its erased result can hide an SDK record.
        if (original->getType()->isPointerType() &&
            (llvm::isa<clang::CallExpr>(original) ||
             llvm::isa<clang::MemberExpr>(original) ||
             llvm::isa<clang::CXXThisExpr>(original)))
          return rejectInquiryOrigin(location);
        for (auto *child : operand->children())
          if (auto *expression = llvm::dyn_cast_or_null<clang::Expr>(child))
            if (!inspectCompilerInquiry(expression, location, true))
              return false;
      }
      return true;
    }
    bool inspectEnumInterface(clang::QualType type, clang::SourceLocation location) {
      if (type.isNull())
        return true;
      if (const auto *pointer = type->getAs<clang::PointerType>())
        return inspectEnumInterface(pointer->getPointeeType(), location);
      if (const auto *reference = type->getAs<clang::ReferenceType>())
        return inspectEnumInterface(reference->getPointeeType(), location);
      if (const auto *array = context.getAsArrayType(type))
        return inspectEnumInterface(array->getElementType(), location);
      if (const auto *atomic = type->getAs<clang::AtomicType>())
        return inspectEnumInterface(atomic->getValueType(), location);
      if (const auto *member = type->getAs<clang::MemberPointerType>()) {
        if (!inspectEnumInterface(member->getPointeeType(), location))
          return false;
        const auto owner = runtimeHeaderMemberClass(member, 0);
        if (owner.isNull()) {
          badLocation = location;
          badName = "<member-pointer qualifier>";
          badReason = "unproved caller member pointer interface";
          return false;
        }
        return inspectEnumInterface(owner, location);
      }
      if (const auto *function = type->getAs<clang::FunctionProtoType>()) {
        if (!inspectEnumInterface(function->getReturnType(), location))
          return false;
        for (const auto parameter : function->param_types())
          if (!inspectEnumInterface(parameter, location))
            return false;
      }
      const auto *enumeration = type->getAs<clang::EnumType>();
      if (enumeration == nullptr)
        return true;
      const auto *declaration = enumeration->getDecl();
      const auto definition = source.getSpellingLoc(declaration->getLocation());
      if (definition.isInvalid() ||
          !sdk.count(source.getFileID(definition).getHashValue()))
        return true;
      badLocation = location;
      badName = declaration->getNameAsString();
      badReason = "unproved caller SDK enum function interface";
      return false;
    }
    bool inspect(clang::QualType type, clang::SourceLocation location,
                 bool permitExplicitAdapters = true) {
      location = source.getExpansionLoc(location);
      if (location.isInvalid() || type.isNull() ||
          !audited.count(source.getFileID(location).getHashValue()))
        return true;
      // A caller's dependent type has no proven exclusion of the mapped SDK
      // records. This includes SFINAE/member probes, not only sizeof and
      // currently instantiated reflection. Keep the admission nondependent;
      // native implementation templates remain independently audited.
      if (type->isDependentType() &&
          !native.count(source.getFileID(location).getHashValue())) {
        if (ascify::ProvesScalar64Type(scalar64, type)) return true;
        // Exact parsed bytes identify the reviewed Sample/profile utilities.
        // Their uninstantiated generic utilities do not establish a texture
        // layout; concrete instantiations still traverse the ordinary checks.
        // Caller templates do not inherit this texture-only exception.
        if (textureOnly && frozenTextureUtilityFiles.count(source.getFileID(location).getHashValue()))
          return true;
        // The exact reviewed patterns contain type constraints, never arbitrary
        // SDK layout probes. Their nondependent uint4 identity argument needs
        // its own scoped proof below. This admits their dependent type
        // observations, not a whole function/trait/body: normal traversal and
        // every concrete instantiation retain all original SDK checks.
        if (!textureOnly && runtimePatternDepth != 0 && runtimeInstanceDepth == 0 &&
            frozenRuntimePatternFiles.count(source.getFileID(location).getHashValue()))
          return true;
        badLocation = location;
        badName = "<dependent type>";
        badReason = "unproved retained dependent type observation";
        return false;
      }
      if (const auto *pointer = type->getAs<clang::PointerType>())
        return inspect(pointer->getPointeeType(), location, permitExplicitAdapters);
      if (const auto *reference = type->getAs<clang::ReferenceType>())
        return inspect(reference->getPointeeType(), location, permitExplicitAdapters);
      if (const auto *array = context.getAsArrayType(type))
        return inspect(array->getElementType(), location, permitExplicitAdapters);
      if (const auto *atomic = type->getAs<clang::AtomicType>())
        return inspect(atomic->getValueType(), location, permitExplicitAdapters);
      if (const auto *member = type->getAs<clang::MemberPointerType>()) {
        if (!inspect(member->getPointeeType(), location, permitExplicitAdapters))
          return false;
        const auto owner = runtimeHeaderMemberClass(member, 0);
        if (owner.isNull()) {
          badLocation = location;
          badName = "<member-pointer qualifier>";
          badReason = "unproved retained member pointer type";
          return false;
        }
        return inspect(owner, location, permitExplicitAdapters);
      }
      if (const auto *function = type->getAs<clang::FunctionProtoType>()) {
        if (!inspect(function->getReturnType(), location, permitExplicitAdapters))
          return false;
        for (const auto parameter : function->param_types())
          if (!inspect(parameter, location, permitExplicitAdapters))
            return false;
      }
      if (!permitExplicitAdapters && !textureOnly)
        if (const auto *enumeration = type->getAs<clang::EnumType>()) {
          const auto *declaration = enumeration->getDecl();
          const auto definition = source.getSpellingLoc(declaration->getLocation());
          if (definition.isValid() &&
              sdk.count(source.getFileID(definition).getHashValue())) {
            badLocation = location;
            badName = declaration->getNameAsString();
            badReason = "unproved SDK enum reflection";
            return false;
          }
        }
      const auto *record = type->getAs<clang::RecordType>();
      if (record == nullptr)
        return true;
      const auto *declaration = record->getDecl();
      if (!textureOnly && runtimePatternDepth != 0 && runtimeInstanceDepth == 0 &&
          std::find(runtimeIdentityArguments.begin(), runtimeIdentityArguments.end(),
                    std::pair<unsigned, const clang::TagDecl *>(
                        location.getRawEncoding(), declaration->getCanonicalDecl())) !=
              runtimeIdentityArguments.end())
        return true;
      const auto definition = source.getSpellingLoc(declaration->getLocation());
      if (definition.isInvalid() ||
          !sdk.count(source.getFileID(definition).getHashValue())) {
        // A caller record can carry a mapped SDK object as a field or base.
        // Reflection on that enclosing record still observes the removed SDK
        // layout, even when the reflected spelling never names the SDK type.
        if (!permitExplicitAdapters) {
          const auto *complete = declaration->getDefinition();
          if (complete != nullptr && reflectedRecords.insert(complete).second) {
            for (const auto *field : complete->fields())
              if (!inspect(field->getType(), location, false))
                return false;
            if (const auto *cxx = llvm::dyn_cast<clang::CXXRecordDecl>(complete))
              for (const auto &base : cxx->bases())
                if (!inspect(base.getType(), location, false))
                  return false;
          }
        }
        return true;
      }
      if (textureOnly && !ascify::IsCudaTextureRecord(declaration)) return true;
      const auto name = declaration->getName();
      // Existing explicit runtime type adapters retain their published
      // contracts. dim3 retains the separately proven host-local initializer
      // rule; native vector records and unadapted SDK records have no ABI proof.
      // Runtime adapters do not prove record reflection or template-type
      // equivalence. In particular CUDA dim3 is default/constexpr constructible
      // whereas the native target class has a different constructor contract.
      if (permitExplicitAdapters &&
          declaration->getDeclContext()->getRedeclContext()->isTranslationUnit() &&
          name == "dim3")
        return true;
      if (permitExplicitAdapters && ascify::IsCudaTextureRecord(declaration))
        return true;
      const auto mapped = CUDA_RUNTIME_TYPE_NAME_MAP.find(name);
      if (permitExplicitAdapters &&
          declaration->getDeclContext()->getRedeclContext()->isTranslationUnit() &&
          (name == "cudaDeviceProp" || name == "cudaFuncAttributes" ||
           ascify::IsCudaTextureRecord(declaration)) &&
          mapped != CUDA_RUNTIME_TYPE_NAME_MAP.end() &&
          !Statistics::isUnsupported(mapped->second))
        return true;
      badLocation = location;
      badName = name.str();
      return false;
    }
    clang::ASTContext &context;
    clang::SourceManager &source;
    const std::set<unsigned> &sdk;
    const std::set<unsigned> &audited;
    const std::set<unsigned> &native;
    const std::set<unsigned> &frozenTextureUtilityFiles;
    const std::set<unsigned> &frozenTextureCgFiles;
    const std::set<unsigned> &frozenRuntimePatternFiles;
    const bool textureOnly;
    const ascify::Scalar64Plan& scalar64;
    const ascify::DynamicWarpGroupStats& dynamicGroups;
    const ascify::NttpVoidCallPlan& nttpVoid;
    const ascify::HostPointerSwapPlan& hostPointerSwap;
    unsigned runtimePatternDepth = 0;
    unsigned runtimeInstanceDepth = 0;
    std::vector<const clang::TemplateTypeParmDecl *> runtimePatternParameters;
    std::vector<const clang::Decl *> runtimeCopiedPatternParts;
    std::vector<std::pair<unsigned, const clang::TagDecl *>> runtimeIdentityArguments;
    std::set<const clang::RecordDecl *> reflectedRecords;
    std::set<const clang::VarDecl *> inquiryVariables;
  } records(compiler.getASTContext(), source, cudaRuntimeApiSdkFileIds, auditedFiles, nativeFiles, frozenTextureUtilityFiles, frozenTextureCgFiles, frozenRuntimePatternFiles, textureOnly, scalar64Plan, dynamicWarpGroups, nttpVoidCallPlan, hostPointerSwapPlan);
  if (!records.TraverseDecl(compiler.getASTContext().getTranslationUnitDecl()))
    {
      const auto diagnostic = compiler.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "CUDA runtime-API header boundary: %0 '%1'");
      compiler.getDiagnostics().Report(records.badLocation, diagnostic)
          << records.badReason << records.badName;
      return false;
    }
  return true;
}

bool AscifyAction::isAdmittedCudaTextureMacroUse(clang::SourceLocation location) {
  auto &source=getCompilerInstance().getSourceManager();
  const auto written=source.getSpellingLoc(location);
  if (written.isInvalid() || !source.isWrittenInMainFile(written)) return false;
  const auto file=source.getFileID(written);
  const unsigned offset=source.getFileOffset(written);
  clang::Token token;
  if (clang::Lexer::getRawToken(written,token,source,getCompilerInstance().getLangOpts(),true))
    return false;
  const auto flagName=clang::Lexer::getSpelling(token,source,getCompilerInstance().getLangOpts());
  const bool arrayFlag=flagName=="cudaArrayLayered" || flagName=="cudaArrayCubemap";
  const NvidiaSampleHelperMacroCandidate *helper=nullptr;
  for (const auto &candidate:nvidiaSampleHelperMacroCandidates) {
    const auto &range=candidate.invocationRange;
    if (range.file==file && range.beginOffset<offset && offset<range.endOffset) {
      if (helper || candidate.kind!=NvidiaSampleHelperMacroKind::CheckCudaErrors ||
          !candidate.statusDomainProven) return false;
      helper=&candidate;
    }
  }
  // Completed callback history catches argument prescan before stringization
  // or token-pasting. Only the exact authenticated helper invocation and the
  // SDK flag's own single-token expansion may enclose this literal token.
  for (const auto &range:cudaLaneIdMacroRanges) {
    if (range.file!=file || offset<range.beginOffset || offset>=range.endOffset) continue;
    if (arrayFlag && range.beginOffset==offset &&
        range.endOffset==offset+token.getLength()) continue;
    if (helper && range.beginOffset==helper->invocationRange.beginOffset &&
        range.endOffset==helper->invocationRange.endOffset) continue;
    return false;
  }
  if (location.isMacroID() && !helper) return false;
  if (helper) cudaTextureRequiredHelperOffsets.insert(helper->invocationRange.beginOffset);
  return true;
}

void AscifyAction::MacroExpands(const clang::Token &MacroNameTok,
                                const clang::MacroDefinition &MD,
                                clang::SourceRange Range,
                                const clang::MacroArgs *Args) {
  if (StandaloneDeviceExecutable)
    ascify::ObserveDeviceNativeMacro(getCompilerInstance(),MacroNameTok,MD,Range,Args,deviceAggregateMacroEvidence);
  observeDirectImageUtilityMacro(MacroNameTok, MD, Range, Args);
  observeCudaRuntimeApiMacroUse(MacroNameTok, MD);
  if (MacroNameTok.getIdentifierInfo() == nullptr)
    return;
  if (originalCgControlName(MacroNameTok.getIdentifierInfo()->getName()))
    originalCgControlChanged = true;
  clang::SourceManager &sourceManager =
      getCompilerInstance().getSourceManager();
  const clang::MacroInfo *macroInfo = MD.getMacroInfo();
  if (MacroNameTok.getIdentifierInfo()->getName() == "__LINE__") {
    const auto spelling = sourceManager.getSpellingLoc(MacroNameTok.getLocation());
    const auto file = sourceManager.getFileID(spelling);
    if (!locationComesFromInitiallyTrustedSystemFile(
            sourceManager, spelling, trustedSystemFileIds) &&
        !cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
        frozenNvidiaSampleHelperFileRole(sourceManager, spelling) == 0 &&
        !isOwnedCudaCompilerInput(file, getCompilerInstance()))
      profilerAvailabilityPositionObserved = true;
  }
  const auto profileUse = sourceManager.getExpansionLoc(MacroNameTok.getLocation());
  if (frontendCompatibility.enabled() && profileUse.isValid()) {
    const auto file = sourceManager.getFileID(profileUse);
    if (verifiedFrontendCgEntryFileIds.count(file.getHashValue()) &&
        !frontendCgMacroExpansionMatches(
            MacroNameTok.getIdentifierInfo()->getName(), macroInfo, file,
            getCompilerInstance(), frontendCompatibility)) {
      macroAlteredFrontendCgFileIds.insert(file.getHashValue());
      if(isNewDynamicProjectionName(MacroNameTok.getIdentifierInfo()->getName()))
        capturedDynamicProjectionName=MacroNameTok.getIdentifierInfo()->getName().str();
    }
  }
  if (frontendCompatibility.enabled() && profileUse.isValid() &&
      sourceManager.getFilename(profileUse) ==
          frontendCompatibility.canonicalRoot + "/host_math.h")
    macroAlteredFrontendHostMathFileIds.insert(
        sourceManager.getFileID(profileUse).getHashValue());
  const auto arrayFlagName=MacroNameTok.getIdentifierInfo()->getName();
  if ((arrayFlagName == "cudaArrayLayered" || arrayFlagName == "cudaArrayCubemap") && macroInfo != nullptr) {
    const auto definition = sourceManager.getSpellingLoc(macroInfo->getDefinitionLoc());
    const auto written = sourceManager.getSpellingLoc(MacroNameTok.getLocation());
    if (definition.isValid() && written.isValid() &&
        cudaRuntimeApiSdkFileIds.count(sourceManager.getFileID(definition).getHashValue()) &&
        sourceManager.isWrittenInMainFile(written) && !macroInfo->isFunctionLike() &&
        macroInfo->getNumTokens()==1 &&
        macroInfo->tokens_begin()->is(clang::tok::numeric_constant) &&
        getCompilerInstance().getPreprocessor().getSpelling(*macroInfo->tokens_begin())==
            (arrayFlagName=="cudaArrayLayered" ? "0x01" : "0x04")) {
      clang::Token token;
      if (!clang::Lexer::getRawToken(written,token,sourceManager,getCompilerInstance().getLangOpts(),true) &&
          clang::Lexer::getSpelling(token,sourceManager,getCompilerInstance().getLangOpts()) == arrayFlagName)
        cudaTextureLiteralOffsets.insert(sourceManager.getFileOffset(written));
    }
  }
  const auto macroBegin = sourceManager.getExpansionLoc(Range.getBegin());
  const auto macroLast = sourceManager.getExpansionLoc(Range.getEnd());
  if (macroBegin.isValid() && macroLast.isValid() &&
      sourceManager.isWrittenInMainFile(macroBegin) &&
      sourceManager.isWrittenInMainFile(macroLast)) {
    const auto macroEnd = clang::Lexer::getLocForEndOfToken(
        macroLast, 0, sourceManager, getCompilerInstance().getLangOpts());
    if (macroEnd.isValid()) {
      const SemanticRewriteRange invocation{
          sourceManager.getFileID(macroBegin),
          sourceManager.getFileOffset(macroBegin),
          sourceManager.getFileOffset(macroEnd)};
      cudaLaneIdMacroRanges.push_back(invocation);
      // Argument prescan can expose physical inner MAX tokens even while an
      // outer macro observes their spelling (for example by stringization).
      // Only the retained, authenticated image-provider MIN is known to use
      // its arguments solely as expression tokens, without # or ##.
      const bool nonReflectiveImageMin =
          MacroNameTok.getIdentifierInfo()->getName() == "MIN" &&
          macroInfo != nullptr && Args != nullptr &&
          !MacroNameTok.getLocation().isMacroID() &&
          !Range.getBegin().isMacroID() && !Range.getEnd().isMacroID() &&
          ((frozenOfficialNvidiaSampleHelperFunctionsSeen &&
            isDirectFrozenFunctionsImageProviderInstance(macroInfo->getDefinitionLoc())) ||
           isProvedDirectImageInstance(macroInfo->getDefinitionLoc())) &&
          activeFrozenHelperImageMinBodyMatches(
              *macroInfo, getCompilerInstance().getPreprocessor());
      if (!nonReflectiveImageMin)
        nvidiaSampleMacroObservationRanges.push_back(invocation);
    }
  }
  // The new retained MAX domain also audits enclosing macro observers in
  // selected headers. Existing main-file candidates keep their old ranges.
  if (macroBegin.isFileID() && macroLast.isFileID() &&
      !sourceManager.isWrittenInMainFile(macroBegin) &&
      sourceManager.getFileID(macroBegin) == sourceManager.getFileID(macroLast) &&
      ownedPreservedMaxLocation(macroBegin)) {
    const auto end = clang::Lexer::getLocForEndOfToken(
        macroLast, 0, sourceManager, getCompilerInstance().getLangOpts());
    if (end.isFileID())
      nvidiaSampleMacroObservationRanges.push_back({sourceManager.getFileID(macroBegin),
          sourceManager.getFileOffset(macroBegin), sourceManager.getFileOffset(end)});
  }
  // A local leaf can depend on macros from the root or an earlier selected
  // sibling. Record the dependency, not an instruction to copy a -D.
  // LocalHeader.cpp still proves a same-position expansion and exact inputs.
  if (macroInfo != nullptr && localHeaderContext != nullptr &&
      localHeaderContext->depth == 0 && localHeaderContext->plan != nullptr) {
    const auto definition = sourceManager.getSpellingLoc(macroInfo->getDefinitionLoc());
    const auto use = sourceManager.getExpansionLoc(MacroNameTok.getLocation());
    if (definition.isValid() && use.isValid() &&
        !sourceManager.isWrittenInMainFile(use)) {
      if (sourceManager.isWrittenInMainFile(definition)) {
        // Preserve the original root-dependency rule, including its later
        // refusal for a non-leaf that genuinely needs parent definitions.
        localHeaderContext->plan->requireInheritedSourceContext(
            sourceManager.getFilename(use).str());
      } else if (sourceManager.getFileID(definition) != sourceManager.getFileID(use) &&
                 localHeaderContext->plan->isSelectedRootHeader(
                     sourceManager.getFilename(definition).str())) {
        // A later include in this file can still establish that it is not a
        // leaf. Decide sibling eligibility only after preprocessing finishes.
        siblingSourceMacroContextUsers.insert(contextualCanonicalPath(
            sourceManager.getFilename(use)));
      } else if (!getCompilerInstance().getLangOpts().CUDA &&
                 !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks &&
                 !macroInfo->isBuiltinMacro() &&
                 localHeaderContext->plan->isSelectedRootHeader(
                     sourceManager.getFilename(use).str()) &&
                 physicalParentHostSystemMacro(
                     sourceManager.getFileID(definition), sourceManager.getFileID(use),
                     getCompilerInstance(), trustedSystemFileIds)) {
        // A later include can prove this header self-contained. Defer the
        // no-own-include qualification until all original edges are known.
        systemSourceMacroContextUsers.insert(contextualCanonicalPath(
            sourceManager.getFilename(use)));
      }
    }
  }
  if (MacroNameTok.getIdentifierInfo()->getName() == "CUDART_CB" &&
      macroInfo != nullptr && !macroInfo->isFunctionLike() &&
      macroInfo->getNumTokens() == 0 &&
      getCompilerInstance().getTarget().getTriple().isOSLinux() &&
      !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks &&
      Range.getBegin().isFileID() && Range.getEnd().isFileID() &&
      MacroNameTok.getLocation().isFileID() &&
      sourceManager.isWrittenInMainFile(MacroNameTok.getLocation())) {
    const auto definition = sourceManager.getSpellingLoc(macroInfo->getDefinitionLoc());
    const auto file = sourceManager.getFileID(definition);
    const auto *entry = sourceManager.getFileEntryForID(file);
    const auto path = sourceManager.getFilename(definition);
    const auto spelling = clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(Range), sourceManager,
        getCompilerInstance().getLangOpts());
    if (entry != nullptr && !sourceManager.isFileOverridden(entry) &&
        llvm::sys::path::filename(path) == "driver_types.h" &&
        isConfiguredCudaSdkDependencyRole(path) &&
        physicalSourceFileMatches(path, *entry, sourceManager) &&
        spelling == "CUDART_CB") {
      const auto end = clang::Lexer::getLocForEndOfToken(
          Range.getEnd(), 0, sourceManager, getCompilerInstance().getLangOpts());
      if (end.isValid()) {
        ct::Replacement edit(sourceManager, Range.getBegin(), spelling.size(), "");
        (void)insertSemanticReplacement(edit,
            clang::FullSourceLoc(Range.getBegin(), sourceManager), Range.getBegin(), end);
      }
    }
  }
  if (isAscifyCudaCompatPublishedMacro(
          MacroNameTok.getIdentifierInfo()->getName()))
    auditExternalNvidiaSampleHelperPreprocessorUse(
        MacroNameTok.getLocation(), MacroNameTok, "expansion");
  auditFrozenNvidiaSampleHelperMacroDependency(
      MacroNameTok.getLocation(), MacroNameTok, MD, "expansion");
  // EXIT_WAIVED is a host-side sample policy constant, not a CUDA capability.
  // Retain its original object-macro contract when the helper is removed.
  // Authenticate both the exact provider and the active token body; a copied
  // spelling, command-line definition, alias expansion, or overridden buffer
  // must not borrow the frozen helper's provenance.
  if (MacroNameTok.getIdentifierInfo()->getName() == "EXIT_WAIVED" &&
      macroInfo != nullptr && Args == nullptr &&
      MacroNameTok.getLocation().isFileID() &&
      Range.getBegin().isFileID() && Range.getEnd().isFileID() &&
      sourceManager.isWrittenInMainFile(MacroNameTok.getLocation()) &&
      activeFrozenHelperFunctionsProviderMacroBodyMatches(
          "EXIT_WAIVED", *macroInfo,
          getCompilerInstance().getPreprocessor())) {
    const auto definition =
        sourceManager.getSpellingLoc(macroInfo->getDefinitionLoc());
    const auto definitionFile = definition.isValid()
        ? sourceManager.getFileID(definition) : clang::FileID();
    const auto *entry = definitionFile.isValid()
        ? sourceManager.getFileEntryForID(definitionFile) : nullptr;
    const unsigned definitionRole = definition.isValid()
        ? frozenNvidiaSampleHelperFileRole(sourceManager, definition)
        : FrozenNvidiaSampleHelperNone;
    const bool frozenCoreProvider =
        entry != nullptr && !sourceManager.isFileOverridden(entry) &&
        (definitionRole == FrozenOfficialNvidiaSampleHelperCuda ||
         definitionRole == FrozenOfficialNvidiaSampleHelperString) &&
        physicalSourceFileMatches(sourceManager.getFilename(definition),
                                  *entry, sourceManager);
    const bool frozenImageProvider =
        frozenOfficialNvidiaSampleHelperFunctionsSeen &&
        isDirectFrozenFunctionsImageProviderInstance(definition);
    bool invalid = false;
    const auto spelling = clang::Lexer::getSourceText(
        clang::CharSourceRange::getTokenRange(Range), sourceManager,
        getCompilerInstance().getLangOpts(), &invalid);
    if (!invalid && spelling == "EXIT_WAIVED" &&
        (frozenCoreProvider || frozenImageProvider)) {
      nvidiaSampleHelperNormalMacroOffsets.insert(
          sourceManager.getFileOffset(MacroNameTok.getLocation()));
      return;
    }
  }
  if (MacroNameTok.getIdentifierInfo()->getName() == "MAX")
    (void)requestDirectImageMaxReplay(MacroNameTok, MD, Range, Args);
  if (MacroNameTok.getIdentifierInfo()->getName() == "MAX" &&
      preservedMaxNeeded) {
    if (!admitPreservedMaxExpansion(MacroNameTok, MD, Range, Args)) {
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs() << "Ascify NVIDIA sample-helper closure: unproved preserved MAX invocation\n";
    }
    return;
  }
  if (macroInfo == nullptr ||
      !locationComesFromRecognizedNvidiaSampleHelper(
          sourceManager, macroInfo->getDefinitionLoc())) {
    return;
  }
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
    if (admittedFrozenInternalMaxExpansion) {
      frozenNvidiaSampleInternalImageMaxExpansionSeen = true;
      return;
    }
    // Ignore implementation-internal expansions from the recognized helper,
    // but never remove the helper when an arbitrary local/user header relies
    // on one of its macros: this prototype does not rewrite that header.
    // The Driver-API helper publishes fallback definitions (MAX) that other
    // frozen sample helpers consume internally. Definition and use both live
    // inside the frozen helper graph, so this stays an implementation detail.
    const unsigned internalExpansionDefinitionRole =
        frozenNvidiaSampleHelperFileRole(
            sourceManager, macroInfo->getDefinitionLoc());
    const unsigned internalExpansionConsumerRole =
        frozenNvidiaSampleHelperFileRole(sourceManager, macroLocation);
    if (!locationComesFromRecognizedNvidiaSampleHelper(
            sourceManager, macroLocation) &&
        !(internalExpansionDefinitionRole ==
              FrozenOfficialNvidiaSampleHelperDrvapi &&
          internalExpansionConsumerRole != FrozenNvidiaSampleHelperNone)) {
      if ((name == "checkCudaErrors" || name == "getLastCudaError") &&
          localHeaderContext != nullptr && localHeaderContext->depth == 0 &&
          localHeaderContext->plan != nullptr)
        localHeaderContext->plan->requireInheritedHelperContext(
            sourceManager.getFilename(macroLocation).str());
      nvidiaSampleHelperUnsupportedMacroUse = true;
      llvm::errs()
          << "Ascify NVIDIA sample-helper closure: helper macro expansion "
          << "outside the main file keeps all helper edits\n";
    }
    return;
  }
  if (name == "MAX" && Args != nullptr &&
      !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks &&
      !MacroNameTok.getLocation().isMacroID() &&
      !Range.getBegin().isMacroID() && !Range.getEnd().isMacroID() &&
      frozenNvidiaSampleHelperFileRole(sourceManager, macroInfo->getDefinitionLoc()) ==
          FrozenOfficialNvidiaSampleHelperCuda &&
      activeFrozenHelperCudaMaxBodyMatches(*macroInfo,
                                          getCompilerInstance().getPreprocessor())) {
    // Preserve this exact macro's unparenthesized argument substitution and
    // repeated evaluation. Renaming to a function or to helper_image's MAX
    // would change behavior for side effects or low-precedence expressions.
    // Clang caches argument pre-expansion; asking for it here is the same
    // expansion the original invocation uses. A consumed __COUNTER__ must
    // retain the invocation, or later target expansions would be renumbered.
    auto &preprocessor = getCompilerInstance().getPreprocessor();
    const auto counterBefore = preprocessor.getCounterValue();
    std::string arguments[2];
    bool admitted = true;
    for (unsigned index = 0; index != 2 && admitted; ++index) {
      const auto &tokens = const_cast<clang::MacroArgs *>(Args)->getPreExpArgument(
          index, preprocessor);
      unsigned count = 0;
      for (const auto &token : tokens) {
        if (token.is(clang::tok::eof)) break;
        if (++count > 512 || token.isAnnotation()) { admitted = false; break; }
        const std::string spelling = preprocessor.getSpelling(token);
        if (spelling.empty()) { admitted = false; break; }
        if (auto *identifier = token.getIdentifierInfo()) {
          // A suppressed/self-referential macro could expand differently when
          // emitted as ordinary source. API identifiers would also need an
          // independent rewrite inside this transaction; retain those cases.
          const auto word = identifier->getName();
          if (preprocessor.getMacroDefinition(identifier) ||
              word.starts_with("cu") ||
              word.starts_with("__") || word == "_Pragma") {
            admitted = false; break;
          }
        }
        switch (token.getKind()) {
        case clang::tok::identifier: case clang::tok::numeric_constant:
        case clang::tok::l_paren: case clang::tok::r_paren:
        case clang::tok::l_square: case clang::tok::r_square:
        case clang::tok::question: case clang::tok::colon:
        case clang::tok::plus: case clang::tok::minus:
        case clang::tok::star: case clang::tok::slash: case clang::tok::percent:
        case clang::tok::less: case clang::tok::greater:
        case clang::tok::lessequal: case clang::tok::greaterequal:
        case clang::tok::equalequal: case clang::tok::exclaimequal:
        case clang::tok::lessless: case clang::tok::greatergreater:
        case clang::tok::amp: case clang::tok::ampamp:
        case clang::tok::pipe: case clang::tok::pipepipe: case clang::tok::caret:
        case clang::tok::exclaim: case clang::tok::tilde:
        case clang::tok::equal: case clang::tok::plusplus: case clang::tok::minusminus:
        case clang::tok::comma: case clang::tok::period: case clang::tok::arrow:
          arguments[index] += spelling + " ";
          break;
        default: admitted = false; break;
        }
        if (!admitted) break;
      }
      if (count == 0) admitted = false;
    }
    if (preprocessor.getCounterValue() != counterBefore) admitted = false;
    const auto after = clang::Lexer::getLocForEndOfToken(
        Range.getEnd(), 0, sourceManager, getCompilerInstance().getLangOpts());
    if (admitted && after.isValid() && sourceManager.isWrittenInMainFile(after) &&
        sourceManager.getFileID(macroLocation) == sourceManager.getFileID(after)) {
      SemanticRewriteRange invocation{sourceManager.getFileID(macroLocation),
          sourceManager.getFileOffset(macroLocation), sourceManager.getFileOffset(after)};
      if (invocation.beginOffset < invocation.endOffset) {
        nvidiaSampleExpandedMacros.push_back({macroLocation, invocation,
            "( " + arguments[0] + "> " + arguments[1] + "? " + arguments[0] +
            ": " + arguments[1] + ")"});
        nvidiaSampleHelperNormalMacroOffsets.insert(invocation.beginOffset);
        return;
      }
    }
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
  if (!sealDirectImageWitness())
    nvidiaSampleHelperUnsupportedMacroUse = true;
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
  if (!provePreservedMaxFinalState())
    nvidiaSampleHelperUnsupportedMacroUse = true;
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

void AscifyAction::recordInactiveNvidiaSampleHelperBranch(
    clang::SourceLocation loc, const clang::Token &nameToken,
    const clang::MacroDefinition &definition) {
  // Capture the condition at its actual evaluation point. The EOF macro table
  // cannot establish whether an earlier conditional body was excluded.
  if (preprocessorConditionalDepth != 0 || definition.getMacroInfo() != nullptr ||
      AscifyAMAP || nvidiaSampleHelperIncludes.empty() ||
      getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks ||
      !loc.isFileID() || !nameToken.getLocation().isFileID() ||
      nameToken.getIdentifierInfo() == nullptr)
    return;
  auto &source = getCompilerInstance().getSourceManager();
  if (!source.isWrittenInMainFile(loc) ||
      source.getFileID(nameToken.getLocation()) != source.getFileID(loc)) return;
  const llvm::StringRef name = nameToken.getIdentifierInfo()->getName();
  auto letter = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
  if (name.empty() || !letter(name.front()) || name.contains("__") ||
      !std::all_of(name.begin(), name.end(), [&](char c) {
        return letter(c) || (c >= '0' && c <= '9') || c == '_';
      }) || isNvidiaSampleHelperObservableMacro(name) ||
      isAscifyCudaCompatPublishedMacro(name)) return;
  const std::string lower = name.lower();
  for (llvm::StringRef prefix : {"cuda", "cu_", "cublas", "cufft", "curand",
                                 "cusolver", "cusparse", "npp", "nvidia",
                                 "ascify", "ascend", "acl"})
    if (llvm::StringRef(lower).starts_with(prefix)) return;
  bool invalid = false;
  const auto file = source.getFileID(loc);
  const auto buffer = source.getBufferData(file, &invalid);
  const unsigned offset = source.getFileOffset(loc);
  if (invalid || offset >= buffer.size()) return;
  const auto previousNewline = buffer.rfind('\n', offset);
  const auto lineBegin = previousNewline == llvm::StringRef::npos ? 0 : previousNewline + 1;
  const auto lineEnd = buffer.find('\n', offset);
  if (lineEnd == llvm::StringRef::npos) return;
  const auto line = buffer.slice(lineBegin, lineEnd);
  auto text = line.trim(" \t");
  if (!text.consume_front("#")) return;
  text = text.ltrim(" \t");
  if (!text.consume_front("ifdef") || text.empty() ||
      (text.front() != ' ' && text.front() != '\t') || text.trim(" \t") != name)
    return;
  const auto hash = lineBegin + line.find('#');
  const auto start = source.getLocForStartOfFile(file);
  nvidiaSampleInactiveBranches.push_back(
      {start.getLocWithOffset(hash), start.getLocWithOffset(lineEnd + 1),
       {file, static_cast<unsigned>(lineEnd + 1), static_cast<unsigned>(lineEnd + 1)},
       false, false});
}

void AscifyAction::proveSkippedNvidiaSampleHelperBranch(
    clang::SourceRange range, clang::SourceLocation endifLocation) {
  if (!range.getBegin().isFileID() || !range.getEnd().isFileID() ||
      !endifLocation.isFileID()) return;
  auto &source = getCompilerInstance().getSourceManager();
  const auto file = source.getFileID(endifLocation);
  if (!source.isWrittenInMainFile(endifLocation) ||
      source.getFileID(range.getBegin()) != file || source.getFileID(range.getEnd()) != file)
    return;
  bool invalid = false;
  const auto buffer = source.getBufferData(file, &invalid);
  const unsigned endifOffset = source.getFileOffset(endifLocation);
  if (invalid || endifOffset >= buffer.size()) return;
  const auto previousNewline = buffer.rfind('\n', endifOffset);
  const auto lineBegin = previousNewline == llvm::StringRef::npos ? 0 : previousNewline + 1;
  const auto lineEnd = buffer.find('\n', endifOffset);
  if (lineEnd == llvm::StringRef::npos) return;
  auto closing = buffer.slice(lineBegin, lineEnd).trim(" \t");
  if (!closing.consume_front("#") || closing.ltrim(" \t") != "endif") return;
  for (auto &branch : nvidiaSampleInactiveBranches) {
    if (branch.bodyRange.file != file || branch.skipped ||
        source.getFileOffset(range.getBegin()) != source.getFileOffset(branch.opener) ||
        source.getFileOffset(range.getEnd()) != lineEnd ||
        branch.bodyRange.beginOffset > lineBegin) {
      continue;
    }
    const auto body = buffer.slice(branch.bodyRange.beginOffset, lineBegin);
    // A deliberately small physical grammar: no directives (including digraph
    // forms), block comments, trigraphs, CR, or escaped physical newlines.
    // String escapes such as "\\n" remain valid ordinary body text.
    if (body.contains('#') || body.contains("%:") || body.contains("??") ||
        body.contains("/*") || body.contains("*/") || body.contains('\r')) continue;
    bool spliced = false;
    llvm::StringRef remaining = body;
    while (!remaining.empty()) {
      const auto split = remaining.split('\n');
      if (split.first.rtrim(" \t").ends_with("\\")) { spliced = true; break; }
      remaining = split.second;
    }
    if (spliced) continue;
    branch.bodyRange.endOffset = lineBegin;
    branch.skipped = true;
  }
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
      name != "findCudaDevice" && name != "_ConvertSMVer2Cores" &&
      name != "checkCudaCapabilities" && name != "findCudaDeviceDRV" &&
      name != "findFatbinPath" && name != "gpuDeviceInit" &&
      name != "gpuGetMaxGflopsDeviceId" && !publishedCompatMacro)
    return;
  const clang::SourceLocation fileLoc =
      sourceManager.getFileLoc(token.getLocation());
  if (fileLoc.isInvalid() || !sourceManager.isWrittenInMainFile(fileLoc))
    return;
  const unsigned offset = sourceManager.getFileOffset(fileLoc);
  if (name == "MAX" &&
      preservedMaxTokenLocations.count(fileLoc.getRawEncoding()) != 0)
    return;
  if (nvidiaSampleHelperNormalMacroOffsets.count(offset) != 0 ||
      nvidiaSampleFindDeviceNormalOffsets.count(offset) != 0 ||
      nvidiaSampleSMCoresNormalOffsets.count(offset) != 0 ||
      nvidiaSampleCheckCapabilitiesNormalOffsets.count(offset) != 0)
    return;
  // A token inside a preprocessor-skipped conditional branch never reaches
  // the target compilation under the same undefined condition (for example
  // _DEBUG), so it is neither rewritten nor a residual helper dependency.
  const clang::SourceLocation tokenLocation =
      sourceManager.getLocForStartOfFile(sourceManager.getFileID(fileLoc))
          .getLocWithOffset(offset);
  for (const clang::SourceRange &skipped : SkippedSourceRanges) {
    if (skipped.getBegin().isValid() && skipped.getEnd().isValid() &&
        tokenLocation == skipped.getBegin()
            ? false
            : (tokenLocation >= skipped.getBegin() &&
               tokenLocation < skipped.getEnd())) {
      return;
    }
  }
  if (!publishedCompatMacro) {
    for (auto &branch : nvidiaSampleInactiveBranches) {
      if (branch.skipped && branch.bodyRange.file == sourceManager.getFileID(fileLoc) &&
          offset >= branch.bodyRange.beginOffset && offset < branch.bodyRange.endOffset) {
        branch.needed = true;
        return;  // A mandatory #error is committed with the helper transaction.
      }
    }
  }
  if (!nvidiaSampleHelperUnsupportedMacroUse) {
    llvm::errs()
        << "Ascify NVIDIA sample-helper closure: residual raw PP use of '"
        << name << "' keeps helper_cuda.h\n";
  }
  if (name == "_ConvertSMVer2Cores" && !nvidiaSampleSMCoresUnsupportedUse) {
    nvidiaSampleSMCoresUnsupportedUse = true;
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "CUDA sample-helper SM table boundary: unproved raw or macro observation");
    getCompilerInstance().getDiagnostics().Report(fileLoc, diagnostic);
    cudaRuntimeApiBoundaryFailed = true;
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

bool AscifyAction::ownedPreservedMaxLocation(clang::SourceLocation location) {
  if (!location.isFileID()) return false;
  auto &source = getCompilerInstance().getSourceManager();
  const auto file = source.getFileID(location);
  const auto *entry = source.getFileEntryForID(file);
  if (file == source.getMainFileID() && localHeaderContext != nullptr &&
      localHeaderContext->depth == 0 && localHeaderContext->plan != nullptr &&
      localHeaderContext->virtualInput != nullptr &&
      localHeaderContext->inputEvidence != nullptr &&
      localHeaderContext->sourcePath == localHeaderContext->plan->rootSourcePath() &&
      source.getFilename(location) == localHeaderContext->plan->rootSourcePath()) {
    bool invalid = false;
    const auto bytes = source.getBufferData(file, &invalid);
    if (invalid || bytes != *localHeaderContext->virtualInput) return false;
    // Only LocalHeader.cpp supplies this in-memory input. Original discovery
    // must equal the physical root; a flattened joint parse is unpublished and
    // still requires the existing full token/final-macro/path/byte proof before
    // LocalHeader.cpp publishes it. Do not generalize this to -remap-file.
    if (localHeaderContext->requireHelperTransaction) return true;
    const auto physical = llvm::MemoryBuffer::getFile(localHeaderContext->sourcePath);
    return physical && (*physical)->getBuffer() == bytes;
  }
  if (file == source.getMainFileID() && entry != nullptr &&
      !source.isFileOverridden(entry) && !getCompilerInstance().getLangOpts().CUDA &&
      localHeaderContext != nullptr && localHeaderContext->depth == 0 &&
      localHeaderContext->plan != nullptr &&
      localHeaderContext->sourcePath == localHeaderContext->plan->rootSourcePath() &&
      source.getFilename(location) == localHeaderContext->plan->rootSourcePath()) {
    // The explicit host-C++ frontend also uses Tool.mapVirtualFile, even for
    // an unmodified root. Its VFS UniqueID is not the physical file's UniqueID.
    // Admit only the bound root with exactly the current physical bytes; other
    // headers and caller remaps retain the normal physical-identity check.
    bool invalid = false;
    const auto bytes = source.getBufferData(file, &invalid);
    const auto physical = llvm::MemoryBuffer::getFile(localHeaderContext->sourcePath);
    return !invalid && physical && (*physical)->getBuffer() == bytes;
  }
  // Reject unselected headers before the expensive physical-buffer proof.
  // CUB/SDK macro observations never acquire ownership from file identity.
  const bool selected = file == source.getMainFileID() ||
      (localHeaderContext != nullptr && localHeaderContext->plan != nullptr &&
       localHeaderContext->plan->isSelectedRootHeader(
           source.getFilename(location).str()));
  if (!selected) return false;
  // The current physical root and selected original headers still require
  // the same complete UID and parsed/physical-buffer identity proof.
  return entry != nullptr && physicalSourceFileMatches(
      source.getFilename(location), *entry, source);
}

bool AscifyAction::ownedNttpVoidCallLocation(clang::SourceLocation location) {
  if (!location.isFileID()) return false;
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  const auto file = source.getFileID(location);
  const auto *entry = source.getFileEntryForID(file);
  const auto &preprocessor = compiler.getPreprocessorOpts();
  if (entry == nullptr || source.isFileOverridden(entry) ||
      !preprocessor.RemappedFiles.empty() || !preprocessor.RemappedFileBuffers.empty()) return false;
  if (ownedPreservedMaxLocation(location)) return true;
  if (file != source.getMainFileID() || localHeaderContext == nullptr ||
      !localHeaderContext->sourceOnlyJointInput || localHeaderContext->requireHelperTransaction ||
      localHeaderContext->depth != 0 || localHeaderContext->plan == nullptr ||
      localHeaderContext->plan->failed() || localHeaderContext->virtualInput == nullptr ||
      localHeaderContext->inputEvidence == nullptr ||
      localHeaderContext->sourcePath != localHeaderContext->plan->rootSourcePath() ||
      localHeaderContext->artifactPath != localHeaderContext->plan->rootArtifactPath() ||
      source.getFilename(location) != localHeaderContext->plan->rootSourcePath()) return false;
  // This is the exact unpublished joint buffer, not the physical root. Rebuild
  // facts in this AST; LocalHeader.cpp still verifies full evidence to publish.
  bool invalid = false;
  const auto bytes = source.getBufferData(file, &invalid);
  return !invalid && bytes == *localHeaderContext->virtualInput;
}

bool AscifyAction::recordPreservedMaxFallback(
    const clang::Token &name, const clang::MacroDefinition &definition) {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto &pp = compiler.getPreprocessor();
  const auto location = name.getLocation();
  if (compiler.getPreprocessorOpts().RetainExcludedConditionalBlocks ||
      !ownedPreservedMaxLocation(location)) return false;
  const auto file = source.getFileID(location);
  bool invalid = false;
  const auto buffer = source.getBufferData(file, &invalid);
  if (invalid) return false;
  const unsigned begin = physicalLineBegin(buffer, source.getFileOffset(location));
  unsigned end = begin;
  for (unsigned line = 0; line != 3; ++line) {
    if (end == buffer.size()) return false;
    const auto newline = buffer.find('\n', end);
    end = newline == llvm::StringRef::npos ? buffer.size() : newline + 1;
  }
  const auto text = buffer.slice(begin, end);
  // Deliberately closed grammar: three physical directives. Comments, phase-2
  // splices and alternate directive spellings do not acquire a new exception.
  if (text.contains('\\') || text.contains('\r') || text.contains('/') ||
      text.contains('%') || !text.contains('?')) return false;
  // Raw Lexer requires a NUL sentinel at BufferEnd, not a view ending in the
  // middle of the source buffer. Its base location retains physical offsets.
  const std::string ownedText = text.str();
  clang::Lexer lexer(source.getLocForStartOfFile(file).getLocWithOffset(begin),
      compiler.getLangOpts(), ownedText.data(), ownedText.data(),
      ownedText.data() + ownedText.size());
  lexer.SetCommentRetentionState(true);
  std::vector<clang::Token> tokens;
  std::vector<std::string> words;
  for (;;) {
    clang::Token token;
    lexer.LexFromRawLexer(token);
    if (token.is(clang::tok::eof)) break;
    if (tokens.size() == 32 || token.is(clang::tok::comment)) return false;
    tokens.push_back(token);
    words.push_back(pp.getSpelling(token));
  }
  if (tokens.size() < 22 || !tokens[7].isAnyIdentifier() ||
      !tokens[9].isAnyIdentifier() || words[7] == words[9] ||
      tokens[2].getLocation() != location) return false;
  const std::string a = words[7], b = words[9];
  for (const auto &parameter : {a, b})
    if (pp.getIdentifierInfo(parameter)->getTokenID() != clang::tok::identifier ||
        CUDA_RENAMES_MAP().count(parameter) || isAscifyCudaCompatReservedMacro(parameter) ||
        llvm::StringRef(parameter).starts_with("__")) return false;
  const std::vector<std::string> prefix =
      {"#", "ifndef", "MAX", "#", "define", "MAX", "(", a, ",", b, ")"};
  if (!std::equal(prefix.begin(), prefix.end(), words.begin()) ||
      words[words.size()-2] != "#" || words.back() != "endif" ||
      source.getFileOffset(tokens[6].getLocation()) !=
          source.getFileOffset(tokens[5].getLocation()) + 3) return false;
  for (unsigned i = 1; i < tokens.size(); ++i)
    if (tokens[i].isAtStartOfLine() != (i == 3 || i == tokens.size()-2))
      return false;
  const std::vector<std::string> body(words.begin()+11, words.end()-2);
  if (body != std::vector<std::string>{"(",a,">",b,"?",a,":",b,")"} &&
      body != std::vector<std::string>{"(","(",a,">",b,")","?",a,":",b,")"} &&
      body != std::vector<std::string>{"(","(",a,"<",b,")","?",b,":",a,")"})
    return false;
  const auto *active = definition.getMacroInfo();
  if (active != nullptr) {
    const bool frozen = !active->isVariadic() &&
        frozenNvidiaSampleHelperFileRole(source, active->getDefinitionLoc()) ==
            FrozenOfficialNvidiaSampleHelperCuda &&
        activeFrozenHelperCudaMaxBodyMatches(*active, pp);
    if ((!frozen && (!preservedMaxProviderIsCaller ||
                     preservedMaxProvider != active)) ||
        (preservedMaxProvider != nullptr && preservedMaxProvider != active))
      return false;
    preservedMaxProvider = active;
    if (frozen) preservedMaxReplayRequired = true;
    if (frozen && file != source.getMainFileID() &&
        localHeaderContext != nullptr && localHeaderContext->depth == 0 &&
        localHeaderContext->plan != nullptr)
      localHeaderContext->plan->requireInheritedHelperContext(
          source.getFilename(location).str());
  } else if (preservedMaxProvider != nullptr) {
    return false; // A second macro epoch is never restored by this proof.
  }
  preservedMaxFallbacks.push_back({location, tokens[5].getLocation(),
      {file, begin, end}, a, b, body, active == nullptr});
  preservedMaxTokenLocations.insert(location.getRawEncoding());
  preservedMaxTokenLocations.insert(tokens[5].getLocation().getRawEncoding());
  preservedMaxNeeded = true;
  return true;
}

bool AscifyAction::recordPreservedMaxDefinition(const clang::Token &name) {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto &pp = compiler.getPreprocessor();
  const auto *info = pp.getMacroDefinition(name.getIdentifierInfo()).getMacroInfo();
  if (info == nullptr || info->isVariadic() || !info->isFunctionLike() ||
      info->getNumParams() != 2) return false;
  if (frozenNvidiaSampleHelperFileRole(source, name.getLocation()) ==
          FrozenOfficialNvidiaSampleHelperCuda &&
      activeFrozenHelperCudaMaxBodyMatches(*info, pp)) {
    if (preservedMaxProvider != nullptr && preservedMaxProvider != info)
      nvidiaSampleHelperUnsupportedMacroUse = true;
    else preservedMaxProvider = info;
    return false; // Existing frozen-owner audit applies to the helper itself.
  }
  if (!ownedPreservedMaxLocation(name.getLocation()) ||
      preservedMaxProvider != nullptr) return false;
  for (const auto &fallback : preservedMaxFallbacks) {
    if (!fallback.originallyActive || fallback.definitionName != name.getLocation())
      continue;
    auto parameter = info->param_begin();
    if ((*parameter++)->getName() != fallback.parameterLeft ||
        (*parameter)->getName() != fallback.parameterRight) return false;
    std::vector<std::string> body;
    for (const auto &token : info->tokens()) body.push_back(pp.getSpelling(token));
    if (body != fallback.replacementTokens) return false;
    preservedMaxProvider = info;
    preservedMaxProviderIsCaller = true;
    return true;
  }
  return false;
}

bool AscifyAction::admitPreservedMaxExpansion(
    const clang::Token &name, const clang::MacroDefinition &definition,
    clang::SourceRange range, const clang::MacroArgs *arguments) {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  const auto location = name.getLocation();
  if (definition.getMacroInfo() != preservedMaxProvider ||
      preservedMaxProvider == nullptr || arguments == nullptr ||
      arguments->getNumMacroArguments() != 2 || !location.isFileID() ||
      !range.getBegin().isFileID() || !range.getEnd().isFileID()) return false;
  const bool image = (frozenOfficialNvidiaSampleHelperFunctionsSeen &&
      isDirectFrozenFunctionsImageProviderInstance(location)) ||
      isProvedDirectImageInstance(location);
  if (!image && !ownedPreservedMaxLocation(location)) return false;
  bool invalidBuffer = false;
  const auto buffer = source.getBufferData(source.getFileID(location), &invalidBuffer);
  if (invalidBuffer) return false;
  const auto line = buffer.slice(physicalLineBegin(buffer, source.getFileOffset(location)),
                                 source.getFileOffset(location)).ltrim();
  if (line.starts_with("#") || line.starts_with("%:")) return false;
  const auto end = clang::Lexer::getLocForEndOfToken(
      range.getEnd(), 0, source, compiler.getLangOpts());
  if (!end.isFileID() || source.getFileID(end) != source.getFileID(location) ||
      range.getBegin() != location) return false;
  // Keep the invocation and all effects exactly as spelled. Outside the
  // authenticated image helper, argument macros and CUDA names require
  // separate semantic proofs and remain excluded from this small domain.
  if (!image) for (unsigned index = 0; index != 2; ++index) {
    const auto *token = arguments->getUnexpArgument(index);
    unsigned count = 0;
    for (; token->isNot(clang::tok::eof); ++token) {
      if (++count > 512 || token->isAnnotation() || token->getLocation().isMacroID() ||
          token->is(clang::tok::hash) || token->is(clang::tok::hashhash)) return false;
      if (const auto *identifier = token->getIdentifierInfo()) {
        const auto word = identifier->getName();
        if (compiler.getPreprocessor().getMacroDefinition(identifier) ||
            CUDA_RENAMES_MAP().count(word) || word.starts_with("cu") ||
            word.starts_with("__") || word == "_Pragma")
          return false;
      }
    }
    if (count == 0) return false;
  }
  const SemanticRewriteRange invocation{source.getFileID(location),
      source.getFileOffset(location), source.getFileOffset(end)};
  if (invocation.beginOffset >= invocation.endOffset) return false;
  preservedMaxInvocationRanges.push_back(invocation);
  preservedMaxTokenLocations.insert(location.getRawEncoding());
  if (!preservedMaxProviderIsCaller) preservedMaxReplayRequired = true;
  return true;
}

void AscifyAction::auditRawPreservedMaxTokens(clang::FileID file) {
  if (!preservedMaxNeeded || file.isInvalid()) return;
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  const auto start = source.getLocForStartOfFile(file);
  const unsigned role = frozenNvidiaSampleHelperFileRole(source, start);
  // These complete authenticated files remain in the output. Their closed
  // fallbacks and the image consumer retain the recorded active definition.
  if (role == FrozenOfficialNvidiaSampleHelperCuda ||
      role == FrozenOfficialNvidiaSampleHelperString ||
      role == FrozenOfficialNvidiaSampleHelperFunctions ||
      role == FrozenOfficialNvidiaSampleHelperImage ||
      role == FrozenOfficialNvidiaSampleHelperTimer) return;
  bool invalid = false;
  const auto buffer = source.getBufferData(file, &invalid);
  if (invalid) { nvidiaSampleHelperUnsupportedMacroUse = true; return; }
  clang::Lexer lexer(start, compiler.getLangOpts(), buffer.begin(), buffer.begin(), buffer.end());
  for (;;) {
    clang::Token token;
    lexer.LexFromRawLexer(token);
    if (token.is(clang::tok::eof)) break;
    if (!token.isAnyIdentifier() || compiler.getPreprocessor().getSpelling(token) != "MAX")
      continue;
    const auto location = token.getLocation();
    if (preservedMaxTokenLocations.count(location.getRawEncoding())) continue;
    if (file == source.getMainFileID() &&
        nvidiaSampleHelperNormalMacroOffsets.count(source.getFileOffset(location))) continue;
    nvidiaSampleHelperUnsupportedMacroUse = true;
    llvm::errs() << "Ascify NVIDIA sample-helper closure: residual preserved MAX token\n";
    return;
  }
}

bool AscifyAction::provePreservedMaxFinalState() {
  if (!preservedMaxNeeded) return true;
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto &pp = compiler.getPreprocessor();
  if (compiler.getPreprocessorOpts().RetainExcludedConditionalBlocks ||
      preservedMaxProvider == nullptr ||
      pp.getMacroDefinition(pp.getIdentifierInfo("MAX")).getMacroInfo() != preservedMaxProvider)
    return false;
  if (preservedMaxProviderIsCaller) {
    if (preservedMaxReplayRequired ||
        !ownedPreservedMaxLocation(preservedMaxProvider->getDefinitionLoc())) return false;
  } else if (!preservedMaxReplayRequired ||
      frozenNvidiaSampleHelperFileRole(source, preservedMaxProvider->getDefinitionLoc()) !=
          FrozenOfficialNvidiaSampleHelperCuda ||
      !activeFrozenHelperCudaMaxBodyMatches(*preservedMaxProvider, pp)) return false;
  auto overlaps = [](const SemanticRewriteRange &a, const SemanticRewriteRange &b) {
    return a.file == b.file && a.beginOffset < b.endOffset && b.beginOffset < a.endOffset;
  };
  for (const auto &fallback : preservedMaxFallbacks) {
    if (!ownedPreservedMaxLocation(fallback.guardName)) return false;
    for (const auto &edit : SemanticRewriteRanges)
      if (overlaps(fallback.range, edit)) return false;
  }
  for (const auto &invocation : preservedMaxInvocationRanges) {
    for (const auto &edit : SemanticRewriteRanges)
      if (overlaps(invocation, edit)) return false;
    for (const auto &observer : nvidiaSampleMacroObservationRanges)
      if (observer.file == invocation.file && observer.beginOffset <= invocation.beginOffset &&
          invocation.endOffset <= observer.endOffset &&
          (observer.beginOffset < invocation.beginOffset || invocation.endOffset < observer.endOffset))
        return false;
  }
  auditRawPreservedMaxTokens(source.getMainFileID());
  for (const auto file : nvidiaSampleRetainedFileIds) auditRawPreservedMaxTokens(file);
  return !nvidiaSampleHelperUnsupportedMacroUse;
}

bool AscifyAction::admitFrozenHelperMaxFallback(
    const clang::Token &macroNameToken) {
  // This does not admit general observations of MAX. A retained, frozen image
  // provider has already executed its internal MAX use. Removing helper_cuda
  // therefore leaves MAX defined by that same provider before this main-file
  // fallback. Only this exact, inactive three-directive fallback is admitted;
  // its tokens remain unchanged and every other PP/token audit still applies.
  if (!frozenNvidiaSampleInternalImageMaxExpansionSeen ||
      preprocessorConditionalDepth != 1 ||
      macroNameToken.getLocation().isMacroID())
    return false;
  auto &sourceManager = getCompilerInstance().getSourceManager();
  auto &preprocessor = getCompilerInstance().getPreprocessor();
  const auto location = macroNameToken.getLocation();
  if (location.isInvalid() || !sourceManager.isWrittenInMainFile(location))
    return false;
  const auto *macroInfo = preprocessor.getMacroDefinition(
      macroNameToken.getIdentifierInfo()).getMacroInfo();
  if (macroInfo == nullptr ||
      frozenNvidiaSampleHelperFileRole(
          sourceManager, macroInfo->getDefinitionLoc()) !=
          FrozenOfficialNvidiaSampleHelperCuda ||
      !activeFrozenHelperCudaMaxBodyMatches(*macroInfo, preprocessor))
    return false;
  const auto file = sourceManager.getFileID(location);
  bool invalid = false;
  const auto buffer = sourceManager.getBufferData(file, &invalid);
  if (invalid)
    return false;
  unsigned lineStart = sourceManager.getFileOffset(location);
  while (lineStart != 0 && buffer[lineStart - 1] != '\n' &&
         buffer[lineStart - 1] != '\r')
    --lineStart;
  clang::Lexer lexer(sourceManager.getLocForStartOfFile(file),
                     getCompilerInstance().getLangOpts(), buffer.begin(),
                     buffer.begin() + lineStart, buffer.end());
  std::vector<clang::Token> tokens;
  std::vector<std::string> spellings;
  for (unsigned i = 0; i != 23; ++i) {
    clang::Token token;
    lexer.LexFromRawLexer(token);
    tokens.push_back(token);
    spellings.push_back(token.is(clang::tok::eof)
                            ? "" : preprocessor.getSpelling(token));
    if (token.is(clang::tok::eof))
      break;
  }
  if (tokens.size() != 23 || !tokens[7].isAnyIdentifier() ||
      !tokens[9].isAnyIdentifier() || spellings[7] == spellings[9])
    return false;
  const auto &a = spellings[7];
  const auto &b = spellings[9];
  const std::vector<std::string> expected = {
      "#", "ifndef", "MAX", "#", "define", "MAX", "(", a, ",", b,
      ")", "(", a, ">", b, "?", a, ":", b, ")", "#", "endif"};
  if (!std::equal(expected.begin(), expected.end(), spellings.begin()) ||
      tokens[2].getLocation() != location ||
      sourceManager.getFileOffset(tokens[6].getLocation()) !=
          sourceManager.getFileOffset(tokens[5].getLocation()) + 3 ||
      !tokens[3].isAtStartOfLine() || !tokens[20].isAtStartOfLine() ||
      (!tokens[22].is(clang::tok::eof) && !tokens[22].isAtStartOfLine()))
    return false;
  // No newline may turn a replacement-list token into a separate directive
  // or statement. The two # tokens above are the only later line starts.
  for (unsigned i = 1; i != 22; ++i)
    if (i != 3 && i != 20 && tokens[i].isAtStartOfLine())
      return false;
  nvidiaSampleHelperNormalMacroOffsets.insert(
      sourceManager.getFileOffset(tokens[2].getLocation()));
  nvidiaSampleHelperNormalMacroOffsets.insert(
      sourceManager.getFileOffset(tokens[5].getLocation()));
  return true;
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
  if (name == "MAX" && directive == "#ifndef" &&
      admitFrozenHelperMaxFallback(macroNameToken))
    return;
  if (name == "MAX" && directive == "#ifndef" &&
      recordPreservedMaxFallback(macroNameToken,
          getCompilerInstance().getPreprocessor().getMacroDefinition(
              macroNameToken.getIdentifierInfo())))
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
      FrozenOfficialNvidiaSampleHelperTimer |
      FrozenOfficialNvidiaSampleHelperDrvapi;
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
      FrozenOfficialNvidiaSampleHelperTimer |
      FrozenOfficialNvidiaSampleHelperDrvapi;
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

bool AscifyAction::directImageIncludeContains(const DirectImageInclude &include,
    clang::SourceLocation location) {
  auto &source = getCompilerInstance().getSourceManager();
  return include.hash.isValid() && include.end.isValid() && include.end.isFileID() &&
      location.isValid() && location.isFileID() &&
      source.getFileID(location) == source.getFileID(include.hash) &&
      source.getFileOffset(location) >= source.getFileOffset(include.hash) &&
      source.getFileOffset(location) <= source.getFileOffset(include.end);
}

bool AscifyAction::directImageNodeMatches(const DirectImageNode &node, unsigned role) {
  auto &source = getCompilerInstance().getSourceManager();
  if (node.file.isInvalid()) return false;
  const auto begin = source.getLocForStartOfFile(node.file);
  const auto *entry = source.getFileEntryForID(node.file);
  bool invalid = false;
  const auto bytes = source.getBufferData(node.file, &invalid);
  return entry != nullptr && !invalid && entry->getSize() == node.size &&
      entry->getUniqueID().getDevice() == node.device &&
      entry->getUniqueID().getFile() == node.inode &&
      contextualCanonicalPath(source.getFilename(begin)) == node.path &&
      frozenNvidiaSampleHelperFileRole(source, begin) == role &&
      bytes.size() == node.size && contextualInputSha256(bytes) == node.sha256 &&
      physicalSourceFileMatches(node.path, *entry, source);
}

void AscifyAction::observeDirectImageInclude(clang::SourceLocation hash,
    const clang::Token &directive, llvm::StringRef name,
    clang::CharSourceRange range, llvm::StringRef resolved,
    const clang::Module *module) {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto &w = directImageWitness;
  const auto &options = compiler.getPreprocessorOpts();
  if (!ordinaryRuntimeProfileInput(compiler) || module != nullptr ||
      !options.RemappedFiles.empty() || !options.RemappedFileBuffers.empty() ||
      !compiler.getHeaderSearchOpts().VFSOverlayFiles.empty() ||
      !hash.isValid() || !hash.isFileID() || !range.getBegin().isFileID() ||
      !range.getEnd().isFileID() || directive.getIdentifierInfo() == nullptr ||
      directive.getIdentifierInfo()->getPPKeywordID() != clang::tok::pp_include ||
      source.getFileID(hash) != source.getFileID(range.getBegin()) ||
      source.getFileID(hash) != source.getFileID(range.getEnd())) return;
  bool invalid = false;
  const auto text = clang::Lexer::getSourceText(range, source, compiler.getLangOpts(), &invalid);
  if (invalid || (text != name && text != "<" + name.str() + ">" &&
      text != "\"" + name.str() + "\"" && text != "<" + name.str() &&
      text != "\"" + name.str())) return;
  auto intent = [&]() {
    DirectImageInclude result;
    const auto path = contextualCanonicalPath(resolved);
    auto entry = source.getFileManager().getFileRef(resolved);
    if (!entry) { llvm::consumeError(entry.takeError()); return result; }
    if (path.empty() || !physicalSourceFileMatches(path, entry->getFileEntry(), source))
      return result;
    result.hash = hash; result.end = range.getEnd(); result.path = path;
    result.device = entry->getFileEntry().getUniqueID().getDevice();
    result.inode = entry->getFileEntry().getUniqueID().getFile();
    if (result.device == 0 && result.inode == 0) return DirectImageInclude();
    return result;
  };
  const auto parent = source.getFileID(hash);
  if (source.isWrittenInMainFile(hash)) {
    if (preprocessorConditionalDepth != 0 || !ownedPreservedMaxLocation(hash)) return;
    if (name == "helper_cuda.h") {
      if (w.cudaRoot.hash.isValid()) { w.invalid = true; return; }
      w.cudaRoot = intent();
      w.rootSha256 = contextualInputSha256(source.getBufferData(source.getMainFileID()));
    } else if (name == "helper_image.h" && w.cuda.file.isValid()) {
      const auto candidate = intent();
      if (w.imageRoot.hash.isValid()) {
        if (!candidate.hash.isValid() || candidate.path != w.imageRoot.path ||
            candidate.device != w.imageRoot.device || candidate.inode != w.imageRoot.inode)
          w.invalid = true;
        else w.repeats.push_back(candidate);
        return;
      }
      w.imageRoot = candidate;
      const auto *macro = compiler.getPreprocessor().getMacroInfo(
          compiler.getPreprocessor().getIdentifierInfo("MAX"));
      if (macro == nullptr || macro != preservedMaxProvider || preservedMaxProviderIsCaller ||
          !macro->getDefinitionLoc().isFileID() ||
          source.getFileID(macro->getDefinitionLoc()) != w.cuda.file ||
          !activeFrozenHelperCudaMaxBodyMatches(*macro, compiler.getPreprocessor()))
        w.invalid = true;
      else w.max = macro;
    }
  } else if (name == "helper_string.h" && parent == w.cuda.file) {
    if (w.cudaString.hash.isValid()) w.invalid = true;
    else w.cudaString = intent();
  } else if (name == "helper_string.h" && parent == w.image.file) {
    if (w.imageString.hash.isValid()) w.invalid = true;
    else {
      w.imageString = intent();
      if (!w.imageString.hash.isValid() || w.string.file.isInvalid() ||
          w.imageString.path != w.string.path || w.imageString.device != w.string.device ||
          w.imageString.inode != w.string.inode) w.invalid = true;
    }
  } else if (name == "exception.h" && parent == w.image.file) {
    if (w.utilityInclude.hash.isValid()) w.invalid = true;
    else w.utilityInclude = intent();
  }
}

void AscifyAction::observeDirectImageEntry(clang::FileID file, bool firstPhysicalEntry) {
  auto &source = getCompilerInstance().getSourceManager();
  auto &w = directImageWitness;
  const auto include = source.getIncludeLoc(file);
  auto capture = [&](const DirectImageInclude &intent, unsigned role, DirectImageNode &node) {
    if (!directImageIncludeContains(intent, include)) return false;
    const auto *entry = source.getFileEntryForID(file);
    bool invalid = false;
    const auto bytes = source.getBufferData(file, &invalid);
    const auto path = contextualCanonicalPath(source.getFilename(source.getLocForStartOfFile(file)));
    if (!firstPhysicalEntry || entry == nullptr || invalid || node.file.isValid() ||
        path != intent.path || entry->getUniqueID().getDevice() != intent.device ||
        entry->getUniqueID().getFile() != intent.inode ||
        frozenNvidiaSampleHelperFileRole(source, source.getLocForStartOfFile(file)) != role ||
        !physicalSourceFileMatches(path, *entry, source)) { w.invalid = true; return true; }
    node = {file, path, contextualInputSha256(bytes), intent.device, intent.inode,
            static_cast<std::uint64_t>(bytes.size())};
    return true;
  };
  if (capture(w.cudaRoot, FrozenOfficialNvidiaSampleHelperCuda, w.cuda)) return;
  if (capture(w.cudaString, FrozenOfficialNvidiaSampleHelperString, w.string)) return;
  if (capture(w.imageRoot, FrozenOfficialNvidiaSampleHelperImage, w.image)) return;
  if (directImageIncludeContains(w.utilityInclude, include)) {
    bool invalid = false;
    const auto bytes = source.getBufferData(file, &invalid);
    if (invalid || bytes.size() != 6168 || !sha256Equals(bytes,
        "ad043023a652bf3610671aad88237383c7bf2e17c66f8da5830d1d668fe7bbe7")) {
      w.invalid = true;
      return;
    }
    capture(w.utilityInclude, FrozenNvidiaSampleHelperNone, w.utility);
    return;
  }
  if (directImageIncludeContains(w.imageString, include)) w.invalid = true; // expected same-guard skip
  for (const auto &repeat : w.repeats)
    if (directImageIncludeContains(repeat, include)) w.invalid = true;
}

void AscifyAction::observeDirectImageSkip(const clang::FileEntry &entry,
    const clang::Token &filename, llvm::StringRef resolved) {
  auto &w = directImageWitness;
  const auto path = contextualCanonicalPath(resolved);
  auto skip = [&](DirectImageInclude &include) {
    if (directImageIncludeContains(include, filename.getLocation()) &&
        include.path == path && entry.getUniqueID().getDevice() == include.device &&
        entry.getUniqueID().getFile() == include.inode) include.skipped = true;
  };
  skip(w.imageString);
  for (auto &repeat : w.repeats) skip(repeat);
}

bool AscifyAction::isProvedDirectImageInstance(clang::SourceLocation location) {
  auto &w = directImageWitness;
  auto &source = getCompilerInstance().getSourceManager();
  return w.requested && !w.invalid && location.isValid() && location.isFileID() &&
      source.getFileID(location) == w.image.file &&
      directImageNodeMatches(w.image, FrozenOfficialNvidiaSampleHelperImage) &&
      getCompilerInstance().getPreprocessor().getMacroInfo(
          getCompilerInstance().getPreprocessor().getIdentifierInfo("MAX")) == w.max;
}

bool AscifyAction::requestDirectImageMaxReplay(const clang::Token &name,
    const clang::MacroDefinition &definition, clang::SourceRange range,
    const clang::MacroArgs *arguments) {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto &pp = compiler.getPreprocessor();
  auto &w = directImageWitness;
  if (w.image.file.isInvalid() || !name.getLocation().isFileID() ||
      source.getFileID(name.getLocation()) != w.image.file) return false;
  if (w.invalid || w.requested || arguments == nullptr || arguments->getNumMacroArguments() != 2 ||
      range.getBegin() != name.getLocation() || !range.getEnd().isFileID() ||
      source.getFileID(range.getEnd()) != w.image.file || definition.isAmbiguous() ||
      !definition.getModuleMacros().empty() || definition.getMacroInfo() != w.max ||
      w.max == nullptr || preservedMaxProvider != w.max || preservedMaxProviderIsCaller ||
      !w.imageString.skipped || !directImageNodeMatches(w.cuda, FrozenOfficialNvidiaSampleHelperCuda) ||
      !directImageNodeMatches(w.string, FrozenOfficialNvidiaSampleHelperString) ||
      !directImageNodeMatches(w.image, FrozenOfficialNvidiaSampleHelperImage)) return false;
  const auto bytes = source.getBufferData(w.image.file);
  const llvm::StringRef call = "MAX((float)epsilon, __MIN_EPSILON_ERROR)";
  const auto offset = bytes.find(call);
  const auto end = clang::Lexer::getLocForEndOfToken(range.getEnd(), 0, source, compiler.getLangOpts());
  if (offset == llvm::StringRef::npos || bytes.find(call, offset + 1) != llvm::StringRef::npos ||
      source.getFileOffset(name.getLocation()) != offset || !end.isFileID() ||
      source.getFileID(end) != w.image.file || source.getFileOffset(end) != offset + call.size()) return false;
  const std::vector<std::string> expected[] = {{"(", "float", ")", "epsilon"}, {"__MIN_EPSILON_ERROR"}};
  for (unsigned i = 0; i != 2; ++i) {
    std::vector<std::string> actual;
    for (const auto *token = arguments->getUnexpArgument(i); token->isNot(clang::tok::eof); ++token) {
      if (actual.size() == 4 || !token->getLocation().isFileID() ||
          source.getFileID(token->getLocation()) != w.image.file) return false;
      actual.push_back(pp.getSpelling(*token));
    }
    if (actual != expected[i]) return false;
  }
  if (pp.getMacroInfo(pp.getIdentifierInfo("float")) || pp.getMacroInfo(pp.getIdentifierInfo("epsilon"))) return false;
  const auto *epsilon = pp.getMacroInfo(pp.getIdentifierInfo("__MIN_EPSILON_ERROR"));
  if (epsilon == nullptr || !epsilon->getDefinitionLoc().isFileID() ||
      source.getFileID(epsilon->getDefinitionLoc()) != w.image.file ||
      !frontendCgMacroTokens(*epsilon, pp, {"1e-3f"})) return false;
  const char *guards[] = {"COMMON_HELPER_CUDA_H_", "COMMON_HELPER_IMAGE_H_", "COMMON_HELPER_STRING_H_"};
  const clang::FileID owners[] = {w.cuda.file, w.image.file, w.string.file};
  for (unsigned i = 0; i != 3; ++i) {
    const auto *guard = pp.getMacroInfo(pp.getIdentifierInfo(guards[i]));
    if (guard == nullptr || guard->isFunctionLike() || guard->getNumTokens() != 0 ||
        !guard->getDefinitionLoc().isFileID() || source.getFileID(guard->getDefinitionLoc()) != owners[i]) return false;
    w.guards[i] = guard;
  }
  w.epsilon = epsilon; w.invocation = name.getLocation(); w.requested = true;
  preservedMaxNeeded = true; preservedMaxReplayRequired = true;
  llvm::errs() << "Ascify direct-image MAX replay v1: one literal consumer; exact CUDA macro epoch; functions_root="
               << (frozenOfficialNvidiaSampleHelperFunctionsSeen ? 1 : 0) << "\n";
  return true;
}

void AscifyAction::recordDirectImageUtilityControl(llvm::StringRef name,
    llvm::StringRef kind, clang::SourceLocation location,
    const clang::MacroInfo *macro) {
  auto &source = getCompilerInstance().getSourceManager();
  const auto file = directImageWitness.utility.file;
  const bool inUtility = file.isValid() && location.isValid() &&
      (source.getFileID(source.getExpansionLoc(location)) == file ||
       source.getFileID(source.getSpellingLoc(location)) == file);
  const bool guard = name == "COMMON_EXCEPTION_H_";
  // The exact input has one outer #ifndef, not arbitrary conditional/pragma
  // consumers. Keep these observations even before the provider is requested.
  if (inUtility && kind != "define" && kind != "expansion" &&
      !(guard && kind == "ifndef"))
    directImageUtilityInvalid = true;
  if (!guard && !(name == "EXIT_FAILURE" && (kind == "define" || kind == "undef")))
    return;
  if (directImageUtilityControls.size() == 64) {
    directImageUtilityInvalid = true;
    return;
  }
  directImageUtilityControls.push_back({name.str(), kind.str(), location, macro});
}

void AscifyAction::observeDirectImageUtilityMacro(const clang::Token &name,
    const clang::MacroDefinition &definition, clang::SourceRange range,
    const clang::MacroArgs *arguments) {
  auto &source = getCompilerInstance().getSourceManager();
  const auto *macro = definition.getMacroInfo();
  if (name.getIdentifierInfo())
    recordDirectImageUtilityControl(name.getIdentifierInfo()->getName(),
        "expansion", name.getLocation(), macro);
  const auto file = directImageWitness.utility.file;
  if (file.isInvalid() || name.getLocation().isInvalid()) return;
  const auto spelling = source.getSpellingLoc(name.getLocation());
  const auto expansion = source.getExpansionLoc(name.getLocation());
  if (source.getFileID(spelling) != file && source.getFileID(expansion) != file)
    return;
  if (name.getIdentifierInfo() == nullptr || directImageUtilityMacroUses.size() == 16) {
    directImageUtilityInvalid = true;
    return;
  }
  DirectImageUtilityMacroUse use;
  use.name = name.getIdentifierInfo()->getName().str();
  use.token = name.getLocation(); use.spelling = spelling; use.expansion = expansion;
  use.definition = macro ? macro->getDefinitionLoc() : clang::SourceLocation();
  use.range = range; use.macro = macro; use.arguments = arguments != nullptr;
  use.ambiguous = definition.isAmbiguous() || !definition.getModuleMacros().empty();
  if (use.definition.isFileID()) {
    const auto providerFile = source.getFileID(use.definition);
    const auto *entry = source.getFileEntryForID(providerFile);
    bool invalid = false;
    const auto bytes = source.getBufferData(providerFile, &invalid);
    if (entry != nullptr && !invalid)
      use.provider = {providerFile, contextualCanonicalPath(source.getFilename(use.definition)),
          contextualInputSha256(bytes), entry->getUniqueID().getDevice(),
          entry->getUniqueID().getFile(), static_cast<std::uint64_t>(bytes.size())};
  }
  // Both endpoints are retained: argument prescan/nested/foreign spelling is
  // not equivalent to a direct default expansion at the consumer's token.
  directImageUtilityMacroUses.push_back(std::move(use));
}

bool AscifyAction::sealDirectImageUtility() {
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto &pp = compiler.getPreprocessor();
  auto &w = directImageWitness;
  const auto file = w.utility.file;
  if (directImageUtilityInvalid || file.isInvalid() ||
      w.utility.size != 6168 || w.utility.sha256 !=
          "ad043023a652bf3610671aad88237383c7bf2e17c66f8da5830d1d668fe7bbe7" ||
      !directImageNodeMatches(w.utility, FrozenNvidiaSampleHelperNone) ||
      !directImageIncludeContains(w.utilityInclude, source.getIncludeLoc(file)) ||
      source.getFileID(w.utilityInclude.hash) != w.image.file ||
      w.utilityInclude.path != w.utility.path ||
      w.utilityInclude.device != w.utility.device ||
      w.utilityInclude.inode != w.utility.inode || directImageUtilityMacroUses.size() != 1)
    return false;
  const auto bytes = source.getBufferData(file);
  const llvm::StringRef guardName = "COMMON_EXCEPTION_H_";
  const auto guardIf = bytes.find(guardName);
  if (guardIf == llvm::StringRef::npos) return false;
  const auto guardDefine = bytes.find(guardName, guardIf + guardName.size());
  const auto *guard = pp.getMacroInfo(pp.getIdentifierInfo(guardName));
  if (guardDefine == llvm::StringRef::npos ||
      guard == nullptr || guard->isFunctionLike() || guard->getNumTokens() != 0 ||
      !guard->getDefinitionLoc().isFileID() ||
      source.getFileID(guard->getDefinitionLoc()) != file ||
      source.getFileOffset(guard->getDefinitionLoc()) != guardDefine) return false;
  const auto &use = directImageUtilityMacroUses.front();
  const auto *macro = use.macro;
  const auto failureOffset = bytes.find("EXIT_FAILURE");
  if (use.name != "EXIT_FAILURE" || macro == nullptr || macro->isBuiltinMacro() ||
      macro->isFunctionLike() || use.arguments || use.ambiguous ||
      !use.token.isFileID() || use.spelling != use.token || use.expansion != use.token ||
      use.range.getBegin() != use.token || use.range.getEnd() != use.token ||
      source.getFileID(use.token) != file || failureOffset == llvm::StringRef::npos ||
      source.getFileOffset(use.token) != failureOffset ||
      !frontendCgMacroTokens(*macro, pp, {"1"}) ||
      pp.getMacroInfo(pp.getIdentifierInfo("EXIT_FAILURE")) != macro ||
      !use.definition.isFileID() || !directImageNodeMatches(use.provider, FrozenNvidiaSampleHelperNone) ||
      !source.isInSystemHeader(use.definition) ||
      !(pathIsWithinPhysicalDirectory(use.provider.path, "/usr/include") ||
        pathIsWithinPhysicalDirectory(use.provider.path, "/usr/lib/gcc"))) return false;
  for (const auto &token : macro->tokens())
    if (!token.getLocation().isFileID() ||
        source.getFileID(token.getLocation()) != use.provider.file) return false;
  unsigned ifs = 0, defines = 0, nativeDefines = 0;
  for (const auto &event : directImageUtilityControls) {
    if (event.name == guardName) {
      if (!event.location.isFileID() || source.getFileID(event.location) != file) return false;
      if (event.kind == "ifndef" && event.macro == nullptr &&
          source.getFileOffset(event.location) == guardIf) ++ifs;
      else if (event.kind == "define" && event.macro == guard &&
               event.location == guard->getDefinitionLoc()) ++defines;
      else return false;
    } else if (event.name == "EXIT_FAILURE") {
      if (event.kind != "define" || event.macro != macro ||
          event.location != use.definition) return false;
      ++nativeDefines;
    }
  }
  if (ifs != 1 || defines != 1 || nativeDefines != 1) return false;
  const auto mentions = [&](llvm::StringRef text) {
    return text.contains(guardName) || text.contains("push_macro") ||
        text.contains("pop_macro") || text.contains("_Pragma") || text.contains("__pragma");
  };
  for (const auto &option : compiler.getPreprocessorOpts().Macros)
    if (mentions(option.first) || llvm::StringRef(option.first).contains("EXIT_FAILURE")) return false;
  // Include inactive directives and quoted/escaped guard-stack spellings in
  // observed caller inputs. Native/owned compiler files retain their policies;
  // an arbitrary -isystem input does not become a native provider here.
  std::set<clang::FileID> inputs = enteredSourceFileIds;
  for (auto info = source.fileinfo_begin(); info != source.fileinfo_end(); ++info)
    inputs.insert(source.translateFile(info->first));
  for (const auto input : inputs) {
    if (input.isInvalid() || input == file) continue;
    const auto *entry = source.getFileEntryForID(input);
    if (entry == nullptr) continue;
    const auto start = source.getLocForStartOfFile(input);
    const auto path = source.getFilename(start);
    if (isOwnedCudaCompilerInput(input, compiler, true) ||
        cudaRuntimeApiSdkFileIds.count(input.getHashValue()) ||
        (source.isInSystemHeader(start) && physicalSourceFileMatches(path, *entry, source) &&
         (pathIsWithinPhysicalDirectory(path, "/usr/include") ||
          pathIsWithinPhysicalDirectory(path, "/usr/lib/gcc")))) continue;
    bool invalid = false;
    const auto data = source.getBufferData(input, &invalid);
    if (invalid) return false;
    clang::Lexer lexer(start, compiler.getLangOpts(), data.begin(), data.begin(), data.end());
    for (;;) {
      clang::Token token; lexer.LexFromRawLexer(token);
      if (token.is(clang::tok::eof)) break;
      if (token.isAnyIdentifier() && mentions(pp.getSpelling(token))) return false;
      if (token.is(clang::tok::string_literal)) {
        clang::StringLiteralParser literal(llvm::ArrayRef<clang::Token>(&token, 1),
            source, compiler.getLangOpts(), compiler.getTarget(), nullptr);
        if (literal.hadError || mentions(literal.GetString())) return false;
      }
    }
  }
  return true;
}

bool AscifyAction::isProvedDirectImageUtilityFile(clang::FileID file) {
  auto &w = directImageWitness;
  return w.requested && !w.invalid && file == w.utility.file &&
      isProvedDirectImageInstance(getCompilerInstance().getSourceManager().getIncludeLoc(file)) &&
      sealDirectImageUtility();
}

bool AscifyAction::sealDirectImageWitness() {
  auto &w = directImageWitness;
  if (!w.requested) return true;
  auto &compiler = getCompilerInstance();
  auto &source = compiler.getSourceManager();
  auto &pp = compiler.getPreprocessor();
  const char *controls[] = {"COMMON_HELPER_CUDA_H_", "COMMON_HELPER_IMAGE_H_",
                           "COMMON_HELPER_STRING_H_", "__MIN_EPSILON_ERROR"};
  const auto fail = [&]() { w.invalid = true; nvidiaSampleHelperUnsupportedMacroUse = true; return false; };
  if (w.invalid || !w.imageString.skipped || nvidiaSampleHelperIncludes.size() != 1 ||
      nvidiaSampleHelperIncludes.front().hashLocation != w.cudaRoot.hash ||
      !ownedPreservedMaxLocation(w.cudaRoot.hash) || !ownedPreservedMaxLocation(w.imageRoot.hash) ||
      contextualInputSha256(source.getBufferData(source.getMainFileID())) != w.rootSha256 ||
      pp.getMacroInfo(pp.getIdentifierInfo("MAX")) != w.max || preservedMaxProvider != w.max ||
      !activeFrozenHelperCudaMaxBodyMatches(*w.max, pp) ||
      pp.getMacroInfo(pp.getIdentifierInfo(controls[3])) != w.epsilon ||
      !frontendCgMacroTokens(*w.epsilon, pp, {"1e-3f"}) ||
      !directImageNodeMatches(w.cuda, FrozenOfficialNvidiaSampleHelperCuda) ||
      !directImageNodeMatches(w.image, FrozenOfficialNvidiaSampleHelperImage) ||
      !directImageNodeMatches(w.string, FrozenOfficialNvidiaSampleHelperString) ||
      !sealDirectImageUtility()) return fail();
  for (unsigned i = 0; i != 3; ++i)
    if (pp.getMacroInfo(pp.getIdentifierInfo(controls[i])) != w.guards[i]) return fail();
  for (const auto &repeat : w.repeats) if (!repeat.skipped) return fail();
  auto mentions = [&](llvm::StringRef text) {
    return std::any_of(std::begin(controls), std::end(controls),
        [&](const char *control) { return text.contains(control); });
  };
  for (const auto &macro : compiler.getPreprocessorOpts().Macros)
    if (mentions(macro.first) || llvm::StringRef(macro.first).contains("push_macro") ||
        llvm::StringRef(macro.first).contains("pop_macro")) return fail();
  const auto main = source.getBufferData(source.getMainFileID());
  clang::Lexer lexer(source.getLocForStartOfFile(source.getMainFileID()), compiler.getLangOpts(),
      main.begin(), main.begin(), main.end());
  for (;;) {
    clang::Token token; lexer.LexFromRawLexer(token);
    if (token.is(clang::tok::eof)) break;
    if (token.is(clang::tok::raw_identifier)) {
      const auto word = pp.getSpelling(token);
      if (mentions(word) || word == "push_macro" || word == "pop_macro") return fail();
    }
    if (token.is(clang::tok::string_literal)) {
      clang::StringLiteralParser literal(llvm::ArrayRef<clang::Token>(&token, 1),
          source, compiler.getLangOpts(), compiler.getTarget(), nullptr);
      if (literal.hadError || mentions(literal.GetString())) return fail();
    }
  }
  if (localHeaderContext && localHeaderContext->inputEvidence) {
    auto &evidence = *localHeaderContext->inputEvidence;
    for (const auto *node : {&w.cuda, &w.image, &w.string, &w.utility,
                            &directImageUtilityMacroUses.front().provider}) {
      const auto path = contextualRecordPathBinding(evidence, node->path);
      if (path.empty()) return fail();
      evidence.fileSha256[path] = node->sha256;
      evidence.providerIdentities[path] = {node->device, node->inode};
      evidence.providerSizes[path] = node->size;
    }
  }
  return true;
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
      FrozenOfficialNvidiaSampleHelperString |
      FrozenOfficialNvidiaSampleHelperDrvapi;
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
  // This exact frozen file contains MAX only in its pure fallback guard and
  // definition. A physically retained, already-recorded caller fallback is
  // therefore not a semantic dependency of the removed helper body.
  if (macroNameToken.getIdentifierInfo()->getName() == "MAX" &&
      useKind == "#ifndef" &&
      frozenConsumerRole == FrozenOfficialNvidiaSampleHelperCuda &&
      preservedMaxNeeded && preservedMaxProviderIsCaller &&
      macroInfo == preservedMaxProvider)
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
  if (!sealDirectImageWitness())
    nvidiaSampleHelperUnsupportedMacroUse = true;
  if (nvidiaSampleHelperIncludes.empty())
    return;
  if (!provePreservedMaxFinalState())
    nvidiaSampleHelperUnsupportedMacroUse = true;
  // Inspect the completed callback history, not just the current prescan
  // stack. Every enclosing observer must be proved; the candidate's own
  // identical range and macro invocations inside its arguments are harmless.
  for (const auto &candidate : nvidiaSampleExpandedMacros)
    for (const auto &observer : nvidiaSampleMacroObservationRanges)
      if (observer.file == candidate.range.file &&
          observer.beginOffset <= candidate.range.beginOffset &&
          candidate.range.endOffset <= observer.endOffset &&
          (observer.beginOffset < candidate.range.beginOffset ||
           candidate.range.endOffset < observer.endOffset))
        nvidiaSampleHelperUnsupportedMacroUse = true;
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
        const bool exactProvenMaxCallee =
            candidate.kind == NvidiaSampleHelperMacroKind::CheckCudaErrors &&
            candidate.statusDomainProven &&
            existing.beginOffset >= candidate.invocationRange.beginOffset + 15U &&
            existing.endOffset < candidate.invocationRange.endOffset &&
            std::any_of(provenKernelAttributeCalleeRanges.begin(),
                        provenKernelAttributeCalleeRanges.end(),
                        [&](const SemanticRewriteRange &proved) {
                          return proved.file == existing.file &&
                                 proved.beginOffset == existing.beginOffset &&
                                 proved.endOffset == existing.endOffset;
                        });
        if (exactProvenMaxCallee)
          continue;
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
  for (const auto &candidate : nvidiaSampleSMCoresCandidates) {
    if (!stagingError.empty()) break;
    for (const auto &existing : SemanticRewriteRanges)
      if (overlaps(candidate.rewriteRange, existing))
        stagingError = "SM table call overlaps an existing rewrite";
    for (const auto &pending : stagedSemanticRanges)
      if (overlaps(candidate.rewriteRange, pending))
        stagingError = "SM table call overlaps helper transaction";
    if (!stagingError.empty()) break;
    const ct::Replacement replacement(sourceManager, candidate.nameLocation,
        candidate.rewriteRange.endOffset - candidate.rewriteRange.beginOffset,
        candidate.target);
    if (!llcompat::insertReplacement(staged, replacement, &stagingError)) break;
    committedMacroReplacements.push_back(replacement);
    committedMacroLocations.push_back(candidate.nameLocation);
    stagedSemanticRanges.push_back(candidate.rewriteRange);
  }
  for (const auto &candidate : nvidiaSampleCheckCapabilitiesCandidates) {
    if (!stagingError.empty()) break;
    for (const auto &existing : SemanticRewriteRanges)
      if (overlaps(candidate.rewriteRange, existing))
        stagingError = "capability check overlaps an existing rewrite";
    for (const auto &pending : stagedSemanticRanges)
      if (overlaps(candidate.rewriteRange, pending))
        stagingError = "capability check overlaps helper transaction";
    if (!stagingError.empty()) break;
    const ct::Replacement replacement(
        sourceManager, candidate.nameLocation,
        candidate.rewriteRange.endOffset - candidate.rewriteRange.beginOffset,
        candidate.target);
    if (!llcompat::insertReplacement(staged, replacement, &stagingError)) break;
    committedMacroReplacements.push_back(replacement);
    committedMacroLocations.push_back(candidate.nameLocation);
    stagedSemanticRanges.push_back(candidate.rewriteRange);
  }
  for (const auto &candidate : nvidiaSampleExpandedMacros) {
    if (!stagingError.empty()) break;
    for (const auto &existing : SemanticRewriteRanges)
      if (overlaps(candidate.range, existing)) stagingError = "expanded macro overlaps an existing rewrite";
    for (const auto &pending : stagedSemanticRanges)
      if (overlaps(candidate.range, pending)) stagingError = "expanded macro overlaps a pending rewrite";
    if (!stagingError.empty()) break;
    ct::Replacement replacement(sourceManager, candidate.location,
        candidate.range.endOffset - candidate.range.beginOffset, candidate.text);
    if (!llcompat::insertReplacement(staged, replacement, &stagingError)) break;
    committedMacroReplacements.push_back(replacement);
    committedMacroLocations.push_back(candidate.location);
    stagedSemanticRanges.push_back(candidate.range);
  }

  for (const NvidiaSampleTemplateGuard &guard : nvidiaSampleTemplateGuards) {
    if (!stagingError.empty())
      break;
    for (const SemanticRewriteRange &existing : SemanticRewriteRanges) {
      if (overlaps(guard.anchorRange, existing)) {
        stagingError = "template domain guard overlaps a semantic rewrite";
        break;
      }
    }
    for (const SemanticRewriteRange &pending : stagedSemanticRanges) {
      if (overlaps(guard.anchorRange, pending)) {
        stagingError = "template domain guard overlaps helper transaction";
        break;
      }
    }
    if (!stagingError.empty())
      break;
    const ct::Replacement replacement(
        sourceManager, guard.insertionLocation, 0U, guard.text);
    if (!llcompat::insertReplacement(staged, replacement, &stagingError))
      break;
    stagedSemanticRanges.push_back(guard.anchorRange);
  }

  for (const auto &branch : nvidiaSampleInactiveBranches) {
    if (!stagingError.empty()) break;
    if (!branch.needed) continue;
    for (const auto &existing : SemanticRewriteRanges)
      if (overlaps(branch.bodyRange, existing))
        stagingError = "inactive helper guard overlaps a semantic rewrite";
    for (const auto &pending : stagedSemanticRanges)
      if (overlaps(branch.bodyRange, pending))
        stagingError = "inactive helper guard overlaps helper transaction";
    if (!stagingError.empty()) break;
    const ct::Replacement guard(sourceManager, branch.insertionLocation, 0U,
        "#error \"Ascify: inactive NVIDIA helper branch was enabled in the target configuration\"\n");
    if (!llcompat::insertReplacement(staged, guard, &stagingError)) break;
    stagedSemanticRanges.push_back(branch.bodyRange);
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
      "#include <helper_string.h>\n"
      // This is the exact public policy macro from every admitted helper
      // profile. Preserve its definition timing and an existing definition,
      // rather than replacing call sites by an enum/constant (which would
      // change #if, defined(), and stringization semantics).
      "#ifndef EXIT_WAIVED\n"
      "#define EXIT_WAIVED 2\n"
      "#endif";
  if (preservedMaxNeeded && preservedMaxReplayRequired)
    includeReplacementText +=
        "\n#ifndef MAX\n#define MAX(a, b) (a > b ? a : b)\n#endif";
  const bool helperNeedsCompat =
      !nvidiaSampleHelperMacroCandidates.empty() ||
      !nvidiaSampleFindDeviceCandidates.empty() ||
      !nvidiaSampleSMCoresCandidates.empty() ||
      !nvidiaSampleCheckCapabilitiesCandidates.empty();
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
  if (!nvidiaSampleSMCoresCandidates.empty())
    llvm::errs() << "Ascify NVIDIA sample-helper closure: committed "
                 << nvidiaSampleSMCoresCandidates.size()
                 << " version-preserving legacy SM table call(s)\n";
  if (!nvidiaSampleTemplateGuards.empty()) {
    llvm::errs() << "Ascify NVIDIA sample-helper closure: committed "
                 << nvidiaSampleTemplateGuards.size()
                 << " explicit template type-domain guard(s)\n";
  }
  if (localHeaderContext != nullptr)
    localHeaderContext->helperTransactionCommitted = true;
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
  if (cudaRuntimeApiBoundaryFailed)
    return;
  if (!sealDynamicWarpGroupInputs()) return;
  if (!sealOriginalCgProviders()) return;
  const bool helperDiscovery = localHeaderContext != nullptr &&
      localHeaderContext->deferredInheritedHelperDiscovery;
  if (!helperDiscovery)
    finalizeCudaAttributeGuards();
  if (localHeaderContext != nullptr && localHeaderContext->depth == 0 &&
      localHeaderContext->plan != nullptr) {
    // References include declarations in signatures and instantiated bodies.
    // Expansion locations keep a reference introduced by a header macro in
    // that header. Declarations must physically belong to the original root
    // or another selected direct header. A sibling declaration is available
    // at this include only because the original parse resolved it; the same
    // position/token/macro proof still governs the inline transaction.
    // No declaration text, initializer or type is synthesized here.
    // The new sibling rule is limited to leaves. A non-leaf may include the
    // declaration provider itself, so it must retain ordinary closure parsing
    // instead of being misclassified as requiring an inherited context.
    const auto *inputEvidence = localHeaderContext->inputEvidence;
    if (inputEvidence != nullptr) {
      for (const auto &path : siblingSourceMacroContextUsers)
        if (!path.empty() && !inputEvidence->quotedIncludeParents.count(path))
          localHeaderContext->plan->requireInheritedSourceContext(path);
      for (const auto &path : systemSourceMacroContextUsers) {
        const bool hasOwnInclude = std::any_of(
            inputEvidence->originalEdges.begin(), inputEvidence->originalEdges.end(),
            [&](const LocalHeaderEdge &edge) { return edge.parentSourcePath == path; });
        // In particular, an angle include may have been FileSkipped because
        // the root already included it. Definition ancestry alone must not
        // turn a self-contained header into a contextual one.
        if (!path.empty() && !hasOwnInclude)
          localHeaderContext->plan->requireInheritedSourceContext(path);
      }
    }
    class ParentSourceDependencies
        : public clang::RecursiveASTVisitor<ParentSourceDependencies> {
      clang::SourceManager &source;
      LocalHeaderClosurePlan &plan;
      const LocalHeaderInputEvidence *inputEvidence;
      void observe(clang::SourceLocation use, const clang::Decl *declaration) {
        if (declaration == nullptr)
          return;
        use = source.getExpansionLoc(use);
        const auto definition = source.getSpellingLoc(declaration->getLocation());
        if (use.isInvalid() || definition.isInvalid() ||
            source.isWrittenInMainFile(use))
          return;
        if (source.isWrittenInMainFile(definition)) {
          plan.requireInheritedSourceContext(source.getFilename(use).str());
          return;
        }
        if (inputEvidence != nullptr &&
            source.getFileID(definition) != source.getFileID(use) &&
            plan.isSelectedRootHeader(source.getFilename(definition).str()) &&
            !inputEvidence->quotedIncludeParents.count(
                contextualCanonicalPath(source.getFilename(use))))
          plan.requireInheritedSourceContext(source.getFilename(use).str());
      }
     public:
      ParentSourceDependencies(clang::SourceManager &source,
                               LocalHeaderClosurePlan &plan,
                               const LocalHeaderInputEvidence *inputEvidence)
          : source(source), plan(plan), inputEvidence(inputEvidence) {}
      bool shouldVisitTemplateInstantiations() const { return true; }
      bool VisitDeclRefExpr(clang::DeclRefExpr *reference) {
        observe(reference->getExprLoc(), reference->getDecl());
        return true;
      }
      bool VisitTagTypeLoc(clang::TagTypeLoc type) {
        observe(type.getBeginLoc(), type.getDecl());
        return true;
      }
      bool VisitTypedefTypeLoc(clang::TypedefTypeLoc type) {
        observe(type.getBeginLoc(), type.getTypePtr()->getDecl());
        return true;
      }
      bool VisitTemplateSpecializationTypeLoc(clang::TemplateSpecializationTypeLoc type) {
        observe(type.getBeginLoc(),
                type.getTypePtr()->getTemplateName().getAsTemplateDecl());
        return true;
      }
    } dependencies(getCompilerInstance().getSourceManager(),
                   *localHeaderContext->plan, inputEvidence);
    dependencies.TraverseDecl(
        getCompilerInstance().getASTContext().getTranslationUnitDecl());
  }
  if (helperDiscovery) {
    // File-entry capture precedes the closing pack pragma. Refresh only raw
    // closed-region metadata now; no layout witness or owned edit is emitted.
    // The strict joint parse still proves records, macros and final edits.
    if (packedRecords.used())
      for (const auto file : enteredSourceFileIds) recordContextPragmaSource(file);
    return;
  }
  finalizePendingNvidiaSampleHelperPragma();
  for (const auto offset : requiredAtomicSymbolOffsets) {
    if (registeredCudaSymbolOffsets.count(offset)) continue;
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Ascify cannot publish global atomic output without its same-module device symbol resolver");
    getCompilerInstance().getDiagnostics().Report(diagnostic);
    return;
  }
  if (cudaDeviceSymbolGeneratedNameConflict) {
    const auto diagnostic =
        getCompilerInstance().getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "Ascify cannot publish CUDA compatibility output because an "
            "input declaration or macro conflicts with a generated device "
            "symbol resolver");
    getCompilerInstance().getDiagnostics().Report(diagnostic);
    return;
  }
  if (!auditKernelAttributeObservations()) return;
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
  std::vector<ct::Replacement> packedOwnedEdits;
  if (!packedRecords.prove(getCompilerInstance(), nvidiaSampleHelperUnsupportedMacroUse,
                           packedOwnedEdits)) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "packed-record boundary: %0");
    getCompilerInstance().getDiagnostics().Report(diagnostic) << packedRecords.error();
    return;
  }
  if (packedRecords.used())
    for (const auto file : enteredSourceFileIds) recordContextPragmaSource(file);
  finalizeNvidiaSampleHelperClosure();
  for (unsigned required:deviceAggregateRequiredHelperOffsets) {
    const bool consumed=localHeaderContext && localHeaderContext->helperTransactionCommitted && std::any_of(nvidiaSampleHelperMacroCandidates.begin(),
        nvidiaSampleHelperMacroCandidates.end(),[&](const auto& candidate) {
          return candidate.invocationRange.beginOffset==required && candidate.statusDomainProven && candidate.rewritten;
        });
    if (!consumed) {
      const auto diagnostic=getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,"device aggregate inline boundary: required helper diagnostic transaction did not commit");
      getCompilerInstance().getDiagnostics().Report(diagnostic);return;
    }
  }
  if (dynamicWarpGroups.used() && !nvidiaSampleHelperIncludes.empty() &&
      (localHeaderContext == nullptr || !localHeaderContext->helperTransactionCommitted)) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "dynamic warp group boundary: included helper requires its committed source transaction");
    getCompilerInstance().getDiagnostics().Report(diagnostic);
    return;
  }
  for (unsigned required:cudaTextureRequiredHelperOffsets) {
    const bool committed=std::any_of(nvidiaSampleHelperMacroCandidates.begin(),
        nvidiaSampleHelperMacroCandidates.end(),[&](const auto &candidate) {
          return candidate.invocationRange.beginOffset==required &&
              candidate.kind==NvidiaSampleHelperMacroKind::CheckCudaErrors &&
              candidate.statusDomainProven && candidate.rewritten;
        });
    if (!committed) {
      const auto diagnostic=getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "CUDA texture boundary: required frozen helper transaction did not commit");
      getCompilerInstance().getDiagnostics().Report(diagnostic);
      return;
    }
  }
  if (localHeaderContext != nullptr &&
      localHeaderContext->requireHelperTransaction &&
      !localHeaderContext->helperTransactionCommitted) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "contextual local-header conversion requires a proven joint helper transaction");
    getCompilerInstance().getDiagnostics().Report(diagnostic);
    return;
  }
  std::string includes;
  if (needsCudaCompatHeader && !hasCudaCompatHeader)
    includes += "#include <ascify/ascify_cuda_compat.hpp>\n";
  if (needsCudaVec3LayoutHeader)
    includes += "#include <ascify/cuda_vec3_layout.hpp>\n";
  if (needsCudaTextureHeader)
    includes += "#include <ascify/texture_compat.hpp>\n";
  if (dynamicWarpGroups.used())
    includes += "#include <ascify/cooperative_groups_compat.hpp>\n"
                "#include <ascify/dynamic_warp_groups_compat.hpp>\n";
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
  if (includes.empty()) {
    finalizePackedRecordEdits(packedOwnedEdits);
    return;
  }

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
  finalizePackedRecordEdits(packedOwnedEdits);
}

bool AscifyAction::finalizePackedRecordEdits(const std::vector<ct::Replacement> &owned) {
  if (!packedRecords.used()) return true;
  auto diagnose = [&](const std::string &reason) {
    const auto id = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "packed-record final edit boundary: %0");
    getCompilerInstance().getDiagnostics().Report(id) << reason;
    return false;
  };
  if (!packedRecords.auditFinalEdits(*replacements, {})) return diagnose(packedRecords.error());
  ct::Replacements staged(*replacements);
  for (const auto &edit : owned) {
    std::string error;
    if (!llcompat::insertReplacement(staged, edit, &error)) return diagnose(error);
  }
  if (!packedRecords.auditFinalEdits(staged, owned)) return diagnose(packedRecords.error());
  *replacements = std::move(staged);
  return true;
}

namespace {

/**
  * A silly little class to proxy PPCallbacks back to the AscifyAction class.
  */
class PPCallbackProxy : public clang::PPCallbacks {
  AscifyAction &ascifyAction;

public:
  explicit PPCallbackProxy(AscifyAction &action): ascifyAction(action) {}

#if LLVM_VERSION_MAJOR >= 23
  // This callback signature is bound to the actual LLVM23 SDK used by the
  // admitted frontend. Older versions retain conditional-profiler refusal.
  void HasInclude(clang::SourceLocation location, StringRef filename, bool angled,
                  clang::OptionalFileEntryRef file,
                  clang::SrcMgr::CharacteristicKind) override {
    ascifyAction.HasInclude(location, filename, angled,
        file ? &file->getFileEntry() : nullptr,
        file ? file->getName() : StringRef());
  }
#endif

  void FileChanged(
      clang::SourceLocation location,
      clang::PPCallbacks::FileChangeReason reason,
      clang::SrcMgr::CharacteristicKind fileType,
      clang::FileID) override {
    ascifyAction.FileChanged(location, reason, fileType);
  }

#if LLVM_VERSION_MAJOR >= 13
  void FileSkipped(const clang::FileEntryRef& file,
                   const clang::Token& filenameToken,
                   clang::SrcMgr::CharacteristicKind) override {
    ascifyAction.FileSkipped(file.getFileEntry(), filenameToken, file.getName());
  }
#endif

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

  void If(clang::SourceLocation location,
          clang::SourceRange condition,
          clang::PPCallbacks::ConditionValueKind value) override {
    ascifyAction.If(location, condition, value);
  }

  void Elif(clang::SourceLocation location, clang::SourceRange,
            clang::PPCallbacks::ConditionValueKind,
            clang::SourceLocation ifLocation) override {
    ascifyAction.Else(location, ifLocation);
  }

  void Else(clang::SourceLocation location,
            clang::SourceLocation ifLocation) override {
    ascifyAction.Else(location, ifLocation);
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

  void Endif(clang::SourceLocation location,
             clang::SourceLocation ifLocation) override {
    ascifyAction.Endif(location, ifLocation);
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
                    const clang::MacroArgs *Args) override {
    ascifyAction.MacroExpands(MacroNameTok, MD, Range, Args);
  }

  void SourceRangeSkipped(clang::SourceRange Range, clang::SourceLocation EndifLoc) override {
    ascifyAction.AddSkippedSourceRange(Range, EndifLoc);
  }
};
}

bool AscifyAction::BeginInvocation(clang::CompilerInstance &CI) {
  clang::DiagnosticsEngine &DE = CI.getDiagnostics();
  // This ClangTool invocation supplies its own FileManager. A driver overlay
  // is not guaranteed to be installed into that VFS, so silently proceeding
  // would parse different bytes from the input explicitly selected by the
  // caller. Reject before preprocessing or publishing any transformation.
  if (!CI.getHeaderSearchOpts().VFSOverlayFiles.empty()) {
    const auto ID = DE.getCustomDiagID(
        clang::DiagnosticsEngine::Error,
        "Ascify cannot translate input selected by VFS overlay arguments; "
        "this frontend does not support -ivfsoverlay");
    DE.Report(ID);
    return false;
  }
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
  if (localHeaderContext == nullptr || localHeaderContext->depth == 0)
    Statistics::current().cudaProfilerAvailabilityEvidence.clear();
  // Register yourself as the preprocessor callback, by proxy.
  PP.addPPCallbacks(std::unique_ptr<PPCallbackProxy>(new PPCallbackProxy(*this)));
#if LLVM_VERSION_MAJOR > 3
  Statistics::cudaVersionUsedByClang = Statistics::convertCudaToolkitVersion(clang::ToCudaVersion(PP.getTargetInfo().getSDKVersion()));
  llvm::errs() << " !!!!!!! CUDA SDK version detected: " << int(clang::ToCudaVersion(PP.getTargetInfo().getSDKVersion())) << "\n";
#endif
  // Snapshot the expanded token stream before any source edits. Length-prefixed
  // spelling includes actual __FILE__/__LINE__/__COUNTER__/include-level uses.
  std::string contextualTokens;
  if (localHeaderContext != nullptr && localHeaderContext->inputEvidence != nullptr) {
    const auto &options = getCompilerInstance().getPreprocessorOpts();
    const auto &language = getCompilerInstance().getLangOpts();
    localHeaderContext->inputEvidence->ordinaryPhysicalInput =
        !language.Modules && !language.CPlusPlusModules && options.ImplicitPCHInclude.empty() &&
        options.PCHThroughHeader.empty() && options.PrecompiledPreambleBytes.first == 0 &&
        !options.PrecompiledPreambleBytes.second &&
        getCompilerInstance().getFrontendOpts().ModuleFiles.empty() &&
        getCompilerInstance().getASTContext().getExternalSource() == nullptr;
#if LLVM_VERSION_MAJOR >= 10
    localHeaderContext->inputEvidence->standardPreprocessing =
        !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks;
#endif
    PP.setTokenWatcher([&](const clang::Token &token) {
      const std::string spelling = token.isAnnotation() ? std::string()
                                                        : PP.getSpelling(token);
      contextualTokens += std::to_string(static_cast<unsigned>(token.getKind())) +
          ":" + std::to_string(spelling.size()) + ":" + spelling + "\n";
    });
  }
  // Now we're done futzing with the lexer, have the subclass proceeed with Sema and AST matching.
  clang::ASTFrontendAction::ExecuteAction();
  if (localHeaderContext != nullptr && localHeaderContext->inputEvidence != nullptr) {
    PP.setTokenWatcher(nullptr);
    auto &evidence = *localHeaderContext->inputEvidence;
    evidence.ordinaryPhysicalInput = evidence.ordinaryPhysicalInput &&
        getCompilerInstance().getASTContext().getExternalSource() == nullptr;
    evidence.tokenSha256 = contextualInputSha256(contextualTokens);
    contextualTokens.clear();
    std::map<std::string, std::string> definitions;
    for (const auto &entry : PP.macros()) {
      const auto *identifier = entry.first;
      const auto *info = PP.getMacroDefinition(identifier).getMacroInfo();
      if (info == nullptr)
        continue;
      std::string definition = std::to_string(info->isBuiltinMacro()) + ":" +
          std::to_string(info->isFunctionLike()) + ":" +
          std::to_string(info->isC99Varargs()) + ":" +
          std::to_string(info->isGNUVarargs()) + ":";
      for (const auto *parameter : info->params())
        definition += parameter->getName().str() + ",";
      for (const auto &token : info->tokens()) {
        const auto spelling = PP.getSpelling(token);
        definition += std::to_string(static_cast<unsigned>(token.getKind())) +
            ":" + std::to_string(token.hasLeadingSpace()) +
            ":" + std::to_string(spelling.size()) + ":" + spelling + "\n";
      }
      definitions[identifier->getName().str()] = std::move(definition);
    }
    std::string macros = "counter=" + std::to_string(PP.getCounterValue()) + "\n";
    for (const auto &entry : definitions)
      macros += std::to_string(entry.first.size()) + ":" + entry.first + ":" +
          std::to_string(entry.second.size()) + ":" + entry.second;
    evidence.macroSha256 = contextualInputSha256(macros);
    evidence.pragmaSha256 = contextualInputSha256(contextualPragmaEvents);
  }
  auto &SM = getCompilerInstance().getSourceManager();
  // The scratch-specific parsing projection must never publish code without
  // proving every participating specialization and its whole-block schedule.
  // Reset the contextual token watcher above before any diagnostic early exit.
  if (getCompilerInstance().getDiagnostics().hasErrorOccurred())
    return;
  // A macro can erase a new class/factory QName before AST inventory. Its
  // actual expansion in the exact pinned projection is itself a new surface
  // observation, including when no remaining registered group name survives.
  if(!capturedDynamicProjectionName.empty()) {
    const auto diagnostic=getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,"dynamic warp group boundary: new projection name '%0' was captured by a macro");
    getCompilerInstance().getDiagnostics().Report(diagnostic)<<capturedDynamicProjectionName;
    cudaRuntimeApiBoundaryFailed=true;return;
  }
  class ProjectionScopeAudit : public clang::RecursiveASTVisitor<ProjectionScopeAudit> {
    std::function<bool(clang::SourceLocation)> projected;
   public:
    clang::SourceLocation bad;
    explicit ProjectionScopeAudit(std::function<bool(clang::SourceLocation)> p):projected(std::move(p)){}
    bool VisitNamespaceDecl(clang::NamespaceDecl* declaration) {
      if(declaration->getName()=="cooperative_groups"&&projected(declaration->getLocation())&&
          !declaration->getDeclContext()->isTranslationUnit()){bad=declaration->getLocation();return false;}
      return true;
    }
  } projectionScope([&](clang::SourceLocation location) {
      if(!location.isFileID())return false;
      const auto file=SM.getFileID(location);
      return verifiedFrontendCgEntryFileIds.count(file.getHashValue())&&
          verifiedFrontendCgFile(file,getCompilerInstance(),frontendCompatibility);
    });
  if(!verifiedFrontendCgEntryFileIds.empty()&&
      !projectionScope.TraverseDecl(getCompilerInstance().getASTContext().getTranslationUnitDecl())) {
    const auto diagnostic=getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,"frontend CG projection must retain its translation-unit namespace (origin %0)");
    getCompilerInstance().getDiagnostics().Report(diagnostic)<<projectionScope.bad.printToString(SM);
    cudaRuntimeApiBoundaryFailed=true;return;
  }
  if (!sealOriginalCgProviders()) return;
  // All PP callbacks and Sema have completed. Seal the entire conditional
  // proof BEFORE either surface audit, and long before output publication.
  // EndSourceFileAction must never create a late audit exception.
  if (!sealProfilerAvailabilityProof()) {
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  // A recognized helper can be provided by a selected parent-context header.
  // Before that header is inlined, its macro transaction cannot be proved.
  // These two discovery parses collect original evidence only. The existing
  // joint parse has this flag disabled and must pass every boundary below.
  if (localHeaderContext != nullptr &&
      localHeaderContext->allowInheritedHelperDiscovery &&
      localHeaderContext->depth == 0 && localHeaderContext->plan != nullptr &&
      !localHeaderContext->plan->failed() &&
      !localHeaderContext->plan->inheritedHelperHeaders().empty() &&
      localHeaderContext->inputEvidence != nullptr &&
      localHeaderContext->inputEvidence->standardPreprocessing &&
      localHeaderContext->inputEvidence->ordinaryPhysicalInput &&
      !localHeaderContext->requireHelperTransaction && !AscifyAMAP &&
      TargetRecipe == "none") {
    localHeaderContext->deferredInheritedHelperDiscovery = true;
    *replacements = ct::Replacements();
    llvm::errs() << "Ascify contextual helper discovery: evidence only; "
                    "strict joint transaction required\n";
    return;
  }
  // Complete physical/macro SDK admission before scalar64's callee proof.
  // Record/type admission follows its expression-specific closed-domain proof.
  if (cudaRuntimeApiHeaderAdmitted && !auditCudaRuntimeApiSurface(false)) {
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  std::map<unsigned, bool> scalar64SdkFiles;
  const auto scalar64Sdk = [&](clang::SourceLocation location) {
    location = SM.getSpellingLoc(location);
    if (location.isInvalid()) return false;
    const auto file = SM.getFileID(location);
    const auto found = scalar64SdkFiles.find(file.getHashValue());
    if (found != scalar64SdkFiles.end()) return found->second;
    const bool proven = cudaRuntimeApiSdkFileIds.count(file.getHashValue()) ||
        isOwnedCudaCompilerInput(file, getCompilerInstance()) ||
        (verifiedFrontendCgEntryFileIds.count(file.getHashValue()) &&
         !macroAlteredFrontendCgFileIds.count(file.getHashValue()) &&
         verifiedFrontendCgFile(file, getCompilerInstance(), frontendCompatibility));
    scalar64SdkFiles[file.getHashValue()] = proven;
    return proven;
  };
  const auto scalar64Shuffle = [&](const clang::FunctionDecl* function) {
    if (!function || !function->getIdentifier() || !function->hasAttr<clang::CUDADeviceAttr>() ||
        !scalar64Sdk(function->getLocation())) return false;
    const auto name = function->getName();
    auto& context = getCompilerInstance().getASTContext();
    if (name == "__shfl_sync" || name == "__shfl_down_sync") {
      if (function->getQualifiedNameAsString() != name || function->getNumParams() != 4)
        return false;
      return context.hasSameUnqualifiedType(function->getParamDecl(0)->getType(), context.UnsignedIntTy) &&
          context.hasSameUnqualifiedType(function->getParamDecl(1)->getType(), function->getReturnType()) &&
          context.hasSameUnqualifiedType(function->getParamDecl(2)->getType(),
              name == "__shfl_sync" ? context.IntTy : context.UnsignedIntTy) &&
          context.hasSameUnqualifiedType(function->getParamDecl(3)->getType(), context.IntTy);
    }
    const auto* method = llvm::dyn_cast<clang::CXXMethodDecl>(function);
    if (!method || (name != "shfl" && name != "shfl_down") || function->getNumParams() != 2)
      return false;
    const auto* parent = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(method->getParent());
    if (!parent || parent->getSpecializedTemplate()->getQualifiedNameAsString() !=
            "cooperative_groups::ascify_detail::tile_register_operations" ||
        parent->getTemplateArgs().size() != 1 ||
        parent->getTemplateArgs()[0].getKind() != clang::TemplateArgument::Integral ||
        parent->getTemplateArgs()[0].getAsIntegral() != 32) return false;
    const auto file = SM.getFileID(SM.getSpellingLoc(function->getLocation()));
    return verifiedFrontendCgEntryFileIds.count(file.getHashValue()) &&
        !macroAlteredFrontendCgFileIds.count(file.getHashValue()) &&
        verifiedFrontendCgFile(file, getCompilerInstance(), frontendCompatibility) &&
        context.hasSameUnqualifiedType(function->getParamDecl(0)->getType(), function->getReturnType()) &&
        context.hasSameUnqualifiedType(function->getParamDecl(1)->getType(),
            name == "shfl" ? context.IntTy : context.UnsignedIntTy);
  };
  // Exact product-owned CG plus-reduction declarations are arithmetic leaves.
  // Their bodies live in the target facade and use the integer binary64 RNE
  // helper. This never authenticates a caller overload by spelling alone.
  const auto scalar64Reduce = [&](const clang::FunctionDecl* function) {
    if (!function || !function->hasAttr<clang::CUDADeviceAttr>() ||
        function->getQualifiedNameAsString()!="cooperative_groups::reduce" ||
        function->getNumParams()!=3 || function->isVariadic() || function->hasBody()) return false;
    auto& context=getCompilerInstance().getASTContext();
    for (const auto* declaration : function->redecls()) {
      const auto at=SM.getSpellingLoc(declaration->getLocation());
      if (!at.isFileID()) return false;
      const auto file=SM.getFileID(at);
      if (!verifiedFrontendCgEntryFileIds.count(file.getHashValue()) ||
          macroAlteredFrontendCgFileIds.count(file.getHashValue()) ||
          !verifiedFrontendCgFile(file,getCompilerInstance(),frontendCompatibility) ||
          contextualCanonicalPath(SM.getFilename(at))!=
              frontendCompatibility.canonicalRoot+"/cooperative_groups/reduce.h") return false;
    }
    const auto value=function->getParamDecl(1)->getType();
    if (!function->getReturnType()->isDependentType() &&
        !context.hasSameUnqualifiedType(value,function->getReturnType())) return false;
    const auto parent=function->getParamDecl(0)->getType();
    if (!parent->isLValueReferenceType() || !parent->getPointeeType().isConstQualified()) return false;
    const auto operation=function->getParamDecl(2)->getType();
    const auto* record=operation->getAsCXXRecordDecl();
    if (record) {
      const auto* spec=llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(record);
      if (!spec || spec->getSpecializedTemplate()->getQualifiedNameAsString()!="cooperative_groups::plus" ||
          spec->getTemplateArgs().size()!=1 ||
          spec->getTemplateArgs()[0].getKind()!=clang::TemplateArgument::Type ||
          !context.hasSameUnqualifiedType(spec->getTemplateArgs()[0].getAsType(),value)) return false;
      for (const auto* declaration : record->redecls())
        if (!scalar64Sdk(declaration->getLocation())) return false;
      return true;
    }
    const auto* spec=operation->getAs<clang::TemplateSpecializationType>();
    const auto* declaration=spec ? spec->getTemplateName().getAsTemplateDecl() : nullptr;
    return declaration && declaration->getQualifiedNameAsString()=="cooperative_groups::plus" &&
        scalar64Sdk(declaration->getLocation()) && spec->template_arguments().size()==1 &&
        spec->template_arguments()[0].getKind()==clang::TemplateArgument::Type &&
        context.hasSameUnqualifiedType(spec->template_arguments()[0].getAsType(),value);
  };
  std::string scalar64Error;
  clang::SourceLocation scalar64ErrorLocation;
  if (!ascify::ProveScalar64Lowering(getCompilerInstance().getASTContext(),
          [&](clang::SourceLocation location) { return ownedPreservedMaxLocation(location); },
          scalar64Sdk, scalar64Shuffle, scalar64Reduce,
          [&](llvm::StringRef name, clang::SourceLocation location) {
            return !PP.getMacroDefinitionAtLoc(PP.getIdentifierInfo(name), location);
          }, !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks,
          scalar64Plan, scalar64ErrorLocation, scalar64Error)) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "scalar64 boundary: %0");
    getCompilerInstance().getDiagnostics().Report(scalar64ErrorLocation, diagnostic) << scalar64Error;
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  if (scalar64Plan.used() && (cudaAttributeControlModified ||
      deviceTemplateAttributeMacroConflict || frontendCgControlModified)) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "scalar64 boundary: device/profile control provenance was modified");
    getCompilerInstance().getDiagnostics().Report(diagnostic);
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  const auto nttpCgFile = [&](clang::SourceLocation location) {
    if (location.isInvalid() || cudaAttributeControlModified ||
        deviceTemplateAttributeMacroConflict || frontendCgControlModified) return false;
    const auto file = SM.getFileID(SM.getSpellingLoc(location));
    return verifiedFrontendCgEntryFileIds.count(file.getHashValue()) &&
        !macroAlteredFrontendCgFileIds.count(file.getHashValue()) &&
        verifiedFrontendCgFile(file, getCompilerInstance(), frontendCompatibility);
  };
  const auto nttpCgRecord = [&](const clang::CXXRecordDecl* record) {
    if (!record || record->getQualifiedNameAsString() != "cooperative_groups::thread_block" ||
        !record->getDefinition()) return false;
    for (const auto* declaration : record->redecls())
      if (!nttpCgFile(declaration->getLocation())) return false;
    return true;
  };
  std::string dynamicError;
  clang::SourceLocation dynamicLocation;
  std::map<unsigned,bool> dynamicCallerFiles;
  const auto dynamicCaller=[&](clang::SourceLocation location) {
    if(location.isInvalid()||!location.isFileID())return false;
    const auto file=SM.getFileID(location);
    const auto old=dynamicCallerFiles.find(file.getHashValue());
    if(old!=dynamicCallerFiles.end())return old->second;
    const auto* entry=SM.getFileEntryForID(file);
    const bool native=scalar64Sdk(location)||
        (!frontendHostMathGuardModified&&!frontendCgControlModified&&!cudaAttributeControlModified&&
         verifiedFrontendHostMathEntryFileIds.count(file.getHashValue())&&
         !macroAlteredFrontendHostMathFileIds.count(file.getHashValue())&&
         verifiedForcedFrontendHostMathFile(file,getCompilerInstance(),frontendCompatibility))||
        (entry&&trustedSystemFileIds.count(file.getHashValue())&&SM.isInSystemHeader(location)&&
         !SM.isFileOverridden(entry)&&physicalSourceFileMatches(SM.getFilename(location),*entry,SM)&&
         (pathIsWithinPhysicalDirectory(SM.getFilename(location),"/usr/include")||
          pathIsWithinPhysicalDirectory(SM.getFilename(location),"/usr/lib/gcc")));
    dynamicCallerFiles[file.getHashValue()]=!native;return !native;
  };
  if (!ascify::ProveDynamicWarpGroups(getCompilerInstance().getASTContext(),
          [&](clang::SourceLocation location) { return ownedNttpVoidCallLocation(location); },
          dynamicCaller,
          [&](const clang::Decl* declaration) { return declaration&&nttpCgFile(declaration->getLocation()); },
          [&](llvm::StringRef name,clang::SourceLocation location) {
            return !PP.getMacroDefinitionAtLoc(PP.getIdentifierInfo(name),location);
          }, ordinaryRuntimeProfileInput(getCompilerInstance())&&TargetPolicy=="dav-c310-vec"&&
              TargetRecipe=="none"&&!AscifyAMAP,
          dynamicWarpGroups,dynamicLocation,dynamicError)) {
    const auto diagnostic=getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,"dynamic warp group boundary: %0");
    getCompilerInstance().getDiagnostics().Report(dynamicLocation,diagnostic)<<dynamicError;
    cudaRuntimeApiBoundaryFailed=true;return;
  }
  if (!sealDynamicWarpGroupInputs()||!sealOriginalCgProviders())return;
  if (dynamicWarpGroups.used()) {
    for (const auto* expression:dynamicWarpGroups.valueExpressions) {
      const auto location=expression->getBeginLoc();
      if (!location.isFileID()||SM.isWrittenInMainFile(location))continue;
      if (!localHeaderContext||!localHeaderContext->plan) {
        const auto diagnostic=getCompilerInstance().getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "dynamic warp group boundary: selected header requires its current joint source transaction");
        getCompilerInstance().getDiagnostics().Report(diagnostic);
        cudaRuntimeApiBoundaryFailed=true;return;
      }
      localHeaderContext->plan->requireInheritedSourceContext(SM.getFilename(location).str());
    }
    needsCudaCompatHeader=true;
    llvm::errs()<<"Ascify dynamic warp groups v1: values="<<dynamicWarpGroups.values
        <<", factories="<<dynamicWarpGroups.factories<<", reductions="<<dynamicWarpGroups.reductions
        <<", shuffles="<<dynamicWarpGroups.shuffles<<", grid_queries="<<dynamicWarpGroups.coordinateQueries<<"\n";
  }
  ascify::FullWarpGroupStats fullWarpGroups;
  std::string fullWarpGroupError;
  clang::SourceLocation fullWarpGroupLocation;
  if (!ascify::ProveFullWarpGroups(getCompilerInstance().getASTContext(),
          [&](clang::SourceLocation location) { return ownedNttpVoidCallLocation(location); },
          [&](const clang::Decl* declaration) {
            return declaration && nttpCgFile(declaration->getLocation());
          },
          [&](const clang::Decl* declaration) {
            return declaration && scalar64Sdk(declaration->getLocation());
          },
          [&](const clang::FunctionDecl* function) {
            if (!function) return false;
            for (const auto* declaration : function->redecls()) {
              const auto location=SM.getSpellingLoc(declaration->getLocation());
              if (!location.isFileID()) return false;
              const auto file=SM.getFileID(location);
              if (cudaRuntimeApiSdkFileIds.count(file.getHashValue()) ||
                  isOwnedCudaCompilerInput(file,getCompilerInstance())) continue;
              const auto* entry=SM.getFileEntryForID(file);
              if (!entry || !trustedSystemFileIds.count(file.getHashValue()) ||
                  !SM.isInSystemHeader(location) || SM.isFileOverridden(entry) ||
                  contextualCanonicalPath(SM.getFilename(location))!="/usr/include/stdio.h" ||
                  !physicalSourceFileMatches(SM.getFilename(location),*entry,SM)) return false;
            }
            return true;
          },
          [&](llvm::StringRef name,clang::SourceLocation location) {
            const auto definition=PP.getMacroDefinitionAtLoc(PP.getIdentifierInfo(name),location);
            if (!definition) return true;
            // Only these exact native error-path macros are benign nonempty
            // host spellings needed by the original sample. They cannot lend
            // an x witness: launchInteger still requires owned FileID nodes.
            if (name!="EXIT_FAILURE" && name!="EXIT_SUCCESS" && name!="stderr") return false;
            const auto* macro=definition.getMacroInfo();
            if (!macro || definition.isAmbiguous() || !definition.getModuleMacros().empty() ||
                macro->isFunctionLike() || macro->getNumTokens()!=1) return false;
            const auto at=SM.getSpellingLoc(macro->getDefinitionLoc());
            if (!at.isFileID()) return false;
            const auto file=SM.getFileID(at);
            const auto* entry=SM.getFileEntryForID(file);
            const auto path=contextualCanonicalPath(SM.getFilename(at));
            if (!entry || !trustedSystemFileIds.count(file.getHashValue()) || !SM.isInSystemHeader(at) ||
                SM.isFileOverridden(entry) || path!=(name=="stderr"?"/usr/include/stdio.h":"/usr/include/stdlib.h") ||
                !physicalSourceFileMatches(SM.getFilename(at),*entry,SM)) return false;
            const auto replacement=clang::Lexer::getSpelling(macro->tokens()[0],SM,getCompilerInstance().getLangOpts());
            return replacement==(name=="stderr"?"stderr":name=="EXIT_FAILURE"?"1":"0");
          },
          [&](clang::FileID file) {
            if (file.isInvalid()) return false;
            if (cudaRuntimeApiSdkFileIds.count(file.getHashValue()) ||
                isOwnedCudaCompilerInput(file,getCompilerInstance()) ||
                fullWarpNativeCompilerInput(file,getCompilerInstance(),trustedSystemFileIds) ||
                (verifiedFrontendCgEntryFileIds.count(file.getHashValue()) && nttpCgFile(SM.getLocForStartOfFile(file))) ||
                (!frontendHostMathGuardModified && !frontendCgControlModified && !cudaAttributeControlModified &&
                 verifiedFrontendHostMathEntryFileIds.count(file.getHashValue()) &&
                 !macroAlteredFrontendHostMathFileIds.count(file.getHashValue()) &&
                 verifiedForcedFrontendHostMathFile(file,getCompilerInstance(),frontendCompatibility))) return true;
            const auto at=SM.getLocForStartOfFile(file);
            const auto* entry=SM.getFileEntryForID(file);
            return entry && trustedSystemFileIds.count(file.getHashValue()) && SM.isInSystemHeader(at) &&
                !SM.isFileOverridden(entry) && physicalSourceFileMatches(SM.getFilename(at),*entry,SM) &&
                (pathIsWithinPhysicalDirectory(SM.getFilename(at),"/usr/include") ||
                 pathIsWithinPhysicalDirectory(SM.getFilename(at),"/usr/lib/gcc"));
          }, !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks &&
              TargetPolicy == "dav-c310-vec" && TargetRecipe == "none",
          fullWarpGroups, fullWarpGroupLocation, fullWarpGroupError)) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "full-warp group boundary: %0");
    getCompilerInstance().getDiagnostics().Report(fullWarpGroupLocation, diagnostic) << fullWarpGroupError;
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  if (fullWarpGroups.used()) {
    for (const auto file : enteredSourceFileIds)
      if (physicalOwnedClangInput(file,getCompilerInstance()) &&
          fullWarpCommandMacroMention(file,getCompilerInstance()))
        fullWarpCompilerIncludeFailure = SM.getLocForStartOfFile(file);
    if (fullWarpCompilerIncludeFailure.isValid()) {
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "full-warp group boundary: private compiler include lookup or macro controls are unproved");
      getCompilerInstance().getDiagnostics().Report(fullWarpCompilerIncludeFailure,diagnostic);
      cudaRuntimeApiBoundaryFailed = true;
      return;
    }
    // Stage both zero-length insertions together so existing token/API edits
    // remain intact. Only the closed, effect-free host launches above obtain
    // lazy initialization; legacy32 and unrelated launch paths are unchanged.
    ct::Replacements staged(*replacements);
    std::string startupError;
    std::set<std::pair<unsigned,unsigned>> startupRanges;
    for (const auto* launch : fullWarpGroups.hostLaunches) {
      const auto begin=launch->getBeginLoc();
      const auto end=clang::Lexer::getLocForEndOfToken(
          launch->getEndLoc(),0,SM,getCompilerInstance().getLangOpts());
      if (!begin.isFileID() || !end.isFileID() || SM.getFileID(begin)!=SM.getFileID(end) ||
          SM.getFileOffset(begin)>=SM.getFileOffset(end)) {
        startupError="prelaunch edit has no complete literal range"; break;
      }
      if (!startupRanges.insert({begin.getRawEncoding(),end.getRawEncoding()}).second) continue;
      for (const auto& range : SemanticRewriteRanges)
        if (range.file==SM.getFileID(begin) && SM.getFileOffset(begin)<range.endOffset &&
            range.beginOffset<SM.getFileOffset(end)) startupError="prelaunch edit overlaps a semantic transaction";
      if (!startupError.empty()) break;
      // ACL's owned success value is0. The literal avoids capturing a caller
      // local named ACL_SUCCESS in the surrounding host lexical scope.
      if (!llcompat::insertReplacement(staged,ct::Replacement(SM,begin,0,
              "(::ascify::cudaRuntimeEnsureReady() == 0 ? (void)("),&startupError) ||
          !llcompat::insertReplacement(staged,ct::Replacement(SM,end,0,
              ") : (void)0)"),&startupError)) break;
    }
    if (!startupError.empty()) {
      const auto diagnostic=getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,"full-warp group boundary: %0");
      getCompilerInstance().getDiagnostics().Report(diagnostic)<<startupError;
      cudaRuntimeApiBoundaryFailed=true;
      return;
    }
    *replacements=std::move(staged);
    needsCudaCompatHeader = true;
    llvm::errs() << "Ascify full-warp groups v1: kernels=" << fullWarpGroups.kernels
                 << ", helpers=" << fullWarpGroups.helpers
                 << ", sync_sites=" << fullWarpGroups.syncs
                 << ", tile16_sync_requires_complete_warp=1\n";
  }
  std::string nttpVoidError;
  clang::SourceLocation nttpVoidLocation;
  if (!ascify::ProveNttpVoidCalls(getCompilerInstance().getASTContext(),
          [&](clang::SourceLocation location) { return ownedNttpVoidCallLocation(location); },
          nttpCgRecord,
          [&](const clang::FunctionDecl* function) {
            return function && function->hasAttr<clang::CUDADeviceAttr>() &&
                nttpCgFile(function->getLocation());
          },
          [&](llvm::StringRef name, clang::SourceLocation location) {
            return !PP.getMacroDefinitionAtLoc(PP.getIdentifierInfo(name), location);
          }, !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks &&
              TargetPolicy == "dav-c310-vec" && TargetRecipe == "none",
          nttpVoidCallPlan, nttpVoidLocation, nttpVoidError)) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "NTTP void call boundary: %0");
    getCompilerInstance().getDiagnostics().Report(nttpVoidLocation, diagnostic) << nttpVoidError;
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  std::string sharedWordError;
  clang::SourceLocation sharedWordLocation;
  const auto trustedSharedQualifier = [&](clang::SourceLocation location) {
    // Same SDK/compiler macro-chain proof as the existing device qualifier
    // guard. The outer physical token alone cannot authenticate a redefinition.
    std::set<const clang::IdentifierInfo *> seen;
    std::vector<const clang::IdentifierInfo *> pending = {PP.getIdentifierInfo("__shared__")};
    while (!pending.empty()) {
      const auto *identifier = pending.back(); pending.pop_back();
      if (!seen.insert(identifier).second) continue;
      if (seen.size() > 16) return false;
      const auto *info = PP.getMacroDefinitionAtLoc(identifier, location).getMacroInfo();
      if (info == nullptr) return false;
      const auto file = SM.getFileID(SM.getSpellingLoc(info->getDefinitionLoc()));
      if (file.isInvalid() || (!cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
                              !isOwnedCudaCompilerInput(file, getCompilerInstance()))) return false;
      for (const auto &token : info->tokens()) {
        if (token.is(clang::tok::hash) || token.is(clang::tok::hashhash)) return false;
        const auto *dependency = token.getIdentifierInfo();
        if (dependency == nullptr) continue;
        bool parameter = false;
        if (info->isFunctionLike())
          for (const auto *argument : info->params()) parameter |= argument == dependency;
        if (!parameter && PP.getMacroDefinitionAtLoc(dependency, location).getMacroInfo())
          pending.push_back(dependency);
      }
    }
    return true;
  };
  if (!ascify::ProveSharedWordAlignment(getCompilerInstance().getASTContext(),
          [&](clang::SourceLocation location) { return ownedNttpVoidCallLocation(location); },
          trustedSharedQualifier,
          [&](llvm::StringRef name, clang::SourceLocation location) {
            return !PP.getMacroDefinitionAtLoc(PP.getIdentifierInfo(name), location);
          }, TargetPolicy == "dav-c310-vec" &&
              !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks,
          sharedWordAlignmentPlan, sharedWordLocation, sharedWordError)) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "shared word alignment boundary: %0");
    getCompilerInstance().getDiagnostics().Report(sharedWordLocation, diagnostic) << sharedWordError;
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  if (sharedWordAlignmentPlan.used()) {
    if (cudaAttributeControlModified || deviceTemplateAttributeMacroConflict) {
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "shared word alignment boundary: CUDA qualifier provenance was modified");
      getCompilerInstance().getDiagnostics().Report(diagnostic);
      cudaRuntimeApiBoundaryFailed = true;
      return;
    }
    if (!sharedWordAlignmentPlan.contextualHeaders.empty() &&
        (localHeaderContext == nullptr || localHeaderContext->plan == nullptr)) {
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "shared word alignment boundary: selected header needs the joint source transaction");
      getCompilerInstance().getDiagnostics().Report(diagnostic);
      cudaRuntimeApiBoundaryFailed = true;
      return;
    }
    for (const auto &path : sharedWordAlignmentPlan.contextualHeaders)
      localHeaderContext->plan->requireInheritedSourceContext(path);
    needsCudaCompatHeader = true;
    llvm::errs() << "Ascify shared word alignment v1: arrays=" << sharedWordAlignmentPlan.arrays
                 << ", closed_helpers=" << sharedWordAlignmentPlan.helpers
                 << ", minimum_object_alignment=4\n";
  }
  if (nttpVoidCallPlan.used() && (cudaAttributeControlModified ||
      deviceTemplateAttributeMacroConflict || frontendCgControlModified)) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "NTTP void call boundary: device/profile control provenance was modified");
    getCompilerInstance().getDiagnostics().Report(diagnostic);
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  std::string hostPointerSwapError;
  clang::SourceLocation hostPointerSwapLocation;
  // Runtime and literal-texture entry points share this exact proof. Do not
  // promote cuda_runtime.h to the broader runtime-API header admission.
  const auto proveHostPointerSwaps = [&]() {
    if (!ascify::ProveHostPointerSwaps(
          getCompilerInstance().getASTContext(),
          [&](clang::SourceLocation location) { return ownedNttpVoidCallLocation(location); },
          [&](const clang::CallExpr* call) {
            const auto* function = call == nullptr ? nullptr : call->getDirectCallee();
            if (!function || function->getQualifiedNameAsString() != "cudaMalloc" ||
                function->isVariadic() || function->getNumParams() != 2 || call->getNumArgs() != 2 ||
                !isAdmittedCudaRuntimeStatusCall(SM, call, trustedSystemFileIds)) return false;
            const auto sdkLocation = [&](clang::SourceLocation location) {
              location = SM.getSpellingLoc(location);
              return location.isFileID() && cudaRuntimeApiSdkFileIds.count(SM.getFileID(location).getHashValue());
            };
            // Initially-system-marked caller headers cannot become allocators.
            for (const auto* declaration : function->redecls())
              if (!sdkLocation(declaration->getLocation())) return false;
            auto& context = getCompilerInstance().getASTContext();
            const auto* status = function->getReturnType()->getAs<clang::EnumType>();
            if (!status || status->getDecl()->getName() != "cudaError" ||
                !status->getDecl()->getDeclContext()->isTranslationUnit() ||
                !context.hasSameType(function->getParamDecl(1)->getType(), context.getSizeType())) return false;
            for (const auto* declaration : status->getDecl()->redecls())
              if (!sdkLocation(declaration->getLocation())) return false;
            const auto first = function->getParamDecl(0)->getType().getCanonicalType();
            const auto* outer = first->getAs<clang::PointerType>();
            const auto* inner = outer == nullptr ? nullptr : outer->getPointeeType()->getAs<clang::PointerType>();
            if (!inner || first.hasLocalQualifiers() || outer->getPointeeType().hasLocalQualifiers() ||
                inner->getPointeeType().hasLocalQualifiers()) return false;
            const auto leaf = inner->getPointeeType();
            if (leaf->isVoidType()) return true; // cuda_runtime_api.h C signature
            // cuda_runtime.h's authenticated typed wrapper is T**,size_t.
            const auto* primary = function->getPrimaryTemplate();
            const auto* arguments = function->getTemplateSpecializationArgs();
            if (!primary || primary->getTemplateParameters()->size() != 1 ||
                !llvm::isa<clang::TemplateTypeParmDecl>(primary->getTemplateParameters()->getParam(0)) ||
                !arguments || arguments->size() != 1 || arguments->get(0).getKind() != clang::TemplateArgument::Type ||
                !leaf->isBuiltinType() || !leaf->isArithmeticType() ||
                !context.hasSameType(arguments->get(0).getAsType(), leaf)) return false;
            for (const auto* declaration : primary->redecls())
              if (!sdkLocation(declaration->getLocation())) return false;
            return true;
          },
          [&](llvm::StringRef name, clang::SourceLocation location) {
            return !PP.getMacroDefinitionAtLoc(PP.getIdentifierInfo(name), location);
          }, !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks &&
              TargetPolicy == "dav-c310-vec" && TargetRecipe == "none",
          hostPointerSwapPlan, hostPointerSwapLocation, hostPointerSwapError)) {
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "host pointer-swap boundary: %0");
      getCompilerInstance().getDiagnostics().Report(hostPointerSwapLocation, diagnostic) << hostPointerSwapError;
      cudaRuntimeApiBoundaryFailed = true;
      return false;
    }
    if (!hostPointerSwapPlan.contextualHeaders.empty()) {
      if (localHeaderContext == nullptr || localHeaderContext->plan == nullptr) {
        const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error, "host pointer-swap boundary: selected header requires its joint source transaction");
        getCompilerInstance().getDiagnostics().Report(diagnostic);
        cudaRuntimeApiBoundaryFailed = true;
        return false;
      }
      for (const auto& path : hostPointerSwapPlan.contextualHeaders)
        localHeaderContext->plan->requireInheritedSourceContext(path);
    }
    return true;
  };
  if (cudaRuntimeApiHeaderAdmitted && !proveHostPointerSwaps()) return;
  if (cudaRuntimeApiHeaderAdmitted && !auditCudaRuntimeApiRecords(false)) {
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  if (cudaProfilerApiHeaderAdmitted && !auditCudaProfilerApiSurface()) {
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  std::map<unsigned, std::string> scalar64CommittedInsertions;
  if (scalar64Plan.used()) {
    ct::Replacements staged(*replacements);
    std::map<unsigned, ascify::Scalar64Edit> insertions;
    std::vector<ascify::Scalar64Edit> edits;
    for (const auto& edit : scalar64Plan.edits) {
      if (!edit.location.isFileID() || !SM.isWrittenInMainFile(edit.location)) {
        scalar64Error = "scalar64 edit has no literal main-file owner"; break;
      }
      const unsigned offset = SM.getFileOffset(edit.location);
      for (const auto& range : SemanticRewriteRanges)
        if (range.file == SM.getMainFileID() && range.beginOffset <= offset &&
            offset < range.endOffset) scalar64Error = "scalar64 edit overlaps another semantic transaction";
      if (!scalar64Error.empty()) break;
      if (edit.length == 0) {
        auto insertion = insertions.find(offset);
        if (insertion == insertions.end()) insertions.emplace(offset, edit);
        else insertion->second.text += edit.text;
      } else edits.push_back(edit);
    }
    for (const auto& insertion : insertions) edits.push_back(insertion.second);
    for (const auto& edit : edits)
      if (!llcompat::insertReplacement(staged, ct::Replacement(SM, edit.location, edit.length, edit.text), &scalar64Error)) break;
    if (!scalar64Error.empty()) {
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "scalar64 boundary: %0");
      getCompilerInstance().getDiagnostics().Report(diagnostic) << scalar64Error;
      cudaRuntimeApiBoundaryFailed = true;
      return;
    }
    *replacements = std::move(staged);
    for (const auto& insertion : insertions)
      scalar64CommittedInsertions.emplace(insertion.first, insertion.second.text);
    for (const auto& edit : edits) if (edit.length)
      SemanticRewriteRanges.push_back({SM.getMainFileID(), SM.getFileOffset(edit.location),
          SM.getFileOffset(edit.location) + edit.length});
    needsCudaCompatHeader = true;
    auditActiveAscifyCudaCompatMacrosAtInsertion();
    llvm::errs() << "Ascify scalar64 v1: additions=" << scalar64Plan.additions
                 << ", assignments=" << scalar64Plan.assignments
                 << ", transports=" << scalar64Plan.transports
                 << ", closed_template_guards=" << scalar64Plan.guards
                 << ", exact_cg_reductions=" << scalar64Plan.reductions
                 << ", masked_accumulations=" << scalar64Plan.maskedAccumulations << "\n";
  }
  std::string vec3AbiError;
  clang::SourceLocation vec3AbiLocation;
  ascify::CudaVec3LayoutPlan vec3Layout;
  const bool vec3RootInput = localHeaderContext == nullptr || localHeaderContext->depth == 0;
  if (vec3RootInput) Statistics::current().cudaVec3LayoutEvidence.clear();
  if (TargetPolicy == "dav-c310-vec" && TargetRecipe == "none" &&
      !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks) {
    const auto selectedVec3Header = [&](clang::SourceLocation use) {
      return use.isFileID() && localHeaderContext != nullptr &&
          localHeaderContext->depth == 0 && localHeaderContext->plan != nullptr &&
          localHeaderContext->plan->isSelectedRootHeader(SM.getFilename(use).str());
    };
    const auto literalVec3Use = [&](clang::SourceLocation use) {
      if (!use.isFileID() || !SM.isWrittenInMainFile(use)) return false;
      const auto file = SM.getFileID(use);
      const auto offset = SM.getFileOffset(use);
      for (const auto& range : cudaLaneIdMacroRanges)
        if (range.file == file && range.beginOffset <= offset && offset < range.endOffset)
          return false;
      return true;
    };
    if (!ascify::ProveCudaVec3LocalLayout(getCompilerInstance().getASTContext(),
            CudaPath.getValue(), ownedClangResourceDirectory(), mappedCubIncludes, selectedVec3Header,
            literalVec3Use, vec3Layout, vec3AbiError, vec3AbiLocation)) {
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "CUDA vec3 ABI boundary: %0");
      getCompilerInstance().getDiagnostics().Report(vec3AbiLocation, diagnostic) << vec3AbiError;
      return;
    }
  }
  if (!ascify::ValidateCudaVec3AbiBoundary(
          getCompilerInstance().getASTContext(), CudaPath.getValue(),
          mappedCubIncludes, vec3AbiError, vec3AbiLocation, &vec3Layout)) {
    const auto diagnostic =
        getCompilerInstance().getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error, "CUDA vec3 ABI boundary: %0");
    getCompilerInstance().getDiagnostics().Report(vec3AbiLocation, diagnostic)
        << vec3AbiError;
    return;
  }
  if (!vec3Layout.uses.empty()) {
    ct::Replacements staged(*replacements);
    std::vector<SemanticRewriteRange> ranges;
    for (const auto& edit : vec3Layout.edits) {
      const auto position = SM.getDecomposedLoc(edit.location);
      for (const auto& range : SemanticRewriteRanges)
        if (range.file == position.first && position.second < range.endOffset &&
            range.beginOffset < position.second + edit.length)
          vec3AbiError = "vec3 edit overlaps an existing semantic transaction";
      if (!vec3AbiError.empty() || !llcompat::insertReplacement(staged,
              ct::Replacement(SM, edit.location, edit.length, edit.replacement), &vec3AbiError)) {
        vec3AbiLocation = edit.location;
        break;
      }
      ranges.push_back({position.first, position.second, position.second + edit.length});
    }
    if (!vec3AbiError.empty()) {
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "CUDA vec3 ABI boundary: %0");
      getCompilerInstance().getDiagnostics().Report(vec3AbiLocation, diagnostic) << vec3AbiError;
      return;
    }
    *replacements = std::move(staged);
    SemanticRewriteRanges.insert(SemanticRewriteRanges.end(), ranges.begin(), ranges.end());
    for (const auto& path : vec3Layout.contextualHeaders)
      localHeaderContext->plan->requireInheritedSourceContext(path);
    needsCudaCompatHeader = true;
    needsCudaVec3LayoutHeader = true;
    auditActiveAscifyCudaCompatMacrosAtInsertion();
    auto& evidence = Statistics::current().cudaVec3LayoutEvidence;
    evidence.insert(evidence.end(), vec3Layout.evidence.begin(), vec3Layout.evidence.end());
    llvm::errs() << "Ascify CUDA vec3 local layout v1: edits=" << vec3Layout.edits.size()
                 << ", uses=" << vec3Layout.uses.size()
                 << ", contextual_headers=" << vec3Layout.contextualHeaders.size() << "\n";
  }
  ascify::UniformBlockReductionStats uniformReduction;
  std::string uniformReductionError;
  const auto uniformInputWitness = [&](ascify::UniformInputWitness kind,
                                       clang::SourceLocation location,
                                       const std::string& name) {
    if (!location.isFileID() || !SM.isWrittenInMainFile(location)) return false;
    auto* identifier=PP.getIdentifierInfo(name);
    const auto definition=PP.getMacroDefinitionAtLoc(identifier,location);
    if (kind==ascify::UniformInputWitness::LiteralIdentifier) return !definition;
    if (kind==ascify::UniformInputWitness::SdkQualifier) {
      const auto* macro=definition.getMacroInfo();
      return (name=="__device__" || name=="__forceinline__") && macro &&
          !macro->isFunctionLike() && !definition.isAmbiguous() &&
          definition.getModuleMacros().empty() && !cudaAttributeControlModified &&
          !deviceTemplateAttributeMacroConflict && scalar64Sdk(macro->getDefinitionLoc());
    }
    if (kind==ascify::UniformInputWitness::EmptyOuterGuard) {
      if (definition || name!=findConventionalMainOuterGuard(getCompilerInstance()) ||
          !hasCompleteMainOuterGuard(getCompilerInstance())) return false;
      for (const auto& command : getCompilerInstance().getPreprocessorOpts().Macros)
        if (llvm::StringRef(command.first).split('=').first==name) return false;
      const auto* history=PP.getLocalMacroDirectiveHistory(identifier);
      if (!history || history->getPrevious() || history->getKind()!=clang::MacroDirective::MD_Define ||
          !history->getLocation().isFileID() || !SM.isWrittenInMainFile(history->getLocation())) return false;
      const auto* macro=history->getMacroInfo();
      return macro && !macro->isFunctionLike() && macro->getNumTokens()==0 &&
          macro->getDefinitionLoc().isFileID() && SM.isWrittenInMainFile(macro->getDefinitionLoc());
    }
    if (kind==ascify::UniformInputWitness::NativeStdioInclude) {
      const auto found=uniformNativeStdioIncludes.find(SM.getFileOffset(location));
      if (name!="stdio.h" || found==uniformNativeStdioIncludes.end() ||
          contextualCanonicalPath(found->second)!="/usr/include/stdio.h") return false;
      for (const auto file : enteredSourceFileIds) {
        const auto at=SM.getLocForStartOfFile(file);
        const auto* entry=SM.getFileEntryForID(file);
        if (entry && trustedSystemFileIds.count(file.getHashValue()) &&
            SM.isInSystemHeader(at) && contextualCanonicalPath(SM.getFilename(at))=="/usr/include/stdio.h" &&
            !SM.isFileOverridden(entry) && physicalSourceFileMatches(SM.getFilename(at),*entry,SM)) return true;
      }
      return false;
    }
    if (kind!=ascify::UniformInputWitness::InactiveArchGuard || name!="__CUDA_ARCH__" ||
        definition || PP.getMacroDefinition(identifier) ||
        getCompilerInstance().getLangOpts().CUDAIsDevice) return false;
    for (const auto& command : getCompilerInstance().getPreprocessorOpts().Macros)
      if (llvm::StringRef(command.first).split('=').first==name) return false;
    for (const auto* history=PP.getLocalMacroDirectiveHistory(identifier);history;history=history->getPrevious()) {
      const auto at=history->getLocation();
      if (!at.isFileID() || !isOwnedCudaCompilerInput(SM.getFileID(at),getCompilerInstance())) return false;
    }
    return true;
  };
  if (!ascify::ValidateUniformBlockReduction(
          getCompilerInstance().getASTContext(), frontendCompatibility,
          uniformReduction, uniformReductionError,
          !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks,
          scalar64Sdk,uniformInputWitness)) {
    const auto diagnostic =
        getCompilerInstance().getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error,
            "uniform block reduction boundary: %0");
    getCompilerInstance().getDiagnostics().Report(diagnostic)
        << uniformReductionError;
    return;
  }
  if (uniformReduction.instantiations != 0) {
    ct::Replacements staged(*replacements);
    std::vector<SemanticRewriteRange> guardRanges;
    for (const auto &guard : uniformReduction.guards) {
      const auto location = guard.insertionLocation;
      if (location.isInvalid() || location.isMacroID() ||
          !SM.isWrittenInMainFile(location)) {
        uniformReductionError = "template guard has no literal main-file insertion point";
        break;
      }
      const auto decomposed = SM.getDecomposedLoc(location);
      if (guard.directive) {
        clang::Token anchor;
        const auto buffer=SM.getBufferData(decomposed.first);
        const auto lineBegin=physicalLineBegin(buffer,decomposed.second);
        if (clang::Lexer::getRawToken(location,anchor,SM,
                getCompilerInstance().getLangOpts(),true) ||
            !anchor.is(clang::tok::hash) ||
            !buffer.slice(lineBegin,decomposed.second).trim().empty()) {
          uniformReductionError="preprocessor guard has no physical directive anchor";
          break;
        }
      }
      if (decomposed.second == 0 && !guard.directive) {
        uniformReductionError = "template guard has no body brace anchor";
        break;
      }
      for (const auto &range : SemanticRewriteRanges) {
        if (range.file == decomposed.first &&
            range.beginOffset <= decomposed.second &&
            decomposed.second < range.endOffset) {
          uniformReductionError = "template guard overlaps an existing semantic rewrite";
          break;
        }
      }
      if (!uniformReductionError.empty())
        break;
      clang::Lexer guardLexer(clang::SourceLocation(),
          getCompilerInstance().getLangOpts(), guard.text.data(),
          guard.text.data(), guard.text.data() + guard.text.size());
      clang::Token token;
      while (!guardLexer.LexFromRawLexer(token)) {
        if (token.isNot(clang::tok::raw_identifier))
          continue;
        const auto name = token.getRawIdentifier();
        if (PP.getMacroDefinitionAtLoc(PP.getIdentifierInfo(name), location)) {
          uniformReductionError = "input macro changes template guard token '" +
                                  name.str() + "' at its insertion point";
          break;
        }
      }
      if (!uniformReductionError.empty())
        break;
      const ct::Replacement edit(SM, location, 0U, guard.text);
      const auto scalarWitness=scalar64CommittedInsertions.find(decomposed.second);
      const std::string scalarInsertion=scalarWitness==scalar64CommittedInsertions.end()
          ? std::string() : scalarWitness->second;
      std::optional<ct::Replacement> knownScalar;
      unsigned scalarMatches=0;
      if (!guard.directive && !scalarInsertion.empty())
        for (const auto& existing : staged)
          if (existing.getFilePath()==edit.getFilePath() &&
              existing.getOffset()==edit.getOffset() && existing.getLength()==0 &&
              existing.getReplacementText()==scalarInsertion) {
            knownScalar=existing;
            ++scalarMatches;
          }
      if (scalarMatches>1) {
        uniformReductionError="template guard has ambiguous scalar insertion witness";
        break;
      }
      if (knownScalar) {
        // Both proof-owned guards precede the first statement. Prefix the
        // complete scalar insertion, which may also open that statement's
        // arithmetic helper when the source has no whitespace after '{'.
        ct::Replacements combined;
        for (const auto& existing : staged) {
          if (existing==*knownScalar) continue;
          if (!llcompat::insertReplacement(combined,existing,&uniformReductionError)) break;
        }
        if (!uniformReductionError.empty() ||
            !llcompat::insertReplacement(combined,
                ct::Replacement(SM,location,0U,guard.text+scalarInsertion),
                &uniformReductionError)) break;
        staged=std::move(combined);
      } else if (!llcompat::insertReplacement(staged,edit,&uniformReductionError)) break;
      guardRanges.push_back(
          {decomposed.first, guard.directive ? decomposed.second : decomposed.second - 1,
           guard.directive ? decomposed.second + 1 : decomposed.second});
    }
    if (!uniformReductionError.empty()) {
      const auto diagnostic =
          getCompilerInstance().getDiagnostics().getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "uniform block reduction boundary: %0");
      getCompilerInstance().getDiagnostics().Report(diagnostic)
          << uniformReductionError;
      return;
    }
    *replacements = std::move(staged);
    SemanticRewriteRanges.insert(SemanticRewriteRanges.end(),
                                 guardRanges.begin(), guardRanges.end());
    needsCudaCompatHeader = true;
    llvm::errs() << "Ascify uniform block reduction: kernels="
                 << uniformReduction.kernels << ", instantiations="
                 << uniformReduction.instantiations << ", collectives="
                 << uniformReduction.collectives << "\n";
  }
  ascifyCudaCompatDeclarationConflict =
      hasInputTopLevelAscifyDeclaration(
          getCompilerInstance().getASTContext(), SM);
  if (!nvidiaSampleHelperIncludes.empty()) {
    proveNvidiaSampleOpenMPParallelPragmas();
    nvidiaSampleHelperUnsupportedDeclarationUse =
        hasUnsupportedNvidiaSampleHelperDeclarationUse();
    rewriteProvenNvidiaSampleHelperMacros();
  }
  if (!prepareKernelThreadBoundHeaderPolicies()) return;
  std::string kernelBoundError;
  clang::SourceLocation kernelBoundLocation;
  const auto closedBoundInvocation = [&](const clang::CallExpr *call) {
    const auto physical = [&](clang::SourceLocation location) {
      while (location.isMacroID()) {
        if (!SM.isMacroArgExpansion(location)) return clang::SourceLocation();
        location = SM.getImmediateSpellingLoc(location);
      }
      return location;
    };
    const auto first = physical(call->getBeginLoc()), last = physical(call->getEndLoc());
    if (!first.isFileID() || !last.isFileID() || SM.getFileID(first) != SM.getFileID(last))
      return false;
    const auto file = SM.getFileID(first);
    const auto end = clang::Lexer::getLocForEndOfToken(last, 0, SM,
                                                       getCompilerInstance().getLangOpts());
    if (!end.isFileID()) return false;
    const unsigned beginOffset = SM.getFileOffset(first), endOffset = SM.getFileOffset(end);
    for (const auto &range : cudaLaneIdMacroRanges) {
      if (range.file != file || range.endOffset <= beginOffset || range.beginOffset >= endOffset)
        continue;
      bool helper = false;
      for (const auto &candidate : nvidiaSampleHelperMacroCandidates)
        if (candidate.kind == NvidiaSampleHelperMacroKind::CheckCudaErrors &&
            candidate.statusDomainProven && candidate.invocationRange.file == file &&
            candidate.invocationRange.beginOffset == range.beginOffset &&
            candidate.invocationRange.endOffset == range.endOffset &&
            range.beginOffset < beginOffset && endOffset < range.endOffset) helper = true;
      if (!helper) return false;
    }
    return true;
  };
  if (!ascify::ProveOccupancyKernelThreadBounds(getCompilerInstance().getASTContext(),
          [&](clang::SourceLocation location) { return ownedNttpVoidCallLocation(location); },
          [&](llvm::StringRef name, clang::SourceLocation location) {
            return trustedKernelBoundQualifier(name, location);
          }, [&](llvm::StringRef name, clang::SourceLocation location) {
            return !PP.getMacroDefinitionAtLoc(PP.getIdentifierInfo(name), location);
          }, [&](const clang::CallExpr *call) {
            const auto *callee = call->getDirectCallee();
            if (!callee || !isAdmittedCudaRuntimeStatusCall(SM, call, trustedSystemFileIds)) return false;
            for (const auto *decl : callee->redecls()) {
              const auto loc = SM.getSpellingLoc(decl->getLocation());
              if (!loc.isFileID() || !cudaRuntimeApiSdkFileIds.count(SM.getFileID(loc).getHashValue()))
                return false;
            }
            return true;
          }, closedBoundInvocation, [&]() {
            // A static helper can otherwise acquire a hidden assembly caller
            // or raw inquiry from an unreviewed include. Keep this domain to
            // the physical root plus already certified SDK/helper/system files.
            std::set<clang::FileID> inputs = enteredSourceFileIds;
            for (auto info = SM.fileinfo_begin(); info != SM.fileinfo_end(); ++info)
              inputs.insert(SM.translateFile(info->first));
            for (const auto file : inputs) {
              if (file.isInvalid() || file == SM.getMainFileID() ||
                  file == PP.getPredefinesFileID()) continue;
              const auto *entry = SM.getFileEntryForID(file);
              const auto start = SM.getLocForStartOfFile(file);
              const auto path = SM.getFilename(start);
              if (!entry || SM.isFileOverridden(entry) ||
                  !physicalSourceFileMatches(path, *entry, SM)) return false;
              // These three exact parser providers already have an entry-time
              // CUDA-control proof. Recheck their current physical bytes and
              // forced-include/CG role; the enclosing directory grants nothing.
              if ((verifiedFrontendHostMathEntryFileIds.count(file.getHashValue()) &&
                   verifiedForcedFrontendHostMathFile(file, getCompilerInstance(), frontendCompatibility)) ||
                  (verifiedFrontendCgEntryFileIds.count(file.getHashValue()) &&
                   verifiedFrontendCgFile(file, getCompilerInstance(), frontendCompatibility))) continue;
              if (isOwnedCudaCompilerInput(file, getCompilerInstance(), true) ||
                  cudaRuntimeApiSdkFileIds.count(file.getHashValue()) ||
                  frozenNvidiaSampleHelperFileRole(SM, start) != 0 ||
                  (trustedSystemFileIds.count(file.getHashValue()) &&
                   (pathIsWithinPhysicalDirectory(path, "/usr/include") ||
                    pathIsWithinPhysicalDirectory(path, "/usr/lib/gcc")))) continue;
              return false;
            }
            return true;
          }, [&](const ascify::PureSimtKernelThreadBound &fact) {
            return stageKernelThreadBoundFact(fact);
          },
          TargetPolicy == "dav-c310-vec" && TargetRecipe == "none" &&
              !cudaAttributeControlModified && !deviceTemplateAttributeMacroConflict &&
              !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks,
          kernelThreadBoundPlan, kernelBoundLocation, kernelBoundError)) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "occupancy kernel-bound boundary: %0");
    getCompilerInstance().getDiagnostics().Report(kernelBoundLocation, diagnostic) << kernelBoundError;
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  if (kernelThreadBoundPlan.used()) needsCudaCompatHeader = true;
  std::vector<clang::SourceLocation> dynamicAbiTailInsertions;
  std::string dynamicAbiTailError;
  clang::SourceLocation dynamicAbiTailLocation;
  if (!ascify::ProveDynamicKernelAbiTails(getCompilerInstance().getASTContext(),
          dynamicWarpGroups,
          [&](clang::SourceLocation location) {
            return (localHeaderContext==nullptr||localHeaderContext->depth==0)&&
                ownedNttpVoidCallLocation(location);
          },dynamicCaller,
          [&](llvm::StringRef name,clang::SourceLocation location) {
            return !PP.getMacroDefinitionAtLoc(PP.getIdentifierInfo(name),location);
          },[&](const clang::CallExpr* call) {
            // Consume only this CI's successfully completed occupancy proof;
            // neither an SDK-looking name nor an offset from another file can
            // grant a new function-reference permission here.
            const auto* callee=llvm::dyn_cast_or_null<clang::DeclRefExpr>(
                call->getCallee()->IgnoreParenImpCasts());
            if(!callee||call->getNumArgs()<=2||!closedBoundInvocation(call))return false;
            auto location=callee->getLocation();
            while(location.isMacroID()) {
              if(!SM.isMacroArgExpansion(location))return false;
              location=SM.getImmediateSpellingLoc(location);
            }
            if(!location.isFileID()||!SM.isWrittenInMainFile(location))return false;
            const auto found=kernelThreadBoundPlan.occupancyCallees.find(SM.getFileOffset(location));
            if(found==kernelThreadBoundPlan.occupancyCallees.end()||
                !llvm::StringRef(found->second).starts_with(
                    "::ascify::cudaOccupancyMaxPotentialBlockSizeMaxThreads<"))return false;
            const auto* reference=llvm::dyn_cast<clang::DeclRefExpr>(
                call->getArg(2)->IgnoreParenImpCasts());
            const auto* kernel=reference?llvm::dyn_cast<clang::FunctionDecl>(reference->getDecl()):nullptr;
            return kernel&&kernelThreadBoundPlan.kernels.count(kernel->getCanonicalDecl());
          },dynamicAbiTailInsertions,dynamicAbiTailLocation,dynamicAbiTailError)) {
    const auto diagnostic=getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,"dynamic kernel ABI boundary: %0");
    getCompilerInstance().getDiagnostics().Report(diagnostic)<<dynamicAbiTailError;
    cudaRuntimeApiBoundaryFailed=true;return;
  }
  if(!dynamicAbiTailInsertions.empty()) {
    ct::Replacements staged(*replacements);std::vector<SemanticRewriteRange> ranges;
    for(const auto location:dynamicAbiTailInsertions) {
      if(!location.isFileID()||!SM.isWrittenInMainFile(location)||
          !ownedNttpVoidCallLocation(location)) {
        dynamicAbiTailError="padding has no literal root insertion owner";break;
      }
      const auto position=SM.getDecomposedLoc(location);
      for(const auto& range:SemanticRewriteRanges)
        if(range.file==position.first&&range.beginOffset<=position.second&&position.second<range.endOffset)
          dynamicAbiTailError="padding overlaps an existing semantic rewrite";
      const ct::Replacement edit(SM,location,0U,", unsigned int = 0u");
      for(const auto& existing:staged)
        if(existing.getFilePath()==edit.getFilePath()&&existing.getOffset()==edit.getOffset()&&existing.getLength()==0)
          dynamicAbiTailError="padding collides with another insertion";
      if(!dynamicAbiTailError.empty()||!llcompat::insertReplacement(staged,edit,&dynamicAbiTailError))break;
      ranges.push_back({position.first,position.second,position.second+1});
    }
    if(!dynamicAbiTailError.empty()) {
      const auto diagnostic=getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,"dynamic kernel ABI boundary: %0");
      getCompilerInstance().getDiagnostics().Report(diagnostic)<<dynamicAbiTailError;
      cudaRuntimeApiBoundaryFailed=true;return;
    }
    *replacements=std::move(staged);
    SemanticRewriteRanges.insert(SemanticRewriteRanges.end(),ranges.begin(),ranges.end());
    llvm::errs()<<"Ascify dynamic kernel ABI tail v1: kernels="<<dynamicAbiTailInsertions.size()<<"\n";
  }
  // Texture macro admission consumes the existing helper status proof.
  // Keep it after that single proof pass and before the raw token rewrites.
  std::string textureError;
  clang::SourceLocation textureErrorLocation;
  // SDK object-like flags have no DeclRefExpr. Prove their completed macro
  // history here, using the same gate as the AST-bound literal identifiers.
  for (unsigned offset:cudaTextureLiteralOffsets) {
    const auto location=SM.getLocForStartOfFile(SM.getMainFileID()).getLocWithOffset(offset);
    if (!isAdmittedCudaTextureMacroUse(location)) {
      const auto diagnostic=getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,"CUDA texture boundary: unproved macro observation of texture flag");
      getCompilerInstance().getDiagnostics().Report(location,diagnostic);
      cudaRuntimeApiBoundaryFailed=true;
      return;
    }
  }
  if (!ascify::ValidateCudaTextureBoundary(getCompilerInstance().getASTContext(),
          cudaRuntimeApiSdkFileIds,cudaTextureLiteralOffsets,
          [&](clang::SourceLocation location) {return isAdmittedCudaTextureMacroUse(location);},
          textureErrorLocation,textureError)) {
    const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error,"CUDA texture boundary: %0");
    getCompilerInstance().getDiagnostics().Report(textureErrorLocation,diagnostic) << textureError;
    cudaRuntimeApiBoundaryFailed = true;
    return;
  }
  // cuda_runtime.h is also a texture entry point and does not set the
  // separate cuda_runtime_api.h admission flag. Its texture records still
  // require the same ABI/reflection proof, without broadening the old raw-PP
  // Runtime boundary to this other entry point.
  if (!cudaRuntimeApiHeaderAdmitted && !cudaTextureLiteralOffsets.empty()) {
    // Texture validation above established actual literal SDK uses. Rebuild
    // the exact host pointer domains in this AST before the texture audit;
    // dormant primaries still receive their mandatory empty-domain guard.
    if (!proveHostPointerSwaps()) return;
    if (!auditCudaRuntimeApiRecords(true)) {
      cudaRuntimeApiBoundaryFailed = true;
      return;
    }
  }
  if (nttpVoidCallPlan.used()) {
    ct::Replacements staged(*replacements);
    std::vector<SemanticRewriteRange> ranges;
    for (const auto location : nttpVoidCallPlan.qualifications) {
      if (!location.isFileID() || !SM.isWrittenInMainFile(location) ||
          !ownedNttpVoidCallLocation(location)) {
        nttpVoidError = "qualification has no literal main-file owner";
        break;
      }
      const auto position = SM.getDecomposedLoc(location);
      for (const auto& range : SemanticRewriteRanges)
        if (range.file == position.first && range.beginOffset <= position.second &&
            position.second < range.endOffset)
          nttpVoidError = "qualification overlaps another semantic transaction";
      if (!nttpVoidError.empty() || !llcompat::insertReplacement(staged,
              ct::Replacement(SM, location, 0U, "::"), &nttpVoidError)) break;
      ranges.push_back({position.first, position.second, position.second + 1});
    }
    if (!nttpVoidCallPlan.contextualHeaders.empty() &&
        (localHeaderContext == nullptr || localHeaderContext->plan == nullptr))
      nttpVoidError = "qualification in a local header requires its source-context transaction";
    if (!nttpVoidError.empty()) {
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "NTTP void call boundary: %0");
      getCompilerInstance().getDiagnostics().Report(diagnostic) << nttpVoidError;
      cudaRuntimeApiBoundaryFailed = true;
      return;
    }
    *replacements = std::move(staged);
    SemanticRewriteRanges.insert(SemanticRewriteRanges.end(), ranges.begin(), ranges.end());
    for (const auto& path : nttpVoidCallPlan.contextualHeaders)
      localHeaderContext->plan->requireInheritedSourceContext(path);
    llvm::errs() << "Ascify closed NTTP void calls v1: patterns=" << nttpVoidCallPlan.patterns
                 << ", concrete_calls=" << nttpVoidCallPlan.concreteCalls
                 << ", qualifications=" << nttpVoidCallPlan.qualifications.size()
                 << ", contextual_headers=" << nttpVoidCallPlan.contextualHeaders.size() << "\n";
  }
  if (hostPointerSwapPlan.used()) {
    ct::Replacements staged(*replacements);
    std::vector<SemanticRewriteRange> ranges;
    for (const auto& guard : hostPointerSwapPlan.guards) {
      if (!guard.location.isFileID() || !SM.isWrittenInMainFile(guard.location) ||
          !ownedNttpVoidCallLocation(guard.location) || SM.getFileOffset(guard.location) == 0) {
        hostPointerSwapError = "guard has no literal joint-root insertion owner";
        break;
      }
      const auto position = SM.getDecomposedLoc(guard.location);
      const SemanticRewriteRange anchor{position.first, position.second - 1, position.second};
      for (const auto& range : SemanticRewriteRanges)
        if (range.file == anchor.file && range.beginOffset < anchor.endOffset &&
            anchor.beginOffset < range.endOffset)
          hostPointerSwapError = "guard overlaps another semantic transaction";
      if (!hostPointerSwapError.empty() || !llcompat::insertReplacement(staged,
              ct::Replacement(SM, guard.location, 0U, guard.text), &hostPointerSwapError)) break;
      ranges.push_back(anchor);
    }
    if (!hostPointerSwapError.empty()) {
      const auto diagnostic = getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error, "host pointer-swap boundary: %0");
      getCompilerInstance().getDiagnostics().Report(diagnostic) << hostPointerSwapError;
      cudaRuntimeApiBoundaryFailed = true;
      return;
    }
    *replacements = std::move(staged);
    SemanticRewriteRanges.insert(SemanticRewriteRanges.end(), ranges.begin(), ranges.end());
    llvm::errs() << "Ascify closed host pointer swaps v2: patterns=" << hostPointerSwapPlan.patterns
                 << ", calls=" << hostPointerSwapPlan.concreteCalls
                 << ", empty_domains=" << hostPointerSwapPlan.emptyPatterns
                 << ", guards=" << hostPointerSwapPlan.guards.size()
                 << ", contextual_headers=" << hostPointerSwapPlan.contextualHeaders.size() << "\n";
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
  // Rewrite mapped CUDA identifiers to their Ascend spellings. The raw lexer gives us enough
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
  // Attribute-only is intentionally separate from the global __device__ map.
  // Run after all ordinary AST gates and the raw pass; a failed optimization
  // proof cannot change their acceptance decision. Header projections do not
  // acquire this main-file-only transformation.
  const bool rootInput = localHeaderContext == nullptr || localHeaderContext->depth == 0;
  if (rootInput) Statistics::current().boundedDeviceTemplateDefinitions = 0;
  if (rootInput && TargetPolicy == "dav-c310-vec" && SimtMathMode == "fast" &&
      TargetRecipe != ascify::DavC310TargetRecipe::SimdRecipeName &&
      !cudaAttributeControlModified && !deviceTemplateAttributeMacroConflict &&
      !getCompilerInstance().getPreprocessorOpts().RetainExcludedConditionalBlocks) {
    const auto ownedQualifier = [&](llvm::StringRef name, clang::SourceLocation location) {
      std::set<const clang::IdentifierInfo *> seen;
      std::vector<const clang::IdentifierInfo *> pending = {PP.getIdentifierInfo(name)};
      while (!pending.empty()) {
        const auto *identifier = pending.back(); pending.pop_back();
        if (!seen.insert(identifier).second) continue;
        if (seen.size() > 16) return false;
        const auto *info = PP.getMacroDefinitionAtLoc(identifier, location).getMacroInfo();
        if (info == nullptr) return false;
        const auto file = SM.getFileID(SM.getSpellingLoc(info->getDefinitionLoc()));
        if (file.isInvalid() || (!cudaRuntimeApiSdkFileIds.count(file.getHashValue()) &&
                                !isOwnedCudaCompilerInput(file, getCompilerInstance())))
          return false;
        for (const auto &token : info->tokens()) {
          if (token.is(clang::tok::hash) || token.is(clang::tok::hashhash)) return false;
          const auto *dependency = token.getIdentifierInfo();
          if (dependency == nullptr) continue;
          bool parameter = false;
          if (info->isFunctionLike())
            for (const auto *argument : info->params()) parameter |= argument == dependency;
          if (!parameter && PP.getMacroDefinitionAtLoc(dependency, location).getMacroInfo())
            pending.push_back(dependency);
        }
      }
      return true;
    };
    const auto edits = ascify::ProveBoundedDeviceTemplateInlining(
        getCompilerInstance().getASTContext(), cudaTextureLiteralOffsets, ownedQualifier);
    ct::Replacements staged(*replacements);
    bool committed = true;
    for (const auto qualifier : edits.qualifiers) {
      const auto location = clang::Lexer::getLocForEndOfToken(qualifier, 0, SM, PP.getLangOpts());
      if (location.isInvalid()) { committed = false; break; }
      const unsigned offset = SM.getFileOffset(location);
      for (const auto &range : SemanticRewriteRanges)
        if (range.file == SM.getFileID(location) && range.beginOffset <= offset &&
            offset <= range.endOffset) committed = false;
      if (!committed) break;
      std::string error;
      if (!llcompat::insertReplacement(staged, ct::Replacement(SM, location, 0,
              " __attribute__((always_inline))"), &error)) {
        committed = false; break;
      }
    }
    if (committed && !edits.qualifiers.empty()) {
      *replacements = std::move(staged);
      Statistics::current().boundedDeviceTemplateDefinitions = edits.qualifiers.size();
      llvm::errs() << "Ascify bounded device-template inlining v1: families=" << edits.families
                   << ", definitions=" << edits.qualifiers.size()
                   << ", instances=" << edits.instances << ", max_depth=64, max_nodes=256\n";
    }
  }
  if (StandaloneDeviceExecutable) {
    std::string aggregateError;
    auto* context=localHeaderContext;
    ascify::DeviceAggregateInlineAnalysis analysis;
    const auto ordinary = context && context->plan && !context->plan->failed() &&
        !context->virtualInput && OptLocalHeadersRecursive && DefaultPreprocessor &&
        TargetPolicy=="dav-c310-vec" && TargetRecipe=="none" &&
        !cudaAttributeControlModified && !deviceAggregateTokenMacroConflict;
    const auto sdkInput = [&](const clang::Decl* declaration) {
      if (!declaration || declaration->getLocation().isInvalid()) return false;
      const auto location=SM.getSpellingLoc(declaration->getLocation());
      const auto file=SM.getFileID(location);const auto* entry=SM.getFileEntryForID(file);
      return entry && (cudaRuntimeApiSdkFileIds.count(file.getHashValue()) ||
          isOwnedCudaCompilerInput(file,getCompilerInstance()) ||
          (verifiedFrontendCgEntryFileIds.count(file.getHashValue()) &&
           verifiedFrontendCgFile(file,getCompilerInstance(),frontendCompatibility))) &&
          physicalSourceFileMatches(SM.getFilename(location),*entry,SM);
    };
    const auto nativeInputUncached = [&](clang::SourceLocation location) {
      if (!location.isFileID()) return false;
      const auto file=SM.getFileID(location);const auto* entry=SM.getFileEntryForID(file);
      if (!entry || !physicalSourceFileMatches(SM.getFilename(location),*entry,SM)) return false;
      if (cudaRuntimeApiSdkFileIds.count(file.getHashValue()) ||
          isOwnedCudaCompilerInput(file,getCompilerInstance(),true) ||
          fullWarpNativeCompilerInput(file,getCompilerInstance(),trustedSystemFileIds) ||
          (verifiedFrontendCgEntryFileIds.count(file.getHashValue()) &&
           verifiedFrontendCgFile(file,getCompilerInstance(),frontendCompatibility)) ||
          (verifiedFrontendHostMathEntryFileIds.count(file.getHashValue()) &&
           verifiedForcedFrontendHostMathFile(file,getCompilerInstance(),frontendCompatibility))) return true;
      if (frozenNvidiaSampleHelperFileRole(SM,location)!=FrozenNvidiaSampleHelperNone) return true;
      return trustedSystemFileIds.count(file.getHashValue()) && SM.isInSystemHeader(location) &&
          (pathIsWithinPhysicalDirectory(SM.getFilename(location),"/usr/include") ||
           pathIsWithinPhysicalDirectory(SM.getFilename(location),"/usr/lib/gcc"));
    };
    // Local to this completed-preprocessing CI. No SDK, AST, or caller
    // ownership is cached or enlarged. System-header state is location-sensitive.
    struct NativeInputSnapshot {
      clang::FileID file;
      const clang::FileEntry* entry=nullptr;
      std::string requested,canonical,uniqueId,parsedSha256;
      std::size_t size=0;
    };
    struct NativeInputMemoValue { clang::SourceLocation location; bool native=false; };
    std::map<unsigned,NativeInputSnapshot> nativeInputSnapshots;
    std::map<std::pair<unsigned,bool>,NativeInputMemoValue> nativeInputMemo;
    bool nativeInputCaptureFailed=false;
    const auto snapshotNativeInput = [&](clang::FileID file,NativeInputSnapshot& value) {
      value.file=file;value.entry=SM.getFileEntryForID(file);
      // A nonphysical FileID still has a stable identity in this CI, but it
      // never gains a physical/native role from this cache.
      if (!value.entry) return true;
      const auto begin=SM.getLocForStartOfFile(file);
      value.requested=SM.getFilename(begin).str();
      value.canonical=contextualCanonicalPath(value.requested);
      llvm::sys::fs::UniqueID first,last;
      if (value.canonical.empty() || SM.isFileOverridden(value.entry) ||
          llvm::sys::fs::getUniqueID(value.requested,first) || first!=value.entry->getUniqueID()) return false;
      bool invalid=false;const auto parsed=SM.getBufferData(file,&invalid);
      const auto physical=llvm::MemoryBuffer::getFile(value.requested);
      if (invalid || !physical || parsed!=(*physical)->getBuffer() ||
          llvm::sys::fs::getUniqueID(value.requested,last) || first!=last ||
          contextualCanonicalPath(value.requested)!=value.canonical) return false;
      value.uniqueId=std::to_string(first.getDevice())+":"+std::to_string(first.getFile());
      value.parsedSha256=contextualInputSha256(parsed);value.size=parsed.size();
      return true;
    };
    const auto captureNativeInputAncestry = [&](clang::FileID first) {
      std::vector<clang::FileID> pending{first};std::set<unsigned> visited;
      while (!pending.empty()) {
        const auto file=pending.back();pending.pop_back();
        if (file.isInvalid() || !visited.insert(file.getHashValue()).second) continue;
        if (!nativeInputSnapshots.count(file.getHashValue())) {
          NativeInputSnapshot value;
          if (!snapshotNativeInput(file,value)) return false;
          nativeInputSnapshots.emplace(file.getHashValue(),std::move(value));
        }
        const auto include=SM.getIncludeLoc(file);
        if (include.isInvalid()) continue;
        // Preserve both physical ancestry paths if an include location is a
        // macro. The uncached classifier itself retains its original rules.
        for (const auto location:{SM.getSpellingLoc(include),SM.getExpansionLoc(include)})
          if (location.isFileID()) pending.push_back(SM.getFileID(location));
      }
      return true;
    };
    const auto nativeInput = [&](clang::SourceLocation location) {
      if (location.isInvalid() || !location.isFileID()) return false;
      const auto file=SM.getFileID(location);
      if (file.isInvalid()) return false;
      const auto key=std::make_pair(file.getHashValue(),SM.isInSystemHeader(location));
      const auto found=nativeInputMemo.find(key);
      if (found!=nativeInputMemo.end()) return found->second.native;
      if (!captureNativeInputAncestry(file)) { nativeInputCaptureFailed=true;return false; }
      const bool native=nativeInputUncached(location);
      nativeInputMemo.emplace(key,NativeInputMemoValue{location,native});
      return native;
    };
    const auto revalidateNativeInputMemo = [&]() {
      if (nativeInputCaptureFailed) return false;
      for (const auto& item:nativeInputSnapshots) {
        const auto& expected=item.second;NativeInputSnapshot actual;
        if (!snapshotNativeInput(expected.file,actual) || actual.entry!=expected.entry ||
            actual.requested!=expected.requested || actual.canonical!=expected.canonical ||
            actual.uniqueId!=expected.uniqueId || actual.size!=expected.size ||
            actual.parsedSha256!=expected.parsedSha256) return false;
      }
      // Once per distinct key, also replay the unchanged classifier. This
      // covers its resource/frozen-parent predicates without trusting a later
      // Bind or a generic transaction to repair this cache's own scope.
      for (const auto& item:nativeInputMemo) {
        const auto& value=item.second;
        if (!value.location.isFileID() || SM.getFileID(value.location).getHashValue()!=item.first.first ||
            SM.isInSystemHeader(value.location)!=item.first.second ||
            nativeInputUncached(value.location)!=value.native) return false;
      }
      return true;
    };
    const auto helperDiagnostic = [&](const ascify::DeviceNativeMacroUse& observed) {
      // No range-wide exception: the builtin must come directly from a
      // frozen helper replacement-list token, never from its value argument.
      if ((observed.name!="__FILE__" && observed.name!="__LINE__") ||
          !observed.unambiguous || !observed.definition ||
          !observed.definition->isBuiltinMacro() || !observed.nameLocation.isMacroID() ||
          SM.isMacroArgExpansion(observed.nameLocation)) return false;
      const auto slot=SM.getImmediateSpellingLoc(observed.nameLocation);
      if (!slot.isFileID() || slot!=SM.getSpellingLoc(observed.nameLocation) || !nativeInput(slot)) return false;
      const auto immediate=SM.getImmediateExpansionRange(observed.nameLocation);
      if (!immediate.isTokenRange() || !immediate.getBegin().isFileID() || !immediate.getEnd().isFileID()) return false;
      const auto after=clang::Lexer::getLocForEndOfToken(immediate.getEnd(),0,SM,getCompilerInstance().getLangOpts());
      if (!after.isFileID() || SM.getFileID(immediate.getBegin())!=SM.getFileID(after)) return false;
      const auto file=SM.getFileID(immediate.getBegin());
      const unsigned begin=SM.getFileOffset(immediate.getBegin()),end=SM.getFileOffset(after);
      const NvidiaSampleHelperMacroCandidate* candidate=nullptr;
      for (const auto& value:nvidiaSampleHelperMacroCandidates)
        if (value.invocationRange.file==file && value.invocationRange.beginOffset==begin && value.invocationRange.endOffset==end) {
          if (candidate || !value.statusDomainProven || !value.nameLocation.isFileID() ||
              !SM.isWrittenInMainFile(value.nameLocation)) return false;
          candidate=&value;
        }
      if (!candidate) return false;
      const llvm::StringRef expected=candidate->kind==NvidiaSampleHelperMacroKind::CheckCudaErrors ? "checkCudaErrors" : "getLastCudaError";
      const ascify::DeviceNativeMacroUse* helper=nullptr;
      for (const auto& use:deviceAggregateMacroEvidence.uses) {
        if (use.name!=expected || use.nameLocation!=candidate->nameLocation) continue;
        if (helper || !use.unambiguous || !use.definition ||
            frozenNvidiaSampleHelperFileRole(SM,use.definition->getDefinitionLoc())!=FrozenOfficialNvidiaSampleHelperCuda ||
            !activeNvidiaSampleHelperMacroBodyMatches(expected,*use.definition,PP)) return false;
        helper=&use;
      }
      if (!helper) return false;
      const auto tokens=helper->definition->tokens();
      const unsigned index=candidate->kind==NvidiaSampleHelperMacroKind::CheckCudaErrors ?
          (observed.name=="__FILE__"?9U:11U) : (observed.name=="__FILE__"?4U:6U);
      if (index>=tokens.size() || tokens[index].getLocation()!=slot ||
          PP.getSpelling(tokens[index])!=observed.name ||
          helper->definition->getParameterNum(tokens[index].getIdentifierInfo())>=0) return false;
      // The same actual PP callback must name this direct literal invocation.
      if (helper->invocation.getBegin()!=immediate.getBegin() || helper->invocation.getEnd()!=immediate.getEnd()) return false;
      deviceAggregateRequiredHelperOffsets.insert(begin);
      return true;
    };
    const auto completeGuard = [&](clang::FileID file) {
      if (!hasCompleteOuterGuardInFile(getCompilerInstance(),file)) return false;
      const auto guard=findConventionalOuterGuardInFile(getCompilerInstance(),file);
      if (guard.empty() || profilerMacroDefinitions[guard]!=1 || profilerMacroUndefinitions[guard]!=0) return false;
      const auto* info=PP.getMacroInfo(PP.getIdentifierInfo(guard));
      return info && !info->isFunctionLike() && info->getNumTokens()==0 &&
          info->getDefinitionLoc().isFileID() && SM.getFileID(info->getDefinitionLoc())==file;
    };
    if (ordinary && context->depth==0) for (const auto& path:deviceAggregateResolvedIncludes) {
      auto file=getCompilerInstance().getFileManager().getFileRef(path);
      if (!file) { llvm::consumeError(file.takeError()); aggregateError="resolved caller input vanished";break; }
      if (SM.translateFile(&file->getFileEntry()).isValid()) continue;
      const bool trustedUnparsed=isConfiguredCudaSdkDependencyRole(path) ||
          pathIsWithinPhysicalDirectory(path,ownedClangResourceDirectory()) ||
          pathIsWithinPhysicalDirectory(path,"/usr/include") || pathIsWithinPhysicalDirectory(path,"/usr/lib/gcc");
      if (!trustedUnparsed) {aggregateError="caller include has no entered original body/incoming-guard proof: "+path;break;}
    }
    if (!ordinary) aggregateError="requires an ordinary recursive physical project and unchanged inline/layout macros";
    else if (!aggregateError.empty()) {}
    else if (!ascify::AnalyzeDeviceAggregateInlining(getCompilerInstance(),
        context->plan->rootSourcePath(),context->sourcePath,
        [&](clang::SourceLocation location){return ownedPreservedMaxLocation(location);},
        sdkInput,nativeInput,deviceAggregateMacroEvidence,helperDiagnostic,completeGuard,
        context->depth==0?nullptr:&context->plan->deviceInlineFacts(),
        context->depth==0?nullptr:&context->plan->deviceRegisterFactories(),analysis,aggregateError)) {}
    else {
      if (context->depth==0) {
        std::vector<ascify::StandaloneDeviceIncludeEdge> edges;
        for (const auto& item:deviceAggregateIncludeEdges) {
          auto selected=getCompilerInstance().getFileManager().getFileRef(item.resolved);
          if (!selected) {llvm::consumeError(selected.takeError());aggregateError="original include edge provider vanished";break;}
          const auto child=SM.translateFile(&selected->getFileEntry());
          if (child.isValid() && nativeInput(SM.getLocForStartOfFile(child))) continue;
          if (!item.literal || !item.hash.isFileID() || child.isInvalid()) {aggregateError="retained caller include has no literal physical edge";break;}
          const auto parent=SM.isWrittenInMainFile(item.hash)?context->sourcePath:SM.getFilename(item.hash).str();
          edges.push_back({parent,item.resolved,item.spelling,item.search,SM.getFileOffset(item.hash),item.angled});
        }
        if (aggregateError.empty() && !ascify::RecordStandaloneCallerEdges(edges,aggregateError)) {}
      }
      // Analyze and root-edge classification have now finished using the
      // cache. Revalidate before any plan fact or source edit is committed.
      if (aggregateError.empty() && !revalidateNativeInputMemo())
        aggregateError="native-data input/ancestry identity changed during owned analysis";
      if (aggregateError.empty() && context->depth==0 && !context->plan->recordDeviceInlineFacts(analysis))
        aggregateError=context->plan->failureReason();
      if (context->depth!=0 && context->plan->deviceInlineProjectStamp()!=analysis.projectStamp)
        aggregateError="project inputs changed between root and selected header";
      ct::Replacements staged(*replacements);
      std::set<unsigned> guards;std::vector<ascify::DeviceAggregateInlineFact> applied;
      for (const auto& fact:analysis.facts) {
        if (fact.sourcePath!=context->sourcePath) continue;
        const auto& expected=context->plan->deviceInlineFacts();
        if (std::find(expected.begin(),expected.end(),fact)==expected.end()) {
          aggregateError="selected helper differs from complete original root AST proof";break;
        }
        const auto start=SM.getLocForStartOfFile(SM.getMainFileID());
        const auto anchor=start.getLocWithOffset(fact.insertOffset);
        for (const auto& range:SemanticRewriteRanges)
          if (range.file==SM.getMainFileID() && range.beginOffset<=fact.insertOffset &&
              fact.insertOffset<range.endOffset) aggregateError="inline annotation overlaps a semantic transaction";
        if (!aggregateError.empty() || !llcompat::insertReplacement(staged,
            ct::Replacement(SM,anchor,0,"inline __attribute__((always_inline)) "),&aggregateError)) break;
        if (guards.insert(fact.guardOffset).second) {
          const std::string guard=" static_assert(sizeof(::"+fact.recordName+")=="+
              std::to_string(fact.recordBytes)+" && alignof(::"+fact.recordName+")=="+
              std::to_string(fact.recordAlign)+", \"Ascify aggregate layout contract\"); ";
          if (!llcompat::insertReplacement(staged,ct::Replacement(SM,
              start.getLocWithOffset(fact.guardOffset),0,guard),&aggregateError)) break;
        }
        applied.push_back(fact);
      }
      std::vector<ascify::DeviceRegisterTile32Fact> registerApplied;
      for (const auto& fact:analysis.registerFactories) {
        if (!aggregateError.empty()) break;
        if (fact.sourcePath!=context->sourcePath) continue;
        const auto& expected=context->plan->deviceRegisterFactories();
        if (std::find(expected.begin(),expected.end(),fact)==expected.end()) {
          aggregateError="selected register factory lacks complete original root proof";break;
        }
        const auto start=SM.getLocForStartOfFile(SM.getMainFileID());
        const ct::Replacement factory(SM,start.getLocWithOffset(fact.beginOffset),
            fact.endOffset-fact.beginOffset,"::ascify_cg::register_partition32_v1");
        for (const auto& range:SemanticRewriteRanges)
          if (range.file==SM.getMainFileID() && range.beginOffset<fact.endOffset && fact.beginOffset<range.endOffset)
            aggregateError="register factory overlaps a semantic transaction";
        ct::Replacements retained;
        const auto bytes=SM.getBufferData(SM.getMainFileID());
        for (const auto& edit:staged) {
          const unsigned end=edit.getOffset()+edit.getLength();
          if (edit.getFilePath()==factory.getFilePath() && edit.getOffset()<fact.endOffset && fact.beginOffset<end) {
            // Absorb only the ordinary namespace token mapping already
            // covered by the full literal callee replacement, never an edit
            // in its argument or an unrelated semantic rewrite.
            if (fact.beginOffset<=edit.getOffset() && end<=fact.endOffset &&
                edit.getReplacementText()=="ascify_cg" && bytes.substr(edit.getOffset(),edit.getLength())=="cooperative_groups") continue;
            aggregateError="register factory overlaps an unproved replacement";break;
          }
          if (!llcompat::insertReplacement(retained,edit,&aggregateError)) break;
        }
        if (!aggregateError.empty() || !llcompat::insertReplacement(retained,factory,&aggregateError)) break;
        staged=std::move(retained);registerApplied.push_back(fact);
      }
      if (aggregateError.empty() && !registerApplied.empty()) {
        const auto start=SM.getLocForStartOfFile(SM.getMainFileID());
        llcompat::insertReplacement(staged,ct::Replacement(SM,start,0,
            "#include <ascify/register_tile32_compat.hpp>\n"),&aggregateError);
      }
      if (aggregateError.empty()) {
        *replacements=std::move(staged);
        for (const auto& fact:registerApplied) context->plan->markDeviceRegisterApplied(fact);
        for (const auto& fact:applied) context->plan->markDeviceInlineApplied(fact);
        llvm::errs()<<"Ascify closed aggregate device helpers v1: definitions="<<applied.size()<<"\n";
      }
    }
    if (!aggregateError.empty()) {
      const auto diagnostic=getCompilerInstance().getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,"device aggregate inline boundary: %0");
      getCompilerInstance().getDiagnostics().Report(diagnostic)<<aggregateError;
      cudaRuntimeApiBoundaryFailed=true;return;
    }
  }
  // Every helper edit remains staged until the raw pass reaches EOF. Any
  // earlier target-recipe return leaves this false, so EndSourceFileAction
  // cannot commit a closure that skipped raw preprocessor auditing.
  nvidiaSampleHelperRawAuditCompleted = true;
}

void AscifyAction::AddSkippedSourceRange(clang::SourceRange Range,
                                        clang::SourceLocation EndifLoc) {
  SkippedSourceRanges.push_back(Range);
  proveSkippedNvidiaSampleHelperBranch(Range, EndifLoc);
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
  if (Result.Nodes.getNodeAs<clang::FunctionDecl>(sCudaAttributeGuard) != nullptr) {
    proveCudaAttributeGuard(Result);
    return;
  }
  if (const auto *function = Result.Nodes.getNodeAs<clang::FunctionDecl>(sGridConstantParameterGuard)) {
    specializeGridConstantParameters(function);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::GCCAsmStmt>(sCudaLaneIdAsm) != nullptr) {
    if (!rewriteCudaLaneIdAsm(Result)) (void)rewriteCudaPackedSadAsm(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::ParmVarDecl>(
          sCudaGlobalScalarDoubleParam) != nullptr) {
    (void)lowerCudaGlobalScalarDoubleParam(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::VarDecl>(sCudaDeviceSymbol) != nullptr) {
    (void)registerCudaDeviceSymbol(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::VarDecl>(sCudaDefaultDim3) != nullptr) {
    (void)rewriteCudaDefaultDim3(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::VarDecl>(sCudaHalf2DirectInit) != nullptr) {
    (void)rewriteCudaHalf2DirectInit(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::CXXOperatorCallExpr>(sCudaHalf2Operator) != nullptr) {
    (void)rewriteCudaHalf2Operator(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::DeclRefExpr>(sFrontendHostMath) != nullptr) {
    (void)rewriteFrontendHostMath(Result);
    return;
  }
  if (Result.Nodes.getNodeAs<clang::CallExpr>(
          sProvenGlobalAtomicCall) != nullptr) {
    if (!rewriteCudaSmDiagnostic(Result) &&
        !rewriteCudaFloatFrexp(Result) &&
        !preserveHostStandardMathCall(Result) &&
        !rewriteUnsupportedCudaSmAttribute(Result) &&
        !rewriteProvenKernelAttributeCall(Result) &&
        !preserveCudaIntegerIntrinsicArgumentConversion(Result) &&
        !rewriteProvenGlobalAtomicCall(Result) &&
        !rewriteProvenGenericAtomicAddCall(Result) &&
        !rewriteProvenGenericAtomicMaxCall(Result) &&
        !rewriteProvenGenericAtomicExchCall(Result))
      (void)rewriteProvenDeviceMemoryCall(Result);
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
