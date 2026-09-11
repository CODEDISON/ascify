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

#pragma once

#include <cstdint>
#include <deque>
#include <set>
#include <utility>
#include <vector>
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "DavC310TargetRecipe.h"
#include "ReplacementsFrontendActionFactory.h"
#include "Statistics.h"

namespace ct = clang::tooling;
namespace mat = clang::ast_matchers;
using namespace llvm;

/**
  * A FrontendAction that ascifies CUDA programs.
  */
class AscifyAction : public clang::ASTFrontendAction,
                     public mat::MatchFinder::MatchCallback {
private:
  struct SemanticRewriteRange {
    clang::FileID file;
    unsigned beginOffset;
    unsigned endOffset;
  };

  struct NvidiaSampleHelperInclude {
    clang::SourceLocation hashLocation;
    clang::SourceLocation filenameEnd;
  };

  enum class NvidiaSampleHelperMacroKind {
    CheckCudaErrors,
    GetLastCudaError,
  };

  struct NvidiaSampleHelperMacroCandidate {
    NvidiaSampleHelperMacroKind kind;
    clang::SourceLocation nameLocation;
    SemanticRewriteRange invocationRange;
    bool statusDomainProven = false;
    bool rewritten = false;
  };

  struct NvidiaSampleFindDeviceCandidate {
    clang::SourceLocation nameLocation;
    SemanticRewriteRange rewriteRange;
    bool rewritten = false;
  };

  ct::Replacements *replacements = nullptr;
  LocalHeaderRewriteContext *localHeaderContext = nullptr;
  ascify::FrontendCompatibilityConfig frontendCompatibility;
  std::set<unsigned> rewrittenFrontendHostMathOffsets;
  bool frontendHostMathBuiltinMacroDefined = false;
  std::map<std::string, clang::SourceLocation> Ifndefs;
  std::vector<clang::SourceRange> SkippedSourceRanges;
  std::vector<SemanticRewriteRange> SemanticRewriteRanges;
  std::vector<SemanticRewriteRange> nvidiaSampleHelperMacroRanges;
  std::vector<NvidiaSampleHelperMacroCandidate>
      nvidiaSampleHelperMacroCandidates;
  std::vector<NvidiaSampleFindDeviceCandidate>
      nvidiaSampleFindDeviceCandidates;
  std::vector<NvidiaSampleHelperInclude> nvidiaSampleHelperIncludes;
  std::set<std::string> recognizedNvidiaSampleHelperPaths;
  std::set<unsigned> nvidiaSampleHelperNormalMacroOffsets;
  std::set<unsigned> nvidiaSampleFindDeviceNormalOffsets;
  std::vector<bool> nvidiaSampleHelperCudaFileStack;
  std::vector<clang::FileID> nvidiaSampleRetainedFileIds;
  std::set<unsigned> nvidiaSampleRetainedFileIdHashes;
  // FileIDs whose first EnterFile callback already carried a system
  // characteristic.  A later #pragma system_header notification must not be
  // able to promote an ordinary project/preinclude file into this set.
  std::set<unsigned> trustedSystemFileIds;
  std::set<std::pair<std::uint64_t, std::uint64_t>>
      initiallyTrustedSystemFileIdentities;
  std::set<std::pair<std::uint64_t, std::uint64_t>>
      initiallyUntrustedSystemFileIdentities;
  // Physical identities admitted on their first EnterFile callback by an
  // exact frozen NVIDIA helper-functions closure member. This is deliberately
  // independent of later system_header state.
  std::set<std::pair<std::uint64_t, std::uint64_t>>
      frozenNvidiaSampleMacroProviderIdentities;
  std::set<std::pair<std::uint64_t, std::uint64_t>>
      frozenNvidiaSampleDirectFunctionsRootIdentities;
  std::set<unsigned> frozenNvidiaSampleDirectFunctionsRootFileIds;
  bool frozenOfficialNvidiaSampleHelperFunctionsSeen = false;
  bool frozenNvidiaSampleInternalImageMaxExpansionSeen = false;
  unsigned preprocessorConditionalDepth = 0;
  bool nvidiaSampleHelperUnsupportedMacroUse = false;
  bool nvidiaSampleHelperUnsupportedDeclarationUse = false;
  bool nvidiaSampleHelperRewriteFailed = false;
  bool nvidiaSampleHelperOutputMacroEverDefined = false;
  bool nvidiaSampleHelperPendingMacroGeneratedPragma = false;
  bool ascifyCudaCompatDeclarationConflict = false;
  bool ascifyCudaCompatIncludeConflict = false;
  bool ascifyCudaCompatActiveMacroConflict = false;
  std::string ascifyCudaCompatActiveMacroConflictName;
  bool nvidiaSampleFrozenProfileMacroConflict = false;
  bool nvidiaSampleHelperReadyToCommit = false;
  bool nvidiaSampleHelperRawAuditCompleted = false;
  unsigned nvidiaSampleCheckCudaErrorsRewrites = 0;
  unsigned nvidiaSampleGetLastCudaErrorRewrites = 0;
  unsigned nvidiaSampleFindCudaDeviceRewrites = 0;
  std::unique_ptr<mat::MatchFinder> Finder;
  ascify::DavC310TargetRecipe davC310TargetRecipe;
  // CUDA implicitly adds its runtime header. We rewrite explicitly-provided CUDA includes with equivalent
  // ones, and track - using this flag - if the result led to us including the hip runtime header. If it did
  // not, we insert it at the top of the file when we finish processing it.
  // This approach means we do the best it's possible to do w.r.t preserving the user's include order.
  bool insertedRuntimeHeader = false;
  bool insertedBLASHeader = false;
  bool insertedBLASHeader_V2 = false;
  bool insertedRANDHeader = false;
  bool insertedRAND_kernelHeader = false;
  bool insertedDNNHeader = false;
  bool insertedFFTHeader = false;
  bool insertedSPARSEHeader = false;
  bool insertedSPARSEHeader_V2 = false;
  bool insertedComplexHeader = false;
  bool insertedSOLVERHeader = false;
  bool insertedFILEHeader = false;
  bool firstHeader = false;
  bool needsCudaCompatHeader = false;
  bool hasCudaCompatHeader = false;
  bool hasCudaCompatHeaderBeforeNvidiaHelper = false;
  bool needsDavC310TargetHeader = false;
  bool hasDavC310TargetHeader = false;
  bool hasDavC310SimdTargetHeader = false;
  bool hasCubCompatHeader = false;
  bool pragmaOnce = false;
  clang::SourceLocation firstHeaderLoc;
  clang::SourceLocation firstCubCompatHeaderLoc;
  clang::SourceLocation pragmaOnceLoc;
  std::deque<clang::Token> rawTokenWindow;
  std::set<unsigned> loweredDeviceDoubleParamOffsets;
  std::set<unsigned> rewrittenCudaDefaultDim3Offsets;
  std::set<unsigned> rewrittenCudaHalf2DirectInitOffsets;
  std::set<unsigned> rewrittenCudaHalf2OperatorOffsets;
  std::set<unsigned> rewrittenGlobalAtomicOffsets;
  std::set<unsigned> rewrittenWarpAddReductionOffsets;
  std::set<unsigned> taggedCanonicalReducerOffsets;
  std::set<std::string> rowwiseSimdMacrosEverDefined;
  std::set<unsigned> rowwiseSimdRawScannedFiles;
  std::string rowwiseSimdRawConflictingDeclaration;
  static constexpr std::size_t kRawTokenWindowCap = 128;
  // Rewrite a string literal to refer to hip, not CUDA.
  void RewriteString(StringRef s, clang::SourceLocation start);
  // Replace a CUDA identifier with the corresponding hip identifier, if applicable.
  // Returns true if the raw lexer was advanced past rewritten text; the caller must not
  // call LexFromRawLexer for the current token again.
  bool RewriteToken(clang::Lexer &rawLex, clang::Token &tok);
  bool isInSemanticRewriteRange(clang::SourceLocation loc);
  bool isInNvidiaSampleHelperMacroRange(clang::SourceLocation loc);
  bool hasUnsupportedNvidiaSampleHelperDeclarationUse();
  void rewriteProvenNvidiaSampleHelperMacros();
  void auditRawNvidiaSampleHelperToken(const clang::Token &token);
  void auditRawPublishedCompatTokensInFile(clang::FileID file);
  void auditActiveAscifyCudaCompatMacrosAtInsertion();
  void auditExternalNvidiaSampleHelperPreprocessorUse(
      clang::SourceLocation location,
      const clang::Token &macroNameToken,
      llvm::StringRef directive);
  void auditSkippedNvidiaSampleHelperElifDirective(
      clang::SourceLocation location,
      clang::SourceRange conditionRange,
      llvm::StringRef directive);
  void auditFrozenNvidiaSampleHelperMacroDependency(
      clang::SourceLocation useLocation,
      const clang::Token &macroNameToken,
      const clang::MacroDefinition &definition,
      llvm::StringRef useKind);
  bool isDirectFrozenFunctionsImageProviderInstance(
      clang::SourceLocation imageLocation);
  bool admitFrozenHelperMaxFallback(const clang::Token &macroNameToken);
  void finalizeNvidiaSampleHelperClosure();
  // Calculate str's SourceLocation in SourceRange sr
  clang::SourceLocation GetSubstrLocation(const std::string &str, const clang::SourceRange &sr);

public:
  explicit AscifyAction(
      ct::Replacements *replacements,
      LocalHeaderRewriteContext *context,
      const ascify::FrontendCompatibilityConfig& compatibility):
    clang::ASTFrontendAction(),
    replacements(replacements),
    localHeaderContext(context),
    frontendCompatibility(compatibility) {}
  // MatchCallback listeners
  bool cudaLaunchKernel(const mat::MatchFinder::MatchResult &Result);
  bool lowerCudaGlobalScalarDoubleParam(const mat::MatchFinder::MatchResult &Result);
  bool rewriteCudaDefaultDim3(const mat::MatchFinder::MatchResult &Result);
  bool rewriteCudaHalf2DirectInit(const mat::MatchFinder::MatchResult &Result);
  bool rewriteCudaHalf2Operator(const mat::MatchFinder::MatchResult &Result);
  bool rewriteFrontendHostMath(const mat::MatchFinder::MatchResult &Result);
  bool rewriteProvenGlobalAtomicCall(
      const mat::MatchFinder::MatchResult &Result);
  bool rewriteCanonicalWarpAddReduction(
      const mat::MatchFinder::MatchResult &Result);
  bool tagCanonicalBinaryReducer(
      const mat::MatchFinder::MatchResult &Result);
  bool cudaDeviceFuncCall(const mat::MatchFinder::MatchResult &Result);
  bool cudaHostFuncCall(const mat::MatchFinder::MatchResult &Result);
  bool cudaOverloadedHostFuncCall(const mat::MatchFinder::MatchResult &Result);
  bool cubNamespacePrefix(const mat::MatchFinder::MatchResult &Result);
  bool cubFunctionTemplateDecl(const mat::MatchFinder::MatchResult &Result);
  bool cubUsingNamespaceDecl(const mat::MatchFinder::MatchResult &Result);
  bool half2Member(const mat::MatchFinder::MatchResult &Result);
  bool dataTypeSelection(const mat::MatchFinder::MatchResult &Result);

  // Record the include-search characteristic at file entry. This is kept
  // separate from SourceManager::isInSystemHeader(), whose current value can
  // be changed later by #pragma clang/GCC system_header.
  void FileChanged(clang::SourceLocation location,
                   clang::PPCallbacks::FileChangeReason reason,
                   clang::SrcMgr::CharacteristicKind fileType);

  // Called by the preprocessor for each include directive during the non-raw lexing pass.
  void InclusionDirective(clang::SourceLocation hash_loc,
                          const clang::Token &include_token,
                          StringRef file_name,
                          bool is_angled,
                          clang::CharSourceRange filename_range,
                          StringRef resolved_file_name,
                          StringRef search_path,
                          StringRef relative_path,
                          const clang::Module *imported);
  // Called by the preprocessor for each pragma directive during the non-raw lexing pass.
  void PragmaDirective(clang::SourceLocation Loc, clang::PragmaIntroducerKind Introducer);
  void PragmaDiagnostic();
  void finalizePendingNvidiaSampleHelperPragma();
  // Called by the preprocessor for each ifndef directive during the non-raw lexing pass.
  // Found ifndef will be used in EndSourceFileAction() for catching include guard controlling macro.
  void Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD);
  void Ifdef(clang::SourceLocation Loc, const clang::Token &MacroNameTok,
             const clang::MacroDefinition &MD);
  void Elifdef(clang::SourceLocation Loc,
               const clang::Token &MacroNameTok,
               const clang::MacroDefinition &MD,
               llvm::StringRef directive);
  void ElifdefSkipped(clang::SourceLocation Loc,
                      clang::SourceRange ConditionRange,
                      llvm::StringRef directive);
  void ConditionalDirectiveEntered();
  void ConditionalDirectiveEnded();
  void Defined(const clang::Token &MacroNameTok,
               const clang::MacroDefinition &MD,
               clang::SourceRange Range);
  void MacroUndefined(const clang::Token &MacroNameTok,
                      const clang::MacroDefinition &MD);
  void MacroDefined(const clang::Token &MacroNameTok);
  // Rewrite only macro expansions whose definition comes from a recognized
  // NVIDIA cuda-samples helper_cuda.h. User macros with the same spelling are
  // intentionally outside this callback's admitted set.
  void MacroExpands(const clang::Token &MacroNameTok,
                    const clang::MacroDefinition &MD,
                    clang::SourceRange Range);
  //
  void AddSkippedSourceRange(clang::SourceRange Range);

protected:
  // Add a Replacement for the current file. These will all be applied after executing the FrontendAction.
  bool insertReplacement(const ct::Replacement &rep,
                         const clang::FullSourceLoc &fullSL);
  bool insertSemanticReplacement(const ct::Replacement &rep,
                                 const clang::FullSourceLoc &fullSL,
                                 clang::SourceLocation begin,
                                 clang::SourceLocation end);
  // FrontendAction entry point.
  void ExecuteAction() override;
  // Callback before starting processing a single input; used by ascify-clang for setting Preprocessor options.
  bool BeginInvocation(clang::CompilerInstance &CI) override;
  // Called at the start of each new file to process.
  void EndSourceFileAction() override;
  // MatchCallback API entry point. Called by the AST visitor while searching the AST for things we registered an interest for.
  void run(const mat::MatchFinder::MatchResult &Result) override;
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, StringRef InFile) override;
  bool Exclude(const dppCounter &hipToken);
  void FindAndReplace(StringRef name, clang::SourceLocation sl, const std::map<StringRef, dppCounter> &repMap, bool bReplace = true);
};
