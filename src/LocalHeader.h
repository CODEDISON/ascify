#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "clang/Tooling/CommonOptionsParser.h"

namespace ct = clang::tooling;

enum class LocalHeaderClosureMode {
  Disabled,
  Direct,
  Recursive,
};

enum class LocalHeaderNodeState {
  Discovered,
  Translating,
  Staged,
  Failed,
};

struct LocalHeaderNode {
  std::string sourcePath;
  std::string artifactPath;
  std::string artifactRelativePath;
  unsigned depth = 0;
  unsigned rootId = 0;
  LocalHeaderNodeState state = LocalHeaderNodeState::Discovered;
};

struct LocalHeaderEdge {
  std::string parentSourcePath;
  std::string originalSpelling;
  std::string childSourcePath;
  std::string emittedSpelling;
  unsigned sourceOffset = 0;
  unsigned resumeLine = 0;
  std::string resumeFile;
  // Original-context evidence only; ordinary closure edges leave defaults.
  bool isAngled = false;
  bool directLiteral = false;
  unsigned filenameOffset = 0;
  bool physicallySkipped = false;
};

// Evidence belongs to one parse of the original root include context. It is
// never an instruction to inject an assumed header or macro environment.
struct LocalHeaderInputEvidence {
  bool standardPreprocessing = false;
  std::string tokenSha256;
  std::string macroSha256;
  // Ordered literal pragma events include their full payload. Clang annotation
  // tokens alone do not expose e.g. the count of an unroll directive.
  std::string pragmaSha256;
  // Plain current-driver metadata; no AST node or producer-granted capability.
  std::string dynamicWarpSdkTuple;
  bool hasPhysicalLineDirective = false;
  bool ordinaryPhysicalInput = false;
  // Discovery may parse a temporary root copy. Bind its actual main buffer,
  // separately from the original source identity used by contextual reparses.
  std::string parsedRootSourcePath;
  std::string parsedRootSha256;
  std::map<std::string, std::string> fileSha256;
  // Unparsed original SDK providers are data witnesses, never AST/macro owners.
  std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> providerIdentities;
  std::map<std::string, std::uint64_t> providerSizes;
  std::map<std::string, std::string> pathBindings;
  std::string changedPathBinding;
  std::map<std::string, unsigned> fileEntries;
  std::set<std::string> pragmaFiles;
  std::set<std::string> literalContextPragmaFiles;
  std::map<std::string, std::vector<unsigned>> contextOnceLineOffsets;
  std::set<std::string> quotedIncludeParents;
  std::vector<LocalHeaderEdge> originalEdges;
};

struct LocalHeaderContextInline {
  std::string sourcePath;
  std::string sourceSha256;
  unsigned includeOffset = 0;
};

// A root AST may require an explicit bound in an independently staged
// selected header. No FunctionDecl/FileID survives between the two parses.
struct LocalKernelThreadBoundPolicy {
  std::string sourcePath, sourceSha256, kernelName, artifactPath;
  std::uint64_t sourceBytes = 0;
  unsigned declarationOffset = 0, qualifierOffset = 0, maximum = 0;
  bool operator==(const LocalKernelThreadBoundPolicy &other) const {
    return sourcePath == other.sourcePath && sourceSha256 == other.sourceSha256 &&
        sourceBytes == other.sourceBytes && kernelName == other.kernelName &&
        declarationOffset == other.declarationOffset && qualifierOffset == other.qualifierOffset &&
        maximum == other.maximum && artifactPath == other.artifactPath;
  }
};
using LocalKernelThreadBoundPolicyKey = std::pair<std::string, unsigned>;
using LocalKernelThreadBoundPolicies =
    std::map<LocalKernelThreadBoundPolicyKey, LocalKernelThreadBoundPolicy>;

struct LocalHeaderIncludeDecision {
  bool redirect = false;
  bool fatal = false;
  std::string emittedSpelling;
  std::string diagnostic;
};

struct LocalHeaderClosureStats {
  std::size_t quotedLiteralEdgesSeen = 0;
  std::size_t angleEdgesIgnored = 0;
  std::size_t externalQuotedEdges = 0;
  std::size_t selectedIncludeEdges = 0;
  std::size_t redirectedRootEdges = 0;
  std::size_t redirectedHeaderEdges = 0;
  std::size_t duplicateEdges = 0;
  std::size_t cycleEdges = 0;
  std::size_t alreadyTranslatedEdges = 0;
  std::size_t missingRequiredEdges = 0;
  std::size_t unsupportedMacroIncludes = 0;
};

class LocalHeaderClosurePlan {
 public:
  LocalHeaderClosurePlan(const std::string &rootSourcePath,
                         const std::string &rootArtifactPath,
                         const std::string &bundlePath,
                         LocalHeaderClosureMode mode);

  LocalHeaderIncludeDecision observeInclude(
      const std::string &parentSourcePath,
      const std::string &parentArtifactPath,
      unsigned parentDepth,
      unsigned sourceOffset,
      const std::string &originalSpelling,
      const std::string &resolvedPath,
      bool isAngled,
      bool isLiteral,
      unsigned resumeLine = 0,
      const std::string &resumeFile = {});

  bool nextDiscovered(std::size_t &index) const;
  LocalHeaderNode &node(std::size_t index) { return nodes_.at(index); }
  const std::vector<LocalHeaderNode> &nodes() const { return nodes_; }
  const std::vector<LocalHeaderEdge> &edges() const { return edges_; }
  const LocalHeaderClosureStats &stats() const { return stats_; }

  void markTranslating(std::size_t index);
  void markStaged(std::size_t index);
  void markFailed(std::size_t index, const std::string &reason);
  void fail(const std::string &reason);

  bool failed() const { return failed_; }
  const std::string &failureReason() const { return failureReason_; }
  const std::string &rootSourcePath() const { return rootSourcePath_; }
  const std::string &rootArtifactPath() const { return rootArtifactPath_; }
  const std::string &bundlePath() const { return bundlePath_; }
  bool isSelectedRootHeader(const std::string &path) const;
  void requireInheritedHelperContext(const std::string &path);
  const std::set<std::string> &inheritedHelperHeaders() const {
    return inheritedHelperHeaders_;
  }
  void requireInheritedSourceContext(const std::string &path);
  const std::set<std::string> &inheritedSourceHeaders() const {
    return inheritedSourceHeaders_;
  }
  void retainContextEvidence(const LocalHeaderInputEvidence &evidence,
                             const std::vector<LocalHeaderContextInline> &inlined);
  const LocalHeaderInputEvidence &contextEvidence() const { return contextEvidence_; }
  const std::vector<LocalHeaderContextInline> &contextInlines() const { return contextInlines_; }
  bool hasPublishedBundle() const { return !nodes_.empty() || !contextInlines_.empty(); }
  bool verifyContextInputs();
  bool validateArtifactIsolation(bool allowInplaceRootAlias);
  bool validateStagedArtifacts(const std::string &rootStagedPath,
                               const std::string &stagedBundlePath);
  bool requireKernelThreadBoundPolicy(const LocalKernelThreadBoundPolicy &policy);
  const LocalKernelThreadBoundPolicies &kernelThreadBoundPolicies() const {
    return kernelThreadBoundPolicies_;
  }
  bool sealKernelThreadBoundPolicies(const std::string &sourcePath,
      const std::string &artifactPath, const std::string &stagedPath,
      const LocalKernelThreadBoundPolicies &applied);
  bool validateKernelThreadBoundPolicies(const std::string &stagedBundlePath);
  const std::map<LocalKernelThreadBoundPolicyKey, std::string> &kernelThreadBoundArtifacts() const {
    return kernelThreadBoundArtifacts_;
  }


 private:
  struct EligibleRoot {
    std::string canonicalPath;
    unsigned id = 0;
  };

  std::string canonicalExisting(const std::string &path) const;
  std::string absoluteNormalized(const std::string &path) const;
  std::string weaklyCanonical(const std::string &path) const;
  bool isWithin(const std::string &path, const std::string &root) const;
  bool isExcluded(const std::string &path) const;
  bool findEligibleRoot(const std::string &path,
                        EligibleRoot &root,
                        std::string &relative) const;
  std::string makeArtifactPath(const EligibleRoot &root,
                               const std::string &relative) const;
  std::string makeRelativeInclude(const std::string &parentArtifact,
                                  const std::string &childArtifact) const;
  bool endsWithDpp(const std::string &path) const;

  std::string rootSourcePath_;
  std::string rootArtifactLexicalPath_;
  std::string bundleLexicalPath_;
  std::string rootArtifactPath_;
  std::string bundlePath_;
  LocalHeaderClosureMode mode_ = LocalHeaderClosureMode::Disabled;
  std::vector<EligibleRoot> roots_;
  std::vector<std::string> excludedRoots_;
  std::set<std::string> dependencySources_;
  std::vector<LocalHeaderNode> nodes_;
  std::map<std::string, std::size_t> nodeBySource_;
  std::vector<LocalHeaderEdge> edges_;
  LocalHeaderClosureStats stats_;
  std::set<std::string> inheritedHelperHeaders_;
  std::set<std::string> inheritedSourceHeaders_;
  LocalKernelThreadBoundPolicies kernelThreadBoundPolicies_;
  std::map<LocalKernelThreadBoundPolicyKey, std::string> kernelThreadBoundArtifacts_;
  LocalHeaderInputEvidence contextEvidence_;
  std::vector<LocalHeaderContextInline> contextInlines_;
  bool failed_ = false;
  std::string failureReason_;
};

struct LocalHeaderRewriteContext {
  LocalHeaderClosurePlan *plan = nullptr;
  std::string sourcePath;
  std::string artifactPath;
  unsigned depth = 0;
  // With a non-null virtualInput, main.cpp parses this exact buffer at the
  // original root path using Tool.run, never runAndSave on an input file.
  const std::string *virtualInput = nullptr;
  LocalHeaderInputEvidence *inputEvidence = nullptr;
  // Internal source-only joint input capability, set solely by contextualize
  // after the original root, include boundaries and leaf bytes are verified.
  // It grants no publication before the outer token/macro/path/input checks.
  bool sourceOnlyJointInput = false;
  // Internal recursive discovery only. A deferred parse supplies evidence,
  // never an artifact; the strict joint parse must commit the helper proof.
  bool allowInheritedHelperDiscovery = false;
  bool deferredInheritedHelperDiscovery = false;
  bool requireHelperTransaction = false;
  bool helperTransactionCommitted = false;
  // Filled only after the independently rebuilt AST fact produced its exact edit.
  LocalKernelThreadBoundPolicies appliedKernelThreadBoundPolicies;

  LocalHeaderIncludeDecision observe(
      unsigned sourceOffset,
      const std::string &originalSpelling,
      const std::string &resolvedPath,
      bool isAngled,
      bool isLiteral,
      unsigned resumeLine = 0,
      const std::string &resumeFile = {}) {
    if (plan == nullptr)
      return {};
    return plan->observeInclude(sourcePath, artifactPath, depth, sourceOffset,
                                originalSpelling, resolvedPath, isAngled,
                                isLiteral, resumeLine, resumeFile);
  }
};

extern bool ascifySingleSource(
    const std::string &srcPath,
    const std::string &dstPath,
    const ct::CompilationDatabase *compDB,
    ct::CommonOptionsParser *OptionsParserPtr,
    const char *ascify_exe_path,
    const std::string &mainContextPath,
    bool preserveTemp,
    LocalHeaderRewriteContext *localHeaderContext = nullptr);

bool ascifySourceWithLocalHeaderClosure(
    const std::string &mainSourceAbsPath,
    const std::string &rootOutputPath,
    const ct::CompilationDatabase *compDB,
    ct::CommonOptionsParser *OptionsParserPtr,
    const char *ascify_exe,
    bool recursive,
    bool inplace,
    bool analysisOnly = false);
