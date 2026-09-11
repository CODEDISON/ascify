#include "LocalHeader.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "ArgParse.h"
#include "LLVMCompat.h"

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

namespace fs = std::filesystem;

namespace {

std::string pathString(const fs::path &path) {
  return path.lexically_normal().generic_string();
}

bool readFile(const std::string &path, std::string &out) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return false;
  out = buffer.get()->getBuffer().str();
  return true;
}

bool existsAndIsRegular(const std::string &path) {
  std::error_code ec;
  return fs::is_regular_file(fs::path(path), ec) && !ec;
}

bool isSymlink(const std::string &path) {
  std::error_code ec;
  return fs::is_symlink(fs::path(path), ec) && !ec;
}

bool existsAndIsNonemptyRegular(const std::string &path) {
  std::error_code ec;
  return fs::is_regular_file(fs::path(path), ec) && !ec &&
         fs::file_size(fs::path(path), ec) > 0 && !ec;
}

bool writeManifest(const std::string &path,
                   const LocalHeaderClosurePlan &plan) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec);
  if (ec)
    return false;
  out << "schema=1\n";
  out << "root_source=" << plan.rootSourcePath() << "\n";
  out << "root_artifact=" << plan.rootArtifactPath() << "\n";
  for (const auto &node : plan.nodes()) {
    out << "header=" << node.sourcePath << "\t" << node.artifactPath
        << "\n";
  }
  return true;
}

bool ownedBundle(const std::string &path) {
  if (isSymlink(path))
    return false;
  const fs::path marker = fs::path(path) / ".ascify-local-closure";
  if (isSymlink(pathString(marker)) ||
      !existsAndIsRegular(pathString(marker)))
    return false;
  std::string content;
  return readFile(pathString(marker), content) &&
         content.rfind("schema=1\n", 0) == 0;
}

bool inspectPath(const std::string &path,
                 bool &exists,
                 fs::file_status &status,
                 std::string &diagnostic) {
  std::error_code ec;
  status = fs::symlink_status(fs::path(path), ec);
  if (ec == std::errc::no_such_file_or_directory ||
      ec == std::errc::not_a_directory) {
    exists = false;
    return true;
  }
  if (ec) {
    diagnostic = ec.message() + ": inspecting " + path;
    return false;
  }
  exists = status.type() != fs::file_type::not_found;
  return true;
}

bool renameChecked(const std::string &from,
                   const std::string &to,
                   const std::string &operation,
                   std::string &diagnostic) {
  std::error_code ec;
  fs::rename(fs::path(from), fs::path(to), ec);
  if (!ec)
    return true;
  diagnostic = ec.message() + ": " + operation + " (" + from + " -> " +
               to + ")";
  return false;
}

bool removeExactChecked(const std::string &path,
                        const std::string &operation,
                        std::string &diagnostic) {
  bool exists = false;
  fs::file_status status;
  if (!inspectPath(path, exists, status, diagnostic))
    return false;
  if (!exists)
    return true;

  std::error_code ec;
  if (fs::is_directory(status) && !fs::is_symlink(status)) {
    const std::uintmax_t removed = fs::remove_all(fs::path(path), ec);
    if (!ec && removed != static_cast<std::uintmax_t>(-1))
      return true;
  } else {
    const bool removed = fs::remove(fs::path(path), ec);
    if (!ec && removed)
      return true;
  }
  diagnostic = (ec ? ec.message() : "path was not removed") + ": " +
               operation + " (" + path + ")";
  return false;
}

bool moveAside(const std::string &path,
               std::string &backup,
               std::string &diagnostic) {
  bool pathExists = false;
  fs::file_status pathStatus;
  if (!inspectPath(path, pathExists, pathStatus, diagnostic))
    return false;
  if (!pathExists) {
    backup.clear();
    return true;
  }
  backup = path + ".ascify-backup";
  bool backupExists = false;
  fs::file_status backupStatus;
  if (!inspectPath(backup, backupExists, backupStatus, diagnostic))
    return false;
  if (backupExists) {
    diagnostic = "stale transaction backup exists: " + backup;
    return false;
  }
  return renameChecked(path, backup, "moving published artifact aside",
                       diagnostic);
}

bool retryRollbackRemove(const std::string &path,
                         std::string &diagnostic) {
  std::string firstError;
  if (removeExactChecked(path, "removing partially published artifact",
                         firstError))
    return true;
  llvm::errs() << sAscify << sError << "Rollback remove failed once; retrying: "
               << firstError << "\n";
  std::string retryError;
  if (removeExactChecked(path, "retrying partial-artifact removal",
                         retryError)) {
    llvm::errs() << sAscify
                 << "Rollback remove recovered after a checked retry\n";
    return true;
  }
  diagnostic = firstError + "; retry failed: " + retryError;
  return false;
}

bool retryRollbackRename(const std::string &backup,
                         const std::string &path,
                         std::string &diagnostic) {
  std::string firstError;
  if (renameChecked(backup, path, "restoring transaction backup", firstError))
    return true;
  llvm::errs() << sAscify << sError << "Rollback rename failed once; retrying: "
               << firstError << "\n";
  std::string retryError;
  if (renameChecked(backup, path, "retrying transaction-backup restore",
                    retryError)) {
    llvm::errs() << sAscify
                 << "Rollback rename recovered after a checked retry\n";
    return true;
  }
  diagnostic = firstError + "; retry failed: " + retryError;
  return false;
}

bool restoreBackup(const std::string &path,
                   const std::string &backup,
                   std::string &diagnostic) {
  if (!retryRollbackRemove(path, diagnostic))
    return false;
  if (backup.empty())
    return true;
  if (!retryRollbackRename(backup, path, diagnostic))
    return false;

  bool restoredExists = false;
  fs::file_status restoredStatus;
  if (!inspectPath(path, restoredExists, restoredStatus, diagnostic))
    return false;
  bool backupExists = false;
  fs::file_status backupStatus;
  if (!inspectPath(backup, backupExists, backupStatus, diagnostic))
    return false;
  if (!restoredExists || backupExists) {
    diagnostic = "rollback identity check failed for " + path;
    return false;
  }
  return true;
}

bool removeBackup(const std::string &backup, std::string &diagnostic) {
  if (backup.empty())
    return true;
  return removeExactChecked(backup, "removing committed transaction backup",
                            diagnostic);
}

bool reportRollback(const std::string &rootArtifact,
                    const std::string &rootBackup,
                    const std::string &bundleArtifact,
                    const std::string &bundleBackup) {
  std::string bundleError;
  const bool bundleRestored =
      restoreBackup(bundleArtifact, bundleBackup, bundleError);
  std::string rootError;
  const bool rootRestored = restoreBackup(rootArtifact, rootBackup, rootError);
  if (!bundleRestored)
    llvm::errs() << sAscify << sError
                 << "Local-header bundle rollback failed: " << bundleError
                 << "\n";
  if (!rootRestored)
    llvm::errs() << sAscify << sError
                 << "Local-header root rollback failed: " << rootError << "\n";
  return bundleRestored && rootRestored;
}

bool publishClosure(const std::string &stagedRoot,
                    const std::string &stagedBundle,
                    const std::string &rootArtifact,
                    const std::string &bundleArtifact,
                    bool hasHeaders) {
  std::string rootBackup;
  std::string bundleBackup;
  const bool replaceExistingOwnedBundle = ownedBundle(bundleArtifact);
  std::string operationError;

  bool rootExists = false;
  fs::file_status rootStatus;
  if (!inspectPath(rootArtifact, rootExists, rootStatus, operationError)) {
    llvm::errs() << sAscify << sError << operationError << "\n";
    return false;
  }

  if (rootExists &&
      (fs::is_symlink(rootStatus) || fs::is_directory(rootStatus))) {
    llvm::errs() << sAscify << sError
                 << "Refusing to replace a symlink or directory as root "
                    "output: "
                 << rootArtifact << "\n";
    return false;
  }

  bool bundleExists = false;
  fs::file_status bundleStatus;
  if (!inspectPath(bundleArtifact, bundleExists, bundleStatus,
                   operationError)) {
    llvm::errs() << sAscify << sError << operationError << "\n";
    return false;
  }
  if (hasHeaders && bundleExists && !replaceExistingOwnedBundle) {
    llvm::errs() << sAscify << sError
                 << "Refusing to replace unowned local-header bundle: "
                 << bundleArtifact << "\n";
    return false;
  }
  if ((hasHeaders || replaceExistingOwnedBundle) &&
      !moveAside(bundleArtifact, bundleBackup, operationError)) {
    llvm::errs() << sAscify << sError
                 << "Cannot stage existing local-header bundle: "
                 << bundleArtifact << ": " << operationError << "\n";
    return false;
  }
  if (!moveAside(rootArtifact, rootBackup, operationError)) {
    std::string rollbackError;
    if (!restoreBackup(bundleArtifact, bundleBackup, rollbackError))
      llvm::errs() << sAscify << sError
                   << "Local-header bundle rollback failed: " << rollbackError
                   << "\n";
    llvm::errs() << sAscify << sError
                 << "Cannot stage existing root output: " << rootArtifact
                 << ": " << operationError << "\n";
    return false;
  }

  if (hasHeaders) {
    if (!renameChecked(stagedBundle, bundleArtifact,
                       "publishing local-header bundle", operationError)) {
      reportRollback(rootArtifact, rootBackup, bundleArtifact, bundleBackup);
      llvm::errs() << sAscify << sError << operationError << "\n";
      return false;
    }
  }

  if (!renameChecked(stagedRoot, rootArtifact, "publishing root output",
                     operationError)) {
    reportRollback(rootArtifact, rootBackup, bundleArtifact, bundleBackup);
    llvm::errs() << sAscify << sError << operationError << "\n";
    return false;
  }

  std::string cleanupError;
  if (!removeBackup(rootBackup, cleanupError)) {
    llvm::errs() << sAscify << sError
                 << "Published root output but could not remove its checked "
                    "transaction backup: "
                 << cleanupError << "\n";
    return false;
  }
  if (!removeBackup(bundleBackup, cleanupError)) {
    llvm::errs() << sAscify << sError
                 << "Published local-header closure but could not remove its "
                    "checked transaction backup: "
                 << cleanupError << "\n";
    return false;
  }
  return true;
}

}  // namespace

LocalHeaderClosurePlan::LocalHeaderClosurePlan(
    const std::string &rootSourcePath,
    const std::string &rootArtifactPath,
    const std::string &bundlePath,
    LocalHeaderClosureMode mode)
    : rootSourcePath_(canonicalExisting(rootSourcePath)),
      rootArtifactLexicalPath_(absoluteNormalized(rootArtifactPath)),
      bundleLexicalPath_(absoluteNormalized(bundlePath)),
      rootArtifactPath_(weaklyCanonical(rootArtifactPath)),
      bundlePath_(weaklyCanonical(bundlePath)),
      mode_(mode) {
  if (rootSourcePath_.empty()) {
    fail("cannot canonicalize root source");
    return;
  }
  if (rootArtifactLexicalPath_.empty() || bundleLexicalPath_.empty() ||
      rootArtifactPath_.empty() || bundlePath_.empty()) {
    fail("cannot weakly canonicalize local-header artifact paths");
    return;
  }
  dependencySources_.insert(rootSourcePath_);

  std::vector<std::string> candidates;
  candidates.push_back(pathString(fs::path(rootSourcePath_).parent_path()));
  for (const std::string &includeDir : IncludeDirs)
    candidates.push_back(includeDir);

  unsigned nextId = 0;
  std::set<std::string> seen;
  for (const std::string &candidate : candidates) {
    const std::string canonical = canonicalExisting(candidate);
    if (canonical.empty() || !seen.insert(canonical).second)
      continue;
    roots_.push_back({canonical, nextId++});
  }

  for (const std::string &excluded :
       std::vector<std::string>{CudaPath.getValue(),
                                ClangResourceDir.getValue()}) {
    const std::string canonical = canonicalExisting(excluded);
    if (!canonical.empty())
      excludedRoots_.push_back(canonical);
  }
}

std::string LocalHeaderClosurePlan::canonicalExisting(
    const std::string &path) const {
  if (path.empty())
    return {};
  std::error_code ec;
  const fs::path canonical = fs::canonical(fs::path(path), ec);
  if (ec)
    return {};
  return pathString(canonical);
}

std::string LocalHeaderClosurePlan::weaklyCanonical(
    const std::string &path) const {
  if (path.empty())
    return {};
  std::error_code ec;
  const fs::path absolute = fs::absolute(fs::path(path), ec);
  if (ec)
    return {};
  const fs::path canonical = fs::weakly_canonical(absolute, ec);
  if (ec)
    return {};
  return pathString(canonical);
}

std::string LocalHeaderClosurePlan::absoluteNormalized(
    const std::string &path) const {
  if (path.empty())
    return {};
  std::error_code ec;
  const fs::path absolute = fs::absolute(fs::path(path), ec);
  if (ec)
    return {};
  return pathString(absolute);
}

bool LocalHeaderClosurePlan::isWithin(const std::string &path,
                                      const std::string &root) const {
  std::error_code ec;
  const fs::path relative = fs::relative(fs::path(path), fs::path(root), ec);
  if (ec || relative.is_absolute())
    return false;
  if (relative.empty() || relative == ".")
    return true;
  for (const auto &part : relative) {
    if (part == "..")
      return false;
  }
  return true;
}

bool LocalHeaderClosurePlan::isExcluded(const std::string &path) const {
  for (const std::string &root : excludedRoots_) {
    if (isWithin(path, root))
      return true;
  }
  return false;
}

bool LocalHeaderClosurePlan::findEligibleRoot(
    const std::string &path,
    EligibleRoot &selected,
    std::string &relative) const {
  bool found = false;
  std::size_t longest = 0;
  for (const EligibleRoot &candidate : roots_) {
    if (!isWithin(path, candidate.canonicalPath))
      continue;
    if (!found || candidate.canonicalPath.size() > longest ||
        (candidate.canonicalPath.size() == longest &&
         candidate.id < selected.id)) {
      found = true;
      longest = candidate.canonicalPath.size();
      selected = candidate;
    }
  }
  if (!found)
    return false;

  std::error_code ec;
  const fs::path rel = fs::relative(fs::path(path),
                                    fs::path(selected.canonicalPath), ec);
  if (ec || rel.empty() || rel.is_absolute())
    return false;
  for (const auto &part : rel) {
    if (part == "..")
      return false;
  }
  relative = pathString(rel);
  return true;
}

std::string LocalHeaderClosurePlan::makeArtifactPath(
    const EligibleRoot &root,
    const std::string &relative) const {
  fs::path artifact = fs::path(bundlePath_) /
                      ("r" + std::to_string(root.id)) /
                      fs::path(relative);
  artifact += ".dpp";
  return pathString(artifact);
}

std::string LocalHeaderClosurePlan::makeRelativeInclude(
    const std::string &parentArtifact,
    const std::string &childArtifact) const {
  std::error_code ec;
  const fs::path relative = fs::relative(
      fs::path(childArtifact), fs::path(parentArtifact).parent_path(), ec);
  if (ec || relative.empty() || relative.is_absolute())
    return {};
  return pathString(relative);
}

bool LocalHeaderClosurePlan::endsWithDpp(const std::string &path) const {
  constexpr const char suffix[] = ".dpp";
  return path.size() >= sizeof(suffix) - 1 &&
         path.compare(path.size() - (sizeof(suffix) - 1),
                      sizeof(suffix) - 1, suffix) == 0;
}

LocalHeaderIncludeDecision LocalHeaderClosurePlan::observeInclude(
    const std::string &parentSourcePath,
    const std::string &parentArtifactPath,
    unsigned parentDepth,
    unsigned sourceOffset,
    const std::string &originalSpelling,
    const std::string &resolvedPath,
    bool isAngled,
    bool isLiteral) {
  LocalHeaderIncludeDecision decision;
  if (mode_ == LocalHeaderClosureMode::Disabled)
    return decision;

  std::string canonical;
  if (!resolvedPath.empty()) {
    canonical = canonicalExisting(resolvedPath);
    if (!canonical.empty())
      dependencySources_.insert(canonical);
  }
  if (isAngled) {
    ++stats_.angleEdgesIgnored;
    return decision;
  }
  if (!isLiteral) {
    ++stats_.unsupportedMacroIncludes;
    if (mode_ == LocalHeaderClosureMode::Recursive) {
      fail("macro-produced quoted include cannot form a strict recursive closure");
      decision.fatal = true;
      decision.diagnostic = failureReason_;
    }
    return decision;
  }

  ++stats_.quotedLiteralEdgesSeen;
  if (endsWithDpp(originalSpelling)) {
    ++stats_.alreadyTranslatedEdges;
    return decision;
  }
  if (resolvedPath.empty()) {
    ++stats_.missingRequiredEdges;
    fail("quoted local include has no resolved FileEntry: " +
         originalSpelling);
    decision.fatal = true;
    decision.diagnostic = failureReason_;
    return decision;
  }

  const fs::path lexical(originalSpelling);
  if (lexical.is_absolute()) {
    fail("absolute quoted include is not eligible: " + originalSpelling);
    decision.fatal = true;
    decision.diagnostic = failureReason_;
    return decision;
  }

  if (canonical.empty()) {
    ++stats_.missingRequiredEdges;
    fail("cannot canonicalize resolved quoted include: " + resolvedPath);
    decision.fatal = true;
    decision.diagnostic = failureReason_;
    return decision;
  }
  if (endsWithDpp(canonical)) {
    ++stats_.alreadyTranslatedEdges;
    return decision;
  }
  if (canonical == rootSourcePath_) {
    fail("a selected local header resolves back to the root input: " +
         originalSpelling);
    decision.fatal = true;
    decision.diagnostic = failureReason_;
    return decision;
  }
  if (isExcluded(canonical)) {
    ++stats_.externalQuotedEdges;
    return decision;
  }

  EligibleRoot root;
  std::string relative;
  if (!findEligibleRoot(canonical, root, relative)) {
    bool hasParentTraversal = false;
    for (const auto &part : lexical) {
      if (part == "..") {
        hasParentTraversal = true;
        break;
      }
    }
    bool lexicalInEligibleRoot = false;
    const fs::path lexicalCandidate =
        fs::path(parentSourcePath).parent_path() / lexical;
    std::error_code lexicalEc;
    if (fs::exists(lexicalCandidate, lexicalEc) && !lexicalEc) {
      const std::string lexicalAbsolute =
          absoluteNormalized(pathString(lexicalCandidate));
      for (const EligibleRoot &candidate : roots_) {
        if (isWithin(lexicalAbsolute, candidate.canonicalPath)) {
          lexicalInEligibleRoot = true;
          break;
        }
      }
    }
    if (hasParentTraversal || lexicalInEligibleRoot) {
      fail("quoted include resolves outside its eligible root: " +
           originalSpelling);
      decision.fatal = true;
      decision.diagnostic = failureReason_;
      return decision;
    }
    ++stats_.externalQuotedEdges;
    return decision;
  }

  auto found = nodeBySource_.find(canonical);
  const bool maySelect = parentDepth == 0 ||
                         mode_ == LocalHeaderClosureMode::Recursive;
  if (found == nodeBySource_.end() && !maySelect)
    return decision;

  std::size_t index = 0;
  if (found == nodeBySource_.end()) {
    LocalHeaderNode node;
    node.sourcePath = canonical;
    node.artifactPath = makeArtifactPath(root, relative);
    std::error_code relEc;
    node.artifactRelativePath = pathString(fs::relative(
        fs::path(node.artifactPath), fs::path(bundlePath_), relEc));
    if (relEc || node.artifactRelativePath.empty()) {
      fail("cannot allocate local-header artifact path for: " + canonical);
      decision.fatal = true;
      decision.diagnostic = failureReason_;
      return decision;
    }
    node.depth = parentDepth + 1;
    node.rootId = root.id;
    index = nodes_.size();
    nodes_.push_back(node);
    nodeBySource_[canonical] = index;
  } else {
    index = found->second;
    ++stats_.duplicateEdges;
    if (nodes_[index].state == LocalHeaderNodeState::Translating ||
        nodes_[index].depth <= parentDepth)
      ++stats_.cycleEdges;
  }

  const std::string relativeInclude = makeRelativeInclude(
      parentArtifactPath, nodes_[index].artifactPath);
  if (relativeInclude.empty()) {
    fail("cannot form relative include from artifact to: " +
         nodes_[index].artifactPath);
    decision.fatal = true;
    decision.diagnostic = failureReason_;
    return decision;
  }

  decision.redirect = true;
  decision.emittedSpelling = "\"" + relativeInclude + "\"";
  ++stats_.selectedIncludeEdges;
  if (parentDepth == 0)
    ++stats_.redirectedRootEdges;
  else
    ++stats_.redirectedHeaderEdges;
  edges_.push_back({canonicalExisting(parentSourcePath), originalSpelling,
                    canonical, decision.emittedSpelling, sourceOffset});
  return decision;
}

bool LocalHeaderClosurePlan::nextDiscovered(std::size_t &index) const {
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i].state == LocalHeaderNodeState::Discovered) {
      index = i;
      return true;
    }
  }
  return false;
}

void LocalHeaderClosurePlan::markTranslating(std::size_t index) {
  nodes_.at(index).state = LocalHeaderNodeState::Translating;
}

void LocalHeaderClosurePlan::markStaged(std::size_t index) {
  nodes_.at(index).state = LocalHeaderNodeState::Staged;
}

void LocalHeaderClosurePlan::markFailed(std::size_t index,
                                        const std::string &reason) {
  nodes_.at(index).state = LocalHeaderNodeState::Failed;
  fail(reason);
}

void LocalHeaderClosurePlan::fail(const std::string &reason) {
  if (!failed_) {
    failed_ = true;
    failureReason_ = reason;
  }
}

bool LocalHeaderClosurePlan::validateArtifactIsolation(
    bool allowInplaceRootAlias) {
  struct ArtifactIdentity {
    const char *label;
    std::string path;
    bool directoryLike;
  };

  std::vector<ArtifactIdentity> artifacts = {
      {"root output", weaklyCanonical(rootArtifactLexicalPath_), false},
      {"local-header bundle", weaklyCanonical(bundleLexicalPath_), true},
      {"root backup",
       weaklyCanonical(rootArtifactLexicalPath_ + ".ascify-backup"), false},
      {"local-header bundle backup",
       weaklyCanonical(bundleLexicalPath_ + ".ascify-backup"), true},
  };
  for (const ArtifactIdentity &artifact : artifacts) {
    if (artifact.path.empty()) {
      fail(std::string("cannot weakly canonicalize ") + artifact.label);
      return false;
    }
  }

  for (std::size_t i = 0; i < artifacts.size(); ++i) {
    for (std::size_t j = i + 1; j < artifacts.size(); ++j) {
      const bool overlap =
          artifacts[i].path == artifacts[j].path ||
          (artifacts[i].directoryLike &&
           isWithin(artifacts[j].path, artifacts[i].path)) ||
          (artifacts[j].directoryLike &&
           isWithin(artifacts[i].path, artifacts[j].path));
      if (overlap) {
        fail(std::string(artifacts[i].label) + " aliases " +
             artifacts[j].label + ": " + artifacts[i].path);
        return false;
      }
    }
  }

  for (const ArtifactIdentity &artifact : artifacts) {
    for (const std::string &dependency : dependencySources_) {
      const bool explicitInplaceRoot =
          allowInplaceRootAlias &&
          std::string(artifact.label) == "root output" &&
          dependency == rootSourcePath_ && artifact.path == dependency;
      if (explicitInplaceRoot)
        continue;
      if (artifact.path == dependency ||
          (artifact.directoryLike && isWithin(dependency, artifact.path))) {
        fail(std::string(artifact.label) +
             " aliases or contains an input dependency: " + dependency);
        return false;
      }
    }
  }
  return true;
}

bool LocalHeaderClosurePlan::validateStagedArtifacts(
    const std::string &rootStagedPath,
    const std::string &stagedBundlePath) {
  if (failed_ || !existsAndIsNonemptyRegular(rootStagedPath))
    return false;

  for (const LocalHeaderNode &node : nodes_) {
    if (node.state != LocalHeaderNodeState::Staged) {
      fail("local-header node was not staged: " + node.sourcePath);
      return false;
    }
    const fs::path staged = fs::path(stagedBundlePath) /
                            fs::path(node.artifactRelativePath);
    if (!existsAndIsNonemptyRegular(pathString(staged))) {
      fail("staged local-header artifact is missing: " + pathString(staged));
      return false;
    }
  }

  for (const LocalHeaderEdge &edge : edges_) {
    std::string parentStaged = rootStagedPath;
    if (edge.parentSourcePath != rootSourcePath_) {
      auto found = nodeBySource_.find(edge.parentSourcePath);
      if (found == nodeBySource_.end()) {
        fail("edge parent is absent from closure plan");
        return false;
      }
      parentStaged = pathString(fs::path(stagedBundlePath) /
                                nodes_[found->second].artifactRelativePath);
    }
    std::string content;
    if (!readFile(parentStaged, content) ||
        content.find(edge.emittedSpelling) == std::string::npos) {
      fail("staged artifact did not contain planned .dpp include: " +
           parentStaged);
      return false;
    }
  }
  return true;
}

bool ascifySourceWithLocalHeaderClosure(
    const std::string &mainSourceAbsPath,
    const std::string &rootOutputPath,
    const ct::CompilationDatabase *compDB,
    ct::CommonOptionsParser *OptionsParserPtr,
    const char *ascify_exe,
    bool recursive,
    bool inplace,
    bool analysisOnly) {
  if (mainSourceAbsPath.size() >= 4 &&
      mainSourceAbsPath.compare(mainSourceAbsPath.size() - 4, 4, ".dpp") ==
          0) {
    llvm::errs() << sAscify << sError
                 << "Refusing to retranslate a .dpp root with local-header "
                    "closure enabled: "
                 << mainSourceAbsPath << "\n";
    return false;
  }

  std::error_code absEc;
  fs::path finalRoot = rootOutputPath.empty()
                           ? fs::path(mainSourceAbsPath + ".dpp")
                           : fs::absolute(fs::path(rootOutputPath), absEc);
  if (absEc) {
    llvm::errs() << sAscify << sError
                 << "Cannot normalize root output path: " << rootOutputPath
                 << "\n";
    return false;
  }
  finalRoot = finalRoot.lexically_normal();
  std::error_code rootCanonicalEc;
  const fs::path sourceCanonical =
      fs::canonical(fs::path(mainSourceAbsPath), rootCanonicalEc);
  if (rootCanonicalEc) {
    llvm::errs() << sAscify << sError
                 << "Cannot canonicalize local-header root input: "
                 << mainSourceAbsPath << "\n";
    return false;
  }
  const fs::path outputCanonical =
      fs::weakly_canonical(finalRoot, rootCanonicalEc);
  if (rootCanonicalEc) {
    llvm::errs() << sAscify << sError
                 << "Cannot canonicalize local-header root output: "
                 << pathString(finalRoot) << "\n";
    return false;
  }
  if (!inplace && sourceCanonical == outputCanonical) {
    llvm::errs() << sAscify << sError
                 << "Root output aliases the input; use --inplace explicitly\n";
    return false;
  }
  const fs::path finalBundle =
      inplace ? fs::path(mainSourceAbsPath + ".ascify-headers")
              : fs::path(pathString(finalRoot) + ".headers");

  LocalHeaderClosurePlan plan(
      mainSourceAbsPath, pathString(finalRoot), pathString(finalBundle),
      recursive ? LocalHeaderClosureMode::Recursive
                : LocalHeaderClosureMode::Direct);
  if (plan.failed()) {
    llvm::errs() << sAscify << sError << plan.failureReason() << "\n";
    return false;
  }

  SmallString<256> stagingPrefix(
      pathString(finalRoot.parent_path()));
  llvm::sys::path::append(stagingPrefix, ".ascify-local-closure");
  SmallString<256> stagingRoot;
  const std::error_code stageEc = llvm::sys::fs::createUniqueDirectory(
      stagingPrefix, stagingRoot);
  if (stageEc) {
    llvm::errs() << sAscify << sError << stageEc.message()
                 << ": creating local-header staging directory\n";
    return false;
  }
  const fs::path staging(stagingRoot.c_str());
  const fs::path stagedRoot = staging / "root.dpp";
  const fs::path stagedBundle = staging / "headers";

  LocalHeaderRewriteContext rootContext;
  rootContext.plan = &plan;
  rootContext.sourcePath = plan.rootSourcePath();
  rootContext.artifactPath = plan.rootArtifactPath();
  rootContext.depth = 0;
  bool ok = ascifySingleSource(
      mainSourceAbsPath, pathString(stagedRoot), compDB, OptionsParserPtr,
      ascify_exe, mainSourceAbsPath, false, &rootContext);
  if (!ok || plan.failed()) {
    llvm::errs() << sAscify << sError
                 << "Local-header root staging failed: "
                 << (plan.failed() ? plan.failureReason() : mainSourceAbsPath)
                 << "\n";
    llvm::sys::fs::remove_directories(stagingRoot);
    return false;
  }

  std::size_t index = 0;
  while (plan.nextDiscovered(index)) {
    plan.markTranslating(index);
    const LocalHeaderNode node = plan.node(index);
    const fs::path stagedHeader =
        stagedBundle / fs::path(node.artifactRelativePath);
    const std::error_code mkdirEc = llvm::sys::fs::create_directories(
        pathString(stagedHeader.parent_path()));
    if (mkdirEc) {
      plan.markFailed(index, "cannot create staged header directory");
      break;
    }

    LocalHeaderRewriteContext headerContext;
    headerContext.plan = &plan;
    headerContext.sourcePath = node.sourcePath;
    headerContext.artifactPath = node.artifactPath;
    headerContext.depth = node.depth;
    ok = ascifySingleSource(node.sourcePath, pathString(stagedHeader), compDB,
                            OptionsParserPtr, ascify_exe, node.sourcePath,
                            false, &headerContext);
    if (!ok || plan.failed()) {
      plan.markFailed(index, "local-header translation failed: " +
                                 node.sourcePath +
                                 " (standalone header parse; parent "
                                 "macro/declaration context is not replayed)");
      break;
    }
    plan.markStaged(index);
  }

  if (!plan.validateArtifactIsolation(inplace)) {
    llvm::errs() << sAscify << sError
                 << "Local-header artifact isolation failed: "
                 << plan.failureReason() << "\n";
    llvm::sys::fs::remove_directories(stagingRoot);
    return false;
  }

  if (!plan.validateStagedArtifacts(pathString(stagedRoot),
                                    pathString(stagedBundle))) {
    llvm::errs() << sAscify << sError
                 << "Local-header closure validation failed: "
                 << plan.failureReason() << "\n";
    llvm::sys::fs::remove_directories(stagingRoot);
    return false;
  }

  if (analysisOnly) {
    llvm::sys::fs::remove_directories(stagingRoot);
    return true;
  }

  if (!plan.nodes().empty()) {
    const std::error_code bundleEc = llvm::sys::fs::create_directories(
        pathString(stagedBundle));
    if (bundleEc ||
        !writeManifest(pathString(stagedBundle /
                                  ".ascify-local-closure"), plan)) {
      llvm::errs() << sAscify << sError
                   << "Cannot write local-header closure manifest\n";
      llvm::sys::fs::remove_directories(stagingRoot);
      return false;
    }
  }

  if (!publishClosure(pathString(stagedRoot), pathString(stagedBundle),
                      pathString(finalRoot), pathString(finalBundle),
                      !plan.nodes().empty())) {
    llvm::sys::fs::remove_directories(stagingRoot);
    return false;
  }

  llvm::sys::fs::remove_directories(stagingRoot);
  llvm::outs() << sAscify << "Published translated local include closure: "
               << plan.nodes().size() << " unique header(s), "
               << plan.stats().selectedIncludeEdges << " redirected edge(s)\n";
  return true;
}
