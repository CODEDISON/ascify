#pragma once

#include <string>
#include <functional>
#include <set>
#include <utility>
#include <vector>
#include "clang/Basic/SourceLocation.h"

namespace clang { class ASTContext; class FileEntry; class SourceManager; class TagDecl; }

namespace ascify {
// A literal main-file CUB include whose semantic replacement was accepted.
// Recording the selected physical path does not itself authenticate a provider.
struct CudaVec3MappedCubInclude {
  clang::SourceLocation hashLocation;
  clang::SourceLocation filenameEnd;
  std::string resolvedPath;
  bool unconditional = false;
};

struct CudaVec3LayoutEdit {
  clang::SourceLocation location;
  unsigned length = 0;
  std::string replacement;
};

// The keys are individual observed uses, never a canonical-record exemption.
// A failed proof returns an empty plan; no partial edits can be published.
struct CudaVec3LayoutPlan {
  std::vector<CudaVec3LayoutEdit> edits;
  std::set<std::pair<unsigned, const clang::TagDecl*>> uses;
  std::set<std::string> contextualHeaders;
  std::vector<std::string> evidence;
};

bool ProveCudaVec3LocalLayout(
    clang::ASTContext& context, const std::string& configuredCudaRoot,
    const std::string& ownedClangResourceRoot,
    const std::vector<CudaVec3MappedCubInclude>& mappedCubIncludes,
    const std::function<bool(clang::SourceLocation)>& selectedHeader,
    const std::function<bool(clang::SourceLocation)>& literalUnobserved,
    CudaVec3LayoutPlan& plan, std::string& error,
    clang::SourceLocation& location);

// A skipped include is removable only when its earlier parsed provider belongs
// to an authenticated, unconditionally mapped CUB umbrella. This is a proof of
// a redundant directive, not independent CUB subheader or vector ABI support.
bool IsRedundantFrozenCubUtilInclude(
    clang::SourceManager& sourceManager,
    const std::string& configuredCudaRoot,
    const std::vector<CudaVec3MappedCubInclude>& mappedCubIncludes,
    const clang::FileEntry& skippedFile,
    clang::SourceLocation includeLocation);

// Reject observed uses of the frozen CUDA record ABI that the native DPP
// float3/uint3 spelling cannot preserve. This grants no rewrite authority.
bool ValidateCudaVec3AbiBoundary(clang::ASTContext& context,
                                const std::string& configuredCudaRoot,
                                const std::vector<CudaVec3MappedCubInclude>&
                                    mappedCubIncludes,
                                std::string& error,
                                clang::SourceLocation& location,
                                const CudaVec3LayoutPlan* plan = nullptr);
}
