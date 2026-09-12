#pragma once

#include <string>
#include <vector>
#include "clang/Basic/SourceLocation.h"

namespace clang { class ASTContext; }
namespace ascify {
struct FrontendCompatibilityConfig;
struct UniformBlockReductionGuard {
  clang::SourceLocation insertionLocation;
  clang::SourceRange kernelRange;
  std::string text;
};
struct UniformBlockReductionStats {
  unsigned kernels = 0;
  unsigned instantiations = 0;
  unsigned collectives = 0;
  std::vector<UniformBlockReductionGuard> guards;
};

// Mandatory publication gate for the scratch-specific parsing projection.
// This does not admit generic multi-warp thread_block_tile operations.
bool ValidateUniformBlockReduction(
    clang::ASTContext& context, const FrontendCompatibilityConfig& profile,
    UniformBlockReductionStats& stats, std::string& error,
    bool standardPreprocessing);
}  // namespace ascify
