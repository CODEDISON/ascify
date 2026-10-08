#pragma once

#include <string>
#include <vector>
#include "clang/Basic/SourceLocation.h"

namespace clang { class ASTContext; }

namespace ascify {
// A literal main-file CUB include whose semantic replacement was accepted.
// Recording the selected physical path does not itself authenticate a provider.
struct CudaVec3MappedCubInclude {
  clang::SourceLocation hashLocation;
  clang::SourceLocation filenameEnd;
  std::string resolvedPath;
};

// Reject observed uses of the frozen CUDA record ABI that the native DPP
// float3/uint3 spelling cannot preserve. This grants no rewrite authority.
bool ValidateCudaVec3AbiBoundary(clang::ASTContext& context,
                                const std::string& configuredCudaRoot,
                                const std::vector<CudaVec3MappedCubInclude>&
                                    mappedCubIncludes,
                                std::string& error,
                                clang::SourceLocation& location);
}
