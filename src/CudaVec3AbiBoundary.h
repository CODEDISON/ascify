#pragma once

#include <string>
#include "clang/Basic/SourceLocation.h"

namespace clang { class ASTContext; }

namespace ascify {
// Reject observed uses of the frozen CUDA record ABI that the native DPP
// float3/uint3 spelling cannot preserve. This grants no rewrite authority.
bool ValidateCudaVec3AbiBoundary(clang::ASTContext& context,
                                std::string& error,
                                clang::SourceLocation& location);
}
