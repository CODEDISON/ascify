#pragma once

#include <set>
#include <string>

#include "clang/Basic/SourceLocation.h"

namespace clang {
class ASTContext;
class CallExpr;
}

namespace ascify {

struct DeviceMemoryRewrite {
  clang::SourceRange range;
  std::string replacement;
  std::string apiName;
};

// Plans a semantic edit only. The caller applies the ordinary macro/conflict
// transaction checks and includes ascify/device_memory_compat.hpp. A false
// result leaves the source operation unchanged; it never forges an address
// space or silently admits a broader memory operation.
bool planDeviceMemoryCall(const clang::CallExpr *call,
                          clang::ASTContext &context,
                          const std::set<unsigned> &trustedSystemFileIds,
                          DeviceMemoryRewrite &result);

} // namespace ascify
