#pragma once

#include <functional>
#include <set>
#include <string>
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"

namespace clang { class ASTContext; }
namespace ascify {
struct SharedWordAlignmentPlan {
  // Original qualifier locations, consumed by the ordinary token replacement.
  // No independent overlapping edit or array/type/extent rewrite is emitted.
  std::set<unsigned> qualifiers;
  std::set<std::string> contextualHeaders;
  unsigned arrays = 0, helpers = 0;
  bool used() const { return arrays != 0; }
};
bool ProveSharedWordAlignment(
    clang::ASTContext &, const std::function<bool(clang::SourceLocation)> &ownedInput,
    const std::function<bool(clang::SourceLocation)> &trustedSharedQualifier,
    const std::function<bool(llvm::StringRef, clang::SourceLocation)> &macroFree,
    bool enabled, SharedWordAlignmentPlan &, clang::SourceLocation &badLocation,
    std::string &error);
} // namespace ascify
