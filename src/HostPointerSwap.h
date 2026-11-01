#pragma once

#include <functional>
#include <set>
#include <string>
#include <vector>
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"

namespace clang { class ASTContext; class Expr; class CallExpr; }
namespace ascify {
struct HostPointerSwapGuard {
  clang::SourceLocation location;
  std::string text;
};
// Exact pattern nodes only. Concrete instances and inquiry operands never
// consume these facts. Header facts require a new, proved joint root parse.
// Guards close primary BODY instantiation domains, including the empty domain.
// They do not distinguish two spellings of the same C++ type; alias provenance
// is a separate original-source proof, not an extra property of __is_same.
struct HostPointerSwapPlan {
  std::set<const void *> typeLocs;
  std::set<const clang::Expr *> expressions;
  std::vector<HostPointerSwapGuard> guards;
  std::set<std::string> contextualHeaders;
  unsigned patterns = 0, concreteCalls = 0, emptyPatterns = 0;
  bool used() const { return patterns != 0; }
};
bool ProveHostPointerSwaps(clang::ASTContext &context,
    const std::function<bool(clang::SourceLocation)> &ownedInput,
    const std::function<bool(const clang::CallExpr *)> &trustedAllocation,
    const std::function<bool(llvm::StringRef, clang::SourceLocation)> &macroFree,
    bool standardPreprocessing, HostPointerSwapPlan &plan,
    clang::SourceLocation &errorLocation, std::string &error);
} // namespace ascify
