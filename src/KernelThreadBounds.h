#pragma once
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
namespace clang { class ASTContext; class FunctionDecl; class CallExpr; }
namespace ascify {
struct PureSimtKernelThreadBound {
  const clang::FunctionDecl *kernel = nullptr;
  unsigned maximum = 0;
  // Valid only for a converter-selected explicit annotation; source bounds
  // are never inferred from an undocumented compiler default or ELF tag.
  clang::SourceLocation policyQualifier;
};
struct KernelThreadBoundPlan {
  std::map<const clang::FunctionDecl *, PureSimtKernelThreadBound> kernels;
  std::map<unsigned, unsigned> policyQualifiers;
  std::map<unsigned, std::string> occupancyCallees;
  bool used() const { return !kernels.empty(); }
};
using KernelBoundOwned = std::function<bool(clang::SourceLocation)>;
using KernelBoundQualifier = std::function<bool(llvm::StringRef, clang::SourceLocation)>;
using KernelBoundMacroFree = std::function<bool(llvm::StringRef, clang::SourceLocation)>;
// A default-bound fact is usable only after its owner stages the policy;
// selected-header annotations are fulfilled by the outer atomic transaction.
bool ProvePureSimtKernelThreadBound(
    clang::ASTContext &, const clang::FunctionDecl *, const KernelBoundOwned &,
    const KernelBoundQualifier &, const KernelBoundMacroFree &,
    PureSimtKernelThreadBound &, clang::SourceLocation &, std::string &);
bool ProveOccupancyKernelThreadBounds(
    clang::ASTContext &, const KernelBoundOwned &, const KernelBoundQualifier &,
    const KernelBoundMacroFree &,
    const std::function<bool(const clang::CallExpr *)> &trustedApi,
    const std::function<bool(const clang::CallExpr *)> &closedMacroInvocation,
    const std::function<bool()> &closedHelperInputContext,
    const std::function<bool(const PureSimtKernelThreadBound &)> &stagePolicy,
    bool enabled, KernelThreadBoundPlan &, clang::SourceLocation &, std::string &);
} // namespace ascify
