#pragma once

#include <functional>
#include <set>
#include <string>
#include <vector>
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class ASTContext;
class CXXRecordDecl;
class Expr;
class FunctionDecl;
}

namespace ascify {
// Facts belong to exact AST nodes, not a record, family body or dependent type.
// Source qualification and these facts are published together after all proofs.
struct NttpVoidCallPlan {
  std::set<const clang::Expr*> expressions;
  std::vector<clang::SourceLocation> qualifications;
  std::set<std::string> contextualHeaders;
  unsigned patterns = 0;
  unsigned concreteCalls = 0;
  bool used() const { return patterns != 0; }
};

bool ProveNttpVoidCalls(
    clang::ASTContext& context,
    const std::function<bool(clang::SourceLocation)>& ownedInput,
    const std::function<bool(const clang::CXXRecordDecl*)>& trustedCgRecord,
    const std::function<bool(const clang::FunctionDecl*)>& trustedCgFunction,
    const std::function<bool(llvm::StringRef, clang::SourceLocation)>& macroFree,
    bool standardPreprocessing, NttpVoidCallPlan& plan,
    clang::SourceLocation& errorLocation, std::string& error);
} // namespace ascify
