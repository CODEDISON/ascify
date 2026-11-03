#pragma once
#include <functional>
#include <set>
#include <string>
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
namespace clang { class ASTContext; class Decl; class CallExpr; class Expr; class NamespaceAliasDecl; }
namespace ascify {
struct DynamicWarpGroupStats {
  unsigned values=0, factories=0, reductions=0, shuffles=0, coordinateQueries=0;
  std::set<const clang::Expr*> scalarExpressions;
  // Exact current-AST nodes only. Future consumers must continue traversing
  // children/inquiries; these sets do not authorize a body or an SDK type.
  std::set<const clang::Expr*> valueExpressions;
  std::set<const clang::Expr*> sourceReferences;
  std::set<const clang::Expr*> directCallees;
  std::set<const clang::Expr*> functorConstructions;
  std::set<const clang::CallExpr*> approvedCalls;
  std::set<const void*> sourceTypeLocations;
  std::set<const clang::NamespaceAliasDecl*> namespaceAliases;
  bool used() const { return values!=0 || factories!=0; }
};
// Current-CompilerInstance witnesses only. The source transaction must also
// authenticate the complete projection, physical SDK and backend token history.
bool ProveDynamicWarpGroups(
    clang::ASTContext& context,
    const std::function<bool(clang::SourceLocation)>& ownedSource,
    const std::function<bool(clang::SourceLocation)>& callerSource,
    const std::function<bool(const clang::Decl*)>& ownedCgDecl,
    const std::function<bool(llvm::StringRef,clang::SourceLocation)>& macroFree,
    bool standardDavC310Vec, DynamicWarpGroupStats& output,
    clang::SourceLocation& errorLocation, std::string& error);
}
