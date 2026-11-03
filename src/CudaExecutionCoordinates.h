#pragma once

#include <functional>
#include <optional>
#include <set>
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Path.h"

namespace ascify {
// Reuse the complete property/base/getter proof used by the half2 leaf gate.
// The caller additionally supplies its frozen physical SDK/resource proof.
// This authenticates a read, never an explicit getter call or user dim3 field.
struct CudaExecutionCoordinate {
  enum class Builtin { ThreadIndex, BlockIndex, BlockExtent, GridExtent } builtin;
  char axis;
  const clang::PseudoObjectExpr* expression;
  const clang::CallExpr* getter;
};
inline std::optional<CudaExecutionCoordinate> ProveCudaExecutionCoordinate(
    const clang::Expr* input, clang::ASTContext& context,
    const std::function<bool(clang::SourceLocation)>& trusted) {
  using namespace clang;
  const auto* expression=dyn_cast_or_null<PseudoObjectExpr>(input ? input->IgnoreParenImpCasts() : nullptr);
  if (!expression || expression->getBeginLoc().isMacroID() || expression->getEndLoc().isMacroID()) return {};
  const auto* call=dyn_cast<CallExpr>(expression->getResultExpr()->IgnoreParenImpCasts());
  const auto* method=call ? dyn_cast_or_null<CXXMethodDecl>(call->getDirectCallee()) : nullptr;
  const auto* property=dyn_cast<MSPropertyRefExpr>(expression->getSyntacticForm()->IgnoreParenImpCasts());
  if (!method || call->getNumArgs()!=0 || !call->getType()->isSpecificBuiltinType(BuiltinType::UInt) ||
      !property || property->isArrow()) return {};
  const auto axis=property->getPropertyDecl()->getName();
  if ((axis!="x" && axis!="y" && axis!="z") || method->getName()!="__fetch_builtin_"+axis.str()) return {};
  const Expr* base=property->getBaseExpr();
  while (base) {
    if (base->getBeginLoc().isMacroID() || base->getEndLoc().isMacroID()) return {};
    if (const auto* value=dyn_cast<OpaqueValueExpr>(base)) base=value->getSourceExpr();
    else if (const auto* value=dyn_cast<ParenExpr>(base)) base=value->getSubExpr();
    else if (const auto* value=dyn_cast<ImplicitCastExpr>(base)) base=value->getSubExpr();
    else break;
  }
  const auto* ref=dyn_cast_or_null<DeclRefExpr>(base);
  const auto* variable=ref ? dyn_cast<VarDecl>(ref->getDecl()) : nullptr;
  if (!variable || variable->isLocalVarDecl() || !variable->hasGlobalStorage() ||
      variable->getType().isVolatileQualified() || variable->getType()->isAtomicType()) return {};
  class References : public RecursiveASTVisitor<References> {
   public:
    std::set<const VarDecl*> variables;
    bool VisitDeclRefExpr(DeclRefExpr* r) {
      if (const auto* v=dyn_cast<VarDecl>(r->getDecl())) variables.insert(v->getCanonicalDecl());
      return true;
    }
  } references;
  references.TraverseStmt(const_cast<PseudoObjectExpr*>(expression));
  if (references.variables.size()!=1 || !references.variables.count(variable->getCanonicalDecl())) return {};
  const auto name=variable->getName();
  CudaExecutionCoordinate::Builtin builtin;
  if (name=="threadIdx") builtin=CudaExecutionCoordinate::Builtin::ThreadIndex;
  else if (name=="blockIdx") builtin=CudaExecutionCoordinate::Builtin::BlockIndex;
  else if (name=="blockDim") builtin=CudaExecutionCoordinate::Builtin::BlockExtent;
  else if (name=="gridDim") builtin=CudaExecutionCoordinate::Builtin::GridExtent;
  else return {};
  const auto* record=variable->getType()->getAsCXXRecordDecl();
  if (!record || record->getQualifiedNameAsString()!="__cuda_builtin_"+name.str()+"_t" ||
      method->getParent()->getCanonicalDecl()!=record->getCanonicalDecl()) return {};
  auto& source=context.getSourceManager();
  const auto physical=[&](SourceLocation location) {
    const auto expanded=source.getExpansionLoc(location);
    return trusted(source.getSpellingLoc(location)) && trusted(expanded) &&
        source.isInSystemHeader(expanded) &&
        llvm::sys::path::filename(source.getFilename(expanded))=="__clang_cuda_builtin_vars.h";
  };
  for (const auto* d:variable->redecls()) if (!physical(d->getLocation())) return {};
  for (const auto* d:record->redecls()) if (!physical(d->getLocation())) return {};
  for (const auto* d:method->redecls()) if (!physical(d->getLocation())) return {};
  if (!physical(property->getPropertyDecl()->getLocation())) return {};
  return CudaExecutionCoordinate{builtin,axis.front(),expression,call};
}
} // namespace ascify
