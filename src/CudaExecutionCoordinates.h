#pragma once

#include <functional>
#include <optional>
#include <set>
#include <string>
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Builtins.h"
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
  const auto sdkLocation=[&](SourceLocation location) {
    const auto expanded=source.getExpansionLoc(location);
    return trusted(source.getSpellingLoc(location)) && trusted(expanded) && source.isInSystemHeader(expanded);
  };
  const auto resource=[&](SourceLocation location) {
    const auto expanded=source.getExpansionLoc(location);
    return sdkLocation(location) &&
        llvm::sys::path::filename(source.getFilename(expanded))=="__clang_cuda_builtin_vars.h";
  };
  // CUDA's SDK may forward-declare blockDim/threadIdx through curand using
  // Clang's temporary dim3 aliases. Every redeclaration must remain real SDK.
  for (const auto* d:variable->redecls()) if (!sdkLocation(d->getLocation())) return {};
  for (const auto* d:record->redecls()) if (!resource(d->getLocation())) return {};
  if (!resource(property->getPropertyDecl()->getLocation())) return {};
  const char* intrinsicRegister=nullptr;
  switch (builtin) {
    case CudaExecutionCoordinate::Builtin::ThreadIndex: intrinsicRegister="tid_"; break;
    case CudaExecutionCoordinate::Builtin::BlockIndex: intrinsicRegister="ctaid_"; break;
    case CudaExecutionCoordinate::Builtin::BlockExtent: intrinsicRegister="ntid_"; break;
    case CudaExecutionCoordinate::Builtin::GridExtent: intrinsicRegister="nctaid_"; break;
  }
  const std::string intrinsicName=std::string("__nvvm_read_ptx_sreg_")+intrinsicRegister+axis.str();
  const auto* definition=method->getDefinition();
  const auto* body=definition ? dyn_cast_or_null<CompoundStmt>(definition->getBody()) : nullptr;
  const auto* ret=body && body->size()==1 ? dyn_cast<ReturnStmt>(*body->body_begin()) : nullptr;
  const Expr* result=ret ? ret->getRetValue()->IgnoreParens() : nullptr;
  bool convertedInt=false;
  if (const auto* cast=dyn_cast_or_null<ImplicitCastExpr>(result)) {
    // Clang's actual NVVM sreg builtin returns signed int. The SDK getter
    // converts its 32 raw bits to uint, preserving the CUDA coordinate value.
    if (cast->getCastKind()!=CK_IntegralCast ||
        !cast->getType()->isSpecificBuiltinType(BuiltinType::UInt) ||
        !cast->getSubExpr()->getType()->isSpecificBuiltinType(BuiltinType::Int)) return {};
    convertedInt=true; result=cast->getSubExpr()->IgnoreParens();
  }
  const auto* intrinsic=dyn_cast_or_null<CallExpr>(result);
  const auto* callee=intrinsic ? intrinsic->getDirectCallee() : nullptr;
  if (!callee || intrinsic->getNumArgs()!=0 || callee->getNumParams()!=0 || callee->isVariadic() ||
      !intrinsic->getType()->isSpecificBuiltinType(convertedInt ? BuiltinType::Int : BuiltinType::UInt) ||
      !callee->getReturnType()->isSpecificBuiltinType(convertedInt ? BuiltinType::Int : BuiltinType::UInt) ||
      context.getTypeSize(context.IntTy)!=32 || context.getTypeSize(context.UnsignedIntTy)!=32 || !callee->getBuiltinID() ||
      callee->getName()!=intrinsicName || context.BuiltinInfo.getName(callee->getBuiltinID())!=intrinsicName) return {};
  for (const auto* d:callee->redecls()) if (!resource(d->getLocation())) return {};
  for (const auto* d:method->redecls()) {
    if (resource(d->getLocation())) continue;
    // ## generates the getter identifier in Clang's scratch buffer. Admit
    // that token only after proving the actual single builtin-return body;
    // the physical expansion still belongs to the owned resource header.
    const auto location=d->getLocation(), expanded=source.getExpansionLoc(location);
    if (!location.isMacroID() || !source.isWrittenInScratchSpace(source.getSpellingLoc(location)) ||
        !trusted(expanded) || !source.isInSystemHeader(expanded) ||
        llvm::sys::path::filename(source.getFilename(expanded))!="__clang_cuda_builtin_vars.h") return {};
  }
  return CudaExecutionCoordinate{builtin,axis.front(),expression,call};
}
} // namespace ascify
