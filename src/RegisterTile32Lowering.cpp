#include "RegisterTile32Lowering.h"

#include <map>
#include <set>
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

namespace ascify {
namespace {
using namespace clang;
const Expr* plain(const Expr* expression) {
  return expression ? expression->IgnoreParenImpCasts() : nullptr;
}
bool tile32(QualType type) {
  if (type.isNull()) return false;
  const auto* record=type.getUnqualifiedType()->getAsCXXRecordDecl();
  const auto* instance=dyn_cast_or_null<ClassTemplateSpecializationDecl>(record);
  if (!instance || instance->getTemplateArgs().size()==0 ||
      instance->getSpecializedTemplate()->getQualifiedNameAsString()!=
          "cooperative_groups::thread_block_tile") return false;
  const auto& size=instance->getTemplateArgs()[0];
  return size.getKind()==TemplateArgument::Integral && size.getAsIntegral()==32;
}
bool partition32(const FunctionDecl* declaration) {
  if (!declaration || declaration->getQualifiedNameAsString()!=
          "cooperative_groups::tiled_partition") return false;
  const auto* args=declaration->getTemplateSpecializationArgs();
  return args && args->size()>0 && args->get(0).getKind()==TemplateArgument::Integral &&
      args->get(0).getAsIntegral()==32;
}
class Inventory : public RecursiveASTVisitor<Inventory> {
 public:
  std::vector<const CallExpr*> calls;
  std::vector<const DeclRefExpr*> references;
  std::vector<const VarDecl*> variables;
  std::vector<const NamedDecl*> names;
  std::vector<TypeLoc> writtenTypes;
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }
  bool VisitCallExpr(CallExpr* call) {
    calls.push_back(call);
    return true;
  }
  bool VisitDeclRefExpr(DeclRefExpr* reference) {
    references.push_back(reference);return true;
  }
  bool VisitVarDecl(VarDecl* variable) { variables.push_back(variable);return true; }
  bool VisitNamedDecl(NamedDecl* declaration) { names.push_back(declaration);return true; }
  bool VisitTypeLoc(TypeLoc type) { writtenTypes.push_back(type);return true; }
};
class Proof {
  ASTContext& context;
  SourceManager& source;
  const std::function<bool(SourceLocation)>& literal;
  const std::function<bool(const Decl*)>& trusted;
  Inventory inventory;
  bool allDecls(const NamedDecl* declaration) const {
    if (!declaration || !trusted(declaration)) return false;
    if (const auto* function=dyn_cast<FunctionDecl>(declaration))
      for (const auto* redeclaration:function->redecls())
        if (!trusted(redeclaration)) return false;
    if (const auto* record=dyn_cast<CXXRecordDecl>(declaration))
      for (const auto* redeclaration:record->redecls())
        if (!trusted(redeclaration)) return false;
    return true;
  }
  bool unsigned32(QualType type) const {
    return !type.isVolatileQualified() &&
        context.hasSameType(type.getUnqualifiedType(),context.UnsignedIntTy) &&
        context.getTypeSize(type)==32;
  }
  bool nativeVector(QualType type) const {
    const auto* record=type.getUnqualifiedType()->getAsCXXRecordDecl();
    if (!record || record->getQualifiedNameAsString()!="uint4" ||
        !record->getDefinition() || !allDecls(record)) return false;
    record=record->getDefinition();
    return record->isStandardLayout() && record->isTriviallyCopyable() &&
        context.getTypeSize(type)==128 && context.getTypeAlign(type)==128;
  }
  const FunctionDecl* owner(const VarDecl* variable) const {
    const auto* function=dyn_cast<FunctionDecl>(variable->getDeclContext());
    if (!function || !function->doesThisDeclarationHaveABody() ||
        function->getTemplatedKind()!=FunctionDecl::TK_NonTemplate ||
        !literal(function->getLocation()) ||
        (!function->hasAttr<CUDADeviceAttr>() && !function->hasAttr<CUDAGlobalAttr>()))
      return nullptr;
    return function;
  }
  bool operation(const DeclRefExpr* reference) const {
    if (!reference->getLocation().isFileID() || !literal(reference->getLocation()))
      return false;
    const Expr* object=reference;
    const MemberExpr* member=nullptr;
    for (;;) {
      const auto parents=context.getParents(*object);
      if (parents.size()!=1) return false;
      if (const auto* paren=parents[0].get<ParenExpr>()) { object=paren;continue; }
      if (const auto* cast=parents[0].get<ImplicitCastExpr>()) {
        if (cast->getSubExpr()!=object ||
            (cast->getCastKind()!=CK_UncheckedDerivedToBase &&
             cast->getCastKind()!=CK_DerivedToBase && cast->getCastKind()!=CK_NoOp)) return false;
        const auto* base=cast->getType().getUnqualifiedType()->getAsCXXRecordDecl();
        const auto* instance=dyn_cast_or_null<ClassTemplateSpecializationDecl>(base);
        if (!instance || !allDecls(instance) || instance->getTemplateArgs().size()==0 ||
            instance->getSpecializedTemplate()->getQualifiedNameAsString()!=
                "cooperative_groups::ascify_detail::tile_register_operations" ||
            instance->getTemplateArgs()[0].getKind()!=TemplateArgument::Integral ||
            instance->getTemplateArgs()[0].getAsIntegral()!=32) return false;
        object=cast;continue;
      }
      member=parents[0].get<MemberExpr>();break;
    }
    if (!member || member->isArrow() || member->getBase()!=object ||
        !literal(member->getMemberLoc()) || !allDecls(member->getMemberDecl())) return false;
    const Expr* callee=member;
    const CallExpr* call=nullptr;
    for (;;) {
      const auto up=context.getParents(*callee);
      if (up.size()!=1) return false;
      if (const auto* cast=up[0].get<ImplicitCastExpr>()) {
        if (cast->getSubExpr()!=callee || cast->getCastKind()!=CK_FunctionToPointerDecay)
          return false;
        callee=cast;continue;
      }
      call=up[0].get<CallExpr>();break;
    }
    if (!call || call->getCallee()!=callee || !literal(call->getExprLoc())) return false;
    const auto* method=dyn_cast_or_null<CXXMethodDecl>(call->getDirectCallee());
    if (!method || !allDecls(method)) return false;
    if (const auto* memberCall=dyn_cast<CXXMemberCallExpr>(call)) {
      if (plain(memberCall->getImplicitObjectArgument())!=reference) return false;
    } else if (!method->isStatic()) return false;
    const auto name=member->getMemberDecl()->getName();
    if (name=="thread_rank" || name=="size" || name=="meta_group_rank" || name=="meta_group_size")
      return call->getNumArgs()==0 && unsigned32(call->getType());
    if (call->getNumArgs()!=2 || !literal(call->getArg(0)->getBeginLoc()) ||
        !literal(call->getArg(1)->getBeginLoc())) return false;
    if (name=="shfl_up")
      return unsigned32(call->getType()) && unsigned32(call->getArg(0)->getType()) &&
          unsigned32(call->getArg(1)->getType());
    if (name!="shfl_xor" || !nativeVector(call->getType()) ||
        !nativeVector(call->getArg(0)->getType())) return false;
    const auto* mask=dyn_cast_or_null<IntegerLiteral>(plain(call->getArg(1)));
    return mask && literal(mask->getLocation()) && mask->getValue().ule(3);
  }
 public:
  Proof(ASTContext& c,const std::function<bool(SourceLocation)>& l,
        const std::function<bool(const Decl*)>& t)
      : context(c),source(c.getSourceManager()),literal(l),trusted(t) {
    inventory.TraverseDecl(context.getTranslationUnitDecl());
  }
  bool run(std::vector<RegisterTile32Factory>& output,std::string& error) {
    std::map<const VarDecl*,const CallExpr*> variables;
    std::set<const Expr*> admittedFactories;
    std::vector<RegisterTile32Factory> staged;
    const auto fail=[&](const char* text){error=text;return false;};
    for (const auto* call:inventory.calls) {
      const auto* callee=call->getDirectCallee();
      if (!partition32(callee)) continue;
      const auto at=call->getExprLoc();
      const bool caller=at.isValid() && (literal(source.getExpansionLoc(at)) ||
          literal(source.getSpellingLoc(at)));
      if (!caller) continue;
      if (!literal(at)) return fail("register tile factory cannot be generated by a caller macro");
      if (!allDecls(callee) || call->getNumArgs()!=1 || !tile32(call->getType()))
        return fail("register tile requires the physical SDK partition32 declaration");
      const auto* reference=dyn_cast_or_null<DeclRefExpr>(plain(call->getCallee()));
      if (!reference || !reference->hasExplicitTemplateArgs() ||
          reference->getNumTemplateArgs()!=1)
        return fail("register tile requires an explicitly written literal width32");
      const auto& argument=reference->getTemplateArgs()[0];
      const Expr* width=argument.getArgument().getKind()==TemplateArgument::Expression ?
          argument.getSourceExpression() : argument.getArgument().getKind()==TemplateArgument::Integral ?
          argument.getSourceIntegralExpression() : nullptr;
      if (const auto* constant=dyn_cast_or_null<ConstantExpr>(width)) width=constant->getSubExpr();
      const auto* number=dyn_cast_or_null<IntegerLiteral>(width);
      if (!number || number->getValue()!=32 || !literal(number->getLocation()))
        return fail("register tile width cannot be a macro, alias or computed constant");
      const auto* parent=dyn_cast_or_null<DeclRefExpr>(plain(call->getArg(0)));
      const auto* block=parent ? parent->getType().getUnqualifiedType()->getAsCXXRecordDecl() : nullptr;
      if (!parent || !literal(parent->getLocation()) || !block ||
          block->getQualifiedNameAsString()!="cooperative_groups::thread_block" ||
          !allDecls(block) || parent->getType().isVolatileQualified())
        return fail("register tile parent requires a physical SDK block lvalue");
      const Expr* initializer=call;
      auto parents=context.getParents(*initializer);
      // Clang represents const-auto class prvalue initialization with exactly
      // one implicit qualification NoOp. No constructor/conversion or cleanup
      // wrapper obtains this factory capability.
      if (parents.size()==1) {
        if (const auto* cast=parents[0].get<ImplicitCastExpr>()) {
          if (cast->getCastKind()!=CK_NoOp || cast->getSubExpr()!=call ||
              !cast->getType().isConstQualified() || cast->getType().isVolatileQualified() ||
              !context.hasSameUnqualifiedType(cast->getType(),call->getType()))
            return fail("register tile initialization contains an unproved conversion");
          initializer=cast;parents=context.getParents(*initializer);
        }
      }
      const auto* variable=parents.size()==1 ? parents[0].get<VarDecl>() : nullptr;
      if (!variable || variable->getInit()!=initializer || !owner(variable) ||
          !variable->isLocalVarDecl() || variable->isStaticLocal() ||
          variable->getTLSKind()!=VarDecl::TLS_None || variable->hasAttrs() ||
          variable->getType()->isReferenceType() || variable->getType().isVolatileQualified() ||
          !variable->getType().isConstQualified() || !variable->getTypeSourceInfo() ||
          !variable->getTypeSourceInfo()->getType()->getContainedAutoType())
        return fail("register tile requires a const auto local initialized by its direct factory");
      const auto calleeRange=reference->getSourceRange();
      const auto begin=calleeRange.getBegin();
      const auto end=Lexer::getLocForEndOfToken(calleeRange.getEnd(),0,source,context.getLangOpts());
      if (!begin.isFileID() || !end.isFileID() || !literal(begin) || !literal(end) ||
          source.getFileID(begin)!=source.getFileID(end) || !variables.emplace(variable,call).second)
        return fail("register tile factory has no unique physical token range");
      admittedFactories.insert(reference);staged.push_back({begin,end,call});
    }
    if (!staged.empty()) {
      for (const auto* declaration:inventory.names) {
        if (!declaration->getIdentifier() || declaration->getName()!="ascify_cg") continue;
        const auto at=declaration->getLocation();
        if (at.isValid() && (literal(source.getExpansionLoc(at)) ||
            literal(source.getSpellingLoc(at))))
          return fail("register tile adapter namespace is reserved for the owned target header");
      }
    }
    for (const auto* reference:inventory.references) {
      const auto* variable=dyn_cast<VarDecl>(reference->getDecl());
      if (variable && variables.count(variable) && !operation(reference))
        return fail("register tile view escapes, is copied, reflected, or uses an unsupported operation");
      const auto at=reference->getLocation();
      const bool caller=at.isValid() && (literal(source.getExpansionLoc(at)) ||
          literal(source.getSpellingLoc(at)));
      if (!caller) continue;
      if (const auto* function=dyn_cast<FunctionDecl>(reference->getDecl()))
        if (partition32(function) && !admittedFactories.count(reference))
          return fail("register tile factory escapes its direct const-auto initialization");
    }
    for (const auto* variable:inventory.variables) {
      const auto at=variable->getLocation();
      if (!at.isValid() || (!literal(source.getExpansionLoc(at)) &&
          !literal(source.getSpellingLoc(at)))) continue;
      auto type=variable->getType();
      while (type->isReferenceType() || type->isPointerType()) type=type->getPointeeType();
      if (tile32(type) && !variables.count(variable))
        return fail("register tile values cannot enter through another local, parameter or global");
    }
    for (const auto type:inventory.writtenTypes) {
      const auto at=type.getBeginLoc();
      if (!at.isValid() || (!literal(source.getExpansionLoc(at)) &&
          !literal(source.getSpellingLoc(at)))) continue;
      if (tile32(type.getType()) && !type.getType()->getContainedAutoType())
        return fail("register tile representation cannot be explicitly named or reflected");
    }
    output=std::move(staged);return true;
  }
};
}  // namespace

bool ProveRegisterTile32Factories(ASTContext& context,
    const std::function<bool(SourceLocation)>& ownedLiteral,
    const std::function<bool(const Decl*)>& ownedCgOrSdkDecl,
    std::vector<RegisterTile32Factory>& output,std::string& error) {
  output.clear();error.clear();
  return Proof(context,ownedLiteral,ownedCgOrSdkDecl).run(output,error);
}
}  // namespace ascify
