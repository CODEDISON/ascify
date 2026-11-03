#include "Scalar64Lowering.h"

#include <map>
#include <set>
#include <cstdint>
#include <utility>
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
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
const FunctionTemplateDecl* family(const FunctionDecl* function) {
  if (!function) return nullptr;
  const auto* result = function->getPrimaryTemplate();
  if (!result) result = function->getDescribedFunctionTemplate();
  return result ? result->getCanonicalDecl() : nullptr;
}
bool device(const FunctionDecl* function) {
  return function && (function->hasAttr<CUDADeviceAttr>() ||
                      function->hasAttr<CUDAGlobalAttr>());
}
bool doubleType(QualType type, ASTContext& context) {
  return !type.isNull() && context.hasSameUnqualifiedType(type, context.DoubleTy);
}
// Refusal only. These are compiler builtin arithmetic wrappers, not records,
// pointers, arrays or a sizeof-based approximation of an arithmetic domain.
bool wrappedDoubleArithmeticType(QualType type, ASTContext& context) {
  if (type.isNull()) return false;
  type = type.getCanonicalType();
  QualType element;
  if (const auto* atomic = type->getAs<AtomicType>()) element = atomic->getValueType();
  else if (const auto* complex = type->getAs<ComplexType>()) element = complex->getElementType();
  else if (const auto* vector = type->getAs<VectorType>()) element = vector->getElementType();
  else return false;
  return doubleType(element, context) || wrappedDoubleArithmeticType(element, context);
}
bool addType(QualType type, ASTContext& context) {
  return !type.isNull() && (context.hasSameUnqualifiedType(type, context.IntTy) ||
      context.hasSameUnqualifiedType(type, context.FloatTy) || doubleType(type, context));
}
bool wideType(QualType type, ASTContext& context) {
  if (type.isNull() || type->isDependentType() || type.isVolatileQualified()) return false;
  type = type.getCanonicalType().getUnqualifiedType();
  const auto* builtin = type->getAs<BuiltinType>();
  if (!builtin || context.getTypeSize(type) != 64) return false;
  switch (builtin->getKind()) {
  case BuiltinType::Long: case BuiltinType::ULong:
  case BuiltinType::LongLong: case BuiltinType::ULongLong:
  case BuiltinType::Double: return true;
  default: return false;
  }
}
bool transportType(QualType type, ASTContext& context) {
  return addType(type, context) ||
      context.hasSameUnqualifiedType(type, context.UnsignedIntTy) || wideType(type, context);
}

struct Operation { const FunctionDecl* owner; const BinaryOperator* expression; };
struct Invocation { const FunctionDecl* owner; const CallExpr* expression; };

class Inventory : public RecursiveASTVisitor<Inventory> {
 public:
  const FunctionDecl* current = nullptr;
  std::vector<const FunctionDecl*> functions;
  std::vector<Operation> operations;
  std::vector<Invocation> calls;
  std::vector<const DeclRefExpr*> references;
  std::vector<const UnresolvedLookupExpr*> lookups;
  std::set<const Expr*> directCallees;
  std::map<const FunctionDecl*, std::vector<const Expr*>> expressions;
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }
  bool TraverseDecl(Decl* declaration) {
    const auto* saved = current;
    if (const auto* function = dyn_cast_or_null<FunctionDecl>(declaration)) {
      current = function;
      functions.push_back(function);
    }
    const bool result = RecursiveASTVisitor<Inventory>::TraverseDecl(declaration);
    current = saved;
    return result;
  }
  bool VisitExpr(Expr* expression) {
    if (current) expressions[current].push_back(expression);
    return true;
  }
  bool VisitBinaryOperator(BinaryOperator* expression) {
    if (current) operations.push_back({current, expression});
    return true;
  }
  bool VisitCallExpr(CallExpr* expression) {
    if (current) calls.push_back({current, expression});
    directCallees.insert(expression->getCallee()->IgnoreParenImpCasts());
    return true;
  }
  bool VisitDeclRefExpr(DeclRefExpr* expression) {
    if (isa<FunctionDecl>(expression->getDecl())) references.push_back(expression);
    return true;
  }
  bool VisitUnresolvedLookupExpr(UnresolvedLookupExpr* expression) {
    lookups.push_back(expression); return true;
  }
};

class Proof {
  ASTContext& context;
  SourceManager& source;
  const std::function<bool(SourceLocation)>& owned;
  const std::function<bool(SourceLocation)>& sdk;
  const std::function<bool(const FunctionDecl*)>& shuffle;
  const std::function<bool(const FunctionDecl*)>& reduce;
  const std::function<bool(llvm::StringRef, SourceLocation)>& macroFree;
  Scalar64Plan& plan;
  SourceLocation& badLocation;
  std::string& error;
  Inventory inventory;
  std::set<const FunctionDecl*> routes, mathRoutes, opaque;
  std::map<const FunctionDecl*, std::set<const FunctionDecl*>> edges;
  std::map<const FunctionTemplateDecl*, const TemplateTypeParmDecl*> parameters;
  std::set<const FunctionTemplateDecl*> arithmeticFamilies;
  std::map<const FunctionTemplateDecl*, const TemplateTypeParmDecl*> reduceGroups;
  std::map<const FunctionDecl*, std::set<const CallExpr*>> exactReductionCalls;
  std::map<unsigned, std::vector<Operation>> sites;
  std::map<unsigned, std::string> prefixes, suffixes;
  std::map<unsigned, Scalar64Edit> tokens;
  std::set<unsigned> editedOperators;

  bool fail(SourceLocation location, const std::string& text) {
    if (error.empty()) { badLocation = location; error = text; }
    return false;
  }
  bool input(const FunctionDecl* function) const {
    return device(function) && function->getLocation().isValid() &&
        !sdk(source.getSpellingLoc(function->getLocation()));
  }
  bool literal(SourceLocation location) const {
    return location.isFileID() && source.isWrittenInMainFile(location) && owned(location);
  }
  bool zero(const Expr* expression) const {
    if (!expression || expression->HasSideEffects(context) ||
        !expression->getType()->isIntegerType() || expression->getType()->isEnumeralType()) return false;
    Expr::EvalResult result;
    return expression->EvaluateAsInt(result, context) && result.Val.getInt() == 0;
  }
  bool isShuffle(const CallExpr* call) const {
    const auto* callee = call->getDirectCallee();
    if (!callee || !callee->getIdentifier() || !shuffle(callee)) return false;
    const auto name = callee->getName();
    return name == "__shfl_sync" || name == "__shfl_down_sync" ||
           name == "shfl" || name == "shfl_down";
  }
  bool reductionGroup(QualType type) const {
    if (type.isNull() || type.isVolatileQualified() || type->isReferenceType()) return false;
    type = type.getUnqualifiedType();
    const auto* record = type->getAsCXXRecordDecl();
    const auto* item = dyn_cast_or_null<ClassTemplateSpecializationDecl>(record);
    if (!item || item->getSpecializationKind() == TSK_ExplicitSpecialization ||
        item->getTemplateArgs().size() != 2 || !record->getDefinition()) return false;
    for (const auto* d : record->redecls()) if (!sdk(source.getSpellingLoc(d->getLocation()))) return false;
    for (const auto* d : item->getSpecializedTemplate()->redecls())
      if (!sdk(source.getSpellingLoc(d->getLocation()))) return false;
    const auto& args=item->getTemplateArgs();
    if (args[0].getKind()!=TemplateArgument::Integral ||
        args[0].getAsIntegral().isNegative() || args[0].getAsIntegral().getActiveBits()>16) return false;
    const auto size=args[0].getAsIntegral().getZExtValue();
    const auto name=item->getSpecializedTemplate()->getQualifiedNameAsString();
    if (name=="cooperative_groups::thread_block_tile") {
      if (size!=32 || args[1].getKind()!=TemplateArgument::Type) return false;
      const auto parent=args[1].getAsType();
      if (parent->isVoidType()) return true;
      const auto* block=parent->getAsCXXRecordDecl();
      if (!block || block->getQualifiedNameAsString()!="cooperative_groups::thread_block") return false;
      for (const auto* d : block->redecls()) if (!sdk(source.getSpellingLoc(d->getLocation()))) return false;
      return true;
    }
    if (name!="cooperative_groups::scratch_thread_block_tile" ||
        !(size==32 || size==64 || size==128 || size==256 || size==512) ||
        args[1].getKind()!=TemplateArgument::Integral || args[1].getAsIntegral().isNegative() ||
        args[1].getAsIntegral().getActiveBits()>16) return false;
    const auto block=args[1].getAsIntegral().getZExtValue();
    return block>=32 && block<=1024 && block%size==0;
  }
  bool exactReduction(const CallExpr* call) const {
    const auto* callee=call ? call->getDirectCallee() : nullptr;
    if (!callee || !reduce(callee) || call->getNumArgs()!=3 ||
        !addType(call->getType(),context) || !reductionGroup(call->getArg(0)->getType())) return false;
    const auto* group=dyn_cast<DeclRefExpr>(plain(call->getArg(0)));
    const auto* variable=group ? dyn_cast<VarDecl>(group->getDecl()) : nullptr;
    if (!variable || (!isa<ParmVarDecl>(variable) && !variable->isLocalVarDecl()) ||
        call->getArg(0)->HasSideEffects(context) ||
        !context.hasSameUnqualifiedType(call->getArg(1)->getType(),call->getType())) return false;
    const auto* operation=dyn_cast<CXXConstructExpr>(plain(call->getArg(2)));
    return operation && operation->getNumArgs()==0 &&
        context.hasSameUnqualifiedType(operation->getType(),callee->getParamDecl(2)->getType()) &&
        sdk(source.getSpellingLoc(operation->getConstructor()->getLocation()));
  }
  bool reduceForwarderPattern(const FunctionTemplateDecl* primary,
                              const TemplateTypeParmDecl*& group) const {
    const auto* params=primary->getTemplateParameters();
    const auto* f=primary->getTemplatedDecl();
    if (params->size()!=2 || f->getNumParams()!=2 || f->hasAttr<CUDAGlobalAttr>() ||
        !f->hasAttr<CUDADeviceAttr>() || !literal(f->getLocation())) return false;
    for (const auto* d : f->redecls()) {
      if (!literal(d->getLocation())) return false;
      for (const auto* a : d->attrs()) if (!isa<CUDADeviceAttr>(a)) return false;
      for (const auto* p : d->parameters()) if (p->hasAttrs()) return false;
    }
    for (const auto* d : primary->redecls()) if (d->hasAttrs()) return false;
    for (const auto* p : *params) if (p->hasAttrs()) return false;
    group=dyn_cast<TemplateTypeParmDecl>(params->getParam(1));
    if (!group || !group->getIdentifier() || group->isParameterPack() ||
        group->hasDefaultArgument() || group->hasAttrs()) return false;
    const auto* first=f->getReturnType()->getAs<TemplateTypeParmType>();
    const auto* scalar=f->getParamDecl(0)->getType()->getAs<TemplateTypeParmType>();
    const auto second=f->getParamDecl(1)->getType();
    const auto* pointee=second->isLValueReferenceType() ?
        second->getPointeeType()->getAs<TemplateTypeParmType>() : nullptr;
    if (!first || !scalar || !pointee || first->getIndex()!=0 || scalar->getIndex()!=0 ||
        pointee->getDecl()!=group || second->getPointeeType().isConstQualified()) return false;
    const auto* body=dyn_cast_or_null<CompoundStmt>(f->getBody());
    if (!body || body->size()!=1) return false;
    const auto* ret=dyn_cast<ReturnStmt>(*body->body_begin());
    const auto* call=ret ? dyn_cast_or_null<CallExpr>(plain(ret->getRetValue())) : nullptr;
    if (!call || call->getNumArgs()!=3 || !literal(call->getExprLoc())) return false;
    const auto* a=dyn_cast<DeclRefExpr>(plain(call->getArg(0)));
    const auto* b=dyn_cast<DeclRefExpr>(plain(call->getArg(1)));
    if (!a || !b || a->getDecl()!=f->getParamDecl(1) || b->getDecl()!=f->getParamDecl(0)) return false;
    const auto* lookup=dyn_cast<UnresolvedLookupExpr>(plain(call->getCallee()));
    if (!lookup || !lookup->getName().getAsIdentifierInfo() ||
        lookup->getName().getAsIdentifierInfo()->getName()!="reduce") return false;
    unsigned candidates=0;
    for (const auto* d : lookup->decls()) {
      const auto* t=dyn_cast<FunctionTemplateDecl>(d);
      if (!t || !reduce(t->getTemplatedDecl())) return false;
      ++candidates;
    }
    // Every concrete specialization below must select exactly the trusted
    // plus<T> overload. The written pattern retains no other statements.
    return candidates!=0;
  }
  bool wideShuffle(const CallExpr* call) const {
    return isShuffle(call) && wideType(call->getType(), context);
  }
  bool shuffleControls(const CallExpr* call) {
    const bool member = isa<CXXMemberCallExpr>(call);
    const auto* callee = call->getDirectCallee();
    const bool index = callee->getName() == "shfl" || callee->getName() == "__shfl_sync";
    const auto known = [&](const Expr* expression, int64_t& value) {
      if (const auto* argument = dyn_cast<CXXDefaultArgExpr>(expression)) expression = argument->getExpr();
      Expr::EvalResult result;
      if (expression->HasSideEffects(context) || !expression->EvaluateAsInt(result, context) ||
          result.Val.getInt().getBitWidth() > 64) return false;
      value = result.Val.getInt().getSExtValue();
      return true;
    };
    int64_t value = 0;
    if (!member && known(call->getArg(3), value) && value != 32)
      return fail(call->getExprLoc(), "scalar64 shuffle admits only width 32");
    if (index && known(call->getArg(member ? 1 : 2), value) && (value < 0 || value >= 32))
      return fail(call->getExprLoc(), "scalar64 index shuffle requires a source lane in [0,32)");
    if (index && !member && known(call->getArg(0), value) &&
        static_cast<uint32_t>(value) != UINT32_MAX)
      return fail(call->getExprLoc(), "scalar64 index shuffle retains the full-warp mask contract");
    return true;
  }
  bool observation(const Expr* expression) const {
    std::vector<DynTypedNode> pending = {DynTypedNode::create(*expression)};
    unsigned visited = 0;
    while (!pending.empty()) {
      if (++visited > 128) return true;
      const auto node = pending.back(); pending.pop_back();
      for (const auto& parent : context.getParents(node)) {
        if (parent.get<UnaryExprOrTypeTraitExpr>() || parent.get<CXXNoexceptExpr>() ||
            parent.get<CXXTypeidExpr>() || parent.get<TypeTraitExpr>() ||
            parent.get<ArrayTypeTraitExpr>() || parent.get<ExpressionTraitExpr>() ||
            parent.get<StaticAssertDecl>() || parent.get<DecltypeType>()) return true;
        if (const auto* variable = parent.get<VarDecl>())
          if (variable->isConstexpr() || isa<ParmVarDecl>(variable) ||
              (variable->hasGlobalStorage() && variable->hasInit() &&
               variable->hasConstantInitialization())) return true;
        if (const auto* location = parent.get<TypeLoc>()) {
          const auto type = location->getType();
          if (!type.isNull() && (isa<DecltypeType>(type.getTypePtr()) ||
                                 isa<TypeOfExprType>(type.getTypePtr()))) return true;
        }
        if (const auto* type = parent.get<Type>())
          if (isa<DecltypeType>(type) || isa<TypeOfExprType>(type) ||
              (isa<FunctionProtoType>(type) &&
               cast<FunctionProtoType>(type)->getNoexceptExpr())) return true;
        if (parent.get<FunctionDecl>()) continue;
        pending.push_back(parent);
      }
    }
    return false;
  }
  bool signatureType(QualType type, const TemplateTypeParmDecl* parameter) const {
    if (type.isNull() || type.isVolatileQualified() || type->isAtomicType()) return false;
    if (type->isVoidType()) return true;
    if (const auto* pointer = type->getAs<PointerType>())
      return signatureType(pointer->getPointeeType(), parameter);
    if (const auto* reference = type->getAs<ReferenceType>())
      return signatureType(reference->getPointeeType(), parameter);
    if (const auto* scalar = type->getAs<TemplateTypeParmType>())
      return parameter && scalar->getDecl() == parameter;
    return !type->isDependentType() && type->isBuiltinType() && type->isArithmeticType();
  }
  bool templateProof(const FunctionTemplateDecl* primary) {
    if (parameters.count(primary)) return true;
    const auto* list = primary->getTemplateParameters();
    const auto* function = primary->getTemplatedDecl();
    if (list->size() == 0 || list->getRequiresClause() ||
        function->getTrailingRequiresClause())
      return fail(function->getLocation(), "open/constrained scalar64 template is not proven");
    const auto* parameter = dyn_cast<TemplateTypeParmDecl>(list->getParam(0));
    if (!parameter || !parameter->getIdentifier() || parameter->isParameterPack() ||
        parameter->hasDefaultArgument())
      return fail(function->getLocation(), "scalar64 requires one named closed builtin type parameter");
    const TemplateTypeParmDecl* reduceGroup=nullptr;
    const bool cgForwarder=reduceForwarderPattern(primary,reduceGroup);
    for (unsigned index = 1; index < list->size(); ++index) {
      if (cgForwarder && index==1) continue;
      const auto* value = dyn_cast<NonTypeTemplateParmDecl>(list->getParam(index));
      if (!value || value->isParameterPack() || value->hasDefaultArgument() ||
          !value->getType()->isBuiltinType() || !value->getType()->isIntegerType())
        return fail(function->getLocation(), "additional scalar64 parameters must be fixed builtin NTTPs");
    }
    if (function->getPreviousDecl() || function->getMostRecentDecl() != function ||
        !signatureType(function->getReturnType(), parameter))
      return fail(function->getLocation(), "scalar64 template redeclaration/return type is not closed");
    for (const auto* formal : function->parameters())
      if (formal->hasDefaultArg() || (!signatureType(formal->getType(), parameter) &&
          !(cgForwarder && formal==function->getParamDecl(1))))
        return fail(formal->getLocation(), "scalar64 signature contains a class/conversion/default argument");
    unsigned concrete = 0;
    const bool math = arithmeticFamilies.count(primary) != 0;
    for (const auto* instance : primary->specializations()) {
      if (instance->getTemplateSpecializationKind() == TSK_ExplicitSpecialization)
        return fail(instance->getLocation(), "scalar64 explicit specialization can bypass the closed primary guard");
      const auto* args = instance->getTemplateSpecializationArgs();
      if (!args || args->size() != list->size() || args->get(0).getKind() != TemplateArgument::Type)
        return fail(instance->getLocation(), "unproved scalar64 template specialization");
      const auto type = args->get(0).getAsType();
      if (!(math ? addType(type, context) : transportType(type, context)))
        return fail(instance->getLocation(), "scalar64 template has an unproved builtin dtype instance");
      for (unsigned index = 1; index < args->size(); ++index)
        if (cgForwarder && index==1) {
          if (args->get(index).getKind()!=TemplateArgument::Type ||
              !reductionGroup(args->get(index).getAsType()))
            return fail(instance->getLocation(), "CG reduce forwarder has an unproved group instance");
          const auto* body=dyn_cast_or_null<CompoundStmt>(instance->getBody());
          const auto* ret=body && body->size()==1 ? dyn_cast<ReturnStmt>(*body->body_begin()) : nullptr;
          const auto* call=ret ? dyn_cast_or_null<CallExpr>(plain(ret->getRetValue())) : nullptr;
          if (!call || !exactReduction(call))
            return fail(instance->getLocation(), "CG reduce forwarder has an unproved concrete reduce call");
        } else if (args->get(index).getKind() != TemplateArgument::Integral)
          return fail(instance->getLocation(), "scalar64 template has an unresolved NTTP instance");
      ++concrete;
    }
    if (!concrete) return fail(function->getLocation(), "open scalar64 template has no concrete instance");
    // Builtin scalar/pointer arguments have no associated class namespaces.
    // Inspect every reference, including unused patterns and overload escapes.
    for (const auto* reference : inventory.references)
      if (family(dyn_cast<FunctionDecl>(reference->getDecl())) == primary &&
          !inventory.directCallees.count(reference))
        return fail(reference->getLocation(), "scalar64 template address/indirect escape is not admitted");
    for (const auto* lookup : inventory.lookups) {
      bool mentions = false; unsigned count = 0;
      for (const auto* declaration : lookup->decls()) {
        if (const auto* shadow = dyn_cast<UsingShadowDecl>(declaration)) declaration = shadow->getTargetDecl();
        const auto* candidate = dyn_cast<FunctionTemplateDecl>(declaration);
        mentions |= candidate && candidate->getCanonicalDecl() == primary;
        ++count;
      }
      if (mentions && (count != 1 || !inventory.directCallees.count(lookup)))
        return fail(lookup->getNameLoc(), "scalar64 overload/ADL lookup is not closed");
    }
    parameters[primary] = parameter;
    plan.scalarParameters.insert(parameter);
    if (cgForwarder) { reduceGroups[primary]=reduceGroup; plan.reduceGroupParameters.insert(reduceGroup); }
    if (const auto* spelling = function->getTypeSourceInfo()) {
      const auto location = spelling->getTypeLoc();
      if (!location.getAs<FunctionProtoTypeLoc>().isNull())
        plan.scalarSignatureTypeLocs.insert(location.getOpaqueData());
    }
    return true;
  }
  bool pureInteger(const Expr* expression, unsigned depth = 0) const {
    if (!expression || depth > 32 || expression->getType().isVolatileQualified()) return false;
    if (const auto* cast = dyn_cast<ImplicitCastExpr>(expression)) return pureInteger(cast->getSubExpr(), depth + 1);
    expression = expression->IgnoreParens();
    if (!expression->getType()->isIntegerType() || expression->getType()->isEnumeralType()) return false;
    if (isa<IntegerLiteral>(expression) || isa<CXXBoolLiteralExpr>(expression)) return true;
    if (const auto* ref = dyn_cast<DeclRefExpr>(expression)) {
      const auto* variable = dyn_cast<VarDecl>(ref->getDecl());
      return variable && (isa<ParmVarDecl>(variable) || variable->isLocalVarDecl()) &&
          !variable->getType().isVolatileQualified() && !variable->getType()->isAtomicType();
    }
    if (const auto* binary = dyn_cast<BinaryOperator>(expression)) {
      const auto op = binary->getOpcode();
      return (op == BO_Add || op == BO_Sub || op == BO_Mul || op == BO_And ||
              op == BO_Or || op == BO_Xor || op == BO_Shl || op == BO_Shr) &&
          pureInteger(binary->getLHS(), depth + 1) && pureInteger(binary->getRHS(), depth + 1);
    }
    return false;
  }
  bool lvalue(const Expr* expression, unsigned depth = 0) const {
    if (!expression || depth > 32) return false;
    expression = expression->IgnoreParens();
    const auto type = expression->getType();
    if (type.isVolatileQualified() || type->isAtomicType()) return false;
    if (const auto* ref = dyn_cast<DeclRefExpr>(expression)) {
      const auto* variable = dyn_cast<VarDecl>(ref->getDecl());
      return variable && (isa<ParmVarDecl>(variable) || variable->isLocalVarDecl()) &&
          !variable->getType().isVolatileQualified() && !variable->getType()->isAtomicType();
    }
    if (const auto* array = dyn_cast<ArraySubscriptExpr>(expression)) {
      const auto* base = dyn_cast<DeclRefExpr>(plain(array->getBase()));
      const auto* variable = base ? dyn_cast<VarDecl>(base->getDecl()) : nullptr;
      return variable && (isa<ParmVarDecl>(variable) || variable->isLocalVarDecl()) &&
          !variable->getType().isVolatileQualified() &&
          (variable->getType()->isPointerType() || variable->getType()->isArrayType()) &&
          pureInteger(array->getIdx(), depth + 1);
    }
    if (const auto* unary = dyn_cast<UnaryOperator>(expression))
      return unary->getOpcode() == UO_Deref &&
          isa<DeclRefExpr>(plain(unary->getSubExpr())) && lvalue(plain(unary->getSubExpr()), depth + 1);
    return false;
  }
  bool scalarValue(const Expr* expression, unsigned depth = 0) const {
    if (!expression || depth > 64 || expression->getType().isVolatileQualified() ||
        expression->getType()->isAtomicType()) return false;
    if (const auto* cast = dyn_cast<ImplicitCastExpr>(expression)) {
      switch (cast->getCastKind()) {
      case CK_LValueToRValue: case CK_NoOp: return scalarValue(cast->getSubExpr(), depth + 1);
      case CK_IntegralToFloating: return zero(cast->getSubExpr());
      default: return false;
      }
    }
    expression = expression->IgnoreParens();
    if (isa<FloatingLiteral>(expression) || isa<IntegerLiteral>(expression)) return true;
    if (lvalue(expression)) return true;
    if (const auto* binary = dyn_cast<BinaryOperator>(expression))
      return binary->getOpcode() == BO_Add && scalarValue(binary->getLHS(), depth + 1) &&
          scalarValue(binary->getRHS(), depth + 1);
    if (const auto* call = dyn_cast<CallExpr>(expression)) {
      if (!isShuffle(call) || observation(call)) return false;
      const unsigned valueIndex = isa<CXXMemberCallExpr>(call) ? 0U : 1U;
      for (unsigned index = 0; index < call->getNumArgs(); ++index) {
        const auto* argument = call->getArg(index);
        if (isa<CXXDefaultArgExpr>(argument)) continue;
        if (!(index == valueIndex ? scalarValue(argument, depth + 1) : pureInteger(argument)))
          return false;
      }
      if (const auto* member = dyn_cast<CXXMemberCallExpr>(call))
        if (!isa<DeclRefExpr>(plain(member->getImplicitObjectArgument()))) return false;
      return true;
    }
    return false;
  }
  bool bodyContract(const FunctionDecl* function) {
    // The published source retains unparsed conditional text. Do not rewrite
    // a body whose target preprocessing could expose a different operator or
    // type family than the AST just proved. Raw tokens distinguish a real
    // directive/pragma from the same characters in a comment or string.
    if (const auto* body = function->getBody()) {
      auto cursor = body->getBeginLoc();
      const auto end = Lexer::getLocForEndOfToken(body->getEndLoc(), 0, source, context.getLangOpts());
      if (!literal(cursor) || !literal(end))
        return fail(body->getBeginLoc(), "scalar64 route requires a literal owned function body");
      while (source.getFileOffset(cursor) < source.getFileOffset(end)) {
        Token token;
        if (Lexer::getRawToken(cursor, token, source, context.getLangOpts(), true))
          return fail(cursor, "scalar64 body token provenance is not proven");
        const auto spelling = Lexer::getSpelling(token, source, context.getLangOpts());
        if (token.is(tok::hash) || token.is(tok::hashhash) ||
            spelling == "_Pragma" || spelling == "__pragma")
          return fail(token.getLocation(), "scalar64 body preprocessing/pragma alternatives are not admitted");
        const auto next = Lexer::getLocForEndOfToken(token.getLocation(), 0, source, context.getLangOpts());
        if (next.isInvalid() || source.getFileOffset(next) <= source.getFileOffset(cursor))
          return fail(cursor, "scalar64 body token range is not proven");
        cursor = next;
      }
    }
    for (const auto* expression : inventory.expressions[function]) {
      if (doubleType(expression->getType(), context) &&
          (expression->getType().isVolatileQualified() || expression->getType()->isAtomicType()))
        return fail(expression->getExprLoc(), "scalar64 route does not admit volatile/atomic binary64 access");
      if (const auto* call = dyn_cast<CallExpr>(expression)) {
        bool fp64 = doubleType(call->getType(), context);
        for (const auto* argument : call->arguments()) fp64 |= doubleType(argument->getType(), context);
        const auto* callee = call->getDirectCallee();
        if (fp64 && !isShuffle(call) && !exactReduction(call) &&
            (!callee || !input(callee) || !callee->hasBody() ||
             !routes.count(callee->getCanonicalDecl())))
          return fail(call->getExprLoc(), "scalar64 route calls an unproved binary64 operation/conversion");
      }
      if (const auto* binary = dyn_cast<BinaryOperator>(expression)) {
        const bool fp64 = doubleType(binary->getType(), context) ||
            doubleType(binary->getLHS()->getType(), context) ||
            doubleType(binary->getRHS()->getType(), context);
        if (fp64 && binary->getOpcode() != BO_Add &&
            binary->getOpcode() != BO_AddAssign && binary->getOpcode() != BO_Assign)
          return fail(binary->getOperatorLoc(), "scalar64 route contains unsupported binary64 arithmetic/comparison");
      }
      if (const auto* unary = dyn_cast<UnaryOperator>(expression))
        if (doubleType(unary->getType(), context) && unary->getOpcode() != UO_Deref)
          return fail(unary->getOperatorLoc(), "scalar64 route contains unsupported binary64 unary arithmetic");
      if (const auto* cast = dyn_cast<CastExpr>(expression)) {
        const bool fp64 = doubleType(cast->getType(), context) ||
            doubleType(cast->getSubExpr()->getType(), context);
        if (!fp64) continue;
        switch (cast->getCastKind()) {
        case CK_LValueToRValue: case CK_NoOp: break;
        case CK_IntegralToFloating:
          if (zero(cast->getSubExpr())) break;
          return fail(cast->getExprLoc(), "scalar64 route does not implement runtime integer-to-binary64 conversion");
        default:
          return fail(cast->getExprLoc(), "scalar64 route contains an unproved binary64 conversion");
        }
      }
    }
    return true;
  }
  bool statement(const BinaryOperator* operation) const {
    const auto parents = context.getParents(*operation);
    if (parents.size() != 1) return false;
    if (parents[0].get<CompoundStmt>()) return true;
    if (const auto* branch = parents[0].get<IfStmt>())
      return branch->getThen() == operation || branch->getElse() == operation;
    if (const auto* loop = parents[0].get<ForStmt>()) return loop->getBody() == operation;
    if (const auto* loop = parents[0].get<WhileStmt>()) return loop->getBody() == operation;
    return false;
  }
  bool injectedTokens(SourceLocation location, const std::string& typeName) {
    for (const auto* name : {"ascify", "detail", "scalar_add_rne", "scalar_add_assign_rne",
                            "static_assert", "__is_same", "sizeof", "int", "float", "double", "long", "unsigned"})
      if (!macroFree(name, location)) return fail(location, std::string("input macro changes scalar64 token '") + name + "'");
    return typeName.empty() || macroFree(typeName, location) ||
        fail(location, "input macro changes scalar64 type parameter");
  }
  bool editSite(const std::vector<Operation>& occurrences) {
    const Operation* selected = nullptr;
    for (const auto& occurrence : occurrences)
      if (occurrence.owner->getDescribedFunctionTemplate()) { selected = &occurrence; break; }
    if (!selected) selected = &occurrences.front();
    const auto* operation = selected->expression;
    const auto begin = operation->getBeginLoc(), op = operation->getOperatorLoc();
    const auto end = Lexer::getLocForEndOfToken(operation->getEndLoc(), 0, source, context.getLangOpts());
    if (!literal(begin) || !literal(op) || !literal(end) ||
        source.getFileID(begin) != source.getFileID(end))
      return fail(op, "scalar64 operator/macro range must be literal owned main-file input");
    const bool assignment = operation->getOpcode() == BO_AddAssign;
    std::string typeName = "double";
    const auto* primary = family(selected->owner);
    const bool dependent = operation->getType()->isDependentType();
    if (dependent) {
      if (!primary || !templateProof(primary)) return false;
      typeName = parameters[primary]->getNameAsString();
    }
    bool observedDouble = false;
    for (const auto& occurrence : occurrences) {
      const auto* value = occurrence.expression;
      if (observation(value)) return fail(op, "scalar64 operator in unevaluated/constant observation is not admitted");
      if (value->getType()->isDependentType()) continue;
      if (!addType(value->getType(), context)) return fail(op, "scalar64 operator has an unproved scalar instance");
      if (!scalarValue(value->getLHS()) || !scalarValue(value->getRHS()))
        return fail(op, "scalar64 operands require builtin values and effect-free addresses/RHS");
      if (assignment && (!statement(value) || !lvalue(value->getLHS())))
        return fail(op, "scalar64 += is admitted only as a discarded nonvolatile lvalue statement");
      if (dependent) {
        const auto* args = occurrence.owner->getTemplateSpecializationArgs();
        if (!args || !context.hasSameUnqualifiedType(value->getType(), args->get(0).getAsType()))
          return fail(op, "shared scalar64 operator changes usual arithmetic type across instances");
      } else if (!doubleType(value->getType(), context)) {
        return fail(op, "nondependent scalar64 operator must retain binary64 type");
      }
      const auto left = value->getLHS()->getType(), right = value->getRHS()->getType();
      if (!context.hasSameUnqualifiedType(left, value->getType()) ||
          !context.hasSameUnqualifiedType(right, value->getType()))
        return fail(op, "mixed scalar64 operand conversion is not proven");
      observedDouble |= doubleType(value->getType(), context);
    }
    if (!observedDouble) return true;
    if (!injectedTokens(op, typeName)) return false;
    const unsigned first = source.getFileOffset(begin), last = source.getFileOffset(end);
    const unsigned offset = source.getFileOffset(op);
    prefixes[first] += "::ascify::detail::" + std::string(assignment ? "scalar_add_assign_rne" : "scalar_add_rne") + "<" + typeName + ">(";
    suffixes[last] += ")";
    tokens[offset] = {op, assignment ? 2U : 1U, ","};
    editedOperators.insert(op.getRawEncoding());
    if (assignment) ++plan.assignments; else ++plan.additions;
    return true;
  }
 public:
  Proof(ASTContext& context, const std::function<bool(SourceLocation)>& owned,
        const std::function<bool(SourceLocation)>& sdk,
        const std::function<bool(const FunctionDecl*)>& shuffle,
        const std::function<bool(const FunctionDecl*)>& reduce,
        const std::function<bool(llvm::StringRef, SourceLocation)>& macroFree,
        Scalar64Plan& plan, SourceLocation& location, std::string& error)
      : context(context), source(context.getSourceManager()), owned(owned), sdk(sdk),
        shuffle(shuffle), reduce(reduce), macroFree(macroFree), plan(plan), badLocation(location), error(error) {}

  bool run(bool standard) {
    inventory.TraverseDecl(context.getTranslationUnitDecl());
    std::vector<Operation> wrappedDoubleAdditions;
    for (const auto& item : inventory.operations) {
      if (!input(item.owner)) continue;
      const bool addition = item.expression->getOpcode() == BO_Add ||
                            item.expression->getOpcode() == BO_AddAssign;
      if (addition &&
          (wrappedDoubleArithmeticType(item.expression->getType(), context) ||
           wrappedDoubleArithmeticType(plain(item.expression->getLHS())->getType(), context) ||
           wrappedDoubleArithmeticType(plain(item.expression->getRHS())->getType(), context)))
        wrappedDoubleAdditions.push_back(item);
      if ((item.expression->getOpcode() == BO_Add || item.expression->getOpcode() == BO_AddAssign) &&
          doubleType(item.expression->getType(), context)) {
        routes.insert(item.owner->getCanonicalDecl());
        mathRoutes.insert(item.owner->getCanonicalDecl());
        if (const auto* primary = family(item.owner)) arithmeticFamilies.insert(primary);
      }
      const auto location = item.expression->getOperatorLoc();
      if (source.isWrittenInMainFile(source.getExpansionLoc(location)))
        sites[location.getRawEncoding()].push_back(item);
    }
    for (const auto& item : inventory.calls) {
      if (!input(item.owner)) continue;
      const auto* callee = item.expression->getDirectCallee();
      if (callee && callee->getIdentifier() && wideType(item.expression->getType(), context) &&
          (callee->getName() == "__shfl_sync" || callee->getName() == "__shfl_down_sync") &&
          !isShuffle(item.expression))
        return fail(item.expression->getExprLoc(), "scalar64 shuffle callee has no authenticated CUDA provider");
      if (callee) edges[item.owner->getCanonicalDecl()].insert(callee->getCanonicalDecl());
      if (!callee || (!callee->hasBody() && !sdk(source.getSpellingLoc(callee->getLocation()))))
        opaque.insert(item.owner->getCanonicalDecl());
      if (wideShuffle(item.expression)) routes.insert(item.owner->getCanonicalDecl());
      if (exactReduction(item.expression) && doubleType(item.expression->getType(),context)) {
        routes.insert(item.owner->getCanonicalDecl()); mathRoutes.insert(item.owner->getCanonicalDecl());
        if (const auto* primary=family(item.owner)) arithmeticFamilies.insert(primary);
        exactReductionCalls[item.owner->getCanonicalDecl()].insert(item.expression);
      }
    }
    // Merely including an SDK/Sample utility must not activate its unused
    // overloads. Main-file device definitions and explicit emitted instances
    // are roots; follow their actual direct callees into ordinary headers.
    std::set<const FunctionDecl*> reachable;
    for (const auto* function : inventory.functions)
      if (input(function) && (source.isWrittenInMainFile(
              source.getExpansionLoc(function->getLocation())) ||
          function->getTemplateSpecializationKind() == TSK_ExplicitInstantiationDefinition))
        reachable.insert(function->getCanonicalDecl());
    bool reached = true;
    while (reached) {
      reached = false;
      for (const auto& edge : edges) if (reachable.count(edge.first))
        for (const auto* callee : edge.second)
          if (reachable.insert(callee).second) reached = true;
    }
    for (auto iterator = routes.begin(); iterator != routes.end();)
      if (!reachable.count(*iterator)) iterator = routes.erase(iterator); else ++iterator;
    for (auto iterator = mathRoutes.begin(); iterator != mathRoutes.end();)
      if (!reachable.count(*iterator)) iterator = mathRoutes.erase(iterator); else ++iterator;
    // Atomic/complex/vector results need not be DoubleTy. Refuse their double
    // arithmetic before the ordinary builtin routes-empty exit.
    // Reuse actual reachability; unused header bodies do not activate a route.
    // This is refusal only: no wrapped type enters helper/type admission.
    for (const auto& item : wrappedDoubleAdditions)
      if (reachable.count(item.owner->getCanonicalDecl()))
        return fail(item.expression->getOperatorLoc(),
                    "atomic/complex/vector binary64 +/+= is outside the scalar64 scalar domain");
    for (const auto& entry : exactReductionCalls) if (routes.count(entry.first))
      for (const auto* call : entry.second) {
        ++plan.reductions; plan.scalarExpressions.insert(call);
      }
    if (routes.empty()) return true;
    if (!standard) return fail({}, "scalar64 requires --default-preprocessor");
    bool changed = true;
    while (changed) {
      changed = false;
      for (const auto& edge : edges)
        for (const auto* callee : edge.second)
          if (routes.count(callee)) {
            if (routes.insert(edge.first).second) changed = true;
            if (mathRoutes.count(callee) && mathRoutes.insert(edge.first).second) changed = true;
          }
    }
    for (const auto* function : inventory.functions) {
      if (!input(function)) continue;
      const bool participating = routes.count(function->getCanonicalDecl()) != 0;
      if (function->hasAttr<CUDAGlobalAttr>())
        for (const auto* parameter : function->parameters())
          if (doubleType(parameter->getType(), context) && (participating || opaque.count(function->getCanonicalDecl())))
            return fail(parameter->getLocation(), "exact scalar64 route refuses global by-value double ABI; float parameter lowering is forbidden for this combination");
      if (!participating) continue;
      if (function->isConstexpr() || function->hasAttr<CUDAHostAttr>() ||
          isa<CXXMethodDecl>(function) || function->isVariadic() ||
          function->getTrailingRequiresClause())
        return fail(function->getLocation(), "scalar64 route refuses constexpr/consteval, dual host-device, class or open function interfaces");
      if (const auto* prototype = function->getType()->getAs<FunctionProtoType>())
        if (prototype->hasExceptionSpec())
          return fail(function->getLocation(), "scalar64 route refuses exception/noexcept interface observations");
      // Header transformations are separately proved in their own physical
      // main-file invocation. They still participate in this invocation's
      // ABI/constexpr/call-graph refusal, but never receive a foreign edit.
      if (!source.isWrittenInMainFile(source.getExpansionLoc(function->getLocation()))) continue;
      if (!family(function)) {
        if (!signatureType(function->getReturnType(), nullptr))
          return fail(function->getLocation(), "scalar64 function return must be builtin scalar/void");
        for (const auto* parameter : function->parameters())
          if (parameter->hasDefaultArg() || !signatureType(parameter->getType(), nullptr))
            return fail(parameter->getLocation(), "scalar64 function parameter has an unproved class/conversion/default");
      }
      if (mathRoutes.count(function->getCanonicalDecl()))
        if (const auto* primary = family(function)) arithmeticFamilies.insert(primary);
      if (const auto* primary = family(function)) if (!templateProof(primary)) return false;
      if (!bodyContract(function)) return false;
      for (const auto* ref : inventory.references)
        if (ref->getDecl()->getCanonicalDecl() == function->getCanonicalDecl() &&
            !inventory.directCallees.count(ref))
          return fail(ref->getLocation(), "scalar64 function pointer/address escape is not proven");
    }
    for (const auto& site : sites) {
      bool doubleAdd = false;
      for (const auto& occurrence : site.second)
        doubleAdd |= routes.count(occurrence.owner->getCanonicalDecl()) &&
            (occurrence.expression->getOpcode() == BO_Add || occurrence.expression->getOpcode() == BO_AddAssign) &&
            doubleType(occurrence.expression->getType(), context);
      if (doubleAdd && !editSite(site.second)) return false;
    }
    for (const auto& call : inventory.calls) {
      if (!input(call.owner) || !routes.count(call.owner->getCanonicalDecl()) || !wideShuffle(call.expression)) continue;
      if (source.isWrittenInMainFile(source.getExpansionLoc(call.expression->getExprLoc())) &&
          !literal(call.expression->getExprLoc()))
        return fail(call.expression->getExprLoc(), "scalar64 transport macro/physical call provenance is not proven");
      if (observation(call.expression) || !scalarValue(call.expression))
        return fail(call.expression->getExprLoc(), "scalar64 shuffle requires evaluated builtin values and effect-free arguments");
      if (!shuffleControls(call.expression)) return false;
      if (source.isWrittenInMainFile(call.expression->getExprLoc())) ++plan.transports;
    }
    for (const auto& entry : parameters) {
      const auto* pattern = entry.first->getTemplatedDecl();
      const auto* body = dyn_cast_or_null<CompoundStmt>(pattern->getBody());
      if (!body || !literal(body->getLBracLoc()))
        return fail(pattern->getLocation(), "scalar64 template guard requires a literal owned body");
      const auto insertion = Lexer::getLocForEndOfToken(body->getLBracLoc(), 0, source, context.getLangOpts());
      const std::string name = entry.second->getNameAsString();
      if (!injectedTokens(insertion, name)) return false;
      const bool math = arithmeticFamilies.count(entry.first) != 0;
      const std::string domain = "__is_same("+name+",int) || __is_same("+name+",float) || __is_same("+name+",double)";
      const std::string wide = " || __is_same("+name+",unsigned int) || ((sizeof("+name+")==8) && (__is_same("+name+",long) || __is_same("+name+",unsigned long) || __is_same("+name+",long long) || __is_same("+name+",unsigned long long)))";
      plan.edits.push_back({insertion, 0, "\n  static_assert("+domain+(math ? "" : wide)+", \"Ascify scalar64 requires the proven closed builtin dtype domain\");\n"});
      ++plan.guards;
      const auto group=reduceGroups.find(entry.first);
      if (group!=reduceGroups.end()) {
        const auto groupName=group->second->getNameAsString();
        if (!injectedTokens(insertion,groupName)) return false;
        plan.edits.push_back({insertion,0,"\n  static_assert(::ascify_cg::uniform_reduce_group_domain<"+groupName+">::value, \"Ascify exact CG reduce requires the proved group domain\");\n"});
        ++plan.guards;
      }
    }
    const auto base = source.getLocForStartOfFile(source.getMainFileID());
    for (const auto& prefix : prefixes) plan.edits.push_back({base.getLocWithOffset(prefix.first), 0, prefix.second});
    for (const auto& suffix : suffixes) plan.edits.push_back({base.getLocWithOffset(suffix.first), 0, suffix.second});
    for (const auto& token : tokens) plan.edits.push_back(token.second);
    // A dependent raw CUDA shuffle has an overload placeholder rather than
    // the literal type T. Prove this one call from its authenticated lookup
    // set and all concrete occurrences; do not exempt dependent member calls,
    // user templates, ADL classes or unrelated calls in the same body.
    for (const auto& item : inventory.calls) {
      const auto* primary = family(item.owner);
      if (!primary || !parameters.count(primary) ||
          !item.expression->getType()->isDependentType()) continue;
      const auto* lookup = dyn_cast<UnresolvedLookupExpr>(plain(item.expression->getCallee()));
      const auto* identifier = lookup ? lookup->getName().getAsIdentifierInfo() : nullptr;
      if (!identifier) continue;
      const auto name = identifier->getName();
      if (name != "__shfl_sync" && name != "__shfl_down_sync") continue;
      unsigned candidates = 0;
      for (const auto* declaration : lookup->decls()) {
        if (const auto* shadow = dyn_cast<UsingShadowDecl>(declaration)) declaration = shadow->getTargetDecl();
        const auto* callee = dyn_cast<FunctionDecl>(declaration);
        if (!callee || !shuffle(callee))
          return fail(lookup->getNameLoc(), "dependent scalar64 shuffle lookup has an unproved provider candidate");
        ++candidates;
      }
      bool concrete = false;
      for (const auto& instance : inventory.calls) {
        if (family(instance.owner) != primary ||
            instance.expression->getExprLoc() != item.expression->getExprLoc() ||
            instance.expression->getType()->isDependentType()) continue;
        if (!isShuffle(instance.expression) ||
            !transportType(instance.expression->getType(), context) ||
            !scalarValue(instance.expression))
          return fail(instance.expression->getExprLoc(), "dependent scalar64 shuffle occurrence is not closed builtin transport");
        concrete = true;
      }
      if (!candidates || !concrete)
        return fail(lookup->getNameLoc(), "dependent scalar64 shuffle has no proved concrete closure");
      plan.scalarExpressions.insert(item.expression);
      plan.scalarExpressions.insert(lookup);
    }
    // Only guarded T-derived types are exempt from SDK-layout exclusion.
    // No void-NTTP, record, inquiry, or entire function-body exception is added.
    for (const auto& item : inventory.expressions) {
      const auto* primary = family(item.first);
      if (!primary || !parameters.count(primary)) continue;
      for (const auto* expression : item.second) {
        if (!expression->getType().isNull() && expression->getType()->isDependentType() &&
            ProvesScalar64Type(plan, expression->getType())) plan.scalarExpressions.insert(expression);
      }
    }
    for (const auto& site : sites)
      if (editedOperators.count(site.first))
        for (const auto& occurrence : site.second) plan.scalarExpressions.insert(occurrence.expression);
    return true;
  }
};
} // namespace

bool ProvesScalar64Type(const Scalar64Plan& plan, QualType type) {
  if (type.isNull()) return false;
  if (const auto* parameter = type->getAs<TemplateTypeParmType>())
    return plan.scalarParameters.count(parameter->getDecl()) != 0 ||
        plan.reduceGroupParameters.count(parameter->getDecl()) != 0;
  if (const auto* pointer = type->getAs<PointerType>()) return ProvesScalar64Type(plan, pointer->getPointeeType());
  if (const auto* reference = type->getAs<ReferenceType>()) return ProvesScalar64Type(plan, reference->getPointeeType());
  if (const auto* array = dyn_cast<ArrayType>(type.getTypePtr())) return ProvesScalar64Type(plan, array->getElementType());
  return false;
}

bool ProveScalar64Lowering(ASTContext& context,
    const std::function<bool(SourceLocation)>& ownedInput,
    const std::function<bool(SourceLocation)>& trustedSdk,
    const std::function<bool(const FunctionDecl*)>& trustedShuffle,
    const std::function<bool(const FunctionDecl*)>& trustedReduce,
    const std::function<bool(llvm::StringRef, SourceLocation)>& macroFree,
    bool standardPreprocessing, Scalar64Plan& output,
    SourceLocation& errorLocation, std::string& error) {
  Scalar64Plan staged;
  if (!Proof(context, ownedInput, trustedSdk, trustedShuffle, trustedReduce, macroFree,
             staged, errorLocation, error).run(standardPreprocessing)) return false;
  output = std::move(staged);
  return true;
}
} // namespace ascify
