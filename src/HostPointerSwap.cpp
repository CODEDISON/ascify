#include "HostPointerSwap.h"

#include <map>
#include <utility>
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

namespace ascify {
namespace {
using namespace clang;
using Family = const FunctionTemplateDecl *;
const Expr *plain(const Expr *expression) {
  return expression ? expression->IgnoreParenImpCasts() : nullptr;
}
Family family(const FunctionDecl *function) {
  if (!function) return nullptr;
  const auto *result = function->getPrimaryTemplate();
  if (!result) result = function->getDescribedFunctionTemplate();
  return result ? result->getCanonicalDecl() : nullptr;
}
const ValueDecl *referenced(const Expr *expression) {
  const auto *reference = dyn_cast_or_null<DeclRefExpr>(plain(expression));
  return reference ? reference->getDecl() : nullptr;
}
bool containsVariable(const Stmt *statement, const VarDecl *variable) {
  std::vector<const Stmt *> pending{statement};
  std::set<const Stmt *> visited;
  while (!pending.empty()) {
    const auto *current = pending.back(); pending.pop_back();
    if (!current || !visited.insert(current).second) continue;
    if (const auto *reference = dyn_cast<DeclRefExpr>(current))
      if (reference->getDecl() == variable) return true;
    for (const auto *child : current->children()) pending.push_back(child);
  }
  return false;
}
bool device(const FunctionDecl *function) {
  return function && (function->hasAttr<CUDADeviceAttr>() || function->hasAttr<CUDAGlobalAttr>());
}
struct Invocation { const FunctionDecl *owner; const CallExpr *call; };
class Inventory : public RecursiveASTVisitor<Inventory> {
  const FunctionDecl *current = nullptr;
public:
  std::set<Family> families;
  std::vector<Invocation> calls;
  std::vector<const DeclRefExpr *> references;
  std::vector<const UnresolvedLookupExpr *> lookups;
  std::set<const NamedDecl *> usingTargets;
  std::vector<const VarDecl *> variables;
  std::vector<const BinaryOperator *> assignments;
  std::vector<const UnaryOperator *> addresses;
  std::vector<const ExplicitCastExpr *> casts;
  std::set<const Expr *> directCallees;
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }
  bool TraverseDecl(Decl *declaration) {
    const auto *saved = current;
    if (const auto *function = dyn_cast_or_null<FunctionDecl>(declaration)) current = function;
    const bool result = RecursiveASTVisitor<Inventory>::TraverseDecl(declaration);
    current = saved;
    return result;
  }
  bool VisitFunctionTemplateDecl(FunctionTemplateDecl *declaration) {
    families.insert(declaration->getCanonicalDecl()); return true;
  }
  bool VisitCallExpr(CallExpr *call) {
    calls.push_back({current, call}); directCallees.insert(plain(call->getCallee())); return true;
  }
  bool VisitDeclRefExpr(DeclRefExpr *reference) { references.push_back(reference); return true; }
  bool VisitUnresolvedLookupExpr(UnresolvedLookupExpr *lookup) { lookups.push_back(lookup); return true; }
  bool VisitUsingDecl(UsingDecl *declaration) {
    for (const auto *shadow : declaration->shadows()) usingTargets.insert(shadow->getTargetDecl());
    return true;
  }
  bool VisitUsingShadowDecl(UsingShadowDecl *declaration) {
    usingTargets.insert(declaration->getTargetDecl()); return true;
  }
  bool VisitVarDecl(VarDecl *variable) { variables.push_back(variable); return true; }
  bool VisitBinaryOperator(BinaryOperator *operation) {
    if (operation->isAssignmentOp()) assignments.push_back(operation); return true;
  }
  bool VisitUnaryOperator(UnaryOperator *operation) {
    if (operation->getOpcode() == UO_AddrOf) addresses.push_back(operation); return true;
  }
  bool VisitExplicitCastExpr(ExplicitCastExpr *expression) { casts.push_back(expression); return true; }
};

class Proof {
  ASTContext &context;
  SourceManager &source;
  const std::function<bool(SourceLocation)> &owned;
  const std::function<bool(const CallExpr *)> &allocation;
  const std::function<bool(llvm::StringRef, SourceLocation)> &macroFree;
  HostPointerSwapPlan staged;
  Inventory inventory;
  SourceLocation &bad;
  std::string &error;
  bool fail(SourceLocation location, const char *message) {
    if (error.empty()) { bad = location; error = message; } return false;
  }
  bool literal(SourceLocation location) const { return location.isFileID() && owned(location); }
  bool tokens(SourceRange range) const {
    auto cursor = range.getBegin();
    const auto end = Lexer::getLocForEndOfToken(range.getEnd(), 0, source, context.getLangOpts());
    if (!literal(cursor) || !literal(range.getEnd()) || !literal(end) ||
        source.getFileID(cursor) != source.getFileID(end)) return false;
    while (source.getFileOffset(cursor) < source.getFileOffset(end)) {
      Token token;
      if (Lexer::getRawToken(cursor, token, source, context.getLangOpts(), true) || token.needsCleaning() ||
          token.is(tok::hash) || token.is(tok::hashhash)) return false;
      const auto word = Lexer::getSpelling(token, source, context.getLangOpts());
      if (word == "_Pragma" || word == "__pragma" ||
          (token.isAnyIdentifier() && !macroFree(word, token.getLocation()))) return false;
      const auto next = Lexer::getLocForEndOfToken(token.getLocation(), 0, source, context.getLangOpts());
      if (!literal(next) || source.getFileOffset(next) <= source.getFileOffset(cursor)) return false;
      cursor = next;
    }
    return true;
  }
  // Do not canonicalize: SDK typedefs and caller aliases may have the same
  // canonical builtin pointee. The actual TypeLoc must spell both '*' and
  // the builtin, in an owned, macro-free source range.
  std::string pointer(QualType type) const {
    if (type.isNull() || type.hasLocalQualifiers()) return {};
    const auto *value = dyn_cast<PointerType>(type.getTypePtr());
    if (!value || value->getPointeeType().hasLocalQualifiers()) return {};
    const auto *builtin = dyn_cast<BuiltinType>(value->getPointeeType().getTypePtr());
    if (!builtin) return {};
    switch (builtin->getKind()) {
    case BuiltinType::Bool: return "bool *";
    case BuiltinType::Char_S: case BuiltinType::Char_U: return "char *";
    case BuiltinType::SChar: return "signed char *";
    case BuiltinType::UChar: return "unsigned char *";
    case BuiltinType::Short: return "short *";
    case BuiltinType::UShort: return "unsigned short *";
    case BuiltinType::Int: return "int *";
    case BuiltinType::UInt: return "unsigned int *";
    case BuiltinType::Long: return "long *";
    case BuiltinType::ULong: return "unsigned long *";
    case BuiltinType::LongLong: return "long long *";
    case BuiltinType::ULongLong: return "unsigned long long *";
    case BuiltinType::Float: return "float *";
    case BuiltinType::Double: return "double *";
    default: return {};
    }
  }
  bool spelledPointer(const TypeSourceInfo *spelling) const {
    if (!spelling || pointer(spelling->getType()).empty()) return false;
    const auto type = spelling->getTypeLoc();
    const auto address = type.getAs<PointerTypeLoc>();
    return !address.isNull() && !address.getPointeeLoc().getAs<BuiltinTypeLoc>().isNull() &&
        tokens(type.getSourceRange());
  }
  bool parameterType(QualType type, const TemplateTypeParmDecl *parameter) const {
    if (type.isNull() || type.hasLocalQualifiers()) return false;
    const auto *value = dyn_cast<TemplateTypeParmType>(type.getTypePtr());
    return value && value->getDecl() == parameter;
  }
  bool referenceType(QualType type, const TemplateTypeParmDecl *parameter) const {
    const auto *reference = dyn_cast<LValueReferenceType>(type.getTypePtr());
    return reference && parameterType(reference->getPointeeType(), parameter);
  }
  bool assignment(const Stmt *statement, const ValueDecl *left, const ValueDecl *right) const {
    const auto *operation = dyn_cast<BinaryOperator>(statement);
    return operation && operation->getOpcode() == BO_Assign &&
        referenced(operation->getLHS()) == left && referenced(operation->getRHS()) == right;
  }
  bool shape(Family primary) const {
    const auto *function = primary->getTemplatedDecl();
    const auto *parameters = primary->getTemplateParameters();
    const auto *body = dyn_cast_or_null<CompoundStmt>(function->getBody());
    if (parameters->size() != 1 || function->getNumParams() != 2 || !body || body->size() != 3 ||
        !function->getReturnType()->isVoidType() ||
        !isa<BuiltinType>(function->getReturnType().getTypePtr())) return false;
    const auto *parameter = dyn_cast<TemplateTypeParmDecl>(parameters->getParam(0));
    if (!parameter || !referenceType(function->getParamDecl(0)->getType(), parameter) ||
        !referenceType(function->getParamDecl(1)->getType(), parameter)) return false;
    auto statement = body->body_begin();
    const auto *declaration = dyn_cast<DeclStmt>(*statement++);
    const auto *temporary = declaration && declaration->isSingleDecl()
        ? dyn_cast<VarDecl>(declaration->getSingleDecl()) : nullptr;
    return temporary && temporary->hasLocalStorage() && !temporary->isStaticLocal() &&
        !temporary->hasAttrs() && parameterType(temporary->getType(), parameter) && temporary->hasInit() &&
        referenced(temporary->getInit()) == function->getParamDecl(0) &&
        assignment(*statement++, function->getParamDecl(0), function->getParamDecl(1)) &&
        assignment(*statement, function->getParamDecl(1), temporary);
  }
  bool pointerValue(const Expr *expression, std::set<const VarDecl *> &seen) const {
    expression = plain(expression);
    if (!expression || expression->getExprLoc().isMacroID()) return false;
    if (isa<CXXNullPtrLiteralExpr>(expression)) return true;
    if (const auto *created = dyn_cast<CXXNewExpr>(expression)) {
      const auto *spelling = created->getAllocatedTypeSourceInfo();
      return created->getNumPlacementArgs() == 0 && spelling &&
          isa<BuiltinType>(created->getAllocatedType().getTypePtr()) &&
          !created->getAllocatedType()->isVoidType() && tokens(spelling->getTypeLoc().getSourceRange());
    }
    const auto *variable = dyn_cast_or_null<VarDecl>(referenced(expression));
    if (!variable || !literal(variable->getLocation()) || !spelledPointer(variable->getTypeSourceInfo())) return false;
    if (!seen.insert(variable).second) return true;
    if (variable->hasInit() && !pointerValue(variable->getInit(), seen)) return false;
    for (const auto *operation : inventory.assignments)
      if (referenced(operation->getLHS()) == variable &&
          (operation->getOpcode() != BO_Assign || !pointerValue(operation->getRHS(), seen))) return false;
    // A reference alias or an address escape can subsequently assign an erased
    // SDK object to this variable. Only a physical SDK cudaMalloc out-parameter
    // is an admitted address use; ordinary by-value uses do not mutate it.
    for (const auto *alias : inventory.variables)
      if (alias != variable && alias->hasInit() && containsVariable(alias->getInit(), variable) &&
          !spelledPointer(alias->getTypeSourceInfo())) return false;
    // reinterpret_cast<void*&>(p) aliases p without an '&p' UnaryOperator.
    // Reject writable cast aliases even when they occur only as call operands.
    for (const auto *cast : inventory.casts)
      if (cast->isLValue() && containsVariable(cast->getSubExpr(), variable)) return false;
    for (const auto *address : inventory.addresses) {
      if (referenced(address->getSubExpr()) != variable) continue;
      bool allocated = false;
      for (const auto &invocation : inventory.calls)
        if (allocation(invocation.call) && invocation.call->getNumArgs() == 2) {
          const auto *argument = invocation.call->getArg(0)->IgnoreParens();
          while (const auto *cast = dyn_cast<CastExpr>(argument)) argument = cast->getSubExpr()->IgnoreParens();
          if (argument == address) allocated = true;
        }
      if (!allocated) return false;
    }
    for (const auto &invocation : inventory.calls) {
      const auto *callee = invocation.call->getDirectCallee();
      for (unsigned i = 0; i < invocation.call->getNumArgs(); ++i) {
        if (!containsVariable(invocation.call->getArg(i), variable)) continue;
        if (!callee || i >= callee->getNumParams()) return false;
        if (callee->getParamDecl(i)->getType()->isReferenceType() &&
            (!shapeFamily(callee) || referenced(invocation.call->getArg(i)) != variable)) return false;
      }
    }
    return true;
  }
  bool shapeFamily(const FunctionDecl *function) const {
    const auto primary = family(function);
    return primary && literal(primary->getLocation()) && shape(primary);
  }
  bool prove(Family primary) {
    const auto *function = primary->getTemplatedDecl();
    const auto *parameters = primary->getTemplateParameters();
    const auto *parameter = cast<TemplateTypeParmDecl>(parameters->getParam(0));
    const auto *body = cast<CompoundStmt>(function->getBody());
    if (device(function) || !function->getDeclContext()->isTranslationUnit() ||
        !function->getIdentifier() || primary->getPreviousDecl() || function->getPreviousDecl() ||
        primary->getMostRecentDecl() != primary || function->getMostRecentDecl() != function ||
        function->isVariadic() || function->isConstexpr() || function->getTrailingRequiresClause() ||
        parameters->getRequiresClause() || parameter->isParameterPack() || parameter->hasDefaultArgument() ||
        parameter->getDepth() != 0 || !parameter->getIdentifier() ||
        function->getParamDecl(0)->hasDefaultArg() || function->getParamDecl(1)->hasDefaultArg() ||
        !tokens(primary->getSourceRange()))
      return fail(function->getLocation(), "host pointer swap requires one owned literal primary without redeclarations or controls");
    for (const auto *attribute : function->attrs())
      if (!attribute->isImplicit() || !isa<CUDAHostAttr>(attribute))
        return fail(attribute->getLocation(), "host pointer swap attributes are not admitted");
    if (parameter->hasAttrs() || function->getParamDecl(0)->hasAttrs() || function->getParamDecl(1)->hasAttrs())
      return fail(function->getLocation(), "host pointer swap parameter attributes are not admitted");
    const auto prototype = function->getType()->getAs<FunctionProtoType>();
    if (!prototype || prototype->hasExceptionSpec())
      return fail(function->getLocation(), "host pointer swap exception interface is not admitted");
    std::set<std::string> domain;
    for (const auto *instance : primary->specializations()) {
      const auto *arguments = instance->getTemplateSpecializationArgs();
      if (instance->getTemplateSpecializationKind() != TSK_ImplicitInstantiation || !instance->hasBody() ||
          !arguments || arguments->size() != 1 || arguments->get(0).getKind() != TemplateArgument::Type)
        return fail(instance->getLocation(), "host pointer swap specialization is not closed");
      const auto spelling = pointer(arguments->get(0).getAsType());
      if (spelling.empty()) return fail(instance->getLocation(), "host pointer swap requires a direct builtin pointer instance");
      domain.insert(spelling);
    }
    unsigned calls = 0;
    for (const auto &invocation : inventory.calls) {
      const auto *callee = invocation.call->getDirectCallee();
      if (family(callee) != primary) continue;
      const auto *reference = dyn_cast_or_null<DeclRefExpr>(plain(invocation.call->getCallee()));
      if (!reference || reference->hasQualifier() || !literal(reference->getLocation()) ||
          !invocation.owner || device(invocation.owner) || !tokens(invocation.call->getSourceRange()) ||
          invocation.call->getNumArgs() != 2)
        return fail(invocation.call->getExprLoc(), "host pointer swap requires owned direct host calls");
      if (reference->hasExplicitTemplateArgs())
        for (const auto &argument : reference->template_arguments())
          if (argument.getArgument().getKind() != TemplateArgument::Type ||
              !spelledPointer(argument.getTypeSourceInfo()))
            return fail(reference->getLocation(), "host pointer swap explicit argument contains an alias or erased type");
      for (const auto *argument : invocation.call->arguments()) {
        std::set<const VarDecl *> seen;
        if (!isa_and_nonnull<DeclRefExpr>(plain(argument)) || !pointerValue(argument, seen))
          return fail(argument->getExprLoc(), "host pointer swap actual has unproved pointer spelling or origin");
      }
      ++calls;
    }
    if (!domain.empty() && !calls)
      return fail(function->getLocation(), "host pointer swap has no direct concrete call");
    if (domain.empty() && calls)
      return fail(function->getLocation(), "empty host pointer swap domain has a call observation");
    for (const auto *reference : inventory.references)
      if (family(dyn_cast<FunctionDecl>(reference->getDecl())) == primary &&
          !inventory.directCallees.count(reference))
        return fail(reference->getLocation(), "host pointer swap address or indirect escape is not admitted");
    for (const auto *lookup : inventory.lookups)
      for (const auto *declaration : lookup->decls()) {
        if (const auto *shadow = dyn_cast<UsingShadowDecl>(declaration)) declaration = shadow->getTargetDecl();
        if (const auto *candidate = dyn_cast<FunctionTemplateDecl>(declaration))
          if (candidate->getCanonicalDecl() == primary)
            return fail(lookup->getNameLoc(), "host pointer swap unresolved or future call domain is not admitted");
      }
    for (const auto *target : inventory.usingTargets) {
      const auto *candidate = dyn_cast<FunctionTemplateDecl>(target);
      if ((candidate && candidate->getCanonicalDecl() == primary) ||
          family(dyn_cast<FunctionDecl>(target)) == primary)
        return fail(target->getLocation(), "host pointer swap using export is not admitted");
    }
    const auto name = parameter->getNameAsString();
    std::string guard = "\nstatic_assert(";
    if (domain.empty()) {
      // The complete specialization/reference inventory is empty. Retain the
      // dormant primary, but reject every future BODY instantiation. A guard
      // cannot reject unevaluated signature uses or extern declarations: the
      // original inventory above must independently exclude those references.
      guard += "!__is_same(" + name + ", " + name + ")";
      guard += ", \"Ascify: host pointer swap has no proven body instance in this translation unit\");\n";
    } else {
      for (const auto &type : domain) {
        if (guard != "\nstatic_assert(") guard += " || ";
        guard += "__is_same(" + name + ", " + type + ")";
      }
      guard += ", \"Ascify: host pointer swap is only proven for the emitted pointer type domain\");\n";
    }
    for (const auto *token : {"static_assert", "__is_same", "bool", "char", "signed", "unsigned",
                              "short", "int", "long", "float", "double"})
      if (!macroFree(token, body->getLBracLoc()))
        return fail(body->getLBracLoc(), "input macro changes the host pointer swap guard");
    class PatternFacts : public RecursiveASTVisitor<PatternFacts> {
      HostPointerSwapPlan &plan;
    public:
      explicit PatternFacts(HostPointerSwapPlan &plan) : plan(plan) {}
      bool shouldVisitImplicitCode() const { return true; }
      bool VisitTypeLoc(TypeLoc type) {
        if (type.getType()->isDependentType()) plan.typeLocs.insert(type.getOpaqueData()); return true;
      }
      bool VisitExpr(Expr *expression) {
        if (!expression->getType().isNull() && expression->getType()->isDependentType())
          plan.expressions.insert(expression);
        return true;
      }
    } facts(staged);
    facts.TraverseDecl(const_cast<FunctionDecl *>(function));
    if (source.isWrittenInMainFile(body->getLBracLoc()))
      staged.guards.push_back({Lexer::getLocForEndOfToken(body->getLBracLoc(), 0, source, context.getLangOpts()), guard});
    else staged.contextualHeaders.insert(source.getFilename(body->getLBracLoc()).str());
    ++staged.patterns; staged.concreteCalls += calls;
    if (domain.empty()) ++staged.emptyPatterns;
    return true;
  }
public:
  Proof(ASTContext &context, const std::function<bool(SourceLocation)> &owned,
      const std::function<bool(const CallExpr *)> &allocation,
      const std::function<bool(llvm::StringRef, SourceLocation)> &macroFree,
      SourceLocation &bad, std::string &error)
      : context(context), source(context.getSourceManager()), owned(owned), allocation(allocation),
        macroFree(macroFree), bad(bad), error(error) {}
  bool run(bool standard, HostPointerSwapPlan &output) {
    inventory.TraverseDecl(context.getTranslationUnitDecl());
    for (const auto primary : inventory.families)
      if (literal(primary->getLocation()) && shape(primary)) {
        if (!standard) return fail(primary->getLocation(), "host pointer swap requires standard preprocessing");
        if (!prove(primary)) return false;
      }
    output = std::move(staged); return true;
  }
};
} // namespace
bool ProveHostPointerSwaps(ASTContext &context,
    const std::function<bool(SourceLocation)> &owned,
    const std::function<bool(const CallExpr *)> &allocation,
    const std::function<bool(llvm::StringRef, SourceLocation)> &macroFree,
    bool standard, HostPointerSwapPlan &plan, SourceLocation &bad, std::string &error) {
  return Proof(context, owned, allocation, macroFree, bad, error).run(standard, plan);
}
} // namespace ascify
