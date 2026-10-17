#include "BoundedDeviceTemplateInlining.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <algorithm>
#include <map>

namespace ascify {
namespace {
using namespace clang;
constexpr unsigned MaxNodes = 256;
constexpr unsigned MaxDepth = 64;

const FunctionTemplateDecl *familyOf(const FunctionDecl *function) {
  if (function == nullptr) return nullptr;
  const auto *family = function->getPrimaryTemplate();
  if (family == nullptr) family = function->getDescribedFunctionTemplate();
  return family == nullptr ? nullptr : family->getCanonicalDecl();
}

// All references, including uninstantiated bodies, are inspected. Recording
// exact callee nodes avoids assuming a unique AST parent for shared patterns.
class References : public RecursiveASTVisitor<References> {
public:
  std::set<const FunctionTemplateDecl *> families;
  std::vector<const DeclRefExpr *> references;
  std::vector<const UnresolvedLookupExpr *> lookups;
  std::set<const Expr *> directCallees;
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool VisitFunctionTemplateDecl(FunctionTemplateDecl *function) {
    families.insert(function->getCanonicalDecl()); return true;
  }
  bool VisitDeclRefExpr(DeclRefExpr *reference) {
    if (isa<FunctionDecl>(reference->getDecl())) references.push_back(reference);
    return true;
  }
  bool VisitUnresolvedLookupExpr(UnresolvedLookupExpr *lookup) {
    lookups.push_back(lookup); return true;
  }
  bool VisitCallExpr(CallExpr *call) {
    if (call->getStmtClass() == Stmt::CallExprClass)
      directCallees.insert(call->getCallee()->IgnoreParenImpCasts());
    return true;
  }
  bool noEscape(const FunctionTemplateDecl *family) const {
    for (const auto *reference : references)
      if (familyOf(cast<FunctionDecl>(reference->getDecl())) == family &&
          (reference->hasQualifier() || reference->getLocation().isMacroID() ||
           !directCallees.count(reference))) return false;
    for (const auto *lookup : lookups) {
      bool mentions = false;
      unsigned count = 0;
      for (const auto *declaration : lookup->decls()) {
        ++count;
        if (const auto *shadow = dyn_cast<UsingShadowDecl>(declaration))
          declaration = shadow->getTargetDecl();
        const auto *function = dyn_cast<FunctionTemplateDecl>(declaration);
        mentions |= function != nullptr && function->getCanonicalDecl() == family;
      }
      if (mentions && (count != 1 || lookup->getQualifier() ||
                       lookup->getNameLoc().isMacroID() ||
                       !directCallees.count(lookup))) return false;
    }
    return true;
  }
};

class FamilyProof {
  ASTContext &context;
  SourceManager &source;
  const FunctionTemplateDecl *family;
  const std::set<unsigned> &textureOffsets;
  const std::function<bool(llvm::StringRef, SourceLocation)> &ownedQualifier;
  const NonTypeTemplateParmDecl *parameter = nullptr;
  std::map<const FunctionDecl *, std::vector<const FunctionDecl *>> graph;
  unsigned patternCalls = 0;

  bool literal(SourceLocation location) const {
    return location.isValid() && location.isFileID() &&
           source.getFileID(location) == source.getMainFileID();
  }
  bool scalar(QualType type) const {
    return !type.isNull() && !type.isVolatileQualified() &&
        !type->isDependentType() && type->isBuiltinType() && type->isArithmeticType();
  }
  bool signature(const FunctionDecl *function) const {
    if (isa<CXXMethodDecl>(function) || !function->getDeclContext()->isFileContext() ||
        function->isVariadic() || function->getTrailingRequiresClause() ||
        !scalar(function->getReturnType())) return false;
    const auto *prototype = function->getType()->getAs<FunctionProtoType>();
    if (prototype == nullptr || prototype->hasExceptionSpec()) return false;
    for (const auto *formal : function->parameters())
      if (!scalar(formal->getType()) || formal->hasDefaultArg()) return false;
    for (const auto *attribute : function->attrs())
      if (!isa<CUDADeviceAttr>(attribute)) return false;
    return function->hasAttr<CUDADeviceAttr>();
  }
  SourceLocation qualifier(const FunctionDecl *function) const {
    if (!literal(function->getLocation()) || !signature(function) ||
        function->getPreviousDecl() != nullptr ||
        function->getMostRecentDecl() != function ||
        !function->isThisDeclarationADefinition()) return {};
    const auto *device = function->getAttr<CUDADeviceAttr>();
    if (device == nullptr || device->isInherited()) return {};
    const auto location = source.getExpansionLoc(device->getLocation());
    Token token;
    if (!literal(location) ||
        Lexer::getRawToken(location, token, source, context.getLangOpts(), true) ||
        Lexer::getSpelling(token, source, context.getLangOpts()) != "__device__" ||
        !ownedQualifier("__device__", location)) return {};
    return location;
  }
  bool integerArgument(const Expr *expression, unsigned depth = 0) const {
    if (expression == nullptr || depth > MaxDepth ||
        !context.hasSameType(expression->getType(), context.IntTy)) return false;
    expression = expression->IgnoreParenImpCasts();
    if (isa<IntegerLiteral>(expression)) return true;
    if (const auto *reference = dyn_cast<DeclRefExpr>(expression))
      return reference->getDecl() == parameter;
    if (const auto *unary = dyn_cast<UnaryOperator>(expression))
      return (unary->getOpcode() == UO_Plus || unary->getOpcode() == UO_Minus) &&
             integerArgument(unary->getSubExpr(), depth + 1);
    if (const auto *binary = dyn_cast<BinaryOperator>(expression))
      return (binary->getOpcode() == BO_Add || binary->getOpcode() == BO_Sub) &&
             integerArgument(binary->getLHS(), depth + 1) &&
             integerArgument(binary->getRHS(), depth + 1);
    return false;
  }
  bool constant(const VarDecl *variable, bool array) const {
    if (variable == nullptr || !variable->hasGlobalStorage() ||
        !literal(variable->getLocation()) || variable->getPreviousDecl() != nullptr ||
        variable->getMostRecentDecl() != variable ||
        !variable->hasAttr<CUDAConstantAttr>()) return false;
    const auto *attribute = variable->getAttr<CUDAConstantAttr>();
    const auto location = source.getExpansionLoc(attribute->getLocation());
    Token token;
    if (attribute->isInherited() || !literal(location) ||
        Lexer::getRawToken(location, token, source, context.getLangOpts(), true) ||
        Lexer::getSpelling(token, source, context.getLangOpts()) != "__constant__" ||
        !ownedQualifier("__constant__", location)) return false;
    const auto *type = context.getAsConstantArrayType(variable->getType());
    return array ? type != nullptr && scalar(type->getElementType())
                 : scalar(variable->getType());
  }
  bool expression(const Expr *value, const FunctionDecl *owner,
                  std::vector<const FunctionDecl *> &calls, unsigned depth = 0) {
    if (value == nullptr || value->getType().isNull() || depth > MaxDepth ||
        value->getType().isVolatileQualified()) return false;
    // Literal macros such as an array radius are values, not hidden callees or
    // declarations. All nonliteral operations require direct source tokens.
    if (isa<IntegerLiteral>(value) || isa<FloatingLiteral>(value) ||
        isa<CXXBoolLiteralExpr>(value)) return scalar(value->getType());
    if (const auto *substitution = dyn_cast<SubstNonTypeTemplateParmExpr>(value))
      return expression(substitution->getReplacement(), owner, calls, depth + 1);
    if (const auto *cast = dyn_cast<ImplicitCastExpr>(value)) {
      if (!scalar(cast->getType())) return false;
      switch (cast->getCastKind()) {
      case CK_LValueToRValue: case CK_NoOp: case CK_IntegralCast:
      case CK_IntegralToFloating: case CK_FloatingToIntegral: case CK_FloatingCast:
      case CK_IntegralToBoolean: case CK_FloatingToBoolean:
        return expression(cast->getSubExpr(), owner, calls, depth + 1);
      default: return false;
      }
    }
    if (const auto *paren = dyn_cast<ParenExpr>(value))
      return literal(paren->getLParen()) && literal(paren->getRParen()) &&
          expression(paren->getSubExpr(), owner, calls, depth + 1);
    if (const auto *cast = dyn_cast<ExplicitCastExpr>(value)) {
      if ((!isa<CStyleCastExpr>(cast) && !isa<CXXStaticCastExpr>(cast) &&
           !isa<CXXFunctionalCastExpr>(cast)) || !scalar(cast->getType()) ||
          !scalar(cast->getSubExpr()->getType()) || !literal(cast->getBeginLoc())) return false;
      return expression(cast->getSubExpr(), owner, calls, depth + 1);
    }
    if (const auto *reference = dyn_cast<DeclRefExpr>(value)) {
      if (reference->hasQualifier() || !literal(reference->getLocation())) return false;
      if (reference->getDecl() == parameter) return true;
      for (const auto *formal : owner->parameters())
        if (reference->getDecl() == formal) return scalar(formal->getType());
      return constant(dyn_cast<VarDecl>(reference->getDecl()), false);
    }
    if (const auto *binary = dyn_cast<BinaryOperator>(value))
      return literal(binary->getOperatorLoc()) &&
          (binary->getOpcode() == BO_Add || binary->getOpcode() == BO_Sub ||
              binary->getOpcode() == BO_Mul) &&
          expression(binary->getLHS(), owner, calls, depth + 1) &&
          expression(binary->getRHS(), owner, calls, depth + 1);
    if (const auto *unary = dyn_cast<UnaryOperator>(value))
      return literal(unary->getOperatorLoc()) &&
          (unary->getOpcode() == UO_Plus || unary->getOpcode() == UO_Minus) &&
          expression(unary->getSubExpr(), owner, calls, depth + 1);
    if (const auto *array = dyn_cast<ArraySubscriptExpr>(value)) {
      const auto *base = dyn_cast<DeclRefExpr>(array->getBase()->IgnoreParenImpCasts());
      return base != nullptr && literal(base->getLocation()) && literal(array->getRBracketLoc()) &&
          !base->hasQualifier() && constant(dyn_cast<VarDecl>(base->getDecl()), true) &&
          scalar(array->getType()) && expression(array->getIdx(), owner, calls, depth + 1);
    }
    const auto *call = dyn_cast<CallExpr>(value);
    if (call == nullptr || call->getStmtClass() != Stmt::CallExprClass) return false;
    const auto *callee = call->getDirectCallee();
    const auto *calleeExpression = call->getCallee()->IgnoreParenImpCasts();
    if (!literal(calleeExpression->getBeginLoc()) || !literal(call->getRParenLoc())) return false;
    if (callee != nullptr && familyOf(callee) != family) {
      const auto *reference = dyn_cast<DeclRefExpr>(calleeExpression);
      // The preceding texture gate authenticated owner, exact float overload,
      // direct call and source token. No other foreign call is admitted here.
      if (reference == nullptr || !literal(reference->getLocation()) ||
          !textureOffsets.count(source.getFileOffset(reference->getLocation())) ||
          (callee->getName() != "tex2D" && callee->getName() != "tex2DLayered")) return false;
    } else {
      const auto *target = family->getTemplatedDecl();
      if (call->getNumArgs() != target->getNumParams()) return false;
      for (unsigned i = 0; i < call->getNumArgs(); ++i)
        if (!context.hasSameType(call->getArg(i)->getType().getCanonicalType().getUnqualifiedType(),
                                target->getParamDecl(i)->getType().getCanonicalType().getUnqualifiedType()))
          return false;
      if (callee == nullptr) {
        const auto *lookup = dyn_cast<UnresolvedLookupExpr>(calleeExpression);
        if (lookup == nullptr || lookup->getQualifier() ||
            lookup->getNumTemplateArgs() != 1 || !lookup->hasExplicitTemplateArgs()) return false;
        auto begin = lookup->decls_begin();
        if (begin == lookup->decls_end()) return false;
        const auto *candidate = dyn_cast<FunctionTemplateDecl>(*begin++);
        if (candidate == nullptr || candidate->getCanonicalDecl() != family ||
            begin != lookup->decls_end()) return false;
        const auto &argument = lookup->template_arguments().begin()->getArgument();
        if (argument.getKind() != TemplateArgument::Expression ||
            !integerArgument(argument.getAsExpr())) return false;
        ++patternCalls;
      } else {
        if (!isa<DeclRefExpr>(calleeExpression)) return false;
        calls.push_back(callee->getCanonicalDecl());
      }
    }
    for (const auto *argument : call->arguments())
      if (!expression(argument, owner, calls, depth + 1)) return false;
    return true;
  }
  bool body(const FunctionDecl *function, std::vector<const FunctionDecl *> &calls) {
    const auto *block = dyn_cast_or_null<CompoundStmt>(function->getBody());
    if (block == nullptr || block->size() != 1 ||
        !literal(block->getLBracLoc()) || !literal(block->getRBracLoc())) return false;
    const auto *result = dyn_cast<ReturnStmt>(*block->body_begin());
    return result != nullptr && literal(result->getReturnLoc()) &&
        expression(result->getRetValue(), function, calls);
  }
public:
  FamilyProof(ASTContext &context, const FunctionTemplateDecl *family,
              const std::set<unsigned> &offsets,
              const std::function<bool(llvm::StringRef, SourceLocation)> &owned)
      : context(context), source(context.getSourceManager()), family(family),
        textureOffsets(offsets), ownedQualifier(owned) {}
  bool prove(std::vector<SourceLocation> &locations, unsigned &instances) {
    const auto *parameters = family->getTemplateParameters();
    if (family->getPreviousDecl() != nullptr || family->getMostRecentDecl() != family ||
        parameters->size() != 1 || parameters->getRequiresClause() != nullptr) return false;
    parameter = dyn_cast<NonTypeTemplateParmDecl>(parameters->getParam(0));
    if (parameter == nullptr || parameter->isParameterPack() || parameter->hasDefaultArgument() ||
        !context.hasSameType(parameter->getType(), context.IntTy)) return false;
    const auto *pattern = family->getTemplatedDecl();
    const auto insertion = qualifier(pattern);
    std::vector<const FunctionDecl *> ignored;
    // A node/depth cap alone does not bound expansion: two identical child
    // calls can double the inlined body at every level. Version 1 proves a
    // chain, with exactly one recursive call in the pattern and each instance.
    if (insertion.isInvalid() || !body(pattern, ignored) || patternCalls != 1) return false;
    locations.push_back(insertion);
    unsigned terminalCount = 0, implicitCount = 0;
    for (const auto *instance : family->specializations()) {
      const auto kind = instance->getTemplateSpecializationKind();
      if (kind != TSK_ImplicitInstantiation && kind != TSK_ExplicitSpecialization) return false;
      const auto *arguments = instance->getTemplateSpecializationArgs();
      if (arguments == nullptr || arguments->size() != 1 ||
          arguments->get(0).getKind() != TemplateArgument::Integral ||
          !context.hasSameType(arguments->get(0).getIntegralType(), context.IntTy) ||
          !instance->isThisDeclarationADefinition() || !signature(instance) ||
          instance->getPreviousDecl() != nullptr || instance->getMostRecentDecl() != instance)
        return false;
      auto &edges = graph[instance->getCanonicalDecl()];
      if (graph.size() > MaxNodes || !body(instance, edges)) return false;
      if (kind == TSK_ExplicitSpecialization) {
        const auto terminal = qualifier(instance);
        if (terminal.isInvalid() || !edges.empty()) return false;
        locations.push_back(terminal); ++terminalCount;
      } else {
        if (edges.size() != 1) return false;
        ++implicitCount;
      }
    }
    if (terminalCount == 0 || implicitCount == 0) return false;
    // A memoized height proves the depth of every path, not just the first DFS
    // route to a shared node. A visiting marker rejects cycles before recursion.
    std::map<const FunctionDecl *, unsigned> height;
    std::set<const FunctionDecl *> visiting;
    std::function<unsigned(const FunctionDecl *, unsigned)> visit =
        [&](const FunctionDecl *node, unsigned depth) -> unsigned {
      if (depth > MaxDepth || !graph.count(node) || visiting.count(node)) return 0;
      const auto found = height.find(node);
      if (found != height.end()) return found->second;
      visiting.insert(node);
      unsigned result = 1;
      for (const auto *next : graph.at(node)) {
        const auto child = visit(next, depth + 1);
        if (child == 0 || child >= MaxDepth) return 0;
        result = std::max(result, child + 1);
      }
      visiting.erase(node); height[node] = result; return result;
    };
    for (const auto &node : graph)
      if (visit(node.first, 1) == 0) return false;
    instances = graph.size(); return true;
  }
};
} // namespace

BoundedDeviceTemplateEdits ProveBoundedDeviceTemplateInlining(
    ASTContext &context, const std::set<unsigned> &textureOffsets,
    const std::function<bool(llvm::StringRef, SourceLocation)> &ownedQualifier) {
  References references;
  references.TraverseDecl(context.getTranslationUnitDecl());
  BoundedDeviceTemplateEdits edits;
  std::set<unsigned> offsets;
  for (const auto *family : references.families) {
    const auto location = family->getLocation();
    if (location.isInvalid() || location.isMacroID() ||
        context.getSourceManager().getFileID(location) != context.getSourceManager().getMainFileID() ||
        !references.noEscape(family)) continue;
    FamilyProof proof(context, family, textureOffsets, ownedQualifier);
    std::vector<SourceLocation> locations;
    unsigned instances = 0;
    if (!proof.prove(locations, instances)) continue;
    for (const auto insertion : locations)
      if (offsets.insert(context.getSourceManager().getFileOffset(insertion)).second)
        edits.qualifiers.push_back(insertion);
    ++edits.families; edits.instances += instances;
  }
  return edits;
}
} // namespace ascify
