#include "NttpVoidCall.h"

#include <map>
#include <set>
#include <utility>
#include <cstdint>
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
using Family = const FunctionTemplateDecl*;
using Site = std::pair<unsigned, unsigned>;

Family family(const FunctionDecl* function) {
  if (!function) return nullptr;
  const auto* result = function->getPrimaryTemplate();
  if (!result) result = function->getDescribedFunctionTemplate();
  return result ? result->getCanonicalDecl() : nullptr;
}
bool device(const FunctionDecl* function) {
  return function && (function->hasAttr<CUDADeviceAttr>() ||
                      function->hasAttr<CUDAGlobalAttr>());
}
const Expr* plain(const Expr* value) {
  if (!value) return nullptr;
  value = value->IgnoreParenImpCasts();
  if (const auto* clean = dyn_cast<ExprWithCleanups>(value)) return plain(clean->getSubExpr());
  return value;
}
struct Invocation {
  const FunctionDecl* owner;
  const CallExpr* call;
  bool observation;
};
class Inventory : public RecursiveASTVisitor<Inventory> {
  const FunctionDecl* current = nullptr;
  unsigned typeDepth = 0, observationDepth = 0;
 public:
  std::vector<Invocation> calls;
  std::set<const FunctionDecl*> functions;
  std::set<const Expr*> directCallees;
  std::set<const Expr*> kernelLaunchCallees;
  std::vector<const DeclRefExpr*> references;
  std::vector<const UnresolvedLookupExpr*> lookups;
  std::vector<const Expr*> expressions;
  struct ObservedType { QualType type; SourceLocation location; const FunctionDecl* owner; };
  std::vector<ObservedType> observedTypes;
  std::map<const Expr*, const FunctionDecl*> expressionOwners;
  std::map<const UnresolvedLookupExpr*, const FunctionDecl*> lookupOwners;
  std::set<std::pair<const FunctionDecl*, const CallExpr*>> seenCalls;
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }
  bool TraverseDecl(Decl* declaration) {
    const auto* saved = current;
    if (const auto* function = dyn_cast_or_null<FunctionDecl>(declaration)) {
      current = function;
      functions.insert(function);
    }
    const bool accepted = RecursiveASTVisitor<Inventory>::TraverseDecl(declaration);
    current = saved;
    return accepted;
  }
  bool TraverseTypeLoc(TypeLoc type, bool TraverseQualifier = true) {
    ++typeDepth;
    const bool accepted = RecursiveASTVisitor<Inventory>::TraverseTypeLoc(type, TraverseQualifier);
    --typeDepth;
    return accepted;
  }
  bool TraverseStmt(Stmt* statement) {
    const bool observes = statement &&
        (isa<UnaryExprOrTypeTraitExpr>(statement) || isa<CXXNoexceptExpr>(statement) ||
         isa<CXXTypeidExpr>(statement) || isa<TypeTraitExpr>(statement));
    observationDepth += observes;
    const bool accepted = RecursiveASTVisitor<Inventory>::TraverseStmt(statement);
    observationDepth -= observes;
    return accepted;
  }
  bool VisitExpr(Expr* expression) {
    expressions.push_back(expression);
    expressionOwners[expression] = current;
    return true;
  }
  bool VisitTypeLoc(TypeLoc type) {
    const auto* written = type.getType().getTypePtr();
    if (isa<DecltypeType>(written) || isa<TypeOfExprType>(written) ||
        isa<TemplateSpecializationType>(written))
      observedTypes.push_back({type.getType(), type.getBeginLoc(), current});
    return true;
  }
  bool VisitCallExpr(CallExpr* call) {
    if (!seenCalls.emplace(current, call).second) return true;
    calls.push_back({current, call, typeDepth != 0 || observationDepth != 0});
    directCallees.insert(plain(call->getCallee()));
    if (isa<CUDAKernelCallExpr>(call)) kernelLaunchCallees.insert(plain(call->getCallee()));
    return true;
  }
  bool VisitDeclRefExpr(DeclRefExpr* reference) {
    references.push_back(reference);
    return true;
  }
  bool VisitUnresolvedLookupExpr(UnresolvedLookupExpr* lookup) {
    lookups.push_back(lookup);
    lookupOwners[lookup] = current;
    return true;
  }
};

class Proof {
  ASTContext& context;
  SourceManager& source;
  const std::function<bool(SourceLocation)>& owned;
  const std::function<bool(const CXXRecordDecl*)>& cgRecord;
  const std::function<bool(const FunctionDecl*)>& cgFunction;
  const std::function<bool(llvm::StringRef, SourceLocation)>& macroFree;
  NttpVoidCallPlan& output;
  SourceLocation& badLocation;
  std::string& error;
  Inventory inventory;
  struct Pattern { Invocation invocation; const UnresolvedLookupExpr* lookup; Family caller, callee; };
  std::map<Site, Pattern> patterns;
  std::set<Family> families;
  std::set<Site> qualifications;
  NttpVoidCallPlan staged;

  bool fail(SourceLocation location, const std::string& message) {
    if (error.empty()) { badLocation = location; error = message; }
    return false;
  }
  bool literal(SourceLocation location) const {
    return location.isFileID() && owned(location);
  }
  Site site(SourceLocation location) const {
    const auto position = source.getDecomposedLoc(source.getSpellingLoc(location));
    return {position.first.getHashValue(), position.second};
  }
  bool sourceFunction(const FunctionDecl* function) const {
    return device(function) && function->getIdentifier() && literal(function->getLocation());
  }
  bool cg(QualType type) const {
    if (type.isNull() || type->isDependentType()) return false;
    const auto* record = type.getNonReferenceType()->getAsCXXRecordDecl();
    return record && cgRecord(record->getCanonicalDecl());
  }
  bool uintType(QualType type) const {
    return !type.isNull() && context.hasSameUnqualifiedType(type, context.UnsignedIntTy);
  }
  bool boolType(QualType type) const {
    return !type.isNull() && context.hasSameUnqualifiedType(type, context.BoolTy);
  }
  bool scalar(QualType type) const {
    return uintType(type) || boolType(type) ||
        (!type.isNull() && context.hasSameUnqualifiedType(type, context.FloatTy));
  }
  bool floatPointer(QualType type) const {
    if (type.isNull()) return false;
    const auto* pointer = type->getAs<PointerType>();
    return pointer && context.hasSameUnqualifiedType(pointer->getPointeeType(), context.FloatTy) &&
        !pointer->getPointeeType()->isAtomicType();
  }
  bool interfaceType(QualType type, bool kernel) const {
    return !type.isNull() && !type->isDependentType() &&
        (scalar(type) || floatPointer(type) || (!kernel && !type->isReferenceType() && cg(type)));
  }
  bool domain(Family primary) const {
    if (!primary) return false;
    const auto* function = primary->getTemplatedDecl();
    const auto* parameters = primary->getTemplateParameters();
    if (!parameters->size() || !function->getReturnType()->isVoidType()) return false;
    for (const auto* declaration : *parameters) {
      const auto* parameter = dyn_cast<NonTypeTemplateParmDecl>(declaration);
      if (!parameter || parameter->isParameterPack() ||
          (!uintType(parameter->getType()) && !boolType(parameter->getType()))) return false;
    }
    for (const auto* parameter : function->parameters())
      if (!interfaceType(parameter->getType(), function->hasAttr<CUDAGlobalAttr>())) return false;
    return true;
  }
  bool bodyTokens(const FunctionDecl* function) {
    const auto* body = function->getBody();
    if (!body || !literal(body->getBeginLoc()) || !literal(body->getEndLoc()))
      return fail(function->getLocation(), "closed void template requires an owned literal definition");
    if (source.getFileID(body->getBeginLoc()) != source.getFileID(body->getEndLoc()))
      return fail(body->getBeginLoc(), "void template body must belong to one physical input file");
    auto cursor = body->getBeginLoc();
    const auto end = Lexer::getLocForEndOfToken(body->getEndLoc(), 0, source, context.getLangOpts());
    if (!literal(end) || source.getFileID(end) != source.getFileID(cursor) ||
        source.getFileOffset(end) <= source.getFileOffset(cursor))
      return fail(cursor, "void template body has no valid monotone literal token range");
    while (cursor.isValid() && source.getFileOffset(cursor) < source.getFileOffset(end)) {
      Token token;
      if (Lexer::getRawToken(cursor, token, source, context.getLangOpts(), true))
        return fail(cursor, "void template body token provenance is not proven");
      const auto spelling = Lexer::getSpelling(token, source, context.getLangOpts());
      if (token.is(tok::hash) || token.is(tok::hashhash) ||
          spelling == "_Pragma" || spelling == "__pragma")
        return fail(token.getLocation(), "void template body preprocessing alternatives are not admitted");
      const auto next = Lexer::getLocForEndOfToken(token.getLocation(), 0, source, context.getLangOpts());
      if (next.isInvalid() || source.getFileOffset(next) <= source.getFileOffset(cursor))
        return fail(cursor, "void template body token range is not proven");
      cursor = next;
    }
    return true;
  }
  bool signature(const FunctionDecl* function, bool kernel) {
    const auto* prototype = function->getType()->getAs<FunctionProtoType>();
    if (!sourceFunction(function) || !function->getDeclContext()->isTranslationUnit() ||
        isa<CXXMethodDecl>(function) || function->isVariadic() ||
        !function->getReturnType()->isVoidType() || function->isConstexpr() ||
        function->isConsteval() || function->hasAttr<CUDAHostAttr>() ||
        !prototype || prototype->getExceptionSpecType() != EST_None ||
        prototype->getCallConv() != CC_C || (!kernel && function->hasAttr<CUDAGlobalAttr>()))
      return fail(function->getLocation(), "void template requires a fixed ordinary device interface");
    for (const auto* parameter : function->parameters())
      if (parameter->hasDefaultArg() || !interfaceType(parameter->getType(), kernel))
        return fail(parameter->getLocation(), "void template has an unproved parameter/default/conversion");
    for (const auto* attribute : function->attrs())
      if (!isa<CUDADeviceAttr>(attribute) && !isa<CUDAGlobalAttr>(attribute) &&
          !isa<AlwaysInlineAttr>(attribute))
        return fail(function->getLocation(), "void template has an unproved attribute or ABI");
    return true;
  }
  bool integral(const TemplateArgument& argument, QualType type, uint64_t& value) const {
    if ((!uintType(type) && !boolType(type)) || argument.getKind() != TemplateArgument::Integral ||
        !context.hasSameUnqualifiedType(argument.getIntegralType(), type)) return false;
    const auto& integer = argument.getAsIntegral();
    if ((integer.isSigned() && integer.isNegative()) || integer.getActiveBits() > 32 ||
        (uintType(type) && context.getIntWidth(type) != 32)) return false;
    value = integer.getZExtValue();
    return !boolType(type) || value <= 1;
  }
  bool values(const FunctionDecl* function, Family primary, std::vector<uint64_t>& result) const {
    if (!function || family(function) != primary) return false;
    const auto* arguments = function->getTemplateSpecializationArgs();
    const auto* parameters = primary->getTemplateParameters();
    if (!arguments || arguments->size() != parameters->size()) return false;
    for (unsigned i = 0; i < parameters->size(); ++i) {
      const auto* parameter = dyn_cast<NonTypeTemplateParmDecl>(parameters->getParam(i));
      uint64_t value = 0;
      if (!parameter || !integral(arguments->get(i), parameter->getType(), value)) return false;
      result.push_back(value);
    }
    return true;
  }
  bool proveFamily(Family primary) {
    if (!primary || families.count(primary)) return primary != nullptr;
    const auto* function = primary->getTemplatedDecl();
    const auto* parameters = primary->getTemplateParameters();
    if (!parameters->size() || parameters->getRequiresClause() ||
        function->getTrailingRequiresClause() || function->getPreviousDecl() ||
        function->getMostRecentDecl() != function)
      return fail(function->getLocation(), "void template constraints/redeclarations are not closed");
    if (!signature(function, function->hasAttr<CUDAGlobalAttr>()) || !bodyTokens(function)) return false;
    for (const auto* declaration : *parameters) {
      const auto* parameter = dyn_cast<NonTypeTemplateParmDecl>(declaration);
      if (!parameter || !parameter->getIdentifier() || parameter->isParameterPack() ||
          parameter->hasDefaultArgument() || (!uintType(parameter->getType()) && !boolType(parameter->getType())))
        return fail(declaration->getLocation(), "void template admits only fixed unsigned-int/bool NTTPs");
      if (!literal(parameter->getLocation()) || !macroFree(parameter->getName(), parameter->getLocation()))
        return fail(parameter->getLocation(), "void template parameter token provenance is not proven");
    }
    unsigned count = 0;
    for (const auto* instance : primary->specializations()) {
      std::vector<uint64_t> arguments;
      if (instance->getTemplateSpecializationKind() == TSK_ExplicitSpecialization ||
          !values(instance, primary, arguments) || !instance->hasBody() ||
          !signature(instance, function->hasAttr<CUDAGlobalAttr>()))
        return fail(instance->getLocation(), "void template has an unproved concrete specialization");
      ++count;
    }
    if (!count) return fail(function->getLocation(), "void template has no closed concrete instantiation");
    families.insert(primary);
    return true;
  }
  bool discarded(const CallExpr* call) const {
    const auto parents = context.getParents(*call);
    if (parents.size() != 1) return false;
    if (parents[0].get<CompoundStmt>()) return true;
    if (const auto* branch = parents[0].get<IfStmt>())
      return branch->getThen() == call || branch->getElse() == call;
    if (const auto* loop = parents[0].get<ForStmt>()) return loop->getBody() == call;
    if (const auto* loop = parents[0].get<WhileStmt>()) return loop->getBody() == call;
    return false;
  }
  Family lookupFamily(const UnresolvedLookupExpr* lookup) const {
    if (!lookup || lookup->getQualifier() || !lookup->hasExplicitTemplateArgs()) return nullptr;
    auto iterator = lookup->decls_begin();
    if (iterator == lookup->decls_end()) return nullptr;
    const auto* declaration = dyn_cast<FunctionTemplateDecl>(*iterator++);
    return iterator == lookup->decls_end() && declaration ? declaration->getCanonicalDecl() : nullptr;
  }
  bool argumentSource(const TemplateArgumentLoc& argument, Family caller, SourceLocation callSite) const {
    const auto range = argument.getSourceRange();
    const auto begin = range.getBegin(), last = range.getEnd();
    if (!literal(begin) || !literal(last) || !literal(callSite) ||
        source.getFileID(begin) != source.getFileID(callSite) ||
        source.getFileID(begin) != source.getFileID(last) ||
        source.getFileOffset(begin) > source.getFileOffset(last)) return false;
    Token token;
    if (Lexer::getRawToken(begin, token, source, context.getLangOpts(), true) ||
        token.getLocation() != begin || token.needsCleaning()) return false;
    const auto tokenEnd = Lexer::getLocForEndOfToken(token.getLocation(), 0, source, context.getLangOpts());
    const auto rangeEnd = Lexer::getLocForEndOfToken(last, 0, source, context.getLangOpts());
    if (!literal(tokenEnd) || tokenEnd != rangeEnd) return false;
    if (token.is(tok::raw_identifier)) {
      const auto name = token.getRawIdentifier();
      if (!macroFree(name, begin)) return false; // Definition-time checks do not prove this use.
      if (name == "true" || name == "false") return true;
      if (caller)
        for (const auto* declaration : *caller->getTemplateParameters())
          if (declaration->getIdentifier() && declaration->getName() == name) return true;
      return false;
    }
    if (token.isNot(tok::numeric_constant)) return false;
    // A single builtin integer token, not a UDL, constexpr call or alias whose
    // folded value merely happens to equal the desired constant.
    const auto spelling = Lexer::getSpelling(token, source, context.getLangOpts());
    for (const char character : spelling)
      if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f') ||
            (character >= 'A' && character <= 'F') || character == 'x' || character == 'X' ||
            character == 'u' || character == 'U' || character == 'l' || character == 'L' ||
            character == '\'')) return false;
    return true;
  }
  bool argumentValue(const TemplateArgument& argument, QualType formal, Family caller,
                     const std::vector<uint64_t>& callerValues, uint64_t& result) const {
    if (argument.getKind() == TemplateArgument::Integral) return integral(argument, formal, result);
    if (argument.getKind() != TemplateArgument::Expression) return false;
    const auto* expression = plain(argument.getAsExpr());
    if (const auto* reference = dyn_cast_or_null<DeclRefExpr>(expression)) {
      const auto* parameters = caller->getTemplateParameters();
      for (unsigned index = 0; index < parameters->size(); ++index)
        if (reference->getDecl() == parameters->getParam(index)) {
          const auto* parameter = dyn_cast<NonTypeTemplateParmDecl>(parameters->getParam(index));
          if (!parameter || !context.hasSameUnqualifiedType(parameter->getType(), formal)) return false;
          result = callerValues[index];
          return true;
        }
      return false;
    }
    if (!expression || (!isa<IntegerLiteral>(expression) && !isa<CXXBoolLiteralExpr>(expression))) return false;
    Expr::EvalResult evaluated;
    if (!expression->EvaluateAsInt(evaluated, context)) return false;
    const auto& integer = evaluated.Val.getInt();
    if ((integer.isSigned() && integer.isNegative()) || integer.getActiveBits() > 32) return false;
    result = integer.getZExtValue();
    return uintType(formal) || (boolType(formal) && result <= 1);
  }
  bool ordinaryAdl(Family primary) {
    // A real CG class has an associated namespace. Check source injection and
    // freeze the proven global binding with ::, rather than assuming no ADL.
    const auto name = primary->getName();
    for (const auto* function : inventory.functions) {
      if (!function->getIdentifier() || function->getName() != name ||
          function->getDeclContext()->isTranslationUnit()) continue;
      const DeclContext* owner = function->getDeclContext();
      while (owner && !owner->isTranslationUnit()) {
        if (const auto* space = dyn_cast<NamespaceDecl>(owner))
          if (space->getQualifiedNameAsString() == "cooperative_groups")
            return fail(function->getLocation(), "void template CG-associated overload/injection is not closed");
        owner = owner->getParent();
      }
    }
    return true;
  }
  bool sameOwner(const VarDecl* variable, const FunctionDecl* owner) const {
    const auto* declaration = variable ? dyn_cast<FunctionDecl>(variable->getDeclContext()) : nullptr;
    return declaration && owner && declaration->getCanonicalDecl() == owner->getCanonicalDecl();
  }
  bool pointer(const Expr* expression, QualType formal, const FunctionDecl* owner) const {
    // Preserve pointee const/volatile. Only standard qualification and array
    // decay may lie between the exact root and this parameter conversion.
    const auto* value = expression;
    while (true) {
      if (const auto* paren = dyn_cast<ParenExpr>(value)) { value = paren->getSubExpr(); continue; }
      const auto* cast = dyn_cast<ImplicitCastExpr>(value);
      if (!cast) break;
      if (cast->getCastKind() != CK_NoOp && cast->getCastKind() != CK_ArrayToPointerDecay &&
          cast->getCastKind() != CK_LValueToRValue) return false;
      value = cast->getSubExpr();
    }
    const auto* reference = dyn_cast<DeclRefExpr>(value);
    const auto* variable = reference ? dyn_cast<VarDecl>(reference->getDecl()) : nullptr;
    if (!variable || !literal(variable->getLocation())) return false;
    if (isa<ParmVarDecl>(variable)) {
      if (!sameOwner(variable, owner)) return false;
    } else {
      const auto* lexical = dyn_cast<FunctionDecl>(variable->getLexicalDeclContext());
      const auto* array = context.getAsArrayType(variable->getType());
      if (!owner || !lexical || lexical->getCanonicalDecl() != owner->getCanonicalDecl() ||
          !variable->isLocalVarDecl() || !array ||
          !context.hasSameUnqualifiedType(array->getElementType(), context.FloatTy)) return false;
      if (variable->getStorageClass() == SC_Extern) {
        // Instantiated block-scope extern shared roots have a nonlocal semantic
        // context. Only this direct CUDA qualifier keeps their lexical owner;
        // the existing Runtime qualifier/provider audits still apply.
        const auto* shared = variable->getAttr<CUDASharedAttr>();
        if (!shared) return false;
        const auto location = source.getExpansionLoc(shared->getLocation());
        Token token;
        if (!literal(location) ||
            Lexer::getRawToken(location, token, source, context.getLangOpts(), true) ||
            token.getLocation() != location || token.isNot(tok::raw_identifier) || token.needsCleaning() ||
            Lexer::getSpelling(token, source, context.getLangOpts()) != "__shared__") return false;
      } else if (!sameOwner(variable, owner)) return false;
    }
    auto actual = expression->getType();
    if (!actual->isPointerType() || !floatPointer(actual) || !floatPointer(formal)) return false;
    return context.hasSameType(actual.getCanonicalType(), formal.getCanonicalType());
  }
  bool group(const Expr* expression, const FunctionDecl* owner, unsigned depth = 0) const {
    if (!expression || depth > 16 || !cg(expression->getType())) return false;
    expression = plain(expression);
    if (const auto* temporary = dyn_cast<MaterializeTemporaryExpr>(expression))
      return group(temporary->getSubExpr(), owner, depth + 1);
    if (const auto* bind = dyn_cast<CXXBindTemporaryExpr>(expression))
      return group(bind->getSubExpr(), owner, depth + 1);
    if (const auto* construct = dyn_cast<CXXConstructExpr>(expression)) {
      const auto* constructor = construct->getConstructor();
      return construct->getNumArgs() == 1 && constructor->isImplicit() &&
          constructor->isCopyConstructor() && cgRecord(constructor->getParent()->getCanonicalDecl()) &&
          group(construct->getArg(0), owner, depth + 1);
    }
    if (const auto* call = dyn_cast<CallExpr>(expression)) {
      const auto* producer = call->getDirectCallee();
      return producer && producer->getQualifiedNameAsString() == "cooperative_groups::this_thread_block" &&
          call->getNumArgs() == 0 && cgFunction(producer) && literal(call->getExprLoc()) &&
          literal(call->getCallee()->getBeginLoc()) && macroFree(producer->getName(), call->getExprLoc());
    }
    const auto* reference = dyn_cast<DeclRefExpr>(expression);
    const auto* variable = reference ? dyn_cast<VarDecl>(reference->getDecl()) : nullptr;
    if (!variable || !sameOwner(variable, owner) || !literal(variable->getLocation())) return false;
    if (isa<ParmVarDecl>(variable)) {
      return owner && !owner->hasAttr<CUDAGlobalAttr>() && families.count(family(owner));
    }
    return variable->hasLocalStorage() && variable->hasInit() && group(variable->getInit(), owner, depth + 1);
  }
  bool actuals(const CallExpr* call, const FunctionDecl* callee, const FunctionDecl* owner) const {
    if (call->getNumArgs() != callee->getNumParams()) return false;
    for (unsigned index = 0; index < call->getNumArgs(); ++index) {
      const auto* argument = call->getArg(index);
      const auto formal = callee->getParamDecl(index)->getType().getUnqualifiedType();
      if (floatPointer(formal)) { if (!pointer(argument, formal, owner)) return false; }
      else if (cg(formal)) { if (!group(argument, owner)) return false; }
      else {
        const auto* value = plain(argument);
        if (!scalar(formal) || !context.hasSameUnqualifiedType(argument->getType(), formal) ||
            value->getType().isVolatileQualified() || value->getType()->isAtomicType() ||
            (!isa<DeclRefExpr>(value) && !isa<IntegerLiteral>(value) &&
             !isa<FloatingLiteral>(value) && !isa<CXXBoolLiteralExpr>(value))) return false;
      }
    }
    return true;
  }
  bool qualify(SourceLocation location) {
    if (!literal(location))
      return fail(location, "void call qualification requires an owned macro-free identifier");
    Token token;
    if (Lexer::getRawToken(location, token, source, context.getLangOpts(), true) ||
        token.isNot(tok::raw_identifier) || token.needsCleaning())
      return fail(location, "void call qualification identifier is not literal");
    if (!macroFree(token.getRawIdentifier(), location))
      return fail(location, "input macro changes the void call identifier");
    if (!qualifications.insert(site(location)).second) return true;
    if (source.isWrittenInMainFile(location)) staged.qualifications.push_back(location);
    else staged.contextualHeaders.insert(source.getFilename(location).str());
    return true;
  }
  bool provePattern(const Pattern& pattern) {
    const auto* parameters = pattern.callee->getTemplateParameters();
    if (pattern.lookup->getNumTemplateArgs() != parameters->size())
      return fail(pattern.lookup->getNameLoc(), "void call must provide every NTTP explicitly");
    for (const auto& argument : pattern.lookup->template_arguments())
      if (!argumentSource(argument, pattern.caller, pattern.lookup->getNameLoc()))
        return fail(pattern.lookup->getNameLoc(), "void call explicit NTTP token/macro provenance is not proven");
    for (const auto* instance : pattern.caller->specializations()) {
      std::vector<uint64_t> callerValues;
      if (!values(instance, pattern.caller, callerValues))
        return fail(instance->getLocation(), "void caller substitution is not closed");
      unsigned found = 0;
      for (const auto& invocation : inventory.calls) {
        if (!invocation.owner || invocation.owner->getCanonicalDecl() != instance->getCanonicalDecl() ||
            site(invocation.call->getExprLoc()) != site(pattern.invocation.call->getExprLoc())) continue;
        const auto* callee = invocation.call->getDirectCallee();
        std::vector<uint64_t> calleeValues;
        if (!callee || !values(callee, pattern.callee, calleeValues) || invocation.observation ||
            !invocation.call->getType()->isVoidType() || !discarded(invocation.call) ||
            !actuals(invocation.call, callee, invocation.owner))
          return fail(invocation.call->getExprLoc(), "void call concrete binding/interface/observation is not proven");
        unsigned index = 0;
        for (const auto& location : pattern.lookup->template_arguments()) {
          const auto* formal = cast<NonTypeTemplateParmDecl>(parameters->getParam(index));
          uint64_t expected = 0;
          if (!argumentValue(location.getArgument(), formal->getType(), pattern.caller, callerValues, expected) ||
              calleeValues[index] != expected)
            return fail(pattern.lookup->getNameLoc(), "void call caller-to-callee NTTP substitution is not exact");
          ++index;
        }
        ++found;
        ++staged.concreteCalls;
      }
      if (found != 1) return fail(pattern.lookup->getNameLoc(), "void call lacks a unique occurrence in every caller instance");
    }
    staged.expressions.insert(pattern.invocation.call);
    staged.expressions.insert(pattern.lookup);
    ++staged.patterns;
    return qualify(pattern.lookup->getNameLoc());
  }
  bool escapeAndConcreteCalls() {
    for (const auto* reference : inventory.references) {
      const auto* function = dyn_cast<FunctionDecl>(reference->getDecl());
      if (function && families.count(family(function)) && !inventory.directCallees.count(reference))
        return fail(reference->getLocation(), "closed void template has an address/indirect escape");
    }
    for (const auto* lookup : inventory.lookups) {
      bool mentions = false;
      for (const auto* declaration : lookup->decls()) {
        if (const auto* shadow = dyn_cast<UsingShadowDecl>(declaration)) declaration = shadow->getTargetDecl();
        const auto* candidate = dyn_cast<FunctionTemplateDecl>(declaration);
        mentions |= candidate && families.count(candidate->getCanonicalDecl());
      }
      if (inventory.kernelLaunchCallees.count(lookup)) continue; // Existing launch policy, no new fact.
      const auto owner = family(inventory.lookupOwners.at(lookup));
      if (mentions && (!families.count(owner) ||
          !patterns.count(site(lookup->getNameLoc())) || !inventory.directCallees.count(lookup)))
        return fail(lookup->getNameLoc(), "closed void template has an unproved lookup/using/observation");
    }
    for (const auto& invocation : inventory.calls) {
      const auto* callee = invocation.call->getDirectCallee();
      if (!callee || !families.count(family(callee)) || callee->hasAttr<CUDAGlobalAttr>()) continue;
      const auto owner = family(invocation.owner);
      if (!families.count(owner))
        return fail(invocation.call->getExprLoc(), "admitted void template is called outside its closed families");
      if (!invocation.owner || !sourceFunction(invocation.owner) || invocation.observation ||
          !discarded(invocation.call) || !actuals(invocation.call, callee, invocation.owner))
        return fail(invocation.call->getExprLoc(), "closed void template has an unproved concrete caller");
      const auto* reference = dyn_cast<DeclRefExpr>(plain(invocation.call->getCallee()));
      if (!reference || isa<UsingShadowDecl>(reference->getFoundDecl()))
        return fail(invocation.call->getExprLoc(), "closed void template concrete callee is not direct");
      if (!reference->hasExplicitTemplateArgs() ||
          reference->getNumTemplateArgs() != family(callee)->getTemplateParameters()->size())
        return fail(reference->getLocation(), "closed void template concrete NTTP list is not explicit");
      for (const auto& argument : reference->template_arguments())
        if (!argumentSource(argument, owner, reference->getLocation()))
          return fail(reference->getLocation(), "closed void call concrete NTTP token/macro provenance is not proven");
      if (!reference->hasQualifier() && !qualify(reference->getLocation())) return false;
    }
    return true;
  }
  bool relatedCg(QualType type, std::set<const Type*>& visited, unsigned depth = 0) const {
    if (type.isNull()) return false;
    if (depth > 32) return true; // An unbounded layout observation is not a proof.
    const auto* raw = type.getTypePtr();
    if (!visited.insert(raw).second) return false;
    if (cg(type)) return true;
    if (type->isPointerType() || type->isReferenceType())
      return relatedCg(type->getPointeeType(), visited, depth + 1);
    if (const auto* array = context.getAsArrayType(type))
      return relatedCg(array->getElementType(), visited, depth + 1);
    const auto* record = type->getAsCXXRecordDecl();
    if (!record) return false;
    if (const auto* specialization = dyn_cast<ClassTemplateSpecializationDecl>(record))
      for (const auto& argument : specialization->getTemplateArgs().asArray())
        if (argument.getKind() == TemplateArgument::Type &&
            relatedCg(argument.getAsType(), visited, depth + 1)) return true;
    if (const auto* definition = record->getDefinition()) {
      for (const auto* field : definition->fields())
        if (relatedCg(field->getType(), visited, depth + 1)) return true;
      for (const auto& base : definition->bases())
        if (relatedCg(base.getType(), visited, depth + 1)) return true;
    }
    return false;
  }
  bool relatedCg(QualType type) const {
    std::set<const Type*> visited;
    return relatedCg(type, visited);
  }
  bool cgUses() {
    for (const auto& type : inventory.observedTypes)
      if (families.count(family(type.owner)) && literal(type.location) && relatedCg(type.type))
        return fail(type.location, "closed void calls do not admit CG type/trait/layout observations");
    for (const auto* expression : inventory.expressions) {
      if (!families.count(family(inventory.expressionOwners.at(expression))) ||
          !literal(expression->getExprLoc())) continue;
      if (const auto* trait = dyn_cast<UnaryExprOrTypeTraitExpr>(expression))
        if (relatedCg(trait->getTypeOfArgument()))
          return fail(trait->getExprLoc(), "closed void calls do not admit CG sizeof/alignof observations");
      if (const auto* trait = dyn_cast<TypeTraitExpr>(expression))
        for (unsigned i = 0; i < trait->getNumArgs(); ++i)
          if (relatedCg(trait->getArg(i)->getType()))
            return fail(trait->getExprLoc(), "closed void calls do not admit CG type traits");
      if (const auto* typeidExpression = dyn_cast<CXXTypeidExpr>(expression)) {
        const auto type = typeidExpression->isTypeOperand() ? typeidExpression->getTypeOperand(context)
                                                          : typeidExpression->getExprOperand()->getType();
        if (relatedCg(type))
          return fail(typeidExpression->getExprLoc(), "closed void calls do not admit CG RTTI");
      }
      if (const auto* cast = dyn_cast<ExplicitCastExpr>(expression))
        if (relatedCg(cast->getType()) || relatedCg(cast->getSubExpr()->getType()))
          return fail(cast->getExprLoc(), "closed void calls do not admit CG conversion/layout casts");
      if (const auto* unary = dyn_cast<UnaryOperator>(expression))
        if (unary->getOpcode() == UO_AddrOf && cg(unary->getSubExpr()->getType()))
          return fail(unary->getExprLoc(), "closed void call CG object has an address escape");
      if (const auto* operation = dyn_cast<CXXOperatorCallExpr>(expression))
        if (operation->getNumArgs() && cg(operation->getArg(0)->getType()))
          return fail(operation->getExprLoc(), "closed void call CG object uses an unproved operator");
    }
    for (const auto& invocation : inventory.calls) {
      if (!families.count(family(invocation.owner)) || !literal(invocation.call->getExprLoc())) continue;
      bool takesGroup = false;
      for (const auto* argument : invocation.call->arguments()) takesGroup |= cg(argument->getType());
      const auto* callee = invocation.call->getDirectCallee();
      if (takesGroup && callee && !families.count(family(callee)) && !cgFunction(callee))
        return fail(invocation.call->getExprLoc(), "closed void call CG object escapes to an unproved function");
    }
    return true;
  }
  bool acyclic() {
    std::map<Family, std::set<Family>> edges;
    for (const auto& invocation : inventory.calls) {
      const auto caller = family(invocation.owner);
      const auto callee = family(invocation.call->getDirectCallee());
      if (families.count(caller) && families.count(callee)) edges[caller].insert(callee);
    }
    std::set<Family> active, finished;
    const std::function<bool(Family)> visit = [&](Family value) {
      if (finished.count(value)) return true;
      if (!active.insert(value).second)
        return fail(value->getLocation(), "closed void template recursion is not admitted");
      for (const auto target : edges[value]) if (!visit(target)) return false;
      active.erase(value);
      finished.insert(value);
      return true;
    };
    for (const auto value : families) if (!visit(value)) return false;
    return true;
  }
 public:
  Proof(ASTContext& ctx, const std::function<bool(SourceLocation)>& input,
        const std::function<bool(const CXXRecordDecl*)>& record,
        const std::function<bool(const FunctionDecl*)>& function,
        const std::function<bool(llvm::StringRef, SourceLocation)>& macros,
        NttpVoidCallPlan& result, SourceLocation& location, std::string& diagnostic)
      : context(ctx), source(ctx.getSourceManager()), owned(input), cgRecord(record),
        cgFunction(function), macroFree(macros), output(result), badLocation(location), error(diagnostic) {}
  bool run(bool standard) {
    inventory.TraverseDecl(context.getTranslationUnitDecl());
    for (const auto& invocation : inventory.calls) {
      const auto* call = invocation.call;
      if (isa<CUDAKernelCallExpr>(call)) continue; // This proof does not project kernel launches.
      const auto* lookup = dyn_cast<UnresolvedLookupExpr>(plain(call->getCallee()));
      const auto callee = lookupFamily(lookup);
      const auto caller = family(invocation.owner);
      // Do not impose this new policy on existing int/enum/type templates or
      // other interfaces. They receive no new facts and keep the old audits.
      if (!domain(callee) || !domain(caller) || !sourceFunction(callee->getTemplatedDecl()) ||
          !callee->getTemplatedDecl()->getReturnType()->isVoidType() ||
          !call->getType()->isDependentType()) continue;
      if (!standard || !caller || !literal(call->getExprLoc()) || invocation.observation || !discarded(call))
        return fail(call->getExprLoc(), "dependent void call requires a closed evaluated literal caller");
      if (!proveFamily(caller) || !proveFamily(callee)) return false;
      patterns.emplace(site(call->getExprLoc()), Pattern{invocation, lookup, caller, callee});
    }
    if (patterns.empty()) return true;
    for (const auto primary : families) if (!ordinaryAdl(primary)) return false;
    for (const auto& entry : patterns) if (!provePattern(entry.second)) return false;
    if (!escapeAndConcreteCalls() || !cgUses() || !acyclic()) return false;
    output = std::move(staged);
    return true;
  }
};
} // namespace

bool ProveNttpVoidCalls(ASTContext& context,
    const std::function<bool(SourceLocation)>& ownedInput,
    const std::function<bool(const CXXRecordDecl*)>& trustedCgRecord,
    const std::function<bool(const FunctionDecl*)>& trustedCgFunction,
    const std::function<bool(llvm::StringRef, SourceLocation)>& macroFree,
    bool standardPreprocessing, NttpVoidCallPlan& plan,
    SourceLocation& errorLocation, std::string& error) {
  NttpVoidCallPlan staged;
  if (!Proof(context, ownedInput, trustedCgRecord, trustedCgFunction, macroFree,
             staged, errorLocation, error).run(standardPreprocessing)) return false;
  plan = std::move(staged);
  return true;
}
} // namespace ascify
