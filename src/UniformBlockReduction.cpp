#include "UniformBlockReduction.h"
#include "FrontendCompatibility.h"

#include <map>
#include <cctype>
#include <set>
#include <string>
#include <vector>

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

const Expr* plain(const Expr* expression) {
  if (!expression) return nullptr;
  expression = expression->IgnoreParenImpCasts();
  if (const auto* clean = dyn_cast<ExprWithCleanups>(expression))
    return plain(clean->getSubExpr());
  if (const auto* temporary = dyn_cast<MaterializeTemporaryExpr>(expression))
    return plain(temporary->getSubExpr());
  if (const auto* bind = dyn_cast<CXXBindTemporaryExpr>(expression))
    return plain(bind->getSubExpr());
  return expression;
}
const DeclRefExpr* reference(const Expr* expression) {
  return dyn_cast_or_null<DeclRefExpr>(plain(expression));
}
std::string templateName(QualType type) {
  if (type.isNull()) return {};
  type = type.getNonReferenceType().getUnqualifiedType();
  if (const auto* record = type->getAsCXXRecordDecl()) {
    if (const auto* specialization = dyn_cast<ClassTemplateSpecializationDecl>(record))
      return specialization->getSpecializedTemplate()->getQualifiedNameAsString();
  }
  if (const auto* specialization = type->getAs<TemplateSpecializationType>()) {
    if (const auto* declaration = specialization->getTemplateName().getAsTemplateDecl())
      return declaration->getQualifiedNameAsString();
  }
  return {};
}
enum class Role { None, Storage, Block, Tile };
Role role(QualType type) {
  const auto name = templateName(type);
  if (name == "cooperative_groups::block_tile_memory") return Role::Storage;
  if (name == "cooperative_groups::scratch_thread_block") return Role::Block;
  if (name == "cooperative_groups::scratch_thread_block_tile") return Role::Tile;
  return Role::None;
}
bool related(QualType type) {
  if (type.isNull()) return false;
  if (role(type) != Role::None) return true;
  if (type->isPointerType() || type->isReferenceType()) return related(type->getPointeeType());
  if (const auto* array = dyn_cast<ArrayType>(type.getTypePtr())) return related(array->getElementType());
  return false;
}
bool integerArgument(QualType type, unsigned index, unsigned& output) {
  const auto* record = type.getNonReferenceType()->getAsCXXRecordDecl();
  const auto* specialization = dyn_cast_or_null<ClassTemplateSpecializationDecl>(record);
  if (!specialization || specialization->getTemplateArgs().size() <= index) return false;
  const auto& argument = specialization->getTemplateArgs()[index];
  if (argument.getKind() != TemplateArgument::Integral) return false;
  const auto& value = argument.getAsIntegral();
  if (value.isNegative() || value.getActiveBits() > 16) return false;
  output = static_cast<unsigned>(value.getZExtValue());
  return true;
}
bool scalar(QualType type, ASTContext& context) {
  return context.hasSameUnqualifiedType(type, context.IntTy) ||
         context.hasSameUnqualifiedType(type, context.FloatTy);
}

struct BodyInventory : RecursiveASTVisitor<BodyInventory> {
  std::vector<const VarDecl*> variables;
  std::vector<const DeclRefExpr*> references;
  std::vector<const CallExpr*> calls;
  std::vector<const CXXConstructExpr*> constructions;
  bool forbiddenExit = false;
  bool VisitVarDecl(VarDecl* variable) { variables.push_back(variable); return true; }
  bool VisitDeclRefExpr(DeclRefExpr* expression) { references.push_back(expression); return true; }
  bool VisitCallExpr(CallExpr* expression) { calls.push_back(expression); return true; }
  bool VisitCXXConstructExpr(CXXConstructExpr* expression) { constructions.push_back(expression); return true; }
  bool VisitReturnStmt(ReturnStmt*) { forbiddenExit = true; return true; }
  bool VisitGotoStmt(GotoStmt*) { forbiddenExit = true; return true; }
  bool VisitIndirectGotoStmt(IndirectGotoStmt*) { forbiddenExit = true; return true; }
  bool VisitCXXThrowExpr(CXXThrowExpr*) { forbiddenExit = true; return true; }
  bool VisitAsmStmt(AsmStmt*) { forbiddenExit = true; return true; }
  bool VisitLambdaExpr(LambdaExpr*) { forbiddenExit = true; return true; }
  bool VisitCXXNewExpr(CXXNewExpr*) { forbiddenExit = true; return true; }
  bool VisitCXXDeleteExpr(CXXDeleteExpr*) { forbiddenExit = true; return true; }
  bool VisitForStmt(ForStmt*) { forbiddenExit = true; return true; }
  bool VisitWhileStmt(WhileStmt*) { forbiddenExit = true; return true; }
  bool VisitDoStmt(DoStmt*) { forbiddenExit = true; return true; }
  bool VisitCXXForRangeStmt(CXXForRangeStmt*) { forbiddenExit = true; return true; }
  bool VisitIfStmt(IfStmt* statement) {
    if (statement->isConstexpr()) forbiddenExit = true;
    return true;
  }
};

class Proof {
 public:
  Proof(ASTContext& context, const FrontendCompatibilityConfig& profile,
        UniformBlockReductionStats& stats, std::string& error, bool standard)
      : context(context), source(context.getSourceManager()), profile(profile),
        stats(stats), error(error), standard(standard) {}

  bool run() {
    Collector collector(*this);
    collector.TraverseDecl(context.getTranslationUnitDecl());
    if (!error.empty()) return false;
    if ((!kernels.empty() || !patterns.empty()) && !standard)
      return fail("scratch proof requires --default-preprocessor");
    if ((!kernels.empty() || !patterns.empty()) &&
        (!context.getLangOpts().CPlusPlus17 || context.getLangOpts().CPlusPlus20))
      return fail("scratch proof requires C++17; constrained templates are outside the proof");
    if ((!kernels.empty() || !patterns.empty()) && userCooperativeDeclaration)
      return fail("source declarations in cooperative_groups or ascify_cg can alter ADL and are outside the scratch proof");
    if ((!kernels.empty() || !patterns.empty()) && userUsingDirective)
      return fail("source using-directives can alter overload lookup and are outside the scratch proof");
    if (kernels.empty()) {
      if (!patterns.empty()) return fail("scratch kernel has no concrete instantiation");
      return auditUses() && auditRawTypes();
    }
    for (const auto* kernel : kernels)
      if (!proveKernel(kernel)) return false;
    for (const auto* pattern : patterns) {
      bool instantiated = false;
      for (const auto* kernel : kernels)
        if (kernel->getTemplateInstantiationPattern() == pattern) instantiated = true;
      if (!instantiated) return fail("scratch kernel template has no concrete instantiation");
    }
    stats.kernels = static_cast<unsigned>(kernelLocations.size());
    return auditUses() && auditRawTypes();
  }

 private:
  class Collector : public RecursiveASTVisitor<Collector> {
   public:
    explicit Collector(Proof& proof) : proof(proof) {}
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool VisitDecl(Decl* declaration) {
      if (proof.system(declaration)) return true;
      if (const auto* space = dyn_cast<NamespaceDecl>(declaration))
        if (space->getQualifiedNameAsString() == "cooperative_groups" ||
            space->getQualifiedNameAsString() == "ascify_cg")
          proof.userCooperativeDeclaration = true;
      for (auto* owner = declaration->getDeclContext(); owner; owner = owner->getParent())
        if (const auto* space = dyn_cast<NamespaceDecl>(owner))
          if (space->getQualifiedNameAsString() == "cooperative_groups" ||
              space->getQualifiedNameAsString() == "ascify_cg")
            proof.userCooperativeDeclaration = true;
      return true;
    }
    bool VisitUsingDirectiveDecl(UsingDirectiveDecl* declaration) {
      if (!proof.system(declaration)) proof.userUsingDirective = true;
      return true;
    }
    bool VisitFunctionDecl(FunctionDecl* function) {
      if (!function->doesThisDeclarationHaveABody() || proof.system(function)) return true;
      BodyInventory inventory;
      inventory.TraverseStmt(function->getBody());
      bool storage = false;
      for (const auto* variable : inventory.variables)
        if (role(variable->getType()) == Role::Storage) storage = true;
      if (!storage) return true;
      if (!function->hasAttr<CUDAGlobalAttr>())
        return proof.fail("scratch storage must belong directly to a CUDA kernel");
      if (function->getDescribedFunctionTemplate()) {
        if (inventory.forbiddenExit)
          return proof.fail("template pattern contains a loop, exit, or constexpr branch");
        proof.patterns.insert(function);
      }
      else proof.kernels.insert(function);
      return true;
    }
    bool VisitDeclRefExpr(DeclRefExpr* expression) {
      if (!proof.trustedLocation(expression->getExprLoc()))
        proof.allReferences.push_back(expression);
      return true;
    }
    bool VisitVarDecl(VarDecl* variable) {
      if (!proof.system(variable) && related(variable->getType()))
        proof.allVariables.insert(variable);
      return true;
    }
    bool VisitTypeAliasDecl(TypeAliasDecl* declaration) {
      if (!proof.system(declaration) && related(declaration->getUnderlyingType()))
        return proof.fail("scratch types may not escape through a type alias");
      return true;
    }
    bool VisitTypedefDecl(TypedefDecl* declaration) {
      if (!proof.system(declaration) && related(declaration->getUnderlyingType()))
        return proof.fail("scratch types may not escape through a typedef");
      return true;
    }
    bool VisitTemplateSpecializationTypeLoc(TemplateSpecializationTypeLoc location) {
      if (proof.trustedLocation(location.getBeginLoc())) return true;
      const auto* declaration = location.getTypePtr()->getTemplateName().getAsTemplateDecl();
      if (!declaration) return true;
      const auto name = declaration->getQualifiedNameAsString();
      if (name == "cooperative_groups::block_tile_memory")
        proof.storageTypeLocations.push_back(location.getTemplateNameLoc());
      if (name == "cooperative_groups::scratch_thread_block" ||
          name == "cooperative_groups::scratch_thread_block_tile")
        return proof.fail("scratch block and tile types must remain unobserved auto values");
      return true;
    }
   private:
    Proof& proof;
  };

  bool fail(const std::string& reason) {
    if (error.empty()) error = "Ascify uniform block reduction: " + reason;
    return false;
  }
  bool system(const Decl* declaration) const {
    if (!declaration || declaration->getLocation().isInvalid()) return true;
    return trustedLocation(declaration->getLocation());
  }
  bool trustedLocation(SourceLocation original) const {
    const auto location = source.getSpellingLoc(original);
    const auto path = source.getFilename(location);
    return path == profile.canonicalRoot + "/cooperative_groups.h" ||
           path == profile.canonicalRoot + "/cooperative_groups/reduce.h";
  }
  bool trustedFunction(const FunctionDecl* function) const {
    if (!function || function->isNoReturn() || function->hasAttr<AsmLabelAttr>() ||
        function->getTemplateSpecializationKind() == TSK_ExplicitSpecialization) return false;
    for (const auto* declaration : function->redecls())
      if (!trustedLocation(declaration->getLocation())) return false;
    return true;
  }
  bool owned(const FunctionDecl* function, const char* name) const {
    if (!trustedFunction(function) || function->getQualifiedNameAsString() !=
                         std::string("cooperative_groups::") + name) return false;
    const auto file = source.getFilename(source.getSpellingLoc(function->getLocation()));
    return file == profile.canonicalRoot + "/cooperative_groups.h" ||
           file == profile.canonicalRoot + "/cooperative_groups/reduce.h";
  }
  bool cleanLocation(const Decl* declaration) const {
    return declaration->getLocation().isValid() && !declaration->getLocation().isMacroID() &&
           source.isWrittenInMainFile(declaration->getLocation());
  }
  const CallExpr* directInitializer(const VarDecl* variable) const {
    return dyn_cast_or_null<CallExpr>(plain(variable->getInit()));
  }
  bool use(const Expr* expression, const VarDecl* expected) {
    const auto* ref = reference(expression);
    if (!ref || ref->getDecl() != expected || ref->getExprLoc().isMacroID()) return false;
    admittedReferences.insert(ref);
    return true;
  }
  bool topDeclaration(const CompoundStmt* body, const VarDecl* variable) const {
    for (const auto* statement : body->body())
      if (const auto* declaration = dyn_cast<DeclStmt>(statement))
        if (declaration->isSingleDecl() && declaration->getSingleDecl() == variable) return true;
    return false;
  }
  bool autoValue(const VarDecl* variable) const {
    return variable->getTypeSourceInfo() &&
           variable->getTypeSourceInfo()->getType()->getContainedAutoType() &&
           !variable->getType()->isReferenceType();
  }
  bool pureForwarder(const FunctionDecl* function) {
    if (!function || !function->hasAttr<CUDADeviceAttr>() ||
        function->hasAttr<CUDAGlobalAttr>() || function->getNumParams() != 2 ||
        !function->hasBody() || !cleanLocation(function)) return false;
    const auto* primary = function->getPrimaryTemplate();
    if (!primary || primary->getTemplateParameters()->size() != 2) return false;
    // The source's ordinary tile may select a specialization that the parsing
    // projection does not select. Check the entire primary, not only this
    // already-selected scratch instantiation.
    for (const auto* specialization : primary->specializations())
      if (specialization->getTemplateSpecializationKind() == TSK_ExplicitSpecialization)
        return false;
    const FunctionDecl* owners[] = {function, primary->getTemplatedDecl()};
    for (const auto* owner : owners)
      for (const auto* declaration : owner->redecls()) {
        if (!cleanLocation(declaration)) return false;
        for (const auto* attribute : declaration->attrs())
          if (!isa<CUDADeviceAttr>(attribute)) return false;
        for (const auto* parameter : declaration->parameters())
          if (parameter->hasAttrs()) return false;
      }
    for (const auto* declaration : primary->redecls())
      if (declaration->hasAttrs()) return false;
    for (const auto* parameter : *primary->getTemplateParameters()) {
      const auto* type = dyn_cast<TemplateTypeParmDecl>(parameter);
      if (!type || type->isParameterPack() || type->hasDefaultArgument() || type->hasAttrs()) return false;
    }
    // A distinct parser tile must not alter source overload selection. Accept
    // only the unconstrained, unique two-type forwarding template.
    std::set<const NamedDecl*> overloads;
    for (const auto* declaration : function->getDeclContext()->lookup(function->getDeclName())) {
      if (const auto* candidate = dyn_cast<FunctionTemplateDecl>(declaration))
        overloads.insert(candidate->getCanonicalDecl());
      else if (const auto* candidate = dyn_cast<FunctionDecl>(declaration))
        overloads.insert(candidate->getCanonicalDecl());
      else return false;
    }
    if (overloads.size() != 1 || !overloads.count(primary->getCanonicalDecl())) return false;
    const auto* pattern = primary->getTemplatedDecl();
    if (pattern->getNumParams() != 2 ||
        !isa<TemplateTypeParmType>(pattern->getReturnType().getTypePtr()) ||
        !isa<TemplateTypeParmType>(pattern->getParamDecl(0)->getType().getTypePtr()) ||
        !pattern->getParamDecl(1)->getType()->isLValueReferenceType()) return false;
    const auto* group = pattern->getParamDecl(1)->getType()->getPointeeType()->getAs<TemplateTypeParmType>();
    if (!group || group->getIndex() != 1 ||
        cast<TemplateTypeParmType>(pattern->getReturnType().getTypePtr())->getIndex() != 0 ||
        cast<TemplateTypeParmType>(pattern->getParamDecl(0)->getType().getTypePtr())->getIndex() != 0)
      return false;
    const auto* body = dyn_cast<CompoundStmt>(function->getBody());
    if (!body || body->size() != 1) return false;
    const auto* ret = dyn_cast<ReturnStmt>(*body->body_begin());
    const auto* call = ret ? dyn_cast_or_null<CallExpr>(plain(ret->getRetValue())) : nullptr;
    if (!call || !owned(call->getDirectCallee(), "reduce") || call->getNumArgs() != 3 ||
        !scalar(function->getReturnType(), context) ||
        role(function->getParamDecl(1)->getType()) != Role::Tile ||
        !use(call->getArg(0), function->getParamDecl(1)) ||
        !use(call->getArg(1), function->getParamDecl(0))) return false;
    const auto* operation = dyn_cast_or_null<CXXConstructExpr>(plain(call->getArg(2)));
    if (!operation || operation->getNumArgs() != 0 ||
        templateName(operation->getType()) != "cooperative_groups::plus") return false;
    admittedVariables.insert(function->getParamDecl(1));
    admittedFunctions.insert(function);
    rawFunctionRanges.insert(pattern);
    admittedCalls.insert(call);
    return true;
  }

  bool proveKernel(const FunctionDecl* kernel) {
    if (!cleanLocation(kernel)) return fail("scratch kernel must be a literal main-file definition");
    const auto* body = dyn_cast<CompoundStmt>(kernel->getBody());
    if (!body) return fail("scratch kernel requires a compound body");
    rawFunctionRanges.insert(kernel);
    if (!kernel->getReturnType()->isVoidType() || kernel->isVariadic())
      return fail("scratch kernel must have a fixed void signature");
    for (const auto* parameter : kernel->parameters()) {
      auto type = parameter->getType();
      if (type->isPointerType()) type = type->getPointeeType();
      if (!(type->isIntegerType() || context.hasSameUnqualifiedType(type, context.FloatTy)))
        return fail("scratch kernel parameters must be scalar int/float or their pointers");
    }
    if (const auto* primary = kernel->getPrimaryTemplate()) {
      const auto* parameters = primary->getTemplateParameters();
      if (parameters->size() != 3) return fail("kernel template requires only T, block size, and tile size");
      const auto* type = dyn_cast<TemplateTypeParmDecl>(parameters->getParam(0));
      if (!type || type->isParameterPack() || type->hasDefaultArgument() || !type->getIdentifier())
        return fail("kernel requires an unconstrained named scalar type parameter");
      for (unsigned index = 1; index != 3; ++index) {
        const auto* value = dyn_cast<NonTypeTemplateParmDecl>(parameters->getParam(index));
        if (!value || value->isParameterPack() || value->hasDefaultArgument() ||
            !value->getType()->isUnsignedIntegerType())
          return fail("kernel size parameters must be unsigned integer values");
      }
      const auto* pattern = primary->getTemplatedDecl();
      const auto* patternBody = dyn_cast<CompoundStmt>(pattern->getBody());
      if (!patternBody || patternBody->getLBracLoc().isMacroID())
        return fail("kernel template guard requires a literal body brace");
      const auto offset = source.getFileOffset(patternBody->getLBracLoc());
      if (guardedBodies.insert(offset).second) {
        const auto insertion = Lexer::getLocForEndOfToken(
            patternBody->getLBracLoc(), 0, source, context.getLangOpts());
        if (insertion.isInvalid())
          return fail("kernel template guard has no token-aligned insertion point");
        stats.guards.push_back({insertion,
            pattern->getSourceRange(),
            "\n  static_assert(::ascify_cg::uniform_scalar_domain<" + type->getNameAsString() +
            ">::value, \"Ascify uniform reduction kernel admits int and float only\");\n"});
      }
    }
    BodyInventory inventory;
    inventory.TraverseStmt(const_cast<CompoundStmt*>(body));
    if (inventory.forbiddenExit) return fail("kernel contains a loop, early exit, asm, lambda, or allocation");
    const VarDecl *storage = nullptr, *block = nullptr, *tile = nullptr;
    for (const auto* variable : inventory.variables) {
      const Role kind = role(variable->getType());
      if (kind == Role::None) {
        if (related(variable->getType())) return fail("scratch pointer or reference alias escapes ownership");
        if (variable->getType()->isRecordType()) return fail("kernel local class lifetime is not proven");
        continue;
      }
      if (!cleanLocation(variable) || !topDeclaration(body, variable))
        return fail("scratch, block, and tile must be unique top-level declarations");
      auto& slot = kind == Role::Storage ? storage : kind == Role::Block ? block : tile;
      if (slot) return fail("kernel must own exactly one scratch, block, and tile");
      slot = variable;
      admittedVariables.insert(variable);
    }
    if (!storage || !block || !tile || !storage->hasAttr<CUDASharedAttr>() ||
        (storage->hasInit() && !isa<CXXConstructExpr>(plain(storage->getInit()))) ||
        !autoValue(block) || !autoValue(tile))
      return fail("requires one shared storage and unobserved auto block/tile");
    for (const auto* variable : {storage, block, tile}) {
      const auto* record = variable->getType()->getAsCXXRecordDecl();
      const auto* specialization = dyn_cast_or_null<ClassTemplateSpecializationDecl>(record);
      if (!record || !trustedLocation(record->getLocation()) ||
          (specialization && specialization->getSpecializationKind() == TSK_ExplicitSpecialization))
        return fail("scratch projection type has an untrusted specialization");
      for (const auto* declaration : record->redecls())
        if (!trustedLocation(declaration->getLocation()))
          return fail("scratch projection type has an untrusted redeclaration");
    }
    for (const auto* constructor : inventory.constructions)
      if (role(constructor->getType()) == Role::None ||
          !trustedFunction(constructor->getConstructor()))
        return fail("kernel has an unproved constructor or temporary lifetime");
    if (const auto* constructor = dyn_cast_or_null<CXXConstructExpr>(plain(storage->getInit())))
      if (constructor->getNumArgs() != 0) return fail("scratch storage must be default constructed");
    unsigned capacity = 0, size = 0, tileCapacity = 0;
    if (!integerArgument(storage->getType(), 0, capacity) || capacity < 32 || capacity > 1024 ||
        capacity % 32 || !integerArgument(tile->getType(), 0, size) ||
        !integerArgument(tile->getType(), 1, tileCapacity) || capacity != tileCapacity ||
        !(size == 32 || size == 64 || size == 128 || size == 256 || size == 512) || capacity % size)
      return fail("unsupported concrete block/tile sizes");
    const auto* blockInit = directInitializer(block);
    const auto* tileInit = directInitializer(tile);
    if (!blockInit || !owned(blockInit->getDirectCallee(), "this_thread_block") ||
        blockInit->getNumArgs() != 1 || !use(blockInit->getArg(0), storage) ||
        !tileInit || !owned(tileInit->getDirectCallee(), "tiled_partition") ||
        tileInit->getNumArgs() != 1 || !use(tileInit->getArg(0), block))
      return fail("block and tile must derive directly from the unique scratch");
    admittedCalls.insert(blockInit);
    admittedCalls.insert(tileInit);
    unsigned collectives = 0;
    for (const auto* statement : body->body()) {
      const auto* assignment = dyn_cast<BinaryOperator>(statement);
      if (!assignment || assignment->getOpcode() != BO_Assign) continue;
      const auto* call = dyn_cast_or_null<CallExpr>(plain(assignment->getRHS()));
      if (!call || call->getNumArgs() != 2 || !reference(call->getArg(1)) ||
          reference(call->getArg(1))->getDecl() != tile) continue;
      const auto* output = reference(assignment->getLHS());
      const auto* input = reference(call->getArg(0));
      const auto* outputVariable = output ? dyn_cast<VarDecl>(output->getDecl()) : nullptr;
      if (!outputVariable || outputVariable->getDeclContext() != kernel ||
          !outputVariable->hasLocalStorage() || !input || output->getDecl() != input->getDecl() ||
          !scalar(output->getType(), context) ||
          !pureForwarder(call->getDirectCallee()) || !use(call->getArg(1), tile))
        return fail("collective requires a pure unique forwarder and local scalar assignment");
      admittedCalls.insert(call);
      ++collectives;
    }
    if (!collectives) return fail("no unconditional top-level reduction was proven");
    for (const auto* call : inventory.calls) {
      if (admittedCalls.count(call)) continue;
      const auto* callee = call->getDirectCallee();
      if (!callee) return fail("indirect calls are not admitted in a scratch kernel");
      if (const auto* member = dyn_cast<CXXMemberCallExpr>(call)) {
        const auto* object = reference(member->getImplicitObjectArgument());
        if (object && (object->getDecl() == block || object->getDecl() == tile)) {
          const auto name = callee->getNameAsString();
          const bool isBlock = object->getDecl() == block;
          if (!trustedFunction(callee) || call->getNumArgs() != 0 ||
              !(name == "thread_rank" || name == "size" ||
                (isBlock && name == "sync") ||
                (!isBlock && (name == "meta_group_rank" || name == "meta_group_size"))))
            return fail("scratch group uses an unproved member operation");
          admittedReferences.insert(object);
          admittedCalls.insert(call);
          continue;
        }
      }
      // Clang represents calls to static members as ordinary CallExpr nodes.
      if (const auto* member = dyn_cast_or_null<MemberExpr>(plain(call->getCallee()))) {
        const auto* object = reference(member->getBase());
        if (object && (object->getDecl() == block || object->getDecl() == tile)) {
          const auto name = callee->getNameAsString();
          const bool isBlock = object->getDecl() == block;
          if (!trustedFunction(callee) || call->getNumArgs() != 0 ||
              !(name == "thread_rank" || name == "size" ||
                (isBlock && name == "sync") ||
                (!isBlock && (name == "meta_group_rank" || name == "meta_group_size"))))
            return fail("scratch group uses an unproved static member");
          admittedReferences.insert(object);
          admittedCalls.insert(call);
          continue;
        }
      }
      if (owned(callee, "sync") && call->getNumArgs() == 1 && use(call->getArg(0), block)) {
        admittedCalls.insert(call);
        continue;
      }
      return fail("kernel calls an unproved helper that may alter participation");
    }
    for (const auto* ref : inventory.references)
      if (related(ref->getType()) && !admittedReferences.count(ref))
        return fail("scratch group escapes, is reassigned, or is used in a divergent collective");
    ++stats.instantiations;
    stats.collectives += collectives;
    kernelLocations.insert(source.getFileOffset(kernel->getLocation()));
    return true;
  }

  bool auditUses() {
    for (const auto* variable : allVariables) {
      if (admittedVariables.count(variable)) continue;
      const auto* function = dyn_cast<FunctionDecl>(variable->getDeclContext());
      if (function && function->isDependentContext()) continue;
      return fail("scratch-related declaration is outside a proven kernel/forwarder");
    }
    for (const auto* ref : allReferences) {
      if (!related(ref->getType()) || admittedReferences.count(ref)) continue;
      // Dependent pattern nodes are not executable. Every concrete template
      // instantiation is traversed independently, including unused explicit ones.
      if (ref->isTypeDependent()) continue;
      const auto* variable = dyn_cast<VarDecl>(ref->getDecl());
      const auto* function = variable ? dyn_cast<FunctionDecl>(variable->getDeclContext()) : nullptr;
      if (function && function->isDependentContext()) continue;
      return fail("an unproved scratch reference remains in the translation unit");
    }
    return true;
  }

  bool auditRawTypes() {
    std::set<unsigned> storageTokens;
    for (const auto location : storageTypeLocations) {
      if (location.isMacroID() || !source.isWrittenInMainFile(location))
        return fail("scratch type spelling must be literal in the main file");
      bool declared = false;
      for (const auto* variable : allVariables) {
        if (role(variable->getType()) != Role::Storage || !variable->getTypeSourceInfo()) continue;
        const auto range = variable->getTypeSourceInfo()->getTypeLoc().getSourceRange();
        if (range.getBegin().isValid() && range.getEnd().isValid() &&
            source.getFileID(source.getSpellingLoc(range.getBegin())) == source.getMainFileID() &&
            source.getFileOffset(location) >= source.getFileOffset(source.getSpellingLoc(range.getBegin())) &&
            source.getFileOffset(location) <= source.getFileOffset(source.getSpellingLoc(range.getEnd())))
          declared = true;
      }
      if (!declared) return fail("scratch storage type is observed outside its unique declaration");
      storageTokens.insert(source.getFileOffset(location));
    }
    bool invalid = false;
    const auto buffer = source.getBufferData(source.getMainFileID(), &invalid);
    if (invalid) return fail("cannot inspect raw scratch source");
    const auto begin = source.getLocForStartOfFile(source.getMainFileID());
    Lexer lexer(begin, context.getLangOpts(), buffer.begin(), buffer.begin(), buffer.end());
    Token token;
    while (!lexer.LexFromRawLexer(token)) {
      if (token.is(tok::hash)) {
        const auto offset = source.getFileOffset(token.getLocation());
        if (!kernels.empty()) {
          const auto lineEnd = buffer.find('\n', offset);
          const auto line = buffer.slice(offset, lineEnd == llvm::StringRef::npos ? buffer.size() : lineEnd);
          std::string compact;
          for (const char character : line)
            if (!std::isspace(static_cast<unsigned char>(character))) compact += character;
          if (compact != "#include<cooperative_groups.h>" &&
              compact != "#include<cooperative_groups/reduce.h>")
            return fail("scratch TU admits only its two authenticated direct includes; conditional or other directives are outside the proof");
        }
        for (const auto* function : rawFunctionRanges) {
          const auto first = source.getSpellingLoc(function->getBeginLoc());
          const auto last = source.getSpellingLoc(function->getEndLoc());
          if (source.getFileID(first) == source.getMainFileID() &&
              offset >= source.getFileOffset(first) && offset <= source.getFileOffset(last))
            return fail("preprocessor directives inside a proven kernel/forwarder are not admitted");
        }
      }
      if (!token.is(tok::raw_identifier) && !token.is(tok::identifier)) continue;
      const auto name = Lexer::getSpelling(token, source, context.getLangOpts());
      if (name == "scratch_thread_block" || name == "scratch_thread_block_tile")
        return fail("source may not name scratch projection types");
      if (!kernels.empty() && name == "ascify_cg")
        return fail("source may not own the target scratch facade namespace");
      if (name == "block_tile_memory" && !storageTokens.count(source.getFileOffset(token.getLocation())))
        return fail("raw or inactive scratch type use lacks an AST proof");
    }
    for (auto file = source.fileinfo_begin(); file != source.fileinfo_end(); ++file) {
      const auto id = source.translateFile(file->first);
      if (id.isInvalid() || id == source.getMainFileID()) continue;
      const auto fileBegin = source.getLocForStartOfFile(id);
      if (trustedLocation(fileBegin)) continue;
      bool unreadable = false;
      const auto data = source.getBufferData(id, &unreadable);
      if (unreadable) return fail("cannot audit a scratch input dependency");
      Lexer headerLexer(fileBegin, context.getLangOpts(), data.begin(), data.begin(), data.end());
      Token headerToken;
      while (!headerLexer.LexFromRawLexer(headerToken)) {
        if (!headerToken.is(tok::raw_identifier) && !headerToken.is(tok::identifier)) continue;
        const auto name = Lexer::getSpelling(headerToken, source, context.getLangOpts());
        if (name == "block_tile_memory" || name == "scratch_thread_block" ||
            name == "scratch_thread_block_tile")
          return fail("scratch types in dependency headers are outside the main-file proof");
      }
    }
    return true;
  }

  ASTContext& context;
  SourceManager& source;
  const FrontendCompatibilityConfig& profile;
  UniformBlockReductionStats& stats;
  std::string& error;
  bool standard;
  bool userCooperativeDeclaration = false;
  bool userUsingDirective = false;
  std::set<const FunctionDecl*> patterns, kernels, admittedFunctions;
  std::set<const VarDecl*> allVariables, admittedVariables;
  std::vector<const DeclRefExpr*> allReferences;
  std::set<const DeclRefExpr*> admittedReferences;
  std::set<const CallExpr*> admittedCalls;
  std::set<unsigned> kernelLocations;
  std::set<unsigned> guardedBodies;
  std::vector<SourceLocation> storageTypeLocations;
  std::set<const FunctionDecl*> rawFunctionRanges;
};
}  // namespace

bool ValidateUniformBlockReduction(
    clang::ASTContext& context, const FrontendCompatibilityConfig& profile,
    UniformBlockReductionStats& stats, std::string& error, bool standardPreprocessing) {
  stats = {};
  error.clear();
  if (!profile.enabled()) return true;
  return Proof(context, profile, stats, error, standardPreprocessing).run();
}
}  // namespace ascify
