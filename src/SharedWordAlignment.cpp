#include "SharedWordAlignment.h"

#include <map>
#include <iterator>
#include <set>
#include <vector>
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

namespace ascify {
namespace {
using namespace clang;
const Expr *plain(const Expr *value) {
  return value ? value->IgnoreParenImpCasts() : nullptr;
}
bool device(const FunctionDecl *function) {
  return function && (function->hasAttr<CUDADeviceAttr>() ||
                      function->hasAttr<CUDAGlobalAttr>());
}
class Inventory : public RecursiveASTVisitor<Inventory> {
 public:
  std::vector<const VarDecl *> variables;
  std::vector<const DeclRefExpr *> references;
  std::vector<const CallExpr *> calls;
  std::vector<const BinaryOperator *> assignments;
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }
  bool VisitVarDecl(VarDecl *v) { variables.push_back(v); return true; }
  bool VisitDeclRefExpr(DeclRefExpr *v) { references.push_back(v); return true; }
  bool VisitCallExpr(CallExpr *v) { calls.push_back(v); return true; }
  bool VisitBinaryOperator(BinaryOperator *v) {
    if (v->getOpcode() == BO_Assign) assignments.push_back(v);
    return true;
  }
};
class Proof {
  ASTContext &context;
  SourceManager &source;
  const std::function<bool(SourceLocation)> &owned;
  const std::function<bool(SourceLocation)> &trustedQualifier;
  const std::function<bool(llvm::StringRef, SourceLocation)> &macroFree;
  SourceLocation &bad;
  std::string &error;
  Inventory inventory;
  std::set<const VarDecl *> roots;
  std::set<const FunctionDecl *> helpers;
  SharedWordAlignmentPlan staged;

  bool fail(SourceLocation at, const char *reason) {
    if (error.empty()) { bad = at; error = reason; }
    return false;
  }
  bool literal(SourceLocation at) const { return at.isFileID() && owned(at); }
  const VarDecl *variable(const Expr *value) const {
    const auto *ref = dyn_cast_or_null<DeclRefExpr>(plain(value));
    const auto *v = ref ? dyn_cast<VarDecl>(ref->getDecl()) : nullptr;
    return v ? v->getCanonicalDecl() : nullptr;
  }
  bool pointerType(QualType type) const {
    if (!type->isPointerType()) return false;
    const auto pointee = type->getPointeeType().getCanonicalType();
    return pointee->isSpecificBuiltinType(BuiltinType::Short) ||
        pointee->isSpecificBuiltinType(BuiltinType::UShort) ||
        pointee->isSpecificBuiltinType(BuiltinType::Int) ||
        pointee->isSpecificBuiltinType(BuiltinType::UInt);
  }
  // Provenance is an object root, never data loaded from that object.
  bool located(const Expr *value, unsigned depth = 0) const {
    if (!value || depth > 64 || !value->isGLValue()) return false;
    if (const auto *paren = dyn_cast<ParenExpr>(value)) return located(paren->getSubExpr(), depth+1);
    if (const auto *cast = dyn_cast<CastExpr>(value)) return located(cast->getSubExpr(), depth+1);
    if (const auto *ref = dyn_cast<DeclRefExpr>(value)) {
      const auto *v = dyn_cast<VarDecl>(ref->getDecl());
      // &p is the address of a pointer holder, not the shared object p names.
      return v && v->hasAttr<CUDASharedAttr>() && v->getType()->isArrayType() &&
          roots.count(v->getCanonicalDecl());
    }
    if (const auto *index = dyn_cast<ArraySubscriptExpr>(value)) return rooted(index->getBase(), depth+1);
    if (const auto *op = dyn_cast<UnaryOperator>(value))
      return op->getOpcode() == UO_Deref ? rooted(op->getSubExpr(), depth+1) :
          (op->isIncrementDecrementOp() && located(op->getSubExpr(), depth+1));
    if (const auto *op = dyn_cast<BinaryOperator>(value))
      return op->isAssignmentOp() && located(op->getLHS(), depth+1);
    return false;
  }
  bool rooted(const Expr *value, unsigned depth = 0) const {
    if (!value || depth > 64) return false;
    if (const auto *cast = dyn_cast<CastExpr>(value)) {
      if (!cast->getType()->isPointerType() &&
          cast->getCastKind() != CK_ArrayToPointerDecay) return false;
      return rooted(cast->getSubExpr(), depth + 1);
    }
    if (const auto *paren = dyn_cast<ParenExpr>(value)) return rooted(paren->getSubExpr(), depth + 1);
    if (const auto *ref = dyn_cast<DeclRefExpr>(value)) {
      const auto *v = dyn_cast<VarDecl>(ref->getDecl());
      return v && roots.count(v->getCanonicalDecl());
    }
    if (const auto *op = dyn_cast<BinaryOperator>(value)) {
      if (!op->getType()->isPointerType()) return false;
      if (op->getOpcode() == BO_Assign)
        return rooted(op->getRHS(), depth+1) || rooted(op->getLHS(), depth+1);
      return (op->getOpcode() == BO_Add || op->getOpcode() == BO_Sub ||
              op->getOpcode() == BO_AddAssign || op->getOpcode() == BO_SubAssign) &&
          (rooted(op->getLHS(), depth + 1) || rooted(op->getRHS(), depth + 1));
    }
    if (const auto *op = dyn_cast<UnaryOperator>(value)) {
      if (op->getOpcode() == UO_AddrOf) return located(op->getSubExpr(), depth+1);
      if (op->isIncrementDecrementOp() && op->getType()->isPointerType())
        return rooted(op->getSubExpr(), depth+1);
    }
    return false;
  }
  bool localPointer(const VarDecl *v) const {
    if (!v || !pointerType(v->getType()) || !literal(v->getLocation()) ||
        v->getType()->isDependentType() || v->isInitCapture() || v->hasAttrs() ||
        v->getTLSKind() != VarDecl::TLS_None) return false;
    const auto *owner = dyn_cast<FunctionDecl>(v->getDeclContext());
    return device(owner) && !isa<CXXMethodDecl>(owner) && !owner->hasAttr<CUDAHostAttr>() &&
        owner->getTemplatedKind() == FunctionDecl::TK_NonTemplate &&
        (isa<ParmVarDecl>(v) || (v->isLocalVarDecl() && v->hasLocalStorage() && v->getStorageClass() == SC_None));
  }
  bool closedHelper(const FunctionDecl *function) const {
    if (!function) return false;
    const auto *definition = function->getDefinition();
    if (!definition || isa<CXXMethodDecl>(definition) || !definition->hasAttr<CUDADeviceAttr>() ||
        definition->hasAttr<CUDAGlobalAttr>() || definition->hasAttr<CUDAHostAttr>() || definition->isVariadic() ||
        definition->getTemplatedKind() != FunctionDecl::TK_NonTemplate ||
        !definition->getReturnType()->isVoidType() || !literal(definition->getLocation()) ||
        !literal(definition->getBody()->getBeginLoc()) ||
        !literal(definition->getBody()->getEndLoc())) return false;
    for (const auto *declaration : function->redecls())
      if (!literal(declaration->getLocation())) return false;
    return true;
  }
  bool declarationRange(SourceLocation begin, SourceLocation end) {
    const auto file = source.getFileID(begin);
    bool invalid = false;
    const auto bytes = source.getBufferData(file, &invalid);
    if (invalid) return fail(begin, "shared declaration has no physical buffer");
    const auto start = source.getLocForStartOfFile(file);
    const unsigned offset = source.getFileOffset(begin);
    Lexer lexer(start, context.getLangOpts(), bytes.begin(), bytes.begin()+offset, bytes.end());
    Token token;
    unsigned brackets = 0, parentheses = 0, braces = 0;
    for (;;) {
      lexer.LexFromRawLexer(token);
      if (token.is(tok::eof) || token.needsCleaning())
        return fail(begin, "shared declaration has no literal complete physical range");
      if (token.is(tok::semi) && !brackets && !parentheses && !braces)
        return source.getFileOffset(token.getLocation()) >= source.getFileOffset(end) ? true :
            fail(begin, "shared declaration ended before its array body");
      if (token.is(tok::comma) && !brackets && !parentheses && !braces)
        return fail(token.getLocation(), "shared word alignment rejects multiple physical declarators");
      if (token.is(tok::raw_identifier) && !brackets) {
        const auto name = Lexer::getSpelling(token, source, context.getLangOpts());
        // Extent macros inside [] keep their original evaluated expressions.
        // Other caller macros, including empty layout prefixes, gain no proof.
        if (name != "__shared__" && !macroFree(name, token.getLocation()))
          return fail(token.getLocation(), "shared declaration has an opaque caller macro");
      }
      if (token.is(tok::l_square)) ++brackets;
      else if (token.is(tok::r_square)) { if (!brackets) return fail(begin, "unbalanced shared array range"); --brackets; }
      else if (token.is(tok::l_paren)) ++parentheses;
      else if (token.is(tok::r_paren)) { if (!parentheses) return fail(begin, "unbalanced shared declaration range"); --parentheses; }
      else if (token.is(tok::l_brace)) ++braces;
      else if (token.is(tok::r_brace)) { if (!braces) return fail(begin, "unbalanced shared initializer range"); --braces; }
    }
  }
  bool collect() {
    std::map<std::pair<unsigned, unsigned>, unsigned> groupSizes;
    for (const auto *v : inventory.variables) {
      const auto at = source.getExpansionLoc(v->getBeginLoc());
      if (at.isValid()) {
        const auto site = source.getDecomposedLoc(at);
        ++groupSizes[{site.first.getHashValue(), site.second}];
      }
    }
    for (const auto *v : inventory.variables) {
      const auto *shared = v->getAttr<CUDASharedAttr>();
      if (!shared || !literal(v->getLocation())) continue;
      const auto *array = context.getAsConstantArrayType(v->getType());
      if (!array || array->getSize().isZero()) continue;
      const auto element = array->getElementType().getCanonicalType();
      if ((!element->isSpecificBuiltinType(BuiltinType::Short) &&
           !element->isSpecificBuiltinType(BuiltinType::UShort)) || context.getTypeSize(element) != 16) continue;
      // Existing explicit alignments, including smaller/dependent ones, remain
      // on their original path. This rule never rewrites or weakens them.
      if (v->hasAttr<AlignedAttr>()) continue;
      if (!v->getTypeSourceInfo()) continue;
      auto written = v->getTypeSourceInfo()->getTypeLoc().getUnqualifiedLoc();
      while (const auto arrayLoc = written.getAs<ArrayTypeLoc>())
        written = arrayLoc.getElementLoc().getUnqualifiedLoc();
      // Constant extent macros remain original expressions. A macro-spelled
      // element declaration does not gain this new object-layout rule.
      if (!literal(written.getBeginLoc()) || !literal(written.getEndLoc())) continue;
      const auto qualifier = source.getExpansionLoc(shared->getLocation());
      Token token;
      if (!literal(qualifier) || Lexer::getRawToken(qualifier, token, source, context.getLangOpts(), true) ||
          token.getLocation() != qualifier || token.needsCleaning() ||
          Lexer::getSpelling(token, source, context.getLangOpts()) != "__shared__") continue;
      if (!trustedQualifier(qualifier))
        return fail(qualifier, "shared word alignment requires the actual SDK CUDA qualifier macro");
      for (const auto *attribute : v->attrs())
        if (!isa<CUDASharedAttr>(attribute))
          return fail(v->getLocation(), "shared word alignment rejects opaque object attributes");
      if (v->getStorageClass() == SC_Extern || v->isStaticDataMember() ||
          v->isTemplated() ||
          v->getType()->isDependentType())
        return fail(v->getLocation(), "shared word alignment requires a non-template static array definition");
      const auto *owner = dyn_cast<FunctionDecl>(v->getDeclContext());
      if ((owner && (!device(owner) || isa<CXXMethodDecl>(owner) || owner->hasAttr<CUDAHostAttr>() ||
                      owner->getTemplatedKind() != FunctionDecl::TK_NonTemplate)) ||
          (!owner && v->getStorageClass() != SC_Static))
        return fail(v->getLocation(), "shared word alignment requires local or internal source ownership");
      const auto declarationBegin = source.getExpansionLoc(v->getBeginLoc());
      const auto end = v->getEndLoc();
      if (!literal(declarationBegin) || !literal(end) || source.getFileID(declarationBegin) != source.getFileID(end) ||
          source.getFileID(declarationBegin) != source.getFileID(qualifier))
        return fail(v->getLocation(), "shared word alignment declaration has no single physical range");
      const auto begin = source.getFileOffset(qualifier) < source.getFileOffset(declarationBegin)
          ? qualifier : declarationBegin;
      if (!declarationRange(begin, end)) return false;
      const auto site = source.getDecomposedLoc(declarationBegin);
      if (groupSizes[{site.first.getHashValue(), site.second}] != 1 ||
          std::distance(v->redecls_begin(), v->redecls_end()) != 1)
        return fail(v->getLocation(), "shared word alignment rejects multiple declarators or redeclarations");
      if (!macroFree("ASCIFY_ALIGN", qualifier))
        return fail(qualifier, "shared word alignment macro is owned by the source");
      roots.insert(v->getCanonicalDecl());
      staged.qualifiers.insert(qualifier.getRawEncoding());
      ++staged.arrays;
      if (!source.isWrittenInMainFile(qualifier))
        staged.contextualHeaders.insert(source.getFilename(qualifier).str());
    }
    return true;
  }
  bool close() {
    for (unsigned round = 0; round < 64; ++round) {
      const auto before = roots.size();
      for (const auto *v : inventory.variables)
        if (v->hasInit() && rooted(v->getInit())) {
          if (isa<ParmVarDecl>(v) || !localPointer(v))
            return fail(v->getLocation(), "shared pointer initializer escapes the closed local object");
          roots.insert(v->getCanonicalDecl());
        }
      for (const auto *assignment : inventory.assignments)
        if (rooted(assignment->getRHS())) {
          const auto *v = variable(assignment->getLHS());
          if (!localPointer(v)) return fail(assignment->getExprLoc(), "shared pointer assignment escapes the closed local object");
          roots.insert(v);
        }
      for (const auto *call : inventory.calls)
        for (unsigned i = 0; i < call->getNumArgs(); ++i)
          if (rooted(call->getArg(i))) {
            const auto *callee = call->getDirectCallee();
            if (!closedHelper(callee) || i >= callee->getNumParams())
              return fail(call->getExprLoc(), "shared pointer call has no closed direct device helper");
            const auto *definition = callee->getDefinition();
            if (!localPointer(definition->getParamDecl(i)))
              return fail(call->getExprLoc(), "shared pointer helper parameter is outside builtin16/32 pointer transport");
            roots.insert(definition->getParamDecl(i)->getCanonicalDecl());
            // References in earlier declarations need no inference by name.
            for (const auto *declaration : callee->redecls())
              roots.insert(declaration->getParamDecl(i)->getCanonicalDecl());
            helpers.insert(definition->getCanonicalDecl());
            if (!source.isWrittenInMainFile(definition->getLocation()))
              staged.contextualHeaders.insert(source.getFilename(definition->getLocation()).str());
          }
      if (roots.size() > 512 || helpers.size() > 64)
        return fail(SourceLocation(), "shared pointer closure exceeds its bounded source domain");
      if (roots.size() == before) return true;
    }
    return fail(SourceLocation(), "shared pointer closure did not converge");
  }
  bool use(const DeclRefExpr *reference) {
    const Stmt *at = reference;
    for (unsigned depth = 0; depth < 128; ++depth) {
      const auto parents = context.getParents(*at);
      if (parents.size() != 1) return fail(reference->getExprLoc(), "shared object use has no unique closed parent");
      if (const auto *v = parents[0].get<VarDecl>())
        return !isa<ParmVarDecl>(v) && roots.count(v->getCanonicalDecl()) && localPointer(v) ? true :
            fail(reference->getExprLoc(), "shared pointer escapes in a declaration");
      const auto *parent = parents[0].get<Stmt>();
      if (!parent) return fail(reference->getExprLoc(), "shared object type or layout observation is not closed");
      if (const auto *inquiry = dyn_cast<UnaryExprOrTypeTraitExpr>(parent)) {
        if (inquiry->getKind() == UETT_SizeOf) return true;
        return fail(inquiry->getExprLoc(), "shared object alignment reflection cannot change");
      }
      if (const auto *cast = dyn_cast<CastExpr>(parent)) {
        if (cast->getCastKind() == CK_LValueToRValue && !cast->getType()->isPointerType()) return true;
        if (cast->getCastKind() != CK_ArrayToPointerDecay &&
            (!pointerType(cast->getType()) || !cast->getSubExpr()->getType()->isPointerType()))
          return fail(cast->getExprLoc(), "shared object numeric address or unsupported pointer cast");
      } else if (const auto *index = dyn_cast<ArraySubscriptExpr>(parent)) {
        if (index->getIdx() == at) return fail(index->getExprLoc(), "shared address used as an index");
      } else if (const auto *op = dyn_cast<UnaryOperator>(parent)) {
        if (op->getOpcode() != UO_Deref && op->getOpcode() != UO_AddrOf &&
            !op->isIncrementDecrementOp())
          return fail(op->getExprLoc(), "shared address unary observation is not closed");
        if (op->getOpcode() == UO_AddrOf && !located(op->getSubExpr()))
          return fail(op->getExprLoc(), "shared pointer holder address escapes its object root");
        if (op->isIncrementDecrementOp() && op->getType()->isPointerType()) {
          const auto *destination = variable(op->getSubExpr());
          if (!localPointer(destination) || !roots.count(destination))
            return fail(op->getExprLoc(), "shared pointer increment has no closed local holder");
        }
      } else if (const auto *op = dyn_cast<BinaryOperator>(parent)) {
        if (op->isAssignmentOp()) {
          if (op->getLHS()->getType()->isPointerType()) {
            const auto *destination = variable(op->getLHS());
            if (!localPointer(destination) || !roots.count(destination))
              return fail(op->getExprLoc(), "shared pointer store escapes the closed local object");
            if (op->getOpcode() != BO_Assign && op->getOpcode() != BO_AddAssign && op->getOpcode() != BO_SubAssign)
              return fail(op->getExprLoc(), "unsupported shared pointer assignment");
          }
          // Assignment can produce a pointer or an addressable scalar lvalue.
          // Follow its parent: leaked=(p=a) and &(a[0]=x) are not discarded stores.
          at = parent;
          continue;
        }
        if ((op->getOpcode() != BO_Add && op->getOpcode() != BO_Sub) ||
            !op->getType()->isPointerType())
          return fail(op->getExprLoc(), "shared numeric address or cross-object layout observation");
      } else if (const auto *call = dyn_cast<CallExpr>(parent)) {
        const auto *callee = call->getDirectCallee();
        const auto *definition = callee ? callee->getDefinition() : nullptr;
        if (!closedHelper(callee) || !definition || !helpers.count(definition->getCanonicalDecl()))
          return fail(call->getExprLoc(), "shared pointer call has no audited concrete helper");
        for (unsigned i = 0; i < call->getNumArgs(); ++i)
          if (call->getArg(i) == at) {
            if (i < definition->getNumParams() && rooted(call->getArg(i)) &&
                localPointer(definition->getParamDecl(i)) &&
                roots.count(definition->getParamDecl(i)->getCanonicalDecl())) return true;
            return fail(call->getExprLoc(), "shared call argument was not transported to its audited formal");
          }
        return fail(call->getExprLoc(), "shared call use is outside an audited concrete argument");
      } else if (const auto *loop = dyn_cast<ForStmt>(parent)) {
        const auto *store = dyn_cast_or_null<BinaryOperator>(plain(dyn_cast<Expr>(at)));
        // An unbraced loop body discards this complete element assignment.
        // Keep walking through the assignment first: &(a[0]=x) and pointer
        // escapes must still be rejected before reaching a statement boundary.
        // Init, condition and increment clauses gain no new admitted use.
        if (loop->getBody() == at && store && store->isAssignmentOp() &&
            !store->getLHS()->getType()->isPointerType() && located(store->getLHS()))
          return true;
        return fail(reference->getExprLoc(), "shared use is not a discarded for-body element assignment");
      } else if (isa<CompoundStmt>(parent)) {
        for (const auto &owner : context.getParents(*parent))
          if (owner.get<StmtExpr>())
            return fail(reference->getExprLoc(), "GNU statement expression may export a shared address");
        // Only a discarded statement ends the value-use walk.
        return true;
      } else if (!isa<ParenExpr>(parent)) {
        return fail(reference->getExprLoc(), "shared pointer escape or unsupported object observation");
      }
      at = parent;
    }
    return fail(reference->getExprLoc(), "shared object use exceeds bounded expression depth");
  }
 public:
  Proof(ASTContext &c, const std::function<bool(SourceLocation)> &o,
        const std::function<bool(SourceLocation)> &q,
        const std::function<bool(llvm::StringRef, SourceLocation)> &m,
        SourceLocation &b, std::string &e) : context(c), source(c.getSourceManager()),
        owned(o), trustedQualifier(q), macroFree(m), bad(b), error(e) {}
  bool run(SharedWordAlignmentPlan &out) {
    inventory.TraverseDecl(context.getTranslationUnitDecl());
    if (!collect()) return false;
    if (!staged.used()) return true;
    if (!close()) return false;
    for (const auto *reference : inventory.references) {
      const auto *v = dyn_cast<VarDecl>(reference->getDecl());
      if (!v || !roots.count(v->getCanonicalDecl())) continue;
      if (!literal(reference->getExprLoc()) || !use(reference))
        return fail(reference->getExprLoc(), "shared object reference has no physical closed use");
    }
    staged.helpers = helpers.size();
    out = std::move(staged);
    return true;
  }
};
} // namespace
bool ProveSharedWordAlignment(ASTContext &context,
    const std::function<bool(SourceLocation)> &owned,
    const std::function<bool(SourceLocation)> &trustedQualifier,
    const std::function<bool(llvm::StringRef, SourceLocation)> &macroFree,
    bool enabled, SharedWordAlignmentPlan &output, SourceLocation &bad, std::string &error) {
  output = SharedWordAlignmentPlan();
  if (!enabled) return true;
  return Proof(context, owned, trustedQualifier, macroFree, bad, error).run(output);
}
} // namespace ascify
