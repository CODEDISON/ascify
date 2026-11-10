#include "KernelThreadBounds.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <algorithm>
#include <utility>

namespace ascify {
namespace {
using namespace clang;
SourceLocation physical(SourceLocation loc, SourceManager &source) {
  while (loc.isMacroID()) {
    if (!source.isMacroArgExpansion(loc)) return {};
    loc = source.getImmediateSpellingLoc(loc);
  }
  return loc;
}
bool occupancyName(llvm::StringRef name) {
  return name == "cudaOccupancyMaxActiveBlocksPerMultiprocessor" ||
         name == "cudaOccupancyMaxPotentialBlockSize";
}
struct Tokens {
  ASTContext &context;
  SourceManager &source;
  const KernelBoundOwned &owned;
  const KernelBoundQualifier &qualifier;
  const KernelBoundMacroFree &macroFree;
  // Preserve the old attribute proof's boundary=2 rule. Empty-expanding
  // source tokens before the AST begin are still part of its declaration.
  bool declarationBoundary(SourceLocation begin) const {
    if (!begin.isFileID()) return false;
    bool invalid = false;
    const auto bytes = source.getBufferData(source.getFileID(begin), &invalid);
    if (invalid) return false;
    Lexer lexer(source.getLocForStartOfFile(source.getFileID(begin)), context.getLangOpts(),
                bytes.begin(), bytes.begin(), bytes.end());
    const unsigned first = source.getFileOffset(begin);
    bool directiveLine = false, hasPrevious = false;
    tok::TokenKind previous = tok::unknown;
    Token token;
    while (true) {
      lexer.LexFromRawLexer(token);
      if (token.is(tok::eof)) return false;
      if (source.getFileOffset(token.getLocation()) >= first)
        // A linkage-specification literal ("C"/"C++") directly precedes an
        // extern-linkage kernel signature; it is part of the declaration,
        // not a previous statement.
        return !hasPrevious || previous == tok::semi ||
            previous == tok::r_brace || previous == tok::string_literal;
      if (token.isAtStartOfLine()) directiveLine = false;
      if (token.is(tok::hash) && token.isAtStartOfLine()) directiveLine = true;
      if (!directiveLine) { previous = token.getKind(); hasPrevious = true; }
    }
  }
  bool scan(SourceRange range, bool signature,
            std::vector<Token> *saved = nullptr) const {
    const auto begin = source.getExpansionLoc(range.getBegin());
    const auto last = source.getExpansionLoc(range.getEnd());
    if (!begin.isFileID() || !last.isFileID() || !owned(begin) || !owned(last) ||
        source.getFileID(begin) != source.getFileID(last)) return false;
    if (signature && !declarationBoundary(begin)) return false;
    const auto end = Lexer::getLocForEndOfToken(last, 0, source, context.getLangOpts());
    if (!end.isFileID()) return false;
    auto cursor = begin;
    while (source.getFileOffset(cursor) < source.getFileOffset(end)) {
      Token token;
      if (Lexer::getRawToken(cursor, token, source, context.getLangOpts(), true) ||
          token.needsCleaning() || token.is(tok::hash) || token.is(tok::hashhash)) return false;
      const auto finish = Lexer::getLocForEndOfToken(token.getLocation(), 0, source,
                                                     context.getLangOpts());
      if (!finish.isFileID() || source.getFileOffset(finish) <= source.getFileOffset(cursor) ||
          source.getFileOffset(finish) > source.getFileOffset(end)) return false;
      if (signature && token.isOneOf(tok::l_square, tok::r_square)) return false;
      if (token.is(tok::raw_identifier)) {
        const auto name = token.getRawIdentifier();
        if (signature && (name == "__attribute__" || name == "__attribute" ||
                          name == "__declspec" || name == "asm" || name == "__asm__")) return false;
        if (signature && (name == "__global__" || name == "__launch_bounds__")) {
          if (!qualifier(name, token.getLocation())) return false;
        } else if (!macroFree(name, token.getLocation())) return false;
      }
      if (saved) saved->push_back(token);
      cursor = finish;
    }
    return true;
  }
  bool closedHelperFile(FileID file) const {
    bool invalid = false;
    const auto bytes = source.getBufferData(file, &invalid);
    if (invalid) return false;
    Lexer lexer(source.getLocForStartOfFile(file), context.getLangOpts(),
                bytes.begin(), bytes.begin(), bytes.end());
    Token token;
    while (true) {
      lexer.LexFromRawLexer(token);
      if (token.is(tok::eof)) return true;
      if (token.is(tok::hash) && token.isAtStartOfLine()) {
        lexer.LexFromRawLexer(token);
        if (!token.is(tok::raw_identifier) ||
            (token.getRawIdentifier() != "include" && token.getRawIdentifier() != "line")) return false;
      }
      if (token.is(tok::raw_identifier)) {
        const auto name = token.getRawIdentifier();
        if (name == "asm" || name == "__asm" || name == "__asm__" ||
            name == "__attribute__" || name == "__attribute" || name == "__declspec") return false;
      }
    }
  }
  bool unconditional(FileID file) const {
    bool invalid = false;
    const auto text = source.getBufferData(file, &invalid);
    if (invalid) return false;
    Lexer lexer(source.getLocForStartOfFile(file), context.getLangOpts(),
                text.begin(), text.begin(), text.end());
    Token token;
    while (true) {
      lexer.LexFromRawLexer(token);
      if (token.is(tok::eof)) return true;
      if (!token.is(tok::hash) || !token.isAtStartOfLine()) continue;
      lexer.LexFromRawLexer(token);
      if (!token.is(tok::raw_identifier)) continue;
      const auto name = token.getRawIdentifier();
      if (name == "if" || name == "ifdef" || name == "ifndef" || name == "elif" ||
          name == "elifdef" || name == "elifndef" || name == "else" || name == "endif") return false;
    }
  }
};
struct CallOwner { const CallExpr *call; const FunctionDecl *owner; };
struct Inventory : RecursiveASTVisitor<Inventory> {
  const FunctionDecl *current = nullptr;
  std::vector<CallOwner> calls;
  std::vector<const DeclRefExpr *> references;
  std::vector<const NamedDecl *> usingTargets;
  std::vector<const UnresolvedLookupExpr *> unresolved;
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }
  bool TraverseDecl(Decl *decl) {
    const auto *saved = current;
    if (const auto *fn = llvm::dyn_cast_or_null<FunctionDecl>(decl)) current = fn;
    const bool result = RecursiveASTVisitor<Inventory>::TraverseDecl(decl);
    current = saved;
    return result;
  }
  bool VisitCallExpr(CallExpr *call) { calls.push_back({call, current}); return true; }
  bool VisitDeclRefExpr(DeclRefExpr *ref) { references.push_back(ref); return true; }
  bool VisitUnresolvedLookupExpr(UnresolvedLookupExpr *expr) { unresolved.push_back(expr); return true; }
  bool VisitUsingShadowDecl(UsingShadowDecl *decl) { usingTargets.push_back(decl->getTargetDecl()); return true; }
};
class OccupancyProof {
  ASTContext &context;
  SourceManager &source;
  const KernelBoundOwned &owned;
  const KernelBoundQualifier &qualifier;
  const KernelBoundMacroFree &macroFree;
  const std::function<bool(const CallExpr *)> &trustedApi, &closedInvocation;
  const std::function<bool()> &closedHelperInputContext;
  const std::function<bool(const PureSimtKernelThreadBound &)> &stagePolicy;
  SourceLocation &bad;
  std::string &error;
  KernelThreadBoundPlan staged;
  Inventory inventory;
  Tokens tokens;
  std::set<const Expr *> apiCallees;
  bool fail(SourceLocation loc, const char *message) {
    if (error.empty()) { bad = loc; error = message; }
    return false;
  }
  bool literalCall(const CallExpr *call) {
    const auto *ref = llvm::dyn_cast<DeclRefExpr>(call->getCallee()->IgnoreParenImpCasts());
    if (!ref || ref->hasQualifier() || ref->hasExplicitTemplateArgs() ||
        !closedInvocation(call)) return false;
    const auto first = physical(call->getBeginLoc(), source);
    const auto last = physical(call->getEndLoc(), source);
    return first.isFileID() && last.isFileID() &&
        source.isWrittenInMainFile(first) && tokens.scan({first, last}, false);
  }
  bool addKernel(const FunctionDecl *kernel, unsigned &bound) {
    PureSimtKernelThreadBound fact;
    if (!ProvePureSimtKernelThreadBound(context, kernel, owned, qualifier, macroFree,
                                       fact, bad, error)) return false;
    if (!stagePolicy(fact)) return fail(kernel->getLocation(), "kernel bound policy has no atomic publication owner");
    const auto *canonical = kernel->getCanonicalDecl();
    const auto found = staged.kernels.find(canonical);
    if (found != staged.kernels.end() && found->second.maximum != fact.maximum)
      return fail(kernel->getLocation(), "inconsistent kernel bound");
    staged.kernels[canonical] = fact;
    if (fact.policyQualifier.isValid() && source.isWrittenInMainFile(fact.policyQualifier))
      staged.policyQualifiers[source.getFileOffset(fact.policyQualifier)] = fact.maximum;
    bound = fact.maximum;
    return true;
  }
  const FunctionDecl *directKernel(const Expr *argument) const {
    const Expr *expr = argument->IgnoreParenImpCasts();
    // CUDA's C occupancy API erases its function token to void*. Only that
    // physical C-style cast is admitted; variables and integer casts are not.
    if (const auto *cast = llvm::dyn_cast<CStyleCastExpr>(expr)) {
      const auto *pointer = cast->getType()->getAs<PointerType>();
      if (!pointer || !pointer->getPointeeType()->isVoidType() ||
          !physical(cast->getBeginLoc(), source).isFileID()) return nullptr;
      expr = cast->getSubExpr()->IgnoreParenImpCasts();
    }
    if (const auto *address = llvm::dyn_cast<UnaryOperator>(expr)) {
      if (address->getOpcode() != UO_AddrOf) return nullptr;
      expr = address->getSubExpr()->IgnoreParenImpCasts();
    }
    const auto *ref = llvm::dyn_cast<DeclRefExpr>(expr);
    if (!ref || ref->hasExplicitTemplateArgs()) return nullptr;
    return llvm::dyn_cast<FunctionDecl>(ref->getDecl());
  }
  bool rawNameUses(SourceRange range, llvm::StringRef name,
                   const std::set<unsigned> &allowed) const {
    const auto first = source.getExpansionLoc(range.getBegin());
    const auto last = source.getExpansionLoc(range.getEnd());
    if (!first.isFileID() || !last.isFileID() ||
        source.getFileID(first) != source.getFileID(last)) return false;
    const auto file = source.getFileID(first);
    bool invalid = false;
    const auto bytes = source.getBufferData(file, &invalid);
    if (invalid) return false;
    Lexer lexer(source.getLocForStartOfFile(file), context.getLangOpts(),
                bytes.begin(), bytes.begin(), bytes.end());
    const unsigned begin = source.getFileOffset(first), end = source.getFileOffset(last);
    Token token;
    while (true) {
      lexer.LexFromRawLexer(token);
      if (token.is(tok::eof)) return true;
      const unsigned offset = source.getFileOffset(token.getLocation());
      if (offset > end) return true;
      if (offset >= begin && token.is(tok::raw_identifier) &&
          token.getRawIdentifier() == name && !allowed.count(offset)) return false;
    }
  }
  bool closedHelper(const ParmVarDecl *parameter, unsigned &bound) {
    const auto *fn = llvm::dyn_cast<FunctionDecl>(parameter->getDeclContext());
    if (!closedHelperInputContext() || !fn || !fn->getIdentifier() ||
        !parameter->getIdentifier() || fn->getStorageClass() != SC_Static ||
        !fn->getDeclContext()->isTranslationUnit() || !fn->hasBody() ||
        fn->getPreviousDecl() || fn->getMostRecentDecl() != fn ||
        fn->getTemplatedKind() != FunctionDecl::TK_NonTemplate ||
        llvm::isa<CXXMethodDecl>(fn) || fn->hasAttr<CUDADeviceAttr>() || fn->hasAttr<CUDAGlobalAttr>() ||
        !parameter->getType()->isPointerType() ||
        !parameter->getType()->getPointeeType()->isVoidType() ||
        !source.isWrittenInMainFile(fn->getLocation()) ||
        !tokens.unconditional(source.getFileID(fn->getLocation())) ||
        !tokens.closedHelperFile(source.getFileID(fn->getLocation())) ||
        !tokens.scan({fn->getBeginLoc(), fn->getBody()->getBeginLoc()}, false))
      return fail(parameter->getLocation(), "unproved erased-kernel helper definition");
    for (const auto *attribute : fn->attrs())
      if (!attribute->isImplicit()) return fail(fn->getLocation(), "unproved helper attribute");
    for (const auto *decl : fn->getDeclContext()->lookup(fn->getDeclName())) {
      const auto *other = llvm::dyn_cast<FunctionDecl>(decl);
      if (!other || other->getCanonicalDecl() != fn->getCanonicalDecl())
        return fail(fn->getLocation(), "overloaded erased-kernel helper");
    }
    for (const auto *target : inventory.usingTargets)
      if (target->getCanonicalDecl() == fn->getCanonicalDecl())
        return fail(fn->getLocation(), "erased-kernel helper using alias");
    struct Body : RecursiveASTVisitor<Body> {
      bool VisitLambdaExpr(LambdaExpr *) { return false; }
      bool VisitGCCAsmStmt(GCCAsmStmt *) { return false; }
      bool VisitStmtExpr(StmtExpr *) { return false; }
    } body;
    if (!body.TraverseStmt(const_cast<Stmt *>(fn->getBody())))
      return fail(fn->getLocation(), "unsupported erased-kernel helper body");
    std::set<const Expr *> directCallees;
    std::set<unsigned> helperTokens{source.getFileOffset(fn->getLocation())};
    bool callerSeen = false;
    for (const auto &entry : inventory.calls) {
      const auto *call = entry.call;
      const auto *callee = call->getDirectCallee();
      if (!callee || callee->getCanonicalDecl() != fn->getCanonicalDecl()) continue;
      if (!entry.owner || entry.owner == fn || entry.owner->hasAttr<CUDADeviceAttr>() ||
          entry.owner->hasAttr<CUDAGlobalAttr>() ||
          entry.owner->getTemplatedKind() != FunctionDecl::TK_NonTemplate ||
          !literalCall(call) || parameter->getFunctionScopeIndex() >= call->getNumArgs())
        return fail(call->getExprLoc(), "unproved erased-kernel helper caller");
      const auto *kernel = directKernel(call->getArg(parameter->getFunctionScopeIndex()));
      unsigned candidate = 0;
      if (!kernel || !addKernel(kernel, candidate))
        return fail(call->getExprLoc(), "unknown erased-kernel helper actual");
      if (callerSeen && candidate != bound)
        return fail(call->getExprLoc(), "erased-kernel helper callers have different bounds");
      bound = candidate; callerSeen = true;
      const auto *ref = llvm::dyn_cast<DeclRefExpr>(call->getCallee()->IgnoreParenImpCasts());
      directCallees.insert(ref);
      helperTokens.insert(source.getFileOffset(physical(ref->getLocation(), source)));
    }
    if (!callerSeen) return fail(fn->getLocation(), "erased-kernel helper has no closed callers");
    std::set<unsigned> parameterTokens{source.getFileOffset(parameter->getLocation())};
    for (const auto *ref : inventory.references) {
      if (ref->getDecl()->getCanonicalDecl() == fn->getCanonicalDecl() &&
          !directCallees.count(ref)) return fail(ref->getLocation(), "erased-kernel helper escapes");
      if (ref->getDecl()->getCanonicalDecl() != parameter->getCanonicalDecl()) continue;
      bool admitted = false;
      for (const auto &entry : inventory.calls) {
        const auto *callee = entry.call->getDirectCallee();
        if (!callee || !callee->getIdentifier() || !occupancyName(callee->getName()) || entry.owner != fn ||
            !trustedApi(entry.call) || !literalCall(entry.call)) continue;
        const unsigned index = callee->getName() == "cudaOccupancyMaxPotentialBlockSize" ? 2 : 1;
        if (index < entry.call->getNumArgs() &&
            entry.call->getArg(index)->IgnoreParenImpCasts() == ref) admitted = true;
      }
      if (!admitted) return fail(ref->getLocation(), "erased kernel is observed or escapes");
      const auto loc = physical(ref->getLocation(), source);
      if (!loc.isFileID()) return fail(ref->getLocation(), "macro-generated erased kernel use");
      parameterTokens.insert(source.getFileOffset(loc));
    }
    const auto file = source.getFileID(fn->getLocation());
    const auto data = source.getBufferData(file);
    const auto start = source.getLocForStartOfFile(file);
    if (!rawNameUses({start, start.getLocWithOffset(data.size() - 1)}, fn->getName(), helperTokens) ||
        !rawNameUses(fn->getSourceRange(), parameter->getName(), parameterTokens))
      return fail(fn->getLocation(), "erased helper raw inquiry or macro observation");
    return true;
  }
public:
  OccupancyProof(ASTContext &c, const KernelBoundOwned &o, const KernelBoundQualifier &q,
      const KernelBoundMacroFree &m, const std::function<bool(const CallExpr *)> &a,
      const std::function<bool(const CallExpr *)> &i,
      const std::function<bool()> &h,
      const std::function<bool(const PureSimtKernelThreadBound &)> &p, KernelThreadBoundPlan plan,
      SourceLocation &b, std::string &e)
      : context(c), source(c.getSourceManager()), owned(o), qualifier(q), macroFree(m),
        trustedApi(a), closedInvocation(i), closedHelperInputContext(h), stagePolicy(p),
        bad(b), error(e), staged(std::move(plan)),
        tokens{c, source, o, q, m} {}
  bool prove(bool enabled, KernelThreadBoundPlan &plan) {
    inventory.TraverseDecl(context.getTranslationUnitDecl());
    if (!enabled && staged.used()) return fail({}, "kernel-bound query requires the pure-SIMT target policy");
    for (const auto *lookup : inventory.unresolved)
      if (owned(physical(lookup->getNameLoc(), source)) && occupancyName(lookup->getName().getAsString()))
        return fail(lookup->getNameLoc(), "unresolved dependent occupancy query");
    for (const auto &entry : inventory.calls) {
      const auto *call = entry.call;
      const auto *callee = call->getDirectCallee();
      if (!callee || !callee->getIdentifier() || !occupancyName(callee->getName()))
        continue;
      if (!owned(physical(call->getExprLoc(), source))) continue;
      if (!enabled || !entry.owner || entry.owner->getTemplatedKind() != FunctionDecl::TK_NonTemplate ||
          entry.owner->hasAttr<CUDADeviceAttr>() || entry.owner->hasAttr<CUDAGlobalAttr>() ||
          call->isTypeDependent() || call->isValueDependent() ||
          !trustedApi(call) || !literalCall(call))
        return fail(call->getExprLoc(), "unproved occupancy API or physical call");
      const bool potential = callee->getName() == "cudaOccupancyMaxPotentialBlockSize";
      const unsigned index = potential ? 2 : 1;
      if (call->getNumArgs() <= index) return fail(call->getExprLoc(), "bad occupancy call shape");
      unsigned bound = 0;
      if (const auto *kernel = directKernel(call->getArg(index))) {
        if (!addKernel(kernel, bound)) return false;
      } else {
        const auto *ref = llvm::dyn_cast<DeclRefExpr>(call->getArg(index)->IgnoreParenImpCasts());
        const auto *parameter = ref ? llvm::dyn_cast<ParmVarDecl>(ref->getDecl()) : nullptr;
        if (!parameter || !closedHelper(parameter, bound))
          return fail(call->getExprLoc(), "occupancy kernel has no closed bound");
      }
      const auto *ref = llvm::dyn_cast<DeclRefExpr>(call->getCallee()->IgnoreParenImpCasts());
      const auto loc = physical(ref->getLocation(), source);
      const unsigned offset = source.getFileOffset(loc);
      const std::string name = potential ? "cudaOccupancyMaxPotentialBlockSizeMaxThreads" :
                                          "cudaOccupancyMaxActiveBlocksPerMultiprocessorMaxThreads";
      staged.occupancyCallees[offset] = "::ascify::" + name + "<" + std::to_string(bound) + ">";
      apiCallees.insert(ref);
    }
    for (const auto *ref : inventory.references) {
      const auto *fn = llvm::dyn_cast<FunctionDecl>(ref->getDecl());
      if (fn && fn->getIdentifier() && occupancyName(fn->getName()) && owned(physical(ref->getLocation(), source)) &&
          !apiCallees.count(ref)) return fail(ref->getLocation(), "occupancy API address or inquiry is unproved");
    }
    plan = std::move(staged);
    return true;
  }
};
} // namespace

bool ProvePureSimtKernelThreadBound(ASTContext &context, const FunctionDecl *kernel,
    const KernelBoundOwned &owned, const KernelBoundQualifier &qualifier,
    const KernelBoundMacroFree &macroFree, PureSimtKernelThreadBound &fact,
    SourceLocation &bad, std::string &error) {
  auto &source = context.getSourceManager();
  const auto reject = [&](const char *message) {
    if (error.empty()) { bad = kernel ? kernel->getLocation() : SourceLocation(); error = message; }
    return false;
  };
  const FunctionDecl *definition = nullptr;
  if (!kernel || !kernel->hasAttr<CUDAGlobalAttr>() ||
      !kernel->getDeclContext()->getRedeclContext()->isTranslationUnit() ||
      kernel->getTemplatedKind() != FunctionDecl::TK_NonTemplate ||
      !kernel->hasBody(definition) || kernel != definition || kernel->getPreviousDecl() ||
      kernel->getMostRecentDecl() != kernel || !kernel->getLocation().isFileID() ||
      !owned(kernel->getLocation())) return reject("unproved concrete pure-SIMT kernel definition");
  for (const auto *attribute : kernel->attrs())
    if (!attribute->isImplicit() && !llvm::isa<CUDAGlobalAttr>(attribute) &&
        !llvm::isa<CUDALaunchBoundsAttr>(attribute)) return reject("unproved kernel attribute");
  Tokens tokens{context, source, owned, qualifier, macroFree};
  std::vector<Token> signature;
  if (!tokens.unconditional(source.getFileID(kernel->getLocation())) ||
      !tokens.scan({kernel->getBeginLoc(), kernel->getBody()->getBeginLoc()}, true, &signature))
    return reject("kernel signature macro or conditional provenance is unproved");
  SourceLocation global;
  for (const auto &token : signature)
    if (token.is(tok::raw_identifier) && token.getRawIdentifier() == "__global__") {
      if (global.isValid()) return reject("multiple global qualifiers");
      global = token.getLocation();
    }
  if (global.isInvalid()) return reject("literal SDK global qualifier is required");
  // This is a converter-selected explicit resource policy, not a claim about
  // an implicit compiler default or the meaning of an undocumented ELF tag.
  unsigned maximum = 1024;
  SourceLocation policy = global;
  if (const auto *attribute = kernel->getAttr<CUDALaunchBoundsAttr>()) {
    if (attribute->getMinBlocks() || attribute->getMaxBlocks())
      return reject("multi-argument kernel bound is outside the proof");
    bool invalid = false;
    auto spelling = Lexer::getSourceText(source.getExpansionRange(attribute->getRange()),
                                        source, context.getLangOpts(), &invalid).trim();
    if (invalid || !spelling.consume_front("__launch_bounds__"))
      return reject("unproved launch-bound spelling");
    spelling = spelling.trim();
    if (!spelling.consume_front("(") || !spelling.consume_back(")"))
      return reject("unproved launch-bound delimiter");
    spelling = spelling.trim();
    Expr::EvalResult value;
    if (spelling.getAsInteger(10, maximum) || maximum < 1 || maximum > 2048 ||
        !attribute->getMaxThreads()->EvaluateAsInt(value, context) ||
        value.Val.getInt().isNegative() || value.Val.getInt().getLimitedValue() != maximum)
      return reject("kernel bound is not the same positive literal");
    policy = {};
  } else {
    // The target spelling introduced below must not inherit caller control.
    if (!qualifier("__launch_bounds__", global))
      return reject("policy annotation requires its trusted SDK spelling");
    // A selected physical header fact still needs its stagePolicy owner;
    // source offsets alone never authorize a different FileID's raw edits.
  }
  fact = {kernel->getCanonicalDecl(), maximum, policy};
  return true;
}

bool ProveOccupancyKernelThreadBounds(ASTContext &context, const KernelBoundOwned &owned,
    const KernelBoundQualifier &qualifier, const KernelBoundMacroFree &macroFree,
    const std::function<bool(const CallExpr *)> &trustedApi,
    const std::function<bool(const CallExpr *)> &closedMacroInvocation,
    const std::function<bool()> &closedHelperInputContext,
    const std::function<bool(const PureSimtKernelThreadBound &)> &stagePolicy,
    bool enabled, KernelThreadBoundPlan &plan, SourceLocation &bad, std::string &error) {
  return OccupancyProof(context, owned, qualifier, macroFree, trustedApi,
                        closedMacroInvocation, closedHelperInputContext, stagePolicy,
                        plan, bad, error).prove(enabled, plan);
}
} // namespace ascify
