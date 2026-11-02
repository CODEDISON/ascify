#include "FullWarpGroups.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>
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
// Select the available primary API without guessing a release boundary.
template <class Context, class Record>
static auto groupRecordType(Context& context, const Record* record, int)
    -> decltype(context.getRecordType(record), clang::QualType()) {
  return context.getRecordType(record);
}
template <class Context, class Record>
static auto groupRecordType(Context& context, const Record* record, long)
    -> decltype(context.getCanonicalTagType(record), clang::QualType()) {
  return context.getCanonicalTagType(record);
}
template <class Member>
static auto groupMemberClass(const Member* member, int)
    -> decltype(member->getClass(), clang::QualType()) {
  return clang::QualType(member->getClass(), 0);
}
template <class Member>
static auto groupMemberClass(const Member* member, long)
    -> decltype(member->getQualifier().getAsType(), clang::QualType()) {
  const auto qualifier = member->getQualifier();
  using Qualifier = decltype(member->getQualifier());
  if (qualifier.getKind() != Qualifier::Kind::Type)
    return {};
  return clang::QualType(qualifier.getAsType(), 0);
}
enum class Role { None, Block, Tile16, Erased, Internal16 };
enum class Origin { Block, Tile16 };
const FunctionDecl* canonical(const FunctionDecl* f) {
  return f ? f->getCanonicalDecl() : nullptr;
}
const Expr* plain(const Expr* e) {
  if (!e) return nullptr;
  e = e->IgnoreParenImpCasts();
  if (const auto* v = dyn_cast<ExprWithCleanups>(e)) return plain(v->getSubExpr());
  if (const auto* v = dyn_cast<MaterializeTemporaryExpr>(e)) return plain(v->getSubExpr());
  if (const auto* v = dyn_cast<CXXBindTemporaryExpr>(e)) return plain(v->getSubExpr());
  return e;
}
Role role(QualType t) {
  if (t.isNull() || t->isDependentType()) return Role::None;
  const auto* r = t.getUnqualifiedType()->getAsCXXRecordDecl();
  if (!r) return Role::None;
  const auto name = r->getQualifiedNameAsString();
  if (name == "cooperative_groups::thread_block") return Role::Block;
  if (name == "cooperative_groups::thread_group") return Role::Erased;
  const auto* s = dyn_cast<ClassTemplateSpecializationDecl>(r);
  if (!s || s->getTemplateArgs().size() == 0 ||
      s->getTemplateArgs()[0].getKind() != TemplateArgument::Integral ||
      s->getTemplateArgs()[0].getAsIntegral() != 16) return Role::None;
  const auto family = s->getSpecializedTemplate()->getQualifiedNameAsString();
  if (family == "cooperative_groups::thread_block_tile") return Role::Tile16;
  if (family == "cooperative_groups::ascify_detail::tile_register_operations")
    return Role::Internal16;
  return Role::None;
}
bool newRole(Role r) { return r == Role::Tile16 || r == Role::Erased || r == Role::Internal16; }
bool carriesNew(QualType t, unsigned depth = 0) {
  if (t.isNull() || depth > 12) return false;
  if (newRole(role(t))) return true;
  if (t->isPointerType() || t->isReferenceType()) return carriesNew(t->getPointeeType(), depth + 1);
  if (const auto* a = dyn_cast<ArrayType>(t.getTypePtr())) return carriesNew(a->getElementType(), depth + 1);
  if (const auto* m = t->getAs<MemberPointerType>())
    return carriesNew(groupMemberClass(m, 0), depth + 1) || carriesNew(m->getPointeeType(), depth + 1);
  if (const auto* s = dyn_cast_or_null<ClassTemplateSpecializationDecl>(t->getAsCXXRecordDecl()))
    for (const auto& a : s->getTemplateArgs().asArray())
      if (a.getKind() == TemplateArgument::Type && carriesNew(a.getAsType(), depth + 1)) return true;
  return false;
}

struct CallSite { const FunctionDecl* owner; const CallExpr* call; bool observation; };
class Inventory : public RecursiveASTVisitor<Inventory> {
  const FunctionDecl* current = nullptr;
  unsigned observed = 0;
 public:
  std::vector<const FunctionDecl*> functions;
  std::vector<const ValueDecl*> values;
  std::vector<const DeclRefExpr*> references;
  std::vector<CallSite> calls;
  std::set<const Expr*> directCallees;
  struct TypeUse { QualType type; SourceLocation location; const FunctionDecl* owner; };
  std::vector<TypeUse> typeUses;
  std::vector<const TypedefNameDecl*> aliases;
  std::vector<const NamespaceDecl*> namespaces;
  std::vector<const NamespaceAliasDecl*> namespaceAliases;
  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }
  bool TraverseDecl(Decl* d) {
    const auto* save = current;
    if (const auto* f = dyn_cast_or_null<FunctionDecl>(d)) { current = f; functions.push_back(f); }
    const bool ok = RecursiveASTVisitor<Inventory>::TraverseDecl(d);
    current = save;
    return ok;
  }
  bool TraverseTypeLoc(TypeLoc t, bool qualifier = true) {
    ++observed;
    const bool ok = RecursiveASTVisitor<Inventory>::TraverseTypeLoc(t, qualifier);
    --observed;
    return ok;
  }
  bool TraverseStmt(Stmt* s) {
    const bool inquiry = s && (isa<UnaryExprOrTypeTraitExpr>(s) || isa<CXXNoexceptExpr>(s) ||
        isa<CXXTypeidExpr>(s) || isa<TypeTraitExpr>(s));
    observed += inquiry;
    const bool ok = RecursiveASTVisitor<Inventory>::TraverseStmt(s);
    observed -= inquiry;
    return ok;
  }
  bool VisitValueDecl(ValueDecl* d) { values.push_back(d); return true; }
  bool VisitDeclRefExpr(DeclRefExpr* e) { references.push_back(e); return true; }
  bool VisitCallExpr(CallExpr* c) {
    calls.push_back({current, c, observed != 0});
    directCallees.insert(plain(c->getCallee()));
    return true;
  }
  bool VisitTypeLoc(TypeLoc t) { typeUses.push_back({t.getType(), t.getBeginLoc(), current}); return true; }
  bool VisitTypedefNameDecl(TypedefNameDecl* d) { aliases.push_back(d); return true; }
  bool VisitNamespaceDecl(NamespaceDecl* d) { namespaces.push_back(d); return true; }
  bool VisitNamespaceAliasDecl(NamespaceAliasDecl* d) { namespaceAliases.push_back(d); return true; }
};

struct Value { bool uniform = false; std::optional<int64_t> constant; };
struct Frame {
  const FunctionDecl* function;
  unsigned threads;
  std::map<const ValueDecl*, Origin> groups;
  std::map<const ValueDecl*, Value> scalars;
};
class Proof {
  ASTContext& context;
  const std::function<bool(SourceLocation)>& owned;
  const std::function<bool(const Decl*)>& cg;
  const std::function<bool(const Decl*)>& sdk;
  const std::function<bool(const FunctionDecl*)>& printfOwner;
  const std::function<bool(llvm::StringRef,SourceLocation)>& macroFree;
  const std::function<bool(FileID)>& nativeInput;
  FullWarpGroupStats& result;
  SourceLocation& bad;
  std::string& error;
  Inventory inventory;
  std::set<const FunctionDecl*> affected, reached, stack;
  std::set<const CallExpr*> approvedCalls;
  std::set<const Expr*> approvedCallees;
  std::set<const CallExpr*> syncSites;
  unsigned invocations = 0;
  mutable std::map<unsigned,bool> sourceDirectiveCache;
  bool fail(SourceLocation l, const std::string& s) {
    if (error.empty()) { bad = l; error = s; }
    return false;
  }
  bool literal(SourceLocation l) const { return l.isFileID() && owned(l); }
  bool directiveFreeSource(SourceLocation location) const {
    if (!literal(location)) return false;
    auto& source=context.getSourceManager();
    const auto file=source.getFileID(location);
    const unsigned key=file.getHashValue();
    const auto prior=sourceDirectiveCache.find(key);
    if (prior!=sourceDirectiveCache.end()) return prior->second;
    bool invalid=false;
    const auto bytes=source.getBufferData(file,&invalid);
    if (invalid) return false;
    Lexer lexer(source.getLocForStartOfFile(file),context.getLangOpts(),
        bytes.begin(),bytes.begin(),bytes.end());
    lexer.SetCommentRetentionState(false);
    Token token; lexer.LexFromRawLexer(token);
    const auto clean=[](const Token& t) { return !t.needsCleaning() && !t.is(tok::hashhash); };
    while (!token.is(tok::eof)) {
      if (!clean(token)) return sourceDirectiveCache[key]=false;
      if (token.is(tok::raw_identifier) && (token.getRawIdentifier()=="_Pragma" ||
          token.getRawIdentifier()=="__pragma" || token.getRawIdentifier()=="__has_include"))
        return sourceDirectiveCache[key]=false;
      if (token.is(tok::raw_identifier)) {
        const auto name=token.getRawIdentifier();
        // Only CUDA qualifier spellings borrow the existing sticky SDK
        // attribute-control proof. All other source tokens, including empty
        // expansions omitted from the AST/launch range, must be macro-free.
        const bool qualifier=name=="__global__" || name=="__device__" || name=="__shared__" ||
            name=="__host__" || name=="__forceinline__" || name=="__align__";
        if (!qualifier && !macroFree(name,token.getLocation())) return sourceDirectiveCache[key]=false;
      }
      if (!token.is(tok::hash)) { lexer.LexFromRawLexer(token); continue; }
      if (!token.isAtStartOfLine()) return sourceDirectiveCache[key]=false;
      lexer.LexFromRawLexer(token);
      if (!clean(token) || token.isAtStartOfLine() || !token.is(tok::raw_identifier) ||
          token.getRawIdentifier()!="include") return sourceDirectiveCache[key]=false;
      lexer.LexFromRawLexer(token);
      if (!clean(token) || token.isAtStartOfLine()) return sourceDirectiveCache[key]=false;
      if (token.is(tok::less)) {
        do {
          lexer.LexFromRawLexer(token);
          if (!clean(token) || token.isAtStartOfLine() || token.is(tok::eof) || token.is(tok::hash))
            return sourceDirectiveCache[key]=false;
        } while (!token.is(tok::greater));
      } else if (token.is(tok::string_literal)) {
        const auto text=Lexer::getSpelling(token,source,context.getLangOpts());
        if (text.size()<2 || text.front()!='"' || text.back()!='"' || text.find('\\')!=std::string::npos)
          return sourceDirectiveCache[key]=false;
      } else return sourceDirectiveCache[key]=false;
      lexer.LexFromRawLexer(token);
      if (!token.is(tok::eof) && !token.isAtStartOfLine()) return sourceDirectiveCache[key]=false;
    }
    return sourceDirectiveCache[key]=true;
  }
  bool trustedRecord(QualType t) const {
    const auto* r = t.isNull() ? nullptr : t->getAsCXXRecordDecl();
    if (!r || !r->getDefinition()) return false;
    for (const auto* d : r->redecls()) if (!cg(d)) return false;
    return true;
  }
  bool writtenType(const VarDecl* v) const {
    const auto* info=v ? v->getTypeSourceInfo() : nullptr;
    return info && literal(info->getTypeLoc().getBeginLoc()) && literal(info->getTypeLoc().getEndLoc());
  }
  bool builtinLocal(const VarDecl* v) const {
    return v && v->hasLocalStorage() && !v->isStaticLocal() && v->getTLSKind()==VarDecl::TLS_None &&
        !v->hasAttrs() && !v->getType()->isReferenceType() && v->getType()->isBuiltinType() &&
        v->getType()->isArithmeticType() && !v->getType().isVolatileQualified() &&
        !v->getType()->isAtomicType() && writtenType(v) &&
        bool(v->getTypeSourceInfo()->getTypeLoc().getUnqualifiedLoc().getAs<BuiltinTypeLoc>());
  }
  bool trustedFunction(const FunctionDecl* f) const {
    if (!f) return false;
    for (const auto* d : f->redecls()) if (d->isNoReturn() || !cg(d)) return false;
    return true;
  }
  bool boundedDiagnostic(const CallExpr* call) const {
    const auto* f=call->getDirectCallee();
    if (!f || f->getQualifiedNameAsString()!="printf" || f->getBuiltinID()==0 ||
        !f->getDeclContext()->getRedeclContext()->isTranslationUnit() || !printfOwner(f) ||
        !f->isExternC() || !f->hasExternalFormalLinkage() || !f->isVariadic() || f->getNumParams()!=1 ||
        !context.hasSameUnqualifiedType(f->getReturnType(),context.IntTy) ||
        call->getNumArgs()==0 || call->getNumArgs()>17) return false;
    for (const auto* d : f->redecls()) {
      if (d->isNoReturn() || d->hasBody() || !d->isExternC() || !d->hasExternalFormalLinkage()) return false;
      for (const auto* a : d->attrs())
        if (!isa<BuiltinAttr>(a) && !isa<FormatAttr>(a) &&
            !isa<CUDADeviceAttr>(a) && !isa<CUDAHostAttr>(a)) return false;
    }
    const QualType parameter=f->getParamDecl(0)->getType();
    if (!parameter->isPointerType() || !parameter->getPointeeType().isConstQualified() ||
        parameter->getPointeeType().isVolatileQualified() ||
        !context.hasSameUnqualifiedType(parameter->getPointeeType(),context.CharTy) ||
        f->getParamDecl(0)->hasDefaultArg()) return false;
    // The return count cannot control participation or escape as a value.
    // Only actual discarded call statements are admitted, including an if
    // body without braces; no nested/comma/argument/result-use exception.
    const auto parents=context.getParents(*call);
    if (parents.size()!=1) return false;
    bool discarded=parents[0].get<CompoundStmt>()!=nullptr;
    if (const auto* branch=parents[0].get<IfStmt>())
      discarded=branch->getThen()==call || branch->getElse()==call;
    if (!discarded) return false;
    const auto* format=dyn_cast_or_null<StringLiteral>(plain(call->getArg(0)));
    if (!format || !literal(format->getBeginLoc()) || format->getCharByteWidth()!=1 ||
        format->getString().size()>4096 || format->getString().contains('\0')) return false;
    unsigned arguments=0;
    const auto text=format->getString();
    for (size_t i=0;i<text.size();++i) if (text[i]=='%') {
      if (++i==text.size()) return false;
      if (text[i]=='%') continue;
      if (text[i]!='d') return false;
      ++arguments;
    }
    if (arguments+1!=call->getNumArgs()) return false;
    for (unsigned i=1;i<call->getNumArgs();++i)
      if (!context.hasSameUnqualifiedType(call->getArg(i)->getType(),context.IntTy)) return false;
    return true;
  }
  bool sourceFunction(const FunctionDecl* f, bool kernel) {
    if (!f || !f->getIdentifier() || !literal(f->getLocation()) || !f->getBody() ||
        !literal(f->getBody()->getBeginLoc()) || !literal(f->getBody()->getEndLoc()) ||
        !f->getDeclContext()->isTranslationUnit() || isa<CXXMethodDecl>(f) || f->isVariadic() ||
        f->getPrimaryTemplate() || f->getDescribedFunctionTemplate() || f->isConstexpr() ||
        f->isConsteval() || f->isNoReturn() || f->hasAttr<CUDAHostAttr>() || !directiveFreeSource(f->getLocation()) ||
        (kernel ? !f->hasAttr<CUDAGlobalAttr>() : !f->hasAttr<CUDADeviceAttr>()))
      return fail(f ? f->getLocation() : SourceLocation(), "group path needs an owned ordinary non-template device definition");
    if (role(f->getReturnType()) != Role::None || carriesNew(f->getReturnType()))
      return fail(f->getLocation(), "group return/representation escape is not admitted");
    if (!f->getReturnType()->isBuiltinType() ||
        (!f->getReturnType()->isVoidType() && !f->getReturnType()->isArithmeticType()) ||
        (kernel && !f->getReturnType()->isVoidType()))
      return fail(f->getLocation(), "group helper return must be an ordinary builtin scalar/void");
    const auto* prototype=f->getType()->getAs<FunctionProtoType>();
    if (!prototype || prototype->getCallConv()!=CC_C || prototype->getExceptionSpecType()!=EST_None ||
        f->getReturnType()->isReferenceType())
      return fail(f->getLocation(), "group helper ABI/exception/return is not admitted");
    for (const auto* a : f->attrs())
      if (!isa<CUDADeviceAttr>(a) && !isa<CUDAGlobalAttr>(a) && !isa<AlwaysInlineAttr>(a))
        return fail(f->getLocation(), "group helper attribute is outside the closed interface");
    for (const auto* d : f->redecls()) if (!literal(d->getLocation()))
      return fail(d->getLocation(), "group helper declaration is outside the physical source closure");
    for (const auto* d : f->getDeclContext()->lookup(f->getDeclName())) {
      const auto* other = dyn_cast<FunctionDecl>(d);
      if (!other || canonical(other) != canonical(f))
        return fail(f->getLocation(), "group helper overload/ADL observation is not admitted");
    }
    if (!kernel) for (const auto* other : inventory.functions) {
      if (!other->getIdentifier() || other->getIdentifier()!=f->getIdentifier()) continue;
      for (const auto* scope=other->getDeclContext();scope;scope=scope->getParent()) {
        const auto* space=dyn_cast<NamespaceDecl>(scope);
        if (space && space->getQualifiedNameAsString()=="cooperative_groups")
          return fail(f->getLocation(), "source helper name collides with the admitted CG ADL surface");
      }
    }
    for (const auto* p : f->parameters()) {
      if (p->hasDefaultArg() || p->getType()->isReferenceType() || p->getType().isVolatileQualified() ||
          (kernel && role(p->getType()) != Role::None) ||
          (role(p->getType()) == Role::None && carriesNew(p->getType())))
        return fail(p->getLocation(), "unproved group helper parameter/default/escape");
      if (role(p->getType())!=Role::None && !writtenType(p))
        return fail(p->getLocation(), "group parameter type is macro-derived or not literal source");
      if (role(p->getType())==Role::None) {
        QualType scalar=p->getType();
        if (scalar->isPointerType()) scalar=scalar->getPointeeType();
        if (!scalar->isBuiltinType() || (!scalar->isArithmeticType() && !scalar->isVoidType()))
          return fail(p->getLocation(), "user record/enum/vector/helper carrier can alter group ADL/overload selection");
      }
    }
    return true;
  }
  void approve(const CallExpr* c) {
    approvedCalls.insert(c); approvedCallees.insert(plain(c->getCallee()));
  }
  bool fits(Role r, Origin o) const {
    return r == Role::Erased || (r == Role::Block && o == Origin::Block) ||
        (r == Role::Tile16 && o == Origin::Tile16);
  }
  std::optional<Origin> origin(const Expr* input, Frame& f) {
    const Expr* e = plain(input);
    if (!e || !literal(e->getExprLoc())) return {};
    if (const auto* v = dyn_cast<DeclRefExpr>(e)) {
      const auto it = f.groups.find(v->getDecl());
      if (it != f.groups.end()) return it->second;
    }
    if (const auto* c = dyn_cast<CXXConstructExpr>(e)) {
      if (!trustedFunction(c->getConstructor()) || !trustedRecord(c->getType()) || c->getNumArgs() != 1)
        return {};
      if (const auto* temporary=dyn_cast<CXXTemporaryObjectExpr>(c)) {
        const auto type=temporary->getTypeSourceInfo()->getTypeLoc();
        if (!literal(type.getBeginLoc()) || !literal(type.getEndLoc())) return {};
      }
      const auto value = origin(c->getArg(0), f);
      if (value && fits(role(c->getType()), *value)) return value;
    }
    if (const auto* c = dyn_cast<CallExpr>(e)) {
      const auto* target = c->getDirectCallee();
      if (!literal(c->getExprLoc()) || !literal(plain(c->getCallee())->getExprLoc()) ||
          !trustedFunction(target) || !trustedRecord(c->getType())) return {};
      const auto name = target->getQualifiedNameAsString();
      if (name == "cooperative_groups::this_thread_block" && c->getNumArgs() == 0 && role(c->getType()) == Role::Block) {
        approve(c); return Origin::Block;
      }
      const auto* args = target->getTemplateSpecializationArgs();
      if (name == "cooperative_groups::tiled_partition" && c->getNumArgs() == 1 && args && args->size() == 1 &&
          args->get(0).getKind() == TemplateArgument::Integral && args->get(0).getAsIntegral() == 16 &&
          role(c->getType()) == Role::Tile16) {
        const auto parent = origin(c->getArg(0), f);
        if (parent && *parent == Origin::Block) { approve(c); return Origin::Tile16; }
      }
    }
    return {};
  }
  bool readOnlyWitness(const VarDecl* v) const {
    if (!v || !v->isLocalVarDecl() || !builtinLocal(v) || !directiveFreeSource(v->getLocation()) ||
        (!context.hasSameUnqualifiedType(v->getType(),context.IntTy) &&
         !context.hasSameUnqualifiedType(v->getType(),context.UnsignedIntTy)) ||
        v->getType().isVolatileQualified() || !v->hasInit()) return false;
    for (const auto* r : inventory.references) if (r->getDecl() == v) {
      const auto parents = context.getParents(*r);
      if (parents.size() != 1) return false;
      const auto* c = parents[0].get<ImplicitCastExpr>();
      if (!c || c->getCastKind() != CK_LValueToRValue) return false;
    }
    return true;
  }
  std::optional<uint64_t> launchInteger(const Expr* input, std::set<const VarDecl*>& visiting) const {
    const Expr* e = input;
    if (!e || !directiveFreeSource(e->getExprLoc())) return {};
    if (const auto* p=dyn_cast<ParenExpr>(e)) return launchInteger(p->getSubExpr(),visiting);
    if (const auto* p=dyn_cast<ExprWithCleanups>(e)) return launchInteger(p->getSubExpr(),visiting);
    if (const auto* p=dyn_cast<ImplicitCastExpr>(e)) {
      auto value=launchInteger(p->getSubExpr(),visiting);
      if (!value || !p->getType()->isIntegerType()) return {};
      const unsigned bits=context.getIntWidth(p->getType());
      if (!bits || bits>64 || p->getType()->isBooleanType() ||
          (bits<64 && *value >= (1ULL << (bits-(p->getType()->isSignedIntegerType()?1:0))))) return {};
      return value;
    }
    if (const auto* n = dyn_cast<IntegerLiteral>(e)) {
      if (n->getValue().getActiveBits() <= 32) return n->getValue().getZExtValue();
      return {};
    }
    if (const auto* r = dyn_cast<DeclRefExpr>(e)) {
      const auto* v = dyn_cast<VarDecl>(r->getDecl());
      if (readOnlyWitness(v) && visiting.insert(v).second) {
        auto value = launchInteger(v->getInit(), visiting); visiting.erase(v); return value;
      }
    }
    return {};
  }
  std::optional<unsigned> blockThreads(const CUDAKernelCallExpr* launch) const {
    const auto* config = launch->getConfig();
    if (!config || config->getNumArgs() < 2) return {};
    const auto* d = dyn_cast_or_null<CXXConstructExpr>(plain(config->getArg(1)));
    if (!d || d->getNumArgs() != 3 || !sdk(d->getConstructor()) ||
        d->getConstructor()->getParent()->getQualifiedNameAsString() != "dim3") return {};
    for (const auto* declaration : d->getConstructor()->redecls()) if (!sdk(declaration)) return {};
    std::set<const VarDecl*> visiting;
    const auto yz = [&](unsigned index) -> std::optional<uint64_t> {
      const auto* argument=dyn_cast_or_null<CXXDefaultArgExpr>(plain(d->getArg(index)));
      if (!argument) return launchInteger(d->getArg(index),visiting);
      // Only the actual authenticated dim3 constructor's y/z default1 may
      // come from SDK source. User x/defaults/macros never borrow this rule.
      bool parameter=false;
      for (const auto* declaration : d->getConstructor()->redecls())
        parameter |= declaration->getNumParams()>index && declaration->getParamDecl(index)==argument->getParam();
      const auto* one=dyn_cast_or_null<IntegerLiteral>(plain(argument->getExpr()));
      if (!parameter || !sdk(argument->getParam()) || !one || one->getValue()!=1 ||
          !one->getExprLoc().isFileID()) return {};
      auto& source=context.getSourceManager();
      if (source.getFileID(one->getExprLoc()) !=
          source.getFileID(source.getSpellingLoc(argument->getParam()->getLocation()))) return {};
      return 1;
    };
    auto x = launchInteger(d->getArg(0), visiting), y = yz(1), z = yz(2);
    if (!x || !y || !z || *y != 1 || *z != 1 || *x < 32 || *x > 1024 || *x % 32 != 0) return {};
    return static_cast<unsigned>(*x);
  }
  Value binary(BinaryOperatorKind op, Value a, Value b) const {
    Value out{a.uniform && b.uniform, {}};
    if (!out.uniform || !a.constant || !b.constant) return out;
    const int64_t x = *a.constant, y = *b.constant;
    if (x < -(1LL<<30) || x > (1LL<<30) || y < -(1LL<<30) || y > (1LL<<30)) return out;
    switch (op) {
      case BO_Add: out.constant=x+y; break;
      case BO_Sub: out.constant=x-y; break;
      case BO_Mul: out.constant=x*y; break;
      case BO_Div: if (y) out.constant=x/y; break;
      case BO_Rem: if (y) out.constant=x%y; break;
      case BO_LT: out.constant=x<y; break; case BO_LE: out.constant=x<=y; break;
      case BO_GT: out.constant=x>y; break; case BO_GE: out.constant=x>=y; break;
      case BO_EQ: out.constant=x==y; break; case BO_NE: out.constant=x!=y; break;
      case BO_LAnd: out.constant=x&&y; break; case BO_LOr: out.constant=x||y; break;
      default: break;
    }
    return out;
  }
  bool expression(const Expr*, Frame&, bool uniformControl, Value&);
  bool statement(const Stmt*, Frame&, bool uniformControl, bool& exits);
  bool function(const FunctionDecl*, unsigned, const std::map<unsigned, Origin>&);
 public:
  Proof(ASTContext& c, const std::function<bool(SourceLocation)>& o,
      const std::function<bool(const Decl*)>& g, const std::function<bool(const Decl*)>& s,
      const std::function<bool(const FunctionDecl*)>& p,
      const std::function<bool(llvm::StringRef,SourceLocation)>& m,
      const std::function<bool(FileID)>& n,
      FullWarpGroupStats& r, SourceLocation& b, std::string& e)
      : context(c), owned(o), cg(g), sdk(s), printfOwner(p), macroFree(m), nativeInput(n), result(r), bad(b), error(e) {}
  bool run(bool enabled);
};

bool Proof::expression(const Expr* input, Frame& f, bool control, Value& out) {
  out = {};
  if (!input) return true;
  if (const auto* e = dyn_cast<ParenExpr>(input)) return expression(e->getSubExpr(), f, control, out);
  if (const auto* e = dyn_cast<ExprWithCleanups>(input)) return expression(e->getSubExpr(), f, control, out);
  if (isa<LambdaExpr>(input) || isa<StmtExpr>(input) || isa<UnaryExprOrTypeTraitExpr>(input) ||
      isa<CXXNoexceptExpr>(input) || isa<CXXTypeidExpr>(input) || isa<TypeTraitExpr>(input))
    return fail(input->getExprLoc(), "group path inquiry/capture/statement expression is not admitted");
  if (const auto* c = dyn_cast<CastExpr>(input)) {
    if (role(c->getType()) != Role::None || carriesNew(c->getType()))
      return fail(c->getExprLoc(), "group cast or representation escape is not admitted");
    if (!expression(c->getSubExpr(), f, control, out)) return false;
    // Keep a finite-loop constant only when the actual destination type can
    // represent it. Never ignore a narrowing/signedness conversion.
    if (out.constant) {
      if (!c->getType()->isIntegerType()) out.constant.reset();
      else if (c->getType()->isBooleanType()) out.constant = bool(*out.constant);
      else {
        const unsigned bits = context.getIntWidth(c->getType());
        if (bits == 0 || bits > 64 || (c->getType()->isUnsignedIntegerType() && *out.constant < 0) ||
            (bits < 64 && (*out.constant < -(1LL << (bits - 1)) ||
                uint64_t(*out.constant < 0 ? 0 : *out.constant) >=
                    (1ULL << (bits - (c->getType()->isSignedIntegerType() ? 1 : 0))))))
          out.constant.reset();
      }
    }
    return true;
  }
  if (const auto* n = dyn_cast<IntegerLiteral>(input)) {
    if (!literal(n->getExprLoc())) return fail(n->getExprLoc(), "group control literal is macro-derived");
    out.uniform = true;
    if (n->getValue().getActiveBits() < 63) out.constant = n->getValue().getZExtValue();
    return true;
  }
  if (const auto* n = dyn_cast<CXXBoolLiteralExpr>(input)) {
    out = {true, n->getValue()}; return true;
  }
  if (const auto* r = dyn_cast<DeclRefExpr>(input)) {
    if (role(r->getType()) != Role::None || carriesNew(r->getType()))
      return fail(r->getExprLoc(), "group value is outside an authenticated origin/method/by-value argument");
    if (!builtinLocal(dyn_cast<VarDecl>(r->getDecl()))) return true;
    const auto i = f.scalars.find(r->getDecl());
    if (i != f.scalars.end()) out = i->second;
    return true;
  }
  if (const auto* call = dyn_cast<CallExpr>(input)) {
    const auto* target = call->getDirectCallee();
    if (!target || !target->getIdentifier() || !literal(call->getExprLoc()) ||
        !literal(plain(call->getCallee())->getExprLoc()))
      return fail(call->getExprLoc(), "group path indirect or macro-derived call is not admitted");
    for (const auto* d : target->redecls()) if (d->isNoReturn())
      return fail(call->getExprLoc(), "termination/noreturn call can remove collective participants");
    const auto* method = dyn_cast<CXXMethodDecl>(target);
    const Expr* object = nullptr;
    if (const auto* c = dyn_cast<CXXMemberCallExpr>(call)) object = c->getImplicitObjectArgument();
    if (const auto* m = dyn_cast<MemberExpr>(plain(call->getCallee()))) {
      if (!literal(m->getMemberLoc())) return fail(m->getMemberLoc(), "macro-derived group/member call is not admitted");
      if (!object) object=m->getBase();
    }
    const bool groupMethod = method && role(groupRecordType(context, method->getParent(), 0)) != Role::None;
    const bool freeSync = target->getQualifiedNameAsString() == "cooperative_groups::sync";
    if (groupMethod || freeSync) {
      if (!trustedFunction(target) || (groupMethod ? call->getNumArgs() != 0 : call->getNumArgs() != 1))
        return fail(call->getExprLoc(), "unproved group method/provider/signature");
      const auto group = origin(groupMethod ? object : call->getArg(0), f);
      if (!group) return fail(call->getExprLoc(), "group method has no closed block/static16 origin");
      const auto name = target->getName();
      if (name == "sync") {
        if (!control || !target->getReturnType()->isVoidType())
          return fail(call->getExprLoc(), "collective requires block-uniform control at one common epoch");
        syncSites.insert(call);
      } else if (groupMethod && (name == "size" || name == "thread_rank")) {
        const auto ownerRole = role(groupRecordType(context, method->getParent(), 0));
        const QualType expected = ownerRole == Role::Erased ? context.UnsignedLongLongTy : context.UnsignedIntTy;
        if (!context.hasSameUnqualifiedType(target->getReturnType(), expected))
          return fail(call->getExprLoc(), "group result type differs from authenticated CUDA interface");
        out.uniform = name == "size";
        if (out.uniform) out.constant = *group == Origin::Block ? f.threads : 16;
      } else return fail(call->getExprLoc(), "only group size/thread_rank/sync are admitted");
      approve(call); return true;
    }
    bool groupCall = affected.count(canonical(target)) != 0;
    for (const auto* a : call->arguments()) groupCall |= role(a->getType()) != Role::None || carriesNew(a->getType());
    if (groupCall) {
      if (!control || call->getNumArgs() != target->getNumParams())
        return fail(call->getExprLoc(), "group helper invocation is not block-uniform/exact");
      std::map<unsigned, Origin> bindings;
      for (unsigned i = 0; i < call->getNumArgs(); ++i) {
        const auto parameter = target->getParamDecl(i)->getType();
        if (role(parameter) != Role::None) {
          const auto value = origin(call->getArg(i), f);
          if (!value || !fits(role(parameter), *value) || parameter->isReferenceType())
            return fail(call->getArg(i)->getExprLoc(), "group actual/formal origin does not close by value");
          bindings.emplace(i, *value);
        } else {
          Value ignored;
          if (!expression(call->getArg(i), f, control, ignored)) return false;
        }
      }
      if (!function(target->getDefinition(), f.threads, bindings)) return false;
      approve(call); return true; // A source helper's scalar result is varying.
    }
    // Default deny external calls, even in uniform control. SDK ownership or
    // a familiar spelling never proves noncollectivity/normal continuation.
    // This excludes all raw block/warp votes, shuffles, match/reduce/sync and
    // trap/unreachable calls; adding one later needs its own semantic proof.
    // The single bounded diagnostic has exact declaration/provider, format,
    // argument and discarded-result checks, and its arguments are still
    // recursively audited for hidden collectives and control mutations.
    if (method || !boundedDiagnostic(call))
      return fail(call->getExprLoc(), "unproved external/SDK collective or terminating call on group path");
    for (const auto* a : call->arguments()) { Value ignored; if (!expression(a, f, control, ignored)) return false; }
    approve(call); return true;
  }
  if (const auto* b = dyn_cast<BinaryOperator>(input)) {
    Value a, v;
    if (!expression(b->getLHS(), f, control, a)) return false;
    const bool shortCircuit = b->getOpcode() == BO_LAnd || b->getOpcode() == BO_LOr;
    if (!expression(b->getRHS(), f, control && (!shortCircuit || a.uniform), v)) return false;
    if (shortCircuit) for (auto& item : f.scalars) item.second.constant.reset();
    if (b->isAssignmentOp()) {
      const auto* ref = dyn_cast_or_null<DeclRefExpr>(plain(b->getLHS()));
      if (ref) {
        if (role(ref->getType()) != Role::None || carriesNew(ref->getType()))
          return fail(b->getExprLoc(), "group reassignment is outside immutable origin flow");
        if (b->isCompoundAssignmentOp()) v = binary(BinaryOperator::getOpForCompoundAssignment(b->getOpcode()), a, v);
        v.uniform &= control && builtinLocal(dyn_cast<VarDecl>(ref->getDecl()));
        if (!v.uniform) v.constant.reset();
        if (v.constant && ref->getType()->isIntegerType()) {
          const unsigned bits=context.getIntWidth(ref->getType());
          if (!bits || bits>64 || ref->getType()->isBooleanType() ||
              (ref->getType()->isUnsignedIntegerType() && *v.constant<0) ||
              (bits<64 && (*v.constant < -(1LL<<(bits-1)) ||
                  uint64_t(*v.constant<0?0:*v.constant) >=
                    (1ULL<<(bits-(ref->getType()->isSignedIntegerType()?1:0))))))
            v.constant.reset();
        }
        f.scalars[ref->getDecl()] = v;
      }
      out = v;
    } else out = binary(b->getOpcode(), a, v);
    return true;
  }
  if (const auto* u = dyn_cast<UnaryOperator>(input)) {
    const auto* ref = dyn_cast_or_null<DeclRefExpr>(plain(u->getSubExpr()));
    if (u->getOpcode() == UO_AddrOf && ref) {
      const auto* var = dyn_cast<VarDecl>(ref->getDecl());
      if (role(ref->getType()) != Role::None || carriesNew(ref->getType()) ||
          (var && var->hasLocalStorage() && var->getType()->isScalarType()))
        return fail(u->getExprLoc(), "address escape of group/local control is not admitted");
    }
    if (!expression(u->getSubExpr(), f, control, out)) return false;
    if (u->isIncrementDecrementOp() && ref) {
      out = binary(u->isIncrementOp() ? BO_Add : BO_Sub, out, {true, 1});
      out.uniform &= control && builtinLocal(dyn_cast<VarDecl>(ref->getDecl()));
      if (!out.uniform) out.constant.reset();
      f.scalars[ref->getDecl()] = out;
    } else if (out.constant) {
      if (u->getOpcode() == UO_Minus && *out.constant > -(1LL<<62)) out.constant = -*out.constant;
      else if (u->getOpcode() == UO_LNot) out.constant = !*out.constant;
      else if (u->getOpcode() != UO_Plus) out.constant.reset();
    }
    if (u->getOpcode() == UO_Deref || u->getOpcode() == UO_AddrOf) out = {};
    return true;
  }
  if (const auto* c = dyn_cast<ConditionalOperator>(input)) {
    Value condition, a, b;
    if (!expression(c->getCond(), f, control, condition)) return false;
    Frame yes=f, no=f;
    if (!expression(c->getTrueExpr(), yes, control && condition.uniform, a) ||
        !expression(c->getFalseExpr(), no, control && condition.uniform, b)) return false;
    for (auto& item : f.scalars) {
      const Value x=yes.scalars[item.first], y=no.scalars[item.first];
      item.second.uniform=x.uniform && y.uniform;
      item.second.constant=item.second.uniform && x.constant==y.constant ? x.constant : std::nullopt;
    }
    out.uniform = condition.uniform && a.uniform && b.uniform;
    if (out.uniform && a.constant == b.constant) out.constant = a.constant;
    return true;
  }
  if (const auto* m = dyn_cast<MemberExpr>(input)) {
    if (const auto* method = dyn_cast<CXXMethodDecl>(m->getMemberDecl()))
      if (role(groupRecordType(context, method->getParent(), 0)) != Role::None)
        return fail(m->getExprLoc(), "group method address/inquiry is not admitted");
  }
  if (role(input->getType()) != Role::None || carriesNew(input->getType()) || isa<CXXConstructExpr>(input))
    return fail(input->getExprLoc(), "unproved group/object construction or observation");
  if (!isa<ArraySubscriptExpr>(input) && !isa<MemberExpr>(input) && !isa<StringLiteral>(input) &&
      !isa<FloatingLiteral>(input) && !isa<CharacterLiteral>(input) && !isa<CXXNullPtrLiteralExpr>(input) &&
      !isa<GNUNullExpr>(input) && !isa<ImplicitValueInitExpr>(input) && !isa<InitListExpr>(input))
    return fail(input->getExprLoc(), "expression is outside the closed group control grammar");
  for (const auto* child : input->children()) {
    const auto* e = dyn_cast_or_null<Expr>(child);
    if (!e) return fail(input->getExprLoc(), "unproved statement in group expression");
    Value ignored; if (!expression(e, f, control, ignored)) return false;
  }
  return true; // Loads/unknown scalar operations never gain uniformity.
}

// Loop counters are read-only in the body. In particular a ref argument or
// address/capture may not silently invalidate the finite iteration witness.
class CounterReads : public RecursiveASTVisitor<CounterReads> {
  ASTContext& context;
  const VarDecl* counter;
 public:
  bool ok = true;
  CounterReads(ASTContext& c, const VarDecl* v) : context(c), counter(v) {}
  bool VisitDeclRefExpr(DeclRefExpr* r) {
    if (r->getDecl() != counter) return true;
    const auto p=context.getParents(*r);
    const auto* cast=p.size()==1 ? p[0].get<ImplicitCastExpr>() : nullptr;
    if (!cast || cast->getCastKind()!=CK_LValueToRValue) ok=false;
    return true;
  }
};
class Writes : public RecursiveASTVisitor<Writes> {
 public:
  std::set<const ValueDecl*> values;
  bool VisitBinaryOperator(BinaryOperator* b) {
    if (b->isAssignmentOp()) if (const auto* r=dyn_cast_or_null<DeclRefExpr>(plain(b->getLHS())))
      values.insert(r->getDecl());
    return true;
  }
  bool VisitUnaryOperator(UnaryOperator* u) {
    if (u->isIncrementDecrementOp()) if (const auto* r=dyn_cast_or_null<DeclRefExpr>(plain(u->getSubExpr())))
      values.insert(r->getDecl());
    return true;
  }
};

bool Proof::statement(const Stmt* s, Frame& f, bool control, bool& exits) {
  exits=false;
  if (!s || isa<NullStmt>(s)) return true;
  if (const auto* block=dyn_cast<CompoundStmt>(s)) {
    bool path=control;
    for (const auto* child : block->body()) {
      bool early=false;
      if (!statement(child, f, path, early)) return false;
      // Even a source-uniform early exit conservatively removes the witness
      // for later collectives. Tail scalar returns remain permitted.
      if (early) { path=false; exits=true; }
    }
    return true;
  }
  if (const auto* decl=dyn_cast<DeclStmt>(s)) {
    for (const auto* d : decl->decls()) {
      const auto* v=dyn_cast<VarDecl>(d);
      if (!v || !literal(v->getLocation())) return fail(d->getLocation(), "unproved declaration in group body");
      const Role r=role(v->getType());
      if (r!=Role::None) {
        if (r==Role::Internal16 || !control || !v->hasLocalStorage() || v->isStaticLocal() ||
            v->getTLSKind()!=VarDecl::TLS_None || v->hasAttrs() || v->getType().isVolatileQualified() ||
            !trustedRecord(v->getType()) || !writtenType(v) || !v->hasInit())
          return fail(v->getLocation(), "group local needs an immutable ordinary uniform origin");
        const auto o=origin(v->getInit(), f);
        if (!o || !fits(r,*o)) return fail(v->getLocation(), "group local initialization has no exact origin");
        f.groups.emplace(v,*o);
      } else {
        if (carriesNew(v->getType())) return fail(v->getLocation(), "group pointer/ref/array/aggregate carrier is not admitted");
        if (v->getType()->isReferenceType())
          return fail(v->getLocation(), "local reference can alias and invalidate group control state");
        Value value;
        if (v->hasInit() && !expression(v->getInit(),f,control,value)) return false;
        value.uniform &= control && builtinLocal(v);
        if (!value.uniform) value.constant.reset();
        f.scalars.emplace(v,value);
      }
    }
    return true;
  }
  if (const auto* branch=dyn_cast<IfStmt>(s)) {
    if (branch->getInit() || branch->getConditionVariable() || branch->isConstexpr())
      return fail(branch->getIfLoc(), "group control uses an unproved if initializer/constexpr branch");
    Value condition;
    if (!expression(branch->getCond(),f,control,condition)) return false;
    Frame yes=f,no=f; bool a=false,b=false;
    if (!statement(branch->getThen(),yes,control&&condition.uniform,a) ||
        !statement(branch->getElse(),no,control&&condition.uniform,b)) return false;
    for (auto& item : f.scalars) {
      const Value x=yes.scalars[item.first],y=no.scalars[item.first];
      item.second.uniform=x.uniform&&y.uniform;
      item.second.constant=item.second.uniform&&x.constant==y.constant ? x.constant : std::nullopt;
    }
    exits=a||b;
    return true;
  }
  if (const auto* loop=dyn_cast<ForStmt>(s)) {
    if (!control || loop->getConditionVariable())
      return fail(loop->getForLoc(), "collective loop must be block-uniform");
    const auto* init=dyn_cast_or_null<DeclStmt>(loop->getInit());
    const auto* counter=init&&init->isSingleDecl() ? dyn_cast<VarDecl>(init->getSingleDecl()) : nullptr;
    if (!counter || !builtinLocal(counter) || (!context.hasSameUnqualifiedType(counter->getType(),context.IntTy) &&
                    !context.hasSameUnqualifiedType(counter->getType(),context.UnsignedIntTy)) ||
        counter->getType().isVolatileQualified())
      return fail(loop->getForLoc(), "collective loop requires one bounded integer induction variable");
    bool ignoredExit=false;
    if (!statement(init,f,control,ignoredExit)) return false;
    const Value initial=f.scalars[counter];
    const auto* condition=dyn_cast_or_null<BinaryOperator>(plain(loop->getCond()));
    const auto* lhs=condition ? dyn_cast_or_null<DeclRefExpr>(plain(condition->getLHS())) : nullptr;
    Value limit;
    const auto* literalLimit=condition ? dyn_cast_or_null<IntegerLiteral>(plain(condition->getRHS())) : nullptr;
    if (!condition || !lhs || lhs->getDecl()!=counter || !initial.uniform || !initial.constant ||
        !literalLimit || !literal(literalLimit->getExprLoc()) ||
        !expression(condition->getRHS(),f,control,limit) || !limit.uniform || !limit.constant)
      return fail(loop->getForLoc(), "collective loop RHS must be an immutable literal finite bound");
    const Expr* step=plain(loop->getInc());
    bool finite=false;
    if (const auto* divide=dyn_cast_or_null<CompoundAssignOperator>(step)) {
      const auto* dst=dyn_cast_or_null<DeclRefExpr>(plain(divide->getLHS()));
      const auto* factor=dyn_cast_or_null<IntegerLiteral>(plain(divide->getRHS()));
      finite=condition->getOpcode()==BO_GT && *limit.constant==0 &&
          *initial.constant>0 && *initial.constant<=1024 && dst && dst->getDecl()==counter &&
          divide->getOpcode()==BO_DivAssign && factor && factor->getValue()==2;
    }
    if (const auto* increment=dyn_cast_or_null<UnaryOperator>(step)) {
      const auto* dst=dyn_cast_or_null<DeclRefExpr>(plain(increment->getSubExpr()));
      finite=condition->getOpcode()==BO_LT && *initial.constant>=0 &&
          *initial.constant<=*limit.constant && *limit.constant<=1024 &&
          dst && dst->getDecl()==counter && increment->isIncrementOp();
    }
    if (!finite) return fail(loop->getForLoc(), "collective loop is outside bounded ++ or positive /=2 grammar");
    CounterReads reads(context,counter); reads.TraverseStmt(const_cast<Stmt*>(loop->getBody()));
    if (!reads.ok) return fail(loop->getForLoc(), "loop induction witness is mutated/escaped in its body");
    Writes writes; writes.TraverseStmt(const_cast<Stmt*>(loop->getBody()));
    // A single abstract body visit cannot prove an epoch whose previously
    // uniform local becomes varying (even after this iteration's last sync).
    // Freeze every loop-entry uniform local except the separately proved
    // induction counter. References/addresses/opaque calls are denied above.
    for (const auto* value : writes.values) {
      const auto entry=f.scalars.find(value);
      if (value!=counter && entry!=f.scalars.end() && entry->second.uniform)
        return fail(loop->getForLoc(), "loop body writes a loop-entry uniform witness across collective epochs");
    }
    bool early=false;
    if (!statement(loop->getBody(),f,control,early) || early)
      return fail(loop->getForLoc(), "collective loop body can leave its common epoch");
    Value ignored;
    if (!expression(loop->getInc(),f,control,ignored)) return false;
    // One abstract body visit proves uniformity, not final values after N
    // iterations. Never let stale constants authorize a later loop/shape.
    writes.values.insert(counter);
    for (const auto* v : writes.values) f.scalars[v].constant.reset();
    return true;
  }
  if (const auto* ret=dyn_cast<ReturnStmt>(s)) {
    Value ignored;
    if (!expression(ret->getRetValue(),f,control,ignored)) return false;
    exits=true; return true;
  }
  if (const auto* e=dyn_cast<Expr>(s)) { Value ignored; return expression(e,f,control,ignored); }
  return fail(s->getBeginLoc(), "switch/while/do/goto/break/continue/exception/asm is outside group control proof");
}

bool Proof::function(const FunctionDecl* f, unsigned threads, const std::map<unsigned,Origin>& bindings) {
  if (++invocations>512) return fail(f?f->getLocation():SourceLocation(), "group call closure exceeds finite proof budget");
  if (!sourceFunction(f,f&&f->hasAttr<CUDAGlobalAttr>())) return false;
  const auto* id=canonical(f);
  if (!stack.insert(id).second) return fail(f->getLocation(), "recursive group helper flow is not admitted");
  Frame frame{f,threads,{},{}};
  for (unsigned i=0;i<f->getNumParams();++i) {
    const auto* p=f->getParamDecl(i);
    if (role(p->getType())!=Role::None) {
      auto b=bindings.find(i);
      if (b==bindings.end() || !fits(role(p->getType()),b->second) || !trustedRecord(p->getType()))
        return fail(p->getLocation(), "group formal has no closed actual origin");
      frame.groups.emplace(p,b->second);
    } else frame.scalars.emplace(p,Value{});
  }
  bool exits=false;
  if (!statement(f->getBody(),frame,true,exits)) return false;
  stack.erase(id); reached.insert(id); return true;
}

bool Proof::run(bool enabled) {
  result={};
  inventory.TraverseDecl(context.getTranslationUnitDecl());
  bool used=false;
  for (const auto& t : inventory.typeUses) if (literal(t.location) && carriesNew(t.type)) {
    used=true;
    if (!t.owner || role(t.type)==Role::None || role(t.type)==Role::Internal16 ||
        isa<DecltypeType>(t.type.getTypePtr()) || isa<TypeOfExprType>(t.type.getTypePtr()))
      return fail(t.location, "new group type inquiry/carrier/global declaration is outside closed by-value flow");
    affected.insert(canonical(t.owner));
  }
  for (const auto* v : inventory.values) if (literal(v->getLocation()) && carriesNew(v->getType())) {
    used=true;
    const auto* owner=dyn_cast<FunctionDecl>(v->getDeclContext());
    if (!owner || role(v->getType())==Role::None || role(v->getType())==Role::Internal16)
      return fail(v->getLocation(), "group value must be an ordinary local/by-value parameter");
    affected.insert(canonical(owner));
  }
  for (const auto& site : inventory.calls) if (site.owner && literal(site.call->getExprLoc()) &&
      newRole(role(site.call->getType()))) {
    used=true; affected.insert(canonical(site.owner));
  }
  if (!used) return true; // Tile32-only and ordinary CUDA keep their old gate.
  if (!enabled) return fail({}, "new group proof requires standard preprocessing and dav-c310-vec");
  // Conditional/empty macros in another caller header can change a physical
  // literal or insert a write only in target preprocessing. Close every
  // entered input, not only files containing an active declaration: genuine
  // SDK/compiler/native files use their existing identity gates; all caller
  // inputs require the restricted literal-include/no-directive/macro grammar.
  auto& source=context.getSourceManager();
  for (auto info=source.fileinfo_begin();info!=source.fileinfo_end();++info) {
    const auto file=source.translateFile(info->first);
    if (file.isInvalid() || nativeInput(file)) continue;
    const auto begin=source.getLocForStartOfFile(file);
    if (!directiveFreeSource(begin))
      return fail(begin,"caller input preprocessing is not closed for full-warp group proof");
  }
  for (const auto* alias : inventory.aliases) if (literal(alias->getLocation()) && carriesNew(alias->getUnderlyingType()))
    return fail(alias->getLocation(), "group type alias/inquiry is not admitted");
  for (const auto* space : inventory.namespaces) if (literal(space->getLocation()) &&
      (space->getQualifiedNameAsString()=="cooperative_groups" || space->getQualifiedNameAsString()=="ascify_cg"))
    return fail(space->getLocation(), "caller group namespace declarations can alter ADL/provider identity");
  for (const auto* alias : inventory.namespaceAliases) if (literal(alias->getLocation()) &&
      alias->getName()=="ascify_cg")
    return fail(alias->getLocation(), "caller alias collides with the published target group namespace");
  // Both directions are deliberate: every group helper caller and every
  // source helper below the path is closed; a hidden block barrier cannot be
  // concealed inside an ostensibly nongroup helper called by varying lanes.
  bool changed=true;
  while (changed) {
    changed=false;
    for (const auto& site : inventory.calls) {
      const auto* target=site.call->getDirectCallee();
      if (!site.owner || !target || isa<CUDAKernelCallExpr>(site.call) ||
          !literal(site.owner->getLocation()) || !literal(target->getLocation())) continue;
      if (affected.count(canonical(target))) changed|=affected.insert(canonical(site.owner)).second;
      if (affected.count(canonical(site.owner))) changed|=affected.insert(canonical(target)).second;
    }
  }
  for (const auto& site : inventory.calls) if (site.owner && affected.count(canonical(site.owner)) && site.observation)
    return fail(site.call->getExprLoc(), "group path call is observed in an unevaluated/type inquiry");
  std::set<const FunctionDecl*> kernels;
  for (const auto* f : affected) if (f->hasAttr<CUDAGlobalAttr>()) kernels.insert(f);
  if (kernels.empty()) return fail({}, "new group definitions have no closed direct-launch kernel root");
  for (const auto* kernel : kernels) {
    bool launched=false;
    for (const auto& site : inventory.calls) if (canonical(site.call->getDirectCallee())==kernel) {
      const auto* launch=dyn_cast<CUDAKernelCallExpr>(site.call);
      if (!launch || !literal(launch->getExprLoc()))
        return fail(site.call->getExprLoc(), "group kernel has an unproved caller/launch");
      const auto threads=blockThreads(launch);
      if (!threads) return fail(launch->getExprLoc(), "group kernel requires a closed immutable 1D block32..1024 multiple32 launch");
      if (!function(kernel->getDefinition(),*threads,{})) return false;
      approve(launch); launched=true;
    }
    if (!launched) return fail(kernel->getLocation(), "group kernel has no visible proved launch");
  }
  for (const auto* f : affected) if (!reached.count(f))
    return fail(f->getLocation(), "group helper has an unclosed caller/origin/definition");
  for (const auto* r : inventory.references) {
    const auto* f=dyn_cast<FunctionDecl>(r->getDecl());
    if (f && affected.count(canonical(f)) && !approvedCallees.count(r))
      return fail(r->getExprLoc(), "group helper/kernel address or indirect caller escapes the closure");
  }
  for (const auto& site : inventory.calls) if (site.owner && affected.count(canonical(site.owner)) &&
      !approvedCalls.count(site.call))
    return fail(site.call->getExprLoc(), "group path contains an unproved call site");
  result.kernels=kernels.size(); result.helpers=reached.size()-kernels.size(); result.syncs=syncSites.size();
  return true;
}
} // namespace

bool ProveFullWarpGroups(ASTContext& context,
    const std::function<bool(SourceLocation)>& owned,
    const std::function<bool(const Decl*)>& cg,
    const std::function<bool(const Decl*)>& sdk,
    const std::function<bool(const FunctionDecl*)>& printfOwner,
    const std::function<bool(llvm::StringRef,SourceLocation)>& macroFree,
    const std::function<bool(FileID)>& nativeInput,
    bool enabled, FullWarpGroupStats& stats, SourceLocation& bad, std::string& error) {
  return Proof(context,owned,cg,sdk,printfOwner,macroFree,nativeInput,stats,bad,error).run(enabled);
}
} // namespace ascify
