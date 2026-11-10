#include "DynamicWarpGroups.h"
#include <map>
#include <set>
#include <vector>
#include <utility>
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Builtins.h"
#include "clang/Lex/Lexer.h"

namespace ascify { namespace {
using namespace clang;
template <class Member>
static auto memberClass(const Member* member,int)
    -> decltype(member->getClass(),clang::QualType()) {return clang::QualType(member->getClass(),0);}
template <class Member>
static auto memberClass(const Member* member,long)
    -> decltype(member->getQualifier().getAsType(),clang::QualType()) {
  const auto qualifier=member->getQualifier();using Qualifier=decltype(member->getQualifier());
  if(qualifier.getKind()!=Qualifier::Kind::Type)return {};
  return clang::QualType(qualifier.getAsType(),0);
}
template <class Context,class Record>
static auto recordType(Context& c,const Record* r,int)
    -> decltype(c.getRecordType(r),clang::QualType()) {return c.getRecordType(r);}
template <class Context,class Record>
static auto recordType(Context& c,const Record* r,long)
    -> decltype(c.getCanonicalTagType(r),clang::QualType()) {return c.getCanonicalTagType(r);}
enum class Kind { None, Coalesced, Grid, Greater };
Kind recordKind(const CXXRecordDecl* record) {
  if(!record)return Kind::None;const auto name=record->getQualifiedNameAsString();
  if(const auto* specialization=dyn_cast<ClassTemplateSpecializationDecl>(record))
    if(specialization->getSpecializedTemplate()->getQualifiedNameAsString()=="cooperative_groups::greater")return Kind::Greater;
  return name=="cooperative_groups::coalesced_group"?Kind::Coalesced:
      name=="cooperative_groups::grid_group"?Kind::Grid:
      name=="cooperative_groups::greater"?Kind::Greater:Kind::None;
}
Kind kind(QualType t) {
  if(t.isNull() || t->isDependentType())return Kind::None;
  const auto* record=t.getUnqualifiedType()->getAsCXXRecordDecl();
  if(!record)return Kind::None;
  return recordKind(record);
}
// Refusal inventory follows the finite canonical type graph, including record
// fields/bases and template packs. Cycles stop by identity; no depth cap can
// hide a representation observation behind pointer/template aliases.
bool related(QualType first) {
  std::vector<QualType> work{first};std::set<const Type*> seen;
  const auto arguments=[&](const auto& values) {
    std::vector<const TemplateArgument*> pending;
    for(const auto& arg:values)pending.push_back(&arg);
    while(!pending.empty()) {
      const auto* arg=pending.back();pending.pop_back();
      if(arg->getKind()==TemplateArgument::Type)work.push_back(arg->getAsType());
      else if(arg->getKind()==TemplateArgument::Pack)
        for(const auto& item:arg->pack_elements())pending.push_back(&item);
      else if(arg->getKind()==TemplateArgument::Expression)work.push_back(arg->getAsExpr()->getType());
      else if(arg->getKind()==TemplateArgument::Declaration)
        work.push_back(arg->getAsDecl()->getType());
    }
  };
  while(!work.empty()) {
    const auto t=work.back();work.pop_back();if(t.isNull())continue;
    if(kind(t)!=Kind::None)return true;
    if(const auto* record=t->getAsCXXRecordDecl())if(recordKind(record)==Kind::Greater)return true;
    const auto canonical=t.getCanonicalType();
    if(!seen.insert(canonical.getTypePtr()).second)continue;
    if(t->isPointerType()||t->isReferenceType())work.push_back(t->getPointeeType());
    if(const auto* array=dyn_cast<ArrayType>(t.getTypePtr()))work.push_back(array->getElementType());
    if(const auto* member=t->getAs<MemberPointerType>()) {
      work.push_back(member->getPointeeType());work.push_back(memberClass(member,0));
    }
    if(const auto* function=t->getAs<FunctionType>())work.push_back(function->getReturnType());
    if(const auto* function=t->getAs<FunctionProtoType>())
      for(const auto parameter:function->param_types())work.push_back(parameter);
    if(const auto* spec=t->getAs<TemplateSpecializationType>()) {
      const auto* declaration=spec->getTemplateName().getAsTemplateDecl();
      if(declaration&&declaration->getQualifiedNameAsString()=="cooperative_groups::greater")return true;
      arguments(spec->template_arguments());
    }
    if(const auto* record=t->getAsCXXRecordDecl()) {
      if(const auto* spec=dyn_cast<ClassTemplateSpecializationDecl>(record))arguments(spec->getTemplateArgs().asArray());
      if(const auto* definition=record->getDefinition()) {
        for(const auto* field:definition->fields())work.push_back(field->getType());
        for(const auto& base:definition->bases())work.push_back(base.getType());
      }
    }
  }
  return false;
}
const Expr* plain(const Expr* e) {return e?e->IgnoreParenImpCasts():nullptr;}
const VarDecl* variable(const Expr* e) {
  const auto* ref=dyn_cast_or_null<DeclRefExpr>(plain(e));
  return ref?dyn_cast<VarDecl>(ref->getDecl()):nullptr;
}
bool groupFunction(const FunctionDecl* f) {
  if(!f)return false;
  if(related(f->getReturnType()))return true;
  for(const auto* parameter:f->parameters())if(related(parameter->getType()))return true;
  const auto* method=dyn_cast<CXXMethodDecl>(f);
  return method && recordKind(method->getParent())!=Kind::None;
}
bool newDeclaration(const NamedDecl* declaration,ASTContext& context) {
  std::vector<const NamedDecl*> work{declaration};std::set<const NamedDecl*> seen;
  while(!work.empty()) {
    const auto* d=work.back();work.pop_back();if(!d||!seen.insert(d).second)continue;
    if(const auto* f=dyn_cast<FunctionDecl>(d)){if(groupFunction(f))return true;}
    if(const auto* f=dyn_cast<FunctionTemplateDecl>(d))work.push_back(f->getTemplatedDecl());
    if(const auto* c=dyn_cast<ClassTemplateDecl>(d))work.push_back(c->getTemplatedDecl());
    if(const auto* record=dyn_cast<CXXRecordDecl>(d)){if(related(recordType(context,record,0)))return true;}
    if(const auto* alias=dyn_cast<TypedefNameDecl>(d)){if(related(alias->getUnderlyingType()))return true;}
    if(const auto* alias=dyn_cast<TypeAliasTemplateDecl>(d))work.push_back(alias->getTemplatedDecl());
    if(const auto* shadow=dyn_cast<UsingShadowDecl>(d))work.push_back(shadow->getTargetDecl());
    if(const auto* usingDeclaration=dyn_cast<UsingDecl>(d))for(const auto* shadow:usingDeclaration->shadows())work.push_back(shadow);
  }
  return false;
}
struct Invocation {const FunctionDecl* owner;const CallExpr* call;bool observation;};
struct TypeUse {TypeLoc location;const FunctionDecl* owner;};
class Inventory:public RecursiveASTVisitor<Inventory> {
  const FunctionDecl* current=nullptr;unsigned observed=0;
  const Builtin::Context& builtins;
 public:
  explicit Inventory(const Builtin::Context& b):builtins(b){}
  std::vector<const ValueDecl*> values;
  std::vector<const DeclRefExpr*> references;
  std::vector<const MemberExpr*> members;
  std::vector<const UnresolvedLookupExpr*> lookups;
  std::vector<const NamedDecl*> declarations;
  std::vector<Invocation> calls;
  std::vector<TypeUse> types;
  std::vector<std::pair<const FunctionDecl*,const Expr*>> signatureObservers;
  std::vector<const AsmStmt*> assemblies;
  std::vector<const FileScopeAsmDecl*> fileAssemblies;
  std::vector<std::pair<const NamedDecl*,SourceLocation>> templateDeclarations;
  std::set<const Expr*> directCallees;
  bool shouldVisitTemplateInstantiations() const{return true;}
  bool shouldVisitImplicitCode() const{return true;}
  bool TraverseDecl(Decl* d) {
    const auto* saved=current;if(const auto* f=dyn_cast_or_null<FunctionDecl>(d))current=f;
    const bool ok=RecursiveASTVisitor<Inventory>::TraverseDecl(d);current=saved;return ok;
  }
  bool TraverseTypeLoc(TypeLoc t,bool qualifier=true) {
    ++observed;const bool ok=RecursiveASTVisitor<Inventory>::TraverseTypeLoc(t,qualifier);--observed;return ok;
  }
  bool TraverseStmt(Stmt* s) {
    const auto* call=dyn_cast_or_null<CallExpr>(s);
    const auto* callee=call?call->getDirectCallee():nullptr;
    const bool builtinInquiry=callee&&callee->getBuiltinID()!=0&&
        builtins.isUnevaluated(callee->getBuiltinID());
    const bool inquiry=builtinInquiry||(s && (isa<UnaryExprOrTypeTraitExpr>(s)||isa<CXXNoexceptExpr>(s)||
        isa<CXXTypeidExpr>(s)||isa<TypeTraitExpr>(s)||isa<ChooseExpr>(s)||isa<GenericSelectionExpr>(s)));
    observed+=inquiry;const bool ok=RecursiveASTVisitor<Inventory>::TraverseStmt(s);observed-=inquiry;return ok;
  }
  bool VisitValueDecl(ValueDecl* d){values.push_back(d);return true;}
  bool VisitNamedDecl(NamedDecl* d){declarations.push_back(d);return true;}
  bool VisitDeclRefExpr(DeclRefExpr* e){references.push_back(e);return true;}
  bool VisitMemberExpr(MemberExpr* e){members.push_back(e);return true;}
  bool VisitUnresolvedLookupExpr(UnresolvedLookupExpr* e){lookups.push_back(e);return true;}
  bool VisitTypeLoc(TypeLoc t){types.push_back({t,current});return true;}
  bool VisitCallExpr(CallExpr* c){calls.push_back({current,c,observed!=0});directCallees.insert(plain(c->getCallee()));return true;}
  bool VisitPredefinedExpr(PredefinedExpr* e){signatureObservers.push_back({current,e});return true;}
  bool VisitSourceLocExpr(SourceLocExpr* e){signatureObservers.push_back({current,e});return true;}
  bool VisitAsmStmt(AsmStmt* s){assemblies.push_back(s);return true;}
  bool VisitFileScopeAsmDecl(FileScopeAsmDecl* d){fileAssemblies.push_back(d);return true;}
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc& a) {
    if(a.getArgument().getKind()==TemplateArgument::Declaration)
      templateDeclarations.push_back({a.getArgument().getAsDecl(),a.getLocation()});
    return RecursiveASTVisitor<Inventory>::TraverseTemplateArgumentLoc(a);
  }
  bool TraverseTemplateArgument(const TemplateArgument& a) {
    if(a.getKind()==TemplateArgument::Declaration)
      templateDeclarations.push_back({a.getAsDecl(),{}});
    // LLVM23 RAV888..963: Pack delegates to derived TraverseTemplateArguments,
    // whose elements call this override. Preserve Type/Expression traversal.
    return RecursiveASTVisitor<Inventory>::TraverseTemplateArgument(a);
  }
};
class Proof {
  ASTContext& context;SourceManager& source;
  const std::function<bool(SourceLocation)>& owned;
  const std::function<bool(SourceLocation)>& caller;
  const std::function<bool(const Decl*)>& sdk;
  const std::function<bool(llvm::StringRef,SourceLocation)>& macroFree;
  DynamicWarpGroupStats& stats;SourceLocation& bad;std::string& error;
  Inventory inventory;
  std::map<const VarDecl*,const FunctionDecl*> groups;
  std::set<const VarDecl*> staticParents,staticBlocks;
  std::set<const void*> greaterTypeLocations;
  std::set<const Expr*> admittedReferences;
  std::set<const Expr*> admittedCallees;
  bool fail(SourceLocation loc,llvm::StringRef message){if(error.empty()){bad=loc;error=message.str();}return false;}
  bool literal(SourceLocation loc) const{return loc.isFileID()&&owned(loc);}
  bool ownExpansion(SourceLocation loc) const{return loc.isValid()&&owned(source.getExpansionLoc(loc));}
  bool callerExpansion(SourceLocation loc) const{return loc.isValid()&&caller(source.getExpansionLoc(loc));}
  bool tokens(SourceRange range) const {
    const auto begin=range.getBegin(),last=range.getEnd();
    if(!literal(begin)||!literal(last)||source.getFileID(begin)!=source.getFileID(last))return false;
    const auto end=Lexer::getLocForEndOfToken(last,0,source,context.getLangOpts());
    bool invalid=false;const auto bytes=source.getBufferData(source.getFileID(begin),&invalid);
    if(invalid||end.isInvalid()||source.getFileOffset(begin)>=source.getFileOffset(end))return false;
    const auto start=source.getFileOffset(begin),finish=source.getFileOffset(end);
    if(finish>bytes.size())return false;
    Lexer lexer(source.getLocForStartOfFile(source.getFileID(begin)),context.getLangOpts(),
                bytes.begin(),bytes.begin()+start,bytes.end());
    for(;;) {
      Token token;lexer.LexFromRawLexer(token);if(token.is(tok::eof))break;
      if(source.getFileOffset(token.getLocation())>=finish)break;
      if(token.needsCleaning()||token.is(tok::hash)||token.is(tok::hashhash))return false;
      if(token.isAnyIdentifier()&&!macroFree(Lexer::getSpelling(token,source,context.getLangOpts()),token.getLocation()))return false;
    }
    return true;
  }
  bool written(const Expr* e) const{return e&&tokens(e->getSourceRange());}
  bool calleeWritten(const CallExpr* call) const {
    const auto* callee=plain(call->getCallee());
    if(const auto* reference=dyn_cast_or_null<DeclRefExpr>(callee))
      return literal(reference->getLocation())&&tokens(reference->getSourceRange())&&
          (!reference->hasQualifier()||tokens(reference->getQualifierLoc().getSourceRange()));
    if(const auto* member=dyn_cast_or_null<MemberExpr>(callee))
      return literal(member->getMemberLoc())&&literal(member->getExprLoc())&&
          tokens(member->getSourceRange())&&
          (!member->hasQualifier()||tokens(member->getQualifierLoc().getSourceRange()));
    return false;
  }
  bool admitCall(const CallExpr* call) {
    const auto* leaf=plain(call->getCallee());
    for(const Expr* node=call->getCallee();node!=leaf;) {
      stats.valueExpressions.insert(node);
      if(const auto* parens=dyn_cast<ParenExpr>(node))node=parens->getSubExpr();
      else if(const auto* cast=dyn_cast<ImplicitCastExpr>(node)) {
        const auto* pointer=cast->getType()->getAs<clang::PointerType>();
        if(cast->getCastKind()!=CK_FunctionToPointerDecay||!pointer||
            !context.hasSameType(pointer->getPointeeType(),cast->getSubExpr()->getType()))return false;
        node=cast->getSubExpr();
      }else return false;
    }
    stats.approvedCalls.insert(call);admittedCallees.insert(leaf);return true;
  }
  bool trusted(const FunctionDecl* f) const {
    if(!f || !f->hasAttr<CUDADeviceAttr>() || f->hasAttr<CUDAHostAttr>() || f->isVariadic())return false;
    for(const auto* d:f->redecls())if(!sdk(d))return false;
    return true;
  }
  bool trustedType(QualType t) const {
    const auto* r=t.isNull()?nullptr:t->getAsCXXRecordDecl();if(!r||!r->getDefinition())return false;
    for(const auto* d:r->redecls())if(!sdk(d))return false;return true;
  }
  bool legacyReduceImport(const NamedDecl* declaration) const {
    const auto target=[&](const NamedDecl* d) {
      const FunctionDecl* function=dyn_cast_or_null<FunctionDecl>(d);
      if(const auto* primary=dyn_cast_or_null<FunctionTemplateDecl>(d)) {
        for(const auto* redecl:primary->redecls())if(!sdk(redecl))return false;
        function=primary->getTemplatedDecl();
      }
      return function&&function->getQualifiedNameAsString()=="cooperative_groups::reduce"&&trusted(function);
    };
    if(const auto* shadow=dyn_cast<UsingShadowDecl>(declaration))return target(shadow->getTargetDecl());
    const auto* import=dyn_cast<UsingDecl>(declaration);if(!import||import->shadow_size()==0)return false;
    for(const auto* shadow:import->shadows())if(!target(shadow->getTargetDecl()))return false;
    return true;
  }
  bool newLookup(const UnresolvedLookupExpr* lookup) const {
    for(const auto* declaration:lookup->decls())if(newDeclaration(declaration,context))return true;
    return false;
  }
  bool legacyReduceLookup(const UnresolvedLookupExpr* lookup) const {
    if(lookup->decls_begin()==lookup->decls_end())return false;
    for(const auto* declaration:lookup->decls()) {
      const auto* primary=dyn_cast<FunctionTemplateDecl>(declaration);
      const auto* function=primary?primary->getTemplatedDecl():dyn_cast<FunctionDecl>(declaration);
      if(!function||function->getQualifiedNameAsString()!="cooperative_groups::reduce"||!trusted(function))return false;
      if(primary)for(const auto* redecl:primary->redecls())if(!sdk(redecl))return false;
    }
    return true;
  }
  bool greaterConstruction(const CXXConstructExpr* expression) {
    const auto* temporary=dyn_cast<CXXTemporaryObjectExpr>(expression);
    const auto* info=temporary?temporary->getTypeSourceInfo():nullptr;
    if(!info||!tokens(info->getTypeLoc().getSourceRange()))return false;
    const auto root=info->getTypeLoc();
    const auto type=root.getUnqualifiedLoc().getAs<TemplateSpecializationTypeLoc>();
    if(!type||type.getNumArgs()!=1)return false;
    const auto& argument=type.getArgLoc(0);
    if(argument.getArgument().getKind()!=TemplateArgument::Type||
        !context.hasSameType(argument.getArgument().getAsType(),context.IntTy))return false;
    const auto* argumentInfo=argument.getTypeSourceInfo();
    if(!argumentInfo||!argumentInfo->getTypeLoc().getAs<BuiltinTypeLoc>())return false;
    const auto* primary=dyn_cast_or_null<ClassTemplateDecl>(type.getTypePtr()->getTemplateName().getAsTemplateDecl());
    if(!primary||primary->getQualifiedNameAsString()!="cooperative_groups::greater")return false;
    for(const auto* redecl:primary->redecls())if(!sdk(redecl))return false;
    greaterTypeLocations.insert(root.getOpaqueData());
    greaterTypeLocations.insert(root.getUnqualifiedLoc().getOpaqueData());
    return true;
  }
  bool owner(const FunctionDecl* f) const {
    if(!f || !literal(f->getLocation()) || f->isDependentContext() || f->isConstexpr() ||
        f->getTemplatedKind()!=FunctionDecl::TK_NonTemplate || f->hasAttr<CUDAHostAttr>() ||
        !(f->hasAttr<CUDADeviceAttr>()||f->hasAttr<CUDAGlobalAttr>()) || isa<CXXMethodDecl>(f))return false;
    return !related(f->getReturnType());
  }
  bool builtin32(QualType t) const {
    return !t.isNull()&&!t.isVolatileQualified()&&(context.hasSameUnqualifiedType(t,context.IntTy)||
        context.hasSameUnqualifiedType(t,context.UnsignedIntTy)||context.hasSameUnqualifiedType(t,context.BoolTy));
  }
  bool tile32(QualType t) const {
    const auto* r=t.isNull()?nullptr:t->getAsCXXRecordDecl();
    const auto* s=dyn_cast_or_null<ClassTemplateSpecializationDecl>(r);
    if(!s||!trustedType(t)||s->getSpecializationKind()==TSK_ExplicitSpecialization||
        s->getSpecializedTemplate()->getQualifiedNameAsString()!="cooperative_groups::thread_block_tile"||
        s->getTemplateArgs().size()!=2)return false;
    for(const auto* d:s->getSpecializedTemplate()->redecls())if(!sdk(d))return false;
    const auto& args=s->getTemplateArgs();
    if(args[0].getKind()!=TemplateArgument::Integral||args[0].getAsIntegral()!=32||
        args[1].getKind()!=TemplateArgument::Type)return false;
    const auto parent=args[1].getAsType();if(parent->isVoidType())return true;
    const auto* p=parent->getAsCXXRecordDecl();return p&&trustedType(parent)&&
        p->getQualifiedNameAsString()=="cooperative_groups::thread_block";
  }
  const Expr* initializer(const VarDecl* v) const {
    const auto* e=v->getInit();
    // Class prvalues acquire at most qualification of a direct const value.
    if(const auto* cast=dyn_cast_or_null<ImplicitCastExpr>(e)) {
      if(cast->getCastKind()!=CK_NoOp || !context.hasSameUnqualifiedType(cast->getType(),cast->getSubExpr()->getType()))return nullptr;
      e=cast->getSubExpr();
    }
    return e;
  }
  bool ordinaryWrittenType(const VarDecl* v) const {
    const auto* info=v->getTypeSourceInfo();if(!info||!tokens(SourceRange(v->getBeginLoc(),v->getLocation())))return false;
    auto t=info->getTypeLoc().getUnqualifiedLoc();if(!tokens(t.getSourceRange()))return false;
    if(auto a=t.getAs<AutoTypeLoc>())return !a.getTypePtr()->isDecltypeAuto();
    return bool(t.getAs<RecordTypeLoc>())&&trustedType(v->getType());
  }
  bool literal32(const TemplateArgumentLoc& argument) const {
    const auto k=argument.getArgument().getKind();
    const Expr* e=k==TemplateArgument::Expression?argument.getSourceExpression():
        k==TemplateArgument::Integral?argument.getSourceIntegralExpression():nullptr;
    e=plain(e);
    while(const auto* constant=dyn_cast_or_null<ConstantExpr>(e))e=plain(constant->getSubExpr());
    const auto* number=dyn_cast_or_null<IntegerLiteral>(e);
    return number&&literal(number->getLocation())&&number->getValue()==32&&written(number);
  }
  bool blockType(QualType type) const {
    const auto* r=type.isNull()?nullptr:type->getAsCXXRecordDecl();
    return r&&r->getQualifiedNameAsString()=="cooperative_groups::thread_block"&&trustedType(type);
  }
  bool staticWrittenType(const VarDecl* value,bool block) const {
    const auto* info=value->getTypeSourceInfo();if(!info||!tokens(SourceRange(value->getBeginLoc(),value->getLocation())))return false;
    const auto root=info->getTypeLoc();
    if(!tokens(root.getSourceRange()))return false;
    const auto t=root.getUnqualifiedLoc();
    if(const auto a=t.getAs<AutoTypeLoc>())return !a.getTypePtr()->isDecltypeAuto();
    if(block)return bool(t.getAs<RecordTypeLoc>())&&blockType(value->getType());
    const auto written=t.getAs<TemplateSpecializationTypeLoc>();
    if(!written||written.getNumArgs()<1||written.getNumArgs()>2||!literal32(written.getArgLoc(0)))return false;
    const auto* primary=dyn_cast_or_null<ClassTemplateDecl>(written.getTypePtr()->getTemplateName().getAsTemplateDecl());
    if(!primary||primary->getQualifiedNameAsString()!="cooperative_groups::thread_block_tile")return false;
    for(const auto* d:primary->redecls())if(!sdk(d))return false;
    if(written.getNumArgs()==2) {
      const auto& argument=written.getArgLoc(1);
      if(argument.getArgument().getKind()!=TemplateArgument::Type)return false;
      const auto* parent=argument.getTypeSourceInfo();
      if(!parent||
          argument.getArgument().getAsType().hasQualifiers())return false;
      const auto p=parent->getTypeLoc().getUnqualifiedLoc();
      if(const auto builtin=p.getAs<BuiltinTypeLoc>()) {
        if(!builtin.getType()->isVoidType())return false;
      }else if(!p.getAs<RecordTypeLoc>()||!blockType(p.getType()))return false;
    }
    return true;
  }
  bool staticTypeUse(TypeLoc use) const {
    if(!tokens(use.getSourceRange()))return false;
    const auto inside=[&](const VarDecl* declaration) {
      const auto root=declaration->getTypeSourceInfo()->getTypeLoc();
      const auto a=source.getDecomposedLoc(root.getBeginLoc()),z=source.getDecomposedLoc(root.getEndLoc());
      const auto x=source.getDecomposedLoc(use.getBeginLoc()),y=source.getDecomposedLoc(use.getEndLoc());
      return a.first==z.first&&a.first==x.first&&a.first==y.first&&a.second<=x.second&&y.second<=z.second;
    };
    for(const auto* value:staticParents)if(inside(value))return true;
    for(const auto* value:staticBlocks)if(inside(value))return true;
    return false;
  }
  bool staticReferences(const VarDecl* value,bool block) const {
    for(const auto* reference:inventory.references)if(reference->getDecl()==value) {
      if(!written(reference)||!literal(reference->getLocation()))return false;
      bool admitted=false;
      for(const auto& occurrence:inventory.calls) {
        const auto* call=occurrence.call;const auto* function=call->getDirectCallee();
        if(!function||!trusted(function)||!written(call)||!calleeWritten(call)||
            occurrence.observation||call->getNumArgs()==0||plain(call->getArg(0))!=reference)continue;
        const auto name=function->getQualifiedNameAsString();
        admitted|=block?(name=="cooperative_groups::tiled_partition"&&call->getNumArgs()==1):
            (name=="cooperative_groups::binary_partition"&&call->getNumArgs()==2)||
            (name=="cooperative_groups::sync"&&call->getNumArgs()==1);
      }
      if(!admitted)return false;
    }
    return true;
  }
  const CallExpr* staticConversionFactory(const VarDecl* value,const ExprWithCleanups* cleanup) const {
    // The owned SDK converts a tile32<thread_block> prvalue to tile32<void>
    // through this one implicit conversion, optionally qualifying its prvalue
    // const. No other cleanup/cast/object path acquires a factory witness.
    if(cleanup->getNumObjects()!=0||cleanup->cleanupsHaveSideEffects()||
        !context.hasSameType(cleanup->getType(),value->getType()))return nullptr;
    const Expr* converted=cleanup->getSubExpr();
    const auto* resultQualification=dyn_cast<ImplicitCastExpr>(converted);
    if(resultQualification&&resultQualification->getCastKind()==CK_NoOp) {
      if(!value->getType().isConstQualified()||value->getType().isVolatileQualified()||
          !resultQualification->isPRValue()||!resultQualification->getSubExpr()->isPRValue()||
          !context.hasSameType(resultQualification->getType(),value->getType())||
          !context.hasSameType(resultQualification->getSubExpr()->getType(),value->getType().getUnqualifiedType()))return nullptr;
      converted=resultQualification->getSubExpr();
    }else resultQualification=nullptr;
    const auto* cast=dyn_cast<ImplicitCastExpr>(converted);
    const auto* call=cast?dyn_cast<CXXMemberCallExpr>(cast->getSubExpr()):nullptr;
    const auto* member=call?dyn_cast<MemberExpr>(call->getCallee()):nullptr;
    const auto* method=call?dyn_cast_or_null<CXXConversionDecl>(call->getDirectCallee()):nullptr;
    if(!cast||cast->getCastKind()!=CK_UserDefinedConversion||!call||call->getNumArgs()!=0||
        !member||member->isArrow()||member->hasQualifier()||member->hasExplicitTemplateArgs()||
        !method||member->getMemberDecl()!=method||!trusted(method))return nullptr;
    const auto target=value->getType().getUnqualifiedType();
    const auto concrete=recordType(context,method->getParent(),0);
    const auto* from=dyn_cast<ClassTemplateSpecializationDecl>(method->getParent());
    const auto* to=dyn_cast_or_null<ClassTemplateSpecializationDecl>(target->getAsCXXRecordDecl());
    if(!tile32(target)||!tile32(concrete)||!from||!to||
        !blockType(from->getTemplateArgs()[1].getAsType())||
        !to->getTemplateArgs()[1].getAsType()->isVoidType()||
        !context.hasSameType(cast->getType(),target)||!context.hasSameType(call->getType(),target))return nullptr;
    for(const auto* declaration:method->redecls()) {
      const auto* conversion=dyn_cast<CXXConversionDecl>(declaration);
      if(!conversion||!conversion->isConst()||conversion->isVolatile()||conversion->isStatic()||
          conversion->isVirtual()||conversion->isDeleted()||conversion->getNumParams()!=0||
          conversion->getRefQualifier()!=RQ_None||
          conversion->getParent()->getCanonicalDecl()!=from->getCanonicalDecl()||
          !context.hasSameType(conversion->getConversionType(),target)||
          !context.hasSameType(conversion->getReturnType(),target))return nullptr;
    }
    const auto* qualification=dyn_cast<ImplicitCastExpr>(member->getBase());
    const auto* temporary=qualification?dyn_cast<MaterializeTemporaryExpr>(qualification->getSubExpr()):nullptr;
    const auto* factory=temporary?dyn_cast<CallExpr>(temporary->getSubExpr()):nullptr;
    if(!qualification||qualification->getCastKind()!=CK_NoOp||
        !qualification->getType().isConstQualified()||qualification->getType().isVolatileQualified()||
        !context.hasSameUnqualifiedType(qualification->getType(),concrete)||
        !temporary||temporary->getExtendingDecl()||temporary->getStorageDuration()!=SD_FullExpression||
        temporary->getType().hasQualifiers()||!context.hasSameType(temporary->getType(),concrete)||
        !factory||!context.hasSameType(factory->getType(),concrete)||
        call->getImplicitObjectArgument()!=member->getBase())return nullptr;
    // All implicit nodes must cover exactly the source-written factory call;
    // an explicitly written conversion/member call cannot use this path.
    for(const Expr* node:{static_cast<const Expr*>(cleanup),static_cast<const Expr*>(cast),
        static_cast<const Expr*>(call),static_cast<const Expr*>(member),
        static_cast<const Expr*>(qualification),static_cast<const Expr*>(temporary)})
      if(node->getBeginLoc()!=factory->getBeginLoc()||node->getEndLoc()!=factory->getEndLoc())return nullptr;
    if(resultQualification&&(resultQualification->getBeginLoc()!=factory->getBeginLoc()||
        resultQualification->getEndLoc()!=factory->getEndLoc()))return nullptr;
    return written(factory)?factory:nullptr;
  }
  bool staticParent(const VarDecl* value,const FunctionDecl* ownerFunction) {
    if(!value||!tile32(value->getType())||!value->isLocalVarDecl()||isa<ParmVarDecl>(value)||
        value->getDeclContext()!=ownerFunction||value->hasAttrs()||value->isStaticLocal()||
        value->getTLSKind()!=VarDecl::TLS_None||value->getType()->isReferenceType()||
        value->getType().isVolatileQualified()||!literal(value->getLocation())||
        !staticReferences(value,false)||!staticWrittenType(value,false))return false;
    const Expr* init=value->getInit();
    if(const auto* cleanup=dyn_cast_or_null<ExprWithCleanups>(init))
      init=staticConversionFactory(value,cleanup);
    else init=plain(init);
    if(const auto* conversion=dyn_cast_or_null<CXXConstructExpr>(init)) {
      if(conversion->getNumArgs()!=1||!tile32(conversion->getType())||
          !trusted(conversion->getConstructor())||!written(conversion))return false;
      init=plain(conversion->getArg(0));
    }
    const auto* partition=dyn_cast_or_null<CallExpr>(init);
    const auto* function=partition?partition->getDirectCallee():nullptr;
    const auto* callee=partition?dyn_cast_or_null<DeclRefExpr>(plain(partition->getCallee())):nullptr;
    if(!partition||partition->getNumArgs()!=1||!function||!trusted(function)||
        function->getQualifiedNameAsString()!="cooperative_groups::tiled_partition"||
        !tile32(partition->getType())||!written(partition)||!calleeWritten(partition)||
        !callee||!callee->hasExplicitTemplateArgs()||callee->getNumTemplateArgs()!=1)return false;
    if(!literal32(callee->getTemplateArgs()[0]))return false;
    const auto* block=variable(partition->getArg(0));
    if(!block||!block->isLocalVarDecl()||isa<ParmVarDecl>(block)||block->isStaticLocal()||
        block->getDeclContext()!=ownerFunction||block->hasAttrs()||block->getTLSKind()!=VarDecl::TLS_None||
        block->getType()->isReferenceType()||block->getType().isVolatileQualified()||
        !literal(block->getLocation())||!written(plain(partition->getArg(0)))||
        !trustedType(block->getType())||
        block->getType()->getAsCXXRecordDecl()->getQualifiedNameAsString()!="cooperative_groups::thread_block"||
        !staticReferences(block,true)||!staticWrittenType(block,true))return false;
    const auto* factory=dyn_cast_or_null<CallExpr>(plain(block->getInit()));
    const bool proved=factory&&factory->getNumArgs()==0&&written(factory)&&calleeWritten(factory)&&
        trusted(factory->getDirectCallee())&&
        factory->getDirectCallee()->getQualifiedNameAsString()=="cooperative_groups::this_thread_block";
    if(proved){staticParents.insert(value);staticBlocks.insert(block);}return proved;
  }
  bool namespaceAlias(const NamespaceAliasDecl* alias) {
    const auto* target=alias->getAliasedNamespace();
    const auto* original=alias->getNamespace();
    if(!original||original->getQualifiedNameAsString()!="cooperative_groups")return true;
    for(const auto* declaration:original->redecls())if(!sdk(declaration))
      return fail(alias->getLocation(),"namespace alias target lacks complete owned CG provenance");
    if(const auto* parent=dyn_cast<NamespaceAliasDecl>(target))
      if(!stats.namespaceAliases.count(parent))return fail(alias->getLocation(),"namespace alias chain lacks an earlier literal witness");
    const auto begin=alias->getNamespaceLoc(),last=alias->getTargetNameLoc();
    if(!literal(begin)||!literal(last)||!literal(alias->getAliasLoc())||source.getFileID(begin)!=source.getFileID(last))
      return fail(alias->getLocation(),"CG namespace alias requires literal selected-source tokens");
    const auto end=Lexer::getLocForEndOfToken(last,0,source,context.getLangOpts());
    bool invalid=false;const auto bytes=source.getBufferData(source.getFileID(begin),&invalid);
    if(invalid||end.isInvalid()||source.getFileOffset(begin)>=source.getFileOffset(end))
      return fail(alias->getLocation(),"CG namespace alias has no complete physical token range");
    const std::string text=bytes.slice(source.getFileOffset(begin),source.getFileOffset(end)).str();
    if(text.find('\\')!=std::string::npos||text.find('\r')!=std::string::npos)
      return fail(alias->getLocation(),"CG namespace alias contains physical splices");
    Lexer lexer(begin,context.getLangOpts(),text.data(),text.data(),text.data()+text.size());
    lexer.SetCommentRetentionState(true);std::vector<std::string> tokens;
    for(;;){Token token;lexer.LexFromRawLexer(token);if(token.is(tok::eof))break;
      if(token.is(tok::comment)||tokens.size()==6)return fail(alias->getLocation(),"CG namespace alias has an unproved token alternative");
      tokens.push_back(Lexer::getSpelling(token,source,context.getLangOpts()));}
    const std::vector<std::string> ordinary={"namespace",alias->getNameAsString(),"=",target->getNameAsString()};
    const std::vector<std::string> global={"namespace",alias->getNameAsString(),"=","::",target->getNameAsString()};
    if(tokens!=ordinary&&tokens!=global)return fail(alias->getLocation(),"CG namespace alias requires exact literal target spelling");
    stats.namespaceAliases.insert(alias);return true;
  }
  bool registerVariable(const ValueDecl* value) {
    const auto* v=dyn_cast<VarDecl>(value);const auto* f=v?dyn_cast<FunctionDecl>(v->getDeclContext()):nullptr;
    if(!v||!literal(v->getLocation())||!owner(f)||!v->hasLocalStorage()||v->isStaticLocal()||v->getTLSKind()!=VarDecl::TLS_None||
        v->hasAttrs()||v->getType()->isReferenceType()||v->getType()->isPointerType()||
        v->getType().isVolatileQualified()||kind(v->getType())==Kind::None||!trustedType(v->getType())||!ordinaryWrittenType(v))
      return fail(value->getLocation(),"dynamic group requires a direct SDK type/auto local value in a non-template device function");
    const auto* call=dyn_cast_or_null<CallExpr>(initializer(v));
    if(!call||!written(call)||!calleeWritten(call)||!trusted(call->getDirectCallee()))return fail(v->getLocation(),"dynamic group must be initialized directly by a literal owned factory");
    const auto name=call->getDirectCallee()->getQualifiedNameAsString();const auto k=kind(v->getType());
    if(k==Kind::Grid) {
      if(name!="cooperative_groups::this_grid"||call->getNumArgs()!=0)return fail(v->getLocation(),"grid value requires exact this_grid factory");
    }else if(name=="cooperative_groups::coalesced_threads") {
      if(call->getNumArgs()!=0)return fail(v->getLocation(),"coalesced_threads requires no arguments");
    }else if(name=="cooperative_groups::binary_partition"||name=="cooperative_groups::labeled_partition") {
      const bool labeled=name=="cooperative_groups::labeled_partition";
      const auto* parent=call->getNumArgs()==2?variable(call->getArg(0)):nullptr;
      const auto registered=groups.find(parent);
      const bool dynamicParent=parent&&kind(parent->getType())==Kind::Coalesced&&
          registered!=groups.end()&&registered->second==f;
      if(call->getNumArgs()!=2||!parent||(!dynamicParent&&(labeled||!staticParent(parent,f)))||
          !parent->isLocalVarDecl()||isa<ParmVarDecl>(parent)||!parent->hasLocalStorage()||
          parent->isStaticLocal()||parent->getDeclContext()!=f||parent->hasAttrs()||
          parent->getType()->isReferenceType()||parent->getType().isVolatileQualified()||
          !literal(parent->getLocation())||!written(plain(call->getArg(0)))||
          !written(plain(call->getArg(1)))||!builtin32(plain(call->getArg(1))->getType())||
          (labeled&&(!context.hasSameUnqualifiedType(call->getArg(1)->getType(),context.IntTy)||
                     !context.hasSameUnqualifiedType(plain(call->getArg(1))->getType(),context.IntTy)))||
          call->getArg(1)->HasSideEffects(context))
        return fail(v->getLocation(),"partition requires an earlier same-function factory parent and effect-free builtin predicate/int label; static parents require SDK tile32 binary");
      if(dynamicParent&&!groupArgument(call->getArg(0),f))
        return fail(v->getLocation(),"partition parent lacks its exact local reference witness");
    }else return fail(v->getLocation(),"unproved dynamic group factory/copy");
    if(!admitCall(call))return fail(call->getExprLoc(),"factory callee has an unproved wrapper");
    groups.emplace(v,f);
    stats.localFactoryFunctions.insert(f);
    stats.valueExpressions.insert(call);stats.valueExpressions.insert(v->getInit());
    ++stats.values;++stats.factories;
    // The type-use walk below also checks all inner locations of this exact
    // declaration range. No decltype/type-trait or alias receives that range.
    return true;
  }
  bool groupArgument(const Expr* e,const FunctionDecl* f) {
    const auto* v=variable(e);const auto found=groups.find(v);
    if(found==groups.end()||found->second!=f||!written(plain(e)))return false;
    admittedReferences.insert(plain(e));
    const Expr* current=e;unsigned qualifications=0;
    while(current!=plain(e)) {
      stats.valueExpressions.insert(current);
      if(const auto* parens=dyn_cast<ParenExpr>(current))current=parens->getSubExpr();
      else if(const auto* cast=dyn_cast<ImplicitCastExpr>(current)) {
        if(cast->getCastKind()!=CK_NoOp||++qualifications!=1||
            !context.hasSameUnqualifiedType(cast->getType(),cast->getSubExpr()->getType())||
            cast->getType().isVolatileQualified()||cast->getSubExpr()->getType().isVolatileQualified())return false;
        current=cast->getSubExpr();
      }
      else return false;
    }
    stats.valueExpressions.insert(plain(e));return true;
  }
  bool operation(const Invocation& entry) {
    const auto* call=entry.call;const auto* f=call->getDirectCallee();
    if(stats.approvedCalls.count(call))return true;
    if(!groupFunction(f))return true;
    if(!owner(entry.owner)||entry.observation||!written(call)||!calleeWritten(call)||!trusted(f))
      return fail(call->getExprLoc(),"dynamic group operation is observed, indirect, macro-generated or lacks SDK provenance");
    if(const auto* member=dyn_cast_or_null<MemberExpr>(plain(call->getCallee()))) {
      if(!groupArgument(member->getBase(),entry.owner))return fail(call->getExprLoc(),"dynamic member object is not the unique local factory value");
      const auto* v=variable(member->getBase());const auto k=kind(v->getType());const auto name=f->getNameAsString();
      if(k==Kind::Grid) {
        if(name=="sync") {
          // Cooperative grid-wide rendezvous backed by the device-global
          // sense-reversing barrier; the AST proof still pins the exact
          // unique local factory value, zero arguments and void result.
          if(call->getNumArgs()!=0||!call->getType()->isVoidType())
            return fail(call->getExprLoc(),"grid sync requires a zero-argument void form");
        }else if((name!="thread_rank"&&name!="size")||call->getNumArgs()!=0||
            !context.hasSameUnqualifiedType(call->getType(),context.UnsignedLongLongTy))return fail(call->getExprLoc(),"grid group exposes only ULL rank/size/sync");
        ++stats.coordinateQueries;stats.scalarExpressions.insert(call);
      }else if((name=="thread_rank"||name=="size"||name=="num_threads"||name=="meta_group_rank"||name=="meta_group_size")&&call->getNumArgs()==0) {
        if(!context.hasSameUnqualifiedType(call->getType(),context.UnsignedIntTy))return fail(call->getExprLoc(),"coalesced query must preserve CUDA unsigned-int type");
      }else if(name=="shfl"&&call->getNumArgs()==2) {
        const auto* rank=dyn_cast<IntegerLiteral>(plain(call->getArg(1)));
        if(!rank||rank->getValue()!=0||!written(rank)||!builtin32(call->getType())||
            context.hasSameUnqualifiedType(call->getType(),context.BoolTy)||
            !context.hasSameUnqualifiedType(call->getType(),call->getArg(0)->getType())||
            call->getArg(0)->HasSideEffects(context))return fail(call->getExprLoc(),"coalesced shuffle admits only builtin int/uint broadcast from literal rank0");
        ++stats.shuffles;
      }else return fail(call->getExprLoc(),"unproved dynamic group member (sync/escape/representation is not admitted)");
    }else if(f->getQualifiedNameAsString()=="cooperative_groups::sync"&&call->getNumArgs()==1) {
      // The free sync(group) form on the grid value is the same cooperative
      // rendezvous as the member call: unique local factory argument only.
      if(!groupArgument(call->getArg(0),entry.owner)||
          kind(call->getArg(0)->getType())!=Kind::Grid||
          !call->getType()->isVoidType())
        return fail(call->getExprLoc(),"free grid sync requires the unique local grid value");
    }else if(f->getQualifiedNameAsString()=="cooperative_groups::reduce"&&call->getNumArgs()==3) {
      if(!groupArgument(call->getArg(0),entry.owner)||kind(call->getArg(0)->getType())!=Kind::Coalesced||
          !context.hasSameUnqualifiedType(call->getType(),context.IntTy)||
          !context.hasSameUnqualifiedType(call->getArg(1)->getType(),context.IntTy)||
          !context.hasSameUnqualifiedType(plain(call->getArg(1))->getType(),context.IntTy)||call->getArg(1)->HasSideEffects(context))
        return fail(call->getExprLoc(),"dynamic reduce requires local coalesced value and effect-free CUDA int");
      const auto* op=dyn_cast<CXXConstructExpr>(plain(call->getArg(2)));
      const auto* spec=op?dyn_cast_or_null<ClassTemplateSpecializationDecl>(op->getType()->getAsCXXRecordDecl()):nullptr;
      if(!op||!written(op)||op->getNumArgs()!=0||!op->getConstructor()->isDefaultConstructor()||
          !op->getConstructor()->isTrivial()||op->getConstructor()->isDeleted()||!spec||spec->getTemplateArgs().size()!=1||
          spec->getTemplateArgs()[0].getKind()!=TemplateArgument::Type||
          !context.hasSameUnqualifiedType(spec->getTemplateArgs()[0].getAsType(),context.IntTy))
        return fail(call->getExprLoc(),"dynamic reduce operation must be an exact owned zero-argument int functor");
      for(const auto* d:op->getConstructor()->redecls())if(!sdk(d))return fail(call->getExprLoc(),"untrusted dynamic functor constructor");
      const auto name=spec->getSpecializedTemplate()->getQualifiedNameAsString();
      if((name!="cooperative_groups::plus"&&name!="cooperative_groups::greater")||!trustedType(op->getType()))
        return fail(call->getExprLoc(),"dynamic reduce admits only owned int plus/max");
      for(const auto* d:spec->getSpecializedTemplate()->redecls())if(!sdk(d))return fail(call->getExprLoc(),"untrusted dynamic functor primary");
      if(name=="cooperative_groups::greater"&&!greaterConstruction(op))
        return fail(call->getExprLoc(),"new max functor requires its exact literal int temporary type");
      stats.functorConstructions.insert(op);
      ++stats.reductions;
    }else return fail(call->getExprLoc(),"dynamic factory/group argument escapes its direct local contract");
    return admitCall(call)||fail(call->getExprLoc(),"operation callee has an unproved wrapper");
  }
 public:
  Proof(ASTContext& c,const std::function<bool(SourceLocation)>& o,
      const std::function<bool(SourceLocation)>& a,const std::function<bool(const Decl*)>& s,
      const std::function<bool(llvm::StringRef,SourceLocation)>& m,
      DynamicWarpGroupStats& st,SourceLocation& l,std::string& e):context(c),source(c.getSourceManager()),owned(o),caller(a),sdk(s),macroFree(m),stats(st),bad(l),error(e),inventory(c.BuiltinInfo){}
  bool run(bool standard) {
    inventory.TraverseDecl(context.getTranslationUnitDecl());bool used=false;
    for(const auto* v:inventory.values)if(callerExpansion(v->getLocation())&&related(v->getType()))used=true;
    for(const auto& entry:inventory.calls)if(callerExpansion(entry.call->getExprLoc())&&groupFunction(entry.call->getDirectCallee()))used=true;
    for(const auto& use:inventory.types)if(callerExpansion(use.location.getBeginLoc())&&related(use.location.getType()))used=true;
    for(const auto* d:inventory.declarations)if(callerExpansion(d->getLocation())&&
        newDeclaration(d,context)&&!legacyReduceImport(d))used=true;
    for(const auto* reference:inventory.references)if(callerExpansion(reference->getLocation())&&
        (related(reference->getType())||groupFunction(dyn_cast<FunctionDecl>(reference->getDecl()))))used=true;
    for(const auto* member:inventory.members)if(callerExpansion(member->getMemberLoc())&&
        (related(member->getType())||groupFunction(dyn_cast<FunctionDecl>(member->getMemberDecl()))))used=true;
    for(const auto* lookup:inventory.lookups)if(callerExpansion(lookup->getNameLoc())&&
        newLookup(lookup)&&!legacyReduceLookup(lookup))used=true;
    if(!used)return true;
    if(!standard||!context.getLangOpts().CPlusPlus17||context.getLangOpts().CPlusPlus20)
      return fail({},"dynamic warp group proof requires exact C++17 standard dav-c310-vec preprocessing");
    // An unselected retained caller header can also inject an overload into
    // the rewritten namespace. Its declaration does not acquire SDK ownership.
    for(const auto* d:inventory.declarations)if(d->getLocation().isValid()&&!sdk(d)&&
        (d->getNameAsString()=="ascify"||d->getNameAsString()=="ascify_cg"||
         d->getNameAsString()=="asc_shfl"||d->getNameAsString()=="asc_ballot"||
         d->getNameAsString()=="asc_activemask"||
         isa<NamespaceDecl>(d)&&d->getNameAsString()=="cooperative_groups"))
      return fail(d->getLocation(),"caller declarations cannot inject into owned namespaces or backend register dependencies");
    for(const auto* d:inventory.declarations)if(callerExpansion(d->getLocation())&&
        (isa<UsingDecl>(d)||isa<UsingShadowDecl>(d)||isa<TypedefNameDecl>(d)||isa<TypeAliasTemplateDecl>(d))&&newDeclaration(d,context))
      return fail(d->getLocation(),"dynamic group type/function aliases are not admitted");
    for(const auto* d:inventory.declarations)if(callerExpansion(d->getLocation())) {
      if(const auto* usingDirective=dyn_cast<UsingDirectiveDecl>(d))
        if(usingDirective->getNominatedNamespace()->getQualifiedNameAsString()=="cooperative_groups")
          return fail(d->getLocation(),"using namespace cannot publish dynamic group lookup");
      if(const auto* alias=dyn_cast<NamespaceAliasDecl>(d))if(!namespaceAlias(alias))return false;
    }
    for(const auto* v:inventory.values)if(ownExpansion(v->getLocation())&&related(v->getType()))if(!registerVariable(v))return false;
    if(!staticParents.empty()) {
      for(const auto& use:inventory.types)
        if(caller(source.getExpansionLoc(use.location.getBeginLoc()))&&
            (tile32(use.location.getType())||blockType(use.location.getType()))) {
          if(!staticTypeUse(use.location))return fail(use.location.getBeginLoc(),
              "static parent/CTA type is observed outside its exact literal local declaration");
          stats.sourceTypeLocations.insert(use.location.getOpaqueData());
        }
      const auto staticMethod=[&](const Decl* declaration) {
        const auto* method=dyn_cast_or_null<CXXMethodDecl>(declaration);
        const auto t=method?recordType(context,method->getParent(),0):QualType();
        return method&&(tile32(t)||blockType(t));
      };
      for(const auto* reference:inventory.references)
        if(caller(source.getExpansionLoc(reference->getLocation()))&&staticMethod(reference->getDecl()))
          return fail(reference->getLocation(),"static parent/CTA method lookup is outside the factory/free-sync domain");
      for(const auto* member:inventory.members)
        if(caller(source.getExpansionLoc(member->getMemberLoc()))&&staticMethod(member->getMemberDecl()))
          return fail(member->getMemberLoc(),"static parent/CTA member observation is outside the factory/free-sync domain");
    }
    for(const auto& entry:inventory.calls)if(ownExpansion(entry.call->getExprLoc())&&!operation(entry))return false;
    for(const auto* ref:inventory.references)if(callerExpansion(ref->getLocation())) {
      const auto* function=dyn_cast<FunctionDecl>(ref->getDecl());
      if((related(ref->getType())||groupFunction(function))&&!admittedReferences.count(ref)&&!admittedCallees.count(ref))return fail(ref->getLocation(),"dynamic value/function address, copy, assignment or escape");
    }
    for(const auto* member:inventory.members)if(callerExpansion(member->getMemberLoc())) {
      const auto* f=dyn_cast<FunctionDecl>(member->getMemberDecl());
      if(groupFunction(f)&&!admittedCallees.count(member))return fail(member->getMemberLoc(),"dynamic member is not a direct evaluated owned call");
    }
    for(const auto* lookup:inventory.lookups)if(callerExpansion(lookup->getNameLoc())&&newLookup(lookup))
      return fail(lookup->getNameLoc(),"unresolved new-group lookup has no exact ordinary local-call witness");
    for(const auto& use:inventory.types)if(callerExpansion(use.location.getBeginLoc())&&related(use.location.getType())) {
      bool admitted=greaterTypeLocations.count(use.location.getOpaqueData());
      for(const auto& item:groups) {
        const auto root=item.first->getTypeSourceInfo()->getTypeLoc();
        const auto a=source.getDecomposedLoc(root.getBeginLoc()),z=source.getDecomposedLoc(root.getEndLoc());
        const auto x=source.getDecomposedLoc(use.location.getBeginLoc()),y=source.getDecomposedLoc(use.location.getEndLoc());
        if(a.first==z.first&&a.first==x.first&&a.first==y.first&&a.second<=x.second&&y.second<=z.second&&
            literal(use.location.getBeginLoc())&&literal(use.location.getEndLoc()))admitted=true;
      }
      if(!admitted)return fail(use.location.getBeginLoc(),"dynamic type alias, reflection, signature or representation is outside a local value declaration");
      stats.sourceTypeLocations.insert(use.location.getOpaqueData());
    }
    stats.sourceReferences=admittedReferences;
    stats.directCallees=admittedCallees;
    return true;
  }
};
}  // namespace
bool ProveDynamicWarpGroups(ASTContext& c,const std::function<bool(SourceLocation)>& o,
    const std::function<bool(SourceLocation)>& a,
    const std::function<bool(const Decl*)>& s,
    const std::function<bool(llvm::StringRef,SourceLocation)>& m,bool standard,DynamicWarpGroupStats& output,
    SourceLocation& location,std::string& error) {
  DynamicWarpGroupStats staged;if(!Proof(c,o,a,s,m,staged,location,error).run(standard))return false;
  output=std::move(staged);return true;
}

bool ProveDynamicKernelAbiTails(ASTContext& context,const DynamicWarpGroupStats& dynamic,
    const std::function<bool(SourceLocation)>& owned,
    const std::function<bool(SourceLocation)>& caller,
    const std::function<bool(llvm::StringRef,SourceLocation)>& macroFree,
    const std::function<bool(const CallExpr*)>& occupancy,
    std::vector<SourceLocation>& output,SourceLocation& bad,std::string& error) {
  auto& source=context.getSourceManager();std::vector<SourceLocation> staged;
  std::set<const FunctionDecl*> kernels;
  const auto fail=[&](SourceLocation location,llvm::StringRef why) {
    if(error.empty()){bad=location;error=why.str();}return false;
  };
  const auto literal=[&](SourceLocation location) {
    return location.isFileID()&&source.isWrittenInMainFile(location)&&owned(location);
  };
  const auto callerAt=[&](SourceLocation location) {
    return location.isValid()&&caller(source.getExpansionLoc(location));
  };
  const auto tokens=[&](SourceRange range) {
    if(!literal(range.getBegin())||!literal(range.getEnd()))return false;
    const auto file=source.getFileID(range.getBegin());
    if(file!=source.getFileID(range.getEnd()))return false;
    const auto end=Lexer::getLocForEndOfToken(range.getEnd(),0,source,context.getLangOpts());
    if(!end.isFileID())return false;
    bool invalid=false;const auto bytes=source.getBufferData(file,&invalid);
    const unsigned start=source.getFileOffset(range.getBegin()),finish=source.getFileOffset(end);
    if(invalid||start>=finish||finish>bytes.size())return false;
    Lexer lexer(source.getLocForStartOfFile(file),context.getLangOpts(),
                bytes.begin(),bytes.begin()+start,bytes.end());
    for(;;){Token token;lexer.LexFromRawLexer(token);
      if(token.is(tok::eof)||source.getFileOffset(token.getLocation())>=finish)break;
      if(token.needsCleaning()||token.hasUCN()||token.is(tok::hash)||token.is(tok::hashhash))return false;
      if(token.isAnyIdentifier()&&!macroFree(Lexer::getSpelling(token,source,context.getLangOpts()),token.getLocation()))return false;
    }
    return true;
  };
  for(const auto* function:dynamic.localFactoryFunctions) {
    if(!function->hasAttr<CUDAGlobalAttr>()||function->getNumParams()<2)continue;
    const auto n=function->getNumParams();const auto last=function->getParamDecl(n-1)->getType();
    if(!context.hasSameType(last,context.UnsignedIntTy)||context.getTypeSize(last)!=32||
        context.getTypeAlign(last)!=32)continue;
    bool prefix=true;
    for(unsigned i=0;i+1<n;++i) {
      const auto t=function->getParamDecl(i)->getType();const auto* pointer=t->getAs<clang::PointerType>();
      if(!pointer||pointer->getPointeeType()->isFunctionType()||context.getTypeSize(t)!=64||
          context.getTypeAlign(t)!=64||t.isVolatileQualified()||t.getAddressSpace()!=LangAS::Default)prefix=false;
    }
    // Other signatures retain the previous domain; this feature neither
    // guesses their padding nor grants them an ABI-completion witness.
    if(!prefix)continue;
    const FunctionDecl* definition=nullptr;
    if(isa<CXXMethodDecl>(function)||function->getTemplatedKind()!=FunctionDecl::TK_NonTemplate||
        !function->getDeclContext()->getRedeclContext()->isTranslationUnit()||
        !function->hasBody(definition)||definition!=function||function->getPreviousDecl()||
        function->getMostRecentDecl()!=function||function->isVariadic()||
        !context.hasSameType(function->getReturnType(),context.VoidTy)||!literal(function->getLocation())||
        !tokens(function->getNameInfo().getSourceRange()))
      return fail(function->getLocation(),"dynamic kernel ABI padding requires one literal main-file ordinary global definition");
    for(const auto* attr:function->attrs())if(!attr->isImplicit()&&
        !isa<CUDAGlobalAttr>(attr)&&!isa<CUDALaunchBoundsAttr>(attr))
      return fail(function->getLocation(),"dynamic kernel ABI padding has an unproved function attribute");
    const auto* type=function->getType()->getAs<FunctionProtoType>();
    if(!type||type->getCallConv()!=CC_C)return fail(function->getLocation(),"dynamic kernel ABI padding requires the ordinary C calling convention");
    for(const auto* parameter:function->parameters())
      if(parameter->hasAttrs()||parameter->hasDefaultArg()||!tokens(parameter->getSourceRange()))
        return fail(parameter->getLocation(),"dynamic kernel ABI padding requires literal nondefault parameter declarations without attributes");
    const auto* info=function->getTypeSourceInfo();
    const auto location=info?info->getTypeLoc().getAs<FunctionProtoTypeLoc>():FunctionProtoTypeLoc();
    const auto close=location?location.getRParenLoc():SourceLocation();Token token;
    if(!literal(close)||Lexer::getRawToken(close,token,source,context.getLangOpts(),true)||
        !token.is(tok::r_paren)||!macroFree("unsigned",close)||!macroFree("int",close))
      return fail(function->getLocation(),"dynamic kernel ABI padding has no literal macro-free parameter-list terminator");
    kernels.insert(function->getCanonicalDecl());staged.push_back(close);
  }
  if(kernels.empty()){output.clear();return true;}
  Inventory inventory(context.BuiltinInfo);inventory.TraverseDecl(context.getTranslationUnitDecl());
  const auto selected=[&](const NamedDecl* declaration) {
    if(const auto* shadow=dyn_cast_or_null<UsingShadowDecl>(declaration))declaration=shadow->getTargetDecl();
    const auto* f=dyn_cast_or_null<FunctionDecl>(declaration);
    return f&&kernels.count(f->getCanonicalDecl());
  };
  std::set<std::string> kernelNames;
  for(const auto* function:kernels)kernelNames.insert(function->getNameAsString());
  const auto sameName=[&](const NamedDecl* declaration) {
    return declaration&&declaration->getIdentifier()&&
        kernelNames.count(declaration->getNameAsString());
  };
  // A new defaulted fifth argument must never create another overload or
  // retarget an existing five-argument call. Keep the spelling unique across
  // all caller declarations, not just the originally selected reference set.
  for(const auto* declaration:inventory.declarations)
    if(callerAt(declaration->getLocation())&&sameName(declaration)&&!selected(declaration))
      return fail(declaration->getLocation(),"dynamic kernel ABI padding rejects caller same-name declarations or overloads");
  for(const auto* reference:inventory.references)
    if(callerAt(reference->getLocation())&&sameName(reference->getDecl())&&!selected(reference->getDecl()))
      return fail(reference->getLocation(),"dynamic kernel ABI padding rejects same-name nonkernel references");
  for(const auto* member:inventory.members)
    if(callerAt(member->getMemberLoc())&&sameName(member->getMemberDecl())&&!selected(member->getMemberDecl()))
      return fail(member->getMemberLoc(),"dynamic kernel ABI padding rejects same-name member observations");
  const auto physicalArgument=[&](SourceLocation location) {
    while(location.isMacroID()) {
      if(!source.isMacroArgExpansion(location))return SourceLocation();
      location=source.getImmediateSpellingLoc(location);
    }
    return location;
  };
  const auto directReference=[&](const Expr* expression,const DeclRefExpr* reference,bool macroArgument) {
    unsigned decays=0;
    while(expression&&expression!=reference) {
      if(const auto* parens=dyn_cast<ParenExpr>(expression))expression=parens->getSubExpr();
      else if(const auto* cast=dyn_cast<ImplicitCastExpr>(expression)) {
        const auto* pointer=cast->getType()->getAs<clang::PointerType>();
        if(cast->getCastKind()!=CK_FunctionToPointerDecay||++decays!=1||!pointer||
            !context.hasSameType(pointer->getPointeeType(),cast->getSubExpr()->getType()))return false;
        expression=cast->getSubExpr();
      }else return false;
    }
    if(expression!=reference)return false;
    const auto first=macroArgument?physicalArgument(reference->getBeginLoc()):reference->getBeginLoc();
    const auto last=macroArgument?physicalArgument(reference->getEndLoc()):reference->getEndLoc();
    return tokens({first,last});
  };
  std::set<const FunctionDecl*> launched;
  for(const auto* reference:inventory.references)if(selected(reference->getDecl())) {
    bool admitted=false;
    for(const auto& entry:inventory.calls) {
      if(entry.observation||!entry.owner||isa<CXXMethodDecl>(entry.owner)||entry.owner->getTemplatedKind()!=FunctionDecl::TK_NonTemplate||
          entry.owner->hasAttr<CUDAGlobalAttr>()||entry.owner->hasAttr<CUDADeviceAttr>())continue;
      if(const auto* launch=dyn_cast<CUDAKernelCallExpr>(entry.call)) {
        if(launch->getDirectCallee()&&selected(launch->getDirectCallee())&&
            directReference(launch->getCallee(),reference,false)) {
          admitted=true;launched.insert(launch->getDirectCallee()->getCanonicalDecl());
        }
      }else if(entry.call->getNumArgs()>2&&occupancy(entry.call)&&
          entry.call->getDirectCallee()&&entry.call->getDirectCallee()->getNameAsString()=="cudaOccupancyMaxPotentialBlockSize"&&
          directReference(entry.call->getArg(2),reference,true))admitted=true;
      else if(entry.call->getNumArgs()>2&&occupancy(entry.call)&&
          entry.call->getDirectCallee()&&entry.call->getDirectCallee()->getNameAsString()=="cudaOccupancyMaxActiveBlocksPerMultiprocessor") {
        for(const auto* argument:entry.call->arguments())
          if(directReference(argument,reference,true))admitted=true;
      }else if(entry.call->getNumArgs()==6&&
          entry.call->getDirectCallee()&&
          entry.call->getDirectCallee()->getNameAsString()=="cudaLaunchCooperativeKernel") {
        // The proven cooperative-launch projection rewrites the whole call
        // into a direct <<<>>> launch; its kernel operand is (void*)kernel
        // or (void*)&kernel around the exact reference.
        const Expr* operand=entry.call->getArg(0)->IgnoreParenImpCasts();
        if(const auto*cast=dyn_cast<CStyleCastExpr>(operand))
          operand=cast->getSubExpr()->IgnoreParenImpCasts();
        if(const auto*address=dyn_cast<UnaryOperator>(operand)) {
          if(address->getOpcode()!=UO_AddrOf)operand=nullptr;
          else operand=address->getSubExpr()->IgnoreParenImpCasts();
        }
        if(operand==reference)admitted=true;
      }
    }
    if(!admitted) {
      return fail(reference->getLocation(),
          "dynamic kernel ABI padding rejects kernel address escape, reflection, alias or unproved call");
    }
  }
  for(const auto* function:kernels)if(!launched.count(function))
    return fail(function->getLocation(),"dynamic kernel ABI padding requires its direct ordinary host launch in the same root");
  for(const auto& observer:inventory.signatureObservers)
    if(observer.first&&kernels.count(observer.first->getCanonicalDecl()))
      return fail(observer.second->getExprLoc(),"dynamic kernel ABI padding rejects function/source signature observations");
  for(const auto* assembly:inventory.assemblies)if(callerAt(assembly->getBeginLoc()))
    return fail(assembly->getBeginLoc(),"dynamic kernel ABI padding rejects caller assembly symbol observations");
  for(const auto* assembly:inventory.fileAssemblies)if(callerAt(assembly->getBeginLoc()))
    return fail(assembly->getBeginLoc(),"dynamic kernel ABI padding rejects caller file-scope assembly symbol observations");
  for(const auto* declaration:inventory.declarations)if(callerAt(declaration->getLocation())) {
    if(const auto* imported=dyn_cast<UsingDecl>(declaration))
      for(const auto* shadow:imported->shadows())if(selected(shadow))
        return fail(declaration->getLocation(),"dynamic kernel ABI padding rejects kernel using aliases");
    if((isa<UsingShadowDecl>(declaration)&&selected(declaration))||
        declaration->hasAttr<AliasAttr>()||declaration->hasAttr<IFuncAttr>()||
        declaration->hasAttr<AsmLabelAttr>()||declaration->hasAttr<WeakRefAttr>())
      return fail(declaration->getLocation(),"dynamic kernel ABI padding rejects caller symbol aliases");
  }
  for(const auto* lookup:inventory.lookups)if(callerAt(lookup->getNameLoc())) {
    if(kernelNames.count(lookup->getName().getAsString()))
      return fail(lookup->getNameLoc(),"dynamic kernel ABI padding rejects unresolved same-name kernel lookup");
    for(const auto* declaration:lookup->decls())if(selected(declaration))
      return fail(lookup->getNameLoc(),"dynamic kernel ABI padding rejects unresolved kernel lookup");
  }
  for(const auto& item:inventory.templateDeclarations)if(selected(item.first))
    return fail(item.second.isValid()?item.second:(*kernels.begin())->getLocation(),
        "dynamic kernel ABI padding rejects kernel non-type template observations");
  output=std::move(staged);return true;
}
}  // namespace ascify
