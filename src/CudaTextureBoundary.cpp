#include "CudaTextureBoundary.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include <algorithm>
#include <vector>

namespace ascify {
namespace {
constexpr const char *Names[] = {
  "cudaMallocPitch", "cudaMallocArray", "cudaMalloc3DArray", "cudaFreeArray",
  "cudaMemcpy2D", "cudaMemset2D", "cudaMemcpyToArray", "cudaMemcpy3D",
  "cudaCreateTextureObject", "cudaDestroyTextureObject", "cudaCreateChannelDesc",
  "make_cudaExtent", "make_cudaPos", "make_cudaPitchedPtr", "tex2D", "tex2DLayered",
  "cudaArray", "cudaArray_t", "cudaArray_const_t", "cudaTextureObject_t",
  "cudaChannelFormatDesc", "cudaChannelFormatKind", "cudaResourceDesc",
  "cudaResourceType", "cudaTextureDesc", "cudaTextureAddressMode",
  "cudaTextureFilterMode", "cudaTextureReadMode", "cudaPitchedPtr", "cudaExtent",
  "cudaPos", "cudaMemcpy3DParms", "cudaArrayLayered", "cudaResourceViewDesc",
  "cudaChannelFormatKindSigned", "cudaChannelFormatKindUnsigned",
  "cudaChannelFormatKindFloat", "cudaChannelFormatKindNone",
  "cudaResourceTypeArray", "cudaResourceTypeMipmappedArray", "cudaResourceTypeLinear",
  "cudaResourceTypePitch2D", "cudaAddressModeWrap", "cudaAddressModeClamp",
  "cudaAddressModeMirror", "cudaAddressModeBorder", "cudaFilterModePoint",
  "cudaFilterModeLinear", "cudaReadModeElementType", "cudaReadModeNormalizedFloat"
};
bool sdkDeclaration(const clang::Decl *declaration, clang::SourceManager &source,
                    const std::set<unsigned> &files) {
  if (declaration == nullptr) return false;
  const auto location = source.getSpellingLoc(declaration->getLocation());
  return location.isValid() && files.count(source.getFileID(location).getHashValue());
}
const clang::Expr *strip(const clang::Expr *expression) {
  return expression == nullptr ? nullptr : expression->IgnoreParenImpCasts();
}
}

bool IsCudaTextureName(llvm::StringRef name) {
  return std::any_of(std::begin(Names),std::end(Names),
                     [&](const char *item) {return name == item;});
}
bool IsCudaTextureRecord(const clang::RecordDecl *record) {
  if (record == nullptr) return false;
  // Anonymous resource-union alternatives are used only through the named
  // descriptor's published fields. Reflection still calls inspect(...,false).
  while (record->getName().empty()) {
    record = llvm::dyn_cast<clang::RecordDecl>(record->getDeclContext());
    if (record == nullptr) return false;
  }
  if (!record->getDeclContext()->getRedeclContext()->isTranslationUnit()) return false;
  const auto name=record->getName();
  return name=="cudaArray" || name=="cudaChannelFormatDesc" ||
      name=="cudaResourceDesc" || name=="cudaTextureDesc" ||
      name=="cudaPitchedPtr" || name=="cudaExtent" || name=="cudaPos" ||
      name=="cudaMemcpy3DParms" || name=="cudaResourceViewDesc";
}

bool IsCudaTextureZeroInitializer(const clang::InitListExpr *expression,
    clang::ASTContext &context,bool permitNullFirstPointer) {
  if (expression==nullptr || expression->getBeginLoc().isMacroID() ||
      expression->getEndLoc().isMacroID()) return false;
  bool invalid=false;
  const auto text=clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(expression->getSourceRange()),
      context.getSourceManager(),context.getLangOpts(),&invalid);
  if (invalid || text.empty()) return false;
  // Raw Lexer requires a NUL sentinel at BufferEnd. A source substring ends
  // before the original semicolon and does not provide that sentinel.
  const std::string owned=text.str();
  clang::Lexer lexer(clang::SourceLocation(),context.getLangOpts(),
                     owned.data(),owned.data(),owned.data()+owned.size());
  std::vector<clang::tok::TokenKind> kinds;
  clang::Token token;
  while (true) {
    lexer.LexFromRawLexer(token);
    // A final valid token can exhaust the raw buffer. Its kind, rather than
    // LexFromRawLexer's boolean, determines whether it must be inspected.
    if (token.is(clang::tok::eof)) break;
    if (token.is(clang::tok::numeric_constant) &&
        llvm::StringRef(token.getLiteralData(),token.getLength())!="0") return false;
    kinds.push_back(token.getKind());
  }
  return kinds==std::vector<clang::tok::TokenKind>{clang::tok::l_brace,clang::tok::r_brace} ||
      (permitNullFirstPointer && kinds==std::vector<clang::tok::TokenKind>{
          clang::tok::l_brace,clang::tok::numeric_constant,clang::tok::r_brace});
}

bool IsCudaTextureZeroSize(const clang::UnaryExprOrTypeTraitExpr *expression,
    clang::ASTContext &context, const std::set<unsigned> &sdkFiles,
    const std::set<unsigned> &nativeFiles) {
  if (expression == nullptr || expression->getKind()!=clang::UETT_SizeOf ||
      expression->getExprLoc().isMacroID()) return false;
  auto &source=context.getSourceManager();
  const auto type=expression->getTypeOfArgument();
  const auto *record=type->getAs<clang::RecordType>();
  if (record==nullptr || !IsCudaTextureRecord(record->getDecl()) ||
      !sdkDeclaration(record->getDecl(),source,sdkFiles)) return false;
  // This proves one exact zeroing use, not an observable SDK byte count.
  const clang::Expr *child=expression;
  const clang::CallExpr *call=nullptr;
  for (unsigned depth=0;depth<8;++depth) {
    const auto parents=context.getParents(*child);
    if (parents.size()!=1) return false;
    if ((call=parents[0].get<clang::CallExpr>())) break;
    const auto *parent=parents[0].get<clang::Expr>();
    if (parent==nullptr || (!llvm::isa<clang::ParenExpr>(parent) &&
                           !llvm::isa<clang::ImplicitCastExpr>(parent))) return false;
    child=parent;
  }
  if (call==nullptr || call->getNumArgs()!=3 || strip(call->getArg(2))!=expression)
    return false;
  const auto *callee=call->getDirectCallee();
  if (callee==nullptr || callee->getBuiltinID()==0 || callee->getName()!="memset") return false;
  for (const auto *declaration:callee->redecls())
    if (!sdkDeclaration(declaration,source,nativeFiles)) return false;
  const auto *zero=llvm::dyn_cast<clang::IntegerLiteral>(strip(call->getArg(1)));
  const auto *address=llvm::dyn_cast<clang::UnaryOperator>(strip(call->getArg(0)));
  if (zero==nullptr || !zero->getValue().isZero() || address==nullptr ||
      address->getOpcode()!=clang::UO_AddrOf || address->getExprLoc().isMacroID()) return false;
  const auto *variable=llvm::dyn_cast<clang::DeclRefExpr>(strip(address->getSubExpr()));
  return variable!=nullptr && llvm::isa<clang::VarDecl>(variable->getDecl()) &&
      !variable->getType().isConstQualified() && !variable->getType().isVolatileQualified() &&
      context.hasSameUnqualifiedType(type,variable->getType());
}

bool ValidateCudaTextureBoundary(clang::ASTContext &context,
    const std::set<unsigned> &sdkFiles,std::set<unsigned> &literalOffsets,
    const std::function<bool(clang::SourceLocation)> &admitMacroUse,
    clang::SourceLocation &errorLocation,std::string &error) {
  class Audit : public clang::RecursiveASTVisitor<Audit> {
    clang::ASTContext &context;
    clang::SourceManager &source;
    const std::set<unsigned> &sdk;
    std::set<unsigned> &offsets;
    const std::function<bool(clang::SourceLocation)> &admitMacroUse;
    clang::SourceLocation &badLocation;
    std::string &reason;
    bool fail(clang::SourceLocation location,llvm::StringRef message) {
      badLocation=location;reason=message.str();return false;
    }
    bool caller(clang::SourceLocation location) const {
      return location.isValid() && source.isWrittenInMainFile(source.getExpansionLoc(location));
    }
    bool mark(clang::SourceLocation location,const clang::NamedDecl *declaration) {
      if (!caller(location)) return true;
      if (!sdkDeclaration(declaration,source,sdk) ||
          !declaration->getDeclContext()->getRedeclContext()->isTranslationUnit())
        return fail(location,"texture name is not owned by the admitted CUDA SDK");
      if (!admitMacroUse(location))
        return fail(location,"texture reference has an unproved enclosing macro");
      if (location.isMacroID()) {
        // The callback above binds the exact frozen helper candidate and all
        // enclosing observations. Publication also requires its committed
        // transaction; a same-named user macro is never sufficient.
        const auto expansion=source.getExpansionLoc(location);
        clang::Token outer;
        if (!source.isMacroArgExpansion(location) ||
            clang::Lexer::getRawToken(expansion,outer,source,context.getLangOpts(),true) ||
            clang::Lexer::getSpelling(outer,source,context.getLangOpts())!="checkCudaErrors")
          return fail(location,"macro-generated texture reference");
        location=source.getSpellingLoc(location);
        if (!source.isWrittenInMainFile(location))
          return fail(location,"texture macro argument is not a main-file token");
      }
      clang::Token token;
      if (clang::Lexer::getRawToken(location,token,source,context.getLangOpts(),true) ||
          clang::Lexer::getSpelling(token,source,context.getLangOpts())!=declaration->getName())
        return fail(location,"texture reference has no exact literal token");
      offsets.insert(source.getFileOffset(location));return true;
    }
  public:
    Audit(clang::ASTContext &c,const std::set<unsigned> &s,std::set<unsigned> &o,
          const std::function<bool(clang::SourceLocation)> &a,
          clang::SourceLocation &l,std::string &r)
        :context(c),source(c.getSourceManager()),sdk(s),offsets(o),admitMacroUse(a),badLocation(l),reason(r) {}
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool VisitNamedDecl(clang::NamedDecl *declaration) {
      if (declaration->getIdentifier()!=nullptr && IsCudaTextureName(declaration->getName()) &&
          caller(declaration->getLocation()) && !sdkDeclaration(declaration,source,sdk))
        return fail(declaration->getLocation(),"input declaration shadows a mapped texture name");
      return true;
    }
    bool VisitDeclRefExpr(clang::DeclRefExpr *reference) {
      const auto *declaration=reference->getDecl();
      if (declaration->getIdentifier()==nullptr || !IsCudaTextureName(declaration->getName()) ||
          !caller(reference->getExprLoc())) return true;
      if (reference->hasQualifier())
        return fail(reference->getExprLoc(),"qualified texture reference is outside the literal adapter domain");
      if (const auto *function=llvm::dyn_cast<clang::FunctionDecl>(declaration)) {
        for (const auto *redecl:function->redecls())
          if (!sdkDeclaration(redecl,source,sdk))
            return fail(reference->getExprLoc(),"texture function has an input redeclaration");
        const clang::Expr *child=reference;
        const clang::CallExpr *call=nullptr;
        for (unsigned depth=0;depth<8;++depth) {
          const auto parents=context.getParents(*child);
          if (parents.size()!=1) break;
          if ((call=parents[0].get<clang::CallExpr>())) break;
          const auto *parent=parents[0].get<clang::Expr>();
          if (parent==nullptr || (!llvm::isa<clang::ParenExpr>(parent) &&
                                 !llvm::isa<clang::ImplicitCastExpr>(parent))) break;
          child=parent;
        }
        if (call==nullptr || call->getDirectCallee()!=function || strip(call->getCallee())!=reference)
          return fail(reference->getExprLoc(),"texture function is not a direct call");
        if (const auto *arguments=function->getTemplateSpecializationArgs()) {
          if (arguments->size()!=1 || arguments->get(0).getKind()!=clang::TemplateArgument::Type ||
              (!context.hasSameType(arguments->get(0).getAsType(),context.FloatTy) &&
               !context.hasSameType(arguments->get(0).getAsType(),context.UnsignedIntTy)))
            return fail(reference->getExprLoc(),"texture template requires exact float32 or unsigned int32");
        }
        if ((function->getName()=="tex2D" && call->getNumArgs()!=3) ||
            (function->getName()=="tex2DLayered" && call->getNumArgs()!=4))
          return fail(reference->getExprLoc(),"unsupported texture fetch overload");
      }
      return mark(reference->getLocation(),declaration);
    }
    bool VisitTagTypeLoc(clang::TagTypeLoc type) {
      const auto *declaration=type.getDecl();
      return declaration->getIdentifier()==nullptr || !IsCudaTextureName(declaration->getName()) ||
          mark(type.getNameLoc(),declaration);
    }
    bool VisitTypedefTypeLoc(clang::TypedefTypeLoc type) {
      const auto *declaration=type.getTypePtr()->getDecl();
      return !IsCudaTextureName(declaration->getName()) || mark(type.getNameLoc(),declaration);
    }
    bool VisitUnresolvedLookupExpr(clang::UnresolvedLookupExpr *expression) {
      const auto *identifier=expression->getName().getAsIdentifierInfo();
      return identifier==nullptr || !IsCudaTextureName(identifier->getName()) ||
          !caller(expression->getExprLoc()) || fail(expression->getExprLoc(),"dependent texture lookup");
    }
  } audit(context,sdkFiles,literalOffsets,admitMacroUse,errorLocation,error);
  return audit.TraverseDecl(context.getTranslationUnitDecl());
}
}
