#include "DeviceMemoryLowering.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

namespace ascify {
namespace {

bool trustedLocation(clang::SourceLocation location,
                     clang::SourceManager &sourceManager,
                     const std::set<unsigned> &trustedFiles) {
  if (location.isInvalid())
    return false;
  const clang::FileID file = sourceManager.getFileID(
      sourceManager.getExpansionLoc(location));
  return file.isValid() && trustedFiles.count(file.getHashValue()) != 0;
}

bool memcpySignature(const clang::FunctionDecl *callee,
                     clang::ASTContext &context) {
  if (callee == nullptr || callee->isVariadic() || callee->getNumParams() != 3 ||
      callee->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate)
    return false;
  if (!context.hasSameType(callee->getReturnType(), context.VoidPtrTy) ||
      !context.hasSameType(callee->getParamDecl(0)->getType().getUnqualifiedType(),
                           context.VoidPtrTy) ||
      !context.hasSameType(callee->getParamDecl(1)->getType().getUnqualifiedType(),
                           context.getPointerType(context.VoidTy.withConst())) ||
      !context.hasSameType(callee->getParamDecl(2)->getType(), context.getSizeType()))
    return false;
  return true;
}

bool trustedFunctionDeclarations(const clang::FunctionDecl *callee,
                                clang::ASTContext &context,
                                const std::set<unsigned> &trustedFiles) {
  auto &sourceManager = context.getSourceManager();
  for (const auto *declaration : callee->getCanonicalDecl()->redecls()) {
    if (!trustedLocation(declaration->getLocation(), sourceManager, trustedFiles) &&
        !(declaration->getLocation().isInvalid() && declaration->getBuiltinID() != 0))
      return false;
    for (const auto *attribute : declaration->attrs()) {
      if (attribute->getLocation().isInvalid() && attribute->isImplicit())
        continue;
      if (!trustedLocation(attribute->getLocation(), sourceManager, trustedFiles))
        return false;
    }
  }
  return true;
}

bool trustedMemcpy(const clang::FunctionDecl *callee,
                   clang::ASTContext &context,
                   const std::set<unsigned> &trustedFiles) {
  if (!memcpySignature(callee, context) ||
      callee->getQualifiedNameAsString() != "memcpy" ||
      !trustedFunctionDeclarations(callee, context, trustedFiles))
    return false;
  if (callee->getBuiltinID() == clang::Builtin::BImemcpy)
    return true;

  // Clang's CUDA resource header supplies a device-only memcpy wrapper, not
  // a BImemcpy declaration. Prove its actual forwarding body rather than
  // admitting a CUDA-like name or an arbitrary user/system wrapper.
  const clang::FunctionDecl *definition = callee->getDefinition();
  if (definition == nullptr || !definition->hasAttr<clang::CUDADeviceAttr>() ||
      definition->hasAttr<clang::CUDAHostAttr>() ||
      definition->hasAttr<clang::CUDAGlobalAttr>() ||
      definition->getStorageClass() != clang::SC_Static ||
      !definition->isInlineSpecified())
    return false;
  const auto *body = llvm::dyn_cast<clang::CompoundStmt>(definition->getBody());
  if (body == nullptr || body->size() != 1)
    return false;
  const auto *returned = llvm::dyn_cast<clang::ReturnStmt>(*body->body_begin());
  const auto *forwarded = returned == nullptr || returned->getRetValue() == nullptr
                              ? nullptr
                              : llvm::dyn_cast<clang::CallExpr>(
                                    returned->getRetValue()->IgnoreParenImpCasts());
  const clang::FunctionDecl *builtin =
      forwarded == nullptr ? nullptr : forwarded->getDirectCallee();
  if (forwarded == nullptr || forwarded->getNumArgs() != 3 ||
      !memcpySignature(builtin, context) ||
      builtin->getBuiltinID() != clang::Builtin::BI__builtin_memcpy ||
      builtin->getQualifiedNameAsString() != "__builtin_memcpy" ||
      !trustedFunctionDeclarations(builtin, context, trustedFiles))
    return false;
  const auto *calleeReference = llvm::dyn_cast<clang::DeclRefExpr>(
      forwarded->getCallee()->IgnoreParenImpCasts());
  if (calleeReference == nullptr || calleeReference->hasQualifier())
    return false;
  for (unsigned index = 0; index != 3; ++index) {
    const auto *argument = llvm::dyn_cast<clang::DeclRefExpr>(
        forwarded->getArg(index)->IgnoreParenImpCasts());
    if (argument == nullptr || argument->getDecl() != definition->getParamDecl(index))
      return false;
  }
  class TrustedBody : public clang::RecursiveASTVisitor<TrustedBody> {
  public:
    TrustedBody(clang::SourceManager &sourceManager,
                const std::set<unsigned> &trustedFiles)
        : sourceManager(sourceManager), trustedFiles(trustedFiles) {}
    bool VisitStmt(clang::Stmt *statement) {
      for (clang::SourceLocation location :
           {statement->getBeginLoc(), statement->getEndLoc()}) {
        if (location.isInvalid() || location.isMacroID() ||
            !trustedLocation(location, sourceManager, trustedFiles))
          return false;
      }
      return true;
    }
  private:
    clang::SourceManager &sourceManager;
    const std::set<unsigned> &trustedFiles;
  } proof(context.getSourceManager(), trustedFiles);
  return proof.TraverseStmt(const_cast<clang::CompoundStmt *>(body));
}

const clang::FunctionDecl *enclosingFunction(const clang::Stmt *statement,
                                             clang::ASTContext &context) {
  clang::DynTypedNode current = clang::DynTypedNode::create(*statement);
  for (unsigned depth = 0; depth != 64; ++depth) {
    const auto parents = context.getParents(current);
    if (parents.size() != 1)
      return nullptr;
    if (const auto *function = parents[0].get<clang::FunctionDecl>())
      return function;
    if (parents[0].get<clang::LambdaExpr>() != nullptr)
      return nullptr;
    current = parents[0];
  }
  return nullptr;
}

class NoMacroExpression : public clang::RecursiveASTVisitor<NoMacroExpression> {
public:
  bool VisitStmt(clang::Stmt *statement) {
    return statement->getBeginLoc().isValid() && statement->getEndLoc().isValid() &&
           !statement->getBeginLoc().isMacroID() && !statement->getEndLoc().isMacroID();
  }
};

bool plainTrivialObject(clang::QualType type, clang::ASTContext &context,
                        unsigned depth = 0) {
  if (depth > 16 || type.isNull() || type.isVolatileQualified() ||
      type->isReferenceType() || type->isIncompleteType() ||
      type->isDependentType() || !type.isTriviallyCopyableType(context))
    return false;
  if (const auto *array = context.getAsConstantArrayType(type))
    return plainTrivialObject(array->getElementType(), context, depth + 1);
  if (const auto *record = type->getAsCXXRecordDecl()) {
    if (!record->hasDefinition() || record->getNumBases() != 0 ||
        record->isUnion())
      return false;
    for (const auto *field : record->fields()) {
      if (!plainTrivialObject(field->getType(), context, depth + 1))
        return false;
    }
  }
  return true;
}

const clang::VarDecl *privateObject(const clang::Expr *expression,
                                    const clang::FunctionDecl *function,
                                    clang::ASTContext &context) {
  expression = expression->IgnoreParenImpCasts();
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expression);
  const auto *variable = reference == nullptr ? nullptr :
      llvm::dyn_cast<clang::VarDecl>(reference->getDecl());
  if (variable == nullptr || llvm::isa<clang::ParmVarDecl>(variable) ||
      variable->getDeclContext() != function || !variable->hasLocalStorage() ||
      variable->getStorageClass() != clang::SC_None ||
      variable->hasAttr<clang::CUDASharedAttr>() ||
      variable->hasAttr<clang::CUDADeviceAttr>() ||
      variable->hasAttr<clang::CUDAConstantAttr>() ||
      variable->getLocation().isMacroID() ||
      !plainTrivialObject(variable->getType(), context) ||
      context.getTypeSize(variable->getType()) != 64 * 8)
    return nullptr;
  return variable;
}

std::string sourceText(const clang::Expr *expression, clang::ASTContext &context) {
  bool invalid = false;
  const auto text = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(expression->getSourceRange()),
      context.getSourceManager(), context.getLangOpts(), &invalid);
  return invalid ? std::string() : text.str();
}

} // namespace

bool planDeviceMemoryCall(const clang::CallExpr *call,
                          clang::ASTContext &context,
                          const std::set<unsigned> &trustedSystemFileIds,
                          DeviceMemoryRewrite &result) {
  if (call == nullptr || call->getNumArgs() != 3 ||
      !trustedMemcpy(call->getDirectCallee(), context, trustedSystemFileIds))
    return false;
  const auto *function = enclosingFunction(call, context);
  if (function == nullptr ||
      (!function->hasAttr<clang::CUDADeviceAttr>() &&
       !function->hasAttr<clang::CUDAGlobalAttr>()) ||
      function->hasAttr<clang::CUDAHostAttr>() || function->isDependentContext() ||
      function->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate)
    return false;
  NoMacroExpression macroProof;
  if (!macroProof.TraverseStmt(const_cast<clang::CallExpr *>(call)))
    return false;
  const auto *calleeReference = llvm::dyn_cast<clang::DeclRefExpr>(
      call->getCallee()->IgnoreParenImpCasts());
  if (calleeReference == nullptr || calleeReference->hasQualifier())
    return false;
  const auto *address = llvm::dyn_cast<clang::UnaryOperator>(
      call->getArg(0)->IgnoreParenImpCasts());
  if (address == nullptr || address->getOpcode() != clang::UO_AddrOf)
    return false;
  const auto *destination = privateObject(address->getSubExpr(), function, context);
  const auto *source = privateObject(call->getArg(1), function, context);
  if (destination == nullptr || source == nullptr || destination == source ||
      destination->getType().isConstQualified() ||
      !source->getType()->isConstantArrayType())
    return false;
  const auto *size = llvm::dyn_cast<clang::UnaryExprOrTypeTraitExpr>(
      call->getArg(2)->IgnoreParenImpCasts());
  if (size == nullptr || size->getKind() != clang::UETT_SizeOf || size->isArgumentType() ||
      privateObject(size->getArgumentExpr(), function, context) != destination)
    return false;
  const std::string destinationText = sourceText(address, context);
  const std::string sourceString = sourceText(call->getArg(1)->IgnoreParenImpCasts(), context);
  if (destinationText.empty() || sourceString.empty())
    return false;
  result.range = call->getSourceRange();
  result.replacement = "::ascify::device_memcpy_private<64>((" + destinationText +
                       "), (" + sourceString + "))";
  result.apiName = "AST-proven private 64-byte memcpy";
  return true;
}

} // namespace ascify
