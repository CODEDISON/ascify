#ifndef ASCIFY_CUDA_TEXTURE_BOUNDARY_H
#define ASCIFY_CUDA_TEXTURE_BOUNDARY_H
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/StringRef.h"
#include <set>
#include <string>
#include <functional>
namespace ascify {
bool IsCudaTextureName(llvm::StringRef name);
bool IsCudaTextureRecord(const clang::RecordDecl *record);
bool IsCudaTextureZeroInitializer(const clang::InitListExpr *expression,
    clang::ASTContext &context, bool permitNullFirstPointer);
bool IsCudaTextureZeroSize(const clang::UnaryExprOrTypeTraitExpr *expression,
    clang::ASTContext &context, const std::set<unsigned> &sdkFiles,
    const std::set<unsigned> &nativeFiles);
bool ValidateCudaTextureBoundary(clang::ASTContext &context,
    const std::set<unsigned> &sdkFiles, std::set<unsigned> &literalOffsets,
    const std::function<bool(clang::SourceLocation)> &admitMacroUse,
    clang::SourceLocation &errorLocation, std::string &error);
}
#endif
