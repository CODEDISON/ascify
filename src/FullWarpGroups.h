#pragma once

#include <functional>
#include <string>
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"

namespace clang { class ASTContext; class Decl; class FunctionDecl; }
namespace ascify {
struct FullWarpGroupStats {
  unsigned kernels = 0, helpers = 0, syncs = 0;
  bool used() const { return kernels != 0; }
};

// Mandatory source-publication proof for the new static16/erased-group domain.
// It does not admit a partial-active/named-mask barrier. Existing tile32-only
// inputs are untouched. Stats publish only after the entire closed TU passes.
bool ProveFullWarpGroups(
    clang::ASTContext& context,
    const std::function<bool(clang::SourceLocation)>& ownedSource,
    const std::function<bool(const clang::Decl*)>& ownedCgDecl,
    const std::function<bool(const clang::Decl*)>& ownedSdkDecl,
    const std::function<bool(const clang::FunctionDecl*)>& ownedDevicePrintf,
    const std::function<bool(llvm::StringRef,clang::SourceLocation)>& macroFree,
    const std::function<bool(clang::FileID)>& trustedNativeInput,
    bool standardDavC310Vec, FullWarpGroupStats& stats,
    clang::SourceLocation& errorLocation, std::string& error);
} // namespace ascify
