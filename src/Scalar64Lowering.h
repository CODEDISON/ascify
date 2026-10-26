#pragma once

#include <functional>
#include <set>
#include <string>
#include <vector>
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
class ASTContext;
class Expr;
class FunctionDecl;
class TemplateTypeParmDecl;
}

namespace ascify {
struct Scalar64Edit {
  clang::SourceLocation location;
  unsigned length = 0;
  std::string text;
};
struct Scalar64Plan {
  std::vector<Scalar64Edit> edits;
  std::set<const clang::Expr*> scalarExpressions;
  std::set<const clang::TemplateTypeParmDecl*> scalarParameters;
  // The outer FunctionProto TypeLoc of a proved declaration only. Its return
  // and parameter TypeLocs are still traversed; no function-type/trait domain
  // or address-taking exception is granted.
  std::set<const void*> scalarSignatureTypeLocs;
  unsigned additions = 0, assignments = 0, transports = 0, guards = 0;
  bool used() const { return additions || assignments || transports; }
};

// These callbacks reuse the converter's physical input/SDK/profile proofs.
// No spelling, basename or sizeof-only rule can authenticate a callee.
bool ProveScalar64Lowering(
    clang::ASTContext& context,
    const std::function<bool(clang::SourceLocation)>& ownedInput,
    const std::function<bool(clang::SourceLocation)>& trustedSdk,
    const std::function<bool(const clang::FunctionDecl*)>& trustedShuffle,
    const std::function<bool(llvm::StringRef, clang::SourceLocation)>& macroFree,
    bool standardPreprocessing, Scalar64Plan& plan,
    clang::SourceLocation& errorLocation, std::string& error);

// Only T / pointers / references / arrays of the exact, guarded builtin
// parameter are proven. Records, decltype, traits and function types are not.
bool ProvesScalar64Type(const Scalar64Plan& plan, clang::QualType type);
} // namespace ascify
