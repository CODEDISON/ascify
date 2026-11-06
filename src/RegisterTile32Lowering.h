#pragma once

#include <functional>
#include <string>
#include <vector>
#include "clang/Basic/SourceLocation.h"

namespace clang { class ASTContext; class Decl; class CallExpr; }
namespace ascify {
struct RegisterTile32Factory {
  clang::SourceLocation begin, end;
  const clang::CallExpr* call = nullptr;
};

// This analysis cannot grant a standalone target. Its caller must first own
// that complete target and subsequently compile/link/publish it atomically.
// It admits only const auto locals and the listed register operations.
bool ProveRegisterTile32Factories(
    clang::ASTContext&,
    const std::function<bool(clang::SourceLocation)>& ownedLiteral,
    const std::function<bool(const clang::Decl*)>& ownedCgOrSdkDecl,
    std::vector<RegisterTile32Factory>&, std::string& error);
}  // namespace ascify
