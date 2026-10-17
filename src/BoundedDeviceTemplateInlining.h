#pragma once

#include <functional>
#include <set>
#include <vector>
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"

namespace clang { class ASTContext; }
namespace ascify {
struct BoundedDeviceTemplateEdits {
  std::vector<clang::SourceLocation> qualifiers;
  unsigned families = 0;
  unsigned instances = 0;
};

// Optional, closed-world optimization of observed int-NTTP families. A failed
// proof leaves the entire family unchanged; it never bypasses a publication
// boundary. Call only after the ordinary texture and record audits succeed.
BoundedDeviceTemplateEdits ProveBoundedDeviceTemplateInlining(
    clang::ASTContext &context, const std::set<unsigned> &textureOffsets,
    const std::function<bool(llvm::StringRef, clang::SourceLocation)> &ownedQualifier);
} // namespace ascify
