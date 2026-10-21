#pragma once

#include <string>

namespace clang {
namespace tooling {
class RefactoringTool;
}  // namespace tooling
}  // namespace clang

namespace ascify {

inline constexpr const char* kNoFrontendCompatibility = "none";
inline constexpr const char* kAdmittedFrontendCompatibilityV1 =
    "ascify-admitted-v1";

struct FrontendCompatibilityConfig {
  std::string profile;
  std::string canonicalRoot;

  bool enabled() const { return !canonicalRoot.empty(); }
};

// Adds a product-owned parser include only for an explicitly selected profile.
// The default profile preserves the official CUDA dependency lane unchanged.
bool ConfigureFrontendCompatibility(
    clang::tooling::RefactoringTool& tool,
    const std::string& profile,
    const char* ascifyExecutable,
    FrontendCompatibilityConfig& config,
    std::string& error);

// Enforces that cooperative-groups includes admitted by an enabled profile
// resolve to the exact verified product header.  Unknown subheaders and local
// shadows fail before translated output can be published.
bool ValidateFrontendCompatibilityInclude(
    const FrontendCompatibilityConfig& config,
    const std::string& includeSpelling,
    const std::string& resolvedPath,
    std::string& error);

// Recognizes only the active product host-math header and its exact parsed
// bytes. Callers must additionally prove physical SourceManager ownership and
// the forced-include instance; this does not admit a copied caller header.
bool IsVerifiedFrontendHostMathInput(
    const FrontendCompatibilityConfig& config,
    const std::string& path,
    const std::string& parsedContents);

// Identifies only the two exact product CG parser headers. This is byte/path
// identity, not preprocessing or AST admission; callers must prove those too.
bool IsVerifiedFrontendCooperativeGroupsInput(
    const FrontendCompatibilityConfig& config,
    const std::string& path,
    const std::string& parsedContents);

}  // namespace ascify
