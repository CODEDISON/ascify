#include "CudaVec3AbiBoundary.h"

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Path.h"
#if LLVM_VERSION_MAJOR >= 13
#include "llvm/Support/SHA256.h"
#endif

namespace ascify {
namespace {

// CUDA 13.4.1 cudart headers, reviewed against their redistribution identity.
// Both reviewed Samples versions use this same CUDA installation. Parsed
// bytes identify rejection candidates; they never authenticate a replacement.
constexpr const char* kVectorTypesSha =
    "ded087a2c3ca89fd3a5150589175aca0aa4714357016209ea3e7a51f94845e75";
constexpr const char* kClangCudaRuntimeWrapperSha =
    "6caeb75ad8b888bc93141b700608cf3581f334fc5cc0971e96e3f699e86ceb45";
constexpr const char* kClangCudaBuiltinVarsSha =
    "32d445ec643c5802efbd54d41e4e8bff952436586d992fc0f91258238cb6e115";

std::string bufferSha(llvm::StringRef contents) {
#if LLVM_VERSION_MAJOR >= 13
  llvm::SHA256 hash;
  hash.update(contents);
  const auto digest = hash.final();
  const char hex[] = "0123456789abcdef";
  std::string result;
  for (const auto byte : digest) {
    const auto value = static_cast<unsigned char>(byte);
    result += hex[value >> 4];
    result += hex[value & 15];
  }
  return result;
#else
  (void)contents;
  return {};
#endif
}

class FrozenProviders {
 public:
  explicit FrozenProviders(clang::SourceManager& sourceManager)
      : sourceManager(sourceManager) {}

  llvm::StringRef hashAt(clang::SourceLocation location) {
    location = sourceManager.getExpansionLoc(location);
    if (location.isInvalid())
      return {};
    const auto file = sourceManager.getFileID(location);
    if (file.isInvalid())
      return {};
    const auto key = file.getHashValue();
    auto found = hashes.find(key);
    if (found == hashes.end()) {
      bool invalid = false;
      const auto bytes = sourceManager.getBufferData(file, &invalid);
      found = hashes.emplace(key, invalid ? std::string() : bufferSha(bytes)).first;
    }
    return found->second;
  }

  bool isProvider(clang::SourceLocation location) {
    // Exempt only these exact provider buffers, not arbitrary system headers.
    // A project header marked -isystem or #pragma system_header is still audited.
    const auto hash = hashAt(location);
    return hash == kVectorTypesSha ||
        hash == "f15e2cc5f6b26a617143551b8aed48bd74e5d997ecbe9e3a7abe7cb430262d6b" ||
        hash == "4d2ef3eff9431eba4e3fe0ae7e2121d24013afb0b7660b40531b1641d9010541" ||
        hash == "601cc73210a046d029e25f189ade5d02004aaa6c045a5100be80b5d5e52b5e57" ||
        hash == "1f5fc20bebef01a4b6eb2b385ec4c590c813c544dc8808dd8ecc238b3922b2f4" ||
        hash == "0db6fe73f95ae49dbd3f74d174d3f1a1bd1f1908805a2f335cd5a372a178b947" ||
        // Matching Clang 23 builtin variables and their conversion bodies.
        hash == kClangCudaBuiltinVarsSha ||
        hash == kClangCudaRuntimeWrapperSha;
  }

 private:
  clang::SourceManager& sourceManager;
  std::map<unsigned, std::string> hashes;
};

using Records = std::map<const clang::TagDecl*, std::pair<int64_t, int64_t>>;

class FindRecords : public clang::RecursiveASTVisitor<FindRecords> {
 public:
  FindRecords(clang::ASTContext& context, FrozenProviders& providers,
              Records& records)
      : context(context), providers(providers), records(records) {}

  bool VisitDecl(clang::Decl* declaration) {
    const auto location = context.getSourceManager().getExpansionLoc(
        declaration->getLocation());
    if (location.isInvalid())
      return true;
    const auto filename = llvm::sys::path::filename(
        context.getSourceManager().getFilename(location));
    if ((filename == "__clang_cuda_runtime_wrapper.h" &&
         providers.hashAt(location) != kClangCudaRuntimeWrapperSha) ||
        (filename == "__clang_cuda_builtin_vars.h" &&
         providers.hashAt(location) != kClangCudaBuiltinVarsSha))
      unknownClangRuntimeProvider = true;
    return true;
  }

  bool VisitRecordDecl(clang::RecordDecl* declaration) {
    if (!declaration->isCompleteDefinition() ||
        !declaration->getDeclContext()->isTranslationUnit() ||
        (declaration->getName() != "float3" && declaration->getName() != "uint3") ||
        providers.hashAt(declaration->getLocation()) != kVectorTypesSha)
      return true;
    auto& sourceManager = context.getSourceManager();
    auto file = sourceManager.getFileID(sourceManager.getExpansionLoc(
        declaration->getLocation()));
    std::set<unsigned> ancestors;
    while (file.isValid() && ancestors.insert(file.getHashValue()).second) {
      const auto start = sourceManager.getLocForStartOfFile(file);
      if (llvm::sys::path::filename(sourceManager.getFilename(start)) ==
              "__clang_cuda_runtime_wrapper.h" &&
          providers.hashAt(start) != kClangCudaRuntimeWrapperSha) {
        unknownClangRuntimeProvider = true;
        return true;
      }
      const auto include = sourceManager.getIncludeLoc(file);
      if (include.isInvalid())
        break;
      file = sourceManager.getFileID(sourceManager.getExpansionLoc(include));
    }
    const auto& layout = context.getASTRecordLayout(declaration);
    // Exact parsed definition bytes and canonical identity are sufficient for
    // a rejection. Do not silently skip a pragma/macro-altered source layout;
    // rejecting it does not claim any replacement ABI has been authenticated.
    records.emplace(declaration->getCanonicalDecl(), std::make_pair(
        layout.getSize().getQuantity(), layout.getAlignment().getQuantity()));
    return true;
  }

  bool hasUnknownClangRuntimeProvider() const {
    return unknownClangRuntimeProvider;
  }

 private:
  clang::ASTContext& context;
  FrozenProviders& providers;
  Records& records;
  bool unknownClangRuntimeProvider = false;
};

class AuditUses : public clang::RecursiveASTVisitor<AuditUses> {
 public:
  AuditUses(FrozenProviders& providers, const Records& records,
            std::string& error, clang::SourceLocation& location)
      : providers(providers), records(records), error(error), location(location) {}

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool VisitTypeLoc(clang::TypeLoc type) {
    return inspect(type.getType(), type.getBeginLoc());
  }
  bool VisitValueDecl(clang::ValueDecl* declaration) {
    return inspect(declaration->getType(), declaration->getLocation());
  }
  bool VisitTypedefNameDecl(clang::TypedefNameDecl* declaration) {
    return inspect(declaration->getUnderlyingType(), declaration->getLocation());
  }
  bool VisitExpr(clang::Expr* expression) {
    return inspect(expression->getType(), expression->getExprLoc());
  }
  bool VisitRecordDecl(clang::RecordDecl* declaration) {
    return inspectRecord(declaration, declaration->getLocation());
  }
  bool VisitUsingShadowDecl(clang::UsingShadowDecl* declaration) {
    return inspectDecl(declaration->getTargetDecl(), declaration->getLocation());
  }
  bool VisitUsingDecl(clang::UsingDecl* declaration) {
    // Using shadows are implicit declarations and are not visited by the
    // default RecursiveASTVisitor traversal. Audit the source using itself.
    for (const auto* shadow : declaration->shadows())
      if (!inspectDecl(shadow->getTargetDecl(), declaration->getLocation()))
        return false;
    return true;
  }
  bool VisitUnresolvedLookupExpr(clang::UnresolvedLookupExpr* expression) {
    for (const auto* declaration : expression->decls())
      if (!inspectDecl(declaration, expression->getNameLoc()))
        return false;
    return true;
  }

 private:
  bool inspectDecl(const clang::Decl* declaration, clang::SourceLocation use) {
    if (const auto* record = llvm::dyn_cast<clang::RecordDecl>(declaration))
      return inspectRecord(record, use);
    if (const auto* value = llvm::dyn_cast<clang::ValueDecl>(declaration))
      return inspect(value->getType(), use);
    if (const auto* alias = llvm::dyn_cast<clang::TypedefNameDecl>(declaration))
      return inspect(alias->getUnderlyingType(), use);
    if (const auto* function = llvm::dyn_cast<clang::FunctionTemplateDecl>(declaration))
      return inspect(function->getTemplatedDecl()->getType(), use);
    return true;
  }

  bool inspectRecord(const clang::RecordDecl* record, clang::SourceLocation use) {
    const auto found = records.find(record->getCanonicalDecl());
    if (found == records.end() || use.isInvalid() ||
        providers.isProvider(use))
      return true;
    location = use;
    error = "CUDA " + record->getNameAsString() +
        " record ABI (size " + std::to_string(found->second.first) +
        ", alignment " + std::to_string(found->second.second) +
        ") is not proved compatible with the native "
        "DPP vector spelling; automatic vec3 adaptation is not proved";
    return false;
  }

  bool inspect(clang::QualType type, clang::SourceLocation use) {
    if (type.isNull() || use.isInvalid() || providers.isProvider(use))
      return true;
    if (const auto* record = type->getAs<clang::RecordType>())
      return inspectRecord(record->getDecl(), use);
    if (type->isPointerType() || type->isReferenceType())
      return inspect(type->getPointeeType(), use);
    if (const auto* array = llvm::dyn_cast<clang::ArrayType>(type.getCanonicalType()))
      return inspect(array->getElementType(), use);
    if (const auto* function = type->getAs<clang::FunctionType>()) {
      if (!inspect(function->getReturnType(), use))
        return false;
      if (const auto* prototype = llvm::dyn_cast<clang::FunctionProtoType>(function))
        for (const auto parameter : prototype->param_types())
          if (!inspect(parameter, use))
            return false;
    }
    if (const auto* member = type->getAs<clang::MemberPointerType>())
      return inspect(member->getPointeeType(), use);
    return true;
  }

  FrozenProviders& providers;
  const Records& records;
  std::string& error;
  clang::SourceLocation& location;
};

}  // namespace

bool ValidateCudaVec3AbiBoundary(clang::ASTContext& context, std::string& error,
                                clang::SourceLocation& location) {
  FrozenProviders providers(context.getSourceManager());
  Records records;
  FindRecords find(context, providers, records);
  find.TraverseDecl(context.getTranslationUnitDecl());
  // Provider exemptions are version-specific. An unreviewed genuine Clang
  // runtime wrapper must not turn its own internal uint3 declarations into a
  // rejection of every scalar program. Such a tuple receives no safety claim.
  if (records.empty() || find.hasUnknownClangRuntimeProvider())
    return true;
  AuditUses audit(providers, records, error, location);
  return audit.TraverseDecl(context.getTranslationUnitDecl());
}

}  // namespace ascify
