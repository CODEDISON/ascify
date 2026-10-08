#include "CudaVec3AbiBoundary.h"

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
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
constexpr const char* kCubEntrySha =
    "c0b0dc5079bb8ead26a87d3dce77ea0f41522a797a5426787ecdfb0f6f3dd749";
constexpr const char* kCubUtilTypeSha =
    "01b79eff9d4d050fb05d9bcc3fd132c6345b284e4dbf972979b48f31a72d970a";
constexpr const char* kTupleVectorTypesSha =
    "aee5cf1924688f28cad93a1b30100534103c9ba7b67d86b1595f60ca047d6a4b";
constexpr const char* kRadixRankOperationsSha =
    "31ef8a823d0b71443482e5b8e4fbc967cccd0925eddd6390debfe3977f5ef39f";
constexpr const char* kThrustActorSha =
    "d8607a5963cbb561ad8b97c33185ae81fba12c5050ae63b872a779a06ad7dda1";
constexpr const char* kCompressedPairSha =
    "c7c2201e28e82c968bd49d15c0dde7d66548310d4687728ec198d40a3b395575";
constexpr const char* kTriviallyRelocatableSha =
    "be2ac5128897e81726a0a8f5a5d692b5892b1d61dae85fa6aaecbc7b40011739";
constexpr const char* kDispatchTransformSha =
    "7d1a82f7ba201164e4fa6a2b6269309ca79acf831da684f962d4a144e18e4b38";
constexpr const char* kZipIteratorSha =
    "69e5b2d743b086ee26ab8ddec0eb9998e1cec9594471ab4ae66822aa9d754433";
constexpr const char* kSubmdspanHelperSha =
    "dd9d25186913810174213f5849fd66cbc1fc6ed0a8f660b8b1ff406d6c6828f6";
constexpr const char* kBinarySearchHelpersSha =
    "c15bad43e5f87ceaa980670491d69571d40b62e3e7012fec91d4fea51267fd30";

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
  FrozenProviders(clang::SourceManager& sourceManager,
                  const std::string& configuredCudaRoot,
                  const std::vector<CudaVec3MappedCubInclude>& mappedCubIncludes)
      : sourceManager(sourceManager), mappedCubIncludes(mappedCubIncludes) {
    llvm::SmallString<256> canonical;
    if (!configuredCudaRoot.empty() &&
        !llvm::sys::fs::real_path(configuredCudaRoot, canonical))
      cudaRoot = canonical.str().str();
  }

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
        hash == kClangCudaRuntimeWrapperSha ||
        ((hash == kCubUtilTypeSha || hash == kTupleVectorTypesSha ||
          hash == kTriviallyRelocatableSha) &&
         isRemovedCubProvider(location));
  }

  bool isRemovedTupleProvider(clang::SourceLocation location) {
    return hashAt(location) == kTupleVectorTypesSha &&
        isRemovedCubProvider(location);
  }

  bool isRemovedDependentTupleSite(clang::SourceLocation location) {
    const auto hash = hashAt(location);
    return (hash == kRadixRankOperationsSha || hash == kThrustActorSha ||
            hash == kCompressedPairSha || hash == kDispatchTransformSha ||
            hash == kZipIteratorSha || hash == kBinarySearchHelpersSha) &&
        isRemovedCubProvider(location);
  }

  bool isRemovedTupleUsingSite(clang::SourceLocation location) {
    return hashAt(location) == kSubmdspanHelperSha &&
        isRemovedCubProvider(location);
  }

 private:
  bool physicalBufferMatches(clang::FileID file, llvm::StringRef expectedPath,
                             llvm::StringRef expectedSha) {
    const auto* entry = sourceManager.getFileEntryForID(file);
    if (entry == nullptr || sourceManager.isFileOverridden(entry))
      return false;
    const auto start = sourceManager.getLocForStartOfFile(file);
    llvm::SmallString<256> actual, expected;
    if (llvm::sys::fs::real_path(sourceManager.getFilename(start), actual) ||
        llvm::sys::fs::real_path(expectedPath, expected) || actual != expected)
      return false;
    llvm::sys::fs::UniqueID physical;
    if (llvm::sys::fs::getUniqueID(expected, physical) ||
        (physical.getDevice() == 0 && physical.getFile() == 0) ||
        physical != entry->getUniqueID() || hashAt(start) != expectedSha)
      return false;
    const auto bytes = llvm::MemoryBuffer::getFile(expected);
    return bytes && bufferSha((*bytes)->getBuffer()) == expectedSha;
  }

  bool isRemovedCubProvider(clang::SourceLocation use) {
    // Authenticate the explicit declaration providers and dependent-lookup
    // sites. Only isProvider's three roles exempt whole-file declarations.
    use = sourceManager.getExpansionLoc(use);
    if (use.isInvalid() || cudaRoot.empty() || mappedCubIncludes.empty())
      return false;
    const auto hash = hashAt(use);
    const char* suffix = nullptr;
    if (hash == kCubUtilTypeSha)
      suffix = "/include/cccl/cub/util_type.cuh";
    else if (hash == kTupleVectorTypesSha)
      suffix = "/include/cccl/cuda/std/__tuple_dir/vector_types.h";
    else if (hash == kRadixRankOperationsSha)
      suffix = "/include/cccl/cub/block/radix_rank_sort_operations.cuh";
    else if (hash == kThrustActorSha)
      suffix = "/include/cccl/thrust/detail/functional/actor.h";
    else if (hash == kCompressedPairSha)
      suffix = "/include/cccl/cuda/std/__memory/compressed_pair.h";
    else if (hash == kTriviallyRelocatableSha)
      suffix = "/include/cccl/thrust/type_traits/is_trivially_relocatable.h";
    else if (hash == kDispatchTransformSha)
      suffix = "/include/cccl/cub/device/dispatch/dispatch_transform.cuh";
    else if (hash == kZipIteratorSha)
      suffix = "/include/cccl/cuda/__iterator/zip_iterator.h";
    else if (hash == kSubmdspanHelperSha)
      suffix = "/include/cccl/cuda/std/__mdspan/submdspan_helper.h";
    else if (hash == kBinarySearchHelpersSha)
      suffix = "/include/cccl/cub/detail/binary_search_helpers.cuh";
    else
      return false;
    auto file = sourceManager.getFileID(use);
    const auto key = file.getHashValue();
    const auto cached = removedCubFiles.find(key);
    if (cached != removedCubFiles.end())
      return cached->second;
    bool removed = false;
    const std::string providerPath = cudaRoot + suffix;
    const std::string entryPath = cudaRoot + "/include/cccl/cub/cub.cuh";
    if (physicalBufferMatches(file, providerPath, hash)) {
      std::set<unsigned> ancestors;
      while (file.isValid() && ancestors.insert(file.getHashValue()).second) {
        const auto include = sourceManager.getIncludeLoc(file);
        if (include.isInvalid())
          break;
        if (physicalBufferMatches(file, entryPath, kCubEntrySha)) {
          const auto parent = sourceManager.getExpansionLoc(include);
          for (const auto& mapped : mappedCubIncludes) {
            llvm::SmallString<256> resolved, expected;
            if (llvm::sys::fs::real_path(mapped.resolvedPath, resolved) ||
                llvm::sys::fs::real_path(entryPath, expected) ||
                resolved != expected || !parent.isFileID() ||
                sourceManager.getFileID(parent) !=
                    sourceManager.getFileID(mapped.hashLocation))
              continue;
            const auto offset = sourceManager.getFileOffset(parent);
            if (offset >= sourceManager.getFileOffset(mapped.hashLocation) &&
                offset <= sourceManager.getFileOffset(mapped.filenameEnd)) {
              removed = true;
              break;
            }
          }
          break;
        }
        file = sourceManager.getFileID(sourceManager.getExpansionLoc(include));
      }
    }
    removedCubFiles.emplace(key, removed);
    return removed;
  }

  clang::SourceManager& sourceManager;
  const std::vector<CudaVec3MappedCubInclude>& mappedCubIncludes;
  std::string cudaRoot;
  std::map<unsigned, std::string> hashes;
  std::map<unsigned, bool> removedCubFiles;
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
  bool VisitCallExpr(clang::CallExpr* expression) {
    // A qualified callee's lookup can be nondependent while its call depends
    // on template arguments/operands. Walk-up precedes traversal of the callee.
    if (expression->isInstantiationDependent())
      if (const auto* lookup = llvm::dyn_cast<clang::UnresolvedLookupExpr>(
              expression->getCallee()->IgnoreParenImpCasts()))
        dependentLookupCalls.insert(lookup);
    return true;
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
    for (const auto* shadow : declaration->shadows()) {
      // These frozen, removed using-declarations import an overload set for
      // dependent tuple operations. They do not select a CUDA vector overload.
      if (providers.isRemovedTupleUsingSite(declaration->getLocation()) &&
          providers.isRemovedTupleProvider(shadow->getTargetDecl()->getLocation())) {
        if (!inspectUnselectedTupleCandidate(shadow->getTargetDecl(),
                                             declaration->getLocation()))
          return false;
        continue;
      }
      if (!inspectDecl(shadow->getTargetDecl(), declaration->getLocation()))
        return false;
    }
    return true;
  }
  bool VisitUnresolvedLookupExpr(clang::UnresolvedLookupExpr* expression) {
    for (const auto* declaration : expression->decls()) {
      // A dependent tuple get lookup lists every vector overload, including
      // uint3, even when the caller has only scalar types. Parameter types of
      // these unselected, authenticated removed-provider candidates are not
      // actual caller ABI uses. Audit the return, and retain the normal full
      // type audit for resolved references and every other declaration.
      if ((expression->isInstantiationDependent() ||
           dependentLookupCalls.count(expression) != 0) &&
          providers.isRemovedDependentTupleSite(expression->getNameLoc()) &&
          providers.isRemovedTupleProvider(declaration->getLocation())) {
        if (!inspectUnselectedTupleCandidate(declaration, expression->getNameLoc()))
          return false;
        continue;
      }
      if (!inspectDecl(declaration, expression->getNameLoc()))
        return false;
    }
    return true;
  }

 private:
  bool inspectUnselectedTupleCandidate(const clang::Decl* declaration,
                                      clang::SourceLocation use) {
    const auto* function = llvm::dyn_cast<clang::FunctionDecl>(declaration);
    if (const auto* templ =
            llvm::dyn_cast<clang::FunctionTemplateDecl>(declaration))
      function = templ->getTemplatedDecl();
    return function ? inspect(function->getReturnType(), use)
                    : inspectDecl(declaration, use);
  }

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
    if (use.isInvalid() || providers.isProvider(use))
      return true;
    if (found == records.end()) {
      // Removing a provider include must not hide a caller's exposed SDK ABI.
      // In particular CubVector<float/unsigned,3> inherits the CUDA record.
      if (const auto* derived = llvm::dyn_cast<clang::CXXRecordDecl>(record))
        if (const auto* definition = derived->getDefinition())
          for (const auto& base : definition->bases())
            if (!inspect(base.getType(), use))
              return false;
      return true;
    }
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
  std::set<const clang::UnresolvedLookupExpr*> dependentLookupCalls;
  std::string& error;
  clang::SourceLocation& location;
};

}  // namespace

bool ValidateCudaVec3AbiBoundary(clang::ASTContext& context,
                                const std::string& configuredCudaRoot,
                                const std::vector<CudaVec3MappedCubInclude>&
                                    mappedCubIncludes, std::string& error,
                                clang::SourceLocation& location) {
  FrozenProviders providers(context.getSourceManager(), configuredCudaRoot,
                            mappedCubIncludes);
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
