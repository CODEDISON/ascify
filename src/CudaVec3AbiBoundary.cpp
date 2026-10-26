#include "CudaVec3AbiBoundary.h"

#include <cstdint>
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
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

// Clang changed both getters in the newer type/qualifier representation.
// Select the available primary API without guessing a release boundary.
template <class Context, class Record>
static auto vec3RecordType(Context& context, const Record* record, int)
    -> decltype(context.getRecordType(record), clang::QualType()) {
  return context.getRecordType(record);
}
template <class Context, class Record>
static auto vec3RecordType(Context& context, const Record* record, long)
    -> decltype(context.getCanonicalTagType(record), clang::QualType()) {
  return context.getCanonicalTagType(record);
}
template <class Member>
static auto vec3MemberClass(const Member* member, int)
    -> decltype(member->getClass(), clang::QualType()) {
  return clang::QualType(member->getClass(), 0);
}
template <class Member>
static auto vec3MemberClass(const Member* member, long)
    -> decltype(member->getQualifier().getAsType(), clang::QualType()) {
  const auto qualifier = member->getQualifier();
  using Qualifier = decltype(member->getQualifier());
  if (qualifier.getKind() != Qualifier::Kind::Type)
    return {};
  return clang::QualType(qualifier.getAsType(), 0);
}

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
                  const std::vector<CudaVec3MappedCubInclude>& mappedCubIncludes,
                  const std::string& ownedClangRoot = {})
      : sourceManager(sourceManager), mappedCubIncludes(mappedCubIncludes) {
    llvm::SmallString<256> canonical;
    if (!configuredCudaRoot.empty() &&
        !llvm::sys::fs::real_path(configuredCudaRoot, canonical))
      cudaRoot = canonical.str().str();
    canonical.clear();
    if (!ownedClangRoot.empty() &&
        !llvm::sys::fs::real_path(ownedClangRoot, canonical))
      clangRoot = canonical.str().str();
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

  bool isRemovedUtilProvider(const clang::FileEntry& entry) {
    const auto file = sourceManager.translateFile(&entry);
    if (file.isInvalid())
      return false;
    const auto start = sourceManager.getLocForStartOfFile(file);
    return hashAt(start) == kCubUtilTypeSha && isRemovedCubProvider(start);
  }

  bool authenticVectorDefinition(clang::SourceLocation location) {
    return authenticFile(location, "vector_types.h", kVectorTypesSha);
  }

  bool authenticFactoryDeclaration(clang::SourceLocation location) {
    return authenticFile(location, "vector_functions.h",
        "f15e2cc5f6b26a617143551b8aed48bd74e5d997ecbe9e3a7abe7cb430262d6b") ||
        authenticFile(location, "vector_functions.hpp",
        "4d2ef3eff9431eba4e3fe0ae7e2121d24013afb0b7660b40531b1641d9010541");
  }

  bool authenticBuiltinUInt3Forward(const clang::RecordDecl* declaration) {
    // This is one known compiler-owned forward declaration, not authority for
    // arbitrary provider uses or a caller-selected copy of ResourceDir.
    if (clangRoot.empty() || declaration->getName() != "uint3" ||
        !declaration->isStruct() || declaration->isThisDeclarationADefinition() ||
        !declaration->decls_empty() || declaration->hasAttrs() ||
        !declaration->getDeclContext()->isTranslationUnit() ||
        !declaration->getLocation().isFileID()) return false;
    const auto file = sourceManager.getFileID(declaration->getLocation());
    if (!physicalBufferMatches(file, clangRoot + "/include/__clang_cuda_builtin_vars.h",
                               kClangCudaBuiltinVarsSha)) return false;
    const auto include = sourceManager.getIncludeLoc(file);
    return include.isFileID() && physicalBufferMatches(sourceManager.getFileID(include),
        clangRoot + "/include/__clang_cuda_runtime_wrapper.h", kClangCudaRuntimeWrapperSha);
  }

 private:
  bool authenticFile(clang::SourceLocation location, llvm::StringRef name,
                     llvm::StringRef expectedSha) {
    location = sourceManager.getSpellingLoc(location);
    return !cudaRoot.empty() && location.isValid() &&
        physicalBufferMatches(sourceManager.getFileID(location),
                              cudaRoot + "/include/" + name.str(), expectedSha);
  }

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
  std::string clangRoot;
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
        (declaration->getName() != "float3" && declaration->getName() != "uint3" &&
         declaration->getName() != "int3") ||
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
            std::string& error, clang::SourceLocation& location,
            const CudaVec3LayoutPlan* plan)
      : providers(providers), records(records), error(error), location(location), plan(plan) {}

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
    if (plan != nullptr && plan->uses.count(
            {use.getRawEncoding(), record->getCanonicalDecl()}) != 0)
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
  std::set<const clang::UnresolvedLookupExpr*> dependentLookupCalls;
  std::string& error;
  clang::SourceLocation& location;
  const CudaVec3LayoutPlan* plan;
};

// An intentionally closed first profile: automatic scalar records and defined
// non-template inline functions. The original ABI auditor still visits every
// use afterwards, including uses this planner does not know how to rewrite.
class LocalLayoutProof : public clang::RecursiveASTVisitor<LocalLayoutProof> {
  using Base = clang::RecursiveASTVisitor<LocalLayoutProof>;
 public:
  LocalLayoutProof(clang::ASTContext& context, FrozenProviders& providers,
                   const Records& records,
                   const std::function<bool(clang::SourceLocation)>& selectedHeader,
                   const std::function<bool(clang::SourceLocation)>& literalUnobserved,
                   CudaVec3LayoutPlan& plan, std::string& error,
                   clang::SourceLocation& location)
      : context(context), source(context.getSourceManager()), providers(providers),
        records(records), selectedHeader(selectedHeader),
        literalUnobserved(literalUnobserved), plan(plan), error(error), location(location) {}

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool TraverseDecl(clang::Decl* declaration) {
    const auto* prior = function;
    if (const auto* next = llvm::dyn_cast_or_null<clang::FunctionDecl>(declaration))
      function = next;
    const bool result = Base::TraverseDecl(declaration);
    function = prior;
    return result;
  }
  bool TraverseLambdaExpr(clang::LambdaExpr* expression) {
    ++lambdaDepth;
    const bool result = Base::TraverseLambdaExpr(expression);
    --lambdaDepth;
    return result;
  }
  bool TraverseTemplateArgumentLoc(const clang::TemplateArgumentLoc& argument) {
    if (argument.getArgument().getKind() == clang::TemplateArgument::Type &&
        observe(argument.getArgument().getAsType(), argument.getLocation()))
      fail(argument.getLocation(), "vec3 template argument has no closed future-instantiation proof");
    return Base::TraverseTemplateArgumentLoc(argument);
  }

  bool VisitNamedDecl(clang::NamedDecl* declaration) {
    if (declaration->getIdentifier() != nullptr &&
        !providers.isProvider(declaration->getLocation())) {
      const auto name = declaration->getName();
      if (name.take_front(12) == "ascify_cuda_" || name.take_front(17) == "ascify_make_cuda_")
        fail(declaration->getLocation(), "input declaration owns a generated vec3 name");
      if (isFactoryName(name))
        if (const auto* fn = llvm::dyn_cast<clang::FunctionDecl>(declaration))
          if (closedInline(fn) && declaration->getLocation().isFileID())
            boundNames.insert(declaration->getLocation().getRawEncoding());
    }
    return true;
  }

  bool VisitTypeLoc(clang::TypeLoc type) {
    const auto use = type.getBeginLoc();
    if (!observe(type.getType(), use)) return true;
    if (!allowedShape(type.getType()) || !ordinaryOwner())
      return fail(use, "pointer/array/dependent vec3 type is outside the local-record profile");
    if (const auto named = type.getAs<clang::RecordTypeLoc>()) {
      if (const auto* record = vectorRecord(named.getType()))
        edit(named.getNameLoc(), record->getName(), "ascify_cuda_" + record->getNameAsString());
    } else if (const auto named = type.getAs<clang::TypedefTypeLoc>()) {
      const auto* alias = named.getTypePtr()->getDecl();
      if (providers.authenticVectorDefinition(alias->getLocation())) {
        const auto* record = vectorRecord(alias->getUnderlyingType());
        if (record == nullptr || alias->getName() != record->getName())
          return fail(use, "unproved SDK vec3 alias");
        edit(named.getNameLoc(), alias->getName(), "ascify_cuda_" + record->getNameAsString());
      }
    } else if (type.getAs<clang::DecltypeTypeLoc>()) {
      return fail(use, "decltype vec3 is not admitted by the local-record profile");
    }
    return true;
  }

  bool VisitTemplateSpecializationTypeLoc(clang::TemplateSpecializationTypeLoc type) {
    if (providers.isProvider(type.getBeginLoc())) return true;
    for (unsigned index = 0; index < type.getNumArgs(); ++index) {
      const auto argument = type.getArgLoc(index).getArgument();
      if (argument.getKind() == clang::TemplateArgument::Type && contains(argument.getAsType())) {
        observe(argument.getAsType(), type.getBeginLoc());
        fail(type.getBeginLoc(), "vec3 template argument has no future-instantiation proof");
      }
    }
    return true;
  }

  bool VisitValueDecl(clang::ValueDecl* declaration) {
    if (!providers.isProvider(declaration->getLocation()))
      if (const auto* variable = llvm::dyn_cast<clang::VarDecl>(declaration))
        if (variable->getType()->isReferenceType() && variable->hasInit() &&
            !contains(variable->getType()) && derivesFromVector(variable->getInit()))
          fail(variable->getLocation(), "vec3 component reference binding is outside the scalar-local profile");
    if (!observe(declaration->getType(), declaration->getLocation())) return true;
    if (const auto* fn = llvm::dyn_cast<clang::FunctionDecl>(declaration)) {
      if (!closedInline(fn)) fail(declaration->getLocation(), "vec3 function ABI is not a defined closed inline function");
    } else if (const auto* parameter = llvm::dyn_cast<clang::ParmVarDecl>(declaration)) {
      if (!closedInline(llvm::dyn_cast<clang::FunctionDecl>(parameter->getDeclContext())) ||
          !allowedScalarOrReference(parameter->getType()) || parameter->hasAttrs())
        fail(parameter->getLocation(), "vec3 parameter is outside the closed inline profile");
    } else if (const auto* variable = llvm::dyn_cast<clang::VarDecl>(declaration)) {
      if (!variable->isLocalVarDecl() || variable->hasGlobalStorage() || variable->hasAttrs() ||
          !ordinaryOwner() || !allowedScalarOrReference(variable->getType()))
        fail(variable->getLocation(), "vec3 storage must be an automatic scalar local");
    } else {
      fail(declaration->getLocation(), "vec3 field, indirect call, or nonlocal declaration is not proved");
    }
    return true;
  }

  bool VisitTypedefNameDecl(clang::TypedefNameDecl* declaration) {
    if (observe(declaration->getUnderlyingType(), declaration->getLocation()) &&
        (!declaration->getDeclContext()->isFunctionOrMethod() || !ordinaryOwner() ||
         !allowedScalarOrReference(declaration->getUnderlyingType())))
      fail(declaration->getLocation(), "vec3 alias must stay inside an ordinary function");
    return true;
  }

  bool VisitRecordDecl(clang::RecordDecl* declaration) {
    if (records.count(declaration->getCanonicalDecl()) &&
        !providers.isProvider(declaration->getLocation())) {
      observe(vec3RecordType(context, declaration, 0), declaration->getLocation());
      fail(declaration->getLocation(), "caller redeclaration of the SDK vec3 record is not proved");
    }
    if (const auto* cxx = llvm::dyn_cast<clang::CXXRecordDecl>(declaration))
      if (cxx->isCompleteDefinition())
        for (const auto& base : cxx->bases())
          if (observe(base.getType(), base.getBeginLoc()))
            fail(base.getBeginLoc(), "vec3 inheritance is not admitted");
    return true;
  }

  bool VisitExpr(clang::Expr* expression) {
    if (!observe(expression->getType(), expression->getExprLoc())) return true;
    if (const auto* decay = llvm::dyn_cast<clang::ImplicitCastExpr>(expression)) {
      if (decay->getCastKind() == clang::CK_FunctionToPointerDecay) {
        const auto* reference = llvm::dyn_cast<clang::DeclRefExpr>(decay->getSubExpr()->IgnoreParenImpCasts());
        const auto* callee = reference == nullptr ? nullptr : llvm::dyn_cast<clang::FunctionDecl>(reference->getDecl());
        if (!ordinaryOwner() || reference == nullptr || !isDirectCallReference(reference) ||
            (!factory(callee) && !closedInline(callee) && !implicitRecordAssignment(callee)))
          fail(decay->getExprLoc(), "vec3 function pointer escapes a proved direct call");
        return true;
      }
    }
    if (!allowedShape(expression->getType()) || !ordinaryOwner())
      return fail(expression->getExprLoc(), "vec3 expression has an unproved type or template owner");
    if (const auto* reference = llvm::dyn_cast<clang::DeclRefExpr>(expression)) {
      if (const auto* callee = llvm::dyn_cast<clang::FunctionDecl>(reference->getDecl())) {
        if (!isDirectCallReference(reference) ||
            (!factory(callee) && !closedInline(callee) && !implicitRecordAssignment(callee)))
          return fail(reference->getExprLoc(), "vec3 function reference is not a proved direct call");
        if (factory(callee)) edit(reference->getLocation(), callee->getName(),
                                 "ascify_" + callee->getNameAsString().substr(0, 5) +
                                 "cuda_" + callee->getNameAsString().substr(5));
        else if (callee->getIdentifier() != nullptr && isFactoryName(callee->getName()) &&
                 reference->getLocation().isFileID())
          boundNames.insert(reference->getLocation().getRawEncoding());
      } else if (!llvm::isa<clang::VarDecl>(reference->getDecl())) {
        return fail(reference->getExprLoc(), "vec3 reference is not a local or inline parameter");
      }
      return true;
    }
    if (const auto* cast = llvm::dyn_cast<clang::ImplicitCastExpr>(expression)) {
      if (cast->getCastKind() != clang::CK_LValueToRValue && cast->getCastKind() != clang::CK_NoOp &&
          cast->getCastKind() != clang::CK_FunctionToPointerDecay)
        fail(cast->getExprLoc(), "vec3 implicit conversion is not a value copy");
      // A function-to-pointer decay is only legal as the proved direct callee;
      // allowedShape deliberately rejects stored function pointers elsewhere.
      return true;
    }
    if (const auto* construct = llvm::dyn_cast<clang::CXXConstructExpr>(expression)) {
      const auto* ctor = construct->getConstructor();
      if (!ctor->isImplicit() || !vectorRecord(construct->getType()) ||
          (!ctor->isCopyOrMoveConstructor() && !ctor->isDefaultConstructor()))
        fail(construct->getExprLoc(), "vec3 construction is not a trivial SDK record operation");
      return true;
    }
    if (llvm::isa<clang::CallExpr>(expression) || llvm::isa<clang::InitListExpr>(expression) ||
        llvm::isa<clang::ParenExpr>(expression) || llvm::isa<clang::MaterializeTemporaryExpr>(expression) ||
        llvm::isa<clang::CXXBindTemporaryExpr>(expression) || llvm::isa<clang::ExprWithCleanups>(expression))
      return true;
    return fail(expression->getExprLoc(), "vec3 expression kind is outside the closed local profile");
  }

  bool VisitCallExpr(clang::CallExpr* expression) {
    if (providers.isProvider(expression->getExprLoc())) return true;
    const auto* callee = expression->getDirectCallee();
    for (unsigned index = 0; index < expression->getNumArgs(); ++index) {
      const auto* argument = expression->getArg(index);
      if (contains(argument->getType()) || !derivesFromVector(argument)) continue;
      if (callee == nullptr || (index < callee->getNumParams() &&
                               callee->getParamDecl(index)->getType()->isReferenceType()))
        fail(argument->getExprLoc(), "vec3 component cannot bind an unproved call reference");
    }
    bool touched = contains(expression->getType());
    for (const auto* argument : expression->arguments()) touched |= contains(argument->getType());
    if (!touched) return true;
    if (callee == nullptr || (!factory(callee) && !closedInline(callee) && !implicitRecordAssignment(callee)))
      fail(expression->getExprLoc(), "vec3 call crosses an unproved SDK, builtin, or external ABI");
    return true;
  }

  bool VisitCXXConstructExpr(clang::CXXConstructExpr* expression) {
    if (providers.isProvider(expression->getExprLoc())) return true;
    const auto* constructor = expression->getConstructor();
    bool argumentUsesVector = false;
    for (unsigned index = 0; index < expression->getNumArgs(); ++index) {
      const auto* argument = expression->getArg(index);
      argumentUsesVector |= contains(argument->getType());
      if (!contains(argument->getType()) && derivesFromVector(argument) &&
          index < constructor->getNumParams() &&
          constructor->getParamDecl(index)->getType()->isReferenceType())
        fail(argument->getExprLoc(), "vec3 component cannot bind an unproved constructor reference");
    }
    if (argumentUsesVector && vectorRecord(expression->getType()) == nullptr)
      fail(expression->getExprLoc(), "vec3 construction crosses an unproved record or dim3 ABI");
    return true;
  }

  bool VisitCastExpr(clang::CastExpr* expression) {
    if (!providers.isProvider(expression->getExprLoc()) &&
        !llvm::isa<clang::ImplicitCastExpr>(expression) &&
        (contains(expression->getType()) || contains(expression->getSubExpr()->getType())))
      fail(expression->getExprLoc(), "explicit cast observes an unproved vec3 representation boundary");
    return true;
  }

  bool VisitReturnStmt(clang::ReturnStmt* statement) {
    const auto* value = statement->getRetValue();
    if (function != nullptr && value != nullptr &&
        !providers.isProvider(statement->getReturnLoc()) &&
        function->getReturnType()->isReferenceType() &&
        !contains(function->getReturnType()) && derivesFromVector(value))
      fail(statement->getReturnLoc(), "vec3 component cannot escape through a reference return");
    return true;
  }

  bool VisitInitListExpr(clang::InitListExpr* expression) {
    if (providers.isProvider(expression->getExprLoc()) || contains(expression->getType())) return true;
    if (const auto* recordType = expression->getType()->getAs<clang::RecordType>()) {
      const auto* record = recordType->getDecl()->getDefinition();
      if (record != nullptr)
        for (const auto* field : record->fields())
          if (field->getType()->isReferenceType() && derivesFromVector(expression))
            fail(expression->getExprLoc(), "vec3 component cannot bind a reference aggregate");
    }
    return true;
  }

  bool VisitUnaryOperator(clang::UnaryOperator* expression) {
    if (!providers.isProvider(expression->getExprLoc()) && expression->getOpcode() == clang::UO_AddrOf &&
        derivesFromVector(expression->getSubExpr()))
      fail(expression->getExprLoc(), "vec3 or component address escapes the scalar-local profile");
    return true;
  }
  bool VisitCXXTypeidExpr(clang::CXXTypeidExpr* expression) {
    const auto type = expression->isTypeOperand() ? expression->getTypeOperand(context)
                                                : expression->getExprOperand()->getType();
    if (observe(type, expression->getExprLoc()))
      fail(expression->getExprLoc(), "vec3 RTTI observes the rewritten type identity");
    return true;
  }
  bool VisitPredefinedExpr(clang::PredefinedExpr* expression) {
    if (function != nullptr && contains(function->getType()) &&
        !providers.isProvider(expression->getExprLoc()))
      fail(expression->getExprLoc(), "vec3 inline signature is observed by a predefined function string");
    return true;
  }
  bool VisitCXXThrowExpr(clang::CXXThrowExpr* expression) {
    if (expression->getSubExpr() != nullptr &&
        observe(expression->getSubExpr()->getType(), expression->getExprLoc()))
      fail(expression->getExprLoc(), "vec3 exception crosses an unproved runtime ABI");
    return true;
  }
  bool VisitCXXCatchStmt(clang::CXXCatchStmt* statement) {
    if (const auto* caught = statement->getExceptionDecl())
      if (observe(caught->getType(), statement->getCatchLoc()))
        fail(statement->getCatchLoc(), "vec3 catch observes an unproved runtime type identity");
    return true;
  }
  bool VisitUsingDecl(clang::UsingDecl* declaration) {
    if (providers.isProvider(declaration->getLocation())) return true;
    for (const auto* shadow : declaration->shadows()) {
      const auto* target = shadow->getTargetDecl();
      if (providers.isRemovedTupleUsingSite(declaration->getLocation()) &&
          providers.isRemovedTupleProvider(target->getLocation())) {
        if (const auto* fn = llvm::dyn_cast<clang::FunctionDecl>(target))
          if (observe(fn->getReturnType(), declaration->getLocation()))
            fail(declaration->getLocation(), "removed tuple using exposes a vec3 return");
        continue;
      }
      if (const auto* value = llvm::dyn_cast<clang::ValueDecl>(target)) {
        if (observe(value->getType(), declaration->getLocation()))
          fail(declaration->getLocation(), "vec3 using declaration is not proved");
      } else if (const auto* alias = llvm::dyn_cast<clang::TypedefNameDecl>(target)) {
        if (observe(alias->getUnderlyingType(), declaration->getLocation()))
          fail(declaration->getLocation(), "vec3 using declaration is not proved");
      }
    }
    return true;
  }
  bool VisitUnresolvedLookupExpr(clang::UnresolvedLookupExpr* expression) {
    if (providers.isProvider(expression->getExprLoc())) return true;
    for (const auto* declaration : expression->decls()) {
      if (providers.isRemovedDependentTupleSite(expression->getExprLoc()) &&
          providers.isRemovedTupleProvider(declaration->getLocation())) {
        if (const auto* fn = llvm::dyn_cast<clang::FunctionDecl>(declaration))
          if (observe(fn->getReturnType(), expression->getExprLoc()))
            fail(expression->getExprLoc(), "removed dependent tuple lookup exposes a vec3 return");
        continue;
      }
      if (const auto* value = llvm::dyn_cast<clang::ValueDecl>(declaration))
        if (observe(value->getType(), expression->getExprLoc()))
          fail(expression->getExprLoc(), "vec3 unresolved overload or dependent lookup is not proved");
    }
    return true;
  }

  bool finish() {
    if (!observed) { plan = {}; error.clear(); location = {}; return true; }
    auditRawBindings();
    if (!error.empty()) { plan = {}; return false; }
    for (const auto& item : edits) plan.edits.push_back(item.second);
    for (const auto& item : buffers)
      plan.evidence.push_back("buffer=" + item.first + ";sha256=" + item.second);
    for (const auto& item : plan.uses) {
      const auto use = clang::SourceLocation::getFromRawEncoding(item.first);
      plan.evidence.push_back("site=" + source.getFilename(use).str() + ":" +
          std::to_string(source.getFileOffset(use)) + ";type=" +
          item.second->getNameAsString() + ";layout=12/4");
    }
    plan.evidence.push_back("provider=CUDA13.4.1;vector_types_sha256=" + std::string(kVectorTypesSha) +
                           ";layout=size12,align4,offsets0/4/8;scope=closed-scalar-local-v1");
    std::sort(plan.evidence.begin(), plan.evidence.end());
    plan.evidence.erase(std::unique(plan.evidence.begin(), plan.evidence.end()), plan.evidence.end());
    return true;
  }

 private:
  const clang::RecordDecl* vectorRecord(clang::QualType type) const {
    if (const auto* record = type->getAs<clang::RecordType>())
      if (records.count(record->getDecl()->getCanonicalDecl())) return record->getDecl();
    return nullptr;
  }
  void collect(clang::QualType type, std::set<const clang::TagDecl*>& result) const {
    if (type.isNull()) return;
    if (const auto* record = vectorRecord(type)) result.insert(record->getCanonicalDecl());
    else if (type->isPointerType() || type->isReferenceType()) collect(type->getPointeeType(), result);
    else if (const auto* array = context.getAsArrayType(type)) collect(array->getElementType(), result);
    else if (const auto* fn = type->getAs<clang::FunctionType>()) {
      collect(fn->getReturnType(), result);
      if (const auto* proto = llvm::dyn_cast<clang::FunctionProtoType>(fn))
        for (const auto parameter : proto->param_types()) collect(parameter, result);
    } else if (const auto* member = type->getAs<clang::MemberPointerType>()) {
      collect(member->getPointeeType(), result);
      collect(vec3MemberClass(member, 0), result);
    }
  }
  bool contains(clang::QualType type) const {
    std::set<const clang::TagDecl*> found;
    collect(type, found);
    return !found.empty();
  }
  bool allowedScalarOrReference(clang::QualType type) const {
    if (type.isVolatileQualified() || type->isAtomicType()) return false;
    if (type->isReferenceType()) type = type->getPointeeType();
    return !type.isVolatileQualified() && vectorRecord(type) != nullptr;
  }
  bool allowedShape(clang::QualType type) const {
    if (allowedScalarOrReference(type)) return true;
    if (const auto* fn = type->getAs<clang::FunctionType>()) {
      if (contains(fn->getReturnType()) && !allowedScalarOrReference(fn->getReturnType())) return false;
      if (const auto* proto = llvm::dyn_cast<clang::FunctionProtoType>(fn))
        for (const auto parameter : proto->param_types())
          if (contains(parameter) && !allowedScalarOrReference(parameter)) return false;
      return true;
    }
    // Implicit callee decay is checked separately, not a stored pointer grant.
    return false;
  }
  bool ordinaryOwner() const {
    return lambdaDepth == 0 && (function == nullptr ||
        (function->getTemplatedKind() == clang::FunctionDecl::TK_NonTemplate &&
         !function->isDependentContext() && !llvm::isa<clang::CXXMethodDecl>(function)));
  }
  bool editable(clang::SourceLocation use) {
    if (!use.isFileID()) return false;
    if (!source.isWrittenInMainFile(use)) {
      if (!selectedHeader(use)) return false;
      plan.contextualHeaders.insert(source.getFilename(use).str());
    } else if (!literalUnobserved(use)) {
      return false;
    }
    const auto file = source.getFileID(use);
    bool invalid = false;
    const auto buffer = source.getBufferData(file, &invalid);
    if (invalid) return false;
    buffers[source.getFilename(use).str()] = bufferSha(buffer);
    bufferFiles.insert(file);
    return true;
  }
  bool validRecord(const clang::TagDecl* tag) {
    const auto* record = llvm::cast<clang::RecordDecl>(tag)->getDefinition();
    if (record == nullptr || !providers.authenticVectorDefinition(record->getLocation()) ||
        !record->getDeclContext()->isTranslationUnit() || record->isUnion()) return false;
    const auto& layout = context.getASTRecordLayout(record);
    if (layout.getSize().getQuantity() != 12 || layout.getAlignment().getQuantity() != 4) return false;
    if (const auto* cxx = llvm::dyn_cast<clang::CXXRecordDecl>(record))
      if (cxx->getNumBases() || !cxx->isStandardLayout() || !cxx->isTriviallyCopyable() ||
          !cxx->isAggregate()) return false;
    for (const auto* member : record->decls()) {
      if (llvm::isa<clang::FieldDecl>(member)) continue;
      if (const auto* injected = llvm::dyn_cast<clang::CXXRecordDecl>(member))
        if (injected->isImplicit() && injected->isInjectedClassName()) continue;
      if (const auto* special = llvm::dyn_cast<clang::FunctionDecl>(member))
        if (special->isImplicit()) continue;
      return false;
    }
    const auto scalar = record->getName() == "float3" ? context.FloatTy :
                        record->getName() == "uint3" ? context.UnsignedIntTy : context.IntTy;
    unsigned index = 0;
    const char* names[] = {"x", "y", "z"};
    for (const auto* field : record->fields()) {
      if (index == 3 || field->getName() != names[index] || field->isBitField() || field->hasInClassInitializer() ||
          field->getAccess() == clang::AS_private || field->getAccess() == clang::AS_protected ||
          !context.hasSameType(field->getType(), scalar) || layout.getFieldOffset(index) != index * 32)
        return false;
      ++index;
    }
    if (index != 3) return false;
    for (const auto* redeclaration : record->redecls()) {
      if (providers.authenticVectorDefinition(redeclaration->getLocation())) continue;
      const auto* forward = llvm::dyn_cast<clang::RecordDecl>(redeclaration);
      if (!forward || !providers.authenticBuiltinUInt3Forward(forward)) return false;
      plan.evidence.push_back("compiler-forward=uint3;provider=" +
          source.getFilename(redeclaration->getLocation()).str() +
          ";sha256=" + kClangCudaBuiltinVarsSha +
          ";included-by-owned-wrapper-sha256=" + kClangCudaRuntimeWrapperSha);
    }
    return true;
  }
  bool observe(clang::QualType type, clang::SourceLocation use) {
    if (use.isInvalid() || providers.isProvider(use)) return false;
    std::set<const clang::TagDecl*> found;
    collect(type, found);
    if (found.empty()) return false;
    observed = true;
    if (!editable(use)) fail(use, "vec3 use has no literal main/selected contextual-header rewrite path");
    for (const auto* record : found) {
      if (!validRecord(record)) {
        const auto layout = records.find(record)->second;
        fail(use, "CUDA " + record->getNameAsString() + " record ABI (size " +
            std::to_string(layout.first) + ", alignment " + std::to_string(layout.second) +
            ") has no authenticated physical SDK definition and 12/4 layout proof");
      }
      plan.uses.insert({use.getRawEncoding(), record});
    }
    return true;
  }
  bool closedInline(const clang::FunctionDecl* fn) const {
    if (fn == nullptr || !fn->isInlined() || !fn->hasBody() ||
        fn->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate || fn->isDependentContext() ||
        llvm::isa<clang::CXXMethodDecl>(fn) || fn->isExternC() || fn->isVariadic() ||
        fn->hasAttr<clang::CUDAGlobalAttr>()) return false;
    const auto* definition = fn->getDefinition();
    if (definition == nullptr || providers.isProvider(definition->getLocation())) return false;
    const auto begin = source.getExpansionLoc(definition->getBeginLoc());
    const auto end = source.getExpansionLoc(definition->getBody()->getBeginLoc());
    if (!begin.isFileID() || !end.isFileID() || source.getFileID(begin) != source.getFileID(end)) return false;
    bool invalid = false;
    const auto bytes = source.getBufferData(source.getFileID(begin), &invalid);
    if (invalid || source.getFileOffset(begin) > source.getFileOffset(end)) return false;
    // Raw Lexer needs a NUL sentinel at BufferEnd; a source slice ends at '{'.
    const std::string ownedPrefix = bytes.slice(source.getFileOffset(begin), source.getFileOffset(end)).str();
    const llvm::StringRef prefix(ownedPrefix);
    clang::Lexer lexer(clang::SourceLocation(), context.getLangOpts(), prefix.begin(), prefix.begin(), prefix.end());
    clang::Token token;
    lexer.LexFromRawLexer(token);
    while (token.isNot(clang::tok::eof)) {
      if (token.is(clang::tok::l_square) || token.is(clang::tok::r_square)) return false;
      if (token.is(clang::tok::raw_identifier)) {
        const auto name = token.getRawIdentifier();
        if (name == "__attribute__" || name == "__attribute" || name == "__declspec") return false;
      }
      lexer.LexFromRawLexer(token);
    }
    for (const auto* redeclaration : fn->redecls()) {
      const auto use = redeclaration->getLocation();
      if (!use.isFileID() || (!source.isWrittenInMainFile(use) && !selectedHeader(use))) return false;
      for (const auto* attr : redeclaration->attrs())
        if (!llvm::isa<clang::CUDAHostAttr>(attr) && !llvm::isa<clang::CUDADeviceAttr>(attr) &&
            !llvm::isa<clang::AlwaysInlineAttr>(attr)) return false;
    }
    return true;
  }
  bool factory(const clang::FunctionDecl* fn) const {
    if (fn == nullptr || fn->getIdentifier() == nullptr || !fn->getDeclContext()->isTranslationUnit() ||
        fn->getNumParams() != 3 || fn->isVariadic() ||
        fn->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate) return false;
    const auto* record = vectorRecord(fn->getReturnType());
    if (record == nullptr || fn->getNameAsString() != "make_" + record->getNameAsString()) return false;
    const auto scalar = record->getName() == "float3" ? context.FloatTy :
                        record->getName() == "uint3" ? context.UnsignedIntTy : context.IntTy;
    for (const auto* parameter : fn->parameters())
      if (parameter->hasDefaultArg() || !context.hasSameType(parameter->getType(), scalar)) return false;
    for (const auto* redeclaration : fn->redecls())
      if (!providers.authenticFactoryDeclaration(redeclaration->getLocation())) return false;
    const auto* definition = fn->getDefinition();
    if (definition == nullptr || !providers.authenticFactoryDeclaration(definition->getLocation())) return false;
    // Frozen bytes do not by themselves prove the body after preprocessing.
    // Admit only the actual three parameter-to-field assignments and return.
    const auto* body = llvm::dyn_cast<clang::CompoundStmt>(definition->getBody());
    if (body == nullptr || body->size() != 5) return false;
    auto statement = body->body_begin();
    const auto* declaration = llvm::dyn_cast<clang::DeclStmt>(*statement++);
    if (declaration == nullptr || !declaration->isSingleDecl()) return false;
    const auto* temporary = llvm::dyn_cast<clang::VarDecl>(declaration->getSingleDecl());
    if (temporary == nullptr || temporary->hasAttrs() || temporary->hasGlobalStorage() ||
        !context.hasSameType(temporary->getType(), fn->getReturnType())) return false;
    if (temporary->hasInit()) {
      const auto* init = llvm::dyn_cast<clang::CXXConstructExpr>(temporary->getInit());
      if (init == nullptr || init->getNumArgs() != 0 || !init->getConstructor()->isImplicit() ||
          !init->getConstructor()->isDefaultConstructor()) return false;
    }
    const char* fields[] = {"x", "y", "z"};
    for (unsigned index = 0; index < 3; ++index) {
      const auto* assignment = llvm::dyn_cast<clang::BinaryOperator>(*statement++);
      if (assignment == nullptr || assignment->getOpcode() != clang::BO_Assign) return false;
      const auto* member = llvm::dyn_cast<clang::MemberExpr>(assignment->getLHS()->IgnoreParenImpCasts());
      const auto* value = llvm::dyn_cast<clang::DeclRefExpr>(assignment->getRHS()->IgnoreParenImpCasts());
      const auto* base = member == nullptr ? nullptr : llvm::dyn_cast<clang::DeclRefExpr>(member->getBase()->IgnoreParenImpCasts());
      if (base == nullptr || base->getDecl() != temporary || member->getMemberNameInfo().getAsString() != fields[index] ||
          value == nullptr || value->getDecl() != definition->getParamDecl(index)) return false;
    }
    const auto* result = llvm::dyn_cast<clang::ReturnStmt>(*statement);
    if (result == nullptr || result->getRetValue() == nullptr) return false;
    const auto* value = result->getRetValue()->IgnoreParenImpCasts();
    if (const auto* copy = llvm::dyn_cast<clang::CXXConstructExpr>(value)) {
      if (copy->getNumArgs() != 1 || !copy->getConstructor()->isImplicit() ||
          !copy->getConstructor()->isCopyOrMoveConstructor()) return false;
      value = copy->getArg(0)->IgnoreParenImpCasts();
    }
    const auto* reference = llvm::dyn_cast<clang::DeclRefExpr>(value);
    return reference != nullptr && reference->getDecl() == temporary;
  }
  bool implicitRecordAssignment(const clang::FunctionDecl* fn) const {
    const auto* method = llvm::dyn_cast_or_null<clang::CXXMethodDecl>(fn);
    return method != nullptr && method->isImplicit() &&
           (method->isCopyAssignmentOperator() || method->isMoveAssignmentOperator()) &&
           records.count(method->getParent()->getCanonicalDecl()) != 0;
  }
  bool isDirectCallReference(const clang::DeclRefExpr* reference) const {
    clang::DynTypedNode node = clang::DynTypedNode::create(*reference);
    for (unsigned depth = 0; depth < 8; ++depth) {
      const auto parents = context.getParents(node);
      if (parents.size() != 1) return false;
      if (const auto* call = parents[0].get<clang::CallExpr>())
        return call->getCallee()->IgnoreParenImpCasts() == reference;
      if (!parents[0].get<clang::ImplicitCastExpr>() && !parents[0].get<clang::ParenExpr>()) return false;
      node = parents[0];
    }
    return false;
  }
  bool derivesFromVector(const clang::Expr* expression) const {
    expression = expression->IgnoreParenImpCasts();
    if (contains(expression->getType())) return true;
    if (const auto* reference = llvm::dyn_cast<clang::DeclRefExpr>(expression))
      if (const auto* field = llvm::dyn_cast<clang::FieldDecl>(reference->getDecl()))
        return records.count(field->getParent()->getCanonicalDecl()) != 0;
    // Reject the first scalar-reference binding, rather than trying to recover
    // escaped aliases later. Recurse through comma/conditional/cast/assignment
    // wrappers and aggregate initializers conservatively. A scalar value copy
    // starts independent storage and is not a vec3 reference alias.
    for (const auto* child : expression->children())
      if (const auto* nested = llvm::dyn_cast_or_null<clang::Expr>(child))
        if (derivesFromVector(nested)) return true;
    return false;
  }
  void edit(clang::SourceLocation use, llvm::StringRef spelling, const std::string& replacement) {
    if (!editable(use)) { fail(use, "vec3 token cannot be edited without observing a macro"); return; }
    clang::Token token;
    if (clang::Lexer::getRawToken(use, token, source, context.getLangOpts(), true) ||
        clang::Lexer::getSpelling(token, source, context.getLangOpts()) != spelling) {
      fail(use, "vec3 token spelling does not match its authenticated declaration"); return;
    }
    boundNames.insert(use.getRawEncoding());
    // Selected headers are reparsed at the original include position by the
    // existing input/token/macro-equivalence transaction before publication.
    if (!source.isWrittenInMainFile(use)) return;
    CudaVec3LayoutEdit item{use, token.getLength(), replacement};
    const auto prior = edits.find(use.getRawEncoding());
    if (prior != edits.end() && (prior->second.length != item.length || prior->second.replacement != replacement)) {
      fail(use, "inconsistent vec3 edit at a shared source token"); return;
    }
    edits[use.getRawEncoding()] = item;
  }
  static bool isFactoryName(llvm::StringRef name) {
    return name == "make_float3" || name == "make_uint3" || name == "make_int3";
  }
  void auditRawBindings() {
    // Include the main buffer even when every observed use is in a contextual
    // header. Dormant conditional/macro spellings cannot silently become a
    // native vec3 use after the target preprocessor changes environments.
    bufferFiles.insert(source.getMainFileID());
    for (const auto file : bufferFiles) {
      bool invalid = false;
      const auto bytes = source.getBufferData(file, &invalid);
      const auto begin = source.getLocForStartOfFile(file);
      if (invalid) { fail(begin, "vec3 source buffer is unavailable for the raw binding audit"); continue; }
      clang::Lexer lexer(begin, context.getLangOpts(), bytes.begin(), bytes.begin(), bytes.end());
      clang::Token token;
      bool previousSquare = false;
      lexer.LexFromRawLexer(token);
      while (token.isNot(clang::tok::eof)) {
        if (previousSquare && token.is(clang::tok::l_square))
          fail(token.getLocation(), "raw C++ attributes have no vec3 ABI proof");
        previousSquare = token.is(clang::tok::l_square);
        if (token.is(clang::tok::raw_identifier)) {
          const auto name = token.getRawIdentifier();
          if (name == "__attribute__" || name == "__attribute" || name == "__declspec")
            fail(token.getLocation(), "raw compiler attributes have no vec3 ABI proof");
          if (name.take_front(12) == "ascify_cuda_" || name.take_front(17) == "ascify_make_cuda_" ||
              name.take_front(17) == "ASCIFY_CUDA_VEC3_")
            fail(token.getLocation(), "raw input owns a reserved vec3 output identifier");
          if ((name == "float3" || name == "uint3" || name == "int3" || isFactoryName(name)) &&
              !boundNames.count(token.getLocation().getRawEncoding()))
            fail(token.getLocation(), "raw vec3 spelling has no authenticated active AST rewrite/binding");
        }
        lexer.LexFromRawLexer(token);
      }
    }
  }
  bool fail(clang::SourceLocation use, const std::string& reason) {
    if (error.empty()) { error = reason; location = use; }
    return true; // Continue auditing; finish discards the whole failed plan.
  }

  clang::ASTContext& context;
  clang::SourceManager& source;
  FrozenProviders& providers;
  const Records& records;
  const std::function<bool(clang::SourceLocation)>& selectedHeader;
  const std::function<bool(clang::SourceLocation)>& literalUnobserved;
  CudaVec3LayoutPlan& plan;
  std::string& error;
  clang::SourceLocation& location;
  const clang::FunctionDecl* function = nullptr;
  unsigned lambdaDepth = 0;
  bool observed = false;
  std::map<unsigned, CudaVec3LayoutEdit> edits;
  std::map<std::string, std::string> buffers;
  std::set<clang::FileID> bufferFiles;
  std::set<unsigned> boundNames;
};

}  // namespace

bool IsRedundantFrozenCubUtilInclude(
    clang::SourceManager& sourceManager,
    const std::string& configuredCudaRoot,
    const std::vector<CudaVec3MappedCubInclude>& mappedCubIncludes,
    const clang::FileEntry& skippedFile,
    clang::SourceLocation includeLocation) {
  if (!includeLocation.isFileID() ||
      !sourceManager.isWrittenInMainFile(includeLocation))
    return false;
  std::vector<CudaVec3MappedCubInclude> priorUnconditionalIncludes;
  for (const auto& mapped : mappedCubIncludes) {
    if (mapped.unconditional && mapped.hashLocation.isFileID() &&
        sourceManager.getFileID(mapped.hashLocation) ==
            sourceManager.getFileID(includeLocation) &&
        sourceManager.getFileOffset(mapped.filenameEnd) <
            sourceManager.getFileOffset(includeLocation))
      priorUnconditionalIncludes.push_back(mapped);
  }
  FrozenProviders providers(sourceManager, configuredCudaRoot,
                            priorUnconditionalIncludes);
  return providers.isRemovedUtilProvider(skippedFile);
}

bool ProveCudaVec3LocalLayout(
    clang::ASTContext& context, const std::string& configuredCudaRoot,
    const std::string& ownedClangResourceRoot,
    const std::vector<CudaVec3MappedCubInclude>& mappedCubIncludes,
    const std::function<bool(clang::SourceLocation)>& selectedHeader,
    const std::function<bool(clang::SourceLocation)>& literalUnobserved,
    CudaVec3LayoutPlan& plan, std::string& error, clang::SourceLocation& location) {
  plan = {};
  error.clear();
  FrozenProviders providers(context.getSourceManager(), configuredCudaRoot,
                            mappedCubIncludes, ownedClangResourceRoot);
  Records records;
  FindRecords find(context, providers, records);
  find.TraverseDecl(context.getTranslationUnitDecl());
  if (records.empty() || find.hasUnknownClangRuntimeProvider()) return true;
  LocalLayoutProof proof(context, providers, records, selectedHeader, literalUnobserved,
                         plan, error, location);
  proof.TraverseDecl(context.getTranslationUnitDecl());
  return proof.finish();
}

bool ValidateCudaVec3AbiBoundary(clang::ASTContext& context,
                                const std::string& configuredCudaRoot,
                                const std::vector<CudaVec3MappedCubInclude>&
                                    mappedCubIncludes, std::string& error,
                                clang::SourceLocation& location,
                                const CudaVec3LayoutPlan* plan) {
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
  AuditUses audit(providers, records, error, location, plan);
  return audit.TraverseDecl(context.getTranslationUnitDecl());
}

}  // namespace ascify
