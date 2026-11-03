#include "FrontendCompatibility.h"

#include <algorithm>
#include <filesystem>
#include <set>
#include <system_error>

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace {

namespace fs = std::filesystem;

constexpr char kAdmissionHeader[] = R"ASCIFY(#ifndef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_
#define ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_

#include <type_traits>
struct uint4;

#if defined(__CUDACC__) || defined(__CUDA__)
#define ASCIFY_FRONTEND_COMPAT_DEVICE_ __device__
#else
#define ASCIFY_FRONTEND_COMPAT_DEVICE_
#endif

namespace cooperative_groups {
class thread_block {
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static void sync();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int size();
};
ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block this_thread_block();
ASCIFY_FRONTEND_COMPAT_DEVICE_ inline void sync(const thread_block& group) {
  group.sync();
}

template <unsigned int Size, typename ParentT = void> class thread_block_tile;
namespace ascify_detail {
// Value domain only; no class/enum obtains this capability from sizeof alone.
template<class T> struct scalar64_transport_type : std::integral_constant<bool,
    (std::is_same<T, long>::value || std::is_same<T, unsigned long>::value ||
     std::is_same<T, long long>::value ||
     std::is_same<T, unsigned long long>::value ||
     std::is_same<T, double>::value) && sizeof(T) == 8> {};
// Full-warp register collectives and synchronization with runtime participation checks.
template <unsigned int Size> class tile_register_operations {
  static_assert(Size == 32, "Ascify admits only full-warp tile size 32; multi-warp subgroup synchronization is not implemented");
 protected:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ tile_register_operations();
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static constexpr unsigned int size() { return Size; }
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value, T>::type
  shfl_up(T value, unsigned int delta) const;
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value ||
      scalar64_transport_type<T>::value, T>::type
  shfl(T value, int source_rank) const;
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value ||
      scalar64_transport_type<T>::value, T>::type
  shfl_down(T value, unsigned int delta) const;
  template <typename T> ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
      std::is_same<T, unsigned int>::value || std::is_same<T, float>::value ||
      std::is_same<T, uint4>::value, T>::type
  shfl_xor(T value, unsigned int lane_mask) const;
};
// Rank/size/sync only. A mandatory source proof establishes complete-warp
// participation at every call/epoch before this projection can publish.
template <> class tile_register_operations<16> {
 protected:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ tile_register_operations();
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static constexpr unsigned int size() { return 16; }
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
};
}  // namespace ascify_detail

template <unsigned int Size, typename ParentT>
class thread_block_tile : public ascify_detail::tile_register_operations<Size> {
  static_assert(std::is_same<ParentT, thread_block>::value,
                "Ascify admits only tiles partitioned from a thread block");
  ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block_tile();
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ operator thread_block_tile<Size, void>() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int meta_group_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int meta_group_size();
};
template <unsigned int Size>
class thread_block_tile<Size, void> : public ascify_detail::tile_register_operations<Size> {
 public:
  template <typename ParentT, typename std::enable_if<
      std::is_same<ParentT, thread_block>::value, int>::type = 0>
  ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block_tile(const thread_block_tile<Size, ParentT>&);
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned int meta_group_rank() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned int meta_group_size() const;
};

template <unsigned int Size>
ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block_tile<Size, thread_block>
tiled_partition(const thread_block& parent);

template <unsigned int Size, typename ParentT>
ASCIFY_FRONTEND_COMPAT_DEVICE_ inline void
sync(const thread_block_tile<Size, ParentT>& group) {
  group.sync();
}

// Erased values may originate only from an authenticated block or static16
// tile. The mandatory whole-TU proof rejects representation observations,
// escapes, source overloads and unproved control/call/launch paths.
class thread_group final {
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_group(const thread_block&);
  template <typename ParentT, typename std::enable_if<
      std::is_same<ParentT, thread_block>::value ||
      std::is_same<ParentT, void>::value, int>::type = 0>
  ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_group(const thread_block_tile<16, ParentT>&);
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned long long size() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned long long thread_rank() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
};
ASCIFY_FRONTEND_COMPAT_DEVICE_ inline void sync(const thread_group& group) {
  group.sync();
}

// A parsing projection for a separately proven whole-block scratch domain.
// Ordinary thread_block_tile still rejects sizes above 32. Ascify's mandatory
// AST gate rejects every unproved use of the distinct types below; their
// identities may not be observed by user code, overloads, or type traits.
template <unsigned BlockSize> struct block_tile_memory {};
template <unsigned BlockSize> struct scratch_thread_block {
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static void sync();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int size();
};
template <unsigned BlockSize>
ASCIFY_FRONTEND_COMPAT_DEVICE_ scratch_thread_block<BlockSize>
this_thread_block(block_tile_memory<BlockSize>& storage);
template <unsigned BlockSize>
ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync(const scratch_thread_block<BlockSize>& group);
template <unsigned Size, unsigned BlockSize> struct scratch_thread_block_tile {
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static constexpr unsigned int size() { return Size; }
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int meta_group_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int meta_group_size();
};
template <unsigned Size, unsigned BlockSize>
ASCIFY_FRONTEND_COMPAT_DEVICE_ scratch_thread_block_tile<Size, BlockSize>
tiled_partition(const scratch_thread_block<BlockSize>& parent);

// Opaque parsing values for a separately authenticated register-only group
// contract. Mandatory whole-TU proof must reject all unproved operations,
// representation observations, aliases, copies and escapes before publication.
class grid_group {
  ASCIFY_FRONTEND_COMPAT_DEVICE_ grid_group();
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned long long thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned long long size();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ bool is_valid() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
};
ASCIFY_FRONTEND_COMPAT_DEVICE_ grid_group this_grid();
class coalesced_group {
 protected:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ coalesced_group();
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned size() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned num_threads() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned thread_rank() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned meta_group_rank() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ unsigned meta_group_size() const;
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
  template<class T> ASCIFY_FRONTEND_COMPAT_DEVICE_ T shfl(T,int) const;
};
ASCIFY_FRONTEND_COMPAT_DEVICE_ coalesced_group coalesced_threads();
template<unsigned Size,class Parent> ASCIFY_FRONTEND_COMPAT_DEVICE_
coalesced_group binary_partition(const thread_block_tile<Size,Parent>&,bool);
ASCIFY_FRONTEND_COMPAT_DEVICE_ coalesced_group binary_partition(const coalesced_group&,bool);
template<class Label> ASCIFY_FRONTEND_COMPAT_DEVICE_
coalesced_group labeled_partition(const coalesced_group&,Label);
template<class Label,unsigned Size,class Parent> ASCIFY_FRONTEND_COMPAT_DEVICE_
coalesced_group labeled_partition(const thread_block_tile<Size,Parent>&,Label);
}  // namespace cooperative_groups
#undef ASCIFY_FRONTEND_COMPAT_DEVICE_
#endif
)ASCIFY";

constexpr char kReductionHeader[] = R"ASCIFY(#ifndef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_REDUCE_H_
#define ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_REDUCE_H_
#include <cooperative_groups.h>
#if defined(__CUDACC__) || defined(__CUDA__)
#define ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_ __device__
#else
#define ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
#endif
namespace cooperative_groups {
template <typename T> struct plus {
  static_assert(std::is_same<T, int>::value || std::is_same<T, float>::value ||
                std::is_same<T, double>::value,
                "Ascify cooperative plus supports int, float and exact binary64 double");
  ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_ T operator()(T left, T right) const;
};
// The target implements a register-only all-reduction for a complete tile32.
// No arbitrary functor or generic multi-warp tile is admitted here.
template <unsigned int Size, typename ParentT, typename T>
ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
typename std::enable_if<Size == 32 && (std::is_same<T, int>::value ||
                                    std::is_same<T, float>::value ||
                                    std::is_same<T, double>::value), T>::type
reduce(const thread_block_tile<Size, ParentT>& group, T value, plus<T> operation);
// This overload is published only when the mandatory whole-TU scratch proof
// succeeds. It does not admit a generic multi-warp thread_block_tile.
template <unsigned Size, unsigned BlockSize, typename T>
ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
typename std::enable_if<std::is_same<T, int>::value ||
                        std::is_same<T, float>::value ||
                        std::is_same<T, double>::value, T>::type
reduce(const scratch_thread_block_tile<Size, BlockSize>& group, T value, plus<T> operation);

// These declarations provide inventory for the mandatory exact int/owned
// plus-or-max gate. They grant neither a generic functor nor a sync capability.
template<class T> struct greater {
  ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_ T operator()(T,T) const;
};
template<class Op> ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
int reduce(const coalesced_group&,int,Op);
}  // namespace cooperative_groups
#undef ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
#endif
)ASCIFY";

constexpr char kHostMathHeader[] = R"ASCIFY(#ifndef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_
#define ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_

// CUDA's float max(a, b) returns fmaxf(a, b), including its NaN behavior.
// Clang's CUDA wrapper exposes only the device overload. These opt-in parser
// templates admit float/float max, int/int min/max, and the two exact
// int/unsigned-int min orders; no argument is narrowed.
// Ascify rewrites proven references to target builtins or a finite helper.
namespace ascify_frontend_compat_detail {
template <bool> struct host_float_max_enabled {};
template <> struct host_float_max_enabled<true> { using type = int; };
template <bool> struct host_int_min_enabled {};
template <> struct host_int_min_enabled<true> { using type = int; };
template <bool> struct host_mixed_uint_min_enabled {};
template <> struct host_mixed_uint_min_enabled<true> { using type = int; };
template <bool> struct host_int_max_enabled {};
template <> struct host_int_max_enabled<true> { using type = int; };
}

template <class A, class B,
          typename ascify_frontend_compat_detail::host_float_max_enabled<
              __is_same(A, float) && __is_same(B, float)>::type = 0>
#if defined(__CUDACC__) || defined(__CUDA__)
__attribute__((host))
#endif
inline float max(A a, B b) {
  return __builtin_fmaxf(a, b);
}

// Use value parameters: CUDA integer min returns a value and each actual
// argument is evaluated once. The target spelling is proven separately.
template <class A, class B,
          typename ascify_frontend_compat_detail::host_int_min_enabled<
              __is_same(A, int) && __is_same(B, int)>::type = 0>
#if defined(__CUDACC__) || defined(__CUDA__)
__attribute__((host))
#endif
inline int min(A a, B b) {
  return a < b ? a : b;
}

// CUDA converts the signed operand to unsigned int before taking the minimum.
// The conditional's usual arithmetic conversion preserves that value domain.
template <class A, class B,
          typename ascify_frontend_compat_detail::host_mixed_uint_min_enabled<
              (__is_same(A, int) && __is_same(B, unsigned int)) ||
              (__is_same(A, unsigned int) && __is_same(B, int))>::type = 0>
#if defined(__CUDACC__) || defined(__CUDA__)
__attribute__((host))
#endif
inline unsigned int min(A a, B b) {
  return a < b ? a : b;
}

template <class A, class B,
          typename ascify_frontend_compat_detail::host_int_max_enabled<
              __is_same(A, int) && __is_same(B, int)>::type = 0>
#if defined(__CUDACC__) || defined(__CUDA__)
__attribute__((host))
#endif
inline int max(A a, B b) {
  return a > b ? a : b;
}

#endif
)ASCIFY";

constexpr char kProfileManifest[] =
    "schema=ascify.frontend-compat-profile.v1\n"
    "profile=ascify-admitted-v1\n"
    "file=cooperative_groups.h;bytes=8893;sha256=428a45c2107e823d08d837fb10c0e0742ba361bfe8de4db78b4cdf8de4d05aa4\n"
    "file=cooperative_groups/reduce.h;bytes=2167;sha256=5229c9dc3f909da824339a588eaab8f38f8804eb39ac35025766df06e0c38165\n"
    "file=host_math.h;bytes=2594;sha256=3b87b69a619cff6333d8fb2728d6af31a90a28fd2073077084b81ba602d70e2f\n";

static_assert(sizeof(kAdmissionHeader) - 1 == 8893,
              "admission header identity drifted");
static_assert(sizeof(kReductionHeader) - 1 == 2167,
              "reduction header identity drifted");
static_assert(sizeof(kHostMathHeader) - 1 == 2594,
              "host math header identity drifted");
static_assert(sizeof(kProfileManifest) - 1 == 393,
              "frontend profile manifest identity drifted");

struct RequiredProfileFile {
  const char* relativePath;
  std::uintmax_t bytes;
  const char* exactContent;
};

constexpr RequiredProfileFile kRequiredProfileFiles[] = {
    {"profile.manifest", 393, kProfileManifest},
    {"cooperative_groups.h", 8893, kAdmissionHeader},
    {"cooperative_groups/reduce.h", 2167, kReductionHeader},
    {"host_math.h", 2594, kHostMathHeader},
};

std::string pathString(const fs::path& path) {
  return path.lexically_normal().generic_string();
}

bool pathIsWithin(const fs::path& root, const fs::path& candidate) {
  const fs::path relative = candidate.lexically_relative(root);
  if (relative.empty() || relative.is_absolute())
    return false;
  for (const fs::path& component : relative) {
    if (component == "..")
      return false;
  }
  return true;
}

bool readFile(const fs::path& path, std::string& content) {
  auto buffer = llvm::MemoryBuffer::getFile(pathString(path));
  if (!buffer)
    return false;
  content = buffer.get()->getBuffer().str();
  return true;
}

bool regularFileWithoutSymlink(const fs::path& path) {
  std::error_code error;
  const fs::file_status linkStatus = fs::symlink_status(path, error);
  return !error && !fs::is_symlink(linkStatus) &&
         fs::is_regular_file(linkStatus);
}

bool ValidateProfileRoot(const fs::path& candidate,
                         std::string& resolved,
                         std::string& error) {
  std::error_code filesystemError;
  if (!fs::exists(candidate, filesystemError) || filesystemError)
    return false;

  const fs::path canonicalRoot = fs::canonical(candidate, filesystemError);
  if (filesystemError || !fs::is_directory(canonicalRoot, filesystemError) ||
      filesystemError) {
    error = "frontend compatibility profile is not a readable directory: " +
            pathString(candidate);
    return false;
  }

  std::set<std::string> expectedFiles;
  for (const RequiredProfileFile& required : kRequiredProfileFiles) {
    expectedFiles.insert(required.relativePath);
    const fs::path lexicalPath = canonicalRoot / required.relativePath;
    if (!regularFileWithoutSymlink(lexicalPath)) {
      error = "frontend compatibility profile is missing required regular "
              "file '" + std::string(required.relativePath) + "'";
      return false;
    }
    const fs::path canonicalFile =
        fs::canonical(lexicalPath, filesystemError);
    if (filesystemError ||
        canonicalFile != (canonicalRoot / required.relativePath)) {
      error = "frontend compatibility profile file escapes or aliases its "
              "verified root: " + std::string(required.relativePath);
      return false;
    }
    std::string content;
    if (!readFile(canonicalFile, content) ||
        content.size() != required.bytes) {
      error = "frontend compatibility profile file has an unexpected size: " +
              std::string(required.relativePath);
      return false;
    }
    if (content != required.exactContent) {
      if (std::string(required.relativePath) == "profile.manifest") {
        error = "frontend compatibility profile manifest/version is not "
                "recognized";
      } else {
        error = "frontend compatibility profile exact content mismatch for '" +
                std::string(required.relativePath) + "'";
      }
      return false;
    }
  }

  std::set<std::string> observedFiles;
  std::set<std::string> observedDirectories;
  for (fs::recursive_directory_iterator iterator(canonicalRoot,
                                                  filesystemError), end;
       !filesystemError && iterator != end; iterator.increment(filesystemError)) {
    const fs::file_status linkStatus =
        fs::symlink_status(iterator->path(), filesystemError);
    if (filesystemError)
      break;
    if (fs::is_symlink(linkStatus)) {
      error = "frontend compatibility profile contains a symlink: " +
              pathString(iterator->path());
      return false;
    }
    const fs::path relative =
        fs::relative(iterator->path(), canonicalRoot, filesystemError);
    if (filesystemError)
      break;
    if (fs::is_regular_file(linkStatus)) {
      observedFiles.insert(pathString(relative));
    } else if (fs::is_directory(linkStatus)) {
      observedDirectories.insert(pathString(relative));
    } else {
      error = "frontend compatibility profile contains an unsupported entry";
      return false;
    }
  }
  if (filesystemError) {
    error = "cannot audit frontend compatibility profile layout: " +
            filesystemError.message();
    return false;
  }
  if (observedFiles != expectedFiles) {
    error = "frontend compatibility profile layout does not match its "
            "closed manifest";
    return false;
  }
  if (observedDirectories != std::set<std::string>{"cooperative_groups"}) {
    error = "frontend compatibility profile directory layout does not match "
            "its closed manifest";
    return false;
  }

  resolved = pathString(canonicalRoot);
  return true;
}

bool ResolveFrontendCompatibilityRoot(
    const std::string& profile,
    const char* ascifyExecutable,
    std::string& resolved,
    std::string& error) {
  static int executableAnchor;
  const std::string executable = llvm::sys::fs::getMainExecutable(
      ascifyExecutable, &executableAnchor);
  llvm::SmallString<256> installed(
      llvm::sys::path::parent_path(executable));
  llvm::sys::path::append(installed,
                          ASCIFY_FRONTEND_COMPAT_INSTALL_FROM_BINDIR);
  llvm::sys::path::append(installed, profile);
  llvm::sys::path::remove_dots(installed, true);
  std::error_code filesystemError;
  const bool installedExists =
      fs::exists(fs::path(installed.str().str()), filesystemError);
  if (filesystemError) {
    error = "cannot inspect installed frontend compatibility profile: " +
            filesystemError.message();
    return false;
  }
  if (installedExists) {
    return ValidateProfileRoot(
        fs::path(installed.str().str()), resolved, error);
  }

  const fs::path canonicalExecutable =
      fs::canonical(fs::path(executable), filesystemError);
  if (filesystemError) {
    error = "cannot canonicalize ascify executable while resolving frontend "
            "compatibility data: " + filesystemError.message();
    return false;
  }
  const fs::path canonicalBuildRoot =
      fs::canonical(fs::path(ASCIFY_FRONTEND_COMPAT_BUILD_DIR),
                    filesystemError);
  if (filesystemError ||
      !pathIsWithin(canonicalBuildRoot, canonicalExecutable)) {
    error = "installed frontend compatibility profile is missing for '" +
            profile + "'";
    return false;
  }

  llvm::SmallString<256> source(ASCIFY_FRONTEND_COMPAT_SOURCE_DIR);
  llvm::sys::path::append(source, profile);
  if (ValidateProfileRoot(fs::path(source.str().str()), resolved, error))
    return true;
  if (error.empty()) {
    error = "cannot locate installed or source-tree frontend compatibility "
            "profile '" + profile + "'";
  }
  return false;
}

enum class CooperativeGroupsIncludeKind {
  Other,
  ExactAdmissionHeader,
  ExactReductionHeader,
  Unadmitted,
};

CooperativeGroupsIncludeKind classifyCooperativeGroupsInclude(
    const std::string& spelling) {
  std::string normalized = spelling;
  std::replace(normalized.begin(), normalized.end(), '\\', '/');
  if (normalized == "cooperative_groups.h")
    return CooperativeGroupsIncludeKind::ExactAdmissionHeader;
  if (normalized == "cooperative_groups/reduce.h")
    return CooperativeGroupsIncludeKind::ExactReductionHeader;
  const fs::path path(normalized);
  if (path.filename() == "cooperative_groups.h" ||
      normalized == "cooperative_groups" ||
      normalized.rfind("cooperative_groups/", 0) == 0 ||
      normalized.find("/cooperative_groups/") != std::string::npos) {
    return CooperativeGroupsIncludeKind::Unadmitted;
  }
  return CooperativeGroupsIncludeKind::Other;
}

}  // namespace

namespace ascify {

bool ResolveDynamicWarpGroupsBackend(const FrontendCompatibilityConfig& config,
                                    std::string& path, std::string& error) {
  static int anchor;
  std::error_code ec;
  const fs::path executable=fs::canonical(
      llvm::sys::fs::getMainExecutable(nullptr,&anchor),ec);
  if(ec||!config.enabled()) { error="dynamic backend has no physical product"; return false; }
  const auto installedProfile=(executable.parent_path()/
      ASCIFY_FRONTEND_COMPAT_INSTALL_FROM_BINDIR/config.profile).lexically_normal();
  fs::path candidate;
  if(installedProfile==fs::path(config.canonicalRoot))
    candidate=(executable.parent_path()/ASCIFY_COMPAT_INCLUDE_INSTALL_FROM_BINDIR/
               "ascify/dynamic_warp_groups_compat.hpp").lexically_normal();
  else {
    const auto build=fs::canonical(fs::path(ASCIFY_FRONTEND_COMPAT_BUILD_DIR),ec);
    if(ec||executable.parent_path()!=build||
        fs::path(config.canonicalRoot)!=fs::path(ASCIFY_FRONTEND_COMPAT_SOURCE_DIR)/config.profile) {
      error="dynamic backend does not belong to the selected physical product"; return false;
    }
    candidate=fs::path(ASCIFY_COMPAT_INCLUDE_SOURCE_DIR)/"ascify/dynamic_warp_groups_compat.hpp";
  }
  if(!regularFileWithoutSymlink(candidate)||fs::canonical(candidate,ec)!=candidate||ec) {
    error="dynamic backend has an unproved physical path"; return false;
  }
  path=pathString(candidate);return true;
}

bool IsVerifiedFrontendHostMathInput(
    const FrontendCompatibilityConfig& config,
    const std::string& path,
    const std::string& parsedContents) {
  return config.enabled() &&
         config.profile == kAdmittedFrontendCompatibilityV1 &&
         path == config.canonicalRoot + "/host_math.h" &&
         parsedContents == kHostMathHeader;
}

bool IsVerifiedFrontendCooperativeGroupsInput(
    const FrontendCompatibilityConfig& config,
    const std::string& path,
    const std::string& parsedContents) {
  return config.enabled() &&
         config.profile == kAdmittedFrontendCompatibilityV1 &&
         ((path == config.canonicalRoot + "/cooperative_groups.h" &&
           parsedContents == kAdmissionHeader) ||
          (path == config.canonicalRoot + "/cooperative_groups/reduce.h" &&
           parsedContents == kReductionHeader));
}

bool ConfigureFrontendCompatibility(
    clang::tooling::RefactoringTool& tool,
    const std::string& profile,
    const char* ascifyExecutable,
    FrontendCompatibilityConfig& config,
    std::string& error) {
  config = FrontendCompatibilityConfig{};
  if (profile == kNoFrontendCompatibility) {
    config.profile = kNoFrontendCompatibility;
    return true;
  }
  if (profile != kAdmittedFrontendCompatibilityV1) {
    error = "unsupported --frontend-compat value '" + profile +
            "'; expected 'none' or 'ascify-admitted-v1'";
    return false;
  }

  std::string root;
  if (!ResolveFrontendCompatibilityRoot(
          profile, ascifyExecutable, root, error)) {
    return false;
  }

  // Apply this adjuster last. Inserting at BEGIN makes this narrow product
  // header win over the CUDA installation without changing any other include.
  const std::string includeArgument = "-I" + root;
  tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(
      includeArgument.c_str(),
      clang::tooling::ArgumentInsertPosition::BEGIN));
  // This verified, explicitly selected parser surface must also apply to
  // host code that never includes cooperative_groups.h. It is not emitted
  // into translated output; proven host math calls are rewritten semantically.
  const clang::tooling::CommandLineArguments hostMathArguments{
      "-include", root + "/host_math.h"};
  tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(
      hostMathArguments, clang::tooling::ArgumentInsertPosition::END));
  config.profile = profile;
  config.canonicalRoot = root;
  return true;
}

bool ValidateFrontendCompatibilityInclude(
    const FrontendCompatibilityConfig& config,
    const std::string& includeSpelling,
    const std::string& resolvedPath,
    std::string& error) {
  if (!config.enabled())
    return true;

  const CooperativeGroupsIncludeKind kind =
      classifyCooperativeGroupsInclude(includeSpelling);
  if (kind == CooperativeGroupsIncludeKind::Other)
    return true;
  if (kind == CooperativeGroupsIncludeKind::Unadmitted) {
    error = "frontend compatibility profile '" + config.profile +
            "' rejects unadmitted cooperative-groups header '" +
            includeSpelling + "'";
    return false;
  }
  const char* admittedHeader =
      kind == CooperativeGroupsIncludeKind::ExactReductionHeader
          ? "cooperative_groups/reduce.h" : "cooperative_groups.h";
  if (resolvedPath.empty()) {
    error = "frontend compatibility profile '" + config.profile +
            "' could not prove the selected " + admittedHeader + " source";
    return false;
  }

  std::error_code filesystemError;
  const fs::path actual = fs::canonical(resolvedPath, filesystemError);
  if (filesystemError) {
    error = "frontend compatibility profile '" + config.profile +
            "' cannot canonicalize selected " + admittedHeader + ": " +
            filesystemError.message();
    return false;
  }
  const fs::path expected =
      fs::path(config.canonicalRoot) / admittedHeader;
  if (actual != expected) {
    error = "frontend compatibility profile '" + config.profile +
            "' requires " + admittedHeader + " from its verified profile; got " +
            pathString(actual);
    return false;
  }
  return true;
}

}  // namespace ascify
