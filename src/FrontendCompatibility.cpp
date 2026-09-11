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

#if defined(__CUDACC__) || defined(__CUDA__)
#define ASCIFY_FRONTEND_COMPAT_DEVICE_ __device__
#else
#define ASCIFY_FRONTEND_COMPAT_DEVICE_
#endif

namespace cooperative_groups {

class thread_block {
 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ void sync() const;
};

ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block this_thread_block();

ASCIFY_FRONTEND_COMPAT_DEVICE_ inline void sync(const thread_block& group) {
  group.sync();
}

// Only the full-warp, register-only tile surface verified against CANN 9.1.
// Tile synchronization is deliberately absent: the native tile sync is only
// a block memory fence, which is not proof of CUDA's collective barrier.
template <unsigned int Size, typename ParentT = void>
class thread_block_tile {
  static_assert(Size == 32, "Ascify admits only full-warp tile size 32");
  static_assert(std::is_same<ParentT, thread_block>::value,
                "Ascify admits only tiles partitioned from a thread block");
  ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block_tile();

 public:
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int thread_rank();
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static constexpr unsigned int size() { return Size; }
  ASCIFY_FRONTEND_COMPAT_DEVICE_ static unsigned int meta_group_rank();

  template <typename T>
  ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
                              std::is_same<T, unsigned int>::value ||
                              std::is_same<T, float>::value, T>::type
  shfl_up(T value, unsigned int delta) const;

  template <typename T>
  ASCIFY_FRONTEND_COMPAT_DEVICE_
  typename std::enable_if<std::is_same<T, int>::value ||
                              std::is_same<T, unsigned int>::value ||
                              std::is_same<T, float>::value, T>::type
  shfl_xor(T value, unsigned int lane_mask) const;
};

template <unsigned int Size>
ASCIFY_FRONTEND_COMPAT_DEVICE_ thread_block_tile<Size, thread_block>
tiled_partition(const thread_block& parent);

}  // namespace cooperative_groups

#undef ASCIFY_FRONTEND_COMPAT_DEVICE_

#endif  // ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_COOPERATIVE_GROUPS_H_
)ASCIFY";

constexpr char kReductionPoison[] =
    "#error \"Ascify frontend compatibility ascify-admitted-v1 does not "
    "admit cooperative_groups/reduce.h\"\n";

constexpr char kHostMathHeader[] = R"ASCIFY(#ifndef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_
#define ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_

// CUDA's float max(a, b) returns fmaxf(a, b), including its NaN behavior.
// Clang's CUDA wrapper exposes only the device overload. This opt-in parser
// template admits exactly float/float host calls; no argument is narrowed.
// Ascify rewrites references proven to use this declaration to the builtin.
namespace ascify_frontend_compat_detail {
template <bool> struct host_float_max_enabled {};
template <> struct host_float_max_enabled<true> { using type = int; };
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

#endif
)ASCIFY";

constexpr char kProfileManifest[] =
    "schema=ascify.frontend-compat-profile.v1\n"
    "profile=ascify-admitted-v1\n"
    "file=cooperative_groups.h;bytes=2291;sha256=1d490bd085dbd3e339742854773ef57e73049f7d1ee2d83d02c573a3709366c4\n"
    "file=cooperative_groups/reduce.h;bytes=101;sha256=75adbe65aeb5c2acfd63c9896376e67d270198e566080b9260a219ab99e2de8a\n"
    "file=host_math.h;bytes=900;sha256=fe115c0ee5be69d87f09e53b5a0f9f7649b468c1ceaf03fdb9df993052a5076c\n";

static_assert(sizeof(kAdmissionHeader) - 1 == 2291,
              "admission header identity drifted");
static_assert(sizeof(kReductionPoison) - 1 == 101,
              "reduction poison identity drifted");
static_assert(sizeof(kHostMathHeader) - 1 == 900,
              "host math header identity drifted");
static_assert(sizeof(kProfileManifest) - 1 == 391,
              "frontend profile manifest identity drifted");

struct RequiredProfileFile {
  const char* relativePath;
  std::uintmax_t bytes;
  const char* exactContent;
};

constexpr RequiredProfileFile kRequiredProfileFiles[] = {
    {"profile.manifest", 391, kProfileManifest},
    {"cooperative_groups.h", 2291, kAdmissionHeader},
    {"cooperative_groups/reduce.h", 101, kReductionPoison},
    {"host_math.h", 900, kHostMathHeader},
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
  llvm::sys::path::append(installed, "..");
  llvm::sys::path::append(installed,
                          ASCIFY_FRONTEND_COMPAT_INSTALL_RELPATH);
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
  Unadmitted,
};

CooperativeGroupsIncludeKind classifyCooperativeGroupsInclude(
    const std::string& spelling) {
  std::string normalized = spelling;
  std::replace(normalized.begin(), normalized.end(), '\\', '/');
  if (normalized == "cooperative_groups.h")
    return CooperativeGroupsIncludeKind::ExactAdmissionHeader;
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
  // into translated output; proven float calls are rewritten semantically.
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
  if (resolvedPath.empty()) {
    error = "frontend compatibility profile '" + config.profile +
            "' could not prove the selected cooperative_groups.h source";
    return false;
  }

  std::error_code filesystemError;
  const fs::path actual = fs::canonical(resolvedPath, filesystemError);
  if (filesystemError) {
    error = "frontend compatibility profile '" + config.profile +
            "' cannot canonicalize selected cooperative_groups.h: " +
            filesystemError.message();
    return false;
  }
  const fs::path expected =
      fs::path(config.canonicalRoot) / "cooperative_groups.h";
  if (actual != expected) {
    error = "frontend compatibility profile '" + config.profile +
            "' requires cooperative_groups.h from its verified profile; got " +
            pathString(actual);
    return false;
  }
  return true;
}

}  // namespace ascify
