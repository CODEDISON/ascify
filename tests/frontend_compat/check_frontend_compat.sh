#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
test_root="$repo_root/tests/frontend_compat"
profile_root="$repo_root/frontend_compat/ascify-admitted-v1"
manifest="$profile_root/profile.manifest"
header="$profile_root/cooperative_groups.h"
poison="$profile_root/cooperative_groups/reduce.h"
host_math="$profile_root/host_math.h"

search_fixed() {
  needle=$1
  file=$2
  if command -v rg >/dev/null 2>&1; then
    rg -F -- "$needle" "$file"
  else
    grep -F -- "$needle" "$file"
  fi
}

require_fixed() {
  needle=$1
  file=$2
  if ! search_fixed "$needle" "$file" >/dev/null; then
    echo "missing '$needle' in $file" >&2
    exit 1
  fi
}

forbid_fixed() {
  needle=$1
  file=$2
  if search_fixed "$needle" "$file" >/dev/null; then
    echo "forbidden '$needle' found in $file" >&2
    exit 1
  fi
}

sha256_file() {
  file=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    echo "SHA-256 utility not found" >&2
    exit 1
  fi
}

require_fixed 'class thread_block {' "$header"
require_fixed 'void sync() const;' "$header"
require_fixed 'thread_block this_thread_block();' "$header"
require_fixed 'inline void sync(const thread_block& group)' "$header"
require_fixed 'group.sync();' "$header"
for forbidden in \
  'coalesced_group' 'reduce' 'ballot' \
  '__shared__' 'dynamic_shared' 'nv/target'; do
  forbid_fixed "$forbidden" "$header"
done
require_fixed \
  'does not admit cooperative_groups/reduce.h' "$poison"
require_fixed 'schema=ascify.frontend-compat-profile.v1' "$manifest"
require_fixed 'profile=ascify-admitted-v1' "$manifest"
require_fixed \
  'file=cooperative_groups.h;bytes=2481;sha256=3e3061687252e956483540e6d9f78364200dc6de07dd7dc5e103a5ae06311a04' \
  "$manifest"
require_fixed \
  'file=cooperative_groups/reduce.h;bytes=101;sha256=75adbe65aeb5c2acfd63c9896376e67d270198e566080b9260a219ab99e2de8a' \
  "$manifest"
require_fixed \
  'file=host_math.h;bytes=1458;sha256=a151400e55d4f484f8321b5d98384f3585fed01678cda680354cd331a664558d' \
  "$manifest"
if [ "$(sha256_file "$header")" != \
    '3e3061687252e956483540e6d9f78364200dc6de07dd7dc5e103a5ae06311a04' ]; then
  echo "frontend admission header SHA-256 mismatch" >&2
  exit 1
fi
if [ "$(sha256_file "$poison")" != \
    '75adbe65aeb5c2acfd63c9896376e67d270198e566080b9260a219ab99e2de8a' ]; then
  echo "frontend reduction poison SHA-256 mismatch" >&2
  exit 1
fi
if [ "$(sha256_file "$host_math")" != \
    'a151400e55d4f484f8321b5d98384f3585fed01678cda680354cd331a664558d' ]; then
  echo "frontend host math header SHA-256 mismatch" >&2
  exit 1
fi

profile_files=$(find "$profile_root" -type f -print | LC_ALL=C sort)
expected_files=$(printf '%s\n' \
  "$header" \
  "$poison" \
  "$host_math" \
  "$manifest" | LC_ALL=C sort)
if [ "$profile_files" != "$expected_files" ]; then
  echo "frontend compatibility profile contains an unmanifested file" >&2
  exit 1
fi
if find "$profile_root" -type l -print | grep . >/dev/null; then
  echo "frontend compatibility profile contains a symlink" >&2
  exit 1
fi

require_fixed 'cl::opt<std::string> FrontendCompat("frontend-compat"' \
  "$repo_root/src/ArgParse.cpp"
require_fixed 'cl::init("none")' "$repo_root/src/ArgParse.cpp"
require_fixed 'std::string(FrontendCompat.ArgStr)' "$repo_root/src/ArgParse.cpp"
require_fixed 'profile != kAdmittedFrontendCompatibilityV1' \
  "$repo_root/src/FrontendCompatibility.cpp"
require_fixed 'ArgumentInsertPosition::BEGIN' \
  "$repo_root/src/FrontendCompatibility.cpp"
require_fixed 'ValidateProfileRoot' \
  "$repo_root/src/FrontendCompatibility.cpp"
require_fixed 'ValidateFrontendCompatibilityInclude' \
  "$repo_root/src/AscifyAction.cpp"
require_fixed 'observedFiles != expectedFiles' \
  "$repo_root/src/FrontendCompatibility.cpp"
require_fixed 'exact content mismatch' \
  "$repo_root/src/FrontendCompatibility.cpp"
require_fixed 'ASCIFY_FRONTEND_COMPAT_BUILD_DIR' \
  "$repo_root/src/FrontendCompatibility.cpp"
require_fixed 'ASCIFY_FRONTEND_COMPAT_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}"' \
  "$repo_root/CMakeLists.txt"
require_fixed 'DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/frontend_compat/' \
  "$repo_root/CMakeLists.txt"

cxx=${CXX:-c++}
if ! command -v "$cxx" >/dev/null 2>&1; then
  echo "C++ compiler not found; frontend compatibility syntax checks skipped" >&2
  exit 0
fi

test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/ascify-frontend-compat.XXXXXX")
cleanup() {
  case "$test_tmp" in
    "${TMPDIR:-/tmp}"/ascify-frontend-compat.*)
      rm -rf -- "$test_tmp"
      ;;
    *)
      echo "refusing to remove unexpected work directory: $test_tmp" >&2
      ;;
  esac
}
trap cleanup 0 1 2 15

"$cxx" -std=c++17 -fsyntax-only -I"$profile_root" \
  "$test_root/block_sync_positive.cpp"

"$cxx" -std=c++17 -fsyntax-only -I"$profile_root" \
  "$test_root/tile32_positive.cpp"
for reject_case in 1 2 3 4 5 6 7; do
  if "$cxx" -std=c++17 -fsyntax-only -I"$profile_root" \
      -DASCIFY_TILE_REJECT_CASE="$reject_case" \
      "$test_root/tile32_reject.cpp" \
      >"$test_tmp/tile-$reject_case.stdout" 2>"$test_tmp/tile-$reject_case.stderr"; then
    echo "unsupported tile operation $reject_case compiled unexpectedly" >&2
    exit 1
  fi
done

for negative in generic_sync_reject tile_name_reject; do
  if "$cxx" -std=c++17 -fsyntax-only -I"$profile_root" \
      "$test_root/$negative.cpp" \
      >"$test_tmp/$negative.stdout" 2>"$test_tmp/$negative.stderr"; then
    echo "$negative compiled unexpectedly" >&2
    exit 1
  fi
done

if "$cxx" -std=c++17 -fsyntax-only -I"$profile_root" \
    "$test_root/reduce_header_reject.cpp" \
    >"$test_tmp/reduce.stdout" 2>"$test_tmp/reduce.stderr"; then
  echo "cooperative_groups/reduce.h compiled unexpectedly" >&2
  exit 1
fi
require_fixed 'does not admit cooperative_groups/reduce.h' \
  "$test_tmp/reduce.stderr"

echo "frontend compatibility admission checks passed"
