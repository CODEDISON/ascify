#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
header="$repo_root/include/ascify/cooperative_groups_compat.hpp"
include_map="$repo_root/src/CUDA2DPP.cpp"
positive_test="$repo_root/tests/rewrite/cooperative_groups_header_syntax.cpp"
negative_test="$repo_root/tests/rewrite/cooperative_groups_reject_sync.cpp"
public_test="$repo_root/tests/rewrite/cooperative_groups_public85_fail_closed.cpp"
legacy_stubs="$repo_root/tests/rewrite/cooperative_stubs/legacy"
public_stubs="$repo_root/tests/rewrite/cooperative_stubs/public85"
golden_rules="$repo_root/tests/rewrite/cooperative_groups_compat.expected"

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

check_static_contract() {
  require_fixed '{"cooperative_groups.h"' "$include_map"
  require_fixed '{"ascify/cooperative_groups_compat.hpp"' "$include_map"
  require_fixed '{"cooperative_groups/reduce.h"' "$include_map"

  require_fixed '#include <simt_api/cooperative_groups.h>' "$header"
  require_fixed '__SIMT_DEVICE_FUNCTIONS_DECL__ inline void sync(const thread_block& group)' "$header"
  require_fixed 'group.sync();' "$header"
  require_fixed 'public CANN 8.5: no native cooperative_groups header' "$header"
  require_fixed 'namespace ascify_cg {' "$header"
  require_fixed 'static unsigned int thread_rank()' "$header"
  require_fixed 'static unsigned int meta_group_rank()' "$header"
  require_fixed 'Size == 32' "$header"
  forbid_fixed 'coalesced_group' "$header"
  require_fixed 'reduce(const thread_block_tile<Size, ParentT>&' "$header"
  forbid_fixed 'ballot' "$header"
  forbid_fixed '__shared__' "$header"
}

check_translated_output() {
  output=$1
  while IFS= read -r rule || [ -n "$rule" ]; do
    case "$rule" in
      ''|'#'*) continue ;;
      '+'*) require_fixed "${rule#?}" "$output" ;;
      '-'*) forbid_fixed "${rule#?}" "$output" ;;
      *)
        echo "invalid golden rule: $rule" >&2
        exit 1
        ;;
    esac
  done < "$golden_rules"
}

translated=
if [ "${1:-}" = "--translated" ]; then
  if [ "$#" -ne 2 ]; then
    echo "usage: $0 [--translated output.cce]" >&2
    exit 2
  fi
  translated=$2
elif [ "$#" -ne 0 ]; then
  echo "usage: $0 [--translated output.cce]" >&2
  exit 2
fi

check_static_contract

cxx=${CXX:-c++}
if command -v "$cxx" >/dev/null 2>&1; then
  test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/ascify-cooperative-groups.XXXXXX")
  cleanup_cooperative_tests() {
    case "$test_tmp" in
      "${TMPDIR:-/tmp}"/ascify-cooperative-groups.*)
        rm -rf -- "$test_tmp"
        ;;
      *)
        echo "refusing to remove unexpected work directory: $test_tmp" >&2
        ;;
    esac
  }
  trap cleanup_cooperative_tests 0 1 2 15

  "$cxx" -std=c++17 -fsyntax-only \
    -I"$legacy_stubs" \
    -I"$repo_root/include" \
    "$positive_test"

  "$cxx" -std=c++17 -O2 \
    -I"$legacy_stubs" -I"$repo_root/include" \
    "$repo_root/tests/rewrite/cooperative_groups_uint4_test.cpp" \
    -o "$test_tmp/uint4-shuffle"
  "$test_tmp/uint4-shuffle"

  "$cxx" -std=c++17 -O2 -pthread \
    -I"$legacy_stubs" -I"$repo_root/include" \
    "$repo_root/tests/rewrite/cooperative_groups_reduce_host_test.cpp" \
    -o "$test_tmp/reduce-host"
  "$test_tmp/reduce-host"

  reject_case=1
  while [ "$reject_case" -le 2 ]; do
    reject_log="$test_tmp/reject-sync-$reject_case.log"
    if "$cxx" -std=c++17 -fsyntax-only \
      -DASCIFY_REJECT_SYNC_CASE="$reject_case" \
      -I"$legacy_stubs" \
      -I"$repo_root/include" \
      "$negative_test" >"$test_tmp/reject-sync.out" 2>"$reject_log"; then
      echo "unsupported cooperative group sync case $reject_case compiled unexpectedly" >&2
      exit 1
    fi
    reject_case=$((reject_case + 1))
  done

  public_log="$test_tmp/public85.log"
  if "$cxx" -std=c++17 -fsyntax-only \
    -I"$public_stubs" \
    -I"$repo_root/include" \
    "$public_test" >"$test_tmp/public85.out" 2>"$public_log"; then
    echo "public CANN 8.5 cooperative-groups case compiled unexpectedly" >&2
    exit 1
  fi
  require_fixed \
    'public CANN 8.5: no native cooperative_groups header' \
    "$public_log"
else
  echo "C++ compiler not found; cooperative-groups host syntax checks skipped" >&2
fi

if [ -n "$translated" ]; then
  check_translated_output "$translated"
fi

echo "cooperative-groups compatibility checks passed"
