#!/usr/bin/env bash
set -euo pipefail

test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$test_dir/../.." && pwd)
scratch_dir=$(mktemp -d)
trap 'rm -rf -- "$scratch_dir"' EXIT
compiler=${CXX:-c++}

"$compiler" -std=c++17 -O0 -Wall -Wextra -Werror -pedantic \
  -I"$test_dir/masked_warp_stubs" -I"$repo_dir/include" \
  "$test_dir/masked_warp_compat_test.cpp" -o "$scratch_dir/masked_warp_test"
"$scratch_dir/masked_warp_test"

for rejected_type in double int64_t uint64_t; do
  cat > "$scratch_dir/reject.cpp" <<EOF
#include <ascify/masked_warp_compat.hpp>
void reject() {
  ascify::detail::converged_shfl_down_sync(UINT32_MAX, $rejected_type{}, 1);
}
EOF
  if "$compiler" -std=c++17 -fsyntax-only \
      -I"$test_dir/masked_warp_stubs" -I"$repo_dir/include" \
      "$scratch_dir/reject.cpp" > "$scratch_dir/reject.log" 2>&1; then
    echo "unsupported shuffle type was admitted: $rejected_type" >&2
    exit 1
  fi
  if ! rg -q 'Ascify converged shuffle supports int32_t, uint32_t and float only' \
      "$scratch_dir/reject.log"; then
    cat "$scratch_dir/reject.log" >&2
    exit 1
  fi
done
echo 'masked_warp unsupported types rejected: double int64_t uint64_t'
