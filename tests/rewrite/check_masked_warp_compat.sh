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

for direction in down up; do
for rejected_type in double int64_t uint64_t int16_t uint16_t bool; do
  cat > "$scratch_dir/reject.cpp" <<EOF
#include <ascify/masked_warp_compat.hpp>
void reject() {
  ascify::detail::converged_shfl_${direction}_sync(UINT32_MAX, $rejected_type{}, 1);
}
EOF
  reject_status=0
  "$compiler" -std=c++17 -fsyntax-only \
      -I"$test_dir/masked_warp_stubs" -I"$repo_dir/include" \
      "$scratch_dir/reject.cpp" > "$scratch_dir/reject.log" 2>&1 || reject_status=$?
  if [[ "$reject_status" == 0 || "$reject_status" -ge 128 ]]; then
    echo "unsupported shuffle type lacked a normal rejection: $direction $rejected_type status=$reject_status" >&2
    exit 1
  fi
  if ! rg -q 'Ascify converged shuffle supports int32_t, uint32_t and float only' \
      "$scratch_dir/reject.log"; then
    cat "$scratch_dir/reject.log" >&2
    exit 1
  fi
done
done
echo 'masked_warp up/down unsupported types rejected: double int64_t uint64_t int16_t uint16_t bool'

# CPU debug makes the SDK trap a no-op; our guard must still actually reject.
"$compiler" -std=c++17 -O0 -DASCENDC_CPU_DEBUG -D__NPU_ARCH__=3510 \
  -I"$test_dir/masked_warp_stubs" -I"$repo_dir/include" \
  "$test_dir/masked_warp_compat_test.cpp" -o "$scratch_dir/masked-cpu-debug"
"$scratch_dir/masked-cpu-debug"

for mode in ordinary cpu_debug; do
  extra=(-O0)
  if [[ "$mode" == cpu_debug ]]; then
    extra=(-DASCENDC_CPU_DEBUG -D__NPU_ARCH__=3510)
  fi
  "$compiler" -std=c++17 -O0 -Wall -Wextra -Werror -pedantic "${extra[@]}" \
    -I"$test_dir/masked_warp_stubs" -I"$repo_dir/include" \
    "$test_dir/masked_up_compat_test.cpp" -o "$scratch_dir/masked-up-$mode"
  "$scratch_dir/masked-up-$mode"
done

for family in legacy public85; do
  extra=(-O0)
  if [[ "$family" == public85 ]]; then
    extra=(-DASCIFY_TEST_EXPECT_PUBLIC_85 -DASCIFY_TEST_PUBLIC_85_ACL \
           -I"$test_dir/stubs_public85")
  fi
  "$compiler" -std=c++17 "${extra[@]}" -I"$test_dir/stubs" -I"$repo_dir/include" \
    "$test_dir/masked_up_routing_test.cpp" -o "$scratch_dir/up-routing-$family"
  "$scratch_dir/up-routing-$family"
done
echo 'masked_up public routing preserves full-mask/public85 and partial-mask type boundaries'


# SDK-specific routing and macro preservation remain active under NDEBUG and
# ASCENDC_DUMP=0; the fake SDK only records calls, not actual device termination.
for macro_state in 0 1; do
  "$compiler" -std=c++17 -DNDEBUG -DASCENDC_DUMP=0 \
    -DASCIFY_TEST_ASSERT_DEFINED="$macro_state" \
    -I"$test_dir/stubs" -I"$repo_dir/include" \
    "$test_dir/device_contract_reject_host_test.cpp" -o "$scratch_dir/reject-routing"
  "$scratch_dir/reject-routing"
done
echo 'SDK rejection routing active with NDEBUG; assertion macros preserved'
