#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_build_dir="$(mktemp -d "${TMPDIR:-/tmp}/ascify-rowwise-simd.XXXXXX")"
trap 'rm -rf "${test_build_dir}"' EXIT

compiler="${CXX:-c++}"
common_flags=(
  -std=c++17
  -Wall
  -Wextra
  -Werror
  -I"${repo_root}/tests/rewrite/stubs"
  -I"${repo_root}/include"
)

"${compiler}" "${common_flags[@]}" \
  -I"${repo_root}/runtime/dav_3510/rowwise/src" \
  "${repo_root}/tests/rewrite/rowwise_simd_selector_test.cpp" \
  -o "${test_build_dir}/rowwise_simd_selector_test"
"${test_build_dir}/rowwise_simd_selector_test"

"${compiler}" "${common_flags[@]}" \
  "${repo_root}/tests/rewrite/rowwise_simd_dispatch_test.cpp" \
  -o "${test_build_dir}/rowwise_simd_dispatch_test"
"${test_build_dir}/rowwise_simd_dispatch_test"

"${compiler}" "${common_flags[@]}" \
  "${repo_root}/tests/rewrite/rowwise_hybrid_registry_test.cpp" \
  -o "${test_build_dir}/rowwise_hybrid_registry_test"
"${test_build_dir}/rowwise_hybrid_registry_test"

"${compiler}" "${common_flags[@]}" \
  -I"${repo_root}/runtime/dav_3510/rowwise/src" \
  "${repo_root}/tests/rewrite/rowwise_vector_core_query_test.cpp" \
  -o "${test_build_dir}/rowwise_vector_core_query_test"
"${test_build_dir}/rowwise_vector_core_query_test"

build_script="${repo_root}/tests/softmax_rmsnorm_950/scripts/build.sh"
run_script="${repo_root}/tests/softmax_rmsnorm_950/scripts/run_smoke.sh"
for script in "${build_script}" "${run_script}"; do
  bash -n "${script}"
  grep -F 'ROWWISE_SIMD_RUNTIME_DIR' "${script}" >/dev/null
done
for library_base in \
  libascify950_softmax_reg_recompute_v1 \
  libascify950_rmsnorm_reg_cached_v1 \
  libascify950_rmsnorm_reg_plain_rowbatch_v1 \
  libascify950_layernorm_reg_cached_v1; do
  grep -F "${library_base}" "${build_script}" >/dev/null
  grep -F "${library_base}" "${run_script}" >/dev/null
done
for script in "${build_script}" "${run_script}"; do
  grep -F 'soname_file="${runtime_dir}/${base}.so.1"' "${script}" >/dev/null
  grep -F 'readelf -d -- "${linker_file}"' "${script}" >/dev/null
  grep -F 'readlink -f -- "${soname_file}"' "${script}" >/dev/null
done
for link_name in \
  ascify950_softmax_reg_recompute_v1 \
  ascify950_rmsnorm_reg_cached_v1 \
  ascify950_rmsnorm_reg_plain_rowbatch_v1 \
  ascify950_layernorm_reg_cached_v1; do
  grep -F -- "-l${link_name}" "${build_script}" >/dev/null
done

abi_header="${repo_root}/include/ascify/target/dav_c310/rowwise_simd_abi.h"
runtime_root="${repo_root}/runtime/dav_3510/rowwise"
for symbol in \
  ascify950_softmax_reg_recompute_launch_v1 \
  ascify950_rmsnorm_reg_cached_launch_v1 \
  ascify950_rmsnorm_reg_plain_rowbatch_launch_v1 \
  ascify950_layernorm_reg_cached_launch_v1; do
  grep -F "${symbol}" "${abi_header}" >/dev/null
  grep -R -F "${symbol}" "${runtime_root}/src" >/dev/null
done
grep -F 'IsSoftmaxSimdDomain' \
  "${runtime_root}/src/softmax_reg_recompute.cce" >/dev/null
grep -F 'QueryVectorCoreCount' \
  "${runtime_root}/src/softmax_reg_recompute.cce" >/dev/null
for attribute in \
  ACL_DEV_ATTR_VECTOR_CORE_NUM \
  ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE; do
  grep -F "${attribute}" \
    "${runtime_root}/src/vector_core_query_v1.hpp" >/dev/null
done
if grep -E 'static const int cached|return 56;' \
    "${runtime_root}/src/softmax_reg_recompute.cce" >/dev/null; then
  echo "Softmax SIMD runtime caches or defaults the vector-core count" >&2
  exit 1
fi
grep -F 'IsRmsNormCachedSimdDomain' \
  "${runtime_root}/src/rmsnorm_reg_cached.cce" >/dev/null
grep -F 'IsRmsNormPlainRowBatchSimdDomain' \
  "${runtime_root}/src/rmsnorm_reg_plain_rowbatch.cce" >/dev/null
grep -F 'IsLayerNormCachedHybridDomain' \
  "${runtime_root}/src/layernorm_reg_cached.cce" >/dev/null
grep -F 'LayerNormNormalizeStoreSimtVF' \
  "${runtime_root}/src/layernorm_reg_cached.cce" >/dev/null
grep -F 'const float mean = meanSlot[0];' \
  "${runtime_root}/src/layernorm_reg_cached.cce" >/dev/null
grep -F 'const float inverseVariance = inverseSlot[0];' \
  "${runtime_root}/src/layernorm_reg_cached.cce" >/dev/null

# A selected v1 call must be genuinely mixed inside one ASC kernel.  SIMD
# owns regular row-wise math and the SIMT VF owns the adapter/store stage;
# the outer selector miss remains the only whole-operator fallback.
for mixed_source in \
  "${runtime_root}/src/softmax_reg_recompute.cce" \
  "${runtime_root}/src/rmsnorm_reg_cached.cce" \
  "${runtime_root}/src/rmsnorm_reg_plain_rowbatch.cce" \
  "${runtime_root}/src/layernorm_reg_cached.cce"; do
  grep -F '__simd_vf__' "${mixed_source}" >/dev/null
  grep -F '__simt_vf__' "${mixed_source}" >/dev/null
  grep -F 'ApplyStoreAdapterSimt' "${mixed_source}" >/dev/null
  grep -F 'KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)' \
    "${mixed_source}" >/dev/null
  if grep -F '__global__ __vector__' "${mixed_source}" >/dev/null; then
    echo "mixed runtime still publishes a vector-only global kernel: ${mixed_source}" >&2
    exit 1
  fi
done
grep -F 'static_cast<int>(blocks), kUbBytes, stream' \
  "${runtime_root}/src/softmax_reg_recompute.cce" >/dev/null
grep -F '<<<gridBlocks, ubBytes, stream>>>' \
  "${runtime_root}/src/rmsnorm_reg_cached.cce" >/dev/null
grep -F '<<<gridBlocks, kMaximumUbBytes, stream>>>' \
  "${runtime_root}/src/rmsnorm_reg_plain_rowbatch.cce" >/dev/null
grep -F '<<<gridBlocks, kMaximumUbBytes, stream>>>' \
  "${runtime_root}/src/layernorm_reg_cached.cce" >/dev/null
if grep -R -E '<<<[^>]*(nullptr|, 0,)[^>]*stream>>>' \
    "${runtime_root}/src"/*.cce >/dev/null; then
  echo "mixed runtime launches a kernel without dynamic UB" >&2
  exit 1
fi
grep -F 'struct RowwiseHybridFacadeV1 final' \
  "${repo_root}/include/ascify/target/dav_c310/rowwise_simd_recipes.hpp" \
  >/dev/null
grep -F 'TrySoftmaxHybrid' \
  "${repo_root}/include/ascify/target/dav_c310/rowwise_simd_recipes.hpp" \
  >/dev/null
grep -F 'TryRmsNormHybrid' \
  "${repo_root}/include/ascify/target/dav_c310/rowwise_simd_recipes.hpp" \
  >/dev/null
grep -F 'TryLayerNormHybrid' \
  "${repo_root}/include/ascify/target/dav_c310/rowwise_simd_recipes.hpp" \
  >/dev/null
grep -F 'DispatchRegisteredHybrid' \
  "${repo_root}/include/ascify/target/dav_c310/rowwise_hybrid_registry_v1.hpp" \
  >/dev/null

if grep -R -E \
    'ascify950_(softmax_reg_recompute|rmsnorm_reg_(cached|plain_rowbatch)|layernorm_reg_cached)_launch\(' \
    "${repo_root}/include/ascify/target/dav_c310" \
    "${runtime_root}" >/dev/null; then
  echo "rowwise SIMD v1 exposes an unversioned C ABI symbol" >&2
  exit 1
fi

grep -F 'VERSION 1.0.0' "${runtime_root}/CMakeLists.txt" >/dev/null
grep -F 'SOVERSION 1' "${runtime_root}/CMakeLists.txt" >/dev/null
grep -F 'add_library(ascify950_rowwise_simd_v1 INTERFACE)' \
  "${runtime_root}/CMakeLists.txt" >/dev/null
grep -F 'install(EXPORT Ascify950RowwiseSimdV1Targets' \
  "${runtime_root}/CMakeLists.txt" >/dev/null
grep -F 'ascify950_layernorm_reg_cached_v1' \
  "${runtime_root}/CMakeLists.txt" >/dev/null

if grep -R -E -i 'compile.?probe|candidate' \
    "${runtime_root}" >/dev/null; then
  echo "rowwise runtime contains provisional naming" >&2
  exit 1
fi

echo "rowwise SIMD+SIMT v1 selector, dispatch, and mixed-stage checks passed"
