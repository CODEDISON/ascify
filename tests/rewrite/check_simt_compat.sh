#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
device_map="$repo_root/src/CUDA2DPP_Device_API.cpp"
runtime_map="$repo_root/src/CUDA2DPP_Runtime_API_functions.cpp"
runtime_type_map="$repo_root/src/CUDA2DPP_Runtime_API_types.cpp"
action_cpp="$repo_root/src/AscifyAction.cpp"
action_header="$repo_root/src/AscifyAction.h"
argparse_cpp="$repo_root/src/ArgParse.cpp"
argparse_header="$repo_root/src/ArgParse.h"
compat_header="$repo_root/include/ascify/ascify_cuda_compat.hpp"
runtime_memory_test="$repo_root/tests/rewrite/runtime_memory_compat_test.cpp"
runtime_lifecycle_test="$repo_root/tests/rewrite/runtime_lifecycle_compat_test.cpp"
header_family_test="$repo_root/tests/rewrite/simt_header_family_compat_test.cpp"
fail_closed_test="$repo_root/tests/rewrite/simt_fail_closed_test.cpp"
global_atomic_test="$repo_root/tests/rewrite/global_atomic_compat_test.cpp"
global_atomic_fail_test="$repo_root/tests/rewrite/global_atomic_fail_closed_test.cpp"
golden_input="$repo_root/tests/rewrite/simt_compat_input.cu"
golden_rules="$repo_root/tests/rewrite/simt_compat.expected"
atomic_golden_input="$repo_root/tests/rewrite/global_atomic_input.cu"
atomic_golden_rules="$repo_root/tests/rewrite/global_atomic.expected"

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
  require_fixed 'm["__shfl_xor_sync"]   = {"ascify::shfl_xor_sync"' "$device_map"
  require_fixed 'm["__syncthreads"]     = {"ascify::syncthreads"' "$device_map"
  require_fixed 'm["__syncthreads_and"] = {"ascify::syncthreads_and"' "$device_map"
  require_fixed 'm["__syncthreads_or"]  = {"ascify::syncthreads_or"' "$device_map"
  require_fixed 'm["__syncthreads_count"] = {"ascify::syncthreads_count"' "$device_map"
  require_fixed 'm["__threadfence_block"] = {"ascify::threadfence_block"' "$device_map"
  require_fixed 'm["__threadfence"]       = {"ascify::threadfence"' "$device_map"
  require_fixed 'm["__threadfence_system"] = {"ascify::threadfence_system"' "$device_map"
  require_fixed 'm["__global__"]         = {"ASCIFY_GLOBAL"' "$device_map"
  require_fixed 'm["__expf"]            = {"expf"' "$device_map"
  require_fixed 'm["__frsqrt_rn"]        = {"rsqrtf"' "$device_map"
  require_fixed 'm["__align__"]          = {"ASCIFY_ALIGN"' "$device_map"
  require_fixed 'm["__syncwarp"]        = {"ascify::syncwarp"' "$device_map"
  require_fixed 'm["cudaDeviceGetAttribute"]' "$runtime_map"
  require_fixed 'm["cudaMalloc"]' "$runtime_map"
  require_fixed '{"ascify::cudaMalloc"' "$runtime_map"
  require_fixed '{"ascify::cudaMallocHost"' "$runtime_map"
  require_fixed '{"ascify::cudaFreeHost"' "$runtime_map"
  require_fixed '{"ascify::cudaMemcpy"' "$runtime_map"
  require_fixed '{"ascify::cudaMemcpyAsync"' "$runtime_map"
  require_fixed '{"ascify::cudaMemset"' "$runtime_map"
  require_fixed '{"ascify::cudaMemsetAsync"' "$runtime_map"
  require_fixed '{"ascify::cudaGetLastError"' "$runtime_map"
  require_fixed '{"ascify::cudaGetErrorString"' "$runtime_map"
  require_fixed '{"ascify::cudaGetDevice"' "$runtime_map"
  require_fixed '{"ascify::cudaSetDevice"' "$runtime_map"
  require_fixed '{"ascify::cudaGetDeviceCount"' "$runtime_map"
  require_fixed '{"ascify::cudaDeviceSynchronize"' "$runtime_map"
  require_fixed '{"ascify::cudaDeviceReset"' "$runtime_map"
  require_fixed '{"ascify::cudaStreamCreate"' "$runtime_map"
  require_fixed '{"ascify::cudaStreamDestroy"' "$runtime_map"
  require_fixed '{"ascify::cudaStreamSynchronize"' "$runtime_map"
  require_fixed '{"ascify::cudaEventCreate"' "$runtime_map"
  require_fixed '{"ascify::cudaEventDestroy"' "$runtime_map"
  require_fixed '{"ascify::cudaEventRecord"' "$runtime_map"
  require_fixed '{"ascify::cudaEventSynchronize"' "$runtime_map"
  require_fixed '{"ascify::cudaEventElapsedTime"' "$runtime_map"
  require_fixed '{"ascify::cudaDeviceGetAttribute"' "$runtime_map"
  require_fixed '{"ascify::cudaPeekAtLastError"' "$runtime_map"
  require_fixed '{"ascify::cudaOccupancyMaxActiveBlocksPerMultiprocessor"' "$runtime_map"
  require_fixed '{"ascify::cudaFuncGetAttributes"' "$runtime_map"
  require_fixed '{"ascify::cudaFuncSetAttribute"' "$runtime_map"
  require_fixed 'SEC::OCCUPANCY, FULL' "$runtime_map"
  require_fixed 'm["cudaDevAttrMaxSharedMemoryPerBlockOptin"]' "$runtime_type_map"
  require_fixed 'm["cudaDevAttrWarpSize"]' "$runtime_type_map"
  require_fixed '{"ascify::cudaDevAttrWarpSize"' "$runtime_type_map"
  require_fixed 'm["cudaDevAttrMaxThreadsPerMultiProcessor"]' "$runtime_type_map"
  require_fixed '{"ascify::cudaDevAttrMaxThreadsPerVectorCore"' "$runtime_type_map"
  require_fixed 'm["cudaDevAttrMaxThreadsPerBlock"]' "$runtime_type_map"
  require_fixed '{"ascify::cudaDevAttrMaxThreadsPerBlock"' "$runtime_type_map"
  require_fixed '{"ascify::cudaDevAttrLocalMemoryPerVectorCore"' "$runtime_type_map"
  require_fixed 'm["cudaFuncAttributes"]' "$runtime_type_map"
  require_fixed 'm["cudaMemcpyKind"]' "$runtime_type_map"
  require_fixed '{"aclrtMemcpyKind"' "$runtime_type_map"
  require_fixed '#include <ascify/ascify_cuda_compat.hpp>' "$action_cpp"
  require_fixed 'NoLowerDeviceDoubleParams' "$argparse_header"
  require_fixed '"no-lower-device-double-params"' "$argparse_cpp"
  require_fixed 'cl::init(false)' "$argparse_cpp"
  require_fixed 'std::string(NoLowerDeviceDoubleParams.ArgStr)' "$argparse_cpp"
  require_fixed '"target-policy"' "$argparse_cpp"
  require_fixed 'cl::init("portable")' "$argparse_cpp"
  require_fixed '"simt-math"' "$argparse_cpp"
  require_fixed 'cl::init("precise")' "$argparse_cpp"
  require_fixed 'extern cl::opt<std::string> TargetRecipe' "$argparse_header"
  require_fixed '"target-recipe"' "$argparse_cpp"
  require_fixed 'cl::init("none")' "$argparse_cpp"
  require_fixed 'std::string(TargetRecipe.ArgStr)' "$argparse_cpp"
  require_fixed 'sCudaGlobalScalarDoubleParam' "$action_cpp"
  require_fixed 'sProvenGlobalAtomicCall' "$action_cpp"
  require_fixed 'lowerCudaGlobalScalarDoubleParam' "$action_cpp"
  require_fixed 'rewriteProvenGlobalAtomicCall' "$action_cpp"
  require_fixed 'isAddressRootedAtCurrentGlobalParameter' "$action_cpp"
  require_fixed 'clang::ParmVarDecl' "$action_cpp"
  require_fixed 'enclosingNonLambdaFunction' "$action_cpp"
  require_fixed 'ASCIFY_GLOBAL' "$device_map"
  require_fixed 'clang::CUDAGlobalAttr' "$action_cpp"
  require_fixed 'clang::CUDADeviceAttr' "$action_cpp"
  require_fixed 'clang::BuiltinTypeLoc' "$action_cpp"
  require_fixed 'getNodeAs<clang::ParmVarDecl>' "$action_cpp"
  require_fixed 'loweredDeviceDoubleParamOffsets' "$action_header"
  require_fixed 'rewrittenGlobalAtomicOffsets' "$action_header"
  require_fixed 'CONV_DEVICE_TYPE' "$action_cpp"
  require_fixed 'lowering by-value scalar double parameter' "$action_cpp"
  require_fixed 'name == "cudaMalloc"' "$action_cpp"
  require_fixed 'name == "cudaMemcpy"' "$action_cpp"
  require_fixed 'name == "cudaGetLastError"' "$action_cpp"
  require_fixed 'name == "cudaDevAttrWarpSize"' "$action_cpp"
  require_fixed 'name == "cudaDevAttrMaxThreadsPerMultiProcessor"' "$action_cpp"
  require_fixed 'name == "cudaDevAttrMaxThreadsPerBlock"' "$action_cpp"
  require_fixed 'name == "cudaDevAttrMaxSharedMemoryPerBlock"' "$action_cpp"
  require_fixed 'name == "cudaDevAttrMaxSharedMemoryPerBlockOptin"' "$action_cpp"
  require_fixed 'needsCudaCompatHeader = true;' "$action_cpp"
  forbid_fixed 'if (cudaLaunchKernel(Result)) return;' "$action_cpp"
  require_fixed '__global__ void NarrowScalarDoubleParams' "$golden_input"
  require_fixed 'const double eps, double scale' "$golden_input"
  require_fixed '+ASCIFY_GLOBAL void NarrowScalarDoubleParams(float* output, const float eps, float scale,' "$golden_rules"
  require_fixed '+const double* pointer, double values[])' "$golden_rules"
  require_fixed '+__aicore__ double KeepDeviceDoubleParam(double value)' "$golden_rules"
  require_fixed '+double KeepHostDoublePrecision(double value)' "$golden_rules"
  require_fixed '+std::is_same<double, double>' "$golden_rules"
  require_fixed '#include <algorithm>' "$compat_header"
  require_fixed '#include <atomic>' "$compat_header"
  require_fixed '#include <math.h>' "$compat_header"
  require_fixed '#include <stdlib.h>' "$compat_header"
  require_fixed '#include <type_traits>' "$compat_header"
  require_fixed '__has_include(<simt_api/kernel_simt_intf.h>)' "$compat_header"
  require_fixed '#include <simt_api/kernel_simt_intf.h>' "$compat_header"
  require_fixed '#define ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85 1' "$compat_header"
  require_fixed '#define ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3 1' "$compat_header"
  require_fixed '#define ASCIFY_GLOBAL __global__ __aicore__' "$compat_header"
  require_fixed '#define ASCIFY_GLOBAL __global__' "$compat_header"
  require_fixed '__has_include(<simt_api/device_atomic_functions.h>)' "$compat_header"
  require_fixed '#define ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS 1' "$compat_header"
  require_fixed 'inline const aclrtDevAttr cudaDevAttrWarpSize' "$compat_header"
  require_fixed 'static_cast<aclrtDevAttr>(202U)' "$compat_header"
  require_fixed 'inline const aclrtDevAttr cudaDevAttrMaxThreadsPerVectorCore' "$compat_header"
  require_fixed 'static_cast<aclrtDevAttr>(203U)' "$compat_header"
  require_fixed 'inline const aclrtDevAttr cudaDevAttrLocalMemoryPerVectorCore' "$compat_header"
  require_fixed 'static_cast<aclrtDevAttr>(204U)' "$compat_header"
  require_fixed 'inline const aclrtDevAttr cudaDevAttrMaxThreadsPerBlock' "$compat_header"
  require_fixed 'static_cast<aclrtDevAttr>(209U)' "$compat_header"
  require_fixed 'inline aclError cudaMalloc(void** device_pointer' "$compat_header"
  require_fixed 'class RuntimeManager' "$compat_header"
  require_fixed 'inline RuntimeManager runtime_manager;' "$compat_header"
  require_fixed 'status == ACL_ERROR_REPEAT_INITIALIZE' "$compat_header"
  require_fixed 'registerExitCleanup(&RuntimeManager::runExitCleanup)' "$compat_header"
  require_fixed 'return ::atexit(callback);' "$compat_header"
  require_fixed 'static void runExitCleanup() noexcept' "$compat_header"
  require_fixed 'static constexpr size_t kTrackedDeviceCapacity = 64U;' "$compat_header"
  require_fixed 'int32_t active_devices_[kTrackedDeviceCapacity] = {};' "$compat_header"
  require_fixed 'return rollback == ACL_SUCCESS ? ACL_ERROR_BAD_ALLOC : rollback;' "$compat_header"
  require_fixed 'inline aclError cudaRuntimeEnsureReady()' "$compat_header"
  require_fixed 'inline aclError cudaSetDevice(int device)' "$compat_header"
  require_fixed 'inline aclError cudaGetDevice(int* device)' "$compat_header"
  require_fixed 'inline aclError cudaGetDeviceCount(int* count)' "$compat_header"
  require_fixed 'inline aclError cudaDeviceSynchronize()' "$compat_header"
  require_fixed 'inline aclError cudaDeviceReset()' "$compat_header"
  require_fixed 'inline aclError cudaStreamCreate(aclrtStream* stream)' "$compat_header"
  require_fixed 'inline aclError cudaEventCreate(aclrtEvent* event)' "$compat_header"
  require_fixed 'ACL_MEM_MALLOC_HUGE_FIRST' "$compat_header"
  require_fixed 'inline aclError cudaMallocHost(void** host_pointer' "$compat_header"
  require_fixed 'inline aclError cudaMemcpy(void* destination' "$compat_header"
  require_fixed 'return aclrtMemcpy(destination, bytes, source, bytes, kind);' "$compat_header"
  require_fixed 'inline aclError cudaMemcpyAsync(void* destination' "$compat_header"
  require_fixed 'return aclrtMemcpyAsync(destination, bytes, source, bytes, kind, stream);' "$compat_header"
  require_fixed 'inline aclError cudaMemset(void* destination' "$compat_header"
  require_fixed 'return aclrtMemset(destination, bytes, value, bytes);' "$compat_header"
  require_fixed 'inline aclError cudaGetLastError()' "$compat_header"
  require_fixed 'inline const char* cudaGetErrorString(aclError error)' "$compat_header"
  require_fixed 'inline aclError cudaDeviceGetAttribute' "$compat_header"
  require_fixed 'inline aclError cudaPeekAtLastError' "$compat_header"
  require_fixed 'cudaOccupancyMaxActiveBlocksPerMultiprocessor' "$compat_header"
  require_fixed 'inline aclError cudaFuncGetAttributes' "$compat_header"
  require_fixed 'inline aclError cudaFuncSetAttribute' "$compat_header"
  require_fixed 'float warp_reduce_add(float value)' "$compat_header"
  require_fixed 'float warp_reduce_max(float value)' "$compat_header"
  require_fixed 'float warp_reduce_min(float value)' "$compat_header"
  require_fixed 'int32_t warp_reduce_add(int32_t value)' "$compat_header"
  require_fixed 'uint32_t warp_reduce_add(uint32_t value)' "$compat_header"
  require_fixed 'return numerator / denominator;' "$compat_header"
  require_fixed 'return ::fdividef(numerator, denominator);' "$compat_header"
  require_fixed 'void syncthreads()' "$compat_header"
  require_fixed 'int syncthreads_and(Predicate)' "$compat_header"
  require_fixed 'no admitted block-wide syncthreads_and API' "$compat_header"
  require_fixed 'void threadfence_system()' "$compat_header"
  require_fixed 'ASCIFY_DEFINE_GLOBAL_ATOMIC_BINARY(atomic_add_global, asc_atomic_add)' "$compat_header"
  require_fixed 'ASCIFY_DEFINE_GLOBAL_ATOMIC_UNSIGNED(atomic_inc_global, asc_atomic_inc)' "$compat_header"
  require_fixed 'ASCIFY_DEFINE_GLOBAL_ATOMIC_CAS(atomic_cas_global, asc_atomic_cas)' "$compat_header"
  require_fixed 'reinterpret_cast<__gm__ int32_t*>(address)' "$compat_header"
  require_fixed 'reinterpret_cast<__gm__ uint32_t*>(address)' "$compat_header"
  require_fixed 'no admitted global atomic API in this SIMT' "$compat_header"
  require_fixed 'if (mask != UINT32_MAX) {' "$compat_header"
  require_fixed 'if (width != 2 && width != 4 && width != 8 && width != 16 && width != 32) {' "$compat_header"
  require_fixed 'reject_device_contract();' "$compat_header"
  require_fixed '__builtin_trap();' "$compat_header"
  require_fixed 'if (bytes == 0) { return ACL_SUCCESS; }' "$compat_header"
  require_fixed 'if (device_pointer == nullptr) { return ACL_SUCCESS; }' "$compat_header"
  require_fixed '#define ASCIFY_ALIGN' "$compat_header"
  forbid_fixed 'm["double"]' "$device_map"
  forbid_fixed 'm["atomicAdd"]' "$device_map"
  forbid_fixed 'm["atomicSub"]' "$device_map"
  forbid_fixed 'm["atomicExch"]' "$device_map"
  forbid_fixed 'm["atomicMax"]' "$device_map"
  forbid_fixed 'm["atomicMin"]' "$device_map"
  forbid_fixed 'm["atomicInc"]' "$device_map"
  forbid_fixed 'm["atomicDec"]' "$device_map"
  forbid_fixed 'm["atomicCAS"]' "$device_map"
  forbid_fixed 'm["atomicAnd"]' "$device_map"
  forbid_fixed 'm["atomicOr"]' "$device_map"
  forbid_fixed 'm["atomicXor"]' "$device_map"
  forbid_fixed 'm["cudaMalloc"]                                              = {"aclrtMalloc"' "$runtime_map"
  forbid_fixed 'm["cudaMemcpy"]                                              = {"aclrtMemcpy"' "$runtime_map"
  forbid_fixed 'm["cudaGetLastError"]                                        = {"aclrtGetLastError"' "$runtime_map"
  forbid_fixed 'm["cudaGetErrorString"]                                      = {"aclGetRecentErrMsg"' "$runtime_map"
  forbid_fixed 'm["cudaGetDevice"]                                           = {"aclrtGetDevice"' "$runtime_map"
  forbid_fixed 'm["cudaSetDevice"]                                           = {"aclrtSetDevice"' "$runtime_map"
  forbid_fixed 'm["cudaGetDeviceCount"]                                      = {"aclrtGetDeviceCount"' "$runtime_map"
  forbid_fixed 'm["cudaDeviceReset"]                                         = {"aclrtResetDevice"' "$runtime_map"
  forbid_fixed 'm["free"]' "$runtime_map"
  forbid_fixed 'tryRewriteMallocDeviceAllocDecl' "$action_cpp"
  forbid_fixed 'scanMallocCallAndSemi' "$action_cpp"
  forbid_fixed '{"ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE"' "$runtime_type_map"
  forbid_fixed '{"ACL_DEV_ATTR_WARP_SIZE"' "$runtime_type_map"
  forbid_fixed '{"ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE"' "$runtime_type_map"
  forbid_fixed '{"ACL_DEV_ATTR_MAX_THREAD_PER_BLOCK"' "$runtime_type_map"
  forbid_fixed 'ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &ub_bytes' "$compat_header"
  forbid_fixed '#include <simt_api/asc_bf16.h>' "$compat_header"
  forbid_fixed '#include <mutex>' "$compat_header"
  forbid_fixed '#include <vector>' "$compat_header"
  forbid_fixed 'malloc(sizeof(ActiveDevice))' "$compat_header"
  forbid_fixed 'free(active)' "$compat_header"
  forbid_fixed 'asc_syncthreads_and(' "$compat_header"
  forbid_fixed 'asc_syncthreads_or(' "$compat_header"
  forbid_fixed 'asc_syncthreads_count(' "$compat_header"
  forbid_fixed 'ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE, &max_threads' "$compat_header"
  forbid_fixed '{"aclrtGetDeviceInfo"' "$runtime_map"
  forbid_fixed '{"aclrtPeekAtLastError"' "$runtime_map"
  require_fixed '__global__ void GlobalAtomicCoverage(' "$atomic_golden_input"
  require_fixed 'atomicAdd(&shared_value, 1)' "$atomic_golden_input"
  require_fixed '+old_values[0] = ascify::atomic_add_global' "$atomic_golden_rules"
  require_fixed '+old_values[10] = ascify::atomic_dec_global' "$atomic_golden_rules"
  require_fixed '+atomicAdd(&shared_value, 1)' "$atomic_golden_rules"
}

check_translated_output() {
  output=$1
  expected="$repo_root/tests/rewrite/simt_compat.expected"
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
  done < "$expected"
}

check_atomic_translated_output() {
  output=$1
  while IFS= read -r rule || [ -n "$rule" ]; do
    case "$rule" in
      ''|'#'*) continue ;;
      '+'*) require_fixed "${rule#?}" "$output" ;;
      '-'*) forbid_fixed "${rule#?}" "$output" ;;
      *)
        echo "invalid atomic golden rule: $rule" >&2
        exit 1
        ;;
    esac
  done < "$atomic_golden_rules"
}

check_static_contract

cxx=${CXX:-c++}
if command -v "$cxx" >/dev/null 2>&1; then
  "$cxx" -std=c++17 -fsyntax-only \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" \
    "$repo_root/tests/rewrite/compat_header_syntax.cpp"

  test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/ascify-simt-compat.XXXXXX")
  cleanup_compat_tests() {
    rm -rf -- "$test_tmp"
  }
  trap cleanup_compat_tests 0 1 2 15
  "$cxx" -std=c++17 \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" \
    "$runtime_memory_test" \
    -o "$test_tmp/runtime-memory"
  "$test_tmp/runtime-memory"

  "$cxx" -std=c++17 \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" \
    "$runtime_lifecycle_test" \
    -o "$test_tmp/runtime-lifecycle"
  "$test_tmp/runtime-lifecycle" owned
  "$test_tmp/runtime-lifecycle" borrowed
  "$test_tmp/runtime-lifecycle" failure
  "$test_tmp/runtime-lifecycle" capacity
  "$test_tmp/runtime-lifecycle" reset-cleanup
  "$test_tmp/runtime-lifecycle" registration-failure-owned
  "$test_tmp/runtime-lifecycle" registration-failure-borrowed
  "$test_tmp/runtime-lifecycle" registration-rollback-error
  for lifecycle_case in \
    late-registration-owned late-registration-borrowed \
    late-registration-reset-error late-registration-finalize-error \
    late-registration-bind-error \
    exit-order-owned exit-order-borrowed exit-order-init-only \
    exit-order-failed-bind exit-order-retry exit-order-late-failure; do
    "$test_tmp/runtime-lifecycle" "$lifecycle_case"
  done

  "$cxx" -std=c++17 \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" \
    "$header_family_test" \
    -o "$test_tmp/header-legacy"
  "$test_tmp/header-legacy"

  "$cxx" -std=c++17 \
    -DASCIFY_TEST_EXPECT_PUBLIC_85 \
    -DASCIFY_TEST_PUBLIC_85_ACL \
    -I"$repo_root/tests/rewrite/stubs_public85" \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" \
    "$header_family_test" \
    -o "$test_tmp/header-public85"
  "$test_tmp/header-public85"

  "$cxx" -std=c++17 \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" \
    "$global_atomic_test" \
    -o "$test_tmp/global-atomic-legacy"
  "$test_tmp/global-atomic-legacy"

  for fail_family in legacy public85; do
    fail_closed_case=1
    while [ "$fail_closed_case" -le 4 ]; do
      fail_log="$test_tmp/fail-closed-$fail_family-$fail_closed_case.log"
      if [ "$fail_family" = public85 ]; then
        fail_family_flags="-DASCIFY_TEST_PUBLIC_85_ACL -I$repo_root/tests/rewrite/stubs_public85"
      else
        fail_family_flags=""
      fi
      # shellcheck disable=SC2086
      if "$cxx" -std=c++17 -fsyntax-only \
        $fail_family_flags \
        -DASCIFY_FAIL_CLOSED_CASE="$fail_closed_case" \
        -I"$repo_root/tests/rewrite/stubs" \
        -I"$repo_root/include" \
        "$fail_closed_test" >"$test_tmp/fail-closed.out" 2>"$fail_log"; then
        echo "$fail_family unsupported case $fail_closed_case compiled unexpectedly" >&2
        exit 1
      fi
      case "$fail_closed_case" in
        1) require_fixed 'no admitted block-wide syncthreads_and API' "$fail_log" ;;
        2) require_fixed 'no admitted block-wide syncthreads_or API' "$fail_log" ;;
        3) require_fixed 'no admitted block-wide syncthreads_count API' "$fail_log" ;;
        4) require_fixed 'no admitted CUDA system-scope fence mapping' "$fail_log" ;;
      esac
      fail_closed_case=$((fail_closed_case + 1))
    done
  done

  atomic_fail_case=1
  while [ "$atomic_fail_case" -le 11 ]; do
    atomic_fail_log="$test_tmp/atomic-fail-closed-$atomic_fail_case.log"
    if "$cxx" -std=c++17 -fsyntax-only \
      -DASCIFY_TEST_PUBLIC_85_ACL \
      -DASCIFY_ATOMIC_FAIL_CLOSED_CASE="$atomic_fail_case" \
      -I"$repo_root/tests/rewrite/stubs_public85" \
      -I"$repo_root/tests/rewrite/stubs" \
      -I"$repo_root/include" \
      "$global_atomic_fail_test" \
      >"$test_tmp/atomic-fail-closed.out" 2>"$atomic_fail_log"; then
      echo "public 8.5 atomic case $atomic_fail_case compiled unexpectedly" >&2
      exit 1
    fi
    require_fixed 'no admitted global atomic API in this SIMT' "$atomic_fail_log"
    atomic_fail_case=$((atomic_fail_case + 1))
  done

  cleanup_compat_tests
  trap - 0 1 2 15
fi

if [ "$#" -eq 2 ] && [ "$1" = "--translated" ]; then
  check_translated_output "$2"
elif [ "$#" -eq 2 ] && [ "$1" = "--atomic-translated" ]; then
  check_atomic_translated_output "$2"
elif [ "$#" -eq 4 ] && [ "$1" = "--translated" ] && \
     [ "$3" = "--atomic-translated" ]; then
  check_translated_output "$2"
  check_atomic_translated_output "$4"
elif [ "$#" -ne 0 ]; then
  echo "usage: sh tests/rewrite/check_simt_compat.sh [--translated OUTPUT] [--atomic-translated OUTPUT]" >&2
  exit 2
fi

echo "SIMT compatibility rewrite checks passed"
