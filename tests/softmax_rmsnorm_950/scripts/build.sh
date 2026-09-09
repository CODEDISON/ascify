#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TEST_ROOT}/../.." && pwd)"

WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/.work/softmax_rmsnorm_950}"
GENERATED_ROOT="${GENERATED_ROOT:-${WORK_ROOT}/generated}"
BIN_DIR="${BIN_DIR:-${WORK_ROOT}/bin}"
PROBE_OBJ_DIR="${PROBE_OBJ_DIR:-${WORK_ROOT}/probes}"
CANN_ROOT="${CANN_ROOT:-}"
ROWWISE_SIMD_RUNTIME_DIR="${ROWWISE_SIMD_RUNTIME_DIR:-}"

validate_rowwise_runtime_dir() {
  local runtime_dir="$1"
  local base
  local linker_file
  local soname_file
  local actual_soname
  if ! command -v readelf >/dev/null 2>&1; then
    echo "readelf is required to validate the row-wise SIMD runtime" >&2
    return 1
  fi
  for base in \
    libascify950_softmax_reg_recompute_v1 \
    libascify950_rmsnorm_reg_cached_v1 \
    libascify950_rmsnorm_reg_plain_rowbatch_v1 \
    libascify950_layernorm_reg_cached_v1; do
    linker_file="${runtime_dir}/${base}.so"
    soname_file="${runtime_dir}/${base}.so.1"
    if [[ ! -f "${linker_file}" || ! -f "${soname_file}" ]]; then
      echo "missing row-wise SIMD linker-name or SONAME file:" >&2
      echo "  ${linker_file}" >&2
      echo "  ${soname_file}" >&2
      return 1
    fi
    actual_soname="$(
      readelf -d -- "${linker_file}" |
        sed -n 's/^.*(SONAME).*\[\([^]]*\)\].*$/\1/p'
    )"
    if [[ "${actual_soname}" != "${base}.so.1" ]]; then
      echo "unexpected row-wise SIMD SONAME in ${linker_file}: ${actual_soname}" >&2
      return 1
    fi
    if [[ "$(readlink -f -- "${linker_file}")" != \
          "$(readlink -f -- "${soname_file}")" ]]; then
      echo "row-wise SIMD linker-name and SONAME files resolve differently:" >&2
      echo "  ${linker_file}" >&2
      echo "  ${soname_file}" >&2
      return 1
    fi
  done
}

validate_rowwise_binary_symbols() {
  local binary="$1"
  shift
  local actual
  local expected
  actual="$({
    readelf --dyn-syms --wide "${binary}" |
      awk '$8 ~ /^ascify950_.*_launch_v1$/ { print $7 "\t" $8 }'
  } | sort)"
  expected="$(printf 'UND\t%s\n' "$@" | sort)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "unexpected row-wise SIMD dynamic-symbol set in ${binary}:" >&2
    echo "expected:" >&2
    printf '%s\n' "${expected}" >&2
    echo "actual:" >&2
    printf '%s\n' "${actual}" >&2
    return 1
  fi
}

TARGET="${1:-all}"
MATH_MODE="${2:-production-fast}"
SOFTMAX_VARIANT="${SOFTMAX_VARIANT:-converted}"
SOFTMAX_BLOCK_THREADS="${SOFTMAX_BLOCK_THREADS:-adaptive}"
SOFTMAX_BLOCK_ROW_MAX="${SOFTMAX_BLOCK_ROW_MAX:-2048}"
RMSNORM_VARIANT="${RMSNORM_VARIANT:-converted}"
RMSNORM_BLOCK_ROW_THREADS="${RMSNORM_BLOCK_ROW_THREADS:-512}"
RMSNORM_BLOCK_ROW_AFFINE_THREADS="${RMSNORM_BLOCK_ROW_AFFINE_THREADS:-256}"

case "${TARGET}" in
  all|check|bench|probe|layernorm-check|layernorm-generated-check) ;;
  *)
    echo "usage: $0 [all|check|bench|probe|layernorm-check|layernorm-generated-check] [strict|production-fast|compiler-fast]" >&2
    exit 2
    ;;
esac

case "${MATH_MODE}" in
  strict|production-fast|compiler-fast) ;;
  *)
    echo "unsupported math mode: ${MATH_MODE}" >&2
    exit 2
    ;;
esac

case "${SOFTMAX_VARIANT}" in
  converted)
    SOFTMAX_BIN_SUFFIX=""
    SOFTMAX_DEFINES=()
    ;;
  native-half2-compact)
    SOFTMAX_BIN_SUFFIX="_native_half2_compact"
    SOFTMAX_DEFINES=(
      -DASCIFY950_USE_TUNED_SOFTMAX
      -DSMN_USE_ASC_REDUCE
      -DSMN_USE_HALF2_EXP
      -DSMN_HALF2_REFINE_COMPACT
    )
    ;;
  native-half2-hybrid)
    case "${SOFTMAX_BLOCK_THREADS}" in
      adaptive|256|512|1024) ;;
      *)
        echo "SOFTMAX_BLOCK_THREADS must be adaptive, 256, 512, or 1024" >&2
        exit 2
        ;;
    esac
    if [[ ! "${SOFTMAX_BLOCK_ROW_MAX}" =~ ^[1-9][0-9]*$ ]]; then
      echo "SOFTMAX_BLOCK_ROW_MAX must be a positive integer" >&2
      exit 2
    fi
    SOFTMAX_BIN_SUFFIX="_native_half2_hybrid"
    SOFTMAX_DEFINES=(
      -DASCIFY950_USE_TUNED_SOFTMAX
      -DASCIFY950_USE_BLOCK_SOFTMAX
      -DSMN_USE_ASC_REDUCE
      -DSMN_USE_HALF2_EXP
      -DSMN_HALF2_REFINE_COMPACT
      -DSMN_USE_BLOCK_PER_ROW
    )
    if [[ "${SOFTMAX_BLOCK_THREADS}" != "adaptive" ]]; then
      SOFTMAX_DEFINES+=(
        "-DSMN_FORCE_BLOCK_THREADS=${SOFTMAX_BLOCK_THREADS}"
        "-DSMN_BLOCK_ROW_MAX=${SOFTMAX_BLOCK_ROW_MAX}"
      )
    fi
    ;;
  *)
    echo "unsupported SOFTMAX_VARIANT: ${SOFTMAX_VARIANT}" >&2
    exit 2
    ;;
esac

case "${RMSNORM_VARIANT}" in
  converted)
    RMSNORM_BIN_SUFFIX=""
    RMSNORM_DEFINES=()
    ;;
  native)
    case "${RMSNORM_BLOCK_ROW_THREADS}" in
      128|256|512|1024) ;;
      *)
        echo "RMSNORM_BLOCK_ROW_THREADS must be 128, 256, 512, or 1024" >&2
        exit 2
        ;;
    esac
    case "${RMSNORM_BLOCK_ROW_AFFINE_THREADS}" in
      128|256|512|1024) ;;
      *)
        echo "RMSNORM_BLOCK_ROW_AFFINE_THREADS must be 128, 256, 512, or 1024" >&2
        exit 2
        ;;
    esac
    RMSNORM_BIN_SUFFIX="_native"
    RMSNORM_DEFINES=(
      -DASCIFY950_USE_TUNED_RMSNORM
      "-DASCIFY950_RMN_BLOCK_ROW_THREADS=${RMSNORM_BLOCK_ROW_THREADS}"
      "-DASCIFY950_RMN_BLOCK_ROW_AFFINE_THREADS=${RMSNORM_BLOCK_ROW_AFFINE_THREADS}"
    )
    ;;
  *)
    echo "unsupported RMSNORM_VARIANT: ${RMSNORM_VARIANT}" >&2
    exit 2
    ;;
esac

if [[ -z "${CANN_ROOT}" ]]; then
  echo "CANN_ROOT must point to a user-owned CANN installation" >&2
  exit 2
fi
if [[ ( "${TARGET}" == "layernorm-check"
        || "${TARGET}" == "layernorm-generated-check" )
      && -z "${ROWWISE_SIMD_RUNTIME_DIR}" ]]; then
  echo "${TARGET} requires ROWWISE_SIMD_RUNTIME_DIR" >&2
  exit 2
fi
if [[ ! -f "${CANN_ROOT}/set_env.sh" ]]; then
  echo "missing user-owned CANN environment: ${CANN_ROOT}/set_env.sh" >&2
  exit 1
fi
if [[ "${TARGET}" == "layernorm-generated-check"
      && ! -f "${GENERATED_ROOT}/oneflow/core/cuda/layer_norm.cuh" ]]; then
  echo "missing converted LayerNorm header:" >&2
  echo "  ${GENERATED_ROOT}/oneflow/core/cuda/layer_norm.cuh" >&2
  exit 1
fi
if [[ "${TARGET}" != "probe" && "${TARGET}" != "layernorm-check"
      && "${TARGET}" != "layernorm-generated-check"
      && ! -f "${GENERATED_ROOT}/oneflow/core/cuda/softmax.cuh" ]]; then
  echo "missing converted header:" >&2
  echo "  ${GENERATED_ROOT}/oneflow/core/cuda/softmax.cuh" >&2
  echo "stage the Ascify output under the preserved OneFlow include topology first" >&2
  exit 1
fi
if [[ "${TARGET}" != "probe" && "${TARGET}" != "layernorm-check"
      && "${TARGET}" != "layernorm-generated-check"
      && ! -f "${GENERATED_ROOT}/oneflow/core/cuda/rms_norm.cuh" ]]; then
  echo "missing converted header:" >&2
  echo "  ${GENERATED_ROOT}/oneflow/core/cuda/rms_norm.cuh" >&2
  echo "stage the Ascify output under the preserved OneFlow include topology first" >&2
  exit 1
fi
if [[ "${TARGET}" != "probe" && "${TARGET}" != "layernorm-check"
      && "${TARGET}" != "layernorm-generated-check"
      && ! -f "${GENERATED_ROOT}/oneflow/core/cuda/layer_norm.cuh" ]]; then
  echo "missing converted RMSNorm dependency:" >&2
  echo "  ${GENERATED_ROOT}/oneflow/core/cuda/layer_norm.cuh" >&2
  exit 1
fi
if [[ "${TARGET}" != "probe" && "${TARGET}" != "layernorm-check"
      && "${TARGET}" != "layernorm-generated-check"
      && ! -f "${GENERATED_ROOT}/ascify950/rmsnorm_affine_store.cuh" ]]; then
  echo "missing converted RMSNorm caller adapter:" >&2
  echo "  ${GENERATED_ROOT}/ascify950/rmsnorm_affine_store.cuh" >&2
  exit 1
fi

mkdir -p "${BIN_DIR}" "${PROBE_OBJ_DIR}"

# set_env.sh only affects this build process. It does not install or alter CANN.
set +u
# shellcheck disable=SC1090
source "${CANN_ROOT}/set_env.sh" >/dev/null
set -u

CCEC="${CCEC:-${CANN_ROOT}/tools/bisheng_compiler/bin/ccec}"
if [[ ! -x "${CCEC}" ]]; then
  echo "ccec is not executable: ${CCEC}" >&2
  exit 1
fi

MODE_SUFFIX="${MATH_MODE//-/_}"
DEFINES=("-DCUDA_VERSION=12000")
OPT_FLAGS=(-O2)
if [[ "${MATH_MODE}" == "production-fast" || "${MATH_MODE}" == "compiler-fast" ]]; then
  DEFINES+=(-DOF_SOFTMAX_USE_FAST_MATH -DOF_LAYER_NORM_USE_FAST_MATH)
fi

INCLUDE_DIRS=(
  "${GENERATED_ROOT}"
  "${TEST_ROOT}"
  "${REPO_ROOT}"
  "${REPO_ROOT}/include"
  "${CANN_ROOT}/include"
  "${CANN_ROOT}/include/ascendc/host_api"
  "${CANN_ROOT}/compiler/ascendc/include/highlevel_api"
  "${CANN_ROOT}/compiler/tikcpp/tikcfw"
  "${CANN_ROOT}/compiler/tikcpp/tikcfw/lib"
  "${CANN_ROOT}/compiler/tikcpp/tikcfw/lib/matmul"
  "${CANN_ROOT}/compiler/tikcpp/tikcfw/impl"
  "${CANN_ROOT}/compiler/tikcpp/tikcfw/interface"
  "${CANN_ROOT}/x86_64-linux/asc/include"
  "${CANN_ROOT}/aarch64-linux/asc/include"
)

INCLUDE_FLAGS=()
for include_dir in "${INCLUDE_DIRS[@]}"; do
  if [[ -d "${include_dir}" ]]; then
    INCLUDE_FLAGS+=("-I${include_dir}")
  fi
done

COMPILE_FLAGS=(
  -x dpp
  --cce-aicore-arch=dav-c310-vec
  -std=c++17
  -DNDEBUG
  "${DEFINES[@]}"
  "${OPT_FLAGS[@]}"
  "${INCLUDE_FLAGS[@]}"
)

LINK_FLAGS=(
  "-L${CANN_ROOT}/lib64"
  -lascendcl
  -lruntime
  -lregister
  -lerror_manager
  -lprofapi
  -lascendalog
  -lmmpa
  -lascend_dump
  -lc_sec
  -lstdc++
  -lm
)

if [[ -n "${ROWWISE_SIMD_RUNTIME_DIR}" ]]; then
  validate_rowwise_runtime_dir "${ROWWISE_SIMD_RUNTIME_DIR}"
  LINK_FLAGS+=(
    "-L${ROWWISE_SIMD_RUNTIME_DIR}"
    -lascify950_softmax_reg_recompute_v1
    -lascify950_rmsnorm_reg_cached_v1
    -lascify950_rmsnorm_reg_plain_rowbatch_v1
    -lascify950_layernorm_reg_cached_v1
  )
fi

build_one() {
  local source_file="$1"
  local output_file="$2"
  local effective_math_mode="$3"
  shift 3
  echo "[build] $(basename "${source_file}") -> ${output_file}"
  "${CCEC}" "${COMPILE_FLAGS[@]}" \
    "-DASCIFY950_MATH_MODE=\"${effective_math_mode}\"" "$@" \
    "${source_file}" -o "${output_file}" "${LINK_FLAGS[@]}"
}

compile_only() {
  local source_file="$1"
  local output_file="$2"
  shift 2
  echo "[compile-only] $(basename "${source_file}") -> ${output_file}"
  "${CCEC}" "${COMPILE_FLAGS[@]}" \
    "$@" "${source_file}" -c -o "${output_file}"
}

"${CCEC}" --version 2>&1 | sed -n '1,2p'
echo "[build] CANN_ROOT=${CANN_ROOT}"
echo "[build] GENERATED_ROOT=${GENERATED_ROOT}"
echo "[build] MATH_MODE=${MATH_MODE}"
echo "[build] SOFTMAX_VARIANT=${SOFTMAX_VARIANT}"
echo "[build] RMSNORM_VARIANT=${RMSNORM_VARIANT}"
if [[ -n "${ROWWISE_SIMD_RUNTIME_DIR}" ]]; then
  echo "[build] ROWWISE_SIMD_RUNTIME_DIR=${ROWWISE_SIMD_RUNTIME_DIR}"
fi
if [[ "${RMSNORM_VARIANT}" == "native" ]]; then
  echo "[build] RMSNORM_BLOCK_ROW_THREADS=${RMSNORM_BLOCK_ROW_THREADS}"
  echo "[build] RMSNORM_BLOCK_ROW_AFFINE_THREADS=${RMSNORM_BLOCK_ROW_AFFINE_THREADS}"
fi
if [[ "${SOFTMAX_VARIANT}" == "native-half2-hybrid" ]]; then
  echo "[build] SOFTMAX_BLOCK_THREADS=${SOFTMAX_BLOCK_THREADS}"
  echo "[build] SOFTMAX_BLOCK_ROW_MAX=${SOFTMAX_BLOCK_ROW_MAX}"
fi

if [[ "${TARGET}" == "all" || "${TARGET}" == "check" ]]; then
  # Never compile either host oracle with -ffast-math. In compiler-fast mode
  # the gates validate the same explicit OneFlow fast-math intrinsics, while
  # the compiler-wide flag is isolated to the benchmark TUs below.
  CHECK_MATH_MODE="${MATH_MODE}"
  if [[ "${MATH_MODE}" == "compiler-fast" ]]; then
    CHECK_MATH_MODE="production-fast"
  fi
  build_one "${TEST_ROOT}/softmax_check.cce" \
    "${BIN_DIR}/softmax_check_${MODE_SUFFIX}${SOFTMAX_BIN_SUFFIX}" \
    "${CHECK_MATH_MODE}" "${SOFTMAX_DEFINES[@]}"
  build_one "${TEST_ROOT}/rmsnorm_check.cce" \
    "${BIN_DIR}/rmsnorm_check_${MODE_SUFFIX}${RMSNORM_BIN_SUFFIX}" \
    "${CHECK_MATH_MODE}" "${RMSNORM_DEFINES[@]}"
fi
if [[ "${TARGET}" == "layernorm-check"
      || ( -n "${ROWWISE_SIMD_RUNTIME_DIR}"
           && ( "${TARGET}" == "all" || "${TARGET}" == "check" ) ) ]]; then
  CHECK_MATH_MODE="${MATH_MODE}"
  if [[ "${MATH_MODE}" == "compiler-fast" ]]; then
    CHECK_MATH_MODE="production-fast"
  fi
  build_one "${TEST_ROOT}/layernorm_hybrid_check.cce" \
    "${BIN_DIR}/layernorm_hybrid_check_${MODE_SUFFIX}" \
    "${CHECK_MATH_MODE}"
  if [[ -n "${ROWWISE_SIMD_RUNTIME_DIR}" && "${TARGET}" != "layernorm-check" ]]; then
    validate_rowwise_binary_symbols \
      "${BIN_DIR}/softmax_check_${MODE_SUFFIX}${SOFTMAX_BIN_SUFFIX}" \
      ascify950_softmax_reg_recompute_launch_v1
    validate_rowwise_binary_symbols \
      "${BIN_DIR}/rmsnorm_check_${MODE_SUFFIX}${RMSNORM_BIN_SUFFIX}" \
      ascify950_rmsnorm_reg_cached_launch_v1 \
      ascify950_rmsnorm_reg_plain_rowbatch_launch_v1
  fi
  if [[ -n "${ROWWISE_SIMD_RUNTIME_DIR}" ]]; then
    validate_rowwise_binary_symbols \
      "${BIN_DIR}/layernorm_hybrid_check_${MODE_SUFFIX}" \
      ascify950_layernorm_reg_cached_launch_v1
  fi
fi
if [[ "${TARGET}" == "layernorm-generated-check" ]]; then
  CHECK_MATH_MODE="${MATH_MODE}"
  if [[ "${MATH_MODE}" == "compiler-fast" ]]; then
    CHECK_MATH_MODE="production-fast"
  fi
  build_one "${TEST_ROOT}/layernorm_hybrid_check.cce" \
    "${BIN_DIR}/layernorm_generated_check_${MODE_SUFFIX}" \
    "${CHECK_MATH_MODE}" -DASCIFY950_USE_GENERATED_LAYERNORM
  validate_rowwise_binary_symbols \
    "${BIN_DIR}/layernorm_generated_check_${MODE_SUFFIX}" \
    ascify950_layernorm_reg_cached_launch_v1
fi
if [[ "${TARGET}" == "all" || "${TARGET}" == "bench" ]]; then
  if [[ "${MATH_MODE}" == "compiler-fast" ]]; then
    build_one "${TEST_ROOT}/softmax_bench.cce" \
      "${BIN_DIR}/softmax_bench_${MODE_SUFFIX}${SOFTMAX_BIN_SUFFIX}" \
      "${MATH_MODE}" "${SOFTMAX_DEFINES[@]}" -ffast-math
    build_one "${TEST_ROOT}/rmsnorm_bench.cce" \
      "${BIN_DIR}/rmsnorm_bench_${MODE_SUFFIX}${RMSNORM_BIN_SUFFIX}" \
      "${MATH_MODE}" "${RMSNORM_DEFINES[@]}" -ffast-math
  else
    build_one "${TEST_ROOT}/softmax_bench.cce" \
      "${BIN_DIR}/softmax_bench_${MODE_SUFFIX}${SOFTMAX_BIN_SUFFIX}" \
      "${MATH_MODE}" "${SOFTMAX_DEFINES[@]}"
    build_one "${TEST_ROOT}/rmsnorm_bench.cce" \
      "${BIN_DIR}/rmsnorm_bench_${MODE_SUFFIX}${RMSNORM_BIN_SUFFIX}" \
      "${MATH_MODE}" "${RMSNORM_DEFINES[@]}"
  fi
fi
if [[ "${TARGET}" == "all" || "${TARGET}" == "probe" ]]; then
  compile_only \
    "${TEST_ROOT}/probes/rowwise_recipe_traits_compile.cce" \
    "${PROBE_OBJ_DIR}/rowwise_recipe_traits_compile.o"
  compile_only \
    "${TEST_ROOT}/probes/rowwise_recipe_traits_compile.cce" \
    "${PROBE_OBJ_DIR}/rowwise_recipe_traits_predefined_aclcub_compile.o" \
    -DASCIFY_ROWWISE_PROBE_PREDEFINED_ACLCUB_WARP_SIZE
  build_one \
    "${TEST_ROOT}/probes/rowwise_recipe_contract_probe.cce" \
    "${BIN_DIR}/rowwise_recipe_contract_probe" \
    "dispatch-contract"
  build_one "${TEST_ROOT}/probes/simt_hw_probes.cce" "${BIN_DIR}/simt_hw_probes" \
    "hardware-probe"
fi

echo "[ok] binaries are confined to ${BIN_DIR}"
