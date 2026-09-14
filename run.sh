#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASCIFY_BINARY="${ASCIFY_BINARY:-${SCRIPT_DIR}/build/ascify-clang}"

usage() {
  cat <<'USAGE'
Usage: CUDA_PATH=/path/to/cuda ./run.sh INPUT [ASCIFY_OPTIONS] [-- CLANG_OPTIONS]

Environment:
  ASCIFY_BINARY             Built or installed ascify-clang executable
                            (default: ./build/ascify-clang)
  CUDA_PATH                 CUDA Toolkit parsing root
  CLANG_RESOURCE_DIRECTORY  Optional explicit Clang resource root; otherwise
                            ascify-clang discovers its build/install resources
USAGE
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ -z "${CUDA_PATH:-}" ]]; then
  echo "Set CUDA_PATH to the CUDA Toolkit parsing root." >&2
  exit 2
fi
if [[ ! -x "${ASCIFY_BINARY}" ]]; then
  echo "Ascify binary is not executable: ${ASCIFY_BINARY}" >&2
  exit 1
fi

input="$1"
shift
ascify_args=("${input}" "--cuda-path=${CUDA_PATH}")
if [[ -n "${CLANG_RESOURCE_DIRECTORY:-}" ]]; then
  ascify_args+=("--clang-resource-directory=${CLANG_RESOURCE_DIRECTORY}")
fi
exec "${ASCIFY_BINARY}" "${ascify_args[@]}" "$@"
