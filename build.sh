#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage: LLVM_BUILD_DIR=/path/to/llvm-prefix ./build.sh [CMAKE_OPTIONS...]

Build ascify-clang with an LLVM/Clang development prefix. A configured LLVM
source build also works. Set LLVM_PROJECT_PATH to use its build/ directory.

Environment:
  LLVM_BUILD_DIR       LLVM/Clang prefix or build tree
  LLVM_PROJECT_PATH    Alternative: LLVM source root (uses build/)
  BUILD_DIR            Ascify build directory (default: ./build)
  INSTALL_ROOT         Installation prefix (default: ./ascify_install)
  ASCIFY_CC/CXX        Host compilers; otherwise use CC/CXX, prefix clang,
                      or CMake's default compiler discovery, in that order
  ASCIFY_BUILD_JOBS    Positive build parallelism (default: 2)
  CMAKE_GENERATOR      Build system (default: Ninja)
  CMAKE_BUILD_TYPE     Build type (default: Release)
  ASCIFY_LINKER       Optional linker executable

Remaining arguments are passed unchanged to CMake configuration.
Install separately with: cmake --install <BUILD_DIR>
USAGE
  exit 0
fi

if [[ -z "${LLVM_BUILD_DIR:-}" ]]; then
  if [[ -z "${LLVM_PROJECT_PATH:-}" ]]; then
    echo "Set LLVM_BUILD_DIR to an LLVM/Clang development prefix, or LLVM_PROJECT_PATH to its source tree." >&2
    exit 2
  fi
  LLVM_BUILD_DIR="${LLVM_PROJECT_PATH}/build"
fi

BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
INSTALL_ROOT="${INSTALL_ROOT:-${SCRIPT_DIR}/ascify_install}"
ASCIFY_BUILD_JOBS="${ASCIFY_BUILD_JOBS:-2}"

if [[ ! -d "${LLVM_BUILD_DIR}" ]]; then
  echo "LLVM_BUILD_DIR is not a directory: ${LLVM_BUILD_DIR}" >&2
  exit 1
fi
if [[ ! "${ASCIFY_BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ASCIFY_BUILD_JOBS must be a positive integer" >&2
  exit 2
fi

cmake_args=(-S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -G "${CMAKE_GENERATOR:-Ninja}"
  "-DCMAKE_INSTALL_PREFIX=${INSTALL_ROOT}"
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}"
  "-DCMAKE_PREFIX_PATH=${LLVM_BUILD_DIR}")
for compiler_kind in CC CXX; do
  override_name="ASCIFY_${compiler_kind}"
  compiler="${!override_name:-${!compiler_kind:-}}"
  if [[ -z "${compiler}" ]]; then
    if [[ "${compiler_kind}" == CC ]]; then
      candidate="${LLVM_BUILD_DIR}/bin/clang"
    else
      candidate="${LLVM_BUILD_DIR}/bin/clang++"
    fi
    if [[ -x "${candidate}" ]]; then
      compiler="${candidate}"
    fi
  fi
  if [[ -z "${compiler}" ]]; then
    continue
  fi
  if ! command -v "${compiler}" >/dev/null 2>&1; then
    echo "Host compiler is not executable: ${compiler}" >&2
    exit 1
  fi
  if [[ "${compiler_kind}" == CC ]]; then
    cmake_args+=("-DCMAKE_C_COMPILER=${compiler}")
  else
    cmake_args+=("-DCMAKE_CXX_COMPILER=${compiler}")
  fi
done

if [[ -n "${ASCIFY_LINKER:-}" ]]; then
  if ! command -v "${ASCIFY_LINKER}" >/dev/null 2>&1; then
    echo "ASCIFY_LINKER is not executable: ${ASCIFY_LINKER}" >&2
    exit 1
  fi
  cmake_args+=("-DCMAKE_LINKER=${ASCIFY_LINKER}")
fi

cmake "${cmake_args[@]}" "$@"

cmake --build "${BUILD_DIR}" --config "${CMAKE_BUILD_TYPE:-Release}" \
  --parallel "${ASCIFY_BUILD_JOBS}"
