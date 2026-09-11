#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/ascify-device-properties.XXXXXX")
trap 'rm -rf -- "$work"' 0 1 2 15
for family in legacy public85; do
  extra=""
  if [ "$family" = public85 ]; then
    extra="-DASCIFY_TEST_PUBLIC_85_ACL -I$repo_root/tests/rewrite/stubs_public85"
  fi
  # shellcheck disable=SC2086
  "${CXX:-c++}" -std=c++17 $extra -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" "$repo_root/tests/rewrite/device_properties_compat_test.cpp" \
    -o "$work/properties"
  "$work/properties"
done
for field in major minor totalGlobalMem; do
  printf '#define __aicore__\n#include <ascify/ascify_cuda_compat.hpp>\nint main(){ascify::cudaDeviceProp p; return p.%s;}\n' "$field" > "$work/reject.cpp"
  if "${CXX:-c++}" -std=c++17 -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" -fsyntax-only "$work/reject.cpp" >"$work/negative.log" 2>&1; then
    echo "unmapped CUDA property $field unexpectedly accepted" >&2
    exit 1
  fi
  grep -F "$field" "$work/negative.log" >/dev/null
done
if [ -n "${ASCIFY_BINARY:-}" ]; then
  : "${ASCIFY_CUDA_PATH:?ASCIFY_CUDA_PATH is required for converter checks}"
  : "${ASCIFY_CLANG_RESOURCE_DIRECTORY:?ASCIFY_CLANG_RESOURCE_DIRECTORY is required}"
  "$ASCIFY_BINARY" "$repo_root/tests/rewrite/device_properties_host_input.cu" \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    -o "$work/host_properties.cpp" -- -std=c++17 \
    >"$work/convert.stdout" 2>"$work/convert.stderr"
  grep -F 'ascify/ascify_cuda_compat.hpp' "$work/host_properties.cpp" >/dev/null
  grep -F 'ascify::cudaDeviceProp' "$work/host_properties.cpp" >/dev/null
  grep -F 'ascify::cudaGetDeviceProperties' "$work/host_properties.cpp" >/dev/null
  if grep -F '__global__' "$work/host_properties.cpp" >/dev/null; then
    echo "host-only property fixture unexpectedly contains a kernel" >&2
    exit 1
  fi
  "${CXX:-c++}" -std=c++17 -D__aicore__= \
    -I"$repo_root/tests/rewrite/stubs" -I"$repo_root/include" \
    -fsyntax-only "$work/host_properties.cpp"
  echo "Host-only property conversion supplies a compilable compat include"
fi
echo "Partial device properties and uniform full-warp vote checks passed"
