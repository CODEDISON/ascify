#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/ascify-texture-uint.XXXXXX")
trap 'rm -rf -- "$scratch"' 0 1 2 15
for test in texture_unsigned_sampling_test texture_unsigned_lifecycle_test texture_array_copy2d_test; do
  "${CXX:-c++}" -std=c++17 -I"$repo_root/tests/rewrite/stubs" -I"$repo_root/include" \
    "$repo_root/tests/rewrite/$test.cpp" -o "$scratch/$test"
  "$scratch/$test"
done
printf '%s\n' 'unsigned texture host wrapper checks passed (no NPU evidence)'
