#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/ascify-texture-mirror.XXXXXX")
trap 'rm -rf -- "$scratch"' 0 1 2 15
for test in texture_mirror_sampling_test texture_mirror_abi_test; do
  "${CXX:-c++}" -std=c++17 -I"$repo_root/tests/rewrite/stubs" -I"$repo_root/include" \
    "$repo_root/tests/rewrite/$test.cpp" -o "$scratch/$test"
  "$scratch/$test"
done
printf '%s\n' 'mirror texture CPU wrapper checks passed (no NPU evidence)'
