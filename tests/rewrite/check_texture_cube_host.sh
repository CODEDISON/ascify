#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/ascify-texture-cube.XXXXXX")
trap 'rm -rf -- "$scratch"' 0 1 2 15
for test in texture_cube_sampling_test texture_cube_lifecycle_test; do
  "${CXX:-c++}" -std=c++17 -I"$repo_root/tests/rewrite/stubs" -I"$repo_root/include" \
    "$repo_root/tests/rewrite/$test.cpp" -o "$scratch/$test"
  "$scratch/$test"
done
printf '%s\n' 'bounded cubemap center CPU checks passed (no NPU evidence)'
