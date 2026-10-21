#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/ascify-texture-malloc.XXXXXX")
trap 'rm -rf -- "$scratch"' 0 1 2 15
"${CXX:-c++}" -std=c++17 -pthread -I"$repo_root/tests/rewrite/stubs" -I"$repo_root/include" \
  "$repo_root/tests/rewrite/texture_malloc_lifecycle_test.cpp" -o "$scratch/check"
"$scratch/check"
printf '%s\n' 'ordinary malloc texture lifecycle host model passed (not NPU evidence)'
