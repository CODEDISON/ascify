#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
test_work=$(mktemp -d "${TMPDIR:-/tmp}/ascify-vec3-layout-host.XXXXXX")
trap 'rm -rf -- "$test_work"' 0 1 2 15
"${CXX:-c++}" -std=c++17 -Wall -Wextra -pedantic -I"$repo_root/include" \
  "$repo_root/tests/rewrite/cuda_vec3_layout_host.cpp" -o "$test_work/layout"
"$test_work/layout"
