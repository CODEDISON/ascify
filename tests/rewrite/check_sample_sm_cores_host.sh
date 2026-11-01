#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/ascify-sm-cores.XXXXXX")
trap 'rm -rf -- "$scratch"' 0 1 2 15
"${CXX:-c++}" -std=c++17 -I"$repo_root/tests/rewrite/stubs" -I"$repo_root/include" \
  "$repo_root/tests/rewrite/sample_sm_cores_host_test.cpp" -o "$scratch/check"
"$scratch/check"
printf '%s\n' 'versioned NVIDIA SM host-table semantics passed (no NPU evidence)'
