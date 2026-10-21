#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/ascify-packed-sad.XXXXXX")
trap 'rm -rf -- "$scratch"' 0 1 2 15
"${CXX:-c++}" -std=c++17 -O2 -I"$repo_root/tests/rewrite/stubs" -I"$repo_root/include" \
  "$repo_root/tests/rewrite/packed_sad_host_test.cpp" -o "$scratch/packed-sad-host"
"$scratch/packed-sad-host"
