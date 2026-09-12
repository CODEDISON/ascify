#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/ascify-uniform-block.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
"${CXX:-c++}" -std=c++17 -O2 -pthread -I"$repo_root/include" \
  "$repo_root/tests/rewrite/uniform_block_reduction_host_test.cpp" -o "$work/model"
"$work/model"
