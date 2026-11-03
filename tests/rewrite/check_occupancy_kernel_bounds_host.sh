#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/ascify-occupancy-host.XXXXXX")
trap 'rm -rf -- "$work"' 0 1 2 15
"${CXX:-c++}" -std=c++17 -I"$root/tests/rewrite/stubs" -I"$root/include" \
  "$root/tests/rewrite/occupancy_kernel_bound_runtime_test.cpp" -o "$work/test"
"$work/test"
