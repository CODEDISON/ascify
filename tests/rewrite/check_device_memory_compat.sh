#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/ascify-device-memory.XXXXXX")
trap 'rm -rf -- "$work"' 0 1 2 15
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -I"$repo_root/include" \
  "$repo_root/tests/rewrite/device_memory_compat_test.cpp" -o "$work/host"
"$work/host"
if "${CXX:-c++}" -std=c++17 -I"$repo_root/include" \
  "$repo_root/tests/rewrite/device_memory_size_reject.cpp" -o "$work/reject" \
  >"$work/reject.stdout" 2>"$work/reject.stderr"; then
  echo 'wrong target object size unexpectedly accepted' >&2
  exit 1
fi
grep -F 'requires the proved target object sizes' "$work/reject.stderr" >/dev/null
echo 'private memcpy 256 byte patterns, word representation, return value, canaries and target-size rejection passed'
