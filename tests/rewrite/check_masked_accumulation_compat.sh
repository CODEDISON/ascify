#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/ascify-masked-accumulation.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
"${CXX:-c++}" -std=c++17 -O2 -fno-fast-math \
  -fsanitize=undefined -fno-sanitize-recover=undefined \
  -D__aicore__= -D__global__= -D__device__= -D__ubuf__= \
  -D__NPU_ARCH__=3510 -DASCENDC_CPU_DEBUG=1 \
  -I"$repo_root/tests/rewrite/masked_accumulation_stubs" \
  -I"$repo_root/tests/rewrite/stubs" -I"$repo_root/include" \
  "$repo_root/tests/rewrite/masked_accumulation_host_test.cpp" -o "$work/model"
"$work/model"
