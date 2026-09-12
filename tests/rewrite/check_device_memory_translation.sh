#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
: "${ASCIFY_BINARY:?}"
: "${ASCIFY_CUDA_PATH:?}"
: "${ASCIFY_CLANG_RESOURCE_DIRECTORY:?}"
work=$(mktemp -d "${TMPDIR:-/tmp}/ascify-device-memory-translation.XXXXXX")
trap 'rm -rf -- "$work"' 0 1 2 15
for fixture in device_memory_input device_memory_unproven_input device_memory_redecl_input; do
  if ! "$ASCIFY_BINARY" "$repo_root/tests/rewrite/$fixture.cu" \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    --target-policy=dav-c310-vec --simt-math=fast --default-preprocessor \
    -o "$work/$fixture.cce" \
    -- -std=c++17 >"$work/$fixture.stdout" 2>"$work/$fixture.stderr"; then
    cat "$work/$fixture.stdout" "$work/$fixture.stderr" >&2
    exit 1
  fi
done
if ! "$ASCIFY_BINARY" "$repo_root/tests/rewrite/device_memory_input.cu" \
  "--cuda-path=$ASCIFY_CUDA_PATH" \
  "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
  --target-policy=dav-c310-vec --simt-math=fast -o "$work/retained.cce" \
  -- -std=c++17 >"$work/retained.stdout" 2>"$work/retained.stderr"; then
  cat "$work/retained.stdout" "$work/retained.stderr" >&2
  exit 1
fi
"${PYTHON:-python3}" -B - "$work" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
positive = (root / 'device_memory_input.cce').read_text()
negative = (root / 'device_memory_unproven_input.cce').read_text()
redecl = (root / 'device_memory_redecl_input.cce').read_text()
assert positive.count('::ascify::device_memcpy_private<64>') == 3
assert '::ascify::device_memcpy_private<64>' not in (root / 'retained.cce').read_text()
assert 'device_memcpy_private' not in negative
assert 'device_memcpy_private' not in redecl
assert 'PRIVATE_COPY_MACRO(&packet, words)' in negative
assert 'memcpy(&packet, global, sizeof(packet))' in negative
assert 'memcpy(&packet, shared_words, sizeof(packet))' in negative
assert 'memcpy(&shared_packet, words, sizeof(shared_packet))' in negative
print('private memcpy translation and shared/global/alias/size/template/macro/host/user/redeclaration/namespace/preprocessing boundaries passed')
PY
