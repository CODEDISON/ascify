#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
: "${ASCIFY_BINARY:?}"
: "${ASCIFY_CUDA_PATH:?}"
: "${ASCIFY_CLANG_RESOURCE_DIRECTORY:?}"
work=$(mktemp -d "${TMPDIR:-/tmp}/ascify-half2-translation.XXXXXX")
trap 'rm -rf -- "$work"' 0 1 2 15
translate() {
  "$ASCIFY_BINARY" "$repo_root/tests/rewrite/$1.cu" \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    --target-policy=dav-c310-vec --simt-math=fast -o "$work/$1.cce" \
    -- -std=c++17 >"$work/$1.stdout" 2>"$work/$1.stderr"
}
translate half2_compat_input
"${PYTHON:-python3}" -B "$repo_root/tests/rewrite/check_half2_compat.py" \
  --translated "$work/half2_compat_input.cce"
for rejected in half2_compat_reject half2_template_reject; do
  if translate "$rejected"; then
    echo "unproven half2 expression unexpectedly converted: $rejected" >&2
    exit 1
  fi
  test ! -e "$work/$rejected.cce"
  grep -F 'separate half rounding could not be preserved' "$work/$rejected.stderr" >/dev/null
done
echo "half2 source conversion and precision boundaries passed"
