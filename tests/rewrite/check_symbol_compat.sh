#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/ascify-symbol-test.XXXXXX")
trap 'rm -rf "$test_tmp"' EXIT
cxx=${CXX:-c++}
args=(-std=c++17 -Wall -Wextra -Werror -Werror=return-type -I"$repo_root/tests/rewrite/symbol_stubs" -I"$repo_root/include")
"$cxx" "${args[@]}" "$repo_root/tests/rewrite/symbol_compat_test.cpp" -o "$test_tmp/contracts"
"$test_tmp/contracts"

cat > "$test_tmp/old-sdk.cpp" <<'EOF'
#define ASCIFY_ASCIFY_CUDA_COMPAT_HPP
#define ASCIFY_TEST_SYMBOL_OLD_SDK 1
#define ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85 1
#include <ascify/symbol_compat.hpp>
#ifdef INVOKE_SYMBOL_COPY
int check(float &symbol) {
  return ascify::cudaMemcpyToSymbol(symbol, &symbol, sizeof(symbol));
}
#endif
int main() { return 0; }
EOF
# GCC checks the non-void template body before its dependent assertion can
# reject an instantiation. Including an unused adapter must still be valid.
"$cxx" "${args[@]}" -fsyntax-only "$test_tmp/old-sdk.cpp"
if "$cxx" "${args[@]}" -DINVOKE_SYMBOL_COPY -fsyntax-only "$test_tmp/old-sdk.cpp" > "$test_tmp/old-sdk.log" 2>&1; then
  echo 'error: unadmitted SDK unexpectedly accepted a symbol copy' >&2
  exit 1
fi
grep -Fq 'Ascify symbol copies require the admitted CANN 9.1' "$test_tmp/old-sdk.log"
echo 'synthetic unsupported-SDK compile boundary passed (no NPU)'
