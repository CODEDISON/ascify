#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/ascify-symbol-test.XXXXXX")
trap 'rm -rf "$test_tmp"' EXIT
cxx=${CXX:-c++}
args=(-std=c++17 -Wall -Wextra -Werror -Werror=return-type -I"$repo_root/tests/rewrite/symbol_stubs" -I"$repo_root/include")
"$cxx" "${args[@]}" "$repo_root/tests/rewrite/symbol_compat_test.cpp" -o "$test_tmp/contracts"
"$test_tmp/contracts"

cat > "$test_tmp/sdk-boundary.cpp" <<'EOF'
#define ASCIFY_ASCIFY_CUDA_COMPAT_HPP
#if defined(ASCIFY_TEST_SYMBOL_PUBLIC_FAMILY)
#define ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85 1
#else
#define ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3 1
#endif
#include <acl/acl.h>
// User namespace lookalikes must not masquerade as SDK declarations via ADL.
namespace application {
struct Symbol {};
aclError aclrtGetSymbolSize(const void *, size_t *);
aclError aclrtMemcpyToSymbol(const void *, const void *, size_t, size_t,
                            aclrtMemcpyKind);
}
#include <ascify/symbol_compat.hpp>
#ifdef ASCIFY_TEST_SYMBOL_LATE_GLOBAL_APIS
// An application declaration after the header is not an SDK capability.
aclError aclrtGetSymbolSize(const void *, size_t *);
aclError aclrtMemcpyToSymbol(const void *, const void *, size_t, size_t,
                            aclrtMemcpyKind);
#endif
#ifdef INVOKE_SYMBOL_COPY
int check(application::Symbol &symbol) {
  return ascify::cudaMemcpyToSymbol(symbol, &symbol, sizeof(symbol));
}
#endif
int main() { return 0; }
EOF
# Both missing APIs and either API alone must reject a use, including when
# the ACL 1.17 / legacy SIMT version signals match an SDK that has the APIs.
# An unused adapter must compile and link without either SDK symbol.
check_boundary() {
  local case_name=$1
  shift
  "$cxx" "${args[@]}" "$@" "$test_tmp/sdk-boundary.cpp" -o "$test_tmp/$case_name"
  "$test_tmp/$case_name"
  if "$cxx" "${args[@]}" "$@" -DINVOKE_SYMBOL_COPY -fsyntax-only "$test_tmp/sdk-boundary.cpp" > "$test_tmp/$case_name.log" 2>&1; then
    echo "error: $case_name unexpectedly accepted a symbol copy" >&2
    exit 1
  fi
  grep -Fq 'Ascify symbol copies require legacy SIMT headers and SDK declarations of aclrtGetSymbolSize and aclrtMemcpyToSymbol' "$test_tmp/$case_name.log"
}
check_boundary old-sdk -DASCIFY_TEST_SYMBOL_OLD_SDK -DASCIFY_TEST_SYMBOL_PUBLIC_FAMILY
check_boundary missing-both -DASCIFY_TEST_SYMBOL_NO_SIZE_API -DASCIFY_TEST_SYMBOL_NO_COPY_API
check_boundary missing-size -DASCIFY_TEST_SYMBOL_NO_SIZE_API
check_boundary missing-copy -DASCIFY_TEST_SYMBOL_NO_COPY_API
check_boundary late-global-copy -DASCIFY_TEST_SYMBOL_NO_COPY_API -DASCIFY_TEST_SYMBOL_LATE_GLOBAL_APIS
check_boundary late-global-both -DASCIFY_TEST_SYMBOL_NO_SIZE_API -DASCIFY_TEST_SYMBOL_NO_COPY_API -DASCIFY_TEST_SYMBOL_LATE_GLOBAL_APIS
check_boundary public-family -DASCIFY_TEST_SYMBOL_PUBLIC_FAMILY
echo 'synthetic unsupported-SDK compile/link boundaries passed (no NPU)'
