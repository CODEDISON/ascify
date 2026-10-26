#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/ascify-texture-rgba.XXXXXX")
trap 'rm -rf -- "$scratch"' 0 1 2 15
for test in texture_rgba_sampling_test texture_rgba_abi_test texture_rgba_lifecycle_test; do
  "${CXX:-c++}" -std=c++17 -I"$repo_root/tests/rewrite/stubs" -I"$repo_root/include" \
    "$repo_root/tests/rewrite/$test.cpp" -o "$scratch/$test"
  "$scratch/$test"
done
cat >"$scratch/scalar.cpp" <<'EOF'
#define __aicore__
#include <ascify/texture_compat.hpp>
struct CallerPixel { unsigned char x,y,z,w; };
static_assert(!ascify::detail::textureRGBAElementType<CallerPixel>::value,"no structural type grant");
int main(){auto c=ascify::cudaCreateChannelDesc<unsigned int>();return c.x==32?0:1;}
EOF
for family in legacy public85; do
  if [ "$family" = public85 ]; then
    "${CXX:-c++}" -std=c++17 -DASCIFY_TEST_PUBLIC_85_ACL \
      -I"$repo_root/tests/rewrite/stubs_public85" -I"$repo_root/tests/rewrite/stubs" \
      -I"$repo_root/include" "$scratch/scalar.cpp" -o "$scratch/scalar-$family"
  else
    "${CXX:-c++}" -std=c++17 -I"$repo_root/tests/rewrite/stubs" \
      -I"$repo_root/include" "$scratch/scalar.cpp" -o "$scratch/scalar-$family"
  fi
  "$scratch/scalar-$family"
done
cat >"$scratch/host-vector-reject.cpp" <<'EOF'
#define __aicore__
#include <ascify/texture_compat.hpp>
int main(){auto c=ascify::cudaCreateChannelDesc<uchar4>();return c.x;}
EOF
if "${CXX:-c++}" -std=c++17 -I"$repo_root/tests/rewrite/stubs" -I"$repo_root/include" \
    "$scratch/host-vector-reject.cpp" -o "$scratch/host-vector-reject" 2>"$scratch/reject.log"; then
  echo 'ordinary host mode unexpectedly admitted native RGBA factory' >&2
  exit 1
fi
grep -F 'Ascify texture channels require' "$scratch/reject.log" >/dev/null
printf '%s\n' 'RGBA texture host wrapper checks passed (no NPU evidence)'
