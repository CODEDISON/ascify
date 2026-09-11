#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
rewrite_dir="$repo_root/tests/rewrite"
python=${PYTHON:-python3}

run_check() {
  printf '[release-check] %s\n' "$1"
  shift
  "$@"
}

if ! command -v "$python" >/dev/null 2>&1; then
  echo "Python 3 executable not found: $python" >&2
  exit 1
fi

cd -- "$repo_root"

run_check "dav-c310 recipe static contract" \
  sh "$rewrite_dir/check_dav_c310_recipe.sh"
run_check "canonical warp reduction static contract" \
  sh "$rewrite_dir/check_warp_reduce_rewrite.sh"
run_check "canonical reducer static contract" \
  sh "$rewrite_dir/check_canonical_reducer_rewrite.sh"
run_check "SIMT compatibility static contract" \
  sh "$rewrite_dir/check_simt_compat.sh"
run_check "partial device properties and warp vote contracts" \
  sh "$rewrite_dir/check_device_properties.sh"
run_check "registered symbol copy argument and error contracts" \
  bash "$rewrite_dir/check_symbol_compat.sh"
run_check "half2 parser ADL and dependent builtin lookup" \
  "$python" -B tests/rewrite/check_half2_parse_lookup.py
run_check "host float max semantic admission" \
  "$python" -B tests/frontend_compat/check_frontend_host_math.py
run_check "target ABI compatibility static contract" \
  sh "$rewrite_dir/check_target_abi_compat.sh"
run_check "NVIDIA sample-helper compatibility contract" \
  sh "$rewrite_dir/check_sample_helper_compat.sh"
run_check "narrow cooperative-groups compatibility contract" \
  sh "$rewrite_dir/check_cooperative_groups_compat.sh"
run_check "opt-in frontend compatibility admission contract" \
  sh "$repo_root/tests/frontend_compat/check_frontend_compat.sh"
run_check "translated local include closure static contract" \
  sh "$rewrite_dir/check_local_header_closure.sh"
run_check "translated local include closure host fixtures" \
  "$python" -B -m unittest -q \
    tests/rewrite/test_local_header_closure.py
run_check "machine-readable migration receipt contract" \
  sh "$rewrite_dir/check_migration_receipt.sh"
run_check "formal-evidence and work-metric Python unit tests" \
  "$python" -B -m unittest -q \
    tests/softmax_rmsnorm_950/scripts/test_validate_formal_run.py \
    tests/softmax_rmsnorm_950/tools/test_derive_work_metrics.py
run_check "row-wise SIMD selector and dispatch contracts" \
  sh tests/rewrite/check_rowwise_simd_recipes.sh
run_check "static kernel proof M0 contract" \
  sh tests/rewrite/check_static_kernel_proof_m0.sh
run_check "row-wise ascify-clang wrapper Python unit tests" \
  "$python" -I -B tests/rewrite/test_generate_rowwise_cce.py -q
run_check "explicit FP16 input profile and FP64 rejection" \
  "$python" -B tests/softmax_rmsnorm_950/scripts/test_prepare_fp16_input_profile.py -q
run_check "RMSNorm full1000 input identities" \
  "$python" -B tests/softmax_rmsnorm_950/tools/test_prepare_rmsnorm_full1000.py -q
run_check "RMSNorm 64-bit stores and complete chunked host oracle" \
  "$python" -B tests/softmax_rmsnorm_950/tools/test_rmsnorm_full1000_host.py -q

if [ -z "${ASCIFY_BINARY:-}" ]; then
  echo "[release-check] translated fixtures skipped (ASCIFY_BINARY is unset)"
  echo "[release-check] all host-only checks passed"
  exit 0
fi

if [ ! -x "$ASCIFY_BINARY" ]; then
  echo "ASCIFY_BINARY is not executable: $ASCIFY_BINARY" >&2
  exit 1
fi
if [ -z "${ASCIFY_CUDA_PATH:-}" ]; then
  echo "ASCIFY_CUDA_PATH is required when ASCIFY_BINARY is set" >&2
  exit 1
fi
if [ ! -d "$ASCIFY_CUDA_PATH" ]; then
  echo "ASCIFY_CUDA_PATH is not a directory: $ASCIFY_CUDA_PATH" >&2
  exit 1
fi
if [ -z "${ASCIFY_CLANG_RESOURCE_DIRECTORY:-}" ]; then
  echo "ASCIFY_CLANG_RESOURCE_DIRECTORY is required when ASCIFY_BINARY is set" >&2
  exit 1
fi
if [ ! -d "$ASCIFY_CLANG_RESOURCE_DIRECTORY/include" ]; then
  echo "Clang resource include directory not found: $ASCIFY_CLANG_RESOURCE_DIRECTORY/include" >&2
  exit 1
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/ascify-release-checks.XXXXXX")
cleanup() {
  case "$work_dir" in
    "${TMPDIR:-/tmp}"/ascify-release-checks.*)
      rm -rf -- "$work_dir"
      ;;
    *)
      echo "refusing to remove unexpected work directory: $work_dir" >&2
      ;;
  esac
}
trap cleanup 0 1 2 15

run_check "real Ascify NVIDIA sample-helper frontend closure" \
  sh "$rewrite_dir/check_sample_helper_compat.sh" \
    --binary "$ASCIFY_BINARY" \
    --cuda-path "$ASCIFY_CUDA_PATH" \
    --resource-dir "$ASCIFY_CLANG_RESOURCE_DIRECTORY"

run_check "real ascify local-header publication negatives" \
  sh "$rewrite_dir/check_local_header_integration.sh"
run_check "half2 syntax and separate rounding conversion" \
  sh "$rewrite_dir/check_half2_translation.sh"
run_check "inherited parent helper conversion boundary" \
  "$python" -B "$rewrite_dir/check_inherited_sample_helper_context.py"

translate_fixture() {
  input=$1
  output=$2
  target_policy=$3
  math_mode=$4
  shift 4
  run_check "translate $(basename "$input") ($target_policy/$math_mode)" \
    "$ASCIFY_BINARY" "$input" \
      "--target-policy=$target_policy" \
      "--simt-math=$math_mode" \
      "--cuda-path=$ASCIFY_CUDA_PATH" \
      "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
      "$@" \
      -o "$output" \
      -- "-I$repo_root/tests/support/cuda" -std=c++17
}

reject_frontend_compat_fixture() {
  input=$1
  output=$2
  profile=$3
  expected_error=$4
  shift 4
  if "$ASCIFY_BINARY" "$input" \
      "--frontend-compat=$profile" \
      "--cuda-path=$ASCIFY_CUDA_PATH" \
      "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
      -o "$output" \
      -- -std=c++17 "$@" \
      >"$output.stdout" 2>"$output.stderr"; then
    echo "frontend compatibility rejection unexpectedly succeeded: $profile" >&2
    exit 1
  fi
  if [ -e "$output" ]; then
    echo "rejected frontend compatibility fixture published output" >&2
    exit 1
  fi
  if ! grep -F -- "$expected_error" "$output.stderr" >/dev/null; then
    echo "frontend compatibility rejection did not report: $expected_error" >&2
    exit 1
  fi
}

reject_damaged_frontend_profile() {
  input=$1
  output=$2
  isolated_prefix="$work_dir/frontend-profile-prefix"
  isolated_binary="$isolated_prefix/bin/ascify-clang"
  isolated_profile_root="$isolated_prefix/libexec/ascify/frontend-compat"
  isolated_profile="$isolated_profile_root/ascify-admitted-v1"
  original_lib=$(CDPATH= cd -- "$(dirname -- "$ASCIFY_BINARY")/../lib" 2>/dev/null && pwd || true)

  mkdir -p -- "$isolated_prefix/bin" "$isolated_profile_root"
  cp -- "$ASCIFY_BINARY" "$isolated_binary"
  chmod +x -- "$isolated_binary"
  if [ -n "$original_lib" ]; then
    ln -s -- "$original_lib" "$isolated_prefix/lib"
  fi

  reset_isolated_profile() {
    case "$isolated_profile" in
      "$work_dir"/*)
        rm -rf -- "$isolated_profile"
        ;;
      *)
        echo "refusing to reset unexpected frontend profile path" >&2
        exit 1
        ;;
    esac
    cp -R -- "$repo_root/frontend_compat/ascify-admitted-v1" \
      "$isolated_profile"
  }

  run_isolated_profile_rejection() {
    rejected_output=$1
    rejected_error=$2
    if "$isolated_binary" "$input" \
      --frontend-compat=ascify-admitted-v1 \
      "--cuda-path=$ASCIFY_CUDA_PATH" \
      "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
      -o "$rejected_output" -- -std=c++17 \
      >"$rejected_output.stdout" 2>"$rejected_output.stderr"; then
      echo "damaged frontend compatibility profile unexpectedly succeeded" >&2
      exit 1
    fi
    if [ -e "$rejected_output" ]; then
      echo "damaged frontend compatibility profile published output" >&2
      exit 1
    fi
    if ! grep -F -- "$rejected_error" "$rejected_output.stderr" >/dev/null; then
      echo "damaged frontend profile did not report: $rejected_error" >&2
      exit 1
    fi
  }

  reset_isolated_profile
  original_bytes=$(wc -c <"$isolated_profile/cooperative_groups.h")
  sed 's/class thread_block {/class thread_blocK {/' \
    "$isolated_profile/cooperative_groups.h" \
    >"$isolated_profile/cooperative_groups.h.mutated"
  mutated_bytes=$(wc -c <"$isolated_profile/cooperative_groups.h.mutated")
  if [ "$original_bytes" != "$mutated_bytes" ]; then
    echo "same-size frontend profile mutation changed file length" >&2
    exit 1
  fi
  mv -- "$isolated_profile/cooperative_groups.h.mutated" \
    "$isolated_profile/cooperative_groups.h"
  run_isolated_profile_rejection \
    "$output.same-size-content" \
    "exact content mismatch for 'cooperative_groups.h'"

  reset_isolated_profile
  sed 's/profile.v1/profile.v2/' "$isolated_profile/profile.manifest" \
    >"$isolated_profile/profile.manifest.mutated"
  mv -- "$isolated_profile/profile.manifest.mutated" \
    "$isolated_profile/profile.manifest"
  run_isolated_profile_rejection \
    "$output.manifest-content" \
    "manifest/version is not recognized"

  reset_isolated_profile
  printf '%s\n' '#error "unmanifested profile file"' \
    >"$isolated_profile/cooperative_groups/scan.h"
  run_isolated_profile_rejection \
    "$output.extra-file" \
    "layout does not match its closed manifest"

  reset_isolated_profile
  rm -- "$isolated_profile/cooperative_groups/reduce.h"
  run_isolated_profile_rejection \
    "$output.missing-poison" \
    "missing required regular file 'cooperative_groups/reduce.h'"

  missing_output="$output.missing-profile"
  rm -rf -- "$isolated_profile"
  if "$isolated_binary" "$input" \
      --frontend-compat=ascify-admitted-v1 \
      "--cuda-path=$ASCIFY_CUDA_PATH" \
      "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
      -o "$missing_output" -- -std=c++17 \
      >"$missing_output.stdout" 2>"$missing_output.stderr"; then
    echo "missing installed frontend compatibility profile unexpectedly succeeded" >&2
    exit 1
  fi
  if [ -e "$missing_output" ]; then
    echo "missing installed frontend profile published output" >&2
    exit 1
  fi
  if ! grep -F -- "installed frontend compatibility profile is missing" \
      "$missing_output.stderr" >/dev/null; then
    echo "missing installed frontend profile did not fail closed" >&2
    exit 1
  fi
}

reject_target_recipe() {
  input=$1
  output=$2
  target_policy=$3
  math_mode=$4
  recipe=$5
  expected_error=$6
  stderr_path="$output.stderr"
  if "$ASCIFY_BINARY" "$input" \
      "--target-policy=$target_policy" \
      "--simt-math=$math_mode" \
      "--target-recipe=$recipe" \
      "--cuda-path=$ASCIFY_CUDA_PATH" \
      "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
      -o "$output" \
      -- "-I$repo_root/tests/support/cuda" -std=c++17 \
      >"$output.stdout" 2>"$stderr_path"; then
    echo "target recipe combination unexpectedly succeeded: $recipe" >&2
    exit 1
  fi
  if ! grep -F -- "$expected_error" "$stderr_path" >/dev/null; then
    echo "target recipe rejection did not report: $expected_error" >&2
    exit 1
  fi
}

translate_rowwise_recipe_fixture() {
  input=$1
  output=$2
  shift 2
  run_check "translate $(basename "$input") (explicit row-wise SIMD recipe)" \
    "$python" -I -B "$repo_root/tools/generate_rowwise_cce.py" \
      "$input" "$output" \
      --mode=simd-simt \
      "--ascify-clang=$ASCIFY_BINARY" \
      "--report=$output.report.json" \
      "--cuda-path=$ASCIFY_CUDA_PATH" \
      "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
      "--include=$repo_root/tests/fixtures/oneflow" \
      --compiler-arg=-std=c++17 \
      --require-recipe \
      -- "$@"
}

reject_rowwise_recipe_fixture() {
  input=$1
  output=$2
  shift 2
  if "$python" -I -B "$repo_root/tools/generate_rowwise_cce.py" \
      "$input" "$output" \
      --mode=simd-simt \
      "--ascify-clang=$ASCIFY_BINARY" \
      "--cuda-path=$ASCIFY_CUDA_PATH" \
      "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
      "--include=$repo_root/tests/fixtures/oneflow" \
      --compiler-arg=-std=c++17 \
      --require-recipe \
      "$@" \
      >"$output.stdout" 2>"$output.stderr"; then
    echo "invalid row-wise primitive branch unexpectedly proved SIMD semantics" >&2
    exit 1
  fi
  if [ -e "$output" ] || [ -e "$output.report.json" ]; then
    echo "rejected row-wise primitive branch published output" >&2
    exit 1
  fi
  if ! grep -F "structurally valid SIMD dispatch" "$output.stderr" >/dev/null; then
    echo "invalid row-wise primitive branch did not fail at the SIMD proof gate" >&2
    exit 1
  fi
}

reject_rowwise_recipe_frontend() {
  input=$1
  output=$2
  expected_error=$3
  shift 3
  if "$python" -I -B "$repo_root/tools/generate_rowwise_cce.py" \
      "$input" "$output" \
      --mode=simd-simt \
      "--ascify-clang=$ASCIFY_BINARY" \
      "--cuda-path=$ASCIFY_CUDA_PATH" \
      "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
      "--include=$repo_root/tests/fixtures/oneflow" \
      --compiler-arg=-std=c++17 \
      --require-recipe \
      "$@" \
      >"$output.stdout" 2>"$output.stderr"; then
    echo "conflicting row-wise SIMD frontend input unexpectedly succeeded" >&2
    exit 1
  fi
  if [ -e "$output" ] || [ -e "$output.report.json" ]; then
    echo "rejected row-wise SIMD frontend input published output" >&2
    exit 1
  fi
  if ! grep -F -- "$expected_error" "$output.stderr" >/dev/null; then
    echo "row-wise SIMD frontend rejection did not report: $expected_error" >&2
    exit 1
  fi
}

check_recipe_report() {
  report=$1
  expected_softmax=$2
  expected_rmsnorm=$3
  "$python" - "$report" "$expected_softmax" "$expected_rmsnorm" <<'PY'
import json
from pathlib import Path
import sys

report_path = Path(sys.argv[1])
expected_softmax = int(sys.argv[2])
expected_rmsnorm = int(sys.argv[3])
data = json.loads(report_path.read_text(encoding="utf-8"))
expected = {
    "dispatch_execution_kind": "intra-kernel-simd-simt",
    "dispatch_function_count": expected_softmax + expected_rmsnorm,
    "dispatch_with_simt_fallback_count": expected_softmax + expected_rmsnorm,
    "recipe_call_token_count": expected_softmax + expected_rmsnorm,
    "recognized": True,
    "schema": "ascify.rowwise-cce-report.v3",
    "simd_header_count": 1,
    "try_softmax_hybrid_count": expected_softmax,
    "try_softmax_simd_count": expected_softmax,
    "try_rmsnorm_hybrid_count": expected_rmsnorm,
    "try_rmsnorm_simd_count": expected_rmsnorm,
}
for key, value in expected.items():
    if data.get(key) != value:
        raise SystemExit(
            f"{report_path}: expected {key}={value!r}, got {data.get(key)!r}"
        )
if not isinstance(data.get("input_kernel_launch_count"), int) \
        or data["input_kernel_launch_count"] <= 0:
    raise SystemExit(f"{report_path}: input SIMT kernel launch not proven")
if data.get("kernel_launch_count") != data["input_kernel_launch_count"]:
    raise SystemExit(f"{report_path}: retained SIMT kernel launch not proven")
for key in ("ascify_binary_sha256", "generator_sha256"):
    value = data.get(key)
    if not isinstance(value, str) or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise SystemExit(f"{report_path}: missing provenance digest {key}")
conversion_id = data.get("conversion_id", "")
if not isinstance(conversion_id, str) \
        or not conversion_id.startswith("sha256:") \
        or len(conversion_id) != 71:
    raise SystemExit(f"{report_path}: invalid conversion_id")
normalized = data.get("normalized_converter_argv")
if not isinstance(normalized, list) or normalized[:2] != [
    "$ASCIFY_CLANG", "$SOURCE"
] or "$OUTPUT" not in normalized:
    raise SystemExit(f"{report_path}: normalized converter argv is incomplete")
PY
}

require_fixed_count() {
  needle=$1
  file=$2
  expected=$3
  if command -v rg >/dev/null 2>&1; then
    actual=$( (rg -F -o -- "$needle" "$file" || true) |
      wc -l | tr -d ' ' )
  else
    actual=$( (grep -F -o -- "$needle" "$file" || true) |
      wc -l | tr -d ' ' )
  fi
  if [ "$actual" -ne "$expected" ]; then
    echo "expected $expected occurrences of '$needle' in $file, found $actual" >&2
    exit 1
  fi
}

check_rowwise_recipe_frontend() {
  softmax_output=$1
  rmsnorm_output=$2
  simd_header='#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>'

  require_fixed_count "$simd_header" "$softmax_output" 1
  require_fixed_count \
    '#pragma push_macro("ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_")' \
    "$softmax_output" 0
  require_fixed_count \
    '::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(' "$softmax_output" 3
  require_fixed_count \
    '::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TryRmsNormHybrid(' "$softmax_output" 0

  require_fixed_count "$simd_header" "$rmsnorm_output" 1
  require_fixed_count \
    '#pragma push_macro("ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_")' \
    "$rmsnorm_output" 0
  require_fixed_count \
    '::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(' "$rmsnorm_output" 0
  require_fixed_count \
    '::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TryRmsNormHybrid(' "$rmsnorm_output" 3

  "$python" - "$softmax_output" "$rmsnorm_output" "$simd_header" <<'PY'
from pathlib import Path
import re
import sys

for value in sys.argv[1:3]:
    lines = Path(value).read_text(encoding="utf-8").splitlines()
    try:
        header_index = lines.index(sys.argv[3])
    except ValueError:
        raise SystemExit(f"{value}: SIMD recipe include is missing")
    try:
        namespace_index = next(
            index for index, line in enumerate(lines)
            if line.strip().startswith("namespace oneflow")
        )
    except StopIteration:
        raise SystemExit(f"{value}: translated OneFlow namespace is missing")
    if header_index >= namespace_index:
        raise SystemExit(f"{value}: SIMD recipe include is not at global scope")


def recipe_blocks(path, expected_call, expected_count):
    text = Path(path).read_text(encoding="utf-8")
    start = re.compile(
        r'#pragma push_macro\("(?P<result>'
        r'ascify_dav_c310_recipe_result(?:_generated(?:_[0-9]+)?)?)"\)\n'
        r'#undef (?P=result)\n'
    )
    blocks = []
    for match in start.finditer(text):
        end_marker = '#pragma pop_macro("{}")'.format(match.group("result"))
        end = text.find(end_marker, match.end())
        if end < 0:
            raise SystemExit(
                f"{path}: generated result {match.group('result')} has no pop_macro"
            )
        block = text[match.start():end + len(end_marker)]
        if expected_call in block:
            blocks.append(block)
    if len(blocks) != expected_count:
        raise SystemExit(
            f"{path}: expected {expected_count} guarded {expected_call} blocks, "
            f"found {len(blocks)}"
        )
    return blocks


def require_macro_triplet(path, block, name):
    directives = (
        f'#pragma push_macro("{name}")',
        f'#undef {name}',
        f'#pragma pop_macro("{name}")',
    )
    lines = block.splitlines()
    counts = tuple(lines.count(item) for item in directives)
    if counts != (1, 1, 1):
        raise SystemExit(
            f"{path}: generated proof identifier {name!r} is not protected "
            f"by exactly one push/undef/pop triplet: {counts}"
        )
    positions = tuple(lines.index(item) for item in directives)
    if not positions[0] < positions[1] < positions[2]:
        raise SystemExit(
            f"{path}: generated proof identifier {name!r} has an invalid "
            "push/undef/pop order"
        )


softmax_blocks = recipe_blocks(
    sys.argv[1], "RowwiseHybridFacadeV1::TrySoftmaxHybrid(", 3
)
for block in softmax_blocks:
    names = [
        "ascify_dav_c310_recipe_result",
        "stream", "load", "store", "rows", "cols", "ComputeType",
        "pack_size", "algorithm", "kSoftmax", "oneflow", "cuda",
        "softmax", "Algorithm",
    ]
    if "cols_per_thread" in block:
        names.extend(
            ["cols_per_thread", "thread_group_width", "rows_per_access", "padding"]
        )
    if "(smem >= 0)" in block:
        names.append("smem")
    for name in names:
        require_macro_triplet(sys.argv[1], block, name)

rmsnorm_blocks = recipe_blocks(
    sys.argv[2], "RowwiseHybridFacadeV1::TryRmsNormHybrid(", 3
)
for block in rmsnorm_blocks:
    names = [
        "ascify_dav_c310_recipe_result",
        "stream", "load", "store", "nrow", "ncol", "eps", "inv_rms",
        "ComputeType", "pack_size",
    ]
    if "max_cols_per_thread" in block:
        names.extend(
            [
                "max_cols_per_thread", "min_cols_per_thread",
                "thread_group_width", "rows_per_access", "padding",
            ]
        )
    if "(smem_size >= 0)" in block:
        names.append("smem_size")
    for name in names:
        require_macro_triplet(sys.argv[2], block, name)
PY

  check_recipe_report "$softmax_output.report.json" 3 0
  check_recipe_report "$rmsnorm_output.report.json" 0 3
}

check_injected_rowwise_header_publication() {
  translated_output=$1
  probe=$2
  "$python" - "$translated_output" "$probe" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
target_include = "#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>"
try:
    include_index = source.index(target_include)
except ValueError:
    raise SystemExit(f"{sys.argv[1]}: injected row-wise SIMD include is absent")

begin = include_index - 1
while begin >= 0 and (
        source[begin].startswith("#pragma push_macro(")
        or source[begin].startswith("#undef ")
        or source[begin] == "#include <ascify/ascify_cuda_compat.hpp>"):
    begin -= 1
begin += 1
if begin == include_index:
    raise SystemExit(f"{sys.argv[1]}: injected row-wise SIMD include has no shield")

end = include_index + 1
while end < len(source) and source[end].startswith("#pragma pop_macro("):
    end += 1
if end == include_index + 1:
    raise SystemExit(f"{sys.argv[1]}: injected row-wise SIMD include has no shield restore")

published = (
    "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_SELECTORS_V1_HPP_",
    "ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_H_",
)
probe = [
    "#include <cmath>",
    "#define ASCIFY_TEST_STUB_MATH_FUNCTIONS_H",
    "#define __aicore__",
    "inline float fdividef(float n, float d) { return n / d; }",
]
probe.extend(source[begin:end])
for macro in published:
    probe.extend(
        [
            f"#ifndef {macro}",
            f'#error "injected header did not publish {macro}"',
            "#endif",
        ]
    )
probe.extend(
    [
        "#if ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION != 1",
        '#error "injected header did not publish row-wise SIMD ABI v1"',
        "#endif",
        target_include,
        "static_assert(ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION == 1,",
        '              "repeat include changed the row-wise SIMD ABI");',
        "int main() { return 0; }",
    ]
)
Path(sys.argv[2]).write_text("\n".join(probe) + "\n", encoding="utf-8")
PY

  cxx=${CXX:-c++}
  if ! command -v "$cxx" >/dev/null 2>&1; then
    echo "C++ compiler not found for injected-header probe: $cxx" >&2
    exit 1
  fi
  "$cxx" -std=c++17 -fsyntax-only \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" \
    "$probe"
}

check_outer_guard_collision_output() {
  translated_output=$1
  preprocessed_output=$2
  expected_body=$3
  guard=ASCIFY_TEST_STUB_ACL_H
  syntax_probe="$preprocessed_output.guard_syntax.cpp"

  require_fixed_count "#define $guard" "$translated_output" 1
  require_fixed_count "#pragma push_macro(\"$guard\")" \
    "$translated_output" 1
  require_fixed_count "#undef $guard" "$translated_output" 1
  require_fixed_count "#pragma pop_macro(\"$guard\")" \
    "$translated_output" 1

  "$python" - "$translated_output" "$guard" "$syntax_probe" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
guard_define = text.index(f"#define {sys.argv[2]}")
guard_push = text.index(f'#pragma push_macro("{sys.argv[2]}")')
target_include = text.index(
    "#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>"
)
guard_pop = text.index(f'#pragma pop_macro("{sys.argv[2]}")')
namespace = text.index("namespace oneflow")
if not guard_push < target_include < guard_pop < guard_define < namespace:
    raise SystemExit(
        f"{sys.argv[1]}: outer guard is not isolated around the global SIMD include"
    )

define_line_end = text.find("\n", guard_define)
if define_line_end < 0:
    raise SystemExit(f"{sys.argv[1]}: outer guard define has no line ending")
prefix = text[:define_line_end + 1]
probe = (
    "#include <cmath>\n"
    "#define ASCIFY_TEST_STUB_MATH_FUNCTIONS_H\n"
    "#define __aicore__\n"
    "inline float fdividef(float n, float d) { return n / d; }\n"
    + prefix
    + "int ascify_outer_guard_sentinel = "
      "ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION;\n"
    + "#endif\n"
    + "int main() { return ascify_outer_guard_sentinel == 1 ? 0 : 1; }\n"
)
Path(sys.argv[3]).write_text(probe, encoding="utf-8")
PY

  cxx=${CXX:-c++}
  "$cxx" -std=c++17 -fsyntax-only \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/include" \
    "$syntax_probe"
  "$cxx" -std=c++17 -E -P -x c++ \
    -I"$repo_root/tests/rewrite/stubs" \
    -I"$repo_root/tests/fixtures/oneflow" \
    -I"$repo_root/include" \
    -I"$repo_root" \
    -I"$ASCIFY_CUDA_PATH/include" \
    "$translated_output" -o "$preprocessed_output"
  if ! grep -F "namespace oneflow" "$preprocessed_output" >/dev/null; then
    echo "$translated_output: colliding guard removed the OneFlow namespace" >&2
    exit 1
  fi
  if ! grep -F "$expected_body" "$preprocessed_output" >/dev/null; then
    echo "$translated_output: colliding guard removed $expected_body" >&2
    exit 1
  fi
}

warp_fast="$work_dir/warp_reduce.fast.cce"
warp_default="$work_dir/warp_reduce.default.cce"
reducer_fast="$work_dir/canonical_reducer.fast.cce"
reducer_default="$work_dir/canonical_reducer.default.cce"
simt_translated="$work_dir/simt_compat.cce"
target_abi_translated="$work_dir/target_abi_compat.cce"
cooperative_groups_translated="$work_dir/cooperative_groups_compat.cce"
recipe_fast="$work_dir/dav_c310_recipe.fast.cce"
recipe_portable_precise="$work_dir/dav_c310_recipe.portable_precise.cce"
recipe_portable_fast="$work_dir/dav_c310_recipe.portable_fast.cce"
recipe_dav_precise="$work_dir/dav_c310_recipe.dav_precise.cce"
oneflow_softmax_recipe="$work_dir/oneflow_softmax.recipe_simd.cce"
oneflow_rmsnorm_recipe="$work_dir/oneflow_rmsnorm.recipe_simd.cce"
annotated_shortcut_input="$work_dir/oneflow_softmax.annotated_shortcut.cuh"
annotated_shortcut_output="$work_dir/oneflow_softmax.annotated_shortcut.cce"
annotated_status_input="$work_dir/oneflow_softmax.annotated_status.cuh"
annotated_status_output="$work_dir/oneflow_softmax.annotated_status.cce"
annotated_status_legacy_output="$work_dir/oneflow_softmax.annotated_status.legacy.cce"
unannotated_status_input="$work_dir/oneflow_softmax.unannotated_status.cuh"
unannotated_status_legacy_output="$work_dir/oneflow_softmax.unannotated_status.legacy.cce"
inactive_fast_primitive_input="$work_dir/oneflow_softmax.inactive_fast_primitive.cuh"
inactive_fast_primitive_output="$work_dir/oneflow_softmax.inactive_fast_primitive.cce"
inactive_precise_primitive_input="$work_dir/oneflow_softmax.inactive_precise_primitive.cuh"
inactive_precise_primitive_output="$work_dir/oneflow_softmax.inactive_precise_primitive.cce"
macro_poison_input="$work_dir/oneflow_softmax.macro_poison.cuh"
macro_poison_output="$work_dir/oneflow_softmax.macro_poison.cce"
semantic_macro_input="$work_dir/oneflow_softmax.semantic_macro.cuh"
semantic_macro_output="$work_dir/oneflow_softmax.semantic_macro.cce"
command_macro_input="$work_dir/oneflow_softmax.command_macro.cuh"
command_macro_output="$work_dir/oneflow_softmax.command_macro.cce"
hybrid_result_macro_input="$work_dir/oneflow_softmax.hybrid_result_macro.cuh"
hybrid_result_macro_output="$work_dir/oneflow_softmax.hybrid_result_macro.cce"
command_keyword_input="$work_dir/oneflow_softmax.command_keyword.cuh"
command_keyword_output="$work_dir/oneflow_softmax.command_keyword.cce"
softmax_dispatch_overload_input="$work_dir/oneflow_softmax.dispatch_overload.cuh"
softmax_dispatch_overload_output="$work_dir/oneflow_softmax.dispatch_overload.cce"
rmsnorm_dispatch_overload_input="$work_dir/oneflow_rmsnorm.dispatch_overload.cuh"
rmsnorm_dispatch_overload_output="$work_dir/oneflow_rmsnorm.dispatch_overload.cce"
softmax_marker_poison_input="$work_dir/oneflow_softmax.marker_poison.cuh"
softmax_marker_poison_output="$work_dir/oneflow_softmax.marker_poison.cce"
rmsnorm_marker_poison_input="$work_dir/oneflow_rmsnorm.marker_poison.cuh"
rmsnorm_marker_poison_output="$work_dir/oneflow_rmsnorm.marker_poison.cce"
softmax_abi_poison_input="$work_dir/oneflow_softmax.abi_poison.cuh"
softmax_abi_poison_output="$work_dir/oneflow_softmax.abi_poison.cce"
rmsnorm_abi_poison_input="$work_dir/oneflow_rmsnorm.abi_poison.cuh"
rmsnorm_abi_poison_output="$work_dir/oneflow_rmsnorm.abi_poison.cce"
softmax_asm_abi_poison_input="$work_dir/oneflow_softmax.asm_abi_poison.cuh"
softmax_asm_abi_poison_output="$work_dir/oneflow_softmax.asm_abi_poison.cce"
rmsnorm_asm_abi_poison_input="$work_dir/oneflow_rmsnorm.asm_abi_poison.cuh"
rmsnorm_asm_abi_poison_output="$work_dir/oneflow_rmsnorm.asm_abi_poison.cce"
softmax_file_asm_input="$work_dir/softmax.file_scope_asm.cuh"
softmax_file_asm_output="$work_dir/softmax.file_scope_asm.cce"
rmsnorm_block_asm_input="$work_dir/rmsnorm.block_scope_asm.cuh"
rmsnorm_block_asm_output="$work_dir/rmsnorm.block_scope_asm.cce"
skipped_asm_dependency="$work_dir/rowwise_skipped_asm.hpp"
skipped_asm_dependency_input="$work_dir/rmsnorm.skipped_dependency_asm.cuh"
skipped_asm_dependency_output="$work_dir/rmsnorm.skipped_dependency_asm.cce"
softmax_facade_declaration_input="$work_dir/oneflow_softmax.facade_declaration.cuh"
softmax_facade_declaration_output="$work_dir/oneflow_softmax.facade_declaration.cce"
rmsnorm_facade_specialization_input="$work_dir/oneflow_rmsnorm.facade_specialization.cuh"
rmsnorm_facade_specialization_output="$work_dir/oneflow_rmsnorm.facade_specialization.cce"
facade_specialization_prelude="$work_dir/facade_specialization_prelude.hpp"
softmax_header_guard_poison_input="$work_dir/oneflow_softmax.header_guard_poison.cuh"
softmax_header_guard_poison_output="$work_dir/oneflow_softmax.header_guard_poison.cce"
rmsnorm_abi_version_poison_input="$work_dir/oneflow_rmsnorm.abi_version_poison.cuh"
rmsnorm_abi_version_poison_output="$work_dir/oneflow_rmsnorm.abi_version_poison.cce"
rowwise_header_reinclude_probe="$work_dir/rowwise_header_reinclude_probe.cpp"
softmax_guard_collision_input="$work_dir/oneflow_softmax.guard_collision.cuh"
softmax_guard_collision_output="$work_dir/oneflow_softmax.guard_collision.cce"
softmax_guard_collision_preprocessed="$work_dir/oneflow_softmax.guard_collision.ii"
rmsnorm_guard_collision_input="$work_dir/oneflow_rmsnorm.guard_collision.cuh"
rmsnorm_guard_collision_output="$work_dir/oneflow_rmsnorm.guard_collision.cce"
rmsnorm_guard_collision_preprocessed="$work_dir/oneflow_rmsnorm.guard_collision.ii"
keyword_dependency="$work_dir/rowwise_keyword_poison.hpp"
keyword_dependency_input="$work_dir/oneflow_softmax.keyword_dependency.cuh"
keyword_dependency_output="$work_dir/oneflow_softmax.keyword_dependency.cce"

"$python" - \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh" \
  "$annotated_shortcut_input" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
needle = """template<>
__inline__ __device__ float Exp<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __expf(x);
#else
  return exp(x);
#endif
}
"""
replacement = """template<>
__attribute__((annotate("ascify.semantic.exp")))
__inline__ __device__ float Exp<float>(float x) {
  return x;
}
"""
if source.count(needle) != 1:
    raise SystemExit("Softmax semantic-shortcut mutation source drift")
Path(sys.argv[2]).write_text(source.replace(needle, replacement), encoding="utf-8")
PY

"$python" - \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh" \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/rms_norm.cuh" \
  "$softmax_guard_collision_input" \
  "$rmsnorm_guard_collision_input" <<'PY'
from pathlib import Path
import sys

colliding_guard = "ASCIFY_TEST_STUB_ACL_H"
for source_path, output_path, original_guard in (
    (sys.argv[1], sys.argv[3], "ONEFLOW_CORE_CUDA_SOFTMAX_H_"),
    (sys.argv[2], sys.argv[4], "ONEFLOW_CORE_CUDA_RMS_NORM_H_"),
):
    source = Path(source_path).read_text(encoding="utf-8")
    if source.count(original_guard) < 2:
        raise SystemExit(f"outer-guard mutation source drift: {source_path}")
    mutated = source.replace(original_guard, colliding_guard)
    if output_path == sys.argv[4]:
        define = f"#define {colliding_guard}\n"
        if mutated.count(define) != 1:
            raise SystemExit(f"valued outer-guard mutation drift: {source_path}")
        mutated = mutated.replace(define, f"#define {colliding_guard} 1\n")
    Path(output_path).write_text(mutated, encoding="utf-8")
PY

"$python" - \
  "$softmax_file_asm_input" \
  "$rmsnorm_block_asm_input" \
  "$skipped_asm_dependency" \
  "$skipped_asm_dependency_input" <<'PY'
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(
    'asm(".globl ascify950_softmax_reg_recompute_launch_v1\\n"\n'
    '    ".set forged_softmax_launch, '
    'ascify950_softmax_reg_recompute_launch_v1");\n'
    "int softmax_file_scope_asm_control;\n",
    encoding="utf-8",
)
Path(sys.argv[2]).write_text(
    "inline void rmsnorm_block_scope_asm_control() {\n"
    '  asm volatile(".globl ascify950_rmsnorm_reg_cached_launch_v1");\n'
    "}\n",
    encoding="utf-8",
)
Path(sys.argv[3]).write_text(
    "#if 0\n"
    '__asm__(".globl ascify950_rmsnorm_reg_plain_rowbatch_launch_v1");\n'
    "#endif\n",
    encoding="utf-8",
)
Path(sys.argv[4]).write_text(
    '#include "rowwise_skipped_asm.hpp"\n'
    "int rmsnorm_skipped_dependency_asm_control;\n",
    encoding="utf-8",
)
PY

"$python" - \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh" \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/rms_norm.cuh" \
  "$softmax_facade_declaration_input" \
  "$rmsnorm_facade_specialization_input" \
  "$facade_specialization_prelude" \
  "$softmax_header_guard_poison_input" \
  "$rmsnorm_abi_version_poison_input" <<'PY'
from pathlib import Path
import sys

softmax = Path(sys.argv[1]).read_text(encoding="utf-8")
rmsnorm = Path(sys.argv[2]).read_text(encoding="utf-8")

Path(sys.argv[3]).write_text(
    softmax + "\nstruct RowwiseHybridFacadeV1 {};\n", encoding="utf-8"
)
Path(sys.argv[5]).write_text(
    "template<typename T> struct RowwiseHybridFacadeV1;\n", encoding="utf-8"
)
Path(sys.argv[4]).write_text(
    rmsnorm + "\ntemplate<> struct RowwiseHybridFacadeV1<int> {};\n",
    encoding="utf-8",
)

Path(sys.argv[6]).write_text(
    "#if 0\n"
    "#define ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_\n"
    "#define ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_SELECTORS_V1_HPP_\n"
    "#define ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_H_\n"
    "#endif\n"
    + softmax,
    encoding="utf-8",
)
Path(sys.argv[7]).write_text(
    "#define ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION 999\n"
    "#undef ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION\n"
    + rmsnorm,
    encoding="utf-8",
)
PY

"$python" - \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh" \
  "$annotated_status_input" \
  "$unannotated_status_input" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
for name in (
    "LaunchSoftmaxWarpImpl",
    "LaunchSoftmaxBlockSMemImpl",
    "LaunchSoftmaxBlockUncachedImpl",
):
    needle = f"inline cudaError_t {name}"
    replacement = (
        '__attribute__((annotate("ascify.semantic.status")))\n'
        f"inline int {name}"
    )
    if source.count(needle) != 1:
        raise SystemExit(f"Softmax status-annotation mutation drift: {name}")
    source = source.replace(needle, replacement)
    signature = source.index(f"inline int {name}")
    body_begin = source.index("{", signature)
    depth = 0
    body_end = None
    for position in range(body_begin, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                body_end = position + 1
                break
    if body_end is None:
        raise SystemExit(f"Softmax status wrapper body drift: {name}")
    body = source[body_begin:body_end]
    local_status = "cudaError_t err = GetNumBlocks("
    if body.count(local_status) != 1:
        raise SystemExit(f"Softmax status local mutation drift: {name}")
    body = body.replace(local_status, "int err = GetNumBlocks(")
    source = source[:body_begin] + body + source[body_end:]
Path(sys.argv[2]).write_text(source, encoding="utf-8")
attribute = '__attribute__((annotate("ascify.semantic.status")))\n'
if source.count(attribute) != 3:
    raise SystemExit("Softmax status annotation cardinality drift")
Path(sys.argv[3]).write_text(source.replace(attribute, ""), encoding="utf-8")
PY

"$python" - \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh" \
  "$inactive_fast_primitive_input" \
  "$inactive_precise_primitive_input" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
primitive = """template<>
__inline__ __device__ float Exp<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __expf(x);
#else
  return exp(x);
#endif
}
"""
if source.count(primitive) != 1:
    raise SystemExit("Softmax conditional primitive mutation source drift")
Path(sys.argv[2]).write_text(
    source.replace(
        primitive, primitive.replace("  return __expf(x);", "  return x;")
    ),
    encoding="utf-8",
)
Path(sys.argv[3]).write_text(
    source.replace(
        primitive, primitive.replace("  return exp(x);", "  return x;")
    ),
    encoding="utf-8",
)
PY

"$python" - \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh" \
  "$macro_poison_input" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
first_include = source.index("#include")
source = (
    source[:first_include]
    + "#define ascify ascify_poisoned_namespace\n"
    + "#define TrySoftmaxSimd(...) ascify_poisoned_try(__VA_ARGS__)\n"
    + "#define ascify_target_data ascify_poisoned_data\n"
    + source[first_include:]
)
namespace = source.index("namespace oneflow")
source = (
    source[:namespace]
    + "#define handled ascify_poisoned_handled\n"
    + "#define status ascify_poisoned_status\n"
    + source[namespace:]
)
Path(sys.argv[2]).write_text(source, encoding="utf-8")
PY

"$python" - \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh" \
  "$command_macro_input" \
  "$hybrid_result_macro_input" \
  "$command_keyword_input" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
Path(sys.argv[2]).write_text(
    "#undef TrySoftmaxHybrid\n" + source, encoding="utf-8"
)
Path(sys.argv[3]).write_text(
    "#undef HybridTryResult\n" + source, encoding="utf-8"
)
Path(sys.argv[4]).write_text("#undef true\n" + source, encoding="utf-8")
PY

"$python" - \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh" \
  "$semantic_macro_input" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
first_include = source.index("#include")
source = (
    source[:first_include]
    + "#if ASCIFY_FUTURE_BUILD\n"
    + "#define expf(x) (x)\n"
    + "#endif\n"
    + source[first_include:]
)
Path(sys.argv[2]).write_text(source, encoding="utf-8")
PY

"$python" - \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh" \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/rms_norm.cuh" \
  "$softmax_dispatch_overload_input" \
  "$rmsnorm_dispatch_overload_input" \
  "$softmax_marker_poison_input" \
  "$rmsnorm_marker_poison_input" \
  "$softmax_abi_poison_input" \
  "$rmsnorm_abi_poison_input" \
  "$softmax_asm_abi_poison_input" \
  "$rmsnorm_asm_abi_poison_input" \
  "$keyword_dependency" \
  "$keyword_dependency_input" <<'PY'
from pathlib import Path
import sys

softmax = Path(sys.argv[1]).read_text(encoding="utf-8")
rmsnorm = Path(sys.argv[2]).read_text(encoding="utf-8")

softmax_overload = r'''
namespace ascify { namespace target { namespace dav_c310 {
struct ForgedSoftmaxResult { bool handled; cudaError_t status; };
template<typename Load, typename Store>
inline ForgedSoftmaxResult TrySoftmaxSimd(
    cudaStream_t, const Load&, const Store&, int64_t, int64_t, float*) {
  return {true, cudaSuccess};
}
}}}  // namespace ascify::target::dav_c310
'''
rmsnorm_overload = r'''
namespace ascify { namespace target { namespace dav_c310 {
struct ForgedRmsNormResult { bool handled; cudaError_t status; };
template<typename Load, typename Store>
inline ForgedRmsNormResult TryRmsNormSimd(
    cudaStream_t, const Load&, const Store&, int64_t, int64_t, double,
    float*, float*) {
  return {true, cudaSuccess};
}
}}}  // namespace ascify::target::dav_c310
'''
Path(sys.argv[3]).write_text(softmax + softmax_overload, encoding="utf-8")
Path(sys.argv[4]).write_text(rmsnorm + rmsnorm_overload, encoding="utf-8")

marker = "struct DirectLoad {"
poisoned = marker + "\n  using ascify_target_direct_load_tag = void;"
if softmax.count(marker) != 1:
    raise SystemExit("Softmax DirectLoad marker-poison mutation source drift")
Path(sys.argv[5]).write_text(
    softmax.replace(marker, poisoned), encoding="utf-8"
)
Path(sys.argv[6]).write_text(
    rmsnorm
    + "\nstruct PoisonedRmsAdapter {\n"
      "  using ascify_target_direct_load_tag = void;\n"
      "};\n",
    encoding="utf-8",
)

Path(sys.argv[7]).write_text(
    softmax
    + '\nextern "C" int ascify950_softmax_reg_recompute_launch_v1();\n',
    encoding="utf-8",
)
Path(sys.argv[8]).write_text(
    rmsnorm
    + '\nextern "C" int ascify950_rmsnorm_reg_cached_launch_v1();\n',
    encoding="utf-8",
)
Path(sys.argv[9]).write_text(
    softmax
    + '\nextern "C" int forged_softmax_abi() '
      '__asm__("ascify950_softmax_reg_recompute_launch_v1");\n',
    encoding="utf-8",
)
Path(sys.argv[10]).write_text(
    rmsnorm
    + '\nextern "C" int forged_rmsnorm_abi() '
      '__asm__("ascify950_rmsnorm_reg_cached_launch_v1");\n',
    encoding="utf-8",
)

Path(sys.argv[11]).write_text(
    "#if CUDA_VERSION == 12000\n"
    "#define if(condition) if (true)\n"
    "#endif\n",
    encoding="utf-8",
)
first_include = softmax.index("#include")
Path(sys.argv[12]).write_text(
    softmax[:first_include]
    + '#include "rowwise_keyword_poison.hpp"\n'
    + softmax[first_include:],
    encoding="utf-8",
)
PY

translate_fixture \
  "$rewrite_dir/warp_reduce_input.cu" "$warp_fast" dav-c310-vec fast
translate_fixture \
  "$rewrite_dir/warp_reduce_input.cu" "$warp_default" portable precise
translate_fixture \
  "$rewrite_dir/canonical_reducer_input.cu" "$reducer_fast" dav-c310-vec fast
translate_fixture \
  "$rewrite_dir/canonical_reducer_input.cu" "$reducer_default" portable precise
translate_fixture \
  "$rewrite_dir/simt_compat_input.cu" "$simt_translated" portable precise
translate_fixture \
  "$rewrite_dir/target_abi_compat_input.cu" \
  "$target_abi_translated" portable precise
translate_fixture \
  "$rewrite_dir/cooperative_groups_compat_input.cu" \
  "$cooperative_groups_translated" portable precise \
  --frontend-compat=ascify-admitted-v1
reject_frontend_compat_fixture \
  "$repo_root/tests/frontend_compat/reduce_input.cu" \
  "$work_dir/frontend_compat_reduce.cce" ascify-admitted-v1 \
  "does not admit cooperative_groups/reduce.h"
reject_frontend_compat_fixture \
  "$repo_root/tests/frontend_compat/block_sync_input.cu" \
  "$work_dir/frontend_compat_unknown.cce" unknown \
  "unsupported --frontend-compat value 'unknown'"
reject_frontend_compat_fixture \
  "$repo_root/tests/frontend_compat/unknown_memcpy_async_input.cu" \
  "$work_dir/frontend_compat_memcpy_async.cce" ascify-admitted-v1 \
  "rejects unadmitted cooperative-groups header 'cooperative_groups/memcpy_async.h'"
reject_frontend_compat_fixture \
  "$repo_root/tests/frontend_compat/unknown_scan_input.cu" \
  "$work_dir/frontend_compat_scan.cce" ascify-admitted-v1 \
  "rejects unadmitted cooperative-groups header 'cooperative_groups/scan.h'"
reject_frontend_compat_fixture \
  "$repo_root/tests/frontend_compat/local_shadow_input.cu" \
  "$work_dir/frontend_compat_local_shadow.cce" ascify-admitted-v1 \
  "requires cooperative_groups.h from its verified profile" \
  -iquote "$repo_root/tests/frontend_compat/local_shadow"
reject_frontend_compat_fixture \
  "$repo_root/tests/frontend_compat/local_alias_spelling_input.cu" \
  "$work_dir/frontend_compat_local_alias_spelling.cce" ascify-admitted-v1 \
  "rejects unadmitted cooperative-groups header './cooperative_groups.h'" \
  -iquote "$repo_root/tests/frontend_compat/local_shadow"
reject_damaged_frontend_profile \
  "$repo_root/tests/frontend_compat/block_sync_input.cu" \
  "$work_dir/frontend_compat_profile_reject.cce"
translate_fixture \
  "$rewrite_dir/dav_c310_recipe_input.cu" "$recipe_fast" dav-c310-vec fast
reject_target_recipe \
  "$rewrite_dir/dav_c310_recipe_input.cu" \
  "$work_dir/dav_c310_recipe.preinstrumented.cce" dav-c310-vec fast \
  dav-3510-rowwise-simd-v1 \
  "ascify_target_adapter_owner_type"
translate_fixture \
  "$rewrite_dir/dav_c310_recipe_input.cu" \
  "$recipe_portable_precise" portable precise
translate_fixture \
  "$rewrite_dir/dav_c310_recipe_input.cu" \
  "$recipe_portable_fast" portable fast
translate_fixture \
  "$rewrite_dir/dav_c310_recipe_input.cu" \
  "$recipe_dav_precise" dav-c310-vec precise
translate_rowwise_recipe_fixture \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/softmax.cuh" \
  "$oneflow_softmax_recipe"
translate_rowwise_recipe_fixture \
  "$repo_root/tests/fixtures/oneflow/oneflow/core/cuda/rms_norm.cuh" \
  "$oneflow_rmsnorm_recipe" \
  -include cuda_fp16.h -include cuda_bf16.h
translate_rowwise_recipe_fixture \
  "$softmax_guard_collision_input" \
  "$softmax_guard_collision_output"
translate_rowwise_recipe_fixture \
  "$rmsnorm_guard_collision_input" \
  "$rmsnorm_guard_collision_output" \
  -include cuda_fp16.h -include cuda_bf16.h
run_check "reject Softmax input-defined SIMD overload" \
  reject_rowwise_recipe_frontend \
    "$softmax_dispatch_overload_input" \
    "$softmax_dispatch_overload_output" \
    "reserved by the row-wise SIMD recipe"
run_check "reject RMSNorm input-defined SIMD overload" \
  reject_rowwise_recipe_frontend \
    "$rmsnorm_dispatch_overload_input" \
    "$rmsnorm_dispatch_overload_output" \
    "reserved by the row-wise SIMD recipe" \
    --compiler-arg=-include --compiler-arg=cuda_fp16.h \
    --compiler-arg=-include --compiler-arg=cuda_bf16.h
run_check "reject Softmax pre-instrumented adapter marker" \
  reject_rowwise_recipe_frontend \
    "$softmax_marker_poison_input" "$softmax_marker_poison_output" \
    "ascify_target_direct_load_tag"
run_check "reject RMSNorm pre-instrumented adapter marker" \
  reject_rowwise_recipe_frontend \
    "$rmsnorm_marker_poison_input" "$rmsnorm_marker_poison_output" \
    "ascify_target_direct_load_tag" \
    --compiler-arg=-include --compiler-arg=cuda_fp16.h \
    --compiler-arg=-include --compiler-arg=cuda_bf16.h
run_check "reject Softmax input-defined launch ABI" \
  reject_rowwise_recipe_frontend \
    "$softmax_abi_poison_input" "$softmax_abi_poison_output" \
    "ascify950_softmax_reg_recompute_launch_v1"
run_check "reject RMSNorm input-defined launch ABI" \
  reject_rowwise_recipe_frontend \
    "$rmsnorm_abi_poison_input" "$rmsnorm_abi_poison_output" \
    "ascify950_rmsnorm_reg_cached_launch_v1" \
    --compiler-arg=-include --compiler-arg=cuda_fp16.h \
    --compiler-arg=-include --compiler-arg=cuda_bf16.h
run_check "reject Softmax assembler-labeled launch ABI" \
  reject_rowwise_recipe_frontend \
    "$softmax_asm_abi_poison_input" "$softmax_asm_abi_poison_output" \
    "ascify950_softmax_reg_recompute_launch_v1"
run_check "reject RMSNorm assembler-labeled launch ABI" \
  reject_rowwise_recipe_frontend \
    "$rmsnorm_asm_abi_poison_input" "$rmsnorm_asm_abi_poison_output" \
    "ascify950_rmsnorm_reg_cached_launch_v1" \
    --compiler-arg=-include --compiler-arg=cuda_fp16.h \
    --compiler-arg=-include --compiler-arg=cuda_bf16.h
run_check "reject Softmax file-scope GNU asm" \
  reject_rowwise_recipe_frontend \
    "$softmax_file_asm_input" "$softmax_file_asm_output" \
    "input declaration or asm 'GNU asm' reserved"
run_check "reject RMSNorm block-scope GNU asm" \
  reject_rowwise_recipe_frontend \
    "$rmsnorm_block_asm_input" "$rmsnorm_block_asm_output" \
    "input declaration or asm 'GNU asm' reserved"
run_check "reject skipped non-system dependency GNU asm" \
  reject_rowwise_recipe_frontend \
    "$skipped_asm_dependency_input" "$skipped_asm_dependency_output" \
    "rejects raw input token 'GNU asm' reserved" \
    "--include=$work_dir"
run_check "reject direct protected façade declaration" \
  reject_rowwise_recipe_frontend \
    "$softmax_facade_declaration_input" \
    "$softmax_facade_declaration_output" \
    "rejects raw input token 'RowwiseHybridFacadeV1' reserved"
run_check "reject protected façade explicit specialization" \
  reject_rowwise_recipe_frontend \
    "$rmsnorm_facade_specialization_input" \
    "$rmsnorm_facade_specialization_output" \
    "rejects raw input token 'RowwiseHybridFacadeV1' reserved" \
    --compiler-arg=-include \
    "--compiler-arg=$facade_specialization_prelude" \
    --compiler-arg=-include --compiler-arg=cuda_fp16.h \
    --compiler-arg=-include --compiler-arg=cuda_bf16.h
run_check "reject skipped injected-header guards" \
  reject_rowwise_recipe_frontend \
    "$softmax_header_guard_poison_input" \
    "$softmax_header_guard_poison_output" \
    "rejects reserved macro 'ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_'"
run_check "reject redefined injected-header ABI version" \
  reject_rowwise_recipe_frontend \
    "$rmsnorm_abi_version_poison_input" \
    "$rmsnorm_abi_version_poison_output" \
    "rejects reserved macro 'ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION'" \
    --compiler-arg=-include --compiler-arg=cuda_fp16.h \
    --compiler-arg=-include --compiler-arg=cuda_bf16.h
run_check "reject inactive dependency keyword macro" \
  reject_rowwise_recipe_frontend \
    "$keyword_dependency_input" "$keyword_dependency_output" \
    "rejects reserved macro 'if'" \
    "--include=$work_dir"
if "$python" -I -B "$repo_root/tools/generate_rowwise_cce.py" \
    "$macro_poison_input" "$macro_poison_output" \
    --mode=simd-simt \
    "--ascify-clang=$ASCIFY_BINARY" \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    "--include=$repo_root/tests/fixtures/oneflow" \
    --compiler-arg=-std=c++17 \
    --require-recipe \
    >"$macro_poison_output.stdout" 2>"$macro_poison_output.stderr"; then
  echo "reserved-macro input unexpectedly generated SIMD output" >&2
  exit 1
fi
if [ -e "$macro_poison_output" ] \
    || [ -e "$macro_poison_output.report.json" ]; then
  echo "reserved-macro rejection published output" >&2
  exit 1
fi
if ! grep -F "rejects reserved macro 'ascify'" \
    "$macro_poison_output.stderr" >/dev/null; then
  echo "reserved-macro rejection did not name the conflicting macro" >&2
  exit 1
fi

if "$python" -I -B "$repo_root/tools/generate_rowwise_cce.py" \
    "$semantic_macro_input" "$semantic_macro_output" \
    --mode=simd-simt \
    "--ascify-clang=$ASCIFY_BINARY" \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    "--include=$repo_root/tests/fixtures/oneflow" \
    --compiler-arg=-std=c++17 \
    --require-recipe \
    >"$semantic_macro_output.stdout" \
    2>"$semantic_macro_output.stderr"; then
  echo "inactive semantic-callee macro unexpectedly generated SIMD output" >&2
  exit 1
fi
if [ -e "$semantic_macro_output" ] \
    || [ -e "$semantic_macro_output.report.json" ]; then
  echo "semantic-callee macro rejection published output" >&2
  exit 1
fi
if ! grep -F "rejects reserved macro 'expf'" \
    "$semantic_macro_output.stderr" >/dev/null; then
  echo "semantic-callee macro rejection did not name expf" >&2
  exit 1
fi

if "$python" -I -B "$repo_root/tools/generate_rowwise_cce.py" \
    "$command_macro_input" "$command_macro_output" \
    --mode=simd-simt \
    "--ascify-clang=$ASCIFY_BINARY" \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    "--include=$repo_root/tests/fixtures/oneflow" \
    --compiler-arg=-std=c++17 \
    --compiler-arg=-DTrySoftmaxHybrid=ascify_command_poison \
    --require-recipe \
    >"$command_macro_output.stdout" \
    2>"$command_macro_output.stderr"; then
  echo "command-line macro followed by source undef unexpectedly generated SIMD output" >&2
  exit 1
fi
if [ -e "$command_macro_output" ] \
    || [ -e "$command_macro_output.report.json" ]; then
  echo "command-line macro rejection published output" >&2
  exit 1
fi
if ! grep -F "rejects reserved macro 'TrySoftmaxHybrid'" \
    "$command_macro_output.stderr" >/dev/null; then
  echo "command-line macro rejection did not preserve definition history" >&2
  exit 1
fi

if "$python" -I -B "$repo_root/tools/generate_rowwise_cce.py" \
    "$hybrid_result_macro_input" "$hybrid_result_macro_output" \
    --mode=simd-simt \
    "--ascify-clang=$ASCIFY_BINARY" \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    "--include=$repo_root/tests/fixtures/oneflow" \
    --compiler-arg=-std=c++17 \
    --compiler-arg=-DHybridTryResult=ascify_result_poison \
    --require-recipe \
    >"$hybrid_result_macro_output.stdout" \
    2>"$hybrid_result_macro_output.stderr"; then
  echo "hybrid result macro unexpectedly generated SIMD output" >&2
  exit 1
fi
if [ -e "$hybrid_result_macro_output" ] \
    || [ -e "$hybrid_result_macro_output.report.json" ]; then
  echo "hybrid result macro rejection published output" >&2
  exit 1
fi
if ! grep -F "rejects reserved macro 'HybridTryResult'" \
    "$hybrid_result_macro_output.stderr" >/dev/null; then
  echo "hybrid result macro rejection did not preserve definition history" >&2
  exit 1
fi

if "$python" -I -B "$repo_root/tools/generate_rowwise_cce.py" \
    "$command_keyword_input" "$command_keyword_output" \
    --mode=simd-simt \
    "--ascify-clang=$ASCIFY_BINARY" \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    "--include=$repo_root/tests/fixtures/oneflow" \
    --compiler-arg=-std=c++17 \
    --compiler-arg=-Dtrue=true \
    --require-recipe \
    >"$command_keyword_output.stdout" \
    2>"$command_keyword_output.stderr"; then
  echo "command-line keyword macro followed by source undef unexpectedly generated SIMD output" >&2
  exit 1
fi
if [ -e "$command_keyword_output" ] \
    || [ -e "$command_keyword_output.report.json" ]; then
  echo "command-line keyword macro rejection published output" >&2
  exit 1
fi
if ! grep -F "rejects reserved macro 'true'" \
    "$command_keyword_output.stderr" >/dev/null; then
  echo "command-line keyword macro rejection did not preserve definition history" >&2
  exit 1
fi

run_check "translate annotated non-status wrappers (legacy control)" \
  "$ASCIFY_BINARY" "$annotated_status_input" \
    --target-policy=dav-c310-vec \
    --simt-math=fast \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    -o "$annotated_status_legacy_output" \
    -- "-I$repo_root/tests/fixtures/oneflow" -std=c++17
run_check "translate unannotated non-status wrappers (legacy control)" \
  "$ASCIFY_BINARY" "$unannotated_status_input" \
    --target-policy=dav-c310-vec \
    --simt-math=fast \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    -o "$unannotated_status_legacy_output" \
    -- "-I$repo_root/tests/fixtures/oneflow" -std=c++17
require_fixed_count \
  '::ascify::target::dav_c310::TrySoftmax(' \
  "$annotated_status_legacy_output" 3
require_fixed_count \
  '::ascify::target::dav_c310::TrySoftmax(' \
  "$unannotated_status_legacy_output" 0

if "$python" -I -B "$repo_root/tools/generate_rowwise_cce.py" \
    "$annotated_shortcut_input" "$annotated_shortcut_output" \
    --mode=simd-simt \
    "--ascify-clang=$ASCIFY_BINARY" \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    "--include=$repo_root/tests/fixtures/oneflow" \
    --compiler-arg=-std=c++17 \
    --require-recipe \
    >"$annotated_shortcut_output.stdout" \
    2>"$annotated_shortcut_output.stderr"; then
  echo "annotated identity helper unexpectedly proved SIMD semantics" >&2
  exit 1
fi
if [ -e "$annotated_shortcut_output" ] \
    || [ -e "$annotated_shortcut_output.report.json" ]; then
  echo "rejected annotated semantic shortcut published output" >&2
  exit 1
fi
if ! grep -F "structurally valid SIMD dispatch" \
    "$annotated_shortcut_output.stderr" >/dev/null; then
  echo "annotated semantic shortcut did not fail at the SIMD proof gate" >&2
  exit 1
fi

if "$python" -I -B "$repo_root/tools/generate_rowwise_cce.py" \
    "$annotated_status_input" "$annotated_status_output" \
    --mode=simd-simt \
    "--ascify-clang=$ASCIFY_BINARY" \
    "--cuda-path=$ASCIFY_CUDA_PATH" \
    "--clang-resource-directory=$ASCIFY_CLANG_RESOURCE_DIRECTORY" \
    "--include=$repo_root/tests/fixtures/oneflow" \
    --compiler-arg=-std=c++17 \
    --require-recipe \
    >"$annotated_status_output.stdout" \
    2>"$annotated_status_output.stderr"; then
  echo "annotated non-status wrappers unexpectedly proved SIMD status" >&2
  exit 1
fi
if [ -e "$annotated_status_output" ] \
    || [ -e "$annotated_status_output.report.json" ]; then
  echo "rejected annotated status shortcut published output" >&2
  exit 1
fi
if ! grep -F "structurally valid SIMD dispatch" \
    "$annotated_status_output.stderr" >/dev/null; then
  echo "annotated status shortcut did not fail at the SIMD proof gate" >&2
  exit 1
fi

run_check "reject invalid inactive fast-math primitive branch" \
  reject_rowwise_recipe_fixture \
    "$inactive_fast_primitive_input" "$inactive_fast_primitive_output"
run_check "reject invalid inactive precise primitive branch" \
  reject_rowwise_recipe_fixture \
    "$inactive_precise_primitive_input" "$inactive_precise_primitive_output" \
    --compiler-arg=-DOF_SOFTMAX_USE_FAST_MATH

reject_target_recipe \
  "$rewrite_dir/dav_c310_recipe_input.cu" \
  "$work_dir/dav_c310_recipe.bad_policy.cce" portable fast \
  dav-3510-rowwise-simd-v1 \
  "--target-recipe=dav-3510-rowwise-simd-v1 requires"
reject_target_recipe \
  "$rewrite_dir/dav_c310_recipe_input.cu" \
  "$work_dir/dav_c310_recipe.bad_math.cce" dav-c310-vec precise \
  dav-3510-rowwise-simd-v1 \
  "--target-recipe=dav-3510-rowwise-simd-v1 requires"
reject_target_recipe \
  "$rewrite_dir/dav_c310_recipe_input.cu" \
  "$work_dir/dav_c310_recipe.bad_name.cce" dav-c310-vec fast \
  unsupported \
  "unsupported --target-recipe value 'unsupported'"

run_check "canonical warp reduction translated goldens" \
  sh "$rewrite_dir/check_warp_reduce_rewrite.sh" \
    --fast "$warp_fast" --default "$warp_default"
run_check "canonical reducer translated goldens" \
  sh "$rewrite_dir/check_canonical_reducer_rewrite.sh" \
    --fast "$reducer_fast" --default "$reducer_default"
run_check "SIMT compatibility translated golden" \
  sh "$rewrite_dir/check_simt_compat.sh" \
    --translated "$simt_translated"
run_check "target ABI compatibility translated golden" \
  sh "$rewrite_dir/check_target_abi_compat.sh" \
    --translated "$target_abi_translated"
run_check "opt-in product frontend cooperative-groups translated golden" \
  sh "$rewrite_dir/check_cooperative_groups_compat.sh" \
    --translated "$cooperative_groups_translated"
run_check "dav-c310 recipe translated goldens" \
  sh "$rewrite_dir/check_dav_c310_recipe.sh" \
    --dav-fast "$recipe_fast" \
    --portable-precise "$recipe_portable_precise" \
    --portable-fast "$recipe_portable_fast" \
    --dav-precise "$recipe_dav_precise"
run_check "explicit row-wise SIMD recipe frontend outputs" \
  check_rowwise_recipe_frontend \
    "$oneflow_softmax_recipe" "$oneflow_rmsnorm_recipe"
run_check "injected row-wise SIMD header publication and repeat include" \
  check_injected_rowwise_header_publication \
    "$oneflow_softmax_recipe" "$rowwise_header_reinclude_probe"
run_check "Softmax colliding outer guard preserves translated body" \
  check_outer_guard_collision_output \
    "$softmax_guard_collision_output" \
    "$softmax_guard_collision_preprocessed" \
    "LaunchSoftmaxWarpImpl"
run_check "RMSNorm colliding outer guard preserves translated body" \
  check_outer_guard_collision_output \
    "$rmsnorm_guard_collision_output" \
    "$rmsnorm_guard_collision_preprocessed" \
    "LaunchRmsNormWarpImpl"

echo "[release-check] all host-only and translated-fixture checks passed"
