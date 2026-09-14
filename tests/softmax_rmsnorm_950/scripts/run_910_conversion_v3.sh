#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TEST_ROOT}/../.." && pwd)"

WORK_ROOT="${WORK_ROOT:-${REPO_ROOT}/.work/softmax_rmsnorm_950}"
INPUT_ROOT="${INPUT_ROOT:-${REPO_ROOT}/tests/fixtures/oneflow}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${WORK_ROOT}/conversion/evidence_v3}"
ASCIFY_BINARY="${ASCIFY_BINARY:-${REPO_ROOT}/build/ascify-clang}"
CUDA_ROOT="${CUDA_ROOT:-${WORK_ROOT}/cuda}"
CLANG_RESOURCE_DIRECTORY="${CLANG_RESOURCE_DIRECTORY:-}"
RMSNORM_ADAPTER_INPUT="${RMSNORM_ADAPTER_INPUT:-${TEST_ROOT}/inputs/rmsnorm_affine_store.cuh}"
CONVERSION_SET_ID="${CONVERSION_SET_ID:-ascify_recipe_conversion_v3}"
VALIDATOR="${SCRIPT_DIR}/validate_formal_run.py"
LOCK_HELPER="${SCRIPT_DIR}/safe_lock_exec.py"

if [[ -z "${CLANG_RESOURCE_DIRECTORY}" ]]; then
  echo "Set CLANG_RESOURCE_DIRECTORY to the matching Clang resource directory (the parent of include/)." >&2
  exit 2
fi
if [[ ! "${CONVERSION_SET_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "CONVERSION_SET_ID must be a safe identifier" >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to package conversion evidence" >&2
  exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  echo "sha256sum is required to package conversion evidence" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to publish conversion evidence safely" >&2
  exit 1
fi

if [[ -e "${WORK_ROOT}" && ! -d "${WORK_ROOT}" ]]; then
  echo "Conversion work root is not a directory: ${WORK_ROOT}" >&2
  exit 1
fi
mkdir -p -- "${WORK_ROOT}"
WORK_ROOT_CANONICAL="$(cd -- "${WORK_ROOT}" && pwd -P)"
EVIDENCE_ROOT="$(
  python3 -B -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
    "${EVIDENCE_ROOT}"
)"
EVIDENCE_PARENT_RAW="$(dirname -- "${EVIDENCE_ROOT}")"
EVIDENCE_NAME="$(basename -- "${EVIDENCE_ROOT}")"
if [[ -z "${EVIDENCE_NAME}" || "${EVIDENCE_NAME}" == "."
      || "${EVIDENCE_NAME}" == ".." ]]; then
  echo "EVIDENCE_ROOT must name a child directory" >&2
  exit 2
fi
case "${EVIDENCE_ROOT}" in
  "${WORK_ROOT_CANONICAL}/"*) ;;
  *)
    echo "EVIDENCE_ROOT must remain under WORK_ROOT=${WORK_ROOT_CANONICAL}" >&2
    exit 2
    ;;
esac
mkdir -p -- "${EVIDENCE_PARENT_RAW}"
EVIDENCE_PARENT="$(cd -- "${EVIDENCE_PARENT_RAW}" && pwd -P)"
EVIDENCE_ROOT="${EVIDENCE_PARENT}/${EVIDENCE_NAME}"

PUBLISH_LOCK="${EVIDENCE_ROOT}.publish.lock"
if [[ ! -f "${LOCK_HELPER}" ]]; then
  echo "missing safe lock helper: ${LOCK_HELPER}" >&2
  exit 1
fi
if [[ "${ASCIFY_CONVERSION_LOCK_READY:-}" != "${PUBLISH_LOCK}" ]]; then
  if [[ -n "${ASCIFY_CONVERSION_LOCK_READY:-}" ]]; then
    echo "conversion lock re-entry path mismatch" >&2
    exit 1
  fi
  ASCIFY_CONVERSION_LOCK_READY="${PUBLISH_LOCK}" \
    exec python3 -B "${LOCK_HELPER}" \
      --fd 7 --path "${PUBLISH_LOCK}" --busy-exit 1 -- "$0" "$@"
fi
if [[ -L "${PUBLISH_LOCK}" ]] \
    || ! { true >&7; } 2>/dev/null || ! flock -n 7; then
  echo "conversion publish lock is unsafe or not held: ${PUBLISH_LOCK}" >&2
  exit 1
fi
if [[ -e "${EVIDENCE_ROOT}" || -L "${EVIDENCE_ROOT}" ]]; then
  echo "refusing to overwrite conversion evidence: ${EVIDENCE_ROOT}" >&2
  exit 1
fi
PARTIAL_ROOT="${EVIDENCE_ROOT}.partial.$$"
if [[ -e "${PARTIAL_ROOT}" || -L "${PARTIAL_ROOT}" ]]; then
  echo "partial conversion evidence path already exists: ${PARTIAL_ROOT}" >&2
  exit 1
fi

INPUT_SOFTMAX="${INPUT_ROOT}/oneflow/core/cuda/softmax.cuh"
INPUT_LAYER_NORM="${INPUT_ROOT}/oneflow/core/cuda/layer_norm.cuh"
INPUT_RMSNORM="${INPUT_ROOT}/oneflow/core/cuda/rms_norm.cuh"
INPUT_RMSNORM_ADAPTER="${RMSNORM_ADAPTER_INPUT}"
RECIPE_SOURCE="${REPO_ROOT}/src/DavC310TargetRecipe.cpp"
RECIPE_HEADER="${REPO_ROOT}/src/DavC310TargetRecipe.h"

for required_file in \
  "${ASCIFY_BINARY}" "${VALIDATOR}" "${LOCK_HELPER}" \
  "${INPUT_SOFTMAX}" "${INPUT_LAYER_NORM}" \
  "${INPUT_RMSNORM}" "${INPUT_RMSNORM_ADAPTER}" \
  "${RECIPE_SOURCE}" "${RECIPE_HEADER}" \
  "${CLANG_RESOURCE_DIRECTORY}/include/__clang_cuda_runtime_wrapper.h"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "missing conversion input: ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -x "${ASCIFY_BINARY}" ]]; then
  echo "Ascify binary is not executable: ${ASCIFY_BINARY}" >&2
  exit 1
fi
if [[ ! -d "${CUDA_ROOT}" ]]; then
  echo "missing user-owned CUDA compatibility tree: ${CUDA_ROOT}" >&2
  exit 1
fi

file_sha256() {
  local digest_line
  digest_line="$(sha256sum -- "$1")"
  printf '%s\n' "${digest_line%% *}"
}
require_digest() {
  local label="$1"
  local expected="$2"
  local path="$3"
  local actual
  actual="$(file_sha256 "${path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${label} changed during conversion evidence capture: ${path}" >&2
    return 1
  fi
}

ORIGIN_ASCIFY_SHA256="$(file_sha256 "${ASCIFY_BINARY}")"
ORIGIN_RECIPE_SOURCE_SHA256="$(file_sha256 "${RECIPE_SOURCE}")"
ORIGIN_RECIPE_HEADER_SHA256="$(file_sha256 "${RECIPE_HEADER}")"
ORIGIN_SOFTMAX_SHA256="$(file_sha256 "${INPUT_SOFTMAX}")"
ORIGIN_LAYER_NORM_SHA256="$(file_sha256 "${INPUT_LAYER_NORM}")"
ORIGIN_RMSNORM_SHA256="$(file_sha256 "${INPUT_RMSNORM}")"
ORIGIN_RMSNORM_ADAPTER_SHA256="$(file_sha256 "${INPUT_RMSNORM_ADAPTER}")"

mkdir -p \
  "${PARTIAL_ROOT}/converter" \
  "${PARTIAL_ROOT}/recipe" \
  "${PARTIAL_ROOT}/inputs/oneflow/core/cuda" \
  "${PARTIAL_ROOT}/outputs/oneflow/core/cuda" \
  "${PARTIAL_ROOT}/outputs/ascify950" \
  "${PARTIAL_ROOT}/logs"

cp -p -- "${ASCIFY_BINARY}" "${PARTIAL_ROOT}/converter/ascify-clang"
cp -p -- "${RECIPE_SOURCE}" "${PARTIAL_ROOT}/recipe/DavC310TargetRecipe.cpp"
cp -p -- "${RECIPE_HEADER}" "${PARTIAL_ROOT}/recipe/DavC310TargetRecipe.h"
cp -p -- "${INPUT_SOFTMAX}" \
  "${PARTIAL_ROOT}/inputs/oneflow/core/cuda/softmax.cuh"
cp -p -- "${INPUT_LAYER_NORM}" \
  "${PARTIAL_ROOT}/inputs/oneflow/core/cuda/layer_norm.cuh"
cp -p -- "${INPUT_RMSNORM}" \
  "${PARTIAL_ROOT}/inputs/oneflow/core/cuda/rms_norm.cuh"
cp -p -- "${INPUT_RMSNORM_ADAPTER}" \
  "${PARTIAL_ROOT}/inputs/rmsnorm_affine_store.cuh"

require_digest packaged_ascify_binary "${ORIGIN_ASCIFY_SHA256}" \
  "${PARTIAL_ROOT}/converter/ascify-clang"
require_digest packaged_recipe_source "${ORIGIN_RECIPE_SOURCE_SHA256}" \
  "${PARTIAL_ROOT}/recipe/DavC310TargetRecipe.cpp"
require_digest packaged_recipe_header "${ORIGIN_RECIPE_HEADER_SHA256}" \
  "${PARTIAL_ROOT}/recipe/DavC310TargetRecipe.h"
require_digest packaged_input_softmax "${ORIGIN_SOFTMAX_SHA256}" \
  "${PARTIAL_ROOT}/inputs/oneflow/core/cuda/softmax.cuh"
require_digest packaged_input_layer_norm "${ORIGIN_LAYER_NORM_SHA256}" \
  "${PARTIAL_ROOT}/inputs/oneflow/core/cuda/layer_norm.cuh"
require_digest packaged_input_rmsnorm "${ORIGIN_RMSNORM_SHA256}" \
  "${PARTIAL_ROOT}/inputs/oneflow/core/cuda/rms_norm.cuh"
require_digest packaged_input_rmsnorm_adapter \
  "${ORIGIN_RMSNORM_ADAPTER_SHA256}" \
  "${PARTIAL_ROOT}/inputs/rmsnorm_affine_store.cuh"

run_conversion() {
  local unit="$1"
  local input_relative="$2"
  local output_relative="$3"
  shift 3
  local stdout_path="${PARTIAL_ROOT}/logs/${unit}.stdout"
  local stderr_path="${PARTIAL_ROOT}/logs/${unit}.stderr"
  local exit_code
  echo "[conversion] ${unit}"
  set +e
  (
    cd -- "${PARTIAL_ROOT}"
    ./converter/ascify-clang \
      "${input_relative}" \
      --target-policy=dav-c310-vec \
      --simt-math=fast \
      "--cuda-path=${CUDA_ROOT}" \
      "--clang-resource-directory=${CLANG_RESOURCE_DIRECTORY}" \
      -o "${output_relative}" \
      -- -Iinputs -std=c++17 "$@"
  ) >"${stdout_path}" 2>"${stderr_path}"
  exit_code=$?
  set -e
  if [[ "${exit_code}" != "0" ]]; then
    echo "Ascify conversion failed for ${unit}; evidence retained at ${PARTIAL_ROOT}" >&2
    return "${exit_code}"
  fi
  if [[ ! -s "${PARTIAL_ROOT}/${output_relative}" ]]; then
    echo "Ascify produced an empty output for ${unit}: ${output_relative}" >&2
    return 1
  fi
}

run_conversion \
  softmax \
  "inputs/oneflow/core/cuda/softmax.cuh" \
  "outputs/oneflow/core/cuda/softmax.cuh"
run_conversion \
  layer_norm \
  "inputs/oneflow/core/cuda/layer_norm.cuh" \
  "outputs/oneflow/core/cuda/layer_norm.cuh" \
  -include cuda_fp16.h -include cuda_bf16.h
run_conversion \
  rmsnorm \
  "inputs/oneflow/core/cuda/rms_norm.cuh" \
  "outputs/oneflow/core/cuda/rms_norm.cuh" \
  -include cuda_fp16.h -include cuda_bf16.h
run_conversion \
  rmsnorm_adapter \
  "inputs/rmsnorm_affine_store.cuh" \
  "outputs/ascify950/rmsnorm_affine_store.cuh" \
  -include cuda_fp16.h -include cuda_bf16.h

require_digest live_ascify_binary "${ORIGIN_ASCIFY_SHA256}" "${ASCIFY_BINARY}"
require_digest live_recipe_source "${ORIGIN_RECIPE_SOURCE_SHA256}" "${RECIPE_SOURCE}"
require_digest live_recipe_header "${ORIGIN_RECIPE_HEADER_SHA256}" "${RECIPE_HEADER}"
require_digest live_input_softmax "${ORIGIN_SOFTMAX_SHA256}" "${INPUT_SOFTMAX}"
require_digest live_input_layer_norm "${ORIGIN_LAYER_NORM_SHA256}" "${INPUT_LAYER_NORM}"
require_digest live_input_rmsnorm "${ORIGIN_RMSNORM_SHA256}" "${INPUT_RMSNORM}"
require_digest live_input_rmsnorm_adapter \
  "${ORIGIN_RMSNORM_ADAPTER_SHA256}" "${INPUT_RMSNORM_ADAPTER}"

export PARTIAL_ROOT CUDA_ROOT CLANG_RESOURCE_DIRECTORY CONVERSION_SET_ID VALIDATOR
python3 -B - <<'PY'
import datetime
import hashlib
import importlib.util
import json
import os
import socket
from pathlib import Path

root = Path(os.environ["PARTIAL_ROOT"])
validator_spec = importlib.util.spec_from_file_location(
    "formal_validator", os.environ["VALIDATOR"]
)
if validator_spec is None or validator_spec.loader is None:
    raise RuntimeError("cannot load formal evidence validator")
validator = importlib.util.module_from_spec(validator_spec)
validator_spec.loader.exec_module(validator)


def digest(relative):
    value = hashlib.sha256()
    with (root / relative).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


artifact_specs = [
    ("ascify_binary", "converter", "converter/ascify-clang"),
    ("recipe_source", "source", "recipe/DavC310TargetRecipe.cpp"),
    ("recipe_header", "source", "recipe/DavC310TargetRecipe.h"),
    ("input_softmax", "input", "inputs/oneflow/core/cuda/softmax.cuh"),
    ("input_layer_norm", "input", "inputs/oneflow/core/cuda/layer_norm.cuh"),
    ("input_rmsnorm", "input", "inputs/oneflow/core/cuda/rms_norm.cuh"),
    ("input_rmsnorm_adapter", "input", "inputs/rmsnorm_affine_store.cuh"),
    (
        "generated_softmax_header",
        "output",
        "outputs/oneflow/core/cuda/softmax.cuh",
    ),
    (
        "generated_layer_norm_header",
        "output",
        "outputs/oneflow/core/cuda/layer_norm.cuh",
    ),
    (
        "generated_rmsnorm_header",
        "output",
        "outputs/oneflow/core/cuda/rms_norm.cuh",
    ),
    (
        "generated_rmsnorm_adapter",
        "output",
        "outputs/ascify950/rmsnorm_affine_store.cuh",
    ),
]
unit_specs = [
    ("softmax", "input_softmax", "generated_softmax_header"),
    ("layer_norm", "input_layer_norm", "generated_layer_norm_header"),
    ("rmsnorm", "input_rmsnorm", "generated_rmsnorm_header"),
    (
        "rmsnorm_adapter",
        "input_rmsnorm_adapter",
        "generated_rmsnorm_adapter",
    ),
]
forced_includes = {
    "softmax": [],
    "layer_norm": [
        "-include",
        "cuda_fp16.h",
        "-include",
        "cuda_bf16.h",
    ],
    "rmsnorm": [
        "-include",
        "cuda_fp16.h",
        "-include",
        "cuda_bf16.h",
    ],
    "rmsnorm_adapter": [
        "-include",
        "cuda_fp16.h",
        "-include",
        "cuda_bf16.h",
    ],
}
def recipe_summary(unit):
    stderr = (root / f"logs/{unit}.stderr").read_text(
        encoding="utf-8", errors="replace"
    )
    lines = [
        line for line in stderr.splitlines()
        if line.startswith("Ascify dav-c310 recipe:")
    ]
    if len(lines) != 1:
        raise RuntimeError(
            f"{unit} emitted {len(lines)} dav-c310 recipe summaries"
        )
    return lines[0]


for unit, _, _ in unit_specs:
    artifact_specs.append((f"log_{unit}_stdout", "log", f"logs/{unit}.stdout"))
    artifact_specs.append((f"log_{unit}_stderr", "log", f"logs/{unit}.stderr"))

artifacts = [
    {
        "logical_id": logical_id,
        "role": role,
        "path": relative,
        "sha256": digest(relative),
    }
    for logical_id, role, relative in artifact_specs
]
conversions = []
for unit, input_id, output_id in unit_specs:
    output_relative = next(
        relative
        for logical_id, _, relative in artifact_specs
        if logical_id == output_id
    )
    argv = [
        "./converter/ascify-clang",
        next(
            relative
            for logical_id, _, relative in artifact_specs
            if logical_id == input_id
        ),
        "--target-policy=dav-c310-vec",
        "--simt-math=fast",
        "--cuda-path=" + os.environ["CUDA_ROOT"],
        "--clang-resource-directory="
        + os.environ["CLANG_RESOURCE_DIRECTORY"],
        "-o",
        output_relative,
        "--",
        "-Iinputs",
        "-std=c++17",
    ]
    argv.extend(forced_includes[unit])
    conversions.append(
        {
            "logical_id": unit,
            "input_artifact": input_id,
            "output_artifact": output_id,
            "stdout_artifact": f"log_{unit}_stdout",
            "stderr_artifact": f"log_{unit}_stderr",
            "argv": argv,
            "exit_code": 0,
            "recipe_placement": validator.recipe_placement(
                root / output_relative, unit
            ),
            "recipe_topology": validator.recipe_topology(
                root / output_relative
            ),
            "recipe_summary": recipe_summary(unit),
        }
    )

document = {
    "schema_version": "ascify-conversion-evidence-v3",
    "conversion_set_id": os.environ["CONVERSION_SET_ID"],
    "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "hostname": socket.gethostname(),
    "policy": {
        "target_policy": "dav-c310-vec",
        "simt_math": "fast",
    },
    "converter": {
        "binary_artifact": "ascify_binary",
        "recipe_source_artifact": "recipe_source",
        "recipe_header_artifact": "recipe_header",
    },
    "environment": {
        "cuda_root_origin": os.environ["CUDA_ROOT"],
        "clang_resource_directory_origin": os.environ[
            "CLANG_RESOURCE_DIRECTORY"
        ],
        "input_include_path": "inputs",
    },
    "artifacts": artifacts,
    "conversions": conversions,
}
with (root / "manifest.json").open("x", encoding="utf-8") as stream:
    json.dump(document, stream, sort_keys=True, indent=2)
    stream.write("\n")
PY

ASCIFY_BINARY_SHA256="${ORIGIN_ASCIFY_SHA256}"
"${VALIDATOR}" conversion \
  --manifest "${PARTIAL_ROOT}/manifest.json" \
  --ascify-binary-sha256 "${ASCIFY_BINARY_SHA256}" \
  --recipe-source "${RECIPE_SOURCE}" \
  --recipe-header "${RECIPE_HEADER}" \
  --staged-softmax \
    "${PARTIAL_ROOT}/outputs/oneflow/core/cuda/softmax.cuh" \
  --staged-layer-norm \
    "${PARTIAL_ROOT}/outputs/oneflow/core/cuda/layer_norm.cuh" \
  --staged-rmsnorm \
    "${PARTIAL_ROOT}/outputs/oneflow/core/cuda/rms_norm.cuh" \
  --staged-rmsnorm-adapter \
    "${PARTIAL_ROOT}/outputs/ascify950/rmsnorm_affine_store.cuh"

require_digest publish_ascify_binary "${ORIGIN_ASCIFY_SHA256}" "${ASCIFY_BINARY}"
require_digest publish_recipe_source "${ORIGIN_RECIPE_SOURCE_SHA256}" "${RECIPE_SOURCE}"
require_digest publish_recipe_header "${ORIGIN_RECIPE_HEADER_SHA256}" "${RECIPE_HEADER}"
require_digest publish_input_softmax "${ORIGIN_SOFTMAX_SHA256}" "${INPUT_SOFTMAX}"
require_digest publish_input_layer_norm "${ORIGIN_LAYER_NORM_SHA256}" \
  "${INPUT_LAYER_NORM}"
require_digest publish_input_rmsnorm "${ORIGIN_RMSNORM_SHA256}" "${INPUT_RMSNORM}"
require_digest publish_input_rmsnorm_adapter \
  "${ORIGIN_RMSNORM_ADAPTER_SHA256}" "${INPUT_RMSNORM_ADAPTER}"
PARTIAL_MANIFEST_SHA256="$(file_sha256 "${PARTIAL_ROOT}/manifest.json")"
if [[ -e "${EVIDENCE_ROOT}" || -L "${EVIDENCE_ROOT}" ]]; then
  echo "conversion evidence appeared before publish; partial retained at ${PARTIAL_ROOT}" >&2
  exit 1
fi
if ! mv -T -n -- "${PARTIAL_ROOT}" "${EVIDENCE_ROOT}"; then
  echo "atomic conversion evidence publish failed; partial retained at ${PARTIAL_ROOT}" >&2
  exit 1
fi
if [[ -e "${PARTIAL_ROOT}" || ! -d "${EVIDENCE_ROOT}"
      || -L "${EVIDENCE_ROOT}" ]]; then
  echo "atomic conversion evidence publish identity check failed" >&2
  exit 1
fi
MANIFEST_SHA256="$(file_sha256 "${EVIDENCE_ROOT}/manifest.json")"
if [[ "${MANIFEST_SHA256}" != "${PARTIAL_MANIFEST_SHA256}" ]]; then
  echo "published conversion manifest differs from validated partial" >&2
  exit 1
fi
"${VALIDATOR}" conversion \
  --manifest "${EVIDENCE_ROOT}/manifest.json" \
  --ascify-binary-sha256 "${ASCIFY_BINARY_SHA256}" \
  --recipe-source "${RECIPE_SOURCE}" \
  --recipe-header "${RECIPE_HEADER}" \
  --staged-softmax \
    "${EVIDENCE_ROOT}/outputs/oneflow/core/cuda/softmax.cuh" \
  --staged-layer-norm \
    "${EVIDENCE_ROOT}/outputs/oneflow/core/cuda/layer_norm.cuh" \
  --staged-rmsnorm \
    "${EVIDENCE_ROOT}/outputs/oneflow/core/cuda/rms_norm.cuh" \
  --staged-rmsnorm-adapter \
    "${EVIDENCE_ROOT}/outputs/ascify950/rmsnorm_affine_store.cuh"
require_digest final_ascify_binary "${ORIGIN_ASCIFY_SHA256}" "${ASCIFY_BINARY}"
require_digest final_recipe_source "${ORIGIN_RECIPE_SOURCE_SHA256}" "${RECIPE_SOURCE}"
require_digest final_recipe_header "${ORIGIN_RECIPE_HEADER_SHA256}" "${RECIPE_HEADER}"
require_digest final_input_softmax "${ORIGIN_SOFTMAX_SHA256}" "${INPUT_SOFTMAX}"
require_digest final_input_layer_norm "${ORIGIN_LAYER_NORM_SHA256}" \
  "${INPUT_LAYER_NORM}"
require_digest final_input_rmsnorm "${ORIGIN_RMSNORM_SHA256}" "${INPUT_RMSNORM}"
require_digest final_input_rmsnorm_adapter \
  "${ORIGIN_RMSNORM_ADAPTER_SHA256}" "${INPUT_RMSNORM_ADAPTER}"
echo "[ok] conversion evidence=${EVIDENCE_ROOT}"
echo "[ok] ascify_binary_sha256=${ASCIFY_BINARY_SHA256}"
echo "[ok] conversion_manifest_sha256=${MANIFEST_SHA256}"
