#!/usr/bin/env python3
"""Prepare explicit FP16 workload inputs without defining device FP64 helpers.

This changes conversion inputs, never converter output.  It accepts only the
versioned OneFlow fixture bytes and replaces six double specializations with
deleted declarations, so an actual call remains a compile-time error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PROFILE_ID = "oneflow-fp16-no-device-fp64-v1"
UPSTREAM_COMMIT = "25c8978c1c8b1371ef6aa4187dae4495bd233c35"
DEFAULT_INPUT_ROOT = Path(__file__).resolve().parents[3] / "tests/fixtures/oneflow"
FIXTURE_SHA256 = {
    "oneflow/core/cuda/softmax.cuh":
        "95ad8fed7bdcdf9d8a755c30116c8909e2c655b3f08f19b5965f7289c1beb8cf",
    "oneflow/core/cuda/layer_norm.cuh":
        "41f7d02e188dcc3908b142e33fc3120532f06dabc49923a4f9e2fce92990cd34",
    "oneflow/core/cuda/rms_norm.cuh":
        "9895efb57f9195cfafab8e812f54c2f651ccd704a199d5e1e107b0465d0b629a",
}
# These are exact source bodies, not a general-purpose C++ text rewrite.
# Full-file hashes above prevent a changed source from inheriting the profile.
SPECIALIZATIONS = {
    "oneflow/core/cuda/softmax.cuh": (
        ("Inf", "", "CUDART_INF"),
        ("Exp", "double x", "exp(x)"),
        ("Div", "double a, double b", "a / b"),
        ("Log", "double x", "log(x)"),
    ),
    "oneflow/core/cuda/layer_norm.cuh": (
        ("Div", "double a, double b", "a / b"),
        ("Rsqrt", "double x", "rsqrt(x)"),
    ),
    "oneflow/core/cuda/rms_norm.cuh": (),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def specialization_edit(name: str, parameters: str, expression: str) -> tuple[str, str]:
    declaration = (
        "template<>\n"
        f"__inline__ __device__ double {name}<double>({parameters})"
    )
    original = declaration + " {\n  return " + expression + ";\n}\n"
    replacement = (
        declaration + " = delete;"
        "  // Ascify FP16 input profile excludes device FP64.\n"
    )
    return original, replacement


def prepare_profile(input_root: Path, output_root: Path) -> dict:
    """Validate all sources before creating a new, non-overwriting output root."""
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"refusing to overwrite profile output: {output_root}")

    outputs: dict[str, bytes] = {}
    records = []
    for relative_path, expected in FIXTURE_SHA256.items():
        source = input_root / relative_path
        data = source.read_bytes()
        actual = sha256(data)
        if actual != expected:
            raise ValueError(
                f"unrecognized fixture {relative_path}: expected {expected}, got {actual}"
            )
        original_text = data.decode("utf-8")
        text = original_text
        edits = []
        for name, parameters, expression in SPECIALIZATIONS[relative_path]:
            original, replacement = specialization_edit(name, parameters, expression)
            if text.count(original) != 1:
                raise ValueError(f"expected one exact {name}<double> body in {relative_path}")
            edits.append({
                "symbol": f"{name}<double>",
                "source_line": original_text[:original_text.index(original)].count("\n") + 1,
                "operation": "replace_definition_with_deleted_declaration",
                "original": original,
                "replacement": replacement,
            })
            text = text.replace(original, replacement, 1)
        prepared = text.encode("utf-8")
        outputs[relative_path] = prepared
        records.append({
            "path": relative_path,
            "source_sha256": actual,
            "profile_sha256": sha256(prepared),
            "source_bytes": len(data),
            "profile_bytes": len(prepared),
            "edits": edits,
        })

    manifest = {
        "schema": "ascify.input-profile.v1",
        "profile": PROFILE_ID,
        "intended_workload": "FP16 input/output, FP32 accumulation",
        "source_upstream_commit": UPSTREAM_COMMIT,
        "rms_norm_fixture_note": "Preserves the existing rows_per_access tail fix byte-for-byte.",
        "boundary": "Six device double helper specializations are deleted; real calls are rejected.",
        "not_a_claim": "No automatic FP64 migration, target compilation, correctness, or performance claim.",
        "files": records,
    }
    # mkdir is exclusive: an existing directory, file, or symlink is never reused.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    try:
        for relative_path, data in outputs.items():
            destination = output_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        (output_root / "profile.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(output_root)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = prepare_profile(args.input_root, args.output_root)
    except (OSError, ValueError) as error:
        parser.exit(1, f"FP16 input profile: {error}\n")
    print(f"Prepared {manifest['profile']}: {args.output_root}")
    print(f"Manifest: {args.output_root / 'profile.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
