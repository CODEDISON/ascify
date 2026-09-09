#!/usr/bin/env python3
"""Export every original RMSNorm shape and A800 measurement without domain filtering."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

FIELDS = ("case_id", "idx", "op", "dtype", "tier", "run_check", "run_bench", "rows", "cols",
          "scenario", "input_pattern", "cuda_pred_path", "predicted_path", "affine", "eps", "notes")


def read_csv(path):
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError(f"missing or duplicate header: {path}")
    return raw, list(reader)


def prepare(shapes_path, baseline_path, out):
    shape_bytes, shapes = read_csv(shapes_path)
    baseline_bytes, baselines = read_csv(baseline_path)
    if len(shapes) != 1000 or [int(s["idx"]) for s in shapes] != list(range(1000)):
        raise ValueError("shapes must preserve exactly original idx=0..999 in order")
    by_key = {(int(r["idx"]), r["variant"]): r for r in baselines}
    expected = {(i, v) for i in range(1000) for v in ("plain", "affine")}
    if len(baselines) != 2000 or len(by_key) != 2000 or set(by_key) != expected:
        raise ValueError("A800 must contain exactly 1000 plain and 1000 affine measurements")
    output_rows, previously_filtered = [], []
    max_elements = 0
    for s in shapes:
        idx, rows, cols = int(s["idx"]), int(s["rows"]), int(s["ncol"])
        if not (0 < rows <= 2**31 - 1 and 0 < cols <= 2**31 - 1):
            raise ValueError(f"idx={idx}: dimensions exceed original kernel parameter range")
        elements = rows * cols
        max_elements = max(max_elements, elements)
        if elements >= 2**31 - 1:
            previously_filtered.append(idx)
        for variant in ("plain", "affine"):
            r = by_key[idx, variant]
            if (int(r["rows"]), int(r["cols"]), r["scenario"], r["pred_path"]) != (
                    rows, cols, s["scenario"], s["predicted_path"]):
                raise ValueError(f"idx={idx}, {variant}: shape identity differs from A800")
            if (r["op"], r["dtype"], r["direction"], int(r["iters"])) != (
                    "rms_norm", "fp16", "fwd", 100):
                raise ValueError(f"idx={idx}, {variant}: wrong A800 protocol")
            wanted_bytes = 4 * elements + 4 * rows + (2 * cols if variant == "affine" else 0)
            if int(r["bytes"]) != wanted_bytes:
                raise ValueError(f"idx={idx}, {variant}: invalid logical bytes")
            for field in ("lat_ms_mean", "lat_ms_median", "lat_ms_min", "lat_ms_p90", "gbps"):
                value = float(r[field])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"idx={idx}, {variant}: invalid {field}")
            output_rows.append(dict(zip(FIELDS, (
                f"rn_full_{idx:04d}{variant[0]}", str(idx), "rms_norm", "fp16", "full", "1", "1",
                s["rows"], s["ncol"], s["scenario"], "random", s["predicted_path"],
                s["predicted_path"], "1" if variant == "affine" else "0", "1e-5",
                "Original full1000 input; 64-bit linear addressing"))))
    # Validate the entire input before creating any output; never overwrite an old run.
    out.mkdir(parents=True, exist_ok=False)
    (out / "original_rmsnorm_shapes.csv").write_bytes(shape_bytes)
    (out / "a800_rmsnorm_full1000.csv").write_bytes(baseline_bytes)
    for name, selected in (
        ("shapes_rmsnorm_full1000_pairs2000.csv", output_rows),
        ("shapes_rmsnorm_previous_exclusions.csv",
         [r for r in output_rows if int(r["idx"]) in previously_filtered]),
    ):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(selected)
    manifest = {
        "schema": "ascify.rmsnorm.full1000-inputs.v1", "shape_count": 1000,
        "performance_rows": 2000, "correctness_rows": 2000,
        "variants": {"plain": 1000, "affine": 1000},
        "previously_filtered_indices": previously_filtered,
        "previously_filtered_rows": 2 * len(previously_filtered),
        "max_elements": max_elements,
        "check_arguments": ["--tier", "full", "--max-elements", str(max_elements),
                            "--host-chunk-elements", "1048576", "--abs-tol", "5e-3",
                            "--scaled-rel-tol", "2e-3", "--inv-abs-tol", "2e-3",
                            "--inv-scaled-rel-tol", "2e-3"],
        "device_validation": "not performed by this exporter",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--a800-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.shapes, args.a800_results, args.out_dir)
    print(json.dumps({k: manifest[k] for k in ("shape_count", "performance_rows", "previously_filtered_rows")}))


if __name__ == "__main__":
    main()
