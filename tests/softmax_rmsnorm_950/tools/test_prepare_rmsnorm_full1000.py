#!/usr/bin/env python3
"""Full-corpus export must preserve original records and reject partial evidence."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("exporter", Path(__file__).with_name("prepare_rmsnorm_full1000.py"))
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ascify-full1000-export-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shapes, self.baselines = [], []
        for idx in range(1000):
            rows, cols = (262144 + (idx >= 6), 8192) if idx < 33 else (7, 5)
            self.shapes.append(dict(idx=idx, rows=rows, ncol=cols, scenario="fixture", predicted_path="block_smem"))
            for variant in ("plain", "affine"):
                self.baselines.append(dict(
                    op="rms_norm", dtype="fp16", direction="fwd", variant=variant, idx=idx,
                    rows=rows, cols=cols, scenario="fixture", pred_path="block_smem", iters=100,
                    lat_ms_mean=1, lat_ms_median=1, lat_ms_min=1, lat_ms_p90=1,
                    bytes=rows * cols * 4 + rows * 4 + (cols * 2 if variant == "affine" else 0), gbps=1,
                ))
        self.write()

    def write(self):
        for name, records in (("shapes.csv", self.shapes), ("baseline.csv", self.baselines)):
            with (self.root / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)

    def export(self):
        return exporter.prepare(self.root / "shapes.csv", self.root / "baseline.csv", self.root / "out")

    def test_33_pairs_and_duplicate_geometry_preserved(self):
        manifest = self.export()
        self.assertEqual(manifest["variants"], {"plain": 1000, "affine": 1000})
        self.assertEqual(manifest["previously_filtered_indices"], list(range(33)))
        self.assertEqual(manifest["previously_filtered_rows"], 66)
        self.assertEqual((self.root / "out/a800_rmsnorm_full1000.csv").read_bytes(), (self.root / "baseline.csv").read_bytes())
        self.assertEqual((self.root / "out/original_rmsnorm_shapes.csv").read_bytes(), (self.root / "shapes.csv").read_bytes())
        with (self.root / "out/shapes_rmsnorm_full1000_pairs2000.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2000)
        self.assertEqual(len({(r["idx"], r["affine"]) for r in rows}), 2000)
        self.assertTrue(all(r["run_check"] == "1" and r["run_bench"] == "1" for r in rows))
        with (self.root / "out/shapes_rmsnorm_previous_exclusions.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 66)

    def test_existing_evidence_never_overwritten(self):
        self.export()
        before = (self.root / "out/manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual((self.root / "out/manifest.json").read_bytes(), before)

    def test_missing_original_pair_is_rejected_before_writing(self):
        self.baselines.pop(0)
        self.write()
        with self.assertRaises(ValueError):
            self.export()
        self.assertFalse((self.root / "out").exists())

    def test_invalid_measurement_identity_and_size_rejected(self):
        for key, bad in (("rows", 1), ("dtype", "bf16"), ("bytes", 1), ("iters", 1),
                         ("lat_ms_median", "nan"), ("gbps", 0)):
            with self.subTest(key=key):
                original = self.baselines[0][key]
                self.baselines[0][key] = bad
                self.write()
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse((self.root / "out").exists())
                self.baselines[0][key] = original


if __name__ == "__main__":
    unittest.main()
