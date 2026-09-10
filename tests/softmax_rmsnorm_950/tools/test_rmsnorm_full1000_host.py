#!/usr/bin/env python3
"""Exercise the actual adapter and checker without claiming device validation."""
import csv
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

SUITE = Path(__file__).resolve().parents[1]
STUB = Path(__file__).with_name("host_stubs") / "rmsnorm_checker_runtime.hpp"


class RmsNormFull1000HostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ascify-rms64-host-")
        cls.root = Path(cls.temporary.name)
        for relative in ("oneflow/core/cuda/layer_norm.cuh", "oneflow/core/cuda/rms_norm.cuh",
                         "common/runner_common.hpp"):
            p = cls.root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('#include "rmsnorm_checker_runtime.hpp"\n')
        (cls.root / "rmsnorm_checker_runtime.hpp").write_bytes(STUB.read_bytes())
        adapter = cls.root / "ascify950/rmsnorm_affine_store.cuh"
        adapter.parent.mkdir()
        adapter.write_bytes((SUITE / "inputs/rmsnorm_affine_store.cuh").read_bytes())
        source = cls.root / "checker.cpp"
        source.write_bytes((SUITE / "rmsnorm_check.cce").read_bytes())
        cls.compile(source, cls.root / "checker")
        cls.shapes = cls.root / "shapes.csv"
        cls.shapes.write_text("21,7,5,0\n24,7,5,1\n250,9,8,0\n918,9,8,1\n")
        address_source = cls.root / "address.cpp"
        address_source.write_text(r'''
#include "ascify950/rmsnorm_affine_store.cuh"
#include <cassert>
#include <sys/mman.h>
#include <iostream>

template<int N, bool Affine>
void Check(uint16_t* output, int64_t row, int64_t column, int64_t stride) {
  const int64_t element = row * stride + column;
  const float source[N] = {7};
  std::vector<uint16_t> weights(static_cast<size_t>(stride), 2);
  output[element - 1] = 0x3210;
  for (int i = 0; i < N; ++i) output[element + i] = 0xffff;
  output[element + N] = 0x4567;
  ascify950_recipe_fixture::RmsNormAffineStore<float, uint16_t, Affine> store(
      output, Affine ? weights.data() : nullptr, stride);
  store.template store<N>(source, row, column);
  assert(output[element - 1] == 0x3210);
  assert(output[element + N] == 0x4567);
  for (int i = 0; i < N; ++i) {
    assert(output[element + i] == static_cast<uint16_t>(source[i]) * (Affine ? 2 : 1));
  }
}
int main() {
  static_assert(sizeof(size_t) == 8, "host address regression requires 64-bit size_t");
  const size_t bytes = 8000000010ULL * sizeof(uint16_t);
  void* allocation = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(allocation != MAP_FAILED);
  auto* output = static_cast<uint16_t*>(allocation);
  const int64_t coordinates[][3] = {
      {262143, 8190, 8192},  // 2^31 - 2, including the final INT32_MAX element for pack2
      {262144, 0, 8192},     // 2^31
      {262144, 2, 8192},     // 2^31 + 2
      {976561, 8190, 8192}  // final elements of original idx232 (~8e9 elements)
  };
  for (const auto& p : coordinates) {
    Check<1, false>(output, p[0], p[1], p[2]);
    Check<1, true>(output, p[0], p[1], p[2]);
    Check<2, false>(output, p[0], p[1], p[2]);
    Check<2, true>(output, p[0], p[1], p[2]);
  }
  assert(munmap(allocation, bytes) == 0);
  std::cout << "PASS: 16 actual adapter stores at int32 boundaries and ~8e9 offsets\n";
}
''')
        cls.compile(address_source, cls.root / "address")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def compile(cls, source, output):
        result = subprocess.run(shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++17", "-pthread", "-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all",
            "-I", str(cls.root), str(source), "-o", str(output)],
            text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def run_checker(self, chunk, corruption=None, extra=()):
        output = self.root / "accuracy.csv"
        env = os.environ.copy()
        env.pop("CORRUPTION", None)
        env.pop("FAIL_READBACK", None)
        if corruption == "readback":
            env["FAIL_READBACK"] = "1"
        elif corruption:
            env["CORRUPTION"] = corruption
        result = subprocess.run([str(self.root / "checker"), "--shapes", str(self.shapes),
                                 "--out", str(output), "--host-chunk-elements", str(chunk),
                                 *extra], text=True, capture_output=True, env=env)
        if output.exists():
            with output.open() as stream:
                return result, list(csv.reader(stream))
        return result, []

    def test_actual_adapter_pack1_pack2_plain_affine_above_int32(self):
        result = subprocess.run([str(self.root / "address")], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("16 actual adapter stores", result.stdout)

    def test_chunked_oracle_matches_one_chunk_for_all_elements(self):
        expected, records = self.run_checker(1000)
        self.assertEqual(expected.returncode, 0, expected.stderr)
        self.assertEqual(len(records), 4)
        self.assertTrue(all(r[2] == "pass" for r in records))
        for chunk in (1, 5, 10, 17, 32, 2**64 - 1):
            with self.subTest(chunk=chunk):
                result, actual = self.run_checker(chunk)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(actual, records)
                for idx, elements, rows in ((21, 35, 7), (24, 35, 7), (250, 72, 9), (918, 72, 9)):
                    self.assertIn(f"idx={idx} elements={elements} rows={rows}", result.stderr)

    def test_final_chunk_corruption_is_never_omitted(self):
        for corruption in ("output", "input", "inverse", "nan", "weight", "guard", "unwritten"):
            with self.subTest(corruption=corruption):
                reference, expected = self.run_checker(1000, corruption)
                result, actual = self.run_checker(10, corruption)
                self.assertEqual(reference.returncode, 2 if corruption == "weight" else 3, reference.stderr)
                self.assertEqual(result.returncode, reference.returncode, result.stderr)
                self.assertEqual(actual, expected)
                self.assertTrue(any(r[2] == "fail" for r in actual))

    def test_partial_readback_cannot_report_pass(self):
        result, actual = self.run_checker(10, "readback")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(actual[0][2], "error")
        self.assertIn("chunk payload copy", actual[0][-1])

    def test_parallel_oracle_preserves_records_and_tail_failures(self):
        for corruption in (None, "output", "input", "inverse", "nan", "weight", "guard", "unwritten", "readback"):
            reference, expected = self.run_checker(17, corruption)
            for workers in (2, 3, 8, 64):
                with self.subTest(corruption=corruption, workers=workers):
                    result, actual = self.run_checker(17, corruption, ("--host-workers", str(workers)))
                    self.assertEqual(result.returncode, reference.returncode, result.stderr)
                    self.assertEqual(actual, expected)
                    self.assertIn(f"host oracle workers: {workers}", result.stderr)

    def test_invalid_workers_rejected_before_device_initialization(self):
        for workers in ("0", "65", "-1", "oops", "2x", "99999999999999999999999"):
            with self.subTest(workers=workers):
                result, _ = self.run_checker(17, extra=("--host-workers", workers))
                self.assertEqual(result.returncode, 1)
                self.assertIn("--host-workers must be", result.stderr)

    def test_zero_chunk_rejected_before_device_initialization(self):
        result, _ = self.run_checker(0)
        self.assertEqual(result.returncode, 1)
        self.assertIn("must be positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
