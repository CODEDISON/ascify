"""Exercise the published command-line contract using a built translator.

Set ASCIFY_BINARY to the ascify-clang executable, or pass --binary when running
this file directly. Host-only discovery explicitly skips these tests when a
translator build is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import unittest


class CliSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not os.environ.get("ASCIFY_BINARY"):
            raise unittest.SkipTest("ASCIFY_BINARY is not set")
        cls.binary = Path(os.environ["ASCIFY_BINARY"]).resolve(strict=True)

    def invoke(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.binary), *arguments],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )

    def test_help_describes_cuda_to_ascend(self) -> None:
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("CUDA to Ascend source translator options", result.stdout)
        for option in (
            "--target-policy",
            "--target-recipe",
            "--migration-receipt",
            "--print-stats-csv",
        ):
            self.assertIn(option, result.stdout)
        self.assertNotRegex(result.stdout, re.compile(r"hipify|hipification|rocm|\bhip\b", re.I))

    def test_versions_report_build_information(self) -> None:
        result = self.invoke("--versions")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("CUDA to Ascend source translator", result.stdout)
        self.assertRegex(result.stdout, r"LLVM build version: \d+")
        self.assertRegex(result.stdout, r"Clang resource version: \d+")
        self.assertIn("docs/validation-matrix.md", result.stdout)
        self.assertNotIn("Supports", result.stdout)
        self.assertNotIn("cuDNN", result.stdout)
        self.assertNotIn("DPP", result.stdout)

    def test_standard_version_alias_matches_versions(self) -> None:
        canonical = self.invoke("--versions")
        alias = self.invoke("--version")
        self.assertEqual(canonical.returncode, 0, canonical.stdout)
        self.assertEqual(alias.returncode, 0, alias.stdout)
        self.assertEqual(alias.stdout, canonical.stdout)

    def test_help_prefixes_do_not_suppress_unknown_option_errors(self) -> None:
        for option in ("-h", "--help", "--help-hidden"):
            with self.subTest(option=option):
                result = self.invoke(option)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("CUDA to Ascend source translator options", result.stdout)
        for option in ("--helpful", "--help-fake-option", "--hello=0", "-helpful"):
            with self.subTest(option=option):
                result = self.invoke(option)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("Unknown command line argument", result.stdout)

    def test_removed_options_fail_instead_of_silently_succeeding(self) -> None:
        removed = (
            "--hip-kernel-execution-syntax",
            "--cuda-kernel-execution-syntax",
            "--perl",
            "--python",
            "--md",
            "--csv",
            "--doc-format=full",
            "--o-ascify-perl-dir=unused",
            "--o-python-map-dir=unused",
            "--experimental",
            "--no-undocumented-features",
            "--no-warnings-on-undocumented-features",
        )
        for option in removed:
            with self.subTest(option=option):
                result = self.invoke(option)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("Unknown command line argument", result.stdout)

    def test_missing_source_fails(self) -> None:
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("Must specify at least 1 positional argument", result.stdout)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--binary", help="path to a built ascify-clang executable")
    options, remaining = parser.parse_known_args()
    if options.binary:
        os.environ["ASCIFY_BINARY"] = options.binary
    unittest.main(argv=[sys.argv[0], *remaining])
