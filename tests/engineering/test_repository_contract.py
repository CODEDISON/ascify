#!/usr/bin/env python3
"""Prevent legacy product claims and artifacts without rejecting legal notices."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("repository_check", REPO / "tools/check_repository.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class RepositoryContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in CHECKER.REQUIRED:
            self.write(name, "Project file\n")
        self.write("src/example.cpp", "// Copyright Advanced Micro Devices, Inc.\nint ownership;\n")

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_upstream_attribution_and_ownership_are_allowed(self):
        self.write("THIRD_PARTY_NOTICES.md", "Derived from HIPIFY; retain the AMD license.\n")
        self.assertTrue(CHECKER.check(self.root)["passed"])

    def test_reintroduced_legacy_option_fails(self):
        self.write("src/options.cpp", 'const char *option = "hip-kernel-execution-syntax";\n')
        result = CHECKER.check(self.root)
        self.assertFalse(result["passed"])
        self.assertEqual(result["errors"][0]["path"], "src/options.cpp")

    def test_unimplemented_documented_generator_fails(self):
        self.write("README.md", "Run `ascify-clang --perl`\n")
        self.assertFalse(CHECKER.check(self.root)["passed"])

    def test_cmake_modules_cannot_reintroduce_legacy_backends(self):
        self.write("cmake/backend.cmake", "set(ASCIFY_BACKEND ROCm)\n")
        self.assertFalse(CHECKER.check(self.root)["passed"])

    def test_artifact_is_rejected_but_ignored_build_is_allowed(self):
        self.write("build/compiler.o", "generated\n")
        self.assertTrue(CHECKER.check(self.root)["passed"])
        self.write("examples/vector_add.cu.dpp", "generated\n")
        self.assertFalse(CHECKER.check(self.root)["passed"])

    def test_missing_local_link_fails_without_fetching_web_links(self):
        self.write("README.md", "[Web](https://example.invalid/) [Guide](docs/missing.md)\n")
        result = CHECKER.check(self.root)
        self.assertFalse(result["passed"])
        self.assertEqual(result["local_links_checked"], 1)

    def test_external_local_link_and_source_symlink_fail(self):
        self.write("README.md", "[External](../private.md)\n")
        (self.root / "src/link.cpp").symlink_to(self.root / "src/example.cpp")
        self.assertEqual(len(CHECKER.check(self.root)["errors"]), 2)

    def test_links_inside_examples_are_not_claims(self):
        self.write("README.md", "```markdown\n[Example](missing.md)\n```\n[Guide](docs/README.md)\n")
        self.assertTrue(CHECKER.check(self.root)["passed"])

    def test_source_directory_symlink_is_rejected_without_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory)
            (external / "legacy.cpp").write_text("int hip_target;\n")
            (self.root / "src/imported").symlink_to(external, target_is_directory=True)
            result = CHECKER.check(self.root)
            self.assertFalse(result["passed"])
            self.assertEqual(len(result["errors"]), 1)
            self.assertEqual(result["errors"][0]["path"], "src/imported")


if __name__ == "__main__":
    unittest.main()
