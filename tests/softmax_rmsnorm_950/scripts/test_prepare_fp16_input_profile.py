#!/usr/bin/env python3
"""Test profile identity and real C++ overload/deletion behavior without CANN."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("prepare_fp16_input_profile.py")
SPEC = importlib.util.spec_from_file_location("fp16_input_profile", SCRIPT)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class InputProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ascify-fp16-profile-")
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / "profile"
        cls.manifest = profile.prepare_profile(profile.DEFAULT_INPUT_ROOT, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_only_six_exact_double_specializations_change(self):
        self.assertEqual(sum(len(f["edits"]) for f in self.manifest["files"]), 6)
        for record in self.manifest["files"]:
            source = (profile.DEFAULT_INPUT_ROOT / record["path"]).read_bytes()
            prepared = (self.output / record["path"]).read_bytes()
            self.assertEqual(profile.sha256(source), record["source_sha256"])
            self.assertEqual(profile.sha256(prepared), record["profile_sha256"])
            restored = prepared.decode()
            for edit in record["edits"]:
                self.assertEqual(restored.count(edit["replacement"]), 1)
                restored = restored.replace(edit["replacement"], edit["original"], 1)
            self.assertEqual(restored.encode(), source)
        rms = "oneflow/core/cuda/rms_norm.cuh"
        self.assertEqual((self.output / rms).read_bytes(),
                         (profile.DEFAULT_INPUT_ROOT / rms).read_bytes())

    def test_deterministic_profile_and_manifest(self):
        other = self.root / "profile-again"
        profile.prepare_profile(profile.DEFAULT_INPUT_ROOT, other)
        for name in [*profile.FIXTURE_SHA256, "profile.json"]:
            self.assertEqual((self.output / name).read_bytes(), (other / name).read_bytes())
        self.assertEqual(json.loads((self.output / "profile.json").read_text()), self.manifest)

    def test_changed_source_rejected_before_publication(self):
        source_root = self.root / "changed-input"
        shutil.copytree(profile.DEFAULT_INPUT_ROOT, source_root)
        source = source_root / "oneflow/core/cuda/softmax.cuh"
        source.write_bytes(source.read_bytes() + b"\n// modified source\n")
        destination = self.root / "changed-output"
        with self.assertRaisesRegex(ValueError, "unrecognized fixture"):
            profile.prepare_profile(source_root, destination)
        self.assertFalse(destination.exists())

    def test_existing_directory_is_never_overwritten(self):
        manifest_before = (self.output / "profile.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            profile.prepare_profile(profile.DEFAULT_INPUT_ROOT, self.output)
        self.assertEqual((self.output / "profile.json").read_bytes(), manifest_before)

    def test_output_symlink_is_never_followed(self):
        link = self.root / "output-link"
        link.symlink_to(self.output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            profile.prepare_profile(profile.DEFAULT_INPUT_ROOT, link)


class DeletedSpecializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not cls.compiler or not shutil.which(cls.compiler[0]):
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.temporary = tempfile.TemporaryDirectory(prefix="ascify-fp16-deleted-")
        cls.root = Path(cls.temporary.name)
        output = cls.root / "profile"
        profile.prepare_profile(profile.DEFAULT_INPUT_ROOT, output)
        # Compile the actual prepared helper declarations/bodies. The rest of
        # the OneFlow header needs CUDA/CANN, so target integration is separate.
        softmax = (output / "oneflow/core/cuda/softmax.cuh").read_text()
        layer = (output / "oneflow/core/cuda/layer_norm.cuh").read_text()
        softmax = softmax[softmax.index("template<typename T>\n__inline__ __device__ T Inf();"):
                          softmax.index("inline cudaError_t GetNumBlocks(")]
        layer = layer[layer.index("template<typename T>\n__inline__ __device__ T Div("):
                      layer.index("template<class Func>\ninline cudaError_t GetNumBlocks(")]
        cls.helpers = (
            "#include <cmath>\n#include <type_traits>\n"
            "#define __device__\n#define CUDART_INF_F INFINITY\n"
            "float rsqrt(float x) { return 1.0f / std::sqrt(x); }\n"
            "namespace softmax {\n" + softmax + "}\n"
            "namespace layer_norm {\n" + layer + "}\n"
        )
        cls.compilation_number = 0

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def compile(self, body, *, link=False):
        type(self).compilation_number += 1
        source = self.root / f"test-{self.compilation_number}.cpp"
        source.write_text(self.helpers + body)
        output = source.with_suffix("")
        arguments = [*self.compiler, "-std=c++17", str(source)]
        arguments += ["-o", str(output)] if link else ["-fsyntax-only"]
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=30)
        return result, output

    def assert_deleted(self, body):
        result, _ = self.compile(body)
        self.assertNotEqual(result.returncode, 0, "FP64 helper unexpectedly accepted")
        self.assertIn("deleted", result.stderr.lower(), result.stderr)

    def test_float_helpers_compile_and_keep_results(self):
        result, executable = self.compile(r'''
static_assert(std::is_same<decltype(softmax::Div<float>(1.0f, 2.0f)), float>::value, "");
static_assert(std::is_same<decltype(layer_norm::Div<float>(1.0f, 2.0f)), float>::value, "");
int main() {
  return !(std::isinf(softmax::Inf<float>())
           && softmax::Div<float>(7.0f, 2.0f) == 3.5f
           && softmax::Exp<float>(0.0f) == 1.0f
           && softmax::Log<float>(1.0f) == 0.0f
           && layer_norm::Div<float>(9.0f, 3.0f) == 3.0f
           && layer_norm::Rsqrt<float>(4.0f) == 0.5f);
}
''', link=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        run = subprocess.run([str(executable)], capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 0)

    def test_every_explicit_double_call_is_rejected(self):
        calls = ["softmax::Inf<double>()", "softmax::Exp<double>(1.0)",
                 "softmax::Div<double>(1.0, 2.0)", "softmax::Log<double>(1.0)",
                 "layer_norm::Div<double>(1.0, 2.0)", "layer_norm::Rsqrt<double>(1.0)"]
        for call in calls:
            with self.subTest(call=call):
                self.assert_deleted(f"double actual_use() {{ return {call}; }}\n")

    def test_deduced_double_call_cannot_fall_back_to_float(self):
        self.assert_deleted("double actual_use() { return softmax::Div(1.0, 2.0); }\n")

    def test_dependent_double_instantiation_is_rejected(self):
        self.assert_deleted("""
template<typename T> T dispatch(T a, T b) { return layer_norm::Div<T>(a, b); }
double actual_use() { return dispatch<double>(1.0, 2.0); }
""")

    def test_double_function_address_is_rejected(self):
        self.assert_deleted("double (*actual_use)(double, double) = &softmax::Div<double>;\n")


if __name__ == "__main__":
    unittest.main()
