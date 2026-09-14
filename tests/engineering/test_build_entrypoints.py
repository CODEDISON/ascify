"""Exercise the public shell entrypoints without a CUDA or LLVM installation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class BuildEntrypoints(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ascify engineering ")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.env = os.environ.copy()
        for name in (
            "LLVM_BUILD_DIR", "LLVM_PROJECT_PATH", "BUILD_DIR", "INSTALL_ROOT",
            "ASCIFY_CC", "ASCIFY_CXX", "CC", "CXX", "ASCIFY_BUILD_JOBS",
            "ASCIFY_LINKER", "CMAKE_GENERATOR", "CMAKE_BUILD_TYPE",
            "ASCIFY_BINARY", "CUDA_PATH", "CLANG_RESOURCE_DIRECTORY",
        ):
            self.env.pop(name, None)
        self.records = self.work / "commands.jsonl"
        self.env["ENTRYPOINT_RECORDS"] = str(self.records)

    def recording_executable(self, name):
        path = self.work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['ENTRYPOINT_RECORDS'], 'a') as output:\n"
            "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        )
        path.chmod(0o755)
        return path

    def run_script(self, name, *arguments):
        return subprocess.run(
            ["bash", str(ROOT / name), *arguments], cwd=self.work,
            env=self.env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )

    def prepare_build(self):
        self.recording_executable("tools/cmake")
        self.env["PATH"] = str(self.work / "tools") + os.pathsep + self.env["PATH"]
        prefix = self.work / "llvm prefix"
        prefix.mkdir()
        self.env["LLVM_BUILD_DIR"] = str(prefix)
        return prefix

    def commands(self):
        return [json.loads(line) for line in self.records.read_text().splitlines()]

    def test_help_requires_no_dependencies_or_environment(self):
        for script in ("build.sh", "run.sh"):
            with self.subTest(script=script):
                result = self.run_script(script, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Usage:", result.stdout)
        self.assertFalse(self.records.exists())

    def test_standalone_llvm_prefix_and_literal_cmake_arguments(self):
        prefix = self.prepare_build()
        cc = self.recording_executable("custom tools/cc")
        cxx = self.recording_executable("custom tools/c++")
        self.env.update(ASCIFY_CC=str(cc), ASCIFY_CXX=str(cxx), ASCIFY_BUILD_JOBS="3")
        option = "-DASCIFY_TEST_LABEL=spaces;literal $(text)"
        result = self.run_script("build.sh", option)
        self.assertEqual(result.returncode, 0, result.stderr)
        configure, build = self.commands()
        self.assertIn(f"-DCMAKE_PREFIX_PATH={prefix}", configure)
        self.assertIn(f"-DCMAKE_C_COMPILER={cc}", configure)
        self.assertIn(f"-DCMAKE_CXX_COMPILER={cxx}", configure)
        self.assertEqual(configure[-1], option)
        self.assertEqual(build[-2:], ["--parallel", "3"])

    def test_environment_compiler_overrides_prefix_default(self):
        prefix = self.prepare_build()
        self.recording_executable("llvm prefix/bin/clang")
        self.recording_executable("llvm prefix/bin/clang++")
        cc = self.recording_executable("environment/cc")
        cxx = self.recording_executable("environment/c++")
        self.env.update(CC=str(cc), CXX=str(cxx))
        result = self.run_script("build.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        configure = self.commands()[0]
        self.assertIn(f"-DCMAKE_C_COMPILER={cc}", configure)
        self.assertIn(f"-DCMAKE_CXX_COMPILER={cxx}", configure)
        self.assertNotIn(f"-DCMAKE_C_COMPILER={prefix}/bin/clang", configure)

    def test_invalid_parallelism_stops_before_configuration(self):
        self.prepare_build()
        self.env["ASCIFY_BUILD_JOBS"] = "0"
        result = self.run_script("build.sh")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("positive integer", result.stderr)
        self.assertFalse(self.records.exists())

    def test_run_usage_precedes_environment_validation(self):
        result = self.run_script("run.sh")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)
        self.assertNotIn("Set CUDA_PATH", result.stderr)

    def test_installed_binary_accepts_automatic_resources_and_literal_input(self):
        binary = self.recording_executable("install tree/bin/ascify-clang")
        self.env.update(ASCIFY_BINARY=str(binary), CUDA_PATH="/cuda toolkit")
        result = self.run_script("run.sh", "input $(literal).cu", "--", "-DVALUE=a b")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), [[
            "input $(literal).cu", "--cuda-path=/cuda toolkit", "--", "-DVALUE=a b",
        ]])

    def test_run_forwards_explicit_resources(self):
        binary = self.recording_executable("install/bin/ascify-clang")
        self.env.update(
            ASCIFY_BINARY=str(binary), CUDA_PATH="/cuda",
            CLANG_RESOURCE_DIRECTORY="/clang resources",
        )
        result = self.run_script("run.sh", "input.cu")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--clang-resource-directory=/clang resources", self.commands()[0])

    def resource_configuration(self, explicit="", directory_headers=False):
        llvm_library = self.work / "llvm library"
        resource = llvm_library / "clang/23"
        headers = resource / "include"
        (headers / "cuda_wrappers").mkdir(parents=True)
        for name in ("__clang_cuda_runtime_wrapper.h", "cuda_wrappers/algorithm"):
            if directory_headers:
                (headers / name).mkdir()
            else:
                (headers / name).write_text("// resource fixture\n")
        script = self.work / "find_resources.cmake"
        script.write_text(
            'set(ASCIFY_INSTALL_CLANG_HEADERS ON)\n'
            'set(LIB_CLANG_RES 23)\n'
            'set(LLVM_VERSION_MAJOR 23)\n'
            f'set(LLVM_TOOLS_BINARY_DIR "{self.work}/no-tools")\n'
            f'set(LLVM_LIBRARY_DIRS "{self.work}/missing" "{llvm_library}")\n'
            f'set(ASCIFY_CLANG_RESOURCE_DIRECTORY "{explicit}" CACHE PATH "")\n'
            f'include("{ROOT}/cmake/AscifyClangResources.cmake")\n'
            'ascify_find_clang_resources()\n'
            f'file(WRITE "{self.work}/resolved.txt" "${{ASCIFY_CLANG_RESOURCE_DIRECTORY}}")\n'
        )
        result = subprocess.run(
            [shutil.which("cmake"), "-P", str(script)], cwd=self.work,
            env=self.env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        return result, resource

    def test_resource_search_handles_multiple_llvm_library_directories(self):
        result, resource = self.resource_configuration()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.work / "resolved.txt").read_text(), str(resource.resolve()))

    def test_invalid_explicit_resources_do_not_fall_back_to_other_headers(self):
        result, _ = self.resource_configuration(str(self.work / "incorrect resources"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Matching Clang resource headers were not found", result.stderr)
        self.assertFalse((self.work / "resolved.txt").exists())

    def test_directories_cannot_masquerade_as_resource_headers(self):
        result, _ = self.resource_configuration(directory_headers=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Matching Clang resource headers were not found", result.stderr)
        self.assertFalse((self.work / "resolved.txt").exists())


if __name__ == "__main__":
    unittest.main()
