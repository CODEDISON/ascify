#!/usr/bin/env python3
"""Exercise reviewed CUDA Samples profiles and atomic refusal through Ascify."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / "tests/rewrite/fixtures/nvidia_samples/Common"
V13_4 = ROOT / "tests/rewrite/fixtures/nvidia_samples_v13_4/Common/helper_cuda.h"
BODY = '''
int selectDevice(int argc, const char** argv) {
  checkCudaErrors(cudaSetDevice(0));
  getLastCudaError("reviewed helper profile");
  return findCudaDevice(argc, argv);
}
'''
INCLUDE_ORDERS = {
    "direct": "#include <helper_cuda.h>\n",
    "functions_first": "#include <helper_functions.h>\n#include <helper_cuda.h>\n",
    "cuda_first": "#include <helper_cuda.h>\n#include <helper_functions.h>\n",
}


def identity(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def diagnostics(result):
    # Include callbacks can emit thousands of lines. Keep the actual frontend
    # errors and distinct closure reasons useful when a configured SDK fails.
    relevant = [line for line in result.stderr.splitlines()
                if "error:" in line or "closure:" in line]
    return "\n".join(list(dict.fromkeys(relevant))[:24] or result.stderr.splitlines()[-24:])


def main():
    if sys.flags.optimize:
        raise RuntimeError("helper profile validation requires Python assertions; disable optimization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cuda-path", required=True)
    parser.add_argument("--resource-dir", required=True)
    args = parser.parse_args()
    fixtures = [*HELPERS.iterdir(), V13_4]
    original_identity = identity(fixtures)
    assert len(V13_4.read_bytes()) == 28219
    assert original_identity[str(V13_4)] == (
        "0494246768a4e3dbfcd688d79f4abad1fa39fc9f2d9dc3264a894f1c96d3785d")

    with tempfile.TemporaryDirectory(prefix="ascify-helper-profiles-") as temp:
        work = Path(temp)

        def helpers_for(label, header):
            directory = work / label / "Common"
            shutil.copytree(HELPERS, directory)
            shutil.copyfile(header, directory / "helper_cuda.h")
            return directory

        def convert(label, helpers, source, accepted, reason="", publication_refused=False):
            source_path = work / (label + ".cu")
            output = work / (label + ".cpp")
            source_path.write_text(source)
            inputs = [source_path, *helpers.iterdir()]
            before = identity(inputs)
            result = subprocess.run([
                args.binary, str(source_path), "--target-policy=dav-c310-vec",
                "--simt-math=fast", "--default-preprocessor",
                "--cuda-path=" + args.cuda_path,
                "--clang-resource-directory=" + args.resource_dir,
                "-o", str(output), "--", "-x", "cuda", "-std=c++17",
                "-I" + str(helpers),
            ], text=True, capture_output=True, timeout=60)
            assert identity(inputs) == before, label + ": input bytes changed"
            detail = label + ": " + diagnostics(result)
            if publication_refused:
                assert result.returncode != 0, detail
                assert not output.exists(), label + ": rejected conversion published output"
                assert reason in result.stderr, detail
                print(label + ": compatibility output publication refused")
                return
            assert result.returncode == 0, detail
            text = output.read_text()
            if accepted:
                assert "#include <helper_cuda.h>" not in text, detail
                assert "#include <helper_string.h>" in text, text
                assert "#include <ascify/ascify_cuda_compat.hpp>" in text, text
                assert "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaSetDevice(0))" in text, text
                assert 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR("reviewed helper profile")' in text, text
                assert "return ::ascify::sampleFindCudaDevice(argc, argv);" in text, text
                assert "include removed; portable surface retained" in result.stderr, detail
                for count in ("check_rewrites=1", "get_last_rewrites=1", "find_device_rewrites=1"):
                    assert count in result.stderr, detail
                if "#include <helper_functions.h>" in source:
                    assert "#include <helper_functions.h>" in text, text
            else:
                assert "#include <helper_cuda.h>" in text, text
                assert "checkCudaErrors(ascify::cudaSetDevice(0))" in text, text
                assert 'getLastCudaError("reviewed helper profile")' in text, text
                assert "return findCudaDevice(argc, argv);" in text, text
                for replacement in ("ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS",
                                    "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR",
                                    "::ascify::sampleFindCudaDevice"):
                    assert replacement not in text, label + ": partial helper transaction\n" + text
                assert "include kept" in result.stderr, detail
                assert reason in result.stderr, detail
                for count in ("check_rewrites=0", "get_last_rewrites=0", "find_device_rewrites=0"):
                    assert count in result.stderr, detail
            print(label + (": helper transaction committed" if accepted else ": complete helper transaction retained"))

        for profile, header in (("b7c5481", HELPERS / "helper_cuda.h"), ("v13_4", V13_4)):
            helpers = helpers_for(profile, header)
            for order, includes in INCLUDE_ORDERS.items():
                convert(profile + "_" + order, helpers, includes + BODY, True)

        mutated = helpers_for("v13_4_changed_bytes", V13_4)
        header = mutated / "helper_cuda.h"
        header.write_bytes(header.read_bytes() + b"\n")
        convert("v13_4_changed_bytes", mutated, INCLUDE_ORDERS["direct"] + BODY,
                False, "residual helper declaration 'findCudaDevice'")

        redirected = helpers_for("v13_4_macro_redirect", V13_4)
        source = ("#define MAX(a, b) ((a) + (b))\n" +
                  INCLUDE_ORDERS["direct"] + BODY)
        convert("v13_4_macro_redirect", redirected, source, False,
                "changed a frozen helper profile")

        # exit belongs to the frozen compatibility identifier surface. Its
        # redirection rejects the whole output, even after a later #undef.
        exit_redirected = helpers_for("v13_4_reserved_exit_redirect", V13_4)
        source = ("#include <stdlib.h>\n#define exit ::exit\n" +
                  INCLUDE_ORDERS["direct"] + "#undef exit\n" + BODY)
        convert("v13_4_reserved_exit_redirect", exit_redirected, source, False,
                "macro 'exit' collides with the frozen compat", publication_refused=True)

        changed_dependency = helpers_for("v13_4_changed_dependency", V13_4)
        helper_string = changed_dependency / "helper_string.h"
        original = helper_string.read_text()
        signature = ("inline bool checkCmdLineFlag(const int argc, const char **argv,\n"
                     "                             const char *string_ref) {\n")
        assert original.count(signature) == 1
        helper_string.write_text(original.replace(
            signature, signature + '  fprintf(stderr, "unreviewed helper_string dependency\\n");\n'))
        convert("v13_4_changed_dependency", changed_dependency,
                INCLUDE_ORDERS["direct"] + BODY, False,
                "residual helper declaration 'findCudaDevice'")

    assert identity(fixtures) == original_identity, "vendored fixture bytes changed"
    print("reviewed helper profiles: six conversions, three atomic refusal boundaries and one publication refusal passed")


if __name__ == "__main__":
    main()
