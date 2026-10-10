#!/usr/bin/env python3
"""Check the Samples host-local initializer path and retained helper boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/rewrite/fixtures"
HELPERS = FIXTURES / "nvidia_samples/Common"
PREAMBLE = """#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
"""
CHECKS = """  float *data = nullptr;
  checkCudaErrors(cudaMalloc((void **)&data, 128));
  checkCudaErrors(cudaFree(data));
  getLastCudaError("scope transaction");
"""
FIND = "findCudaDevice(argc, (const char **)argv)"


def local_body(declaration):
    # The original template Sample uses this ordinary runTest local, followed
    # by direct runtime checks. Keep its kernel in the complete Sample gate.
    return ("void runTest(int argc, char **argv) {\n  " + declaration +
            "\n" + CHECKS + "  (void)devID;\n}\n")


def cases():
    for name, declaration in (
        ("template_local", "int devID = " + FIND + ";"),
        ("const_local", "const int devID = " + FIND + ";"),
        ("auto_local", "auto devID = " + FIND + ";"),
        ("direct_local", "int devID(" + FIND + ");"),
        ("list_local", "int devID{" + FIND + "};"),
        ("existing_assignment", "int devID; devID = " + FIND + ";"),
    ):
        yield name, local_body(declaration), True
    yield ("host_local_class",
           "void outer(int argc, char **argv) {\n"
           "  struct Selector {\n" + local_body("int devID = " + FIND + ";") +
           "  };\n  Selector selector; selector.runTest(argc, argv);\n}\n", True)

    later = "void laterChecks() {\n" + CHECKS + "}\n"
    for name, body in (
        ("global", "int devID = findCudaDevice(0, nullptr);\n"),
        ("namespace", "namespace project { int devID = findCudaDevice(0, nullptr); }\n"),
        ("static_local", local_body("static int devID = " + FIND + ";")),
        ("thread_local", local_body("thread_local int devID = " + FIND + ";")),
        ("lambda_body", "void runTest(int argc, char **argv) {\n"
         "  auto select = [&]() { int devID = " + FIND + "; return devID; };\n"
         "  (void)select();\n}\n"),
        ("lambda_capture", "void runTest(int argc, char **argv) {\n"
         "  auto select = [devID = " + FIND + "] { return devID; };\n"
         "  (void)select();\n}\n"),
    ):
        yield name, body + later, False
    # Device/global functions retain the converter's CUDA-attribute rejection,
    # but are not claimed as tested here: Clang rejects the host-only frozen
    # helper call even in sizeof, before that semantic boundary can be reached.
    # A valid local candidate visited first must not be partially published
    # when a later static initializer disqualifies the helper transaction.
    yield ("mixed_local_static",
           local_body("int devID = " + FIND + ";") +
           "int other(int argc, char **argv) { static int devID = " + FIND +
           "; return devID; }\n", False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sys.flags.optimize:
        raise RuntimeError("scope validation requires Python assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cuda-path", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    frozen = [*HELPERS.iterdir(),
              FIXTURES / "nvidia_samples_v13_4/Common/helper_cuda.h"]
    before = {str(path): sha(path) for path in frozen}
    with tempfile.TemporaryDirectory(prefix="ascify-find-scope-") as temporary:
        work = args.evidence_dir or Path(temporary)
        work.mkdir(parents=True, exist_ok=True)
        profiles = {"v13_3": HELPERS}
        newer = work / "v13_4/Common"
        shutil.copytree(HELPERS, newer)
        shutil.copyfile(frozen[-1], newer / "helper_cuda.h")
        profiles["v13_4"] = newer
        results = []
        for profile, helpers in profiles.items():
            for name, body, accepted in cases():
                # All scope cases use the first frozen profile. Also exercise
                # the unchanged complete-Sample form against the second one.
                if profile == "v13_4" and name != "template_local":
                    continue
                label = profile + "_" + name
                directory = work / label
                directory.mkdir()
                source = directory / "input.cu"
                output = directory / "output.cpp"
                receipt = directory / "receipt.json"
                source.write_text(PREAMBLE + body)
                input_sha = sha(source)
                command = [
                    args.binary, str(source), "--target-policy=dav-c310-vec",
                    "--simt-math=fast", "--target-recipe=none",
                    "--default-preprocessor", "--cuda-path=" + args.cuda_path,
                    "--clang-resource-directory=" + args.resource_dir,
                    "--migration-receipt=" + str(receipt), "-o", str(output),
                    "--", "-x", "cuda", "-std=c++17", "-I" + str(helpers),
                ]
                result = subprocess.run(command, text=True, capture_output=True, timeout=60)
                (directory / "argv.json").write_text(json.dumps(command, indent=2) + "\n")
                (directory / "stdout.log").write_text(result.stdout)
                (directory / "stderr.log").write_text(result.stderr)
                assert result.returncode == 0, (label, result.stderr)
                assert sha(source) == input_sha, label + ": input modified"
                assert json.loads(receipt.read_text())["status"] == "succeeded", label
                text = output.read_text()
                if accepted:
                    assert "#include <helper_cuda.h>" not in text, (label, result.stderr)
                    assert "::ascify::sampleFindCudaDevice(argc, (const char **)argv)" in text, label
                    assert "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaMalloc" in text, label
                    assert 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR("scope transaction")' in text, label
                    assert "find_device_rewrites=1" in result.stderr, label
                    assert "include removed" in result.stderr, label
                else:
                    # Existing refusal contract: publish a conversion with the
                    # entire original helper transaction retained, not a
                    # partially rewritten host selector or error checker.
                    assert "#include <helper_cuda.h>" in text, (label, result.stderr)
                    assert "findCudaDevice(" in text, label
                    assert "checkCudaErrors(ascify::cudaMalloc" in text, label
                    assert 'getLastCudaError("scope transaction")' in text, label
                    for forbidden in ("::ascify::sampleFindCudaDevice",
                                      "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS",
                                      "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR"):
                        assert forbidden not in text, (label, forbidden)
                    assert "residual helper declaration 'findCudaDevice'" in result.stderr, label
                    for count in ("find_device_rewrites=0", "check_rewrites=0", "get_last_rewrites=0"):
                        assert count in result.stderr, label
                    assert "include kept" in result.stderr, label
                results.append({"case": label, "accepted": accepted,
                                "input_sha256": input_sha, "output_sha256": sha(output),
                                "returncode": result.returncode})
                print(label + (": local helper transaction committed" if accepted
                               else ": complete helper output retained"))
        assert {str(path): sha(path) for path in frozen} == before
        (work / "summary.json").write_text(json.dumps({"passed": True, "cases": results}, indent=2) + "\n")
        print(f"find-device scope: {len(results)} cases passed")


if __name__ == "__main__":
    main()
