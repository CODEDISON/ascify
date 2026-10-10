#!/usr/bin/env python3
"""Exercise adjacent local CUDA status proofs with the real Ascify frontend."""
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
# Preserve sortingNetworks' direct header graph and host error-check shape.
# Its original kernels and CPU validation belong to the complete Sample run.
PREAMBLE = """#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
"""
SYNC = "error = cudaDeviceSynchronize();\n  checkCudaErrors(error);"


def cases():
    yield "adjacent", "cudaError_t error;\n  " + SYNC, True, ""
    yield "sorting_repeated_loop", """void *data = nullptr;
  cudaError_t error;
  error = cudaMalloc(&data, 64);
  checkCudaErrors(error);
  for (int iteration = 0; iteration < 2; ++iteration) {
    error = cudaDeviceSynchronize();
    checkCudaErrors(error);
  }
  error = cudaFree(data);
  checkCudaErrors(error);""", True, ""

    # All negative cases are legal CUDA C++. Enum casts below deliberately
    # avoid invalid implicit integer-to-enum assignments masking the proof.
    negatives = (
        ("type_without_assignment", "cudaError_t error = cudaDeviceSynchronize();\n"
         "  checkCudaErrors(error);", ""),
        ("numeric_status", "cudaError_t error;\n"
         "  error = static_cast<cudaError_t>(0);\n  checkCudaErrors(error);", ""),
        ("user_return", "cudaError_t error;\n"
         "  error = projectStatus();\n  checkCudaErrors(error);",
         "cudaError_t projectStatus() { return cudaSuccess; }\n"),
        ("explicit_cast", "cudaError_t error;\n"
         "  error = static_cast<cudaError_t>(cudaDeviceSynchronize());\n"
         "  checkCudaErrors(error);", ""),
        ("intervening_statement", "cudaError_t error;\n"
         "  error = cudaDeviceSynchronize();\n  (void)0;\n"
         "  checkCudaErrors(error);", ""),
        ("conditional_assignment", "cudaError_t error = cudaSuccess;\n"
         "  if (argc > 0) error = cudaDeviceSynchronize();\n"
         "  checkCudaErrors(error);", ""),
        ("nonstandalone_check", "cudaError_t error;\n"
         "  error = cudaDeviceSynchronize();\n"
         "  (checkCudaErrors(error), (void)0);", ""),
        ("address_escape", "cudaError_t error;\n"
         "  cudaError_t *escaped = &error;\n  " + SYNC + "\n  (void)escaped;", ""),
        ("reference_escape", "cudaError_t error;\n"
         "  cudaError_t &escaped = error;\n  " + SYNC + "\n  (void)escaped;", ""),
        ("lambda_capture", "cudaError_t error;\n"
         "  auto observe = [&error] { return error; };\n  " + SYNC +
         "\n  (void)observe;", ""),
        # Assignment yields an lvalue. An alias can therefore escape without
        # taking a reference/address of a direct DeclRefExpr to error.
        ("assignment_reference_escape", "cudaError_t error;\n"
         "  auto &alias = (error = cudaSuccess);\n  " + SYNC +
         "\n  (void)alias;", ""),
        ("assignment_address_escape", "cudaError_t error;\n"
         "  auto *alias = &(error = cudaSuccess);\n  " + SYNC +
         "\n  (void)alias;", ""),
        ("assignment_capture_escape", "cudaError_t error;\n"
         "  auto observe = [&alias = (error = cudaSuccess)] { return alias; };\n  " +
         SYNC + "\n  (void)observe;", ""),
        ("different_variable", "cudaError_t error = cudaSuccess;\n"
         "  cudaError_t other;\n  other = cudaDeviceSynchronize();\n"
         "  checkCudaErrors(error);\n  (void)other;", ""),
        ("static_local", "static cudaError_t error;\n  " + SYNC, ""),
        ("thread_local", "thread_local cudaError_t error;\n  " + SYNC, ""),
        ("global_status", SYNC, "cudaError_t error;\n"),
        ("volatile_local", "volatile cudaError_t error;\n  " + SYNC, ""),
        ("macro_assignment", "cudaError_t error;\n  ASSIGN_ERROR();\n"
         "  checkCudaErrors(error);",
         "#define ASSIGN_ERROR() error = cudaDeviceSynchronize()\n"),
        ("second_check", "cudaError_t error;\n  " + SYNC +
         "\n  checkCudaErrors(error);", ""),
    )
    for name, body, prefix in negatives:
        yield name, body, False, prefix


def source_text(body, prefix):
    # Keep independently admitted helper operations before and after the local
    # status use. A failed proof must retain these too, not publish a subset.
    return (PREAMBLE + prefix +
            "void runTest(int argc, char **argv) {\n"
            "  int device = findCudaDevice(argc, (const char **)argv);\n"
            "  checkCudaErrors(cudaSetDevice(device));\n  " + body + "\n"
            '  getLastCudaError("local status transaction");\n}\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sys.flags.optimize:
        raise RuntimeError("local-status validation requires Python assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cuda-path", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    binary = Path(args.binary).resolve()
    frozen = [*HELPERS.iterdir(),
              FIXTURES / "nvidia_samples_v13_4/Common/helper_cuda.h"]
    before = {str(path): sha(path) for path in frozen}
    with tempfile.TemporaryDirectory(prefix="ascify-local-status-") as temporary:
        work = args.evidence_dir or Path(temporary)
        if args.evidence_dir:
            work.mkdir(parents=True, exist_ok=False)
        newer = work / "v13_4/Common"
        shutil.copytree(HELPERS, newer)
        shutil.copyfile(frozen[-1], newer / "helper_cuda.h")
        summary = {"schema": "ascify.sample-local-status.v1", "passed": False,
                   "binary_sha256": sha(binary), "cases": [],
                   "scope": "real CUDA parsing and helper transaction; no target compilation or device execution"}

        def save():
            (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        save()
        try:
            for profile, helpers in (("v13_3", HELPERS), ("v13_4", newer)):
                for name, body, accepted, prefix in cases():
                    if profile == "v13_4" and name != "sorting_repeated_loop":
                        continue
                    label = profile + "_" + name
                    directory = work / label
                    directory.mkdir()
                    source = directory / "input.cu"
                    output = directory / "output.cpp"
                    receipt = directory / "receipt.json"
                    original = source_text(body, prefix)
                    source.write_text(original)
                    row = {"case": label, "expected_admitted": accepted,
                           "input_sha256": sha(source), "passed": False}
                    summary["cases"].append(row)
                    command = [
                        str(binary), str(source), "--target-policy=dav-c310-vec",
                        "--simt-math=fast", "--target-recipe=none", "--default-preprocessor",
                        "--cuda-path=" + args.cuda_path,
                        "--clang-resource-directory=" + args.resource_dir,
                        "--migration-receipt=" + str(receipt), "-o", str(output),
                        "--", "-x", "cuda", "-std=c++17", "-I" + str(helpers),
                    ]
                    (directory / "argv.json").write_text(json.dumps(command, indent=2) + "\n")
                    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
                    (directory / "stdout.log").write_text(result.stdout)
                    (directory / "stderr.log").write_text(result.stderr)
                    row["returncode"] = result.returncode
                    save()
                    # An invalid CUDA source or infrastructure error can never
                    # stand in for the negative semantic result below.
                    assert result.returncode == 0, (label, result.stderr)
                    assert sha(source) == row["input_sha256"], label + ": source modified"
                    assert json.loads(receipt.read_text())["status"] == "succeeded", label
                    text = output.read_text()
                    check_count = original.count("checkCudaErrors(")
                    if accepted:
                        assert "#include <helper_cuda.h>" not in text, (label, result.stderr)
                        assert "::ascify::sampleFindCudaDevice(argc, (const char **)argv)" in text, label
                        assert text.count("ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(") == check_count, label
                        assert text.count("ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(error)") == body.count("checkCudaErrors(error)"), label
                        assert 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR("local status transaction")' in text, label
                        assert "include removed" in result.stderr, (label, result.stderr)
                    else:
                        assert "#include <helper_cuda.h>" in text, (label, result.stderr)
                        assert "findCudaDevice(argc, (const char **)argv)" in text, label
                        assert text.count("checkCudaErrors(") == check_count, label
                        assert text.count("checkCudaErrors(error)") == body.count("checkCudaErrors(error)"), label
                        assert 'getLastCudaError("local status transaction")' in text, label
                        for forbidden in ("::ascify::sampleFindCudaDevice",
                                          "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS",
                                          "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR"):
                            assert forbidden not in text, (label, forbidden)
                        assert "status domain not proven" in result.stderr, (label, result.stderr)
                        assert "include kept" in result.stderr, label
                        for count in ("find_device_rewrites=0", "check_rewrites=0", "get_last_rewrites=0"):
                            assert count in result.stderr, (label, count)
                    row.update({"passed": True, "output_sha256": sha(output),
                                "receipt_sha256": sha(receipt)})
                    save()
                    print(label + (": helper transaction committed" if accepted
                                   else ": complete helper transaction retained"), flush=True)
            assert {str(path): sha(path) for path in frozen} == before
            assert sha(binary) == summary["binary_sha256"], "converter changed during matrix"
            summary.update({"passed": True, "total": len(summary["cases"]),
                            "admitted": sum(row["expected_admitted"] for row in summary["cases"])})
            summary["retained"] = summary["total"] - summary["admitted"]
            save()
        except Exception as error:
            summary["failure"] = str(error)
            save()
            raise
        print(f"local-status frontend: {summary['total']} cases passed")


if __name__ == "__main__":
    main()
