#!/usr/bin/env python3
"""Convert the public FP32 example with an installed Ascify (no target execution).

Omit --resource-dir to check automatic discovery of the installed Clang headers.
Temporary conversion outputs are removed after their identities are recorded.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    return result, {"returncode": result.returncode,
                    "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                    "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--cuda-path", required=True, type=Path)
    parser.add_argument("--resource-dir", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    require(binary.is_file() and os.access(binary, os.X_OK), "--binary must be executable")
    require((args.cuda_path / "include/cuda_runtime.h").is_file(), "CUDA parsing root lacks cuda_runtime.h")
    source = REPO / "examples/vector_add.cu"
    identity = {"binary_sha256": digest(binary), "source_sha256": digest(source)}
    require(not re.search(r"\bdouble\b", source.read_text()), "Public example must use the supported FP32 domain")
    with tempfile.TemporaryDirectory(prefix="ascify-install-check-") as directory:
        work = Path(directory)
        output = work / "vector_add.cce"
        receipt = work / "receipt.json"
        command = [str(binary), str(source), "--cuda-path=" + str(args.cuda_path.resolve()),
                   "--target-policy=dav-c310-vec", "--simt-math=fast",
                   "--frontend-compat=ascify-admitted-v1", "-o", str(output),
                   "--migration-receipt=" + str(receipt)]
        if args.resource_dir:
            command.append("--clang-resource-directory=" + str(args.resource_dir.resolve()))
        result, conversion = run(command + ["--", "-std=c++17"])
        require(result.returncode == 0, "Example conversion failed:\n" + result.stderr[-8000:])
        require(output.is_file() and output.stat().st_size > 0, "Example conversion produced no output")
        data = json.loads(receipt.read_text())
        require(data["schema"] == "ascify.migration-receipt" and data["status"] == "succeeded",
                "Conversion receipt did not report success")
        require(len(data["inputs"]) == 1 and data["inputs"][0]["status"] == "succeeded",
                "The input receipt did not report success")
        translated = output.read_text()
        for expected in ("ascify/ascify_cuda_compat.hpp", "ascify::cudaMalloc",
                         "ascify::cudaMemcpy", "ascify::cudaFree"):
            require(expected in translated, "Missing expected Ascend compatibility interface: " + expected)
        require(not re.search(r"\b(?:hip\w*|rocm\w*)\b", translated, re.I), "Output names a legacy target")
        conversion["output_sha256"] = digest(output)
        conversion["receipt_sha256"] = digest(receipt)

        # An explicit invalid resource path must not silently use the installation.
        sentinel = b"previous output must survive a failed conversion\n"
        output.write_bytes(sentinel)
        invalid_resource = work / "invalid-resources"
        invalid_resource.mkdir()
        invalid = [str(binary), str(source), "--cuda-path=" + str(args.cuda_path.resolve()),
                   "--clang-resource-directory=" + str(invalid_resource), "-o", str(output),
                   "--", "-std=c++17"]
        rejected, invalid_record = run(invalid)
        require(rejected.returncode != 0 and "resource" in rejected.stderr.lower(),
                "Explicit invalid resource directory was not diagnosed")
        require(output.read_bytes() == sentinel, "Failed conversion replaced existing output")
        require(identity["binary_sha256"] == digest(binary) and identity["source_sha256"] == digest(source),
                "Binary or source changed while checking installation")
        print(json.dumps({"schema": "ascify.install-check.v1", "passed": True,
                          **identity, "resource_discovery": "explicit" if args.resource_dir else "automatic",
                          "example_conversion": conversion, "invalid_resource_rejection": invalid_record,
                          "input_and_binary_unchanged": True,
                          "target_compilation": "not_run", "device_execution": "not_run"}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        print("Installation check failed: " + str(error), file=sys.stderr)
        sys.exit(1)
