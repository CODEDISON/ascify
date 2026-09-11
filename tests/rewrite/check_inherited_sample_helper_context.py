#!/usr/bin/env python3
"""Pin the current inherited-helper boundary without accepting partial output."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/rewrite/fixtures/inherited_sample_helper"
HELPERS = ROOT / "tests/rewrite/fixtures/nvidia_samples/Common"


def main():
    binary = os.environ["ASCIFY_BINARY"]
    cuda = os.environ["ASCIFY_CUDA_PATH"]
    resource = os.environ["ASCIFY_CLANG_RESOURCE_DIRECTORY"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in FIXTURE.iterdir()}
    with tempfile.TemporaryDirectory(prefix="ascify-inherited-helper-") as temp:
        work = Path(temp)

        def convert(output, recursive):
            args = [binary, str(FIXTURE / "root.cu"),
                    f"--cuda-path={cuda}", f"--clang-resource-directory={resource}",
                    "-o", str(output), "-I" + str(HELPERS)]
            if recursive:
                args.append("--local-headers-recursive")
            args += ["--", "-std=c++17", "-I" + str(HELPERS)]
            return subprocess.run(args, text=True, capture_output=True)

        # Establish that the fixture is a legal CUDA TU with the parent helper
        # include. Failure here is not evidence for the recursive boundary.
        root_only = work / "root-only.cce"
        result = convert(root_only, False)
        assert result.returncode == 0, result.stderr
        assert "#include <helper_cuda.h>" in root_only.read_text()
        assert "helper macro expansion outside the main file" in result.stderr
        print("original parent TU parsed successfully; external helper transaction retained")

        # The real recursive request must fail at the independently parsed
        # child's inherited macro, and must publish no root/header bundle.
        output = work / "recursive.cce"
        result = convert(output, True)
        assert result.returncode != 0, "unsupported inherited context was accepted"
        assert "use of undeclared identifier 'getLastCudaError'" in result.stderr, result.stderr
        assert "parent macro/declaration context is not replayed" in result.stderr, result.stderr
        assert not output.exists() and not Path(str(output) + ".headers").exists()
        assert not list(work.glob(".ascify-local-closure*")), "staging leaked"
        print("recursive standalone child rejected inherited helper; no output or staging published")

        # A failed retry must not replace a previously existing root artifact.
        output.write_text("// retained previous output\n")
        prior = output.read_bytes()
        result = convert(output, True)
        assert result.returncode != 0 and output.read_bytes() == prior
        assert not Path(str(output) + ".headers").exists()
        print("failed recursive retry preserved previous output")
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in FIXTURE.iterdir()}
    print("fixed input unchanged; all temporary probes removed; tool boundary, not hardware")


if __name__ == "__main__":
    main()
