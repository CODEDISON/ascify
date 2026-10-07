#!/usr/bin/env python3
"""Require explicit preparse refusal of unsupported driver filesystem overlays."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

MESSAGE = "Ascify cannot translate input selected by VFS overlay arguments"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cuda-path", required=True)
    parser.add_argument("--resource-dir", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ascify-overlay-refusal-") as temp:
        work = Path(temp)
        source = work / "input.cu"
        source.write_text('#include "chosen.h"\nint value = SELECTED;\n')
        header = work / "chosen.h"
        header.write_text("#define SELECTED 1\n")
        shadow = work / "selected.h"
        shadow.write_text("#define SELECTED 7\n")
        original = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (source, header, shadow)}
        for name, selected in [("main", source), ("header", header),
                               ("unreferenced", work / "unused.h")]:
            overlay = work / (name + ".json")
            overlay.write_text(json.dumps({"version": 0, "roots": [{
                "type": "file", "name": str(selected),
                "external-contents": str(shadow)}]}))
            output = work / (name + ".dpp")
            result = subprocess.run([
                args.binary, str(source), "--default-preprocessor",
                "--target-policy=dav-c310-vec", "--simt-math=fast",
                "--cuda-path=" + args.cuda_path,
                "--clang-resource-directory=" + args.resource_dir,
                "-o", str(output), "--", "-x", "cuda", "-std=c++17",
                "-I" + str(work), "-ivfsoverlay", str(overlay),
            ], capture_output=True, text=True, timeout=60)
            assert result.returncode != 0 and MESSAGE in result.stderr, (
                name, result.returncode, result.stderr)
            assert not output.exists(), name
            assert {p: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in original} == original
            print(name + ": explicit VFS configuration refused before publication")
    print("VFS configuration refusal passed; no overlay AST admission claimed")


if __name__ == "__main__":
    main()
