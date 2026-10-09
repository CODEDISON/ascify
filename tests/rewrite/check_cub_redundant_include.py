#!/usr/bin/env python3
"""Check frozen CUB redundant-include removal and retained safety boundaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cuda-path", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--source-clang")
    parser.add_argument("--baseline-binary")
    parser.add_argument("--ccec")
    parser.add_argument("--target-include", action="append", default=[])
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    cccl = Path(args.cuda_path) / "include/cccl"
    providers = {
        "cub/cub.cuh": "c0b0dc5079bb8ead26a87d3dce77ea0f41522a797a5426787ecdfb0f6f3dd749",
        "cub/util_type.cuh": "01b79eff9d4d050fb05d9bcc3fd132c6345b284e4dbf972979b48f31a72d970a",
    }
    if any(not (cccl / name).is_file() or sha(cccl / name) != digest
           for name, digest in providers.items()):
        print("CUB redundant-include matrix skipped: requires frozen CUDA 13.4.1 providers")
        return
    temporary = None
    if args.evidence_dir:
        require(not args.evidence_dir.exists(), "Use a new evidence directory")
        root = args.evidence_dir
        root.mkdir(parents=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="ascify-cub-redundant-")
        root = Path(temporary.name)
    env = os.environ.copy()
    for key in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "OBJC_INCLUDE_PATH"):
        env.pop(key, None)
    umbrella = "#include <cub/cub.cuh>\n"
    util = "#include <cub/util_type.cuh>\n"
    scalar = "__global__ void observer(float* p){p[0]=1;}\n"
    cases = {
        "angle": (umbrella + util + scalar, "removed", 1),
        "quoted": (umbrella + '#include "cub/util_type.cuh"\n' + scalar, "removed", 1),
        "repeated": (umbrella + util + util + scalar, "removed", 2),
        "comment_tail": (umbrella + '#include <cub/util_type.cuh> /* redundant */ // end\n' + scalar, "removed", 1),
        "block_reduce": (umbrella + util + "__global__ void observer(float* p){"
                         "using Reduce=cub::BlockReduce<float,32>;"
                         "__shared__ Reduce::TempStorage storage;"
                         "float r=Reduce(storage).Sum(p[threadIdx.x]);"
                         "if(threadIdx.x==0)p[0]=r;}\n", "removed", 1),
        "conditional_subheader": (umbrella + "#if 1\n" + util + "#endif\n" + scalar, "retained", 0),
        "conditional_umbrella": ("#if 1\n" + umbrella + "#endif\n" + util + scalar, "retained", 0),
        "macro_subheader": (umbrella + '#define SUBHEADER "cub/util_type.cuh"\n#include SUBHEADER\n' + scalar, "retained", 0),
        "relative_subheader": (umbrella + '#include <cub/../cub/util_type.cuh>\n' + scalar, "retained", 0),
        "transitive_subheader": (umbrella + '#include "holder.h"\n' + scalar, "retained", 0),
        "multiline_directive": (umbrella + '#include \\\n<cub/util_type.cuh>\n' + scalar, "retained", 0),
        "multiline_tail_comment": (umbrella + '#include <cub/util_type.cuh> /* first\nsecond */\n' + scalar, "retained", 0),
        "ignored_tail_tokens": (umbrella + '#include <cub/util_type.cuh> using Hidden = float3;\n' + scalar, "retained", 0),
        "continued_tail": (umbrella + '#include <cub/util_type.cuh> \\\nusing Hidden = float3;\n' + scalar, "retained", 0),
        "standalone": (util + scalar, "rejected", 0),
        "subheader_first": (util + umbrella + scalar, "rejected", 0),
        "macro_umbrella": ('#define UMBRELLA "cub/cub.cuh"\n#include UMBRELLA\n' + util + scalar, "rejected", 0),
        "transitive_umbrella": ('#include "entry.h"\n' + util + scalar, "rejected", 0),
        "copied_umbrella": (umbrella + util + scalar, "rejected", 0),
        "copied_util": (umbrella + util + scalar, "rejected", 0),
        "caller_float3": (umbrella + util + 'using Caller = float3;\n' + scalar, "rejected", 1),
        "caller_uint3": (umbrella + util + 'void caller(){uint3 v={1,2,3};}\n' + scalar, "rejected", 1),
        "caller_inherited_vec3": (umbrella + util + 'using Caller = cub::CubVector<float,3>;\n' + scalar, "rejected", 1),
    }
    marker = "Ascify CUB redundant include: removed frozen cub/util_type.cuh"

    def run(command, directory):
        directory.mkdir()
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
        (directory / "argv.json").write_text(json.dumps(command, indent=2) + "\n")
        (directory / "stdout.log").write_text(result.stdout)
        (directory / "stderr.log").write_text(result.stderr)
        (directory / "process.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
        return result

    rows = []
    try:
        for name, (body, expected, removals) in cases.items():
            directory = root / name
            directory.mkdir()
            source = directory / "input.cu"
            output = directory / "output.cce"
            receipt = directory / "receipt.json"
            source.write_text("#include <cuda_runtime.h>\n" + body)
            (directory / "holder.h").write_text(util)
            (directory / "entry.h").write_text(umbrella)
            extra = []
            if name in ("copied_umbrella", "copied_util"):
                shadow = directory / "shadow/cub"
                shadow.mkdir(parents=True)
                header = "cub.cuh" if name == "copied_umbrella" else "util_type.cuh"
                shutil.copyfile(cccl / "cub" / header, shadow / header)
                extra.extend(["-I", str(shadow.parent)])
            extra.extend(["-I", str(cccl)])
            command = [args.binary, str(source), "--frontend-compat=ascify-admitted-v1",
                       "--target-policy=dav-c310-vec", "--simt-math=fast", "--default-preprocessor",
                       "--cuda-path=" + args.cuda_path, "--clang-resource-directory=" + args.resource_dir,
                       "--migration-receipt=" + str(receipt), "-o", str(output),
                       "--", "-x", "cuda", "-std=c++17", *extra]
            row = {"name": name, "expected": expected, "expected_removals": removals,
                   "source_sha256": sha(source)}
            if args.source_clang:
                source_result = run([args.source_clang, "-x", "cuda", "--cuda-host-only",
                                     "-nocudalib", "--cuda-path=" + args.cuda_path,
                                     "-resource-dir=" + args.resource_dir, "-std=c++17",
                                     "-fsyntax-only", *extra, str(source)], directory / "source")
                row["source_returncode"] = source_result.returncode
                require(source_result.returncode == 0, name + ": invalid CUDA source\n" + source_result.stderr)
            result = run(command, directory / "candidate")
            row["candidate_returncode"] = result.returncode
            row["removals"] = result.stderr.count(marker)
            require(row["removals"] == removals, name + ": unexpected removal proof\n" + result.stderr)
            if expected == "rejected":
                require(result.returncode != 0 and not output.exists() and receipt.is_file()
                        and json.loads(receipt.read_text()).get("status") == "failed"
                        and "CUDA vec3 ABI boundary:" in result.stderr,
                        name + ": expected caller/provider rejection\n" + result.stderr)
                sentinel = b"existing complete output\n"
                output.write_bytes(sentinel)
                repeated = run(command, directory / "preservation")
                require(repeated.returncode != 0 and output.read_bytes() == sentinel,
                        name + ": rejected conversion changed prior output")
            else:
                require(result.returncode == 0 and output.is_file(), name + ": generation failed\n" + result.stderr)
                row["output_sha256"] = sha(output)
                if expected == "removed":
                    require("util_type.cuh" not in output.read_text(), name + ": redundant header remained")
                    if args.baseline_binary:
                        baseline_output = directory / "baseline.cce"
                        baseline_command = command.copy()
                        baseline_command[0] = args.baseline_binary
                        baseline_command[baseline_command.index(str(output))] = str(baseline_output)
                        baseline_command[baseline_command.index("--migration-receipt=" + str(receipt))] = (
                            "--migration-receipt=" + str(directory / "baseline.receipt.json"))
                        baseline = run(baseline_command, directory / "baseline")
                        require(baseline.returncode == 0 and baseline_output.is_file()
                                and "util_type.cuh" in baseline_output.read_text(),
                                name + ": baseline did not reproduce retained include")
                        row["baseline_returncode"] = baseline.returncode
                    if args.ccec:
                        obj = directory / "output.o"
                        target = run([args.ccec, "-x", "dpp", "--cce-aicore-arch=dav-c310-vec",
                                      "-std=c++17", "-O2", "-c",
                                      *["-I" + path for path in args.target_include],
                                      str(output), "-o", str(obj)], directory / "target")
                        row["target_returncode"] = target.returncode
                        require(target.returncode == 0 and obj.is_file() and obj.read_bytes().startswith(b"\x7fELF"),
                                name + ": target object failed\n" + target.stderr)
            require(sha(source) == row["source_sha256"], name + ": source changed")
            rows.append(row)
        (root / "summary.json").write_text(json.dumps({"passed": True, "cases": rows}, indent=2) + "\n")
        print(f"CUB redundant-include matrix passed: {len(rows)} cases")
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
