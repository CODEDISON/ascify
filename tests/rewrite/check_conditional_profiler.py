#!/usr/bin/env python3
"""Real-SDK source/transaction checks for a positive profiler header probe.

Never execute a translated program or use NPU. A backend header/provider is
deliberately not declared verified by these frontend tests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--clang", type=Path, required=True)
parser.add_argument("--cuda-path", type=Path, required=True)
parser.add_argument("--resource-dir", type=Path, required=True)
parser.add_argument("--evidence-dir", type=Path, required=True)
parser.add_argument("--source-only", action="store_true")
args = parser.parse_args()
for key in ("binary", "clang", "cuda_path", "resource_dir"):
    setattr(args, key, getattr(args, key).resolve(strict=True))
args.evidence_dir = args.evidence_dir.resolve()
assert (args.cuda_path / "include/cuda_profiler_api.h").is_file()
assert (args.resource_dir / "include/__clang_cuda_runtime_wrapper.h").is_file()
version = subprocess.check_output([str(args.clang), "--version"], text=True)
assert re.search(r"\bclang version 23(?:\.|\s)", version), version
args.evidence_dir.mkdir(parents=True, exist_ok=False)

runtime = "#include <cuda_runtime.h>\n"
condition = "#if defined(__has_include) && __has_include(<cuda_profiler_api.h>)\n"
bare = "#if __has_include(<cuda_profiler_api.h>)\n"
include = "#include <cuda_profiler_api.h>\n"
flag = "#define ORIGINAL_PROFILE_FLAG 1\n"
block = condition + include + flag + "#endif\n"
calls = ("int main(){cudaError_t a=cudaProfilerStart();"
         "if(a!=cudaSuccess)return 1;"
         "cudaError_t b=cudaProfilerStop();return b==cudaSuccess?0:2;}\n")
guarded_calls = ("int main(){\n#ifdef ORIGINAL_PROFILE_FLAG\n"
                 "cudaError_t a=cudaProfilerStart();if(a!=cudaSuccess)return 1;\n"
                 "cudaError_t b=cudaProfilerStop();return b==cudaSuccess?0:2;\n"
                 "#else\nreturn 3;\n#endif\n}\n")
empty = "int main(){return 0;}\n"
cases = [
    ("defined_cuda", runtime + block + guarded_calls, "cuda", False, True, False),
    ("bare_cuda", runtime + bare + include + "#endif\n" + calls,
     "cuda", False, False, False),
    ("defined_host_cpp", "#include <cuda_runtime_api.h>\n" + block + guarded_calls,
     "c++", False, True, False),
    ("commented_fresh_flag",
     runtime + (block + guarded_calls).replace("ORIGINAL_PROFILE_FLAG", "MY_LOCAL_CAPTURE")
     .replace(" && ", " /* retained */ && "), "cuda", False, True, False),
    ("recursive_root", runtime + block + guarded_calls, "cuda", False, True, True),
    ("implicit_builtin_line", runtime + block +
     'constexpr int observed=__builtin_LINE();\nstatic_assert(observed==6,"source position changed");\n' + guarded_calls,
     "cuda", False, True, False),
    ("outer_condition", runtime + "#if 1\n" + block + "#endif\n" + guarded_calls,
     "cuda", True, True, False),
    ("original_else", runtime + condition + include + "#else\nint alternate;\n#endif\n" + calls,
     "cuda", True, False, False),
    ("original_elif", runtime + condition + include + "#elif 1\nint alternate;\n#endif\n" + calls,
     "cuda", True, False, False),
    ("probe_or", runtime + bare.replace(")\n", ") || 0\n") + include + "#endif\n" + calls,
     "cuda", True, False, False),
    ("negated_probe", runtime + bare.replace("#if ", "#if !") + include + "#endif\n" + empty,
     "cuda", True, False, False),
    ("macro_condition", runtime + "#define QUERY __has_include\n" +
     bare.replace("__has_include", "QUERY") + include + "#endif\n" + calls,
     "cuda", True, False, False),
    ("macro_header", runtime + "#define PROFILE_HEADER <cuda_profiler_api.h>\n" +
     "#if __has_include(PROFILE_HEADER)\n#include PROFILE_HEADER\n#endif\n" + calls,
     "cuda", True, False, False),
    ("builtin_redefine_before", runtime +
     "#undef __has_include\n#define __has_include(x) 1\n" + block + guarded_calls,
     "cuda", True, True, False),
    ("builtin_redefine_after", runtime + block + guarded_calls +
     "#undef __has_include\n#define __has_include(x) 1\n",
     "cuda", True, True, False),
    ("flag_predefined", runtime + flag + block + guarded_calls, "cuda", True, True, False),
    ("flag_earlier_observation", runtime +
     "#if defined(ORIGINAL_PROFILE_FLAG)\nint earlier;\n#endif\n" + block + guarded_calls,
     "cuda", True, True, False),
    ("flag_late_undef", runtime + block + guarded_calls + "#undef ORIGINAL_PROFILE_FLAG\n",
     "cuda", True, True, False),
    ("flag_late_redefine", runtime + block + guarded_calls +
     "#undef ORIGINAL_PROFILE_FLAG\n#define ORIGINAL_PROFILE_FLAG 2\n",
     "cuda", True, True, False),
    ("flag_cpp_observation", runtime + block +
     "int observed=ORIGINAL_PROFILE_FLAG;\n" + guarded_calls,
     "cuda", True, True, False),
    ("guard_observation", runtime + block + guarded_calls +
     "#ifdef __CUDA_PROFILER_API_H__\nint observed;\n#endif\n",
     "cuda", True, True, False),
    ("other_inquiry", runtime + block + guarded_calls +
     "#if __has_include(<stdio.h>)\nint other;\n#endif\n",
     "cuda", True, True, False),
    ("second_candidate", runtime + bare + include + "#endif\n" +
     bare + include + "#endif\n" + calls, "cuda", True, False, False),
    ("skipped_sdk", runtime + include + block + guarded_calls, "cuda", True, True, False),
    ("pragma_inside", runtime + condition + include +
     '#pragma message("source-valid negative")\n#endif\n' + calls,
     "cuda", True, False, False),
    ("position_observation", runtime + block + "int observed=__LINE__;\n" + guarded_calls,
     "cuda", True, True, False),
    ("input_line_directive", "#line 700\n" + runtime + block + guarded_calls,
     "cuda", True, True, False),
    ("input_gnu_marker", '# 700 "caller_label.cpp"\n' + runtime + block + guarded_calls,
     "cuda", True, True, False),
    ("late_audit_recursive", runtime + block + guarded_calls +
     "#ifdef __CUDA_PROFILER_API_H__\nint observed;\n#endif\n",
     "cuda", True, True, True),
    ("shadow_identical_sdk", runtime + block + guarded_calls, "cuda", True, True, False),
    ("host_cpp_recursive_root", "#include <cuda_runtime_api.h>\n" + block + guarded_calls,
     "c++", False, True, True),
    ("host_cpp_implicit_builtin_line", "#include <cuda_runtime_api.h>\n" + block +
     'constexpr int observed=__builtin_LINE();\nstatic_assert(observed==6,"source position changed");\n' + guarded_calls,
     "c++", False, True, False),
    ("host_cpp_outer_condition", "#include <cuda_runtime_api.h>\n#if 1\n" +
     block + "#endif\n" + guarded_calls, "c++", True, True, False),
    ("host_cpp_shadow_identical_sdk", "#include <cuda_runtime_api.h>\n" + block + guarded_calls,
     "c++", True, True, False),
    ("host_cpp_contextual", '#define PARENT_FACTOR 3\n#include "caller_leaf.h"\n' +
     "#include <cuda_runtime_api.h>\n" + block + guarded_calls,
     "c++", True, True, True),
    ("host_cpp_vfs_overlay", "#include <cuda_runtime_api.h>\n" + block + guarded_calls,
     "c++", True, True, False),
]

summary = {
    "schema": "ascify.conditional-profiler-frontend-tests.v1",
    "source_only": args.source_only, "binary_sha256": sha(args.binary),
    "clang_sha256": sha(args.clang), "programs_executed": 0,
    "backend_status": "unverified", "cases": [], "passed": False,
}


def save():
    (args.evidence_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def run(directory, stage, argv):
    argv = list(map(str, argv))
    (directory / (stage + ".argv.json")).write_text(json.dumps(argv, indent=2) + "\n")
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=90)
    (directory / (stage + ".stdout.log")).write_text(result.stdout)
    (directory / (stage + ".stderr.log")).write_text(result.stderr)
    (directory / (stage + ".process.json")).write_text(
        json.dumps({"returncode": result.returncode, "argv": argv}, indent=2) + "\n")
    return result


save()
try:
    for name, source, language, refused, has_flag, recursive in cases:
        directory = args.evidence_dir / name
        directory.mkdir()
        original = directory / ("input.cu" if language == "cuda" else "input.cpp")
        original.write_text(source)
        converted, receipt = directory / "output.cce", directory / "receipt.json"
        extra = []
        if name in ("shadow_identical_sdk", "host_cpp_shadow_identical_sdk"):
            # A private negative fixture only: identical bytes are not proof
            # that lookup selected the configured physical SDK provider.
            shadow = directory / "caller_include"
            shadow.mkdir()
            (shadow / "cuda_profiler_api.h").write_bytes(
                (args.cuda_path / "include/cuda_profiler_api.h").read_bytes())
            extra = ["-I" + str(shadow)]
        if name == "host_cpp_contextual":
            (directory / "caller_leaf.h").write_text(
                "#ifndef ORIGINAL_CALLER_LEAF_H\n#define ORIGINAL_CALLER_LEAF_H\n"
                "constexpr int inherited=PARENT_FACTOR;\n#endif\n")
        if name == "host_cpp_vfs_overlay":
            shadow = directory / "caller_root.cpp"
            shadow.write_text(source)
            overlay = directory / "caller_overlay.json"
            overlay.write_text(json.dumps({"version": 0, "use-external-names": False,
                "roots": [{"type": "file", "name": str(original),
                           "external-contents": str(shadow)}]}))
            extra = ["-ivfsoverlay", str(overlay)]
        input_hashes = {str(p): sha(p) for p in directory.rglob("*") if p.is_file()}
        common = ["-x", language, "-std=c++17", *extra,
                  "-I" + str(args.cuda_path / "include"),
                  *(["--cuda-path=" + str(args.cuda_path)] if language == "cuda" else []),
                  "-resource-dir=" + str(args.resource_dir)]
        syntax = [args.clang, original, *common, "-fsyntax-only",
                  "-Wno-unknown-cuda-version"]
        if language == "cuda":
            syntax += ["--cuda-host-only", "-nocudalib", "-fgpu-rdc"]
        row = {"name": name, "source_sha256": sha(original),
               "expected_refusal": refused, "passed": False}
        summary["cases"].append(row)
        result = run(directory, "original_syntax", syntax)
        row["syntax_rc"] = result.returncode
        save()
        assert result.returncode == 0, (name, result.stderr)
        if not args.source_only:
            argv = [args.binary, original, "--target-policy=dav-c310-vec",
                    "--simt-math=fast", "--target-recipe=none", "--default-preprocessor",
                    "--frontend-compat=" + ("ascify-admitted-v1" if language == "cuda" else "none"),
                    "--cuda-path=" + str(args.cuda_path),
                    "--clang-resource-directory=" + str(args.resource_dir),
                    "--migration-receipt=" + str(receipt), "-o", converted]
            if recursive:
                argv.append("--local-headers-recursive")
            argv += ["--", *common]
            if language == "cuda":
                argv.append("-fgpu-rdc")
            result = run(directory, "convert", argv)
            row["conversion_rc"] = result.returncode
            record = json.loads(receipt.read_text())
            if refused:
                assert result.returncode != 0 and not converted.exists(), (name, result.stderr)
                assert record["status"] == "failed", record
                assert not any("cuda_profiler_availability" in item for item in record["inputs"]), record
            else:
                assert result.returncode == 0, (name, result.stderr)
                text = converted.read_text()
                assert "__has_include(<acl/acl_prof.h>)" in text, (name, text)
                assert "__has_include(<cuda_profiler_api.h>)" not in text, (name, text)
                assert ('#else\n#error "Ascify admitted profiler branch requires the target '
                        'ACL profiler header"\n#endif') in text, (name, text)
                resume_line = source[:source.index("#endif\n") + len("#endif\n")].count("\n") + 1
                assert "#line " + str(resume_line) + "\n" in text, (name, text)
                if name in ("implicit_builtin_line", "host_cpp_implicit_builtin_line"):
                    assert 'static_assert(observed==6,"source position changed")' in text, text
                assert "ascify::cudaProfilerStart()" in text and "ascify::cudaProfilerStop()" in text, text
                flag_name = "MY_LOCAL_CAPTURE" if name == "commented_fresh_flag" else "ORIGINAL_PROFILE_FLAG"
                if has_flag:
                    assert "#define " + flag_name + " 1" in text, text
                    assert "#ifdef " + flag_name in text, text
                assert record["status"] == "succeeded", record
                assert len(record["inputs"]) == 1, record
                proof = record["inputs"][0]["cuda_profiler_availability"]
                assert proof["rule"] == "positive-physical-sdk-probe-v1", proof
                assert proof["backend_status"] == "unverified", proof
                assert proof["target_header"] == "acl/acl_prof.h", proof
                assert len(proof["evidence"]) == 1 and "sdk_sha256=bb8e03c2" in proof["evidence"][0], proof
                assert ";resume_line=" + str(resume_line) in proof["evidence"][0], proof
                row["output_sha256"] = sha(converted)
            row["receipt_sha256"] = sha(receipt)
        assert sha(original) == row["source_sha256"]
        assert all(sha(Path(path)) == digest for path, digest in input_hashes.items())
        row["passed"] = True
        save()
        print(name, "PASS", flush=True)
    assert sha(args.binary) == summary["binary_sha256"]
    assert sha(args.clang) == summary["clang_sha256"]
    summary.update(passed=True, total=len(cases))
    save()
except Exception as error:
    summary["error"] = str(error)
    save()
    raise
