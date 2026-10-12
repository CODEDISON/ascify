#!/usr/bin/env python3
"""Check source-valid CUDA int-parameter intrinsic conversions and boundaries."""
import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


PREAMBLE = """#include <cuda_runtime.h>
static_assert(sizeof(int) == 4, "CUDA int width");
static_assert(sizeof(long) == 8, "LP64 fixture");
static_assert(sizeof(long long) == 8, "CUDA long long width");
"""
REGION = re.compile(r"/\*ffs_begin:(\w+)\*/(.*?)/\*ffs_end:\1\*/", re.S)


@dataclass
class Case:
    name: str
    code: str
    expected: dict
    headers: dict = field(default_factory=dict)


def mark(name, expression):
    return "/*ffs_begin:" + name + "*/" + expression + "/*ffs_end:" + name + "*/"


def device_case(name, expression, expected=None, parameters="unsigned x", prefix=""):
    return Case(name, prefix + "\n__device__ int sample(" + parameters + ") {\n"
                "  return " + mark("value", expression) + ";\n}\n",
                {"value": expression if expected is None else expected})


def ffs_cases():
    # Preserve the original Sobol expression and its two already-int calls.
    yield Case("sobol_unsigned32", """__device__ int sample(unsigned i, int stride,
                                       unsigned v_stridemask) {
  int first = """ + mark("first", "__ffs(stride)") + ";\n  int second = " +
               mark("second", "__ffs(stride)") + ";\n  return first + second + " +
               mark("value", "__ffs(~((i - stride) | v_stridemask))") + ";\n}\n",
               {"first": "__ffs(stride)", "second": "__ffs(stride)",
                "value": "__ffs(static_cast<int>(~((i - stride) | v_stridemask)))"})

    # CUDA narrows all of these to int. Target long/long-long overloads must
    # not silently select a different width, including on LP64 platforms.
    widths = (("unsigned64", "ull"), ("signed64", "ll"),
              ("unsigned_long", "ul"), ("signed_long", "sl"))
    yield Case("wide_integer_inputs", "__global__ void sample(int *out, "
               "unsigned long long ull, long long ll, unsigned long ul, long sl) {\n" +
               "".join("  out[" + str(index) + "] = " + mark(name, "__ffs(" + var + ")") +
                       ";\n" for index, (name, var) in enumerate(widths)) + "}\n",
               {name: "__ffs(static_cast<int>(" + var + "))" for name, var in widths})

    # This is a frontend spelling check, not a claim of device execution.
    literals = (("zero", "0u"), ("bit31", "0x80000000u"),
                ("truncated_bit32", "0x100000000ULL"),
                ("all_bits64", "0xffffffffffffffffULL"))
    yield Case("zero_and_high_bits", "__global__ void sample(int *out) {\n" +
               "".join("  out[" + str(index) + "] = " + mark(name, "__ffs(" + value + ")") +
                       ";\n" for index, (name, value) in enumerate(literals)) + "}\n",
               {name: "__ffs(static_cast<int>(" + value + "))" for name, value in literals})
    yield device_case("postincrement_once", "__ffs(x++)", "__ffs(static_cast<int>(x++))")
    yield device_case("assignment_once", "__ffs((x += 1u))",
                      "__ffs(static_cast<int>((x += 1u)))")
    # The operator token is macro-expanded, but both boundaries of the
    # BinaryOperator and its operands are ordinary source locations. Preserve
    # that token's original spelling; this is not a macro-expanded callee or
    # argument-expression node boundary (the retained cases below cover those).
    yield device_case("operator_token_macro", "__ffs(x PROJECT_BIT_OR 1u)",
                      "__ffs(static_cast<int>(x PROJECT_BIT_OR 1u))",
                      prefix="#define PROJECT_BIT_OR |\n")
    yield device_case("small_integer_promotion", "__ffs(x)", "__ffs(static_cast<int>(x))",
                      parameters="short x")

    yield device_case("already_int", "__ffs(x)", parameters="int x")
    yield device_case("explicit_cast", "__ffs(static_cast<int>(x))")
    yield device_case("other_intrinsic", "__ffsll(x)", parameters="unsigned long long x")
    yield device_case("floating_conversion", "__ffs(x)", parameters="float x")
    yield device_case("project_global_overload", "__ffs(x)",
                      prefix="__device__ int __ffs(unsigned value) { return value != 0; }\n")
    yield device_case("project_namespace", "project::__ffs(x)",
                      prefix="namespace project { __device__ int __ffs(int value) "
                             "{ return value != 0; } }\n")
    # A legal redeclaration taints the SDK declaration chain. A system-header
    # pragma on a project file must not change its physical ownership.
    yield device_case("project_redeclaration", "__ffs(x)",
                      prefix="__device__ int __ffs(int);\n")
    system_redecl = device_case("project_system_redeclaration", "__ffs(x)",
                               prefix='#include "project.h"\n')
    system_redecl.headers = {"project.h": "#pragma clang system_header\n"
                            "__device__ int __ffs(int);\n"}
    yield system_redecl
    yield device_case("macro_call", "PROJECT_FFS(x)",
                      prefix="#define PROJECT_FFS(value) __ffs(value)\n")
    yield device_case("macro_argument", "__ffs(PROJECT_VALUE)",
                      prefix="#define PROJECT_VALUE (x + 1u)\n")
    yield device_case("nested_call_argument", "__ffs(projectValue(x))",
                      prefix="__device__ unsigned projectValue(unsigned value) "
                             "{ return value + 1u; }\n")
    yield Case("indirect_call", "__device__ int sample(unsigned x) {\n"
               "  auto intrinsic = __ffs;\n  return " + mark("value", "intrinsic(x)") +
               ";\n}\n", {"value": "intrinsic(x)"})
    yield Case("dependent_template", "template<class T> __device__ int sample(T x) {\n"
               "  return " + mark("value", "__ffs(x)") + ";\n}\n",
               {"value": "__ffs(x)"})
    yield Case("nondependent_template", "template<class T> __device__ int sample(unsigned x) {\n"
               "  return " + mark("value", "__ffs(x)") + ";\n}\n",
               {"value": "__ffs(x)"})
    # Calling the device-only SDK function from host or host/device code is
    # invalid CUDA. Exercise the legal host overload boundary instead.
    yield Case("host_project_overload", "int __ffs(unsigned value) { return value != 0; }\n"
               "int sample(unsigned x) { return " + mark("value", "__ffs(x)") + "; }\n",
               {"value": "__ffs(x)"})


def cases(intrinsic="ffs"):
    if intrinsic == "ffs":
        yield from ffs_cases()
        return
    if intrinsic != "clz":
        raise ValueError("unsupported intrinsic: " + intrinsic)

    def clz(text):
        return text.replace("__ffs", "__clz").replace("PROJECT_FFS", "PROJECT_CLZ")

    # Use the unchanged mergeSort helper expression as the project case.
    # Both families then exercise the same exact declaration/source guards.
    for case in ffs_cases():
        if case.name == "sobol_unsigned32":
            yield Case("mergesort_unsigned32", "__device__ unsigned nextPowerOfTwo(unsigned x) {\n"
                       "  return x == 0 ? 0 : 1U << (32 - " +
                       mark("value", "__clz(x - 1)") + ");\n}\n",
                       {"value": "__clz(static_cast<int>(x - 1))"})
        else:
            yield Case(case.name, clz(case.code),
                       {key: clz(value) for key, value in case.expected.items()},
                       {name: clz(text) for name, text in case.headers.items()})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regions(text):
    found = REGION.findall(text)
    result = dict(found)
    if len(result) != len(found):
        raise AssertionError("duplicate expression markers")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsic", choices=("ffs", "clz"), default="ffs")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--cuda-path", required=True, type=Path)
    parser.add_argument("--resource-dir", required=True, type=Path)
    parser.add_argument("--clang", type=Path,
                        help="matching clang++; otherwise try the binary/resource prefixes, then PATH")
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    cuda = args.cuda_path.resolve()
    resource = args.resource_dir.resolve()
    clang = args.clang.resolve() if args.clang else None
    if clang is None:
        choices = [binary.parent / "clang++", resource.parents[2] / "bin/clang++"]
        on_path = shutil.which("clang++")
        if on_path:
            choices.append(Path(on_path))
        clang = next((path.resolve() for path in choices if path.is_file()), None)
    if clang is None or not clang.is_file():
        parser.error("supply --clang with the frontend's matching clang++")
    if not binary.is_file() or not (cuda / "include/cuda_runtime.h").is_file():
        parser.error("--binary and --cuda-path must identify real installed inputs")
    if not (resource / "include/__clang_cuda_runtime_wrapper.h").is_file():
        parser.error("--resource-dir must contain the complete CUDA resource headers")
    version = subprocess.run([str(clang), "--version"], text=True, capture_output=True,
                             timeout=30, check=True).stdout
    clang_major = re.search(r"\bclang version (\d+)(?:\.|\s)", version)
    resource_major = re.match(r"(\d+)(?:\.|$)", resource.name)
    if (clang_major is None or resource_major is None or
            clang_major.group(1) != resource_major.group(1)):
        parser.error("clang major version does not match --resource-dir; "
                     "supply --clang with the frontend's matching clang++")
    with tempfile.TemporaryDirectory(prefix="ascify-" + args.intrinsic + "-conversion-") as temporary:
        work = args.evidence_dir.resolve() if args.evidence_dir else Path(temporary)
        if args.evidence_dir:
            work.mkdir(parents=True, exist_ok=False)
        summary = {"schema": "ascify." + args.intrinsic + "-conversion.v1", "passed": False,
                   "intrinsic": "__" + args.intrinsic,
                   "binary": str(binary), "binary_sha256": sha(binary),
                   "clang": str(clang), "clang_sha256": sha(clang), "clang_version": version,
                   "clang_resource_major": int(resource_major.group(1)),
                   "cuda_path": str(cuda), "resource_dir": str(resource), "cases": [],
                   "scope": "independent real CUDA syntax and generated expressions; "
                            "no target compilation, linking or device execution"}

        def save():
            (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        def run(directory, label, command):
            (directory / (label + ".argv.json")).write_text(json.dumps(command, indent=2) + "\n")
            result = subprocess.run(command, text=True, capture_output=True, timeout=60)
            (directory / (label + ".stdout.log")).write_text(result.stdout)
            (directory / (label + ".stderr.log")).write_text(result.stderr)
            return result

        save()
        try:
            for case in cases(args.intrinsic):
                directory = work / case.name
                directory.mkdir()
                source = directory / "input.cu"
                output = directory / "output.cpp"
                receipt = directory / "receipt.json"
                source.write_text(PREAMBLE + case.code)
                for name, text in case.headers.items():
                    (directory / name).write_text(text)
                input_hashes = {path.name: sha(path) for path in
                                [source, *(directory / name for name in case.headers)]}
                original_regions = regions(source.read_text())
                if original_regions.keys() != case.expected.keys():
                    raise AssertionError(case.name + ": incomplete expression expectations")
                changed = sum(original_regions[name] != expected
                              for name, expected in case.expected.items())
                row = {"case": case.name, "input_sha256": input_hashes,
                       "expected_changed_expressions": changed, "passed": False}
                summary["cases"].append(row)
                syntax = run(directory, "syntax", [
                    str(clang), str(source), "-x", "cuda", "-std=c++17", "-fsyntax-only",
                    "--cuda-host-only", "-nocudalib", "--cuda-path=" + str(cuda),
                    "-resource-dir=" + str(resource), "-Wno-unknown-cuda-version",
                ])
                row["syntax_returncode"] = syntax.returncode
                save()
                if syntax.returncode != 0:
                    raise AssertionError(case.name + ": source syntax failed: " + syntax.stderr)
                conversion = run(directory, "conversion", [
                    str(binary), str(source), "--target-policy=dav-c310-vec", "--simt-math=fast",
                    "--target-recipe=none", "--default-preprocessor", "--cuda-path=" + str(cuda),
                    "--clang-resource-directory=" + str(resource),
                    "--migration-receipt=" + str(receipt), "-o", str(output),
                    "--", "-x", "cuda", "-std=c++17",
                ])
                row["conversion_returncode"] = conversion.returncode
                save()
                if conversion.returncode != 0:
                    raise AssertionError(case.name + ": conversion failed: " + conversion.stderr)
                if json.loads(receipt.read_text())["status"] != "succeeded":
                    raise AssertionError(case.name + ": receipt was not successful")
                actual = output.read_text()
                if regions(actual) != case.expected:
                    raise AssertionError(case.name + ": expression mismatch: " +
                                         repr(regions(actual)) + " expected " + repr(case.expected))
                # Also catch a duplicate or unrelated conversion elsewhere,
                # including a macro definition or a project declaration body.
                expected_casts = source.read_text().count("static_cast<int>(") + changed
                if actual.count("static_cast<int>(") != expected_casts:
                    raise AssertionError(case.name + ": unexpected cast outside marked expressions")
                for name, digest in input_hashes.items():
                    if sha(directory / name) != digest:
                        raise AssertionError(case.name + ": source/header modified: " + name)
                row.update({"passed": True, "output_sha256": sha(output),
                            "receipt_sha256": sha(receipt)})
                save()
                print(case.name + ": source valid; " + str(changed) + " expression(s) converted",
                      flush=True)
            if sha(binary) != summary["binary_sha256"] or sha(clang) != summary["clang_sha256"]:
                raise AssertionError("compiler changed during matrix")
            summary.update({"passed": True, "total": len(summary["cases"]),
                            "conversion_cases": sum(row["expected_changed_expressions"] > 0
                                                    for row in summary["cases"]),
                            "changed_expressions": sum(row["expected_changed_expressions"]
                                                       for row in summary["cases"])})
            summary["retained_cases"] = summary["total"] - summary["conversion_cases"]
            save()
        except Exception as error:
            summary["failure"] = str(error)
            save()
            raise
        print("__" + args.intrinsic + " frontend: " + str(summary["total"]) + " cases passed")


if __name__ == "__main__":
    main()
