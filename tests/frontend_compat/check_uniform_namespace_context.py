#!/usr/bin/env python3
"""Real frontend namespace-import boundaries for the uniform scratch proof."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cuda-path", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--stdlib-debug-header", type=Path,
                        help="Optional actual debug/debug.h; requires the combined VFS configuration refusal patch")
    args = parser.parse_args()
    original = Path(__file__).with_name("uniform_block_reduction_input.cu").read_text()
    anchor = "namespace cg = cooperative_groups;"
    marker = "source using-directives can alter overload lookup"
    cases = {
        "ordinary": (original, True, "", []),
        "empty_namespace_import": (original.replace(anchor, anchor + "\nnamespace empty_provider {}\nnamespace importer { using namespace empty_provider; }"), True, "", []),
        "empty_std_namespace_reopen": (original + "\nnamespace std { namespace __debug {} }\n", True, "", []),
        "anonymous_unrelated_function": (original.replace(anchor, anchor + "\nnamespace { __device__ int unrelated_observer(int value) { return value + 1; } }"), True, "", []),
        "global_empty_import": (original.replace(anchor, anchor + "\nnamespace empty_provider {}\nusing namespace empty_provider;"), False, marker, []),
        "user_namespace_reactivated": (original.replace(anchor, anchor + "\nnamespace provider {}\nnamespace importer { using namespace provider; }\nnamespace provider { int observation; }"), False, marker, []),
        "std_namespace_reactivated": (original + "\nnamespace std { namespace __debug { int observation; } }\n", False, marker, []),
        "nested_namespace_export": (original + "\nnamespace std { namespace __debug { namespace future {} } }\n", False, marker, []),
        "inline_namespace_export": (original + "\nnamespace std { namespace __debug { inline namespace future {} } }\n", False, marker, []),
        "namespace_alias_export": (original + "\nnamespace std { namespace __debug { namespace future = ::cooperative_groups; } }\n", False, marker, []),
        "using_shadow_export": (original + "\nnamespace std { namespace __debug { using ::forward_sum; } }\n", False, marker, []),
        "using_directive_export": (original + "\nnamespace std { namespace __debug { using namespace ::cooperative_groups; } }\n", False, marker, []),
        "dormant_template_export": (original + "\nnamespace std { namespace __debug { template<class Future> struct Observer; } }\n", False, marker, []),
        "dormant_function_template_export": (original + "\nnamespace std { namespace __debug { template<class Future> __attribute__((device)) Future observe(Future); } }\n", False, marker, []),
        "nominated_namespace_alias": (original.replace(anchor, anchor + "\nnamespace empty_provider {}\nnamespace alias = empty_provider;\nnamespace importer { using namespace alias; }"), False, marker, []),
        "nominated_inline_namespace": (original.replace(anchor, anchor + "\nnamespace provider { inline namespace empty {} }\nnamespace importer { using namespace provider::empty; }"), False, marker, []),
        "macro_namespace_import": (original.replace(anchor, anchor + "\nnamespace empty_provider {}\n#define EMPTY_IMPORT using namespace empty_provider;\nnamespace importer { EMPTY_IMPORT }"), False, marker, []),
        "adl_namespace_reopen": (original.replace(anchor, anchor + "\nnamespace cooperative_groups { template<class T, unsigned N, class P> __device__ T forward_sum(T value, thread_block_tile<N, P>&) { return value + T(1); } }"), False, "source declarations in cooperative_groups", []),
        "future_double_instantiation": (original + "\ntemplate __global__ void uniform_sum<double, 128, 64>(const double*, double*);\n", False, "Ascify cooperative plus supports only int and float", []),
        "future_long_instantiation": (original + "\ntemplate __global__ void uniform_sum<long, 128, 64>(const long*, long*);\n", False, "Ascify cooperative plus supports only int and float", []),
        "future_ascify_namespace_import": (original.replace(anchor, anchor + "\nnamespace ascify {}\nnamespace importer { using namespace ascify; }"), False, "input owns the top-level name 'ascify'", []),
        "future_facade_namespace_import": (original.replace(anchor, anchor + "\nnamespace ascify_cg {}\nnamespace importer { using namespace ascify_cg; }"), False, "source declarations in cooperative_groups", []),
        "anonymous_using_overload": (original.replace(anchor, anchor + "\nnamespace { template<class T, unsigned N, class P> __device__ T forward_sum(T value, cg::thread_block_tile<N, P>&) { return value + T(1); } }"), False, "collective requires a pure unique forwarder", []),
        "anonymous_using_shadow": (original.replace(anchor, anchor + "\nnamespace observer { template<class T, unsigned N, class P> __device__ T forward_sum(T value, cg::thread_block_tile<N, P>&) { return value + T(1); } }\nnamespace { using observer::forward_sum; }"), False, "collective requires a pure unique forwarder", []),
        "anonymous_inline_export": (original.replace(anchor, anchor + "\nnamespace { inline namespace observer { template<class T, unsigned N, class P> __device__ T forward_sum(T value, cg::thread_block_tile<N, P>&) { return value + T(1); } } }"), False, "collective requires a pure unique forwarder", []),
        "anonymous_dormant_template": (original.replace(anchor, anchor + "\nnamespace { template<class Future> __device__ Future forward_sum(Future); }"), False, "collective requires a pure unique forwarder", []),
        # The existing six CUDA SDK instantiations do not select this overload;
        # a future admitted int/512/256 instantiation does. Its name must already
        # be excluded before publishing the dependent kernel template.
        "anonymous_future_allowed_instantiation": (original.replace(anchor, anchor + "\nnamespace { template<class T, unsigned N, class P, typename std::enable_if<std::is_same<T,int>::value && N == 256, int>::type = 0> __device__ T forward_sum(T value, cg::thread_block_tile<N,P>&) { return value + T(1); } }"), False, "collective requires a pure unique forwarder", []),
        "anonymous_nested_export": (original.replace(anchor, anchor + "\nnamespace { namespace { template<class T, unsigned N, class P> __device__ T forward_sum(T value, cg::thread_block_tile<N, P>&) { return value + T(1); } } }"), False, "collective requires a pure unique forwarder", []),
        "anonymous_late_reactivation": (original.replace(anchor, anchor + "\nnamespace { __device__ int unrelated_observer(int value) { return value; } }") + "\nnamespace { template<class T, unsigned N, class P> __device__ T forward_sum(T value, cg::thread_block_tile<N, P>&) { return value + T(1); } }\n", False, "collective requires a pure unique forwarder", []),
    }
    temporary = None
    if args.evidence_dir:
        require(not args.evidence_dir.exists(), "Use a new evidence directory")
        root = args.evidence_dir
        root.mkdir(parents=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="ascify-uniform-namespace-")
        root = Path(temporary.name)
    try:
        headers = root / "observers"
        headers.mkdir()
        observer = headers / "observer.h"
        observer.write_text("namespace std { namespace __debug { template<class Future> struct Observer; } }\n")
        cases["isystem_observer"] = (original.replace(anchor, "#include <observer.h>\n" + anchor), False, marker, ["-isystem", str(headers)])
        using_header = headers / "using_observer.h"
        using_header.write_text("namespace cooperative_groups { template<class T, unsigned N, class P> __attribute__((device)) T forward_sum(T value, thread_block_tile<N, P>&) { return value + T(1); } }\n")
        cases["isystem_adl_observer"] = (original.replace(anchor, anchor + "\n#include <using_observer.h>"), False,
                                          "source declarations in cooperative_groups", ["-isystem", str(headers)])
        if args.stdlib_debug_header:
            header = args.stdlib_debug_header.resolve()
            require(header.is_file(), "Missing actual stdlib debug header")
            replacement = headers / "debug_reactivated.h"
            replacement.write_bytes(header.read_bytes() + b"\nnamespace std { namespace __debug { template<class Future> struct Observer; } }\n")
            overlay = root / "observer_vfs.json"
            overlay.write_text(json.dumps({"version": 0, "use-external-names": False,
                "roots": [{"type": "file", "name": str(header), "external-contents": str(replacement)}]}))
            # The current LibTooling entry does not load overlays. A combined
            # entry patch must reject the configuration before AST traversal;
            # this case therefore does not claim to observe namespace reopening.
            cases["vfs_configuration_rejected"] = (original, False,
                "Ascify cannot translate input selected by VFS overlay arguments", ["-ivfsoverlay", str(overlay)])
        rows = []
        for name, (text, accepted, reason, extra) in cases.items():
            dest = root / name
            dest.mkdir()
            source, output = dest / "input.cu", dest / "output.cce"
            source.write_text(text)
            before = {str(path): sha(path) for path in [source, *headers.iterdir()]}
            command = [args.binary, str(source), "--frontend-compat=ascify-admitted-v1",
                       "--target-policy=dav-c310-vec", "--simt-math=fast", "--default-preprocessor",
                       "--cuda-path=" + args.cuda_path, "--clang-resource-directory=" + args.resource_dir,
                       "-o", str(output), "--", "-x", "cuda", "-std=c++17", "-fgpu-rdc", *extra]
            (dest / "argv.json").write_text(json.dumps(command, indent=2) + "\n")
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            (dest / "stdout.log").write_text(result.stdout)
            (dest / "stderr.log").write_text(result.stderr)
            require({str(path): sha(path) for path in [source, *headers.iterdir()]} == before, name + ": input bytes changed")
            if accepted:
                require(result.returncode == 0 and output.is_file(), name + ": positive failed\n" + result.stderr)
                require(output.read_text().count("static_assert(::ascify_cg::uniform_scalar_domain<T>::value") == 1,
                        name + ": future scalar guard missing/duplicated")
                require("instantiations=6" in result.stderr, name + ": concrete instantiation proof missing")
            else:
                require(result.returncode != 0 and not output.exists(), name + ": unsafe output was published")
                require(reason in result.stderr, name + ": did not reach intended proof gate\n" + result.stderr)
                sentinel = b"previous complete output\n"
                output.write_bytes(sentinel)
                repeated = subprocess.run(command, capture_output=True, text=True, timeout=90)
                (dest / "repeat.stdout.log").write_text(repeated.stdout)
                (dest / "repeat.stderr.log").write_text(repeated.stderr)
                require(repeated.returncode != 0 and output.read_bytes() == sentinel,
                        name + ": refusal changed prior output")
            rows.append({"case": name, "accepted": accepted, "returncode": result.returncode,
                         "source_sha256": sha(source), "stderr_sha256": sha(dest / "stderr.log"),
                         "guard_checked": accepted, "atomic_refusal_checked": not accepted})
            print("uniform namespace " + name + ": " + ("accepted" if accepted else "rejected without publication"), flush=True)
        (root / "summary.json").write_text(json.dumps({"complete": True, "positive": sum(r["accepted"] for r in rows),
            "negative": sum(not r["accepted"] for r in rows), "vfs_configuration_refusal_checked": bool(args.stdlib_debug_header),
            "scope": "Actual frontend matrix; no target/device execution claim", "cases": rows}, indent=2) + "\n")
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
