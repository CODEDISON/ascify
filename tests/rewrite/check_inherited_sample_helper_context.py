#!/usr/bin/env python3
"""Exercise contextual helper proof and publication through the real frontend."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/rewrite/fixtures/inherited_sample_helper"
HELPERS = ROOT / "tests/rewrite/fixtures/nvidia_samples/Common"


def identity(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    binary = os.environ["ASCIFY_BINARY"]
    cuda = os.environ["ASCIFY_CUDA_PATH"]
    resource = os.environ["ASCIFY_CLANG_RESOURCE_DIRECTORY"]
    fixture_before = identity(FIXTURE.iterdir())
    with tempfile.TemporaryDirectory(prefix="ascify-inherited-helper-") as temp:
        work = Path(temp)
        inputs = work / "input"
        shutil.copytree(FIXTURE, inputs)
        root = inputs / "root.cu"
        header = inputs / "launch_check.cuh"
        original_root, original_header = root.read_text(), header.read_text()

        def convert(output, recursive=True, standard=True):
            args = [binary, str(root), f"--cuda-path={cuda}",
                    f"--clang-resource-directory={resource}", "-o", str(output),
                    "-I" + str(HELPERS)]
            if recursive:
                args.append("--local-headers-recursive")
            if standard:
                args.append("--default-preprocessor")
            args += ["--", "-std=c++17", "-I" + str(HELPERS)]
            before = identity(inputs.iterdir())
            result = subprocess.run(args, text=True, capture_output=True)
            assert identity(inputs.iterdir()) == before, "frontend changed input bytes"
            assert not list(work.glob(".ascify-local-closure*")), "staging leaked"
            return result

        root_only = work / "root-only.cce"
        result = convert(root_only, False)
        assert result.returncode == 0, result.stderr
        assert "#include <helper_cuda.h>" in root_only.read_text()
        assert "helper macro expansion outside the main file" in result.stderr
        print("original parent TU parsed; nonrecursive helper transaction retained")

        output = work / "recursive.cce"
        bundle = Path(str(output) + ".headers")
        marker = bundle / ".ascify-local-closure"

        # The default converter visits excluded conditional blocks, which is
        # unsuitable for proving actual C++ token/counter equivalence.
        result = convert(output, standard=False)
        assert result.returncode != 0 and "require standard preprocessing" in result.stderr, result.stderr
        assert not output.exists() and not bundle.exists()
        print("retained conditional mode rejected without publication")

        def successful():
            result = convert(output)
            assert result.returncode == 0, result.stdout + result.stderr
            text = output.read_text()
            assert "#include <helper_cuda.h>" not in text, text
            assert '#include "launch_check.cuh"' not in text, text
            assert "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR" in text, text
            proof = marker.read_text()
            assert "context_proof=expanded-tokens-and-final-macros-v1" in proof, proof
            assert "context_standard_preprocessing=1" in proof, proof
            for path in (root, header):
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                assert f"context_input={path}\t{sha}\n" in proof, proof
                assert f"context_path_binding={path}\t{path.resolve()}\n" in proof, proof
            assert f"context_inline={header}\t" in proof, proof
            assert f"context_original_edge={root}\t{header}\t" in proof, proof
            print("joint helper conversion published with original input and include-graph evidence")

        successful()
        # Actual builtins must agree, including deferred macro expansion after
        # return to the parent file. #line also proves a non-default parent line.
        root.write_text('#line 200 "parent-logical.cu"\n' + original_root +
                        '\nconstexpr int parent_line = __LINE__;\n'
                        'constexpr int parent_counter = __COUNTER__;\n'
                        'const char *parent_file = DEFERRED_FILE;\n')
        header.write_text('#ifndef CONTEXT_LEAF\n#define CONTEXT_LEAF\n'
                          '#define DEFERRED_FILE __FILE__\n'
                          'const char *child_file = __FILE__;\n'
                          'constexpr int child_line = __LINE__;\n'
                          'constexpr int child_counter = __COUNTER__;\n' +
                          original_header + '#endif\n')
        successful()
        published = identity([output, marker])
        result = convert(output, standard=False)
        assert result.returncode != 0 and "require standard preprocessing" in result.stderr, result.stderr
        assert identity([output, marker]) == published

        def rejected(label, child_text, root_text=original_root, reason="contextual"):
            root.write_text(root_text)
            header.write_text(child_text)
            result = convert(output)
            assert result.returncode != 0, f"{label}: unsafe context accepted"
            assert reason in result.stderr, f"{label}: {result.stderr}"
            assert identity([output, marker]) == published, f"{label}: previous bundle changed"
            fresh = work / (label + ".cce")
            result = convert(fresh)
            assert result.returncode != 0 and not fresh.exists(), label
            assert not Path(str(fresh) + ".headers").exists(), label
            print(f"{label}: rejected without publication; prior root/bundle unchanged")

        guarded = '#ifndef CONTEXT_LEAF\n#define CONTEXT_LEAF\n' + original_header + '#endif\n'
        rejected("multiple_include", guarded,
                 original_root.replace('#include "launch_check.cuh"',
                                       '#include "launch_check.cuh"\n#include "launch_check.cuh"'),
                 "one direct include occurrence")
        rejected("include_level_token", 'constexpr int child_level = __INCLUDE_LEVEL__;\n' +
                 original_header, reason="expanded tokens or final macro state differ")
        rejected("final_macro_state", '#if __INCLUDE_LEVEL__ == 1\n#define FINAL_STATE 1\n'
                 '#else\n#define FINAL_STATE 2\n#endif\n' + original_header,
                 reason="expanded tokens or final macro state differ")
        rejected("counter_state", '#if __INCLUDE_LEVEL__ == 1\n#if __COUNTER__ == 0\n'
                 '#endif\n#endif\n' + original_header,
                 reason="expanded tokens or final macro state differ")
        rejected("pragma", '#pragma pack(push, 1)\n' + original_header +
                 '#pragma pack(pop)\n', reason="leaf without pragmas")
        rejected("macro_pragma", 'PARENT_PRAGMA\n' + original_header,
                 '#define PARENT_PRAGMA _Pragma("pack(1)")\n' + original_root)
        (inputs / "definitions.h").write_text("constexpr int leaf_value = 7;\n")
        rejected("nested_include", '#include "definitions.h"\n' + original_header,
                 reason="leaf without pragmas")
        rejected("joint_proof", original_header,
                 original_root + '\n#ifdef getLastCudaError\nconstexpr int observes_helper = 1;\n#endif\n',
                 reason="joint helper transaction")
        print("context success plus nine fail-closed boundaries and atomic retries passed")
    assert fixture_before == identity(FIXTURE.iterdir()), "fixed fixture modified"


if __name__ == "__main__":
    main()
