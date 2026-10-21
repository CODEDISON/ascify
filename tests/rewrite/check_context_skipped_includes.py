#!/usr/bin/env python3
"""Real frontend proof for a direct provider and physically skipped includes."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / "tests/rewrite/fixtures/nvidia_samples/Common"


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def main():
    binary = os.environ["ASCIFY_BINARY"]
    cuda = os.environ["ASCIFY_CUDA_PATH"]
    resource = os.environ["ASCIFY_CLANG_RESOURCE_DIRECTORY"]
    frozen_before = inventory(HELPERS)
    with tempfile.TemporaryDirectory(prefix="ascify-context-skip-") as temp:
        work = Path(temp)
        inputs = work / "inputs"
        inputs.mkdir()
        root = inputs / "root.cu"
        provider = inputs / "provider.h"
        child = inputs / "consumer.cuh"
        provider_text = ('#ifndef LOCAL_PROVIDER_GUARD\n#define LOCAL_PROVIDER_GUARD\n'
                         '#include <helper_cuda.h>\n#define PROVIDER_BIAS 3\n#endif\n')
        child_text = ('#include "provider.h"\n'
                      'inline void check_launch() { getLastCudaError("context"); }\n')
        root_text = ('#include "provider.h"\n#include "consumer.cuh"\n'
                     'int main() { check_launch(); return 0; }\n')

        def restore():
            provider.write_text(provider_text)
            child.write_text(child_text)
            root.write_text(root_text)
            for path in inputs.iterdir():
                if path not in (provider, child, root):
                    path.unlink()

        def convert(destination, recursive=True, standard=True):
            args = [binary, str(root), f"--cuda-path={cuda}",
                    f"--clang-resource-directory={resource}",
                    "-I" + str(inputs), "-I" + str(HELPERS), "-o", str(destination)]
            if recursive:
                args.append("--local-headers-recursive")
            if standard:
                args.append("--default-preprocessor")
            args += ["--", "-std=c++17", "-I" + str(inputs), "-I" + str(HELPERS)]
            before = inventory(inputs)
            result = subprocess.run(args, text=True, capture_output=True)
            assert inventory(inputs) == before, "converter modified original inputs"
            assert not list(work.glob(".ascify-local-closure*")), "staging leaked"
            return result

        restore()
        result = convert(work / "retained.cce", recursive=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "included transitively" in result.stderr, result.stderr
        assert '"provider.h"' in (work / "retained.cce").read_text()

        output = work / "joint.cce"
        bundle = Path(str(output) + ".headers")

        def success(label, expected_skips):
            result = convert(output)
            assert result.returncode == 0, label + ": " + result.stdout + result.stderr
            text = output.read_text()
            assert "#include <helper_cuda.h>" not in text, text
            assert '#include "provider.h"' not in text, text
            assert "ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR" in text, text
            proof = (bundle / ".ascify-local-closure").read_text()
            assert "context_proof=expanded-tokens-and-final-macros-v1" in proof, proof
            skipped = [row for row in proof.splitlines()
                       if row.startswith(f"context_skipped_edge={inputs}/")]
            assert len(skipped) == expected_skips, (label, skipped)
            assert all(row.split("\t")[1] == str(provider) for row in skipped), skipped
            for path in inputs.iterdir():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                assert f"context_input={path}\t{digest}\n" in proof, (path, proof)
            assert f"context_inline={provider}\t" in proof, proof
            assert f"context_inline={child}\t" in proof, proof
            print(label + ": original tokens/macros and skipped physical edges proven")

        success("direct_provider_one_consumer", 1)
        root.write_text(root_text.replace('int main()',
            '#include "second.cuh"\nint main()'))
        (inputs / "second.cuh").write_text(
            '#include "provider.h"\ninline void check_second() { getLastCudaError("second"); }\n')
        success("direct_provider_two_consumers", 2)
        restore()
        child.write_text('#line 70 "logical-consumer.cuh"\n' + child_text +
                         'const char *logical_file = __FILE__;\n'
                         'constexpr int logical_line = __LINE__;\n')
        success("preserved_logical_locations", 1)
        published = {"root": hashlib.sha256(output.read_bytes()).hexdigest(),
                     "bundle": inventory(bundle)}

        def rejection(label, root_value=root_text, child_value=child_text,
                      provider_value=provider_text, extra=None):
            restore()
            root.write_text(root_value)
            child.write_text(child_value)
            provider.write_text(provider_value)
            for name, value in (extra or {}).items():
                (inputs / name).write_text(value)
            result = convert(output)
            assert result.returncode > 0, label + ": unsafe input was accepted"
            assert hashlib.sha256(output.read_bytes()).hexdigest() == published["root"], label
            assert inventory(bundle) == published["bundle"], label
            fresh = work / (label + ".cce")
            result = convert(fresh)
            assert result.returncode > 0, label
            assert not fresh.exists() and not Path(str(fresh) + ".headers").exists(), label
            print(label + ": refused; no fresh output and previous bundle unchanged")

        rejection("duplicate_root", root_text.replace('#include "provider.h"',
                  '#include "provider.h"\n#include "provider.h"'))
        rejection("nested_first_entry", root_text.replace(
                  '#include "provider.h"\n#include "consumer.cuh"',
                  '#include "consumer.cuh"\n#include "provider.h"'))
        rejection("real_nested_entry", child_value='#include "new.h"\n' + child_text,
                  extra={"new.h": "#define NEW_HEADER_VALUE 4\n"})
        rejection("guard_reentry", child_value='#undef LOCAL_PROVIDER_GUARD\n' + child_text)
        rejection("macro_include", child_value=child_text.replace(
                  '#include "provider.h"', '#define PROVIDER_FILE "provider.h"\n#include PROVIDER_FILE'))
        rejection("angle_include", child_value=child_text.replace('"provider.h"', '<provider.h>'))
        rejection("digraph_include", child_value=child_text.replace('#include', '%:include'))
        rejection("continued_include", child_value=child_text.replace(
                  '#include "provider.h"', '#include \\\n"provider.h"'))
        rejection("unentered_predefined_guard", root_value=
                  '#include <helper_cuda.h>\n#define LOCAL_PROVIDER_GUARD\n' + root_text)
        rejection("active_pack", child_value='#pragma pack(push, 1)\n' + child_text +
                  '#pragma pack(pop)\n')
        rejection("inactive_pack", child_value='#if __INCLUDE_LEVEL__ == 0\n'
                  '#pragma pack(1)\n#endif\n' + child_text)
        rejection("inactive_digraph_pack", child_value='#if __INCLUDE_LEVEL__ == 0\n'
                  '%:pragma pack(1)\n#endif\n' + child_text)
        rejection("inactive_opaque_pragma", child_value='#if __INCLUDE_LEVEL__ == 0\n'
                  '_Pragma("pack(1)")\n#endif\n' + child_text)
        rejection("include_level", child_value=child_text +
                  'constexpr int child_level = __INCLUDE_LEVEL__;\n')
        rejection("helper_observer", root_value=root_text +
                  '#ifdef getLastCudaError\nconstexpr int helper_visible = 1;\n#endif\n')
        restore()
        result = convert(work / "nonstandard.cce", standard=False)
        assert result.returncode > 0, result.stdout + result.stderr
        assert not (work / "nonstandard.cce").exists()
        assert not Path(str(work / "nonstandard.cce") + ".headers").exists()
        print("3 positive contexts, 15 atomic refusal pairs and 2 mode boundaries passed")
    assert inventory(HELPERS) == frozen_before, "frozen provider fixtures changed"


if __name__ == "__main__":
    main()
