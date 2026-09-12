#!/usr/bin/env python3
"""Exercise the real converter's scratch proof and publication failure paths."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cuda-path", required=True)
    parser.add_argument("--resource-dir", required=True)
    args = parser.parse_args()
    original = Path(__file__).with_name("uniform_block_reduction_input.cu").read_text()
    assignment = "  value = forward_sum(value, tile);"
    variants = {
        "positive": (original, True),
        "digraph_braces": (original.replace("{", "<%").replace("}", "%>"), True),
        "early_return": (original.replace(assignment,
                           "  if (block.thread_rank() == 0) return;\n" + assignment, 1), False),
        "divergent_call": (original.replace(assignment,
                             "  if (tile.meta_group_rank() == 0)\n" + assignment, 1), False),
        "alias": (original.replace("  T value =", "  auto copied_tile = tile;\n  T value ="), False),
        "scratch_escape": (original.replace("  T value =", "  auto* escaped = &scratch;\n  T value ="), False),
        "not_shared": (original.replace("__shared__ ", ""), False),
        "extra_storage": (original.replace("  auto block =", "  __shared__ cg::block_tile_memory<BlockSize> second;\n  auto block ="), False),
        "observed_type": (original.replace("  T value =", "  using observed = decltype(tile);\n  T value ="), False),
        "storage_sizeof": (original.replace("  T value =", "  int observed = sizeof(cg::block_tile_memory<BlockSize>);\n  T value ="), False),
        "altered_forwarder": (original.replace("  return cg::reduce", "  value += T(1);\n  return cg::reduce"), False),
        "forwarder_attribute": (original.replace("__device__ T forward_sum", "__device__ __attribute__((noinline)) T forward_sum"), False),
        "forwarder_enable_if": (original.replace("Group& group) {", "Group& group) __attribute__((enable_if(sizeof(Group) > 0, \"nonempty\"))) {"), False),
        "overloaded_forwarder": (original.replace("template <typename T, unsigned BlockSize", "template <typename T>\n__device__ T forward_sum(T value, cg::thread_block_tile<64>& group);\n\ntemplate <typename T, unsigned BlockSize"), False),
        "explicit_specialization": (original.replace("template <typename T, unsigned BlockSize", "template<> __device__ float forward_sum<float, cg::thread_block_tile<64, cg::thread_block>>(float value, cg::thread_block_tile<64, cg::thread_block>&) { return value + 1; }\n\ntemplate <typename T, unsigned BlockSize"), False),
        "adl_forwarder": (original.replace("namespace cg = cooperative_groups;", "namespace cg = cooperative_groups;\nnamespace cooperative_groups { template<class T, unsigned N, class P> __device__ T forward_sum(T value, thread_block_tile<N, P>&) { return value + T(1); } }"), False),
        "adl_inline_namespace": (original.replace("namespace cg = cooperative_groups;", "namespace cg = cooperative_groups;\nnamespace cooperative_groups { inline namespace custom { template<class T, unsigned N, class P> __device__ T forward_sum(T value, thread_block_tile<N, P>&) { return value + T(1); } } }"), False),
        "using_directive_overload": (original.replace("namespace cg = cooperative_groups;", "namespace cg = cooperative_groups;\nnamespace vendor { template<class T, unsigned N, class P> __device__ T forward_sum(T value, cg::thread_block_tile<N, P>&) { return value + T(1); } }\nusing namespace vendor;"), False),
        "no_instantiation": ("\n".join(line for line in original.splitlines()
                                         if not line.startswith("template __global__")) + "\n", False),
        "inactive_storage": (original + "\n#if 0\ncg::block_tile_memory<64> unproved;\n#endif\n", False),
        "kernel_directive": (original.replace(assignment, "#if 1\n" + assignment + "\n#endif", 1), False),
        "macro_collective": (original.replace("namespace cg = cooperative_groups;", "namespace cg = cooperative_groups;\n#define COLLECTIVE(v, t) forward_sum(v, t)")
                              .replace(assignment, "  value = COLLECTIVE(value, tile);", 1), False),
        "helper_asm": (original.replace("namespace cg = cooperative_groups;", "namespace cg = cooperative_groups;\n__device__ void unsafe_helper() { asm(\"\"); }")
                       .replace("  T value =", "  unsafe_helper();\n  T value ="), False),
        "loop_before_collective": (original.replace("  T value =", "  while (block.thread_rank() == 0) {}\n  T value ="), False),
        "constexpr_future_exit": (original.replace("  T value =", "  if constexpr (GroupSize == 512) { if (block.thread_rank() == 0) return; }\n  T value ="), False),
        "class_lifetime": (original.replace("namespace cg = cooperative_groups;", "namespace cg = cooperative_groups;\nstruct Lifetime { __device__ Lifetime() {} __device__ ~Lifetime() {} };")
                           .replace("  T value =", "  Lifetime object;\n  T value ="), False),
        "extra_template_argument": (original.replace("unsigned GroupSize>", "unsigned GroupSize, typename Hidden = int>"), False),
    }
    # Renaming identifiers must leave the cooperative API's spelling intact.
    variants["renamed"] = (original.replace("uniform_sum", "unrelated_kernel_name")
                           .replace("forward_sum", "sum_bridge")
                           .replace("scratch;", "owned_buffer;")
                           .replace("(scratch)", "(owned_buffer)")
                           .replace("auto tile =", "auto portion =")
                           .replace(", tile)", ", portion)"), True)
    with tempfile.TemporaryDirectory(prefix="ascify-uniform-proof-") as temporary:
        root = Path(temporary)
        for name, (source, accepted) in variants.items():
            path, output = root / f"{name}.cu", root / f"{name}.cce"
            path.write_text(source)
            command = [args.binary, str(path), "--frontend-compat=ascify-admitted-v1",
                       "--target-policy=dav-c310-vec", "--simt-math=fast", "--default-preprocessor",
                       f"--cuda-path={args.cuda_path}",
                       f"--clang-resource-directory={args.resource_dir}", "-o", str(output),
                       "--", "-x", "cuda", "-std=c++17", "-fgpu-rdc"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            if accepted:
                if result.returncode or not output.is_file():
                    raise AssertionError(f"{name}: positive conversion failed\n{result.stderr}")
                emitted = output.read_text()
                assert "ascify_cg" in emitted and "block_tile_memory" in emitted
                assert emitted.count("static_assert(::ascify_cg::uniform_scalar_domain<T>::value") == 1, emitted
                assert "instantiations=6" in result.stderr, result.stderr
            else:
                if result.returncode == 0 or output.exists():
                    raise AssertionError(f"{name}: unsafe source was published\n{result.stderr}")
                if "Ascify uniform block reduction:" not in result.stderr:
                    raise AssertionError(f"{name}: rejection did not reach proof gate\n{result.stderr}")
                # Failure must also preserve a previous complete output.
                sentinel = b"existing complete output\n"
                output.write_bytes(sentinel)
                repeated = subprocess.run(command, capture_output=True, text=True, timeout=90)
                assert repeated.returncode != 0 and output.read_bytes() == sentinel, name
            print(f"uniform proof {name}: {'accepted' if accepted else 'rejected without publication'}")
        path, output = root / "standard_required.cu", root / "standard_required.cce"
        path.write_text(original)
        command = [args.binary, str(path), "--frontend-compat=ascify-admitted-v1",
                   "--target-policy=dav-c310-vec", "--simt-math=fast",
                   f"--cuda-path={args.cuda_path}", f"--clang-resource-directory={args.resource_dir}",
                   "-o", str(output), "--", "-x", "cuda", "-std=c++17", "-fgpu-rdc"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert result.returncode and not output.exists() and "requires --default-preprocessor" in result.stderr, result.stderr
        print("uniform proof retained-preprocessor: rejected without publication")
        command.insert(3, "--default-preprocessor")
        command[-2] = "-std=c++20"
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert result.returncode and not output.exists() and "scratch proof requires C++17" in result.stderr, result.stderr
        print("uniform proof C++20: rejected without publication")
        command[-2] = "-std=c++17"
        injected = root / "preincluded_target_namespace.h"
        injected.write_text("namespace ascify_cg { template<class Group> __attribute__((device)) float forward_sum(float value, Group&) { return value + 1.0f; } }\n")
        command += ["-include", str(injected)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert result.returncode and not output.exists() and "Ascify uniform block reduction:" in result.stderr, result.stderr
        sentinel = b"existing complete output\n"
        output.write_bytes(sentinel)
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert result.returncode and output.read_bytes() == sentinel, result.stderr
        print("uniform proof preincluded-target-namespace: rejected without publication")


if __name__ == "__main__":
    main()
