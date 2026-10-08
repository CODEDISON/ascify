#!/usr/bin/env python3
"""Real frontend CUDA vec3 rejection and output-preservation checks."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cuda-path", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--baseline-binary")
    parser.add_argument("--ccec")
    parser.add_argument("--target-include", action="append", default=[])
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    vector_types = Path(args.cuda_path) / "include/vector_types.h"
    if not vector_types.is_file() or sha(vector_types) != "ded087a2c3ca89fd3a5150589175aca0aa4714357016209ea3e7a51f94845e75":
        print("vec3 ABI matrix skipped: this SDK is outside the frozen CUDA13.4.1 rejection profile")
        return
    temporary = None
    if args.evidence_dir:
        require(not args.evidence_dir.exists(), "Use a new evidence directory")
        root = args.evidence_dir
        root.mkdir(parents=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="ascify-vec3-abi-")
        root = Path(temporary.name)
    env = os.environ.copy()
    for key in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "OBJC_INCLUDE_PATH"):
        env.pop(key, None)
    marker = "CUDA vec3 ABI boundary:"
    prefix = "#include <cuda_runtime.h>\n"
    cases = {
        "scalar": ("__global__ void scalar(float* p) { p[0] += 1; }", True, []),
        "dim3_scalar_ctor": ("void observer() { dim3 d(1, 2, 3); }", True, []),
        "vector2_vector4": ("__global__ void vectors(float2* f2, float4* f4, uint2* u2, uint4* u4) { f2[0] = make_float2(1,2); f4[0] = make_float4(1,2,3,4); u2[0] = make_uint2(1,2); u4[0] = make_uint4(1,2,3,4); }", True, []),
        "own_namespace": ("namespace own { struct float3 { double x,y,z; }; using uint3 = unsigned long long; }\nstatic_assert(sizeof(own::float3)==24 && sizeof(own::uint3)==8, \"own types\");\n__global__ void observer(own::float3* p) { p[0].z=3; }", True, []),
        "own_helper": ("namespace own { struct float3 { int x,y,z; }; __device__ float3 make_float3(int x,int y,int z){return {x,y,z};} }\n__global__ void observer(own::float3* p){p[0]=own::make_float3(1,2,3);}", True, []),
        "own_simd_vector": ("namespace own { using float3 = float __attribute__((ext_vector_type(3))); using uint3 = unsigned int __attribute__((ext_vector_type(3))); }\nstatic_assert(sizeof(own::float3)==16 && alignof(own::float3)==16, \"SIMD layout\");\n__global__ void observer(own::float3* p,own::uint3* u){p[0]={1,2,3};u[0]={1,2,3};}", True, []),
        "own_global_without_sdk": ("struct float3 { double x,y,z; }; using uint3 = double;\nstatic_assert(sizeof(float3)==24 && sizeof(uint3)==8, \"own global types\");\nvoid observer(float3* p){p->z=3;}", True, ["-nocudainc", "-nocudalib"]),
        "local_float3": ("__global__ void observer(){float3 f={1,2,3};}", False, []),
        "local_uint3": ("__global__ void observer(){uint3 u={1,2,3};}", False, []),
        "qualified_elaborated": ("__global__ void observer(){struct ::float3 f={1,2,3};}", False, []),
        "array": ("__global__ void observer(){uint3 u[2]={{1,2,3},{4,5,6}};}", False, []),
        "pointer": ("__global__ void observer(const float3* p,float3* out){out[0]=p[0];}", False, []),
        "reference": ("__device__ float observer(const float3& f){return f.z;}", False, []),
        "unused_alias": ("using Future=float3;", False, []),
        "embedding": ("struct Storage { uint3 data; };", False, []),
        "external_by_value": ("extern \"C\" float3 external(float3);", False, []),
        "function_pointer": ("using Callback=float3(*)(uint3);", False, []),
        "helper_auto": ("__global__ void observer(){auto f=make_float3(1,2,3);}", False, []),
        "helper_address": ("auto callback=&make_uint3;", False, []),
        "dormant_template": ("template<class Future> __device__ auto observer(Future x){return make_float3(1,2,3);}", False, []),
        "generic_lambda": ("void observer(){auto fn=[](auto x){return make_float3(1,2,3);};}", False, []),
        "dormant_record_template": ("template<class Future> struct Observer { uint3 data; };", False, []),
        "template_type_argument": ("template<class T> struct Observer {}; Observer<float3> v;", False, []),
        "decltype_helper": ("using Future=decltype(make_float3(1,2,3));", False, []),
        "sizeof": ("static_assert(sizeof(float3)==12, \"CUDA layout\");", False, []),
        "forward_redecl": ("struct float3;", False, []),
        "using_shadow": ("namespace future { using ::uint3; }", False, []),
        "dim3_uint3_consumer": ("void observer(){dim3 d(make_uint3(1,2,3));}", False, []),
        "macro_type": ("#define VEC float3\n__global__ void observer(){VEC v={1,2,3};}", False, []),
        "line_spoof": ("#line 370 \"vector_types.h\"\n__global__ void observer(){float3 v={1,2,3};}", False, []),
    }
    headers = root / "headers"
    headers.mkdir()
    (headers / "vec_observer.h").write_text("struct Observer { float3 v; };\n")
    (headers / "pragma_observer.h").write_text("#pragma GCC system_header\nstruct Observer { uint3 v; };\n")
    no_sdk = headers / "no_sdk"
    no_sdk.mkdir()
    # Ascify explicitly injects cuda_runtime.h even with -nocudainc. This
    # controlled empty shim isolates user-owned global names; it grants no
    # SDK provenance or target compile claim.
    (no_sdk / "cuda_runtime.h").write_text("// Controlled no-SDK user-type fixture.\n")
    cases["own_global_without_sdk"] = (cases["own_global_without_sdk"][0], True,
        ["-nocudainc", "-nocudalib", "-I", str(no_sdk)])
    cases["isystem_header"] = ("#include <vec_observer.h>", False, ["-isystem", str(headers)])
    cases["pragma_system_header"] = ("#include <pragma_observer.h>", False, ["-I", str(headers)])
    cases["local_header_closure"] = ('#include "local_observer.h"', False, [])
    cases["first_forward_redecl"] = ("struct float3; struct uint3;\n#include <vector_types.h>\nvoid observer(){float3 v={1,2,3};}", False,
        ["-nocudainc", "-nocudalib", "-I", str(no_sdk), "-I", str(Path(args.cuda_path) / "include")])
    cases["packed_sdk_layout"] = ('#pragma pack(push,1)\n#include <vector_types.h>\nstatic_assert(alignof(float3)==1, "packed CUDA record");\nvoid observer(){float3 v={1,2,3};}\n#pragma pack(pop)', False,
        ["-nocudainc", "-nocudalib", "-I", str(no_sdk), "-I", str(Path(args.cuda_path) / "include")])
    cccl = Path(args.cuda_path) / "include/cccl"
    cub_extra = ["-I", str(cccl)]
    cases["real_cub_scalar"] = ('#include <cub/cub.cuh>\n__global__ void observer(float* p){p[0]=1;}', True, cub_extra)
    cases["real_cub_block_reduce"] = ('#include <cub/cub.cuh>\n__global__ void observer(float* p){'
        'using Reduce=cub::BlockReduce<float,32>; __shared__ Reduce::TempStorage storage;'
        'float result=Reduce(storage).Sum(p[threadIdx.x]); if(threadIdx.x==0)p[0]=result;}', True, cub_extra)
    cases["real_cub_caller_sdk_vec3"] = ('#include <cub/cub.cuh>\nvoid observer(){uint3 v={1,2,3};}', False, cub_extra)
    cases["real_cub_caller_export_vec3"] = ('#include <cub/cub.cuh>\nusing Future=cub::CubVector<float,3>;', False, cub_extra)
    cases["direct_util_type_kept"] = ('#include <cub/util_type.cuh>\n__global__ void observer(float* p){p[0]=1;}', False, cub_extra)
    cases["direct_tuple_interface_kept"] = ('#include <cuda/std/tuple>\n__global__ void observer(float* p){p[0]=1;}', False, cub_extra)
    copied_entry = headers / "copied_cub/cub"
    copied_entry.mkdir(parents=True)
    # Exact entry bytes at a different physical origin do not gain authority.
    shutil.copyfile(cccl / "cub/cub.cuh", copied_entry / "cub.cuh")
    cases["copied_cub_entry_scalar"] = ('#include <cub/cub.cuh>\n__global__ void observer(float* p){p[0]=1;}', False,
        ["-I", str(copied_entry.parent), *cub_extra])
    alternate_resource = root / "alternate_resource"
    shutil.copytree(Path(args.resource_dir) / "include", alternate_resource / "include")
    alternate_wrapper = alternate_resource / "include/__clang_cuda_runtime_wrapper.h"
    with alternate_wrapper.open("ab") as stream:
        stream.write(b"\n// Alternate genuine Clang provider bytes: semantic no-op.\n")
    cases["unreviewed_provider_scalar"] = ("__global__ void observer(float* p){p[0]=1;}", True, [])
    cases["unreviewed_provider_vec3"] = ("__global__ void observer(){float3 v={1,2,3};}", True, [])
    alternate_builtin_resource = root / "alternate_builtin_resource"
    shutil.copytree(Path(args.resource_dir) / "include", alternate_builtin_resource / "include")
    with (alternate_builtin_resource / "include/__clang_cuda_builtin_vars.h").open("ab") as stream:
        stream.write(b"\n// Alternate genuine builtin-variable bytes: semantic no-op.\n")
    cases["unreviewed_provider_builtin_scalar"] = ("__global__ void observer(float* p){p[0]=1;}", True, [])
    cases["unreviewed_provider_builtin_vec3"] = ("__global__ void observer(){float3 v={1,2,3};}", True, [])

    def run(command, dest):
        dest.mkdir(parents=True)
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, env=env)
        (dest / "argv.json").write_text(json.dumps(command, indent=2) + "\n")
        (dest / "stdout.log").write_text(result.stdout)
        (dest / "stderr.log").write_text(result.stderr)
        (dest / "process.json").write_text(json.dumps({"returncode": result.returncode}, indent=2) + "\n")
        return result

    rows = []
    try:
        for name, (body, accepted, extra) in cases.items():
            dest = root / name
            dest.mkdir()
            source, output = dest / "input.cu", dest / "output.cce"
            receipt = dest / "receipt.json"
            source.write_text(("" if name == "own_global_without_sdk" else prefix) + body + "\n")
            local_headers = []
            if name == "local_header_closure":
                header = dest / "local_observer.h"
                header.write_text("struct Observer { float3 data; };\n")
                local_headers.append(header)
            before = {str(path): sha(path) for path in [source, *local_headers, *[p for p in headers.rglob("*") if p.is_file()]]}
            command = [args.binary, str(source), "--frontend-compat=ascify-admitted-v1",
                       "--target-policy=dav-c310-vec", "--simt-math=fast", "--default-preprocessor",
                       "--cuda-path=" + args.cuda_path,
                       "--clang-resource-directory=" + args.resource_dir, "-o", str(output),
                       "--migration-receipt=" + str(receipt),
                       "--", "-x", "cuda", "-std=c++17", *extra]
            if name == "local_header_closure":
                command.insert(command.index("--"), "--local-headers-recursive")
            unreviewed_provider = name.startswith("unreviewed_provider_")
            if unreviewed_provider:
                chosen_resource = alternate_builtin_resource if name.startswith("unreviewed_provider_builtin_") else alternate_resource
                command[command.index("--clang-resource-directory=" + args.resource_dir)] = "--clang-resource-directory=" + str(chosen_resource)
            result = run(command, dest / "candidate")
            row = {"name": name, "expected_accept": accepted, "candidate_returncode": result.returncode,
                   "candidate_output_exists": output.is_file(), "source_sha256": sha(source)}
            if accepted:
                require(result.returncode == 0 and output.is_file(), name + ": positive failed\n" + result.stderr)
                if unreviewed_provider:
                    row["profile_scope"] = "outside reviewed provider tuple; generation preserves old behavior and proves no native vec3 ABI"
                if args.ccec and name != "own_global_without_sdk" and not unreviewed_provider:
                    obj = dest / "output.o"
                    target = [args.ccec, "-x", "dpp", "--cce-aicore-arch=dav-c310-vec",
                              "-std=c++17", "-O2", "-c", *["-I" + inc for inc in args.target_include],
                              str(output), "-o", str(obj)]
                    target_result = run(target, dest / "target")
                    row["target_returncode"] = target_result.returncode
                    require(target_result.returncode == 0 and obj.is_file() and
                            obj.read_bytes().startswith(b"\x7fELF"), name + ": target failed\n" + target_result.stderr)
                elif args.ccec:
                    row["target_scope"] = ("not compiled: unreviewed provider ABI is outside this matrix" if unreviewed_provider else
                                           "not compiled: native global name collision is an independent target boundary")
            else:
                require(result.returncode != 0 and not output.exists() and receipt.is_file() and
                        json.loads(receipt.read_text()).get("status") == "failed" and marker in result.stderr,
                        name + ": expected ABI rejection\n" + result.stderr)
                if name == "packed_sdk_layout":
                    require("size 12, alignment 1" in result.stderr, "Packed AST layout was not reported accurately")
                sentinel = b"existing complete output\n"
                output.write_bytes(sentinel)
                receipt.write_bytes(b"existing complete receipt\n")
                repeated = run(command, dest / "preservation")
                require(repeated.returncode != 0 and marker in repeated.stderr and output.read_bytes() == sentinel
                        and json.loads(receipt.read_text()).get("status") == "failed",
                        name + ": previous output changed")
                row["previous_output_preserved"] = True
                row["failure_receipt_status"] = "failed"
                if args.baseline_binary:
                    baseline_output = dest / "baseline.cce"
                    baseline = command.copy()
                    baseline[0] = args.baseline_binary
                    baseline[baseline.index("-o") + 1] = str(baseline_output)
                    baseline[baseline.index("--migration-receipt=" + str(receipt))] = "--migration-receipt=" + str(dest / "baseline.receipt.json")
                    baseline_result = run(baseline, dest / "baseline")
                    row["baseline_returncode"] = baseline_result.returncode
                    row["baseline_generated"] = baseline_output.is_file()
                    require(baseline_result.returncode == 0 and baseline_output.is_file(),
                            name + ": baseline did not admit source\n" + baseline_result.stderr)
                    if name == "local_header_closure":
                        bundle = Path(str(baseline_output) + ".headers")
                        require(bundle.is_dir(), "Baseline local-header bundle was not produced")
                        old_files = [baseline_output, *[p for p in bundle.rglob("*") if p.is_file()]]
                        old_hashes = {str(p): sha(p) for p in old_files}
                        candidate_existing = baseline.copy()
                        candidate_existing[0] = args.binary
                        refused = run(candidate_existing, dest / "owned_bundle_preservation")
                        require(refused.returncode != 0 and marker in refused.stderr and
                                old_hashes == {str(p): sha(p) for p in old_files},
                                "Vec3 rejection modified a complete root/local-header bundle")
                        row["owned_local_header_bundle_preserved"] = True
            require(before == {path: sha(Path(path)) for path in before}, name + ": source/header changed")
            row["inputs_unchanged"] = True
            rows.append(row)
            print(name + (": accepted" if accepted else ": rejected, previous output preserved"))
        summary = {"passed": True, "cases": rows, "device_execution_performed": False,
                   "scope": "Frozen CUDA13.4.1 record uses; no automatic type adaptation or corpus gain"}
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
