#!/usr/bin/env python3
"""Real frontend, atomic-output and optional CCEC checks for the closed vec3 profile.

No device is executed. Positive CCEC rows prove compilation only. Arrays/GM,
cross-TU ABI, templates, and angle-only headers intentionally remain rejected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--cuda-path', required=True, type=Path)
    parser.add_argument('--resource-dir', required=True, type=Path)
    parser.add_argument('--ccec', type=Path)
    parser.add_argument('--target-include', action='append', default=[])
    parser.add_argument('--evidence-dir', required=True, type=Path)
    args = parser.parse_args()
    require(not args.evidence_dir.exists(), 'Use a fresh evidence directory')
    require(sha(args.cuda_path / 'include/vector_types.h') ==
            'ded087a2c3ca89fd3a5150589175aca0aa4714357016209ea3e7a51f94845e75', 'Wrong CUDA provider')
    args.evidence_dir.mkdir(parents=True)
    environment = os.environ.copy()
    for key in ('CPATH', 'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH', 'OBJC_INCLUDE_PATH'):
        environment.pop(key, None)
    cases = {
        'float_local': (True, '__global__ void k(float* out){float3 v={1,2,3};out[0]=v.x+v.y+v.z;}'),
        'uint_local': (True, '__global__ void k(unsigned* out){uint3 v={0,0xffffffffu,3};out[0]=v.x+v.y+v.z;}'),
        'int_local': (True, '__global__ void k(int* out){int3 v={-1,2,-3};out[0]=v.x+v.y+v.z;}'),
        'float_factory': (True, '__global__ void k(float* out){auto v=make_float3(1,2,3);out[0]=v.z;}'),
        'uint_factory': (True, '__global__ void k(unsigned* out){auto v=make_uint3(1,2,3);out[0]=v.z;}'),
        'int_factory': (True, '__global__ void k(int* out){auto v=make_int3(-1,2,-3);out[0]=v.z;}'),
        'qualified_record': (True, '__global__ void k(float* out){struct ::float3 v={1,2,3};out[0]=v.y;}'),
        'local_alias_copy': (True, '__global__ void k(float* out){using V=float3;V v={1,2,3};V copied=v;V& ref=copied;out[0]=ref.z;}'),
        'local_assignment': (True, '__global__ void k(float* out){float3 a={1,2,3},b={4,5,6};a=b;out[0]=a.z;}'),
        'inline_value': (True, '__device__ inline float3 helper(float3 a){a.z+=2;return a;}\n__global__ void k(float* out){float3 a={1,2,3};auto b=helper(a);out[0]=b.z;}'),
        'inline_scalar_array_body': (True, '__device__ inline float3 helper(float3 a){int offsets[1]={2};a.z+=offsets[0];return a;}\n__global__ void k(float* out){float3 a={1,2,3};auto b=helper(a);out[0]=b.z;}'),
        'inline_reference': (True, '__device__ inline float helper(const float3& a){return a.z;}\n__global__ void k(float* out){float3 a={1,2,3};out[0]=helper(a);}'),
        'helper_overload': (True, '__host__ __device__ inline float3 make_float3(float x){return make_float3(x,x,x);}\n__global__ void k(float* out){auto a=make_float3(3.0f);out[0]=a.z;}'),
        'layout_inquiry': (True, 'static_assert(sizeof(float3)==12 && alignof(float3)==4 && __builtin_offsetof(float3,z)==8,"layout");\n__global__ void k(float* out){float3 a={1,2,3};out[0]=a.z;}'),
        'quoted_context': (True, '#include "local_vec.h"\n__global__ void k(float* out){auto a=local_value();out[0]=a.z;}'),
        'component_value_call': (True, '__device__ inline float scalar(float x){return x+2;}\n__global__ void k(float* out){float3 a={1,2,3};out[0]=scalar(a.x);}'),
        'component_value_copy_reference': (True, '__global__ void k(float* out){float3 a={1,2,3};float value=a.x;float& ref=value;ref+=2;out[0]=ref;}'),
        'array_rejected': (False, '__global__ void k(){float3 a[2];}'),
        'gm_pointer_rejected': (False, '__global__ void k(float3* p){p[0].z=3;}'),
        'int_array_rejected': (False, '__global__ void k(){int3 a[2];}'),
        'embedding_rejected': (False, 'struct Ray {float3 origin,direction;};'),
        'global_rejected': (False, 'float3 value;'),
        'volatile_rejected': (False, '__global__ void k(){volatile float3 a={1,2,3};}'),
        'external_abi_rejected': (False, 'extern "C" float3 external(float3);'),
        'noninline_abi_rejected': (False, '__device__ float3 helper(float3 a){return a;}'),
        'function_pointer_rejected': (False, 'auto callback=&make_float3;'),
        'global_alias_rejected': (False, 'using Future=float3;'),
        'macro_type_rejected': (False, '#define V float3\n__global__ void k(){V a={1,2,3};}'),
        'macro_factory_rejected': (False, '#define CALL(x) x\n__global__ void k(){auto a=CALL(make_float3(1,2,3));}'),
        'inactive_type_rejected': (False, '#if 0\nfloat3 future;\n#endif\n__global__ void k(){float3 a={1,2,3};}'),
        'dormant_template_rejected': (False, 'template<class T> __device__ inline float3 helper(T x){return make_float3(1,2,3);}'),
        'generic_lambda_rejected': (False, 'void k(){auto f=[](auto x){return make_float3(1,2,3);};}'),
        'template_argument_rejected': (False, 'template<class T> struct Box {}; Box<float3> box;'),
        'function_template_argument_rejected': (False, 'template<class T> __device__ int probe(){return 1;} __global__ void k(int* out){out[0]=probe<float3>();}'),
        'throw_rejected': (False, 'void k(){float3 a={1,2,3};throw a;}'),
        'catch_rejected': (False, 'void k(){try{}catch(float3 a){}}'),
        'dim3_bridge_rejected': (False, 'void k(){dim3 d(make_uint3(1,2,3));}'),
        'builtin_whole_vector_rejected': (False, '__global__ void k(){uint3 a=threadIdx;}'),
        'component_address_rejected': (False, '__global__ void k(){float3 a={1,2,3};float* p=&a.x;}'),
        'component_reference_alias_rejected': (False, 'void external(float&); void k(){float3 a={1,2,3};float& ref=a.x;external(ref);}'),
        'component_const_reference_alias_rejected': (False, 'void take(const float*); void k(){float3 a={1,2,3};const float& ref=a.x;take(&ref);}'),
        'component_reference_parameter_rejected': (False, 'void external(float&); void k(){float3 a={1,2,3};external(a.x);}'),
        'component_inline_reference_parameter_rejected': (False, 'inline float take(const float& x){return x;} void k(){float3 a={1,2,3};take(a.x);}'),
        'component_constructor_reference_rejected': (False, 'struct Holder{Holder(float&);}; void k(){float3 a={1,2,3};Holder value(a.x);}'),
        'component_aggregate_reference_rejected': (False, 'struct Holder{const float& ref;}; void k(){float3 a={1,2,3};Holder value{a.x};}'),
        'component_reference_return_rejected': (False, 'inline float& get(float3& a){return a.x;} void k(){float3 a={1,2,3};get(a)=4;}'),
        'component_conditional_reference_rejected': (False, 'void k(bool choose){float3 a={1,2,3};float other=4;float& ref=choose?a.x:other;}'),
        'component_cast_reference_rejected': (False, 'void k(){float3 a={1,2,3};float& ref=static_cast<float&>(a.x);}'),
        'component_assignment_reference_rejected': (False, 'void k(){float3 a={1,2,3};float& ref=(a.x=4);}'),
        'member_pointer_rejected': (False, 'void k(){auto member=&float3::x;}'),
        'cast_rejected': (False, '__global__ void k(){float3 a={1,2,3};auto b=(float3)a;}'),
        'signature_observation_rejected': (False, 'inline const char* name(float3 a){return __PRETTY_FUNCTION__;}'),
        'unknown_attribute_rejected': (False, '[[unknown::attribute]] inline float3 helper(float3 a){return a;}'),
        'unknown_gnu_attribute_rejected': (False, 'inline __attribute__((unknown)) float3 helper(float3 a){return a;}'),
        'generated_declaration_rejected': (False, 'struct ascify_cuda_float3 {double x,y,z;}; void k(){float3 a={1,2,3};}'),
        'generated_macro_rejected': (False, '#define ascify_cuda_float3 int\n#undef ascify_cuda_float3\nvoid k(){float3 a={1,2,3};}'),
        'angle_header_rejected': (False, '#include <local_vec.h>\nvoid k(){auto a=local_value();}'),
        'copied_provider_rejected': (False, 'void k(){float3 a={1,2,3};}'),
        'mutated_factory_body_rejected': (False, 'void k(){auto a=make_float3(1,2,3);}'),
        'caller_uint3_forward_rejected': (False, 'struct uint3; __global__ void k(){uint3 a={1,2,3};}'),
        'copied_resource_uint3_rejected': (False, '__global__ void k(){uint3 a={1,2,3};}'),
    }

    def run(argv, directory):
        directory.mkdir()
        result = subprocess.run(argv, capture_output=True, env=environment, timeout=120)
        (directory / 'argv.json').write_text(json.dumps(argv, indent=2) + '\n')
        (directory / 'stdout.log').write_bytes(result.stdout)
        (directory / 'stderr.log').write_bytes(result.stderr)
        (directory / 'process.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
        return result

    rows = []
    for name, (accepted, body) in cases.items():
        directory = args.evidence_dir / name
        directory.mkdir()
        source, output = directory / 'input.cu', directory / 'output.cce'
        receipt = directory / 'receipt.json'
        header = directory / 'local_vec.h'
        header.write_text('#ifndef LOCAL_VEC_H\n#define LOCAL_VEC_H\n__host__ __device__ inline float3 local_value(){return make_float3(1,2,3);}\n#endif\n')
        source.write_text('#include <cuda_runtime.h>\n' + body + '\n')
        resource = args.resource_dir
        if name == 'copied_resource_uint3_rejected':
            # Exact resource bytes at an arbitrary caller path are not owned
            # by the executable; do not grant the new uint3 forward proof.
            resource = directory / 'copied_resource'
            shutil.copytree(args.resource_dir, resource)
        extra = ['-I', str(directory)]
        if name in ('copied_provider_rejected', 'mutated_factory_body_rejected'):
            # Exact provider bytes at another physical path do not gain authority.
            shim = directory / 'shim'
            shim.mkdir()
            if name == 'copied_provider_rejected':
                (shim / 'cuda_runtime.h').write_text('#include "copied_vector_types.h"\n')
                (shim / 'copied_vector_types.h').write_bytes((args.cuda_path / 'include/vector_types.h').read_bytes())
            else:
                # Keep actual SDK bytes/paths while changing the parsed return.
                # This is a synthetic input fixture, never an SDK mutation.
                (shim / 'cuda_runtime.h').write_text('#include <cuda_runtime_api.h>\n'
                    '#include <vector_types.h>\n#define return return t.x=99,\n'
                    '#include <vector_functions.hpp>\n#undef return\n')
            extra = ['-nocudainc', '-nocudalib', '-I', str(shim), '-I', str(args.cuda_path / 'include')]
        bundle = directory / 'output.cce.headers'
        # Positive publication starts fresh. An unowned pre-existing bundle is
        # correctly refused by the existing local-header ownership protocol.
        if not accepted:
            output.write_text('preserve previous output\n')
            bundle.mkdir()
            (bundle / 'prior.txt').write_text('preserve previous bundle\n')
        inputs = [source, header]
        if name == 'copied_resource_uint3_rejected':
            inputs += sorted(path for path in resource.rglob('*') if path.is_file())
        if (directory / 'shim').exists(): inputs += sorted((directory / 'shim').rglob('*.h'))
        original = {str(path): sha(path) for path in inputs}
        prior_output = sha(output) if not accepted else None
        prior_bundle = sha(bundle / 'prior.txt') if not accepted else None
        argv = [str(args.binary), str(source), '--frontend-compat=ascify-admitted-v1',
                '--target-policy=dav-c310-vec', '--simt-math=fast', '--default-preprocessor',
                '--cuda-path=' + str(args.cuda_path), '--clang-resource-directory=' + str(resource),
                '-o', str(output), '--migration-receipt=' + str(receipt)]
        if name == 'quoted_context': argv += ['--local-headers-recursive']
        argv += ['--', '-x', 'cuda', '-std=c++17', *extra]
        result = run(argv, directory / 'convert')
        stderr = result.stderr.decode(errors='replace')
        require((result.returncode == 0) == accepted, (name, result.returncode, stderr))
        require(all(sha(Path(path)) == digest for path, digest in original.items()), 'Original source modified: ' + name)
        parsed = json.loads(receipt.read_text())
        if accepted:
            require('ascify/cuda_vec3_layout.hpp' in output.read_text(), 'Missing compatibility header: ' + name)
            require(any(item.get('cuda_vec3_local_layout', {}).get('evidence') for item in parsed['inputs']),
                    'Missing typed provider/site proof receipt: ' + name)
            if name in ('uint_local', 'uint_factory'):
                require(any('compiler-forward=uint3;provider=' in evidence
                            for item in parsed['inputs']
                            for evidence in item.get('cuda_vec3_local_layout', {}).get('evidence', [])),
                        'Missing executable-owned builtin uint3 forward proof: ' + name)
            if name == 'helper_overload':
                text = output.read_text()
                require('make_float3(float x)' in text and 'ascify_make_cuda_float3(x,x,x)' in text,
                        'User overload not preserved with bound SDK factory rewrite')
            if args.ccec:
                object_file = directory / 'target.o'
                command = [str(args.ccec), '-O2', '-std=c++17', '-x', 'dpp',
                           '--cce-aicore-arch=dav-c310-vec', '-c', str(output), '-o', str(object_file)]
                for include in args.target_include: command += ['-I', include]
                target = run(command, directory / 'target_compile')
                require(target.returncode == 0 and object_file.read_bytes().startswith(b'\x7fELF'),
                        (name, target.stderr.decode(errors='replace')))
        else:
            require('CUDA vec3 ABI boundary:' in stderr, 'Failure did not reach the intended semantic gate: ' + name)
            require(sha(output) == prior_output and sha(bundle / 'prior.txt') == prior_bundle,
                    'Failed conversion changed existing output/bundle: ' + name)
            require(not any(item.get('cuda_vec3_local_layout', {}).get('evidence') for item in parsed['inputs']),
                    'Failed conversion claimed a committed vec3 proof: ' + name)
            # The same negative must also reject without publishing artifacts
            # when there is no earlier output or owned bundle to preserve.
            fresh_output = directory / 'fresh_output.cce'
            fresh_bundle = directory / 'fresh_output.cce.headers'
            fresh_receipt = directory / 'fresh_receipt.json'
            require(not fresh_output.exists() and not fresh_bundle.exists(), 'Fresh negative output already exists')
            fresh_argv = list(argv)
            fresh_argv[fresh_argv.index('-o') + 1] = str(fresh_output)
            receipt_index = fresh_argv.index('--migration-receipt=' + str(receipt))
            fresh_argv[receipt_index] = '--migration-receipt=' + str(fresh_receipt)
            fresh_result = run(fresh_argv, directory / 'convert_fresh')
            require(fresh_result.returncode != 0 and 'CUDA vec3 ABI boundary:' in fresh_result.stderr.decode(errors='replace'),
                    'Fresh negative missed the intended semantic gate: ' + name)
            require(not fresh_output.exists() and not fresh_bundle.exists(), 'Fresh negative published output/bundle: ' + name)
            fresh_parsed = json.loads(fresh_receipt.read_text())
            require(not any(item.get('cuda_vec3_local_layout', {}).get('evidence') for item in fresh_parsed['inputs']),
                    'Fresh negative claimed a committed vec3 proof: ' + name)
            require(all(sha(Path(path)) == digest for path, digest in original.items()), 'Fresh negative modified input: ' + name)
            require(sha(output) == prior_output and sha(bundle / 'prior.txt') == prior_bundle,
                    'Fresh negative changed earlier output/bundle: ' + name)
        rows.append({'case': name, 'expected_conversion_success': accepted, 'passed': True,
                     'publication_checks': 'fresh_positive' if accepted else 'existing_preserved_and_fresh_absent',
                     'target_compiled': bool(accepted and args.ccec), 'device_executed': False})
    (args.evidence_dir / 'summary.json').write_text(json.dumps({'passed': True, 'rows': rows,
        'case_count': len(rows), 'device_executed': False, 'sample_count_delta': 0}, indent=2) + '\n')
    print('CUDA vec3 closed local matrix:', len(rows), 'PASS; no device execution')


if __name__ == '__main__':
    main()
