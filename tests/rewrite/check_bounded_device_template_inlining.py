#!/usr/bin/env python3
"""Real CUDA syntax first; attribute selection, receipt and nonpublication gates.

This suite never invokes CCEC or a device and is not a Sample acceptance test.
Cases outside the optional optimization may either retain their prior ordinary
conversion or fail an existing boundary. Neither result may contain new attrs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ATTRIBUTE = '__attribute__((always_inline))'
BASE = '''template<int n> __device__ float chain(float x){return x+chain<n-1>(x);}
template<> __device__ float chain<0>(float x){return x;}
__global__ void kernel(float*p){p[0]=chain<4>(1.0f);}
'''
CASES = [
    ('finite_chain', BASE, 2),
    ('precise_mode', BASE, 0),
    ('renamed_depth', BASE.replace('chain', 'arbitrary_family').replace('<4>', '<11>'), 2),
    ('step_two_terminal_three', BASE.replace('n-1', 'n-2').replace('<0>', '<3>').replace('<4>', '<9>'), 2),
    ('increasing_chain', BASE.replace('n-1', 'n+2').replace('<0>', '<9>').replace('<4>', '<3>'), 2),
    ('multiple_roots', BASE+'__global__ void second(float*p){p[0]=chain<2>(2.0f);}\n', 2),
    ('constant_texture', '''#define RADIUS 4
__constant__ float weights[9];
template<int q> __device__ float filter(float x,float y,cudaTextureObject_t t){
 return tex2D<float>(t,x+(float)(RADIUS-q),y)*weights[q]+filter<q-1>(x,y,t);}
template<> __device__ float filter<-1>(float,float,cudaTextureObject_t){return 0;}
__global__ void kernel(float*p,cudaTextureObject_t t){p[0]=filter<8>(.5f,.5f,t);}
''', 2),
    ('two_families', BASE+BASE.replace('chain', 'other').replace('kernel', 'second'), 4),
    ('cycle', BASE.replace('n-1', 'n'), 0),
    ('repeated_edge', BASE.replace('x+chain<n-1>(x)', 'chain<n-1>(x)+chain<n-1>(x)'), 0),
    ('diamond', BASE.replace('x+chain<n-1>(x)', 'chain<n-1>(x)+chain<n-2>(x)').replace(
        '__global__ void kernel', 'template<> __device__ float chain<1>(float x){return x;}\n__global__ void kernel'), 0),
    ('budget_depth', BASE.replace('<4>', '<70>'), 0),
    ('function_pointer', BASE+'__device__ auto escaped=&chain<2>;\n', 0),
    ('address_in_uninstantiated_body', BASE+'template<int q> __device__ void unused(){auto p=&chain<q>; (void)p;}\n', 0),
    ('forward_declaration', 'template<int n> __device__ float chain(float);\n'+BASE, 0),
    ('terminal_forward', BASE.replace('template<> __device__ float chain<0>(float x){return x;}',
        'template<> __device__ float chain<0>(float);\ntemplate<> __device__ float chain<0>(float x){return x;}'), 0),
    ('terminal_inherited_qualifier', BASE.replace('template<> __device__', 'template<>'), 0),
    ('terminal_raw_attribute', BASE.replace('template<> __device__', 'template<> __attribute__((device))'), 0),
    ('extern_template', BASE+'extern template __device__ float chain<8>(float);\n', 0),
    ('noinline', BASE.replace('__device__ float chain', '__device__ __attribute__((noinline)) float chain'), 0),
    ('host_device', BASE.replace('__device__', '__host__ __device__'), 0),
    ('qualifier_macro', '#define DEVICE __device__\n'+BASE.replace('__device__', 'DEVICE'), 0),
    ('callee_macro', '#define NEXT chain\n'+BASE.replace('x+chain<n-1>', 'x+NEXT<n-1>'), 0),
    ('arithmetic_macro', '#define ADD(a,b) ((a)+(b))\n'+BASE.replace('x+chain<n-1>(x)', 'ADD(x,chain<n-1>(x))'), 0),
    ('hidden_body', '#define BODY return x+chain<n-1>(x);\n'+BASE.replace('return x+chain<n-1>(x);', 'BODY'), 0),
    ('attribute_macro_history', '#define always_inline noinline\n#undef always_inline\n'+BASE, 0),
    ('local_variable', BASE.replace('return x+chain<n-1>(x);', 'float y=x;return y+chain<n-1>(x);'), 0),
    ('unknown_call', '__device__ float unknown(float x){return x;}\n'+BASE.replace('return x+chain', 'return unknown(x)+chain'), 0),
    ('volatile_read', '__device__ volatile float value;\n'+BASE.replace('return x+chain', 'return value+chain'), 0),
    ('body_reflection', BASE.replace('return x+chain', 'return float(sizeof(cudaTextureDesc))+chain'), 0),
    ('no_instantiation', BASE[:BASE.index('__global__')], 0),
    ('header_definition', '#include "family.h"\n__global__ void kernel(float*p){p[0]=chain<4>(1.0f);}\n', 0),
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def main():
    parser = argparse.ArgumentParser()
    for name in ('binary', 'clang', 'cuda-path', 'resource-dir', 'evidence-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--frontend-compat', choices=('none', 'ascify-admitted-v1'), default='none')
    args = parser.parse_args()
    for name in ('binary', 'clang', 'cuda_path', 'resource_dir'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    version = subprocess.check_output([str(args.clang), '--version'], text=True)
    major = re.search(r'\bclang version (\d+)', version)
    require(major is not None and major.group(1) == args.resource_dir.name.split('.')[0],
            'clang/resource mismatch')
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    summary = {'passed': False, 'device_executed': False, 'original_sample_successes': 0,
               'binary_sha256': sha(args.binary), 'clang_sha256': sha(args.clang),
               'runner_sha256': sha(Path(__file__)), 'cases': []}
    def save():
        (args.evidence_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    def run(directory, role, argv):
        argv = list(map(str, argv))
        (directory/(role+'.argv.json')).write_text(json.dumps(argv, indent=2)+'\n')
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        (directory/(role+'.stdout.log')).write_text(result.stdout)
        (directory/(role+'.stderr.log')).write_text(result.stderr)
        return result
    for name, code, expected in CASES:
        directory = args.evidence_dir/name
        directory.mkdir()
        source, output, receipt = (directory/p for p in ('input.cu', 'output.cce', 'receipt.json'))
        source.write_text('#include <cuda_runtime.h>\n'+code+'\ncudaTextureDesc descriptor{};\n')
        if name == 'header_definition':
            (directory/'family.h').write_text(BASE[:BASE.index('__global__')])
        identities = {str(p): sha(p) for p in directory.iterdir() if p.is_file()}
        row = {'name': name, 'expected_definitions': expected, 'source_sha256': sha(source), 'passed': False}
        summary['cases'].append(row)
        save()
        try:
            syntax = run(directory, 'syntax', [args.clang, source, '-x', 'cuda', '-std=c++17',
                '--cuda-host-only', '-fsyntax-only', '-nocudalib', '--cuda-path='+str(args.cuda_path),
                '-resource-dir='+str(args.resource_dir), '-Wno-unknown-cuda-version'])
            row['syntax_returncode'] = syntax.returncode
            require(syntax.returncode == 0, 'original CUDA syntax failed: '+syntax.stderr)
            converted = run(directory, 'convert', [args.binary, source, '--target-policy=dav-c310-vec',
                '--simt-math='+('precise' if name == 'precise_mode' else 'fast'),
                '--target-recipe=none', '--default-preprocessor',
                '--frontend-compat='+args.frontend_compat, '--cuda-path='+str(args.cuda_path),
                '--clang-resource-directory='+str(args.resource_dir), '--migration-receipt='+str(receipt),
                '-o', output, '--', '-x', 'cuda', '-std=c++17'])
            row['conversion_returncode'] = converted.returncode
            require(converted.returncode >= 0, 'converter crashed')
            data = json.loads(receipt.read_text())
            require(len(data['inputs']) == 1, 'expected one receipt input')
            decision = data['inputs'][0]['bounded_device_template_inlining']
            require(decision == {'rule': 'int-nttp-scalar-dag-v1', 'definitions': expected},
                    'unexpected receipt rule/count: '+str(decision))
            if converted.returncode == 0:
                require(data['status'] == 'succeeded', 'success receipt missing')
                emitted = output.read_text()
                require(emitted.count(ATTRIBUTE) == expected, 'attribute count mismatch')
                require('ASCIFY_FORCEINLINE' not in emitted, 'language inline must not be introduced')
                row['output_sha256'] = sha(output)
            else:
                require(expected == 0 and data['status'] == 'failed', 'positive gate failed')
                require(not output.exists() and not output.is_symlink() and
                        not Path(str(output)+'.headers').exists() and
                        not Path(str(output)+'.headers').is_symlink(), 'failed publication')
                require('fatal error:' not in converted.stderr and any(marker in converted.stderr for marker in
                    ('CUDA texture boundary', 'CUDA runtime-API header boundary',
                     'Ascify cannot publish CUDA compatibility output')), 'environment/unclassified rejection')
            require(all(sha(Path(p)) == digest for p, digest in identities.items()), 'input changed')
            row['passed'] = True
        except (RuntimeError, subprocess.TimeoutExpired, KeyError, OSError, ValueError) as error:
            row['error'] = str(error)
        save()
        print(name, 'passed' if row['passed'] else 'FAILED', flush=True)
    require(sha(Path(__file__)) == summary['runner_sha256'] and
            sha(args.binary) == summary['binary_sha256'] and sha(args.clang) == summary['clang_sha256'],
            'binary changed during run')
    summary['passed'] = all(row['passed'] for row in summary['cases'])
    save()
    require(summary['passed'], 'bounded inlining matrix contains failures')

if __name__ == '__main__':
    main()
