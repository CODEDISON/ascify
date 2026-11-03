#!/usr/bin/env python3
"""Versioned int/float/double scratch, loop, shared-conversion and raw-input gates.

Every refusal fixture is first checked by the actual source CUDA frontend.
This exercises publication and source contracts; it makes no device claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tempfile


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def loop_cases(original):
    base=original;anchor='  value = forward_sum(value, tile);';loop='''  unsigned index=block.thread_rank(), stride=128, bound=128;
      while(index<bound) { value += input[index]; index += stride; }
    '''
    helper='''
    template<class T> struct EmptyShared {
     __device__ operator T*(){extern __shared__ int memory[];return (T*)memory;}
     __device__ operator const T*() const {extern __shared__ int memory[];return (T*)memory;}
    };
    '''
    withloop=base.replace(anchor,loop+anchor,1)
    withshared=base.replace('namespace cg = cooperative_groups;','namespace cg = cooperative_groups;\n'+helper).replace('  T value =','  T* memory=EmptyShared<T>();\n  T value =',1)
    cases = {
        'legacy_int_float': (base,True),
        'double_scratch': (base+'\ntemplate __global__ void uniform_sum<double,128,64>(const double*,double*);\n',True),
        'guard_no_whitespace_first_add': (base.replace('output) {\n  __shared__','output) {output[0] += input[0];\n  __shared__')+'\ntemplate __global__ void uniform_sum<double,128,64>(const double*,double*);\n',True),
        'guard_macro_conflict': (base.replace('namespace cg = cooperative_groups;','namespace cg = cooperative_groups;\n#define uniform_scalar_domain shadow_domain')+'\ntemplate __global__ void uniform_sum<double,128,64>(const double*,double*);\n',False),
        'grid_stride_loop': (withloop,True),
        'nested_builtin_loops': (withloop.replace('value += input[index];','for(unsigned j=0;j<2;++j){value += T(0);}\n value += input[index];'),True),
        'loop_collective': (withloop.replace('value += input[index];','value = forward_sum(value,tile);'),False),
        'loop_block_sync': (withloop.replace('value += input[index];','block.sync();'),False),
        'loop_unknown_call': (withloop.replace('namespace cg = cooperative_groups;','namespace cg = cooperative_groups;\n__device__ void opaque();').replace('value += input[index];','opaque();'),False),
        'loop_mutates_bound': (withloop.replace('value += input[index];','--bound; value += input[index];'),False),
        'loop_preformed_bound_pointer': (withloop.replace('  while','  unsigned* alias=&bound;\n  while').replace('value += input[index];','*alias=1; value += input[index];'),False),
        'loop_preformed_bound_reference': (withloop.replace('  while','  unsigned& alias=bound;\n  while').replace('value += input[index];','alias=1; value += input[index];'),False),
        'loop_bound_is_reference': (withloop.replace('stride=128, bound=128;','stride=128, n=128; unsigned& bound=n;').replace('value += input[index];','n=1; value += input[index];'),False),
        'loop_volatile_bound': (withloop.replace('stride=128, bound=128;','stride=128; volatile unsigned bound=128;'),False),
        'loop_no_progress': (base.replace(anchor,'  while(block.thread_rank()==0) {}\n'+anchor,1),False),
        'shared_pointer_int_float': (withshared,True),
        'shared_pointer_double': (withshared.replace('};','};\ntemplate<> struct EmptyShared<double>{__device__ operator double*(){extern __shared__ double double_memory[];return (double*)double_memory;} __device__ operator const double*()const{extern __shared__ double double_memory[];return (double*)double_memory;}};',1)+'\ntemplate __global__ void uniform_sum<double,128,64>(const double*,double*);\n',True),
        'shared_constructor_effect': (withshared.replace('struct EmptyShared {','struct EmptyShared {\n__device__ EmptyShared(){}'),False),
        'fake_pointer_conversion': (withshared.replace('extern __shared__ int memory[];return (T*)memory;','return (T*)0;'),False),
        'shared_const_interface': (withshared.replace('operator T*(){','operator T*() const {'),False),
        'shared_double_wrong_element': (withshared+'\ntemplate __global__ void uniform_sum<double,128,64>(const double*,double*);\n',False),
        'shared_conversion_attribute': (withshared.replace('__device__ operator T*()', '__device__ __attribute__((noinline)) operator T*()'),False),
        'shared_macro_body': (withshared.replace('extern __shared__ int memory[];return (T*)memory;','MEMORY_BODY',1).replace('namespace cg = cooperative_groups;','namespace cg = cooperative_groups;\n#define MEMORY_BODY extern __shared__ int memory[];return (T*)memory;'),False),
    }
    return cases


def raw_cases(original):
    base=original+'\ntemplate __global__ void uniform_sum<double,128,64>(const double*,double*);\n'
    primary='template<class T> __device__ __forceinline__ T unrelated_reduce(unsigned int mask,T value){return value;}\n'
    arch='''#if __CUDA_ARCH__ >= 800
    template <> __device__ __forceinline__ int unrelated_reduce<int>(unsigned int mask, int mySum)
    {
     mySum = __reduce_add_sync(mask, mySum);
     return mySum;
    }
    #endif
    '''
    full=base+primary+arch
    guard=lambda src:'#ifndef VECTOR_REDUCTION_INPUT_GUARD\n#define VECTOR_REDUCTION_INPUT_GUARD\n'+src+'\n#endif\n'
    stdio=lambda src:src.replace('#include <cooperative_groups.h>','#include <cooperative_groups.h>\n#include <stdio.h>',1)
    cases = {
        'outer_guard': (guard(base),True,[]),
        'native_stdio': (stdio(base),True,[]),
        'inactive_arch': (full,True,[]),
        'combined': (guard(stdio(full)),True,[]),
        'renamed_family_parameters': (full.replace('unrelated_reduce','other_sum').replace('mySum','result').replace('mask','members'),True,[]),
        'outer_trailing_decl': (guard(base)+'int outside_guard;\n',False,[]),
        'outer_value_definition': (guard(base).replace('#define VECTOR_REDUCTION_INPUT_GUARD','#define VECTOR_REDUCTION_INPUT_GUARD 1'),False,[]),
        'outer_else': (guard(base).rsplit('#endif',1)[0]+'#else\nint alternative;\n#endif\n',False,[]),
        'outer_late_undef': (guard(base).rsplit('#endif',1)[0]+'#undef VECTOR_REDUCTION_INPUT_GUARD\n#endif\n',False,[]),
        'outer_cli_history': (guard(base),False,['-DVECTOR_REDUCTION_INPUT_GUARD=1','-UVECTOR_REDUCTION_INPUT_GUARD']),
        'outer_cli_undef': (guard(base),False,['-UVECTOR_REDUCTION_INPUT_GUARD']),
        'condition_900': (full.replace('>= 800','>= 900'),False,[]),
        'arch_else': (full.rsplit('#endif',1)[0]+'#else\nint alternative;\n#endif\n',False,[]),
        'arch_nested': (full.replace(' mySum =',' #if 0\n int inactive_extra;\n #endif\n mySum ='),False,[]),
        'arch_hidden_cg_reference': (full.replace(' mySum =',' cooperative_groups::block_tile_memory<128> hidden;\n mySum ='),False,[]),
        'arch_extra_asm': (full.replace(' mySum =',' asm("nop");\n mySum ='),False,[]),
        'arch_changed_body': (full.replace('return mySum;','return mySum+1;'),False,[]),
        'arch_twice': (full+arch.replace('unrelated_reduce','unrelated_other'),False,[]),
        'arch_spliced_condition': (full.replace('__CUDA_ARCH__ >=','__CUDA_ARCH__ \\\n >='),False,[]),
        'arch_cli_define': (full,False,['-D__CUDA_ARCH__=800']),
        'arch_cli_undef': (full,False,['-U__CUDA_ARCH__']),
        'arch_fake_intrinsic_redecl': (base+'__device__ int __reduce_add_sync(unsigned,int);\n'+primary+arch,False,[]),
        'arch_macro_intrinsic': (full,False,['-D__reduce_add_sync=other_reduce_intrinsic']),
        'stdio_caller_wrapper': (stdio(base),False,['SHADOW_STDIO']),
    }
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--cuda-path', type=Path, required=True)
    parser.add_argument('--resource-dir', type=Path, required=True)
    parser.add_argument('--clang', required=True)
    parser.add_argument('--evidence-dir', type=Path)
    args = parser.parse_args()
    original = Path(__file__).with_name('uniform_block_reduction_input.cu').read_text()
    cases = {'loop_' + name: (source, accepted, [])
             for name, (source, accepted) in loop_cases(original).items()}
    cases.update({'raw_' + name: case for name, case in raw_cases(original).items()})
    require(len(cases) == 47 and len(set(cases)) == 47 and
            sum(case[1] for case in cases.values()) == 12 and
            sum(not case[1] for case in cases.values()) == 35,
            'The versioned matrix must contain 47 unique cases, 12 positive and 35 negative')
    temporary = None
    if args.evidence_dir:
        root = args.evidence_dir
        require(not root.exists(), 'Use a new evidence directory')
        root.mkdir(parents=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix='ascify-uniform-extended-')
        root = Path(temporary.name)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    compiler = Path(shutil.which(args.clang) or args.clang).resolve()
    identity = lambda source: {'source_sha256': sha(source), 'binary_sha256': sha(args.binary),
                               'compiler_sha256': sha(compiler)}
    write = lambda path, value: path.write_text(json.dumps(value, indent=2) + '\n')
    state = {'schema': 'ascify.uniform-extended-tests.v1', 'complete': False,
             'binary_sha256': sha(args.binary), 'compiler_sha256': sha(compiler),
             'harness_sha256': sha(Path(__file__)),
             'source_fixture_sha256': sha(Path(__file__).with_name('uniform_block_reduction_input.cu')),
             'CCEC': 0, 'NPU': 0, 'cases': []}
    try:
        for name, (source, accepted, extra) in cases.items():
            dest = root / name
            dest.mkdir()
            path, output = dest / 'input.cu', dest / 'output.cce'
            bundle = Path(str(output) + '.headers')
            path.write_text(source)
            before_identity = identity(path)
            if extra == ['SHADOW_STDIO']:
                shadow = dest / 'shadow'
                shadow.mkdir()
                (shadow / 'stdio.h').write_text('#include_next <stdio.h>\n')
                extra = ['-I' + str(shadow)]
            original_command = [args.clang, '-x', 'cuda', '--cuda-host-only', '-nocudalib',
                                '--cuda-path=' + str(args.cuda_path), '-Wno-unknown-cuda-version',
                                '-std=c++17', '-fgpu-rdc', '-I' + str(args.cuda_path / 'include/cccl'),
                                *extra, '-fsyntax-only', str(path)]
            command = [str(args.binary), str(path), '--frontend-compat=ascify-admitted-v1',
                       '--target-policy=dav-c310-vec', '--simt-math=fast', '--default-preprocessor',
                       '--cuda-path=' + str(args.cuda_path),
                       '--clang-resource-directory=' + str(args.resource_dir), '-o', str(output),
                       '--', '-x', 'cuda', '-std=c++17', '-fgpu-rdc', *extra]
            write(dest / 'original.argv.json', original_command)
            write(dest / 'argv.json', command)
            original_result = subprocess.run(original_command, capture_output=True, timeout=90)
            (dest / 'original.stdout.log').write_bytes(original_result.stdout)
            (dest / 'original.stderr.log').write_bytes(original_result.stderr)
            require(original_result.returncode == 0,
                    name + ': original CUDA source is invalid\n' + original_result.stderr.decode())
            before = sha(path)
            repeat_returncode = None
            result = subprocess.run(command, capture_output=True, timeout=90)
            (dest / 'stdout.log').write_bytes(result.stdout)
            (dest / 'stderr.log').write_bytes(result.stderr)
            require(sha(path) == before, name + ': source bytes changed')
            if accepted:
                require(result.returncode == 0 and output.is_file(),
                        name + ': positive conversion failed\n' + result.stderr.decode())
                emitted = output.read_text()
                require(emitted.count('static_assert(::ascify_cg::uniform_scalar_domain<T>::value') == 1,
                        name + ': scalar guard was lost or duplicated')
                expected = sum(line.startswith('template __global__') for line in source.splitlines())
                require(f'instantiations={expected}' in result.stderr.decode(),
                        name + ': concrete instances were not all proven')
                if '__CUDA_ARCH__ >= 800' in source:
                    require(emitted.index('#error "Ascify inactive integer reduction') <
                            emitted.index('#if __CUDA_ARCH__ >= 800'),
                            name + ': target branch guard missing')
            else:
                require(result.returncode != 0 and not output.exists() and not bundle.exists(),
                        name + ': unsafe output or bundle published')
                sentinel = b'previous complete output\n'
                output.write_bytes(sentinel)
                repeated = subprocess.run(command, capture_output=True, timeout=90)
                repeat_returncode = repeated.returncode
                (dest / 'repeat.stdout.log').write_bytes(repeated.stdout)
                (dest / 'repeat.stderr.log').write_bytes(repeated.stderr)
                require(repeated.returncode != 0 and output.read_bytes() == sentinel and not bundle.exists(),
                        name + ': refusal changed the previous output or created a bundle')
            after_identity = identity(path)
            require(before_identity == after_identity, name + ': source/compiler/binary identity changed')
            state['cases'].append({'name': name, 'accepted': accepted,
                                   'source_valid': original_result.returncode == 0,
                                   'returncode': result.returncode, 'source_sha256': before,
                                   'original_returncode': original_result.returncode,
                                   'repeat_returncode': repeat_returncode,
                                   'identity_before': before_identity, 'identity_after': after_identity,
                                   'bundle_exists': bundle.exists(),
                                   'stderr_sha256': sha(dest / 'stderr.log'),
                                   'atomic_refusal_checked': not accepted})
            write(root / 'summary.json', state)
            print('uniform extended ' + name + ': ' + ('accepted' if accepted else 'refused'), flush=True)
        state['complete'] = True
        state['positive'] = sum(row['accepted'] for row in state['cases'])
        state['negative'] = sum(not row['accepted'] for row in state['cases'])
    finally:
        write(root / 'summary.json', state)
        if temporary:
            temporary.cleanup()


if __name__ == '__main__':
    main()
