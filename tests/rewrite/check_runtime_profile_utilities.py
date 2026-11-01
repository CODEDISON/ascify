#!/usr/bin/env python3
"""Syntax-first authenticated Runtime utility patterns; no target execution."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


GUARD = 'ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_HOST_MATH_H_'
PREFIX = '#include <cuda_runtime.h>\n#include <cuda_runtime_api.h>\n'
CG = '#include <cooperative_groups.h>\n'


def cases(profile):
    def item(name, code='int retained;\n', fragments=None, extra=(), headers=None,
             recursive=False, default=True):
        return dict(name=name, code=PREFIX + code, fragments=fragments,
                    extra=list(extra), headers=headers or {}, recursive=recursive,
                    default=default)

    yield item('unused_host_patterns', fragments=['int retained;'])
    yield item('unused_cg_primary_and_partial', CG + 'int retained;\n', ['int retained;'])
    yield item('guarded_cg_repeat', CG + CG + 'int retained;\n', ['int retained;'])
    yield item('block_sync', CG + '__global__ void k(){cooperative_groups::this_thread_block().sync();}\n',
               ['ascify_cg::this_thread_block().sync()'])
    yield item('host_float_max', 'float f(){return max(1.0f,2.0f);}\n', ['__builtin_fmaxf(1.0f,2.0f)'])
    yield item('host_int_min', 'int f(){return min(1,2);}\n', ['__builtin_elementwise_min(1,2)'])
    yield item('host_mixed_min', 'unsigned f(){return min(-1,1u);}\n', ['::ascify::cuda_host_min_unsigned(-1,1u)'])
    yield item('host_mixed_min_reverse', 'unsigned f(){return ::min(1u,-1);}\n', ['::ascify::cuda_host_min_unsigned(1u,-1)'])
    yield item('late_unrelated_macro', '#define UnusedCallerMarker 1\n#undef UnusedCallerMarker\nint retained;\n', ['int retained;'])
    child = '#ifndef PROJECT_KERNEL_CUH\n#define PROJECT_KERNEL_CUH\n' + PREFIX + CG + 'int retained;\n#endif\n'
    yield item('selected_local_header', '#include "kernel.cuh"\n', ['int retained;'],
               headers={'kernel.cuh': child}, recursive=True)
    yield item('caller_dependent_template', 'template<class T>T identity(T x){return x;}\n')
    yield item('caller_void_nttp', 'template<int N>void recurse(){recurse<N-1>();}\n')
    yield item('caller_sdk_reflection', 'size_t f(){return sizeof(cudaDeviceProp);}\n')
    yield item('caller_sdk_trait', 'bool f(){return __is_trivial(cudaDeviceProp);}\n')
    yield item('caller_template_sdk_reflection', 'template<class T>size_t bytes(){return sizeof(T);}\nsize_t f(){return bytes<cudaDeviceProp>();}\n')
    yield item('concrete_product_class', CG + 'extern template class cooperative_groups::ascify_detail::tile_register_operations<32>;\n')
    yield item('caller_product_specialization', CG + 'template<>class cooperative_groups::thread_block_tile<32,void>{cudaDeviceProp property;};\nsize_t f(){return sizeof(cooperative_groups::thread_block_tile<32,void>);}\n')
    yield item('runtime_reduce_not_admitted', '#include <cooperative_groups/reduce.h>\nint retained;\n')
    yield item('guard_command_define', extra=['-D' + GUARD])
    yield item('guard_command_undef', extra=['-U' + GUARD])
    yield item('guard_undef_restore', '#undef ' + GUARD + '\n#define ' + GUARD + '\nint retained;\n')
    yield item('guard_push_pop', '#pragma push_macro("' + GUARD + '")\n#pragma pop_macro("' + GUARD + '")\nint retained;\n')
    yield item('guard_inactive_pop', '#if 0\n#pragma pop_macro("' + GUARD + '")\n#endif\nint retained;\n')
    yield item('guard_operator', '_Pragma("push_macro(\\"' + GUARD + '\\")")\nint retained;\n')
    yield item('guard_spliced_operator', '_Pra\\\ngma("push_macro(\\"' + GUARD + '\\")")\nint retained;\n')
    yield item('guard_inactive_operator', '#if 0\n_Pragma("push_macro(\\"' + GUARD + '\\")")\n#endif\nint retained;\n')
    yield item('guard_imported_operator', '#include "pragma.cuh"\nPROJECT_PRAGMA("push_macro(\\"' + GUARD + '\\")")\nint retained;\n',
               headers={'pragma.cuh': '#define PROJECT_PRAGMA(x) _Pragma(x)\n'})
    yield item('guard_digraph', '%:undef ' + GUARD + '\n%:define ' + GUARD + '\nint retained;\n')
    yield item('cuda_control_restore', '#undef __CUDA__\n#define __CUDA__ 1\nint retained;\n')
    yield item('cuda_control_inactive', '#if 0\n#if defined(__CUDACC__)\nint ignored;\n#endif\n#endif\nint retained;\n')
    yield item('parameter_macro_at_entry', extra=['-DA=CallerType'])
    yield item('inline_macro_at_entry', extra=['-Dinline='])
    yield item('owner_macro_at_entry', extra=['-Dhost_float_max_enabled=CallerEnabled'])
    yield item('mixed_owner_macro_at_entry', 'unsigned f(){return min(-1,1u);}\n',
               extra=['-Dhost_mixed_uint_min_enabled=CallerEnabled'])
    yield item('preinclude_macro_at_entry', '#undef A\nint retained;\n', extra=['-include', '{case}/control.cuh'],
               headers={'control.cuh': '#define A CallerType\n'})
    yield item('hostmath_reentry_namespace', '#undef ' + GUARD + '\nnamespace caller{\n#include <host_math.h>\n}\nint retained;\n')
    copied = (profile / 'host_math.h').read_text().replace(GUARD, 'CALLER_HOST_MATH_H_')
    yield item('copied_hostmath_namespace', 'namespace caller{\n#include "copied.h"\n}\nint retained;\n',
               headers={'copied.h': copied})
    yield item('copied_mixed_hostmath_namespace',
               'namespace caller{\n#include "copied.h"\n}\n'
               'unsigned f(){return caller::min(-1,1u);}\n', headers={'copied.h': copied})
    yield item('copied_cg', '#include "cooperative_groups.h"\nint retained;\n',
               headers={'cooperative_groups.h': (profile / 'cooperative_groups.h').read_text()})
    yield item('nested_cg', '#include <type_traits>\nnamespace caller{\n' + CG + '}\nint retained;\n')
    yield item('conditional_retention_mode', default=False)

    # The owned CG selector is a TypeLoc-only proof, not caller uint4 ABI or
    # type-query support. These remain syntax-first atomic rejection cases.
    yield item('caller_uint4_identity_template', CG +
               'template<class T>bool same(){return std::is_same<T,uint4>::value;}\n')
    yield item('caller_uint4_sizeof', CG + 'size_t f(){return sizeof(uint4);}\n')
    yield item('caller_uint4_alignof', CG + 'size_t f(){return alignof(uint4);}\n')
    yield item('caller_uint4_builtin_trait', CG + 'bool f(){return __is_trivial(uint4);}\n')
    yield item('caller_uint4_alias', CG + 'using Packed=uint4; int retained;\n')
    yield item('caller_uint4_redeclaration', CG + 'struct uint4; int retained;\n')
    yield item('caller_identity_redeclaration', CG +
               'namespace std {template<class A,class B>struct is_same;} int retained;\n')
    yield item('caller_identity_partial', CG +
               'namespace std {template<class T>struct is_same<T,uint4>{static constexpr bool value=true;};}\n')
    # The real SDK's one-level `typedef struct uint4 uint4` is permitted only
    # at the owned selector use. A caller may not extend that typedef chain.
    yield item('caller_uint4_typedef_redeclaration', CG +
               'typedef struct uint4 uint4; int retained;\n')
    yield item('caller_uint4_typedef_alias', CG +
               'typedef uint4 CallerPacked; int retained;\n')

    # An implicitly instantiated class retains a distinct member-template
    # pattern and concrete constructor specialization. Only the former has
    # the copied-pattern domain; later caller/concrete observations stay gated.
    tile = ('__global__ void tile_kernel(){'
            'auto block=cooperative_groups::this_thread_block();'
            'cooperative_groups::thread_block_tile<32> tile='
            'cooperative_groups::tiled_partition<32>(block);tile.sync();}\n')
    yield item('copied_member_pattern_and_constructor', CG + tile, ['tile.sync()'])
    yield item('copied_member_named_parent', CG + tile.replace(
               'thread_block_tile<32>', 'thread_block_tile<32,cooperative_groups::thread_block>'),
               ['tile.sync()'])
    guarded_tile = '#ifndef COPIED_TILE_CUH\n#define COPIED_TILE_CUH\n' + PREFIX + CG + tile + '#endif\n'
    yield item('copied_member_selected_header', '#include "tile.cuh"\n', ['tile.sync()'],
               headers={'tile.cuh': guarded_tile}, recursive=True)
    yield item('copied_member_then_caller_dependent', CG + tile +
               'template<class T>T caller_identity(T x){return x;}\n')
    yield item('copied_member_then_sdk_reflection', CG + tile +
               'size_t caller_bytes(){return sizeof(cudaDeviceProp);}\n')
    yield item('explicit_tile_class_instantiation', CG +
               'extern template class cooperative_groups::thread_block_tile<32,void>;\n')
    yield item('copied_member_caller_class_redeclaration', CG +
               'namespace cooperative_groups {template<unsigned int Size,class ParentT>class thread_block_tile;}\n' + tile)
    yield item('copied_member_concrete_sdk_shuffle', CG +
               '__global__ void k(uint4* out){auto block=cooperative_groups::this_thread_block();'
               'cooperative_groups::thread_block_tile<32> tile=cooperative_groups::tiled_partition<32>(block);'
               'uint4 value=out[0];out[0]=tile.shfl_xor(value,1);}\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('binary', 'cuda-path', 'resource-dir', 'evidence-dir'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--clang', type=Path)
    args = parser.parse_args()
    assert not sys.flags.optimize
    for name in ('binary', 'cuda_path', 'resource_dir', 'evidence_dir'):
        setattr(args, name, getattr(args, name).resolve())
    candidates = [args.clang] if args.clang else [args.resource_dir.parents[2] / 'bin/clang++',
                                                Path(shutil.which('clang++') or '/missing')]
    args.clang = next((path.resolve() for path in candidates if path.is_file()), None)
    if args.clang is None:
        parser.error('matching real Clang required')
    version = subprocess.check_output([str(args.clang), '--version'], text=True)
    major = re.search(r'\bclang version (\d+)(?:\.|\s)', version)
    if major is None or major.group(1) != args.resource_dir.name.split('.')[0]:
        parser.error('Clang/resource major mismatch')
    profile = args.binary.parent.parent / 'libexec/ascify/frontend-compat/ascify-admitted-v1'
    inputs = [Path(__file__).resolve(), args.binary, args.clang,
              profile / 'host_math.h', profile / 'cooperative_groups.h',
              profile / 'cooperative_groups/reduce.h']
    identities = {str(path): sha(path) for path in inputs}
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(passed=False, device_executed=False, original_sample_successes=0,
                   identities=identities, cases=[])

    def save():
        (args.evidence_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    def run(directory, name, argv):
        argv = list(map(str, argv))
        (directory / (name + '.argv.json')).write_text(json.dumps(argv, indent=2) + '\n')
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        (directory / (name + '.stdout.log')).write_text(result.stdout)
        (directory / (name + '.stderr.log')).write_text(result.stderr)
        (directory / (name + '.process.json')).write_text(json.dumps(dict(returncode=result.returncode)) + '\n')
        return result

    save()
    try:
        for case in cases(profile):
            directory = args.evidence_dir / case['name']; directory.mkdir()
            for name, code in case['headers'].items():
                path = directory / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(code)
            source = directory / 'input.cu'; source.write_text(case['code'])
            frozen = {str(path): sha(path) for path in directory.rglob('*') if path.is_file()}
            extra = [value.replace('{case}', str(directory)) for value in case['extra']]
            common = ['-x', 'cuda', '-std=c++17', '-iquote' + str(directory), *extra]
            row = dict(name=case['name'], passed=False, fragments=case['fragments'], input_identity=frozen)
            summary['cases'].append(row); save()
            # Explicitly include the exact selected profile for the syntax
            # oracle. The converter independently chooses its own forced file.
            syntax = run(directory, 'syntax', [args.clang, source, '-I' + str(profile), *common, '--cuda-host-only',
                         '-fsyntax-only', '-nocudalib', '--cuda-path=' + str(args.cuda_path),
                         '-resource-dir=' + str(args.resource_dir), '-Wno-unknown-cuda-version',
                         '-include', profile / 'host_math.h'])
            row['syntax_returncode'] = syntax.returncode; save()
            assert syntax.returncode == 0, (case['name'], 'invalid CUDA fixture', syntax.stderr)
            accepted = case['fragments'] is not None
            row['publications'] = []
            for mode in (['fresh'] if accepted else ['fresh', 'existing']):
                output = directory / (mode + '.cce'); receipt = directory / (mode + '.receipt.json')
                sentinel = 'EXISTING_UNCHANGED_' + case['name'] + '\n'
                if mode == 'existing': output.write_text(sentinel)
                result = run(directory, mode, [args.binary, source, '--target-policy=dav-c310-vec',
                             '--simt-math=fast', '--target-recipe=none', '--frontend-compat=ascify-admitted-v1',
                             *(['--default-preprocessor'] if case['default'] else []),
                             '--cuda-path=' + str(args.cuda_path), '--clang-resource-directory=' + str(args.resource_dir),
                             '--migration-receipt=' + str(receipt), '-o', output,
                             *(['--local-headers-recursive'] if case['recursive'] else []), '--', *common])
                row['publications'].append(dict(mode=mode, returncode=result.returncode)); save()
                assert not re.search(r'file not found|no such file or directory|fatal error:', result.stderr, re.I), (case['name'], result.stderr)
                state = json.loads(receipt.read_text())
                if accepted:
                    assert result.returncode == 0 and state['status'] == 'succeeded', (case['name'], result.stderr)
                    text = output.read_text()
                    texts = [text] + [path.read_text() for path in Path(str(output) + '.headers').rglob('*') if path.is_file()]
                    assert '#include <ascify/ascify_cuda_compat.hpp>' in '\n'.join(texts), (case['name'], texts)
                    assert all(any(fragment in text for text in texts) for fragment in case['fragments']), (case['name'], texts)
                    row['output_sha256'] = sha(output)
                else:
                    assert result.returncode > 0 and state['status'] == 'failed', (case['name'], result.stderr)
                    assert not output.is_symlink()
                    assert (not output.exists()) if mode == 'fresh' else output.read_text() == sentinel
                    assert not Path(str(output) + '.headers').exists() and not Path(str(output) + '.headers').is_symlink()
                    assert any(marker in result.stderr for marker in ('CUDA runtime-API header boundary', 'CUDA texture boundary',
                                'frontend compatibility', 'Ascify cannot publish CUDA compatibility output')), (case['name'], result.stderr)
                assert {name: sha(Path(name)) for name in frozen} == frozen
            row['passed'] = True; save(); print(case['name'] + ': passed', flush=True)
        assert {name: sha(Path(name)) for name in identities} == identities
        summary.update(passed=True, total=len(summary['cases']))
    except Exception as error:
        summary['error'] = str(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
