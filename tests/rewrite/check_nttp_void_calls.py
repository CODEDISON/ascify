#!/usr/bin/env python3
"""Real-CUDA syntax first, closed void-call source transactions second.

No CCEC, translated-program execution, ACL calls or device queries. Every
refusal must first pass the independent original-CUDA syntax command. This
script is intentionally not executed while its source review is pending.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

# The explicit Runtime API entry exercises the actual retained-type audit, as
# opposed to making refusal depend on an unrelated include being incidental.
PREFIX = ('#include <cuda_runtime_api.h>\n#include <cuda_runtime.h>\n'
          '#include <cooperative_groups.h>\nnamespace cg=cooperative_groups;\n')
LEAF = '''template<unsigned Tile>
__device__ void leaf(volatile float* p,float value,unsigned lane,cg::thread_block block){
  p[lane]=value;
  cg::sync(block);
}
'''
RELAY = '''template<unsigned Width,bool Power>
__device__ void relay(const float* input,float* output,unsigned n,cg::thread_block block){
  extern __shared__ float scratch[];
  const unsigned lane=threadIdx.x;
  const float value=input[lane];
  leaf<Width>(scratch,value,lane,block);
  output[lane]=scratch[lane];
}
'''
KERNELS = '''template<unsigned Count,bool Flag>
__global__ void first_kernel(const float* input,float* output,unsigned n){
  cg::thread_block block=cg::this_thread_block();
  relay<Count,Flag>(input,output,n,block);
}
template<unsigned Count,bool Flag>
__global__ void second_kernel(const float* input,float* output,unsigned n){
  cg::thread_block block=cg::this_thread_block();
  relay<Count,Flag>(input,output,n,block);
  extern __shared__ float tail[];
  const unsigned lane=threadIdx.x;
  const float value=input[lane];
  leaf<Count>(tail,value,lane,block);
}
'''


def complete(leaf=LEAF, relay=RELAY, kernels=KERNELS, before='', after=''):
    launches = ''.join(
        f'  {name}<{b},{flag}><<<1,{b},{4*b}>>>(input,output,{b});\n'
        for name in ('first_kernel', 'second_kernel')
        for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        for flag in ('false', 'true'))
    return (PREFIX + before + leaf + relay + kernels +
            'void instantiate(float* input,float* output){\n' + launches + '}\n' + after)


def fixtures():
    base = complete()

    def case(name, source, accepted=True, files=None, flags=(), needs=True,
             retained=(), qualifiers=('::leaf<', '::relay<'), recursive=False,
             remap_file=None):
        result = dict(name=name, source=source, accepted=accepted, files=files or {},
                      flags=list(flags), needs=needs, retained=list(retained),
                      qualifiers=list(qualifiers), recursive=recursive)
        if remap_file is not None:
            result['remap_file'] = remap_file
        return result

    yield case('all40', base, retained=('volatile float* p', 'const float* input',
               'cg::thread_block block', 'first_kernel<512,true>', 'second_kernel<1,false>'))
    renamed = base.replace('leaf', 'write_lane').replace('relay', 'forward_lane')
    yield case('renamed', renamed, qualifiers=('::write_lane<', '::forward_lane<'))
    # Value-domain coverage is source-only: no unsafe launch is executed.
    yield case('full_unsigned_domain', '''#include <cuda_runtime.h>
template<unsigned Value,bool Flag> __device__ void tag(float* p){p[0]=0.0f;}
template<unsigned Value,bool Flag> __global__ void mark(float* p){tag<Value,Flag>(p);}
void instantiate(float* p){mark<0,false><<<1,1>>>(p);mark<4294967295u,true><<<1,1>>>(p);}
''', qualifiers=('::tag<',))
    for order in ('before', 'after'):
        source = ('#include "local.cuh"\n#include <cuda_runtime.h>\n' if order == 'before' else
                  '#include <cuda_runtime.h>\n#include "local.cuh"\n')
        yield case('header_' + order, source, files={'local.cuh': base})
    yield case('no_new_candidate', '''#include <cuda_runtime.h>
template<int Offset> __device__ float add_offset(float x){return x+Offset;}
__global__ void ordinary(float* output,const float* input){output[threadIdx.x]=add_offset<3>(input[threadIdx.x]);}
''', needs=False, qualifiers=())
    yield case('int_void_no_explicit_sdk', '''template<int N>
__device__ void int_leaf(float* p){p[0]=N;}
template<int N> __global__ void int_caller(float* p){int_leaf<N>(p);}
void instantiate(float* p){int_caller<3><<<1,1>>>(p);}
''', needs=False, qualifiers=())
    yield case('ordinary_no_explicit_sdk', 'int ordinary_host(int value){return value+1;}\n',
               needs=False, qualifiers=())

    adl = '''namespace cooperative_groups {
template<unsigned N> __device__ int leaf(float* p,float value,unsigned lane,thread_block block){
  p[lane]=value;return 1;
}}
'''
    yield case('adl_better_cv', complete(before=adl), accepted=False)
    yield case('default_nttp', complete(leaf=LEAF.replace('unsigned Tile>', 'unsigned Tile=64>')),
               accepted=False)
    enum = base.replace('namespace cg=cooperative_groups;\n',
                        'namespace cg=cooperative_groups;\nenum class Key:unsigned{one=1};\n')
    enum = enum.replace('unsigned Tile', 'Key Tile').replace('unsigned Width', 'Key Width')
    enum = enum.replace('unsigned Count', 'Key Count')
    enum = re.sub(r'<(?:1|2|4|8|16|32|64|128|256|512),(false|true)>', r'<Key::one,\1>', enum)
    yield case('enum_nttp', enum, accepted=False)
    yield case('pointer_cast', complete(relay=RELAY.replace(
        'leaf<Width>(scratch,', 'leaf<Width>(static_cast<volatile float*>(scratch),')), accepted=False)
    yield case('foreign_array_root', complete(before='__device__ float foreign[512];\n',
        relay=RELAY.replace('leaf<Width>(scratch,', 'leaf<Width>(foreign,')), accepted=False)
    yield case('extern_device_global_alias', complete(before='__device__ float foreign[512];\n',
        relay=RELAY.replace('extern __shared__ float scratch[];',
                            'extern __device__ float foreign[512];').replace('scratch', 'foreign')),
        accepted=False)
    yield case('volatile_scalar_root', complete(relay=RELAY.replace(
        'const float value=', 'volatile float value=')), accepted=False)
    yield case('unknown_group_producer', complete(before='__device__ cg::thread_block manufacture();\n',
        kernels=KERNELS.replace('cg::this_thread_block()', 'manufacture()')), accepted=False)
    yield case('address_escape', complete(after='''__device__ void escape(){
  auto pointer=&leaf<64>;(void)pointer;
}
'''), accepted=False)
    yield case('decltype_observation', complete(relay=RELAY.replace('  output[lane]=',
        '  using Observed=decltype(leaf<Width>(scratch,value,lane,block));\n  output[lane]=')),
        accepted=False)
    yield case('noexcept_observation', complete(relay=RELAY.replace('  output[lane]=',
        '  (void)noexcept(leaf<Width>(scratch,value,lane,block));\n  output[lane]=')), accepted=False)
    yield case('constexpr_callee', '''#include <cuda_runtime.h>
template<unsigned N> __device__ constexpr void constant_leaf(float*){}
template<unsigned N> __global__ void caller(float* p){constant_leaf<N>(p);}
void instantiate(float* p){caller<64><<<1,64>>>(p);}
''', accepted=False)
    macro = '#define WRITE_LANE(N) leaf<N>(scratch,value,lane,block)\n'
    yield case('macro_call', complete(before=macro, relay=RELAY.replace(
        'leaf<Width>(scratch,value,lane,block)', 'WRITE_LANE(Width)')), accepted=False)
    yield case('cli_name_alias', base.replace('leaf', 'MY_LEAF'), accepted=False,
               flags=('-DMY_LEAF=leaf',))
    yield case('cli_nttp_literal', complete(relay=RELAY.replace(
        'leaf<Width>', 'leaf<MY_NTTP>')), accepted=False, flags=('-DMY_NTTP=64',))
    yield case('source_nttp_alias', complete(before='#define MY_NTTP Width\n',
        relay=RELAY.replace('leaf<Width>', 'leaf<MY_NTTP>')), accepted=False)
    yield case('late_nttp_redefinition', complete(before='#define MY_NTTP 64\n',
        relay=RELAY.replace('leaf<Width>', 'leaf<MY_NTTP>') +
        '#undef MY_NTTP\n#define MY_NTTP 128\n'), accepted=False)
    cross_body = LEAF[:LEAF.index('{') + 1] + '\n#include "body_end.cuh"\n'
    yield case('cross_include_body', complete(leaf=cross_body), accepted=False,
        files={'body_end.cuh': '  p[lane]=value;\n  cg::sync(block);\n}\n'}, recursive=True)
    yield case('cg_layout', complete(leaf=LEAF.replace('  p[lane]=',
        '  const unsigned layout=sizeof(cg::thread_block);(void)layout;\n  p[lane]=')), accepted=False)
    specialization = '''template<> __device__ void leaf<128>(volatile float* p,float value,unsigned lane,cg::thread_block){
  p[lane]=value;
}
'''
    yield case('explicit_specialization', complete(leaf=LEAF + specialization), accepted=False)
    late = '''template<int K> __device__ void other(float* p){p[0]=0;}
template<int K> __global__ void late_kernel(float* p){other<K>(p);}
void instantiate_late(float* p){late_kernel<3><<<1,1>>>(p);}
'''
    yield case('late_family_atomicity', complete(after=late), accepted=False)
    # UInt/Bool calls above engage the new plan. These extra calls reference
    # the SAME admitted family, so an int/non-template caller cannot bypass
    # closure merely because its literal call is already nondependent.
    mixed_int = '''template<int Index> __global__ void mixed_int(float* p){
  cg::thread_block block=cg::this_thread_block();
  const unsigned lane=threadIdx.x;
  const float value=p[lane];
  leaf<64>(p,value,lane,block);
}
void instantiate_mixed_int(float* p){mixed_int<7><<<1,64>>>(p);}
'''
    yield case('mixed_int_fixed_call', complete(after=mixed_int), accepted=False)
    yield case('mixed_int_dependent_lookup', complete(after=mixed_int.replace(
        'leaf<64>', 'leaf<static_cast<unsigned>(Index)>')), accepted=False)
    ordinary = '''__global__ void mixed_ordinary(float* p){
  cg::thread_block block=cg::this_thread_block();
  const unsigned lane=threadIdx.x;
  const float value=p[lane];
  leaf<64>(p,value,lane,block);
}
void instantiate_mixed_ordinary(float* p){mixed_ordinary<<<1,64>>>(p);}
'''
    yield case('mixed_ordinary_fixed_call', complete(after=ordinary), accepted=False)
    fake = '''#include <cuda_runtime.h>
namespace cooperative_groups {
class thread_block{};
__device__ thread_block this_thread_block();
__device__ void sync(const thread_block&);
}
'''
    yield case('same_named_cg_header', base, accepted=False,
               files={'cooperative_groups.h': fake})
    yield case('line_spoofed_cg_header', base, accepted=False,
               files={'cooperative_groups.h': '#line 1 "cooperative_groups.h"\n' + fake})
    yield case('caller_remapped_root', base, accepted=False,
               files={'remapped.cu': base}, remap_file='remapped.cu')

    # New origin checks retain the original 35 fixtures above byte-for-byte.
    def origin(name, flags=(), accepted=False, files=None, copy=False, source=base,
               provider='shadow/cooperative_groups.h'):
        result = case(name, source, accepted=accepted, flags=flags, files=files)
        result['original_provider'] = provider
        result['origin_boundary_expected'] = not accepted
        if copy:
            result['sdk_copy'] = {'shadow/cooperative_groups.h': 'cooperative_groups.h'}
        return result

    shadow = {'shadow/cooperative_groups.h': fake}
    yield origin('origin_system_shadow', ('-isystem', '{case}/shadow'), files=shadow)
    yield origin('origin_I_byte_copy', ('-I{case}/shadow',), copy=True)
    yield origin('origin_system_byte_copy', ('-isystem', '{case}/shadow'), copy=True)
    yield origin('origin_explicit_sdk', ('-I{cuda}/include',), accepted=True,
                 provider='{cuda}/include/cooperative_groups.h')
    yield origin('origin_sdk_before_shadow', ('-I{cuda}/include', '-I{case}/shadow'),
                 accepted=False, files=shadow, provider='shadow/cooperative_groups.h')
    yield origin('origin_shadow_before_sdk', ('-I{case}/shadow', '-I{cuda}/include'), files=shadow)
    repeated = base.replace('#include <cooperative_groups.h>\n',
                            '#include <cooperative_groups.h>\n#include <cooperative_groups.h>\n')
    yield origin('origin_repeated_sdk', accepted=True, source=repeated,
                 provider='{cuda}/include/cooperative_groups.h')
    for name, source, flags in [
        ('origin_guard_observation', base + '#ifdef _COOPERATIVE_GROUPS_H_\nint guard_seen;\n#endif\n', ()),
        ('origin_abi_cli', base, ('-D_CG_ABI_VERSION=2',)),
        ('origin_control_alias', '#define CALLER_Q _CG_QUALIFIER\n' + base, ()),
        ('origin_late_restore', base + '#undef _CG_ABI_VERSION\n#define _CG_ABI_VERSION 2\n#undef _CG_ABI_VERSION\n', ()),
        ('origin_raw_inactive', base + '#if 0\n_CG_VERSION\n#endif\n', ()),
        ('origin_stack_restore', '#pragma push_macro("_CG_ABI_VERSION")\n' + base +
         '#pragma pop_macro("_CG_ABI_VERSION")\n', ()),
        ('origin_include_next', base.replace('#include <cooperative_groups.h>',
                                             '#include_next <cooperative_groups.h>'), ()),
        ('origin_has_include', '#if __has_include(<cooperative_groups.h>)\nint cg_available;\n#endif\n' + base, ()),
    ]:
        yield origin(name, flags, source=source, provider='{cuda}/include/cooperative_groups.h')
    yield dict(case('origin_no_profile_control', '#include <cuda_runtime.h>\nint untouched=7;\n',
                    needs=False, qualifiers=()), profile='none')
    # Selection by physical product identity must not bypass the spelling and
    # original-provider contract, including preprocessor forced includes.
    for name, source, flags in [
        ('origin_product_alias', base.replace('#include <cooperative_groups.h>', '#include <H>'), ()),
        ('origin_forced_product_alias', base.replace('#include <cooperative_groups.h>\n', ''),
         ('-include', '{case}/H')),
    ]:
        yield dict(case(name, source, accepted=False, flags=flags),
                   product_alias={'H': 'cooperative_groups.h'}, origin_boundary_expected=True)
    for name, source, flags in [
        ('compiler_default_cli_same_value', base, ('-D_CG_CLUSTER_INTRINSICS_AVAILABLE=1',)),
        ('compiler_default_late_redefine', base +
         '#undef _CG_CLUSTER_INTRINSICS_AVAILABLE\n#define _CG_CLUSTER_INTRINSICS_AVAILABLE 1\n', ()),
        ('compiler_default_observation', base +
         '#ifdef _CG_CLUSTER_INTRINSICS_AVAILABLE\nint cluster_control_seen;\n#endif\n', ()),
    ]:
        yield origin(name, flags, source=source, provider='{cuda}/include/cooperative_groups.h')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def published_sources(output):
    """Follow only this published root's literal local bundle edges, including .dpp.

    Fixture local includes are literal and unconditional. Angle includes belong
    to the separately bound SDK/product inputs; they cannot supply fixture
    qualifications. An unreferenced staged/bundle file is never inspected.
    """
    output = output.absolute()
    bundle = Path(str(output) + '.headers')
    assert output.is_file() and not output.is_symlink(), output
    assert not bundle.is_symlink(), bundle
    result, seen = [], set()

    def visit(path):
        path = path.resolve(strict=True)
        if path in seen:
            return
        seen.add(path)
        result.append(path)
        text = path.read_text()
        for spelling in re.findall(r'(?m)^[ \t]*#[ \t]*include[ \t]*"([^"\r\n]+)"', text):
            relative = Path(spelling)
            assert not relative.is_absolute(), ('absolute published local include', spelling)
            candidate = path.parent / relative
            target = candidate.resolve(strict=True)
            try:
                target.relative_to(bundle.resolve(strict=True))
            except ValueError:
                raise AssertionError(('published include escapes its bundle', spelling, target))
            # Resolving inside the bundle is insufficient if an intermediate
            # component is a symlink. The test requires actual owned artifacts.
            for component in (candidate, *candidate.parents):
                assert not component.is_symlink(), ('symlink in published include', component)
                if component == bundle:
                    break
            assert target.is_file(), ('published include is not a regular file', target)
            visit(target)

    visit(output)
    return result


def launch_inventory(text):
    # Runtime launch configuration may gain whitespace/type constructors.
    # Match the complete literal kernel-template tuple immediately before <<<.
    return Counter((name, int(size), flag == 'true') for name, size, flag in re.findall(
        r'\b(first_kernel|second_kernel)\s*<\s*(\d+)\s*,\s*(false|true)\s*>\s*<<<', text))


def check_all_launches(case, generated):
    original = launch_inventory(case['source'] + '\n' + '\n'.join(case['files'].values()))
    if not original:
        return False
    expected = Counter((name, size, flag)
                       for name in ('first_kernel', 'second_kernel')
                       for size in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
                       for flag in (False, True))
    assert original == expected and sum(original.values()) == 40, ('fixture launch inventory', case['name'])
    actual = launch_inventory(generated)
    assert actual == expected and sum(actual.values()) == 40, (
        case['name'], 'published launch tuples/count changed', expected - actual, actual - expected)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--clang', type=Path, required=True)
    parser.add_argument('--cuda-path', type=Path, required=True)
    parser.add_argument('--resource-dir', type=Path, required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    for key in ('binary', 'clang', 'cuda_path', 'resource_dir'):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    work = args.evidence_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    cccl = args.cuda_path / 'include/cccl'
    cccl_header = cccl / 'cuda/std/type_traits'
    sdk_flags = ['-I' + str(cccl)] if cccl_header.is_file() else []
    key_inputs = [args.binary, args.clang, args.cuda_path / 'include/cuda_runtime.h',
                  args.cuda_path / 'include/cuda_runtime_api.h', args.cuda_path / 'include/cooperative_groups.h',
                  args.resource_dir / 'include/__clang_cuda_runtime_wrapper.h']
    before = {str(p): sha(p) for p in key_inputs}
    if cccl_header.is_file():
        before[str(cccl_header)] = sha(cccl_header)
    summary = dict(schema='ascify.nttp-void.frontend-regression.v1', passed=False,
                   device_executed=False, compiled_target=False, cases=[], inputs=before,
                   binary_sha256=before[str(args.binary)])

    def command(argv, path):
        (path.with_suffix('.argv.json')).write_text(json.dumps(argv, indent=2) + '\n')
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        (path.with_suffix('.stdout.log')).write_text(result.stdout)
        (path.with_suffix('.stderr.log')).write_text(result.stderr)
        return result

    try:
        for case in fixtures():
            directory = work / case['name']
            directory.mkdir()
            source = directory / 'input.cu'
            source.write_text(case['source'])
            for name, text in case['files'].items():
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            for target, original_header in case.get('sdk_copy', {}).items():
                copied = directory / target
                copied.parent.mkdir(parents=True, exist_ok=True)
                copied.write_bytes((args.cuda_path / 'include' / original_header).read_bytes())
            aliases = {}
            for name, role in case.get('product_alias', {}).items():
                target = (args.binary.parent.parent / 'libexec/ascify/frontend-compat' /
                          'ascify-admitted-v1' / role).resolve(strict=True)
                assert target.is_file() and not target.is_symlink(), target
                alias = directory / name
                alias.symlink_to(target)
                aliases[str(alias)] = str(target)
                before[str(target)] = sha(target)
            inputs = {str(p): sha(p) for p in directory.rglob('*') if p.is_file()}
            extra = [flag.replace('{case}', str(directory)).replace('{cuda}', str(args.cuda_path))
                     for flag in case['flags']]
            language = ['-x', 'cuda', '-std=c++17', '-fgpu-rdc',
                        '-I' + str(directory), *sdk_flags, *extra]
            if 'remap_file' in case:
                remapped = directory / case['remap_file']
                assert case['remap_file'] in case['files'] and remapped.is_file()
                language += ['-Xclang', '-remap-file', '-Xclang', str(source) + ';' + str(remapped)]
            syntax = [str(args.clang), '--cuda-host-only', '-nocudalib',
                      '--cuda-path=' + str(args.cuda_path), '-Wno-unknown-cuda-version',
                      '-fsyntax-only', *language, str(source)]
            if 'original_provider' in case:
                syntax += ['-H']
            original = command(syntax, directory / 'original')
            assert original.returncode == 0, (case['name'], 'original syntax invalid', original.stderr)
            if 'original_provider' in case:
                expected_provider = case['original_provider'].replace('{cuda}', str(args.cuda_path))
                expected_provider = Path(expected_provider)
                if not expected_provider.is_absolute():
                    expected_provider = directory / expected_provider
                selected = re.findall(r'(?m)^\.+ (.+[/\\]cooperative_groups\.h)\s*$', original.stderr)
                assert selected and Path(selected[0]).resolve(strict=True) == expected_provider.resolve(strict=True), (
                    case['name'], 'original provider mismatch', selected, str(expected_provider))
            output = directory / 'converted.cce'
            receipt = directory / 'receipt.json'
            convert = [str(args.binary), str(source), '--target-policy=dav-c310-vec',
                       '--target-recipe=none', '--simt-math=precise', '--default-preprocessor',
                       '--frontend-compat=' + case.get('profile', 'ascify-admitted-v1'),
                       '--cuda-path=' + str(args.cuda_path),
                       '--clang-resource-directory=' + str(args.resource_dir),
                       '--migration-receipt=' + str(receipt)]
            if case['recursive'] or 'local.cuh' in case['files']:
                convert.append('--local-headers-recursive')
            result = command(convert + ['-o', str(output), '--', *language], directory / 'convert')
            assert all(sha(Path(p)) == value for p, value in inputs.items()), 'fixture mutated'
            assert all(Path(p).is_symlink() and str(Path(p).readlink()) == target
                       for p, target in aliases.items()), 'fixture alias mutated'
            record = json.loads(receipt.read_text())
            assert record['status'] == ('succeeded' if case['accepted'] else 'failed'), record
            if case['accepted']:
                assert result.returncode == 0 and output.is_file(), (case['name'], result.stderr)
                files = published_sources(output)
                assert all(str(p) not in inputs for p in files), ('original source used as output', files)
                generated = '\n'.join(p.read_text() for p in files)
                all40 = check_all_launches(case, generated)
                for spelling in case['qualifiers']:
                    assert spelling in generated, (case['name'], 'qualification absent', spelling)
                for spelling in case['retained']:
                    projected = spelling.replace('cg::thread_block', 'ascify_cg::thread_block')
                    assert spelling in generated or projected in generated, (
                        case['name'], 'source contract changed', projected)
                if case['needs']:
                    assert 'Ascify closed NTTP void calls v1:' in result.stderr
                else:
                    assert 'Ascify closed NTTP void calls v1:' not in result.stderr
            else:
                assert result.returncode != 0 and not output.exists(), (case['name'], result.stderr)
                # Earlier physical-profile/macro gates may reject before this
                # proof. Original syntax has already succeeded independently.
                assert ('boundary' in result.stderr or 'cooperative' in result.stderr or
                        'profile' in result.stderr), (case['name'], result.stderr)
                if case.get('origin_boundary_expected'):
                    assert 'frontend compatibility original CG provider boundary:' in result.stderr, (
                        case['name'], 'original-provider refusal not exercised', result.stderr)
            summary['cases'].append(dict(name=case['name'], accepted=case['accepted'],
                source_sha256=inputs[str(source)], original_returncode=original.returncode,
                convert_returncode=result.returncode, input_unchanged=True,
                all40_published_launches_verified=all40 if case['accepted'] else False,
                published_sources=[dict(path=str(p), sha256=sha(p)) for p in files] if case['accepted'] else [],
                output_sha256=sha(output) if output.exists() else None))
        assert all(sha(Path(p)) == value for p, value in before.items()), 'tool/SDK input changed'
        summary['passed'] = True
        summary['accepted'] = sum(item['accepted'] for item in summary['cases'])
        summary['source_valid_refusals'] = len(summary['cases']) - summary['accepted']
    finally:
        (work / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
