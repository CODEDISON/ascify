#!/usr/bin/env python3
"""Syntax-first host pointer template domains; no CCEC, link, ACL or device.

Each refusal is tested against fresh and existing output paths. Accepted
emitted guards are then isolated verbatim and compiled with both their proved
domain and future SDK/wrong builtin pointer instances.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

PREFIX = '#include <cuda_runtime_api.h>\n#include <cuda_runtime.h>\n'
SWAP = 'template<class T> inline void swap_values(T &a,T &b){T temporary=a;a=b;b=temporary;}\n'
CALL = 'void use(float *p,float *q){swap_values(p,q);}\n'
GUARD_MESSAGE = 'host pointer swap is only proven for the emitted pointer type domain'
EMPTY_GUARD_MESSAGE = 'host pointer swap has no proven body instance in this translation unit'


def fixtures():
    def case(name, body=SWAP + CALL, accepted=False, files=None, recursive=False, flags=(), domain=('float *',)):
        return dict(name=name, source=PREFIX + body, accepted=accepted, files=files or {},
                    recursive=recursive, flags=list(flags), domain=list(domain))
    yield case('float_pointer', accepted=True)
    yield case('renamed', (SWAP + CALL).replace('swap_values', 'rotate_addresses').replace(
        'temporary', 'saved').replace('class T', 'typename Element').replace('T &', 'Element &').replace(
        'T saved', 'Element saved'), accepted=True)
    yield case('two_builtin_pointer_types', SWAP + CALL +
               'void integer_use(int *p,int *q){swap_values(p,q);}\n', accepted=True, domain=('float *', 'int *'))
    yield case('new_and_sdk_allocation', SWAP +
               'void use(){float *p=new float[4];float *q;cudaMalloc(&q,16);swap_values(p,q);}\n', accepted=True)
    yield case('fake_system_allocator_signature', '#include <fake_allocator.h>\n' + SWAP +
               'void use(float *p,float *q){cudaMalloc((void**)&p,16);swap_values(p,q);}\n',
               files={'fake_allocator.h': '#pragma GCC system_header\ncudaError_t cudaMalloc(void **p,int n);\n'}, flags=['isystem'])
    yield case('caller_allocator_redeclaration', 'extern "C" __host__ cudaError_t cudaMalloc(void **p,size_t n);\n' + SWAP +
               'void use(float *p,float *q){cudaMalloc((void**)&p,size_t(16));swap_values(p,q);}\n')
    yield case('selected_header_joint', '#include "values.h"\n' + CALL,
               accepted=True, files={'values.h': '#ifndef VALUES_H\n#define VALUES_H\n' + SWAP + '#endif\n'}, recursive=True)
    yield case('unselected_header', '#include "values.h"\n' + CALL, files={'values.h': SWAP})
    yield case('zero_instance_main', SWAP, accepted=True, domain=())
    yield case('zero_instance_header', '#include "values.h"\nint retained;\n',
               accepted=True, files={'values.h': SWAP}, recursive=True, domain=())
    yield case('zero_using_export', SWAP + 'namespace other{using ::swap_values;}\n')
    yield case('zero_unevaluated_signature', SWAP + 'using Signature=decltype(&swap_values<float *>);\n')
    yield case('zero_address_escape', SWAP + 'auto escaped=&swap_values<float *>;\n')
    yield case('zero_default_template_argument', SWAP + 'template<class F=decltype(&swap_values<float *>)>struct Later{};\n')
    yield case('zero_extern_instantiation', SWAP + 'extern template void swap_values<float *>(float *&,float *&);\n')
    yield case('zero_unresolved_future_call', SWAP + 'template<class U>void future(U &a,U &b){swap_values(a,b);}\n')
    yield case('proof_then_SDK_inquiry', SWAP + CALL + 'unsigned bytes=sizeof(cudaDeviceProp);\n')
    yield case('header_proof_then_SDK_inquiry', '#include "values.h"\n' + CALL +
               'unsigned bytes=sizeof(cudaDeviceProp);\n', files={'values.h': SWAP}, recursive=True)
    for name, declaration in [
        ('SDK_record', 'cudaDeviceProp'), ('SDK_pointee', 'cudaDeviceProp *'),
        ('SDK_enum_pointee', 'cudaError_t *'), ('SDK_scalar_alias_pointee', 'cudaTextureObject_t *'),
        ('void_pointer', 'void *'), ('nested_pointer', 'float **'),
    ]:
        yield case(name, SWAP + 'void use(){' + declaration + ' p{};' + declaration + ' q{};swap_values(p,q);}\n')
    yield case('caller_leaf_alias', SWAP + 'using Scalar=float;void use(Scalar *p,Scalar *q){swap_values(p,q);}\n')
    yield case('caller_pointer_alias', SWAP + 'using Pointer=float *;void use(Pointer p,Pointer q){swap_values(p,q);}\n')
    yield case('explicit_alias_argument', SWAP + 'using Pointer=float *;void use(float *p,float *q){swap_values<Pointer>(p,q);}\n')
    yield case('SDK_record_wrapper', 'struct Wrapper{cudaDeviceProp property;};\n' + SWAP +
               'void use(Wrapper *p,Wrapper *q){swap_values(p,q);}\n')
    yield case('cast_erased_SDK_initializer', SWAP +
               'void use(){cudaDeviceProp value;float *p=reinterpret_cast<float*>(&value);float *q=nullptr;swap_values(p,q);}\n')
    yield case('cast_erased_SDK_assignment', SWAP +
               'void use(){cudaDeviceProp value;float *p;float *q=nullptr;p=reinterpret_cast<float*>(&value);swap_values(p,q);}\n')
    yield case('pointer_reference_alias', SWAP +
               'void use(float *p,float *q){float *&alias=p;swap_values(p,q);}\n')
    yield case('pointer_reference_aggregate_wrapper', SWAP +
               'struct Alias{float *&value;};void use(float *p,float *q){Alias alias{p};swap_values(p,q);}\n')
    yield case('pointer_address_escape', 'void change(void **);\n' + SWAP +
               'void use(float *p,float *q){change((void**)&p);swap_values(p,q);}\n')
    yield case('pointer_reference_escape', 'void change(float *&);\n' + SWAP +
               'void use(float *p,float *q){change(p);swap_values(p,q);}\n')
    yield case('pointer_cast_reference_escape', 'void change(void *&);\n' + SWAP +
               'void use(float *p,float *q){change(reinterpret_cast<void *&>(p));swap_values(p,q);}\n')
    yield case('pointer_cast_reference_assignment', SWAP +
               'void use(float *p,float *q){cudaDeviceProp value;reinterpret_cast<void *&>(p)=&value;swap_values(p,q);}\n')
    yield case('template_address_escape', SWAP + CALL + 'auto escaped=&swap_values<float *>;\n')
    yield case('redeclared_primary', 'template<class T>void swap_values(T&,T&);\n' + SWAP + CALL)
    yield case('explicit_specialization', SWAP + 'template<>inline void swap_values<float *>(float *&a,float *&b){}\n' + CALL)
    yield case('explicit_instantiation', SWAP + 'template void swap_values<float *>(float *&,float *&);\n' + CALL)
    yield case('default_template_argument', SWAP.replace('class T', 'class T=float *') + CALL)
    yield case('future_uninstantiated_caller', SWAP + CALL + 'template<class U>void later(U &a,U &b){swap_values(a,b);}\n')
    yield case('macro_parameter_restore', '#define T Rebound\n' + SWAP + '#undef T\n' + CALL)
    yield case('macro_builtin_spelling', SWAP + '#define Scalar float\nvoid use(Scalar *p,Scalar *q){swap_values(p,q);}\n#undef Scalar\n')
    yield case('macro_guard_trait_restore', '#define __is_same(a,b) true\n' + SWAP + '#undef __is_same\n' + CALL)
    yield case('macro_guard_assert_restore', '#define static_assert(...)\n' + SWAP + '#undef static_assert\n' + CALL)
    for name, observation in [
        ('sizeof_T', '(void)sizeof(T);'), ('alignof_T', '(void)alignof(T);'),
        ('offsetof_SDK', '(void)__builtin_offsetof(cudaDeviceProp,warpSize);'),
        ('trait_T', '(void)__is_trivial(T);'), ('SDK_member', 'cudaDeviceProp p{};(void)p.warpSize;'),
        ('future_branch', 'if constexpr(__is_same(T,float *)){}else{(void)sizeof(cudaDeviceProp);}'),
    ]:
        yield case(name, SWAP.replace('b=temporary;', 'b=temporary;' + observation) + CALL)
    yield case('inactive_future_branch', SWAP.replace('T temporary', '\n#if 0\n(void)sizeof(cudaDeviceProp);\n#endif\nT temporary') + CALL)
    yield case('unselected_system_header', '#include <values.h>\n' + CALL,
               files={'values.h': SWAP}, flags=['isystem'])
    yield case('remapped_root', files={'replacement.cu': PREFIX + SWAP + CALL}, flags=['remap'])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emitted_template(text, empty=False):
    match = re.search(r'template\s*<\s*(?:class|typename)\s+(\w+)\s*>\s*inline\s+void\s+(\w+)\s*\([^)]*\)\s*\{', text)
    assert match, 'emitted exact swap pattern missing'
    end = text.find('}', match.end())
    assert end >= 0
    result = text[match.start():end + 1] + '\n'
    assert (EMPTY_GUARD_MESSAGE if empty else GUARD_MESSAGE) in result and result.count('static_assert(') == 1
    if empty:
        assert '!__is_same(' + match.group(1) + ', ' + match.group(1) + ')' in result
    return result, match.group(2)


def main():
    if not __debug__:
        raise RuntimeError('optimized Python refused')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['binary', 'clang', 'cuda-path', 'resource-dir', 'evidence-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--case', action='append', default=[])
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    identities = {str(p.resolve()): sha(p) for p in [Path(__file__), args.binary, args.clang]}
    summary = dict(passed=False, identities=identities, cases=[], device_executed=False,
                   CCEC_invocations=0, original_sample_successes=0)
    def save():
        (args.evidence_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    def run(folder, label, argv):
        argv = list(map(str, argv))
        (folder / (label + '.argv.json')).write_text(json.dumps(argv, indent=2) + '\n')
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        for kind in ['stdout', 'stderr']:
            (folder / (label + '.' + kind + '.log')).write_text(getattr(result, kind))
        (folder / (label + '.process.json')).write_text(json.dumps({'returncode': result.returncode}) + '\n')
        return result
    def syntax(folder, label, source, flags):
        return run(folder, label, [args.clang, '--cuda-host-only', '-nocudalib',
            '--cuda-path=' + str(args.cuda_path), '-resource-dir=' + str(args.resource_dir),
            '-Wno-unknown-cuda-version', '-fsyntax-only', *flags, source])
    save()
    try:
        for case in fixtures():
            if args.case and case['name'] not in args.case:
                continue
            folder = args.evidence_dir / case['name'];folder.mkdir()
            source = folder / 'input.cu';source.write_text(case['source'])
            for name, text in case['files'].items(): (folder / name).write_text(text)
            inputs = {str(p): sha(p) for p in folder.iterdir() if p.is_file()}
            flags = ['-x', 'cuda', '-std=c++17', '-I' + str(folder)]
            if 'isystem' in case['flags']:
                flags = ['-x', 'cuda', '-std=c++17', '-isystem', str(folder)]
            if 'remap' in case['flags']:
                flags += ['-Xclang', '-remap-file', '-Xclang', str(source) + ';' + str(folder / 'replacement.cu')]
            row = dict(name=case['name'], accepted=case['accepted'], passed=False, inputs=inputs)
            summary['cases'].append(row);save()
            original = syntax(folder, 'original_syntax', source, flags)
            assert original.returncode == 0, (case['name'], 'invalid original CUDA', original.stderr)
            for mode in (['fresh'] if case['accepted'] else ['fresh', 'existing']):
                output = folder / (mode + '.cce');receipt = folder / (mode + '.receipt.json')
                sentinel = 'EXISTING_UNCHANGED_' + case['name'] + '\n'
                if mode == 'existing': output.write_text(sentinel)
                converted = run(folder, mode, [args.binary, source, '--target-policy=dav-c310-vec',
                    '--target-recipe=none', '--simt-math=fast', '--default-preprocessor',
                    '--frontend-compat=ascify-admitted-v1', '--cuda-path=' + str(args.cuda_path),
                    '--clang-resource-directory=' + str(args.resource_dir), '--migration-receipt=' + str(receipt),
                    '-o', output, *(['--local-headers-recursive'] if case['recursive'] else []), '--', *flags])
                report = json.loads(receipt.read_text())
                if not case['accepted']:
                    assert converted.returncode > 0 and report['status'] == 'failed', (case['name'], converted.stderr)
                    assert output.read_text() == sentinel if mode == 'existing' else not output.exists()
                    assert not Path(str(output) + '.headers').exists()
                    assert re.search(r'host pointer-swap boundary|runtime-API header boundary|contextual.*(?:could not|requires)', converted.stderr), converted.stderr
                    continue
                assert converted.returncode == 0 and report['status'] == 'succeeded', (case['name'], converted.stderr)
                texts = [output.read_text()] + [p.read_text() for p in Path(str(output) + '.headers').rglob('*') if p.is_file()]
                empty = not case['domain']
                template, name = emitted_template('\n'.join(texts), empty)
                assert all('__is_same(' in template and ', ' + domain + ')' in template for domain in case['domain'])
                row['emitted_template_sha256'] = hashlib.sha256(template.encode()).hexdigest()
                if empty:
                    probe = folder / 'empty_primary_uninstantiated.cu'
                    probe.write_text(PREFIX + template + 'int retained;\n')
                    assert syntax(folder, 'empty_primary_uninstantiated', probe, ['-x', 'cuda', '-std=c++17']).returncode == 0
                for label, dtype, accepted in [(('guard_domain_' + str(i)), dtype, True) for i, dtype in enumerate(case['domain'])] + [
                    ('guard_future_SDK_record', 'cudaDeviceProp *', False),
                    ('guard_future_SDK_enum', 'cudaError_t *', False),
                    ('guard_future_SDK_scalar_alias', 'cudaTextureObject_t *', False),
                    ('guard_future_void_pointer', 'void *', False),
                    ('guard_future_unobserved_builtin', 'short *', False)]:
                    probe = folder / (label + '.cu')
                    probe.write_text(PREFIX + template + 'void future(){' + dtype + ' a=nullptr;' + dtype +
                                     ' b=nullptr;' + name + '(a,b);}\n')
                    result = syntax(folder, label, probe, ['-x', 'cuda', '-std=c++17'])
                    assert (result.returncode == 0) is accepted, (case['name'], label, result.stderr)
                    if not accepted:
                        assert (EMPTY_GUARD_MESSAGE if empty else GUARD_MESSAGE) in result.stderr, result.stderr
            assert all(sha(Path(name)) == digest for name, digest in inputs.items())
            row['passed'] = True;save()
        assert summary['cases'] and all(sha(Path(name)) == digest for name, digest in identities.items())
        summary.update(passed=True, case_count=len(summary['cases']))
    except Exception as error:
        summary['error'] = str(error);raise
    finally:
        save()


if __name__ == '__main__':
    main()
