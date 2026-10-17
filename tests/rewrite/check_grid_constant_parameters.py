#!/usr/bin/env python3
"""Exercise the grid-constant value-only proof with real CUDA ASTs.

Every case must first pass the selected CUDA frontend. Negative cases require
the precise semantic boundary diagnostic and no published output or closure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


RECORD = 'struct Packet { int items[17]; };\n'
SIGNATURE = '__global__ void dispatch(__grid_constant__ const Packet p, int *out)'
BODY = '{ int total = 0; for (int i=0; i<17; ++i) total += p.items[i]; out[0] = total; }\n'
KERNEL = SIGNATURE + BODY


def cases():
    def case(name, text, accepted, **extra):
        return dict(name=name, text=text, accepted=accepted, **extra)
    values = [
        case('scalar_reads', RECORD+KERNEL, True),
        case('one_element', (RECORD+KERNEL).replace('17', '1'), True),
        case('large_value', (RECORD+KERNEL).replace('17', '8000'), True),
        case('renamed_alias', (RECORD+'using Alias=Packet;\n'+KERNEL.replace('const Packet', 'const Alias')).replace('items', 'payload'), True),
        case('parenthesized_read', RECORD+KERNEL.replace('p.items[i]', '(((p).items)[i])'), True),
        case('scalar_argument', RECORD+'__device__ int twice(int x) { return 2*x; }\n'+KERNEL.replace('p.items[i]', 'twice(p.items[i])'), True),
        case('forward_declaration', RECORD+SIGNATURE+';\n'+KERNEL, True),
        case('two_value_parameters', RECORD+SIGNATURE.replace('int *out', '__grid_constant__ const Packet q, int *out')+BODY.replace('p.items[i]', 'p.items[i]+q.items[i]'), False),
        case('payload_last', RECORD+SIGNATURE.replace('__grid_constant__ const Packet p, int *out', 'unsigned before, int *out, unsigned after, __grid_constant__ const Packet p')+BODY, True),
        case('guarded_large_payload', (RECORD+SIGNATURE.replace('int *out', 'unsigned before, int *out, unsigned after')+BODY).replace('17', '8000'), True),
        case('launch_retained', RECORD+KERNEL+'void launch(Packet p, int *out) { dispatch<<<2,4>>>(p,out); }\n', True),
        case('parameter_address', RECORD+KERNEL.replace('int total = 0;', 'const Packet *escaped = &p; int total = escaped->items[0];'), False),
        case('element_address', RECORD+KERNEL.replace('int total = 0;', 'const int *escaped = &p.items[0]; int total = *escaped;'), False),
        case('array_decay', RECORD+'__device__ int read(const int *p) { return p[0]; }\n'+KERNEL.replace('int total = 0;', 'int total = read(p.items);'), False),
        case('parameter_reference', RECORD+'__device__ int read(const Packet &p) { return p.items[0]; }\n'+KERNEL.replace('int total = 0;', 'int total = read(p);'), False),
        case('element_reference', RECORD+'__device__ int read(const int &p) { return p; }\n'+KERNEL.replace('int total = 0;', 'int total = read(p.items[0]);'), False),
        case('whole_copy', RECORD+KERNEL.replace('int total = 0;', 'Packet copy = p; int total = copy.items[0];'), False),
        case('lambda_capture', RECORD+KERNEL.replace('int total = 0;', 'auto copy = [p] { return p.items[0]; }; int total = copy();'), False),
        case('sizeof_record', RECORD+KERNEL.replace('int total = 0;', 'int total = sizeof(p);'), False),
        case('sizeof_read', RECORD+KERNEL.replace('int total = 0;', 'int total = sizeof(p.items[0]+0);'), False),
        case('decltype_record', RECORD+KERNEL.replace('int total = 0;', 'using Observed=decltype(p); int total = sizeof(Observed);'), False),
        case('noexcept_read', RECORD+KERNEL.replace('int total = 0;', 'int total = noexcept(p.items[0]+0);'), False),
        case('extra_field', RECORD.replace('int items[17];', 'int items[17]; int extra;')+KERNEL, False),
        case('float_array', (RECORD+KERNEL).replace('int items[17]', 'float items[17]'), False),
        case('volatile_elements', RECORD.replace('int items', 'volatile int items')+KERNEL, False),
        case('mutable_field', RECORD.replace('int items', 'mutable int items')+KERNEL, False),
        case('overaligned_record', RECORD.replace('struct Packet', 'struct alignas(64) Packet')+KERNEL, False),
        case('record_method', RECORD.replace('int items[17];', 'int items[17]; __device__ int first() const { return items[0]; }')+KERNEL, False),
        case('template_kernel', RECORD+'template<int N>\n'+KERNEL, False),
        case('declaration_only', RECORD+SIGNATURE+';\n', False),
        case('alias_qualifier', '#define GRID __grid_constant__\n'+RECORD+KERNEL.replace('__grid_constant__', 'GRID'), False),
        case('direct_attribute', RECORD+KERNEL.replace('__grid_constant__', '__attribute__((grid_constant))'), False),
        case('user_qualifier', '#undef __grid_constant__\n#define __grid_constant__ __attribute__((grid_constant))\n'+RECORD+KERNEL, False),
        case('user_dependency', '#undef __location__\n#define __location__(a) __attribute__((a))\n'+RECORD+KERNEL, False),
        case('redecl_alias', '#define GRID __grid_constant__\n'+RECORD+SIGNATURE.replace('__grid_constant__', 'GRID')+';\n'+KERNEL, False),
        case('header_declaration', '#include "declaration.cuh"\n'+KERNEL, False, headers={'declaration.cuh': RECORD+SIGNATURE+';\n'}),
        case('above_validated_bound', (RECORD+KERNEL).replace('17', '8001'), False),
        case('field_macro', '#define FIELD items\n'+RECORD.replace('items', 'FIELD')+KERNEL, False),
        case('volatile_parameter', RECORD+KERNEL.replace('const Packet', 'const volatile Packet'), False),
        case('redecl_restored_macro', RECORD+'#pragma push_macro("__grid_constant__")\n#undef __grid_constant__\n#define __grid_constant__ __attribute__((grid_constant))\n'+SIGNATURE+';\n#pragma pop_macro("__grid_constant__")\n'+KERNEL, False),
        case('header_macro_reentry', RECORD+'#pragma push_macro("__grid_constant__")\n#include "attribute.cuh"\n'+SIGNATURE+';\n#pragma pop_macro("__grid_constant__")\n#include "attribute.cuh"\n'+KERNEL, False, headers={'attribute.cuh': '#undef __grid_constant__\n#define __grid_constant__ __attribute__((grid_constant))\n'}),
        case('nonconst_invalid_cuda', RECORD+KERNEL.replace('const Packet', 'Packet'), False, syntax_valid=False),
        case('additional_aggregate', RECORD+SIGNATURE.replace('int *out', 'Packet second, int *out')+BODY, False),
        case('additional_wide_scalar', RECORD+SIGNATURE.replace('int *out', 'long long extra, int *out')+BODY, False),
        case('extra_pointer_shape', (RECORD+SIGNATURE.replace('int *out', 'int *first, int *out, int *last')+BODY).replace('17', '8000'), False),
        case('large_payload_last', (RECORD+SIGNATURE.replace('__grid_constant__ const Packet p, int *out', 'unsigned before, int *out, unsigned after, __grid_constant__ const Packet p')+BODY).replace('17', '8000'), True),
        case('payload_in_middle', RECORD+SIGNATURE.replace('__grid_constant__ const Packet p, int *out', 'unsigned before, __grid_constant__ const Packet p, int *out')+BODY, False),
        case('header_definition_recursive', '#include "kernel.cuh"\n', False, headers={'kernel.cuh': RECORD+KERNEL}),
        case('header_definition_nonrecursive', '#include "kernel.cuh"\n', False, headers={'kernel.cuh': RECORD+KERNEL}, recursive=False),
        case('redecl_volatile_qualifier', RECORD+SIGNATURE.replace('const Packet', 'const volatile Packet')+';\n'+KERNEL, False),
        case('too_many_parameters', RECORD+SIGNATURE.replace('int *out', 'int a, int b, int c, int *out')+BODY, False),
    ]
    return values


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, directory, timeout):
    directory.mkdir()
    (directory/'argv.json').write_text(json.dumps(argv, indent=2)+'\n')
    start = time.time()
    with (directory/'stdout.txt').open('wb') as out, (directory/'stderr.txt').open('wb') as err:
        process = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    result = dict(returncode=process.returncode, timed_out=timed_out, elapsed_seconds=time.time()-start)
    (directory/'process.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', default=os.environ.get('ASCIFY_BINARY'))
    ap.add_argument('--clang', default=os.environ.get('CLANG', 'clang++'))
    ap.add_argument('--cuda', default=os.environ.get('ASCIFY_CUDA_PATH'), required=not os.environ.get('ASCIFY_CUDA_PATH'))
    ap.add_argument('--resource', default=os.environ.get('ASCIFY_CLANG_RESOURCE_DIRECTORY'), required=not os.environ.get('ASCIFY_CLANG_RESOURCE_DIRECTORY'))
    ap.add_argument('--evidence-dir', type=Path, required=True)
    ap.add_argument('--syntax-only', action='store_true')
    args = ap.parse_args()
    if sys.flags.optimize:
        raise SystemExit('Python optimization is forbidden')
    if not args.syntax_only and not args.binary:
        ap.error('--binary required for conversion')
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(schema_version=1, script_sha256=sha(__file__), binary_sha256=sha(args.binary) if args.binary else None,
                   clang_sha256=sha(args.clang), syntax_only=args.syntax_only, cases=[], passed=False)
    for spec in cases():
        directory = args.evidence_dir/spec['name']
        directory.mkdir()
        source = directory/'source.cu'
        source.write_text('#include <cuda_runtime.h>\n'+spec['text'])
        for name, text in spec.get('headers', {}).items():
            (directory/name).write_text(text)
        inputs = {p.name: sha(p) for p in directory.iterdir() if p.is_file()}
        syntax = run([args.clang, '-x', 'cuda', '--cuda-host-only', '--cuda-gpu-arch=sm_80', '-nocudalib',
                      '--cuda-path='+args.cuda, '-resource-dir='+args.resource, '-Wno-unknown-cuda-version',
                      '-std=c++17', '-fsyntax-only', str(source)], directory/'syntax', 60)
        result = dict(name=spec['name'], expected_acceptance=spec['accepted'], inputs_sha256=inputs, syntax=syntax, passed=False)
        if not spec.get('syntax_valid', True):
            passed = (syntax['returncode'] != 0 and not syntax['timed_out'] and
                      'grid_constant' in (directory/'syntax/stderr.txt').read_text())
            result.update(status='original_cuda_rejection' if passed else 'invalid_syntax_expectation', passed=passed)
        elif syntax['returncode'] or syntax['timed_out']:
            result['status'] = 'invalid_source_or_frontend_environment'
        elif args.syntax_only:
            result.update(status='source_valid', passed=True)
        else:
            output = directory/'output.cce'
            converted = run([args.binary, str(source), '--target-policy=dav-c310-vec', '--default-preprocessor',
                             *(['--local-headers-recursive'] if spec.get('recursive', True) else []), '--cuda-path='+args.cuda,
                             '--clang-resource-directory='+args.resource, '-o', str(output), '--', '-x', 'cuda',
                             '--cuda-gpu-arch=sm_80', '-std=c++17'], directory/'convert', 90)
            result['convert'] = converted
            diagnostic = (directory/'convert/stderr.txt').read_text()
            if spec['accepted']:
                generated = output.read_text() if output.exists() else ''
                # The value-only proof must preserve this original kernel body
                # byte for byte. Other existing rewrites still handle CUDA launch.
                body = spec['text'][spec['text'].find('{ int total'):]
                body = body[:body.find('}\n')+2] if body.startswith('{ int total') else None
                spelling = 'const Alias p' if spec['name'] == 'renamed_alias' else 'const Packet p'
                passed = (converted['returncode'] == 0 and not converted['timed_out'] and output.exists()
                          and 'grid-constant value-only specialization:' in diagnostic
                          and '__grid_constant__ const' not in generated
                          and spelling in generated)
                if body:
                    passed = passed and body in generated
                if spec['name'] == 'launch_retained':
                    passed = passed and '<<<2,4>>>(p,out)' in generated
                result.update(passed=passed, status='admitted' if passed else 'unexpected_rejection_or_changed_source')
                if output.exists(): result['output_sha256'] = sha(output)
            else:
                passed = (converted['returncode'] != 0 and not converted['timed_out']
                          and 'Ascify grid-constant boundary:' in diagnostic
                          and not output.exists() and not Path(str(output)+'.headers').exists())
                result.update(passed=passed, status='semantic_rejection' if passed else 'unexpected_admission_or_unrelated_failure')
        if {name: sha(directory/name) for name in inputs} != inputs:
            result.update(passed=False, status='input_mutation')
        summary['cases'].append(result)
        (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        (args.evidence_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(spec['name']+': '+result['status'], flush=True)
    summary['passed'] = all(c['passed'] for c in summary['cases'])
    summary['total'] = len(summary['cases'])
    summary['expected_positive'] = sum(c['expected_acceptance'] for c in summary['cases'])
    (args.evidence_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
