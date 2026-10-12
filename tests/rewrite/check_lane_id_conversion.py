#!/usr/bin/env python3
"""Source-valid CUDA lane-ID instruction rewrites and retained boundaries."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ASM = 'asm("mov.u32 %0, %%laneid;" : "=r"(value))'
LOWERED = '(value) = ::ascify::cuda_lane_id()'
REGION = re.compile(r'/\*lane_begin\*/(.*?)/\*lane_end\*/', re.S)


@dataclass
class Case:
    name: str
    code: str
    expected: str
    refused: bool = False


def case(name, statement=ASM, expected=None, owner='__device__',
         local='unsigned int value;', prefix='', suffix='', parameters=''):
    return Case(name, prefix + '\n' + owner + ' void sample(' + parameters + ') {\n' +
                local + '\n/*lane_begin*/' + statement + '/*lane_end*/;\n' + suffix + '\n}\n',
                statement if expected is None else expected)


def cases():
    yield case('device_local', expected=LOWERED)
    yield case('global_local', owner='__global__', expected=LOWERED)
    yield case('parenthesized_output', statement=ASM.replace('(value)', '((value))'),
               expected='((value)) = ::ascify::cuda_lane_id()')
    yield case('unsigned_typedef', prefix='using word = unsigned int;',
               local='word value;', expected=LOWERED)
    yield case('builtin_local_shadow', local='unsigned value; int threadIdx = 9, blockDim = 7;',
               expected=LOWERED)
    yield case('laneid_local_shadow', local='unsigned value; int laneid = 99;', expected=LOWERED)
    yield case('same_decoded_literal', statement=ASM.replace('"mov.u32 %0, %%laneid;"',
               '"mov.u32 %0, " "%%laneid;"'), expected=LOWERED)
    yield case('unbraced_if_else', statement='if (flag) ' + ASM + '; else value = 7u',
               expected='if (flag) ' + LOWERED + '; else value = 7u', parameters='bool flag')
    yield case('two_independent_instructions', statement=ASM + '; ' + ASM,
               expected=LOWERED + '; ' + LOWERED)
    yield case('signed_output', local='int value;')
    yield case('wide_output', local='unsigned long long value;')
    yield case('volatile_output', local='volatile unsigned int value;')
    yield case('reference_output', local='unsigned storage; unsigned &value = storage;')
    yield case('parameter_output', parameters='unsigned value', local='')
    yield case('static_output', local='static unsigned int value;')
    yield case('shared_output', local='__shared__ unsigned int value;')
    yield case('global_output', prefix='__device__ unsigned value;', local='')
    yield case('member_output', statement=ASM.replace('(value)', '(object.value)'),
               local='struct S { unsigned value; }; S object;')
    yield case('subscript_output', statement=ASM.replace('(value)', '(values[0])'),
               local='unsigned values[1];')
    yield case('side_effect_output', statement=ASM.replace('(value)', '(values[index++])'),
               local='unsigned values[2]; int index = 0;')
    yield case('dereference_output', statement=ASM.replace('(value)', '(*out)'),
               parameters='unsigned *out', local='')
    yield case('volatile_asm', statement=ASM.replace('asm(', 'asm volatile('))
    yield case('input_operand', statement=ASM[:-1] + ' : "r"(value))', local='unsigned value = 0;')
    yield case('readwrite_constraint', statement=ASM.replace('"=r"', '"+r"'), local='unsigned value = 0;')
    yield case('earlyclobber_constraint', statement=ASM.replace('"=r"', '"=&r"'))
    yield case('named_output', statement=ASM.replace('"=r"', '[output] "=r"'))
    yield case('multiple_outputs', statement=ASM[:-1] + ', "=r"(other))',
               local='unsigned value, other;')
    yield case('memory_clobber', statement=ASM[:-1] + ' : : "memory")')
    yield case('goto_asm', statement=ASM.replace('asm(', 'asm goto(')[:-1] + ' : : : target)',
               suffix='target:;')
    yield case('other_register', statement=ASM.replace('%%laneid', '%%warpid'))
    yield case('another_instruction', statement=ASM.replace('%%laneid;', '%%laneid; add.u32 %0, %0, 1;'))
    yield case('host_function', owner='')
    yield case('host_device_function', owner='__host__ __device__')
    yield case('template_function', owner='template<class T> __device__')
    yield Case('template_class_method', 'template<class T> struct Box { __device__ void sample() {\n'
               'unsigned value;\n/*lane_begin*/' + ASM + '/*lane_end*/;\n} };\n', ASM)
    yield Case('lambda', '__device__ void sample() { auto f = [] __device__ () { unsigned value;\n'
               '/*lane_begin*/' + ASM + '/*lane_end*/; }; f(); }\n', ASM)
    yield case('macro_template', prefix='#define PROJECT_TEMPLATE "mov.u32 %0, %%laneid;"',
               statement=ASM.replace('"mov.u32 %0, %%laneid;"', 'PROJECT_TEMPLATE'))
    yield case('macro_concatenated_template', prefix='#define PROJECT_SUFFIX "%%laneid;"',
               statement=ASM.replace('"mov.u32 %0, %%laneid;"', '"mov.u32 %0, " PROJECT_SUFFIX'))
    yield case('macro_constraint', prefix='#define PROJECT_CONSTRAINT "=r"',
               statement=ASM.replace('"=r"', 'PROJECT_CONSTRAINT'))
    yield case('macro_output', prefix='#define PROJECT_VALUE value',
               statement=ASM.replace('(value)', '(PROJECT_VALUE)'))
    yield case('macro_punctuation', prefix='#define PROJECT_COLON :',
               statement=ASM.replace(' : ', ' PROJECT_COLON '))
    yield case('macro_entire_statement', prefix='#define PROJECT_ASM ' + ASM, statement='PROJECT_ASM')
    for name, prefix in [
        ('helper_macro_collision', '#define cuda_lane_id() 17u'),
        ('helper_macro_undefined', '#define cuda_lane_id() 17u\n#undef cuda_lane_id'),
        ('namespace_macro_collision', '#define ascify project'),
        ('namespace_declaration_collision', 'namespace ascify {}'),
    ]:
        row = case(name, prefix=prefix)
        row.refused = True
        yield row


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--cuda-path', required=True, type=Path)
    parser.add_argument('--resource-dir', required=True, type=Path)
    parser.add_argument('--clang', type=Path)
    parser.add_argument('--evidence-dir', type=Path)
    parser.add_argument('--source-only', action='store_true',
                        help='Validate inputs only; does not claim rewrite or target success')
    args = parser.parse_args()
    binary, cuda, resource = args.binary.resolve(), args.cuda_path.resolve(), args.resource_dir.resolve()
    clang = args.clang.resolve() if args.clang else None
    if clang is None:
        choices = [binary.parent / 'clang++', resource.parents[2] / 'bin/clang++']
        on_path = shutil.which('clang++')
        if on_path:
            choices.append(Path(on_path))
        clang = next((p.resolve() for p in choices if p.is_file()), None)
    if clang is None or not clang.is_file() or not binary.is_file():
        parser.error('supply a real binary and its matching --clang')
    if not (cuda / 'include/cuda_runtime.h').is_file() or not (resource / 'include/__clang_cuda_runtime_wrapper.h').is_file():
        parser.error('supply real CUDA and complete resource headers')
    version = subprocess.run([str(clang), '--version'], text=True, capture_output=True, check=True).stdout
    major = re.search(r'\bclang version (\d+)(?:\.|\s)', version)
    if major is None or major.group(1) != resource.name.split('.')[0]:
        parser.error('clang version must match resource directory')
    with tempfile.TemporaryDirectory(prefix='ascify-lane-id-') as temporary:
        work = args.evidence_dir.resolve() if args.evidence_dir else Path(temporary)
        if args.evidence_dir:
            work.mkdir(parents=True, exist_ok=False)
        summary = {'schema': 'ascify.lane-id-conversion.v1', 'passed': False,
                   'binary': str(binary), 'binary_sha256': sha(binary),
                   'clang': str(clang), 'clang_sha256': sha(clang), 'clang_version': version,
                   'source_only': args.source_only, 'cases': [],
                   'scope': 'real CUDA source syntax and frontend boundaries; no target build or device execution'}
        def save():
            (work / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        def run(directory, label, command):
            (directory / (label + '.argv.json')).write_text(json.dumps(command, indent=2) + '\n')
            result = subprocess.run(command, text=True, capture_output=True, timeout=60)
            (directory / (label + '.stdout.log')).write_text(result.stdout)
            (directory / (label + '.stderr.log')).write_text(result.stderr)
            return result
        save()
        try:
            for item in cases():
                directory = work / item.name
                directory.mkdir()
                source, output, receipt = (directory / name for name in ('input.cu', 'output.cpp', 'receipt.json'))
                source.write_text('#include <cuda_runtime.h>\n' + item.code)
                original = source.read_text()
                regions = REGION.findall(original)
                if len(regions) != 1:
                    raise AssertionError('one statement region required')
                row = {'case': item.name, 'input_sha256': sha(source), 'passed': False,
                       'expected_rewrite': regions[0] != item.expected,
                       'expected_publication_refusal': item.refused}
                summary['cases'].append(row)
                syntax = run(directory, 'syntax', [str(clang), str(source), '-x', 'cuda',
                    '-std=c++17', '-fsyntax-only', '--cuda-host-only', '-nocudalib',
                    '--cuda-path=' + str(cuda), '-resource-dir=' + str(resource), '-Wno-unknown-cuda-version'])
                row['syntax_returncode'] = syntax.returncode
                save()
                if syntax.returncode:
                    raise AssertionError(item.name + ': invalid CUDA source: ' + syntax.stderr)
                if not args.source_only:
                    conversion = run(directory, 'conversion', [str(binary), str(source),
                        '--target-policy=dav-c310-vec', '--simt-math=fast', '--target-recipe=none',
                        '--default-preprocessor', '--cuda-path=' + str(cuda),
                        '--clang-resource-directory=' + str(resource), '--migration-receipt=' + str(receipt),
                        '-o', str(output), '--', '-x', 'cuda', '-std=c++17'])
                    row['conversion_returncode'] = conversion.returncode
                    save()
                    if item.refused:
                        if conversion.returncode == 0 or output.exists():
                            raise AssertionError(item.name + ': unsafe output was published')
                        if 'Ascify cannot publish CUDA compatibility output' not in conversion.stderr:
                            raise AssertionError(item.name + ': unexpected rejection: ' + conversion.stderr)
                    else:
                        if conversion.returncode or json.loads(receipt.read_text())['status'] != 'succeeded':
                            raise AssertionError(item.name + ': conversion failed: ' + conversion.stderr)
                        actual = output.read_text()
                        if REGION.findall(actual) != [item.expected]:
                            raise AssertionError(item.name + ': statement mismatch: ' + repr(REGION.findall(actual)))
                        if actual.count('::ascify::cuda_lane_id()') != item.expected.count('::ascify::cuda_lane_id()'):
                            raise AssertionError(item.name + ': unexpected helper invocation')
                        row['output_sha256'] = sha(output)
                    row['receipt_sha256'] = sha(receipt)
                if sha(source) != row['input_sha256']:
                    raise AssertionError('input source changed')
                row['passed'] = True
                save()
                print(item.name + ': source valid' + ('; rewrite boundary verified' if not args.source_only else ''), flush=True)
            if sha(binary) != summary['binary_sha256'] or sha(clang) != summary['clang_sha256']:
                raise AssertionError('frontend changed during validation')
            summary.update(passed=True, total=len(summary['cases']))
            save()
        except Exception as error:
            summary['failure'] = str(error)
            save()
            raise
        print('CUDA lane ID: ' + str(summary['total']) + ' cases passed' + (' (source only)' if args.source_only else ''))


if __name__ == '__main__':
    main()
