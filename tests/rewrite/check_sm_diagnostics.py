#!/usr/bin/env python3
"""Real CUDA source proofs for display-only CUDA SM diagnostics on Ascend."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Optional

PRINT = 'printf("SM %d.%d\\n", p.major, p.minor)'
REWRITE = 'printf("SM N/A (Ascend)\\012")'
REGION = re.compile(r'/\*lane_begin\*/(.*?)/\*lane_end\*/', re.S)

@dataclass
class Case:
    name: str
    code: str
    expected: str
    refused: bool = False
    dead_region: Optional[str] = None

def case(name, statement=PRINT, expected=None, local='cudaDeviceProp p{};',
         owner='void', params='', prefix='', suffix=''):
    return Case(name, prefix + '\n' + owner + ' sample(' + params + ') {\n' + local +
                '\n/*lane_begin*/' + statement + '/*lane_end*/;\n' + suffix + '\n}\n',
                statement if expected is None else expected)

def cases(clang_major):
    yield case('plain', expected=REWRITE)
    yield case('other_arguments', statement='printf("name %s has %d MP, SM %d.%d\\n", p.name, p.multiProcessorCount, p.major, p.minor)',
               expected='printf("name %s has %d MP, SM N/A (Ascend)\\012", p.name, p.multiProcessorCount)')
    yield case('parenthesized_fields', statement=PRINT.replace('p.major, p.minor','(p.major), (p.minor)'), expected=REWRITE)
    yield case('typedef_record', prefix='using Props = cudaDeviceProp;', local='Props p{};', expected=REWRITE)
    yield case('const_object', local='const cudaDeviceProp p{};', expected=REWRITE)
    yield case('concatenated_literal', statement=PRINT.replace('"SM %d.%d\\n"','"SM " "%d.%d\\n"'), expected=REWRITE)
    yield case('escaped_percent', statement=PRINT.replace('SM ', '100%% SM '), expected=REWRITE.replace('SM ', '100%% SM '))
    yield case('capability_label', statement=PRINT.replace('SM ', 'compute capability '), expected=REWRITE.replace('SM ', 'compute capability '))
    yield case('compute_label', statement=PRINT.replace('SM ', 'Compute '), expected=REWRITE.replace('SM ', 'Compute '))
    attribute_local = ('int device=0, major=0, minor=0;\n'
        'cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);\n'
        'cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);')
    attribute_print = 'printf("compute capability %d.%d\\n", major, minor)'
    attribute_output = 'printf("compute capability N/A (Ascend)\\012")'
    yield case('attribute_locals', local=attribute_local, statement=attribute_print, expected=attribute_output)
    yield case('attribute_literal_device', local=attribute_local.replace(', device)', ', 0)'), statement=attribute_print, expected=attribute_output)
    yield case('attribute_renamed_locals', local=attribute_local.replace('major','first').replace('minor','second'),
               statement=attribute_print.replace('major','first').replace('minor','second'), expected=attribute_output)
    yield case('attribute_extra_read', local=attribute_local+' int observed=major;', statement=attribute_print)
    yield case('attribute_extra_write', local=attribute_local+' major=5;', statement=attribute_print)
    yield case('attribute_escape', prefix='void escape(int *);', local=attribute_local+' escape(&major);', statement=attribute_print)
    yield case('attribute_capture', local=attribute_local+' auto read=[&major] { return major; };', statement=attribute_print)
    yield case('attribute_wrong_kind', local=attribute_local.replace('cudaDevAttrComputeCapabilityMajor','cudaDevAttrWarpSize'), statement=attribute_print)
    yield case('attribute_other_device', local=attribute_local.replace('minor, cudaDevAttrComputeCapabilityMinor, device','minor, cudaDevAttrComputeCapabilityMinor, 1'), statement=attribute_print)
    yield case('attribute_duplicate_query', local=attribute_local+' cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);', statement=attribute_print)
    yield case('attribute_nested_query', local=attribute_local.replace('cudaDeviceGetAttribute(&major','if (device==0) cudaDeviceGetAttribute(&major'), statement=attribute_print)
    yield case('attribute_return_used', local=attribute_local.replace('cudaDeviceGetAttribute(&major','auto status=cudaDeviceGetAttribute(&major'), statement=attribute_print)
    yield case('attribute_plain_numbers', local='int major=9, minor=0;', statement=attribute_print)
    dead = 'p.major=0; p.minor=0;'
    def stores(name, after='', before='', retained=False):
        item = case(name, local='cudaDeviceProp p{}; '+before+'\n/*dead_begin*/'+dead+'/*dead_end*/\n'+after,
                    expected=REWRITE)
        item.dead_region = dead if retained else '(void)0; (void)0;'
        return item
    yield stores('dead_initial_stores', after='cudaGetDeviceProperties(&p, 0);')
    yield stores('dead_stores_no_query')
    yield stores('retain_store_numeric_read', after='int number=p.major;', retained=True)
    yield stores('retain_store_record_copy', after='auto copied=p;', retained=True)
    yield stores('retain_store_reference', after='auto &ref=p;', retained=True)
    yield stores('retain_store_lambda_capture', after='auto copy=[p] {};', retained=True)
    yield stores('retain_store_field_address', after='int *ptr=&p.multiProcessorCount;', retained=True)
    yield stores('retain_store_property_address', after='auto *ptr=&p;', retained=True)
    yield stores('retain_store_member_reference', after='auto &ref=p.major;', retained=True)
    yield stores('retain_store_other_display', after='printf("%d",p.major);', retained=True)
    item = stores('retain_store_nonzero', retained=True)
    item.code = item.code.replace('p.major=0;', 'p.major=1;')
    item.dead_region = 'p.major=1; p.minor=0;'
    yield item
    yield case('read_return_value', statement='int written = '+PRINT)
    yield case('return_value', owner='int', statement='return '+PRINT)
    yield case('conditional_call', statement='if (true) '+PRINT)
    yield case('comma_expression', statement='('+PRINT+', 0)')
    yield case('parameter_object', local='', params='cudaDeviceProp p')
    yield case('static_object', local='static cudaDeviceProp p{};')
    yield case('thread_local_object', local='thread_local cudaDeviceProp p{};')
    yield case('global_object', local='', prefix='cudaDeviceProp p{};')
    yield case('reference_object', local='cudaDeviceProp value{}; cudaDeviceProp &p=value;')
    yield case('volatile_object', local='volatile cudaDeviceProp p{};')
    yield case('arrow_object', local='cudaDeviceProp value{}; cudaDeviceProp *p=&value;', statement=PRINT.replace('p.', 'p->'))
    yield case('two_objects', local='cudaDeviceProp p{}, other{};', statement=PRINT.replace('p.minor', 'other.minor'))
    yield case('user_record', prefix='namespace project { struct cudaDeviceProp { int major, minor; }; }', local='project::cudaDeviceProp p{};')
    yield case('template_owner', owner='template<class T> void')
    yield case('device_owner', owner='__device__ void')
    yield case('global_owner', owner='__global__ void')
    yield case('lambda_owner', statement='auto f = [&]() { '+PRINT+'; }; f()')
    yield case('macro_format', prefix='#define FORMAT "SM %d.%d\\n"', statement=PRINT.replace('"SM %d.%d\\n"','FORMAT'))
    yield case('macro_suffix', prefix='#define SUFFIX "%d.%d\\n"', statement=PRINT.replace('"SM %d.%d\\n"','"SM " SUFFIX'))
    yield case('macro_member', prefix='#define MAJOR major', statement=PRINT.replace('p.major','p.MAJOR'))
    yield case('macro_comma', prefix='#define COMMA ,', statement=PRINT.replace('p.major, p.minor','p.major COMMA p.minor'))
    yield case('macro_callee', prefix='#define PRINTF printf', statement=PRINT.replace('printf','PRINTF'))
    yield case('printf_redeclaration', prefix='extern "C" int printf(const char *, ...);')
    yield case('qualified_call', statement=PRINT.replace('printf','::printf'))
    yield case('unrelated_label', statement=PRINT.replace('SM ','version '))
    yield case('unsigned_format', statement=PRINT.replace('%d','%u'))
    yield case('width_format', statement=PRINT.replace('%d','%2d'))
    yield case('star_format', statement='printf("SM %*d.%d\\n", 2, p.major, p.minor)')
    yield case('positional_format', statement='printf("SM %1$d.%2$d\\n", p.major, p.minor)')
    yield case('swapped_members', statement='printf("SM %d.%d\\n", p.minor, p.major)')
    yield case('nonadjacent_pair', statement='printf("SM %d / %d\\n", p.major, p.minor)')
    yield case('multiple_pairs', statement='printf("SM %d.%d SM %d.%d\\n", p.major, p.minor, p.major, p.minor)')
    yield case('side_effect_argument', local='cudaDeviceProp p{}; int i=0;', statement='printf("value %d SM %d.%d\\n", i++, p.major, p.minor)')
    yield case('nested_call_argument', prefix='int next();', statement='printf("value %d SM %d.%d\\n", next(), p.major, p.minor)')
    yield case('write_through_format', local='cudaDeviceProp p{}; int n;', statement='printf("%n SM %d.%d\\n", &n, p.major, p.minor)')
    yield case('embedded_null_format', statement=PRINT.replace('SM ', 'SM \\0'))
    yield case('capability_arithmetic', statement=PRINT.replace('p.major, p.minor','p.major + 1, p.minor'))
    yield case('mutable_format', local='cudaDeviceProp p{}; char format[]="SM %d.%d\\n";', statement=PRINT.replace('"SM %d.%d\\n"','format'))
    yield case('extra_argument', statement=PRINT[:-1]+', 17)')

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
    with tempfile.TemporaryDirectory(prefix='ascify-sm-diagnostic-') as temporary:
        work = args.evidence_dir.resolve() if args.evidence_dir else Path(temporary)
        if args.evidence_dir:
            work.mkdir(parents=True, exist_ok=False)
        summary = {'schema': 'ascify.sm-diagnostic-conversion.v1', 'passed': False,
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
            for item in cases(int(major.group(1))):
                directory = work / item.name
                directory.mkdir()
                source, output, receipt = (directory / name for name in ('input.cu', 'output.cpp', 'receipt.json'))
                source.write_text('#include <cuda_runtime.h>\n#include <stdio.h>\n' + item.code)
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
                        if actual.count('N/A (Ascend)') != item.expected.count('N/A (Ascend)'):
                            raise AssertionError(item.name + ': unexpected diagnostic replacement')
                        if item.dead_region is not None:
                            region = re.findall(r'/\*dead_begin\*/(.*?)/\*dead_end\*/', actual, re.S)
                            if region != [item.dead_region]:
                                raise AssertionError(item.name + ': dead-store boundary mismatch: ' + repr(region))
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
        print('CUDA SM diagnostics: ' + str(summary['total']) + ' cases passed' + (' (source only)' if args.source_only else ''))


if __name__ == '__main__':
    main()
