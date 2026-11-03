#!/usr/bin/env python3
"""CPU-only source-contract regression using an actual CUDA SDK and compiler.

Six exact pure helpers enter the masked accumulation contract. Twelve source-
valid neighbors retain old lowering, while four retain their existing refusal.
This script never executes emitted device code or counts sample acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

MASKED = "::ascify::detail::scalar_masked_add_assign_rne<"
SAFE = "::ascify::detail::masked_accumulation_delta("
OLD = "::ascify::detail::scalar_add_assign_rne<"
REFUSED = {
    'extra_statement': 'scalar64 operands require builtin values and effect-free addresses/RHS',
    'narrow_width': 'scalar64 shuffle admits only width 32',
    'subtract_assign': 'unsupported binary64 arithmetic/comparison',
    'inactive_body_directive': 'scalar64 body preprocessing/pragma alternatives are not admitted',
}


def fixtures():
    helper = 'template<class T> __device__ T fold(unsigned mask,T value){\n  for(int offset=warpSize/2;offset>0;offset/=2){\n    value += __shfl_down_sync(mask,value,offset);\n  }\n  return value;\n}\n'
    uses = '__device__ double use_d(double v){return fold(0xffffffffu,v);}\n__device__ float use_f(float v){return fold(0xffffffffu,v);}\n__device__ int use_i(int v){return fold(0xffffffffu,v);}\n'
    cases = []

    def case(name, body, positive=False, prefix='', use=uses):
        cases.append({'name': name, 'positive': positive, 'source': '#include <cuda_runtime.h>\n' + prefix + body + use})
    case('default_width', helper, True)
    case('literal_width32', helper.replace('mask,value,offset)', 'mask,value,offset,32)'), True)
    case('shift_halving', helper.replace('offset/=2', 'offset>>=1'), True)
    case('single_statement_loop', helper.replace('offset/=2){', 'offset/=2)').replace('  }\n  return', '  return'), True)
    case('renamed', helper.replace('fold', 'combine').replace('mask', 'membership').replace('value', 'acc').replace('offset', 'distance'), True, use=uses.replace('fold', 'combine'))
    case('parenthesized_operands', helper.replace('mask,value,offset', '(mask),(value),(offset)'), True)
    case('extra_statement', helper.replace('  return value;', '  value += T(0);\n  return value;'))
    case('alias_value', helper.replace('  for(', '  T& alias=value;\n  for(').replace('    value +=', '    alias +='))
    case('condition_not_equal', helper.replace('offset>0', 'offset!=0'))
    case('stride_four', helper.replace('offset/=2', 'offset>>=2'))
    case('literal_initial16', helper.replace('warpSize/2', '16'))
    case('different_value_expression', helper.replace('mask,value,offset', 'mask,value+value,offset'))
    case('narrow_width', helper.replace('mask,value,offset)', 'mask,value,offset,16)'))
    case('index_shuffle', helper.replace('__shfl_down_sync', '__shfl_sync'))
    case('subtract_assign', helper.replace('value +=', 'value -='))
    case('macro_value_leaf', helper.replace('mask,value,offset', 'mask,(VALUE),offset'), prefix='#define VALUE value\n')
    case('macro_callee_leaf', helper.replace('__shfl_down_sync(mask', '(DOWN)(mask'), prefix='#define DOWN __shfl_down_sync\n')
    case('macro_literal_leaf', helper.replace('warpSize/2', 'warpSize/(TWO)'), prefix='#define TWO 2\n')
    case('inactive_body_directive', helper.replace('    value +=', '#if defined(__NPU_ARCH__)\n    value += value;\n#else\n    value +=').replace('  }\n  return', '#endif\n  }\n  return'))
    case('caller_warp_constant', helper.replace('warpSize/2', 'caller_warp/2'), prefix='const int caller_warp=32;\n')
    case('reference_scalar_parameter', helper.replace('T value)', 'T& value)'))
    case('int_float_only', helper, use=uses.splitlines(True)[1] + uses.splitlines(True)[2])
    assert len(cases) == 22 and sum((row['positive'] for row in cases)) == 6
    assert len({row['name'] for row in cases}) == 22
    for row in cases:
        row['expected'] = 'masked' if row['positive'] else 'refused' if row['name'] in REFUSED else 'fallback'
    assert sum((row['expected'] == 'fallback' for row in cases)) == 12
    return cases


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--cuda-path', required=True)
    parser.add_argument('--resource-dir', required=True)
    parser.add_argument('--clang', default=os.environ.get('CLANG', 'clang++'))
    parser.add_argument('--nvcc', help='Optional real NVIDIA compiler for additional source validity')
    parser.add_argument('--nvcc-host-cxx', help='Optional NVCC host compiler path')
    parser.add_argument('--evidence-dir')
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    context = None if args.evidence_dir else tempfile.TemporaryDirectory(prefix='ascify-masked-source-')
    work = Path(args.evidence_dir if args.evidence_dir else context.name).resolve()
    work.mkdir(parents=True, exist_ok=True)
    assert not list(work.iterdir()), 'Fresh evidence directory required'
    sdk = Path(args.cuda_path).resolve()
    assert (sdk / 'include/cuda.h').is_file() and (sdk / 'include/cuda_runtime.h').is_file()
    header_identity = {str(p): sha(p) for p in [sdk / 'include/cuda.h', sdk / 'include/cuda_runtime.h']}
    cccl = sdk / 'include/cccl'
    sdk_flags = ['-I' + str(cccl)] if (cccl / 'cuda/std/type_traits').is_file() else []
    binary = Path(args.binary).resolve()
    before = dict(binary_sha256=sha(binary), sdk_headers=header_identity, script_sha256=sha(__file__))
    write(work / 'inputs.before.json', before)
    env = os.environ.copy()
    for key in ['LD_PRELOAD', 'LD_AUDIT', 'CPATH', 'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH',
                'LIBRARY_PATH', 'PYTHONPATH']:
        env.pop(key, None)
    rows = []
    summary = dict(schema='ascify.masked-accumulation-source-regression.v1', passed=False,
                   cases=rows, binary_sha256=before['binary_sha256'],
                   optional_native_NVCC=bool(args.nvcc), CCEC=0, NPU=0, sample_acceptance=0)

    def command(directory, name, argv):
        argv = list(map(str, argv))
        write(directory / (name + '.argv.json'), argv)
        start = time.monotonic()
        result = subprocess.run(argv, env=env, capture_output=True, timeout=120)
        (directory / (name + '.stdout.log')).write_bytes(result.stdout)
        (directory / (name + '.stderr.log')).write_bytes(result.stderr)
        write(directory / (name + '.process.json'), dict(returncode=result.returncode,
              elapsed_seconds=time.monotonic() - start))
        return result

    def no_bundle(directory):
        assert not list(directory.glob('*.ascify*')), 'Refused source published an unexpected bundle'

    try:
        cases = fixtures()
        write(work / 'cases.json', cases)
        for item in cases:
            directory = work / item['name']
            directory.mkdir()
            source = directory / 'original.cu'
            source.write_text(item['source'])
            digest = sha(source)
            if args.nvcc:
                argv = [args.nvcc, '-std=c++17', '-arch=sm_75', '-dc']
                if args.nvcc_host_cxx:
                    argv += ['-ccbin', args.nvcc_host_cxx]
                result = command(directory, 'NVCC', argv + [source, '-o', directory / 'original.o'])
                assert result.returncode == 0, (item['name'], 'Original CUDA rejected by NVCC', result.stderr.decode())
            for view in ['host', 'device']:
                result = command(directory, 'Clang_' + view, [args.clang, '-x', 'cuda',
                    '--cuda-' + view + '-only', '--cuda-path=' + str(sdk), '--cuda-gpu-arch=sm_75',
                    '-nocudalib', '-std=c++17', '-fgpu-rdc', *sdk_flags,
                    '-Wno-unknown-cuda-version', '-fsyntax-only', source])
                assert result.returncode == 0, (item['name'], 'Original CUDA ' + view + ' view rejected', result.stderr.decode())
            output, receipt = directory / 'source.cce', directory / 'receipt.json'
            argv = [binary, source, '--target-policy=dav-c310-vec', '--simt-math=precise',
                    '--default-preprocessor', '--frontend-compat=ascify-admitted-v1',
                    '--cuda-path=' + str(sdk), '--clang-resource-directory=' + str(Path(args.resource_dir).resolve()),
                    '--migration-receipt=' + str(receipt), '-o', output, '--', '-x', 'cuda',
                    '-std=c++17', '-fgpu-rdc', *sdk_flags]
            result = command(directory, 'converter', argv)
            assert sha(source) == digest, 'Converter changed original input'
            text = output.read_text() if output.exists() else ''
            markers = re.findall(r'masked_accumulations=(\d+)', result.stderr.decode())
            count, safe = text.count(MASKED), text.count(SAFE)
            if item['expected'] == 'masked':
                assert result.returncode == 0 and markers == ['1'] and count == safe == 1, (item['name'], result.stderr.decode())
                mask, delta = ('membership', 'distance') if item['name'] == 'renamed' else ('mask', 'offset')
                if item['name'] == 'parenthesized_operands':
                    mask, delta = '(' + mask + ')', '(' + delta + ')'
                compact = re.sub(r'\s+', '', text)
                assert SAFE + mask + ',' + delta + ')' in compact, 'Inner safe delta lost original metadata'
                assert '),' + mask + ',' + delta + ')' in compact, 'Outer predicate lost original metadata'
                assert 'Ascify scalar64 requires the proven closed builtin dtype domain' in text
            elif item['expected'] == 'fallback':
                assert result.returncode == 0 and count == safe == 0, (item['name'], result.stderr.decode())
                if item['name'] == 'int_float_only':
                    assert markers == [] and OLD not in text
                else:
                    assert markers == ['0'] and OLD in text, 'Old scalar lowering did not remain available'
            else:
                assert result.returncode != 0 and not output.exists(), (item['name'], result.stderr.decode())
                assert 'scalar64 boundary:' in result.stderr.decode() and REFUSED[item['name']] in result.stderr.decode()
                assert json.loads(receipt.read_text())['status'] == 'failed'
                no_bundle(directory)
                sentinel = b'preserve-old-output\n'
                output.write_bytes(sentinel)
                # A failed migration receipt is intentional diagnostic output.
                # The previously published product output must stay untouched.
                repeated = command(directory, 'refusal_atomic_repeat', argv)
                assert repeated.returncode == result.returncode and output.read_bytes() == sentinel
                assert json.loads(receipt.read_text())['status'] == 'failed'
                assert sha(source) == digest
                no_bundle(directory)
            rows.append(dict(name=item['name'], source_sha256=digest, expected=item['expected'],
                             Clang_both_views_valid=True, native_NVCC_valid=bool(args.nvcc),
                             converter_returncode=result.returncode, masked_markers=markers,
                             masked_helper_calls=count, safe_delta_calls=safe))
            print(item['name'] + ': ' + item['expected'], flush=True)
        assert len(rows) == 22
        summary.update(passed=True, masked=6, old_fallback=12, old_refused=4)
    finally:
        after = dict(binary_sha256=sha(binary), sdk_headers={name: sha(name) for name in header_identity},
                     script_sha256=sha(__file__))
        write(work / 'inputs.after.json', after)
        summary['inputs_unchanged'] = before == after
        summary['passed'] = summary['passed'] and before == after
        write(work / 'summary.json', summary)
        if context:
            context.cleanup()
        assert before == after, 'Bound compiler inputs or converter changed'
    assert summary['passed']


if __name__ == '__main__':
    main()
