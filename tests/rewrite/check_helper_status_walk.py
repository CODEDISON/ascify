#!/usr/bin/env python3
"""Syntax-first status collection after residual helpers; never opens an NPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

if not __debug__:
    raise RuntimeError('Helper status-walk checks require assertions')

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / 'tests/rewrite/fixtures/nvidia_samples/Common'
PREFIX = ('#include <cuda.h>\n'
          'static const char* _cudaGetErrorEnum(CUresult) { return "driver status"; }\n'
          '#include <cuda_runtime.h>\n#include <helper_cuda.h>\n')
RESIDUAL = 'int observe_unmapped_helper(){return ftoi(1.5f);}\n'
LEGAL = 'void memory_check(){checkCudaErrors(cudaDeviceSynchronize());}\n'
TEMPLATE = ('template<class T> void memory_check(T* p){checkCudaErrors(cudaFree(p));}\n'
            'void instantiate(){memory_check<int>(nullptr);}\n')
FOREIGN = ('static int foreign_status_source(){return 0;}\n'
           'void mixed_check(){checkCudaErrors(cudaDeviceSynchronize());'
           'checkCudaErrors((cudaError_t)foreign_status_source());}\n')
DRIVER = ('void mixed_check(){checkCudaErrors(cudaDeviceSynchronize());'
          'checkCudaErrors(cuInit(0));}\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'clang', 'cuda-path', 'resource-dir', 'evidence-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--helper-root', type=Path, default=HELPERS)
    a = parser.parse_args()
    out = a.evidence_dir.resolve()
    out.mkdir()
    helpers = a.helper_root.resolve()
    helper_before = inventory(helpers)
    # Result kinds distinguish helper closure from conversion or run success.
    cases = [('runtime_control', LEGAL, 'committed')]
    for label, body, kind in [('runtime', LEGAL, 'retained'),
                              ('template', TEMPLATE, 'retained'),
                              ('foreign_int', FOREIGN, 'refused'),
                              ('driver', DRIVER, 'refused')]:
        cases += [(label + '_residual_before', RESIDUAL + body, kind),
                  (label + '_residual_after', body + RESIDUAL, kind)]
    cases += [('first_residual_preserved', RESIDUAL +
               'const char* observe_other_helper(){return _ConvertSMVer2ArchName(7,0);}\n' + LEGAL, 'retained')]
    rows = []

    def run(argv, directory):
        directory.mkdir()
        argv = list(map(str, argv))
        (directory / 'argv.json').write_text(json.dumps(argv) + '\n')
        process = subprocess.run(argv, capture_output=True, timeout=120)
        (directory / 'stdout.log').write_bytes(process.stdout)
        (directory / 'stderr.log').write_bytes(process.stderr)
        (directory / 'process.json').write_text(json.dumps({'returncode': process.returncode}) + '\n')
        return process

    for name, body, kind in cases:
        case = out / name
        case.mkdir()
        inputs = case / 'inputs'
        inputs.mkdir()
        source = inputs / 'input.cu'
        source.write_text(PREFIX + body)
        before = inventory(inputs)
        flags = ['-x', 'cuda', '-std=c++17', '-I' + str(helpers), '-I' + str(inputs)]
        syntax = run([a.clang, '--cuda-host-only', '-nocudalib',
            '--cuda-path=' + str(a.cuda_path), '-resource-dir=' + str(a.resource_dir),
            '-Wno-unknown-cuda-version', *flags, '-fsyntax-only', source], case / 'source_syntax')
        assert syntax.returncode == 0, (name, syntax.stderr.decode(errors='replace'))
        output = case / 'published.cce'
        bundle = Path(str(output) + '.headers')
        if kind == 'refused':
            output.write_text('previous root must remain\n')
            bundle.mkdir()
            (bundle / 'previous.txt').write_text('previous bundle must remain\n')
            prior = sha(output), inventory(bundle)
        destinations = [output] if kind != 'refused' else [output, case / 'fresh.cce']
        for index, destination in enumerate(destinations):
            result = run([a.binary, source, '--target-policy=dav-c310-vec',
                '--simt-math=fast', '--target-recipe=none', '--default-preprocessor',
                '--cuda-path=' + str(a.cuda_path), '--clang-resource-directory=' + str(a.resource_dir),
                '--frontend-compat=ascify-admitted-v1', '--local-headers-recursive',
                '-I' + str(helpers), '-I' + str(inputs), '-o', destination,
                '--', *flags], case / ('convert_' + str(index)))
            assert inventory(inputs) == before, (name, 'source mutation')
            diagnostic = result.stderr.decode(errors='replace')
            if kind != 'committed':
                assert "residual helper declaration 'ftoi'" in diagnostic, (name, diagnostic)
            if kind == 'refused':
                assert result.returncode != 0, (name, 'foreign/Driver status admitted')
                assert 'status domain not proven; conversion refused' in diagnostic, (name, diagnostic)
                assert (sha(output), inventory(bundle)) == prior, (name, 'previous artifacts changed')
                if destination != output:
                    assert not destination.exists() and not Path(str(destination) + '.headers').exists(), name
            else:
                assert result.returncode == 0, (name, diagnostic)
                text = destination.read_text()
                assert 'status domain not proven' not in diagnostic, (name, 'valid later status was not collected', diagnostic)
                if kind == 'retained':
                    assert '#include <helper_cuda.h>' in text and RESIDUAL in text, name
                    assert 'unsupported_declaration=1' in diagnostic and 'include kept' in diagnostic, name
                    for counter in ('check_rewrites=0', 'get_last_rewrites=0', 'find_device_rewrites=0'):
                        assert counter in diagnostic, (name, counter)
                    for emitted in ('ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS',
                                    'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR',
                                    '::ascify::sampleFindCudaDevice'):
                        assert emitted not in text, (name, 'partial helper transaction', emitted)
                    if name == 'first_residual_preserved':
                        assert "residual helper declaration '_ConvertSMVer2ArchName'" not in diagnostic, name
                else:
                    assert '#include <helper_cuda.h>' not in text, name
                    assert 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' in text and 'include removed' in diagnostic, name
            assert not list(case.glob('.ascify-local-closure*')), (name, 'staging leaked')
        rows.append({'name': name, 'passed': True, 'original_syntax_passed': True,
                     'expected_helper_transaction': kind,
                     'runtime_ready': False, 'original_sample_successes': 0})
    assert inventory(helpers) == helper_before, 'frozen helper mutation'
    (out / 'summary.json').write_text(json.dumps({'passed': True, 'cases': rows,
        'total': len(rows), 'retained_outputs_are_not_runtime_ready': True,
        'original_sample_successes': 0, 'device_executed': False}, indent=2) + '\n')
    print('Helper status walk:', len(rows), 'syntax-first cases PASS; residuals remain unready; no NPU')


if __name__ == '__main__':
    main()
