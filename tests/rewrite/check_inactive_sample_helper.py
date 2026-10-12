#!/usr/bin/env python3
"""Real CUDA preprocessing + transactional inactive-helper boundary checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / 'tests/rewrite/fixtures/nvidia_samples/Common'
MESSAGE = 'Ascify: inactive NVIDIA helper branch was enabled in the target configuration'
ACTIVE = 'checkCudaErrors(cudaFree(p));\n'
INACTIVE = 'checkCudaErrors(cudaDeviceSynchronize());\n'
BLOCK = '#ifdef PROJECT_OPTION\n' + INACTIVE + '#endif\n'
MAX = ('cudaFuncAttributes a;\n'
       'checkCudaErrors(cudaFuncGetAttributes(&a, kernel));\n'
       'int limit=a.maxThreadsPerBlock; (void)limit;\n')

# name, prelude, body, suffix, guards, helper removed, expected ordinary refusal
CASES = [
    ('simple', '', BLOCK, '', 1, True, False),
    ('indented', '', BLOCK.replace('#', '  # '), '', 1, True, False),
    ('two_bodies', '', BLOCK + BLOCK, '', 2, True, False),
    ('no_inactive_helper', '', BLOCK.replace(INACTIVE, 'int ignored=0;\n'), '', 0, True, False),
    ('undefined_then_defined', '', BLOCK, '#define PROJECT_OPTION\n', 1, True, False),
    ('defined_then_undefined', '#define PROJECT_OPTION\n', BLOCK, '#undef PROJECT_OPTION\n', 0, True, False),
    ('if_zero', '', '#if 0\n' + INACTIVE + '#endif\n', '', 0, False, False),
    ('ifndef', '#define PROJECT_OPTION\n', BLOCK.replace('ifdef', 'ifndef'), '', 0, False, False),
    ('else', '', BLOCK.replace('#endif', '#else\nint active=0;\n#endif'), '', 0, False, False),
    ('elif', '', BLOCK.replace('#endif', '#elif 0\nint ignored=0;\n#endif'), '', 0, False, False),
    ('nested_body', '', BLOCK.replace(INACTIVE, '#ifdef INNER_OPTION\n' + INACTIVE + '#endif\n'), '', 0, False, False),
    ('nested_parent', '', '#if 1\n' + BLOCK + '#endif\n', '', 0, False, False),
    ('body_directive', '', BLOCK.replace(INACTIVE, '#define BODY_OPTION\n' + INACTIVE), '', 0, False, False),
    ('body_block_comment', '', BLOCK.replace(INACTIVE, '/* limited grammar */\n' + INACTIVE), '', 0, False, False),
    ('spliced_opener', '', BLOCK.replace('ifdef ', 'ifdef \\\n'), '', 0, False, False),
    ('spliced_body', '', BLOCK.replace(INACTIVE, '// line \\\ncontinuation\n' + INACTIVE), '', 0, False, False),
    ('cr_lines', '', BLOCK.replace('\n', '\r\n'), '', 0, False, False),
    ('reserved_macro', '', BLOCK.replace('PROJECT_OPTION', '_PROJECT_OPTION'), '', 0, False, False),
    ('sdk_macro', '', BLOCK.replace('PROJECT_OPTION', 'CUDA_PROJECT_OPTION'), '', 0, False, False),
    ('mixed_unsafe_raw', '', BLOCK + '#if 0\n' + INACTIVE + '#endif\n', '', 0, False, False),
    ('mixed_unsafe_status', 'cudaError_t customStatus();\n', BLOCK + 'checkCudaErrors(customStatus());\n', '', 0, False, False),
    ('published_macro_body', '', BLOCK.replace(INACTIVE, INACTIVE + 'ASCIFY_FORCEINLINE\n'), '', 0, False, False),
    ('kernel_query', '#include "kernel.cuh"\n', MAX, '', 0, True, False),
    ('kernel_query_and_inactive', '#include "kernel.cuh"\n', MAX + BLOCK, '', 1, True, False),
    ('kernel_query_unsafe_raw', '#include "kernel.cuh"\n', MAX + BLOCK + '#if 0\n' + INACTIVE + '#endif\n', '', 0, False, False),
    ('kernel_metadata_refused', '#include "kernel.cuh"\n', MAX.replace('cudaFuncAttributes a;', 'cudaFuncAttributes a={0,7};') + BLOCK, '', 0, False, True),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--cuda-path', type=Path, required=True)
    parser.add_argument('--resource-dir', type=Path, required=True)
    parser.add_argument('--clang', type=Path)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.clang is None:
        args.clang = args.resource_dir.resolve().parents[2] / 'bin/clang'
        assert args.clang.is_file(), 'matching Clang not found; provide --clang explicitly'
    out = args.evidence_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {'passed': False, 'binary_sha256': sha(args.binary),
               'test_sha256': sha(Path(__file__)), 'cases': [],
               'scope': 'actual CUDA/frontend/preprocessor; no NPU or Sample count'}

    def save():
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    def run(directory, phase, argv):
        argv = list(map(str, argv))
        (directory / (phase + '.argv.json')).write_text(json.dumps(argv, indent=2) + '\n')
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        (directory / (phase + '.stdout.log')).write_text(result.stdout)
        (directory / (phase + '.stderr.log')).write_text(result.stderr)
        return result

    save()
    for name, prelude, body, suffix, guards, removed, refused in CASES:
        directory = out / name
        directory.mkdir()
        source, output = directory / 'input.cu', directory / 'output.cpp'
        receipt = directory / 'receipt.json'
        source.write_text('#include <helper_cuda.h>\n' + prelude +
                          'void host(void *p) {\n' + ACTIVE + body + '}\n' + suffix)
        (directory / 'kernel.cuh').write_text('__global__ void kernel(unsigned *p) { p[threadIdx.x]=threadIdx.x; }\n')
        row = {'name': name, 'passed': False, 'source_sha256': sha(source),
               'expected_guards': guards, 'expected_removed': removed}
        summary['cases'].append(row)
        save()
        flags = ['-x', 'cuda', '-std=c++17', '-I' + str(HELPERS), '-I' + str(directory)]
        syntax = [args.clang, source, *flags, '-fsyntax-only', '--cuda-host-only',
                  '-nocudalib', '--cuda-path=' + str(args.cuda_path),
                  '-resource-dir=' + str(args.resource_dir), '-Wno-unknown-cuda-version']
        result = run(directory, 'syntax', syntax)
        row['syntax_returncode'] = result.returncode
        save()
        assert result.returncode == 0, (name, result.stderr)
        result = run(directory, 'convert', [args.binary, source,
            '--target-policy=dav-c310-vec', '--simt-math=fast', '--target-recipe=none',
            '--default-preprocessor', '--local-headers-recursive',
            '--cuda-path=' + str(args.cuda_path),
            '--clang-resource-directory=' + str(args.resource_dir),
            '--migration-receipt=' + str(receipt), '-o', output, '--', *flags])
        row['conversion_returncode'] = result.returncode
        save()
        if refused:
            assert result.returncode > 0 and not output.exists(), (name, result.stderr)
            assert json.loads(receipt.read_text())['status'] == 'failed'
            assert 'unproved per-kernel maxThreadsPerBlock' in result.stderr
        else:
            assert result.returncode == 0, (name, result.stderr)
            assert json.loads(receipt.read_text())['status'] == 'succeeded'
            text = output.read_text()
            assert text.count(MESSAGE) == guards, (name, text, result.stderr)
            if guards:
                original_block = BLOCK.replace('#', '  # ') if name == 'indented' else BLOCK
                without_guards = text.replace('#error "' + MESSAGE + '"\n', '')
                assert without_guards.count(original_block) == guards, (name, text)
            assert ('#include <helper_cuda.h>' not in text) == removed, (name, result.stderr)
            assert ('ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' in text) == removed, (name, text)
            if not removed:
                # The helper transaction leaves its macro names unchanged;
                # independent, pre-existing CUDA API mappings still apply.
                assert text.count('checkCudaErrors(') == source.read_text().count('checkCudaErrors('), (name, text)
            if 'kernel_query' in name:
                assert text.count('::ascify::cudaFuncGetAttributesMaxThreads<1024>') == 1, (name, text)
            if name in ('simple', 'indented', 'kernel_query_and_inactive'):
                # Original CUDA is valid in both configurations. Only the
                # translated configuration restriction causes target failure.
                original_enabled = run(directory, 'original_enabled', syntax + ['-DPROJECT_OPTION'])
                row['original_enabled_returncode'] = original_enabled.returncode
                assert original_enabled.returncode == 0, original_enabled.stderr
                target = [args.clang, '-E', '-x', 'c++', '-std=c++17', '-D__aicore__=',
                          '-I' + str(ROOT / 'tests/rewrite/stubs'),
                          '-I' + str(ROOT / 'include'), '-I' + str(HELPERS), output]
                disabled = run(directory, 'target_default', target)
                enabled = run(directory, 'target_enabled', target + ['-DPROJECT_OPTION'])
                row.update(target_default_returncode=disabled.returncode,
                           target_enabled_returncode=enabled.returncode)
                assert disabled.returncode == 0, disabled.stderr
                assert enabled.returncode > 0 and MESSAGE in enabled.stderr, enabled.stderr
        assert sha(source) == row['source_sha256']
        row['passed'] = True
        save()
        print(name + ' passed', flush=True)
    assert sha(args.binary) == summary['binary_sha256']
    summary.update(passed=True, total=len(CASES))
    save()


if __name__ == '__main__':
    main()
