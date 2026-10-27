#!/usr/bin/env python3
"""Original-CUDA-syntax-first, additive utility-provider frontend controls only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inventory(p):
    return {str(q.relative_to(p)): sha(q) for q in sorted(p.rglob('*')) if q.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'clang', 'cuda-path', 'resource-dir', 'helper-root', 'evidence-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    assert __debug__
    for key in ('binary', 'clang', 'cuda_path', 'resource_dir', 'helper_root'):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    out = args.evidence_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec_path = Path(__file__).parent / 'fixtures/direct_image_utility_controls.json'
    cases = json.loads(spec_path.read_text())['cases']
    assert len(cases) == 8 and sum(c['accepted'] for c in cases) == 1
    helper_before = inventory(args.helper_root)
    tools = [args.binary, args.clang, Path(__file__).resolve(), spec_path,
             args.resource_dir / 'include/__clang_cuda_runtime_wrapper.h',
             args.cuda_path / 'include/cuda_runtime.h', args.cuda_path / 'include/cuda_runtime_api.h']
    before = {str(p): sha(p) for p in tools}
    state = dict(schema='ascify.direct-image-utility-controls.v1', passed=False,
                 binary_sha256=sha(args.binary), cases=[], source_inputs=before,
                 target_compiler_executed=False, device_executed=False, original_sample_successes=0)

    def command(argv, path):
        path.mkdir()
        (path / 'argv.json').write_text(json.dumps(list(map(str, argv)), indent=2) + '\n')
        result = subprocess.run(list(map(str, argv)), capture_output=True, timeout=180)
        (path / 'stdout.log').write_bytes(result.stdout)
        (path / 'stderr.log').write_bytes(result.stderr)
        (path / 'process.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
        return result

    try:
        for case in cases:
            directory = out / case['name']
            directory.mkdir()
            inputs = directory / 'input'
            inputs.mkdir()
            source = inputs / 'root.cu'
            source.write_text(case['source'])
            helpers = args.helper_root
            if case['setup'].get('mutate'):
                helpers = inputs / 'changed_helpers'
                shutil.copytree(args.helper_root, helpers)
                altered = helpers / case['setup']['mutate']
                altered.write_bytes(altered.read_bytes() + b'\n// private fixture mutation\n')
            extra = list(case['flags'])
            if case['setup'].get('remap'):
                original = helpers / case['setup']['remap']
                remapped = inputs / 'remapped.h'
                remapped.write_bytes(original.read_bytes())
                extra += ['-Xclang', '-remap-file', '-Xclang', str(original) + ';' + str(remapped)]
            language = ['-x', 'cuda', '-std=c++17', '-fgpu-rdc', '-I' + str(helpers), '-I' + str(inputs)]
            cccl = args.cuda_path / 'include/cccl'
            if (cccl / 'cuda/std/type_traits').is_file():
                language += ['-I' + str(cccl)]
            snapshot = inventory(inputs)
            syntax = command([args.clang, '--cuda-host-only', '-nocudalib',
                              '--cuda-path=' + str(args.cuda_path),
                              '-resource-dir=' + str(args.resource_dir), '-Wno-unknown-cuda-version',
                              *language, *extra, '-fsyntax-only', source], directory / 'original_syntax')
            assert syntax.returncode == 0, (case['name'], 'invalid original CUDA fixture', syntax.stderr)
            output = directory / 'published.cce'
            receipt = directory / 'receipt.json'
            converted = command([args.binary, source, '--cuda-path=' + str(args.cuda_path),
                                 '--clang-resource-directory=' + str(args.resource_dir),
                                 '--frontend-compat=ascify-admitted-v1', '--target-policy=dav-c310-vec',
                                 '--target-recipe=none', '--simt-math=precise', '--default-preprocessor',
                                 '--local-headers-recursive', '--migration-receipt=' + str(receipt),
                                 '-o', output, '--', *language, *extra], directory / 'convert')
            record = json.loads(receipt.read_text())
            assert inventory(inputs) == snapshot, (case['name'], 'input modified')
            if case['accepted']:
                assert converted.returncode == 0 and record['status'] == 'succeeded', (case['name'], converted.stderr)
                assert output.is_file() and not Path(str(output) + '.headers').exists()
                assert b'Ascify direct-image MAX replay v1:' in converted.stderr
                assert '#include <helper_cuda.h>' not in output.read_text()
            else:
                assert converted.returncode != 0 and record['status'] == 'failed', (case['name'], converted.stderr)
                assert b'fatal error:' not in converted.stderr, (case['name'], 'unrelated dependency failure')
                assert not output.exists() and not Path(str(output) + '.headers').exists()
            state['cases'].append(dict(name=case['name'], passed=True, accepted=case['accepted'],
                                       original_syntax_returncode=syntax.returncode, input_sha256=snapshot))
            (out / 'summary.json').write_text(json.dumps(state, indent=2) + '\n')
        assert inventory(args.helper_root) == helper_before
        assert all(sha(Path(p)) == value for p, value in before.items())
        state['passed'] = True
    finally:
        (out / 'summary.json').write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
