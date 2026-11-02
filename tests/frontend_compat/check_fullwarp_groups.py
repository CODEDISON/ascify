#!/usr/bin/env python3
"""Full-source static16/erased-group admission and atomic refusal, CPU only.

The 61 frozen original CUDA fixtures are checked without the parser projection
before conversion. This does not build an original Sample, compile CCEC code,
execute a kernel, or establish a memory-visibility/device result.
"""
import argparse
import hashlib
import json
import os
import shlex
import struct
from pathlib import Path
import re
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError('optimized Python refused')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['binary', 'clang', 'cuda-path', 'resource-dir', 'evidence-dir',
                 'nvcc-sdk', 'nvcc-frozen-manifest', 'native-host-compiler']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    nvcc = args.nvcc_sdk / 'bin/nvcc'
    native_env = os.environ.copy()
    for name in ['LD_PRELOAD', 'LD_AUDIT', 'CPATH', 'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH',
                 'OBJC_INCLUDE_PATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'NVCC_PREPEND_FLAGS',
                 'NVCC_APPEND_FLAGS']:
        native_env.pop(name, None)
    native_env['PATH'] = str(args.nvcc_sdk / 'bin') + ':/usr/bin:/bin'
    frozen = json.loads(args.nvcc_frozen_manifest.read_text())
    native_inputs = {str(args.nvcc_frozen_manifest.resolve()): sha(args.nvcc_frozen_manifest),
                     str(args.native_host_compiler.resolve()): sha(args.native_host_compiler)}
    for item in frozen:
        path = args.nvcc_sdk / item['path']
        if item['type'] == 'file':
            assert path.is_file() and path.stat().st_size == item['size'] and sha(path) == item['sha256']
            native_inputs[str(path)] = item['sha256']
        else:
            assert item['type'] == 'symlink' and path.is_symlink() and str(path.readlink()) == item['target']
            if path.is_file(): native_inputs[str(path)] = sha(path)
    root = Path(__file__).resolve().parents[2]
    fixtures = Path(__file__).resolve().parent / 'fullwarp_groups'
    manifest = fixtures / 'cases.json'
    spec = json.loads(manifest.read_text())
    cases = spec['cases']
    assert spec['schema'] == 'ascify.fullwarp-group.checker-fixtures.v1'
    assert len(cases) == spec['prepared_total'] == 61
    assert len({c['name'] for c in cases}) == len(cases)
    assert len({c['file'] for c in cases}) == len(cases)
    assert sum(c['expected_frontend_after_implementation'] == 'commit' for c in cases) == 4
    assert sum(c['expected_frontend_after_implementation'] == 'refuse' for c in cases) == 56
    assert sum(c['expected_frontend_after_implementation'] == 'legacy_contract_unchanged' for c in cases) == 1
    expected_files = {'cases.json'} | {c['file'] for c in cases} | {e['file'] for e in spec['extra_files']}
    assert {p.name for p in fixtures.iterdir()} == expected_files
    identities = {str(p.resolve()): sha(p) for p in [
        Path(__file__), args.binary, args.clang, manifest,
        root / 'frontend_compat/ascify-admitted-v1/cooperative_groups.h',
        root / 'frontend_compat/ascify-admitted-v1/profile.manifest',
        root / 'include/ascify/cooperative_groups_compat.hpp',
        root / 'src/AscifyCudaCompatIdentifiers.inc']}
    identities.update(native_inputs)
    for entry in cases + spec['extra_files']:
        path = fixtures / entry['file']
        assert path.name == entry['file'] and not path.is_symlink()
        assert sha(path) == entry['sha256'], (entry['file'], 'fixture changed')
        identities[str(path.resolve())] = entry['sha256']
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(schema='ascify.fullwarp-group.checker-results.v1', passed=False,
                   source_validation_kind='nvidia_nvcc_native_object_single_arch_rdc',
                   total=61, expected_commits=4, expected_refusals=56,
                   expected_legacy_unchanged=1, cases=[], identities_before=identities,
                   original_syntax_invocations=0, NVCC_dependency_invocations=0, NVCC_object_invocations=0,
                   converter_invocations=0,
                   CCEC_invocations=0, device_executed=False, original_sample_successes=0)

    def save():
        (args.evidence_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    def run(folder, label, argv, env=None):
        argv = list(map(str, argv))
        (folder / (label + '.argv.json')).write_text(json.dumps(argv, indent=2) + '\n')
        result = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=env)
        for stream in ['stdout', 'stderr']:
            (folder / (label + '.' + stream + '.log')).write_text(getattr(result, stream))
        (folder / (label + '.process.json')).write_text(json.dumps({'returncode': result.returncode}) + '\n')
        return result

    save()
    try:
        for case in cases:
            folder = args.evidence_dir / case['name']; folder.mkdir()
            source = folder / 'input.cu'
            shutil.copyfile(fixtures / case['file'], source)
            for extra in spec['extra_files']:
                shutil.copyfile(fixtures / extra['file'], folder / extra['file'])
            inputs = {str(p): sha(p) for p in folder.iterdir() if p.is_file()}
            expectation = case['expected_frontend_after_implementation']
            accepted = expectation != 'refuse'
            row = dict(name=case['name'], expected=expectation, passed=False,
                       source_syntax_passed=False, inputs=inputs, conversions=[])
            summary['cases'].append(row); save()
            # Real unmodified NVCC/SDK source oracle. No projection, Clang
            # version override, source shim, link or device execution.
            arch = case.get('original_syntax_arch', 'sm_75')
            assert arch in ('sm_75', 'sm_80')
            flags = ['-x', 'cuda', '-std=c++17', '-fgpu-rdc', '-I' + str(folder),
                     '-I' + str(args.cuda_path / 'include/cccl')]
            native_flags = [nvcc, '-ccbin', args.native_host_compiler, '-std=c++17',
                '--extended-lambda', '-rdc=true', '-Xcompiler=-fPIC', '-lineinfo', '--threads=1',
                '-Wno-deprecated-gpu-targets', '--generate-code=arch=compute_' + arch[3:] +
                ',code=[compute_' + arch[3:] + ',' + arch + ']',
                '-I' + str(args.nvcc_sdk / 'include'),
                '-I' + str(args.nvcc_sdk / 'include/cccl'), '-I' + str(folder)]
            dependency_file = folder / 'original_native.d'
            dependencies = run(folder, 'original_dependencies',
                [*native_flags, '-M', source, '-o', dependency_file], native_env)
            summary['NVCC_dependency_invocations'] += 1; save()
            assert dependencies.returncode == 0, (case['name'], 'native dependency scan failed', dependencies.stderr)
            dependency_text = dependency_file.read_text().replace('\\\n', ' ')
            assert ':' in dependency_text
            dependency_paths = shlex.split(dependency_text.split(':', 1)[1])
            assert str(source) in dependency_paths
            assert all(Path(name).is_absolute() and Path(name).is_file() for name in dependency_paths)
            dependency_identity = {name: sha(Path(name)) for name in dependency_paths}
            for name, digest in dependency_identity.items():
                assert name not in identities or identities[name] == digest, 'native dependency changed across cases'
                identities[name] = digest
            row['native_dependency_identity'] = dependency_identity; save()
            obj = folder / 'original_native.o'
            original = run(folder, 'original_syntax', [*native_flags, '-c', source, '-o', obj], native_env)
            summary['NVCC_object_invocations'] += 1
            summary['original_syntax_invocations'] += 1
            row.update(source_syntax_passed=original.returncode == 0,
                       source_syntax_arch=arch, source_syntax_returncode=original.returncode)
            save()
            assert original.returncode == 0, (case['name'], 'invalid original CUDA', original.stderr)
            header = obj.read_bytes()[:64]
            assert len(header) == 64 and header[:6] == b'\x7fELF\x02\x01'
            assert struct.unpack_from('<HH', header, 16) == (1, 183), 'expected AArch64 ET_REL original object'
            assert {name: sha(Path(name)) for name in dependency_paths} == dependency_identity
            row.update(native_object_sha256=sha(obj), native_object_bytes=obj.stat().st_size)
            for mode in (['fresh'] if accepted else ['fresh', 'existing']):
                output = folder / (mode + '.cce')
                receipt = folder / (mode + '.receipt.json')
                sentinel = ('EXISTING_UNCHANGED_' + case['name'] + '\n').encode()
                if mode == 'existing': output.write_bytes(sentinel)
                converted = run(folder, mode, [args.binary, source,
                    '--target-policy=dav-c310-vec', '--target-recipe=none',
                    '--simt-math=fast', '--default-preprocessor',
                    '--frontend-compat=ascify-admitted-v1', '--cuda-path=' + str(args.cuda_path),
                    '--clang-resource-directory=' + str(args.resource_dir),
                    '--migration-receipt=' + str(receipt), '-o', output, '--', *flags,
                    '--cuda-gpu-arch=' + arch])
                summary['converter_invocations'] += 1
                report = json.loads(receipt.read_text())
                item = dict(mode=mode, returncode=converted.returncode,
                            receipt_status=report['status'])
                row['conversions'].append(item); save()
                if not accepted:
                    assert converted.returncode > 0 and report['status'] == 'failed', (case['name'], converted.stderr)
                    assert output.read_bytes() == sentinel if mode == 'existing' else not output.exists()
                    assert not Path(str(output) + '.headers').exists()
                    # All negative originals above are syntactically valid.
                    # Refusal must come from the production proof/provenance
                    # transaction, never an unrelated syntax/compiler error.
                    assert re.search(r'full-warp group boundary|runtime-API header boundary|CUDA.*(?:attribute|macro).*boundary',
                                     converted.stderr), (case['name'], converted.stderr)
                    item.update(atomic_refusal=True, output_preserved=True)
                else:
                    assert converted.returncode == 0 and report['status'] == 'succeeded', (case['name'], converted.stderr)
                    text = output.read_text()
                    assert '#include <ascify/cooperative_groups_compat.hpp>' in text
                    assert not re.search(r'#\s*include\s*[<"]cooperative_groups(?:/|\.h)', text)
                    assert not re.search(r'\bcooperative_groups\s*::', text)
                    marker = re.search(r'Ascify full-warp groups v1: kernels=(\d+), helpers=(\d+), sync_sites=(\d+)',
                                       converted.stderr)
                    if expectation == 'commit':
                        assert marker and int(marker[1]) > 0 and int(marker[3]) > 0, (case['name'], converted.stderr)
                        item['source_proof_stats'] = dict(zip(['kernels', 'helpers', 'sync_sites'], map(int, marker.groups())))
                    else:
                        assert marker is None, 'tile32-only input unexpectedly entered the new proof domain'
                    item.update(output_sha256=sha(output), output_bytes=output.stat().st_size)
            assert all(sha(Path(name)) == digest for name, digest in inputs.items())
            assert {name: sha(Path(name)) for name in dependency_paths} == dependency_identity
            row['passed'] = True; save(); print(case['name'] + ': passed', flush=True)
        assert summary['NVCC_dependency_invocations'] == summary['NVCC_object_invocations'] == 61
        assert summary['original_syntax_invocations'] == 61
        assert summary['converter_invocations'] == 117  # 4 commit + 56*2 refusal + 1 legacy.
        after = {name: sha(Path(name)) for name in identities}
        assert after == identities, 'tool/source/profile/fixture identity changed'
        summary.update(passed=True, accepted=4, refused=56, legacy_unchanged=1,
                       identities_after=after)
    except Exception as error:
        summary['error'] = str(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
