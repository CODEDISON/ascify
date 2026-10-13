#!/usr/bin/env python3
"""Exact v13.0 helper dependency identity; no sample or device execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / 'tests/rewrite/fixtures/nvidia_samples/Common'
OLD = ROOT / 'tests/rewrite/fixtures/nvidia_samples_v13_0/Common/helper_string.h'
OLD_SHA = '616ac8b2588a6e73aed207f85faf0da5e617404390df1790f6637b1042dbcc94'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'cuda-path', 'resource-dir', 'evidence-dir'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--clang', required=True)
    ap.add_argument('--syntax-only', action='store_true')
    a = ap.parse_args()
    out = a.evidence_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert OLD.stat().st_size == 14875 and sha(OLD) == OLD_SHA
    # All source inputs are valid CUDA; negative cases preserve the complete
    # helper transaction, rather than counting a parser error as a refusal.
    cases = [
        ('direct_check', '', 'checkCudaErrors(cudaFree(p));', '', True),
        ('last_error', '', 'getLastCudaError("after work");', '', True),
        ('find_device', '', 'int device = findCudaDevice(argc, argv); (void)device;', '', True),
        ('portable_flag', '', 'bool flag = checkCmdLineFlag(argc, argv, "help"); (void)flag; checkCudaErrors(cudaFree(p));', '', True),
        ('portable_file', '', 'char *file = sdkFindFilePath("data.txt", argv[0]); free(file); checkCudaErrors(cudaFree(p));', '', True),
        ('functions_first', '#include <helper_functions.h>\n', 'checkCudaErrors(cudaFree(p));', '', True),
        ('functions_after', '', 'checkCudaErrors(cudaFree(p));', '#include <helper_functions.h>\n', True),
        ('string_first', '#include <helper_string.h>\n', 'checkCudaErrors(cudaFree(p));', '', True),
        ('changed_comment', '', 'checkCudaErrors(cudaFree(p));', '', False),
        ('changed_function', '', 'checkCudaErrors(cudaFree(p));', '', False),
        ('policy_redefined', '', 'checkCudaErrors(cudaFree(p));', '#undef EXIT_WAIVED\n#define EXIT_WAIVED 9\n', False),
        ('observable_body', '', 'const char *s = SHOW(checkCudaErrors(cudaFree(p))); (void)s;', '#define STR(x) #x\n#define SHOW(x) STR(x)\n', False),
    ]
    summary = {'schema': 'ascify.sample-helper-v13.0-boundaries.v1', 'passed': False,
               'syntax_only': a.syntax_only, 'binary_sha256': sha(a.binary),
               'test_sha256': sha(Path(__file__)), 'source_tag': 'v13.0',
               'source_commit': '3f1c50965017932fc81e6d94a3fc9e04c105b312',
               'old_helper_string_sha256': OLD_SHA, 'device_executed': False,
               'new_original_device_successes': 0, 'cases': []}
    def save():
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    def run(d, phase, argv):
        argv = list(map(str, argv))
        (d / (phase + '.argv.json')).write_text(json.dumps(argv, indent=2) + '\n')
        r = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        (d / (phase + '.stdout.log')).write_text(r.stdout)
        (d / (phase + '.stderr.log')).write_text(r.stderr)
        return r
    save()
    for name, before, body, after, removed in cases:
        d = out / name
        d.mkdir()
        helpers = d / 'Common'
        shutil.copytree(COMMON, helpers)
        shutil.copyfile(OLD, helpers / 'helper_string.h')
        header = helpers / 'helper_string.h'
        if name == 'changed_comment':
            text = header.read_text()
            assert 'Copyright' in text
            header.write_text(text.replace('Copyright', 'CopyRight', 1))
        if name == 'changed_function':
            text = header.read_text()
            assert 'const char *searchPath[]' in text
            header.write_text(text.replace('"./",', '"../",', 1))
            assert sha(header) != OLD_SHA
        identity = {str(p.relative_to(helpers)): sha(p) for p in helpers.rglob('*') if p.is_file()}
        source, output, receipt = d/'input.cu', d/'output.cpp', d/'receipt.json'
        source.write_text(before + '#include <helper_cuda.h>\n' + after +
                          'void host(void *p, int argc, const char **argv) {\n  ' + body + '\n}\n')
        flags = ['-x', 'cuda', '-std=c++17', '-I' + str(helpers)]
        r = run(d, 'syntax', [a.clang, source, *flags, '-fsyntax-only', '--cuda-host-only', '-nocudalib',
            '--cuda-path=' + str(a.cuda_path), '-resource-dir=' + str(a.resource_dir), '-Wno-unknown-cuda-version'])
        row = {'name': name, 'expected_helper_removed': removed, 'source_sha256': sha(source),
               'helper_inputs': identity, 'syntax_returncode': r.returncode, 'passed': False}
        summary['cases'].append(row)
        save()
        assert r.returncode == 0, (name, r.stderr)
        if not a.syntax_only:
            r = run(d, 'convert', [a.binary, source, '--target-policy=dav-c310-vec', '--simt-math=fast',
                '--target-recipe=none', '--default-preprocessor', '--print-stats', '--local-headers-recursive',
                '--cuda-path=' + str(a.cuda_path), '--clang-resource-directory=' + str(a.resource_dir),
                '--migration-receipt=' + str(receipt), '-o', output, '--', *flags])
            row['conversion_returncode'] = r.returncode
            save()
            assert r.returncode == 0, (name, r.stderr)
            assert json.loads(receipt.read_text())['status'] == 'succeeded'
            text = output.read_text()
            assert ('#include <helper_cuda.h>' not in text) == removed, (name, text, r.stderr)
            if removed:
                assert '#include <helper_string.h>' in text, (name, text)
                assert 'closure: include removed; portable surface retained' in r.stderr, (name, r.stderr)
                assert 'checkCudaErrors(' not in text and 'getLastCudaError(' not in text
                assert 'findCudaDevice(' not in text
                for portable in ('checkCmdLineFlag', 'sdkFindFilePath'):
                    assert text.count(portable + '(') == source.read_text().count(portable + '(')
            else:
                assert text.count('checkCudaErrors(') == source.read_text().count('checkCudaErrors(')
                assert 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' not in text
            assert identity == {str(p.relative_to(helpers)): sha(p) for p in helpers.rglob('*') if p.is_file()}
        row['passed'] = True
        save()
    assert sha(OLD) == OLD_SHA and sha(a.binary) == summary['binary_sha256']
    summary['passed'] = True
    save()
    print(json.dumps({'passed': True, 'cases': len(cases), 'syntax_only': a.syntax_only, 'device_executed': False}))


if __name__ == '__main__':
    main()
