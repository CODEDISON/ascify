#!/usr/bin/env python3
"""Original-CUDA syntax first, then versioned SM-helper frontend boundaries; no NPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

if not __debug__:
    raise RuntimeError('SM helper checks require assertions')

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ('#include <stdio.h>\n#include <cuda.h>\n'
          'static const char* _cudaGetErrorEnum(CUresult){return "driver status";}\n'
          '#include <cuda_runtime.h>\n')
CASES = [
    ('direct', '', 'int f(int a,int b){return _ConvertSMVer2Cores(a,b);}', 'commit'),
    ('once', '', 'int f(){int a=7,b=0;return _ConvertSMVer2Cores(a++,b++);}', 'commit'),
    ('heuristic', '', 'float f(int n){return max(192.f/(_ConvertSMVer2Cores(7,0)*float(n)),1.f);}', 'commit'),
    ('status', '', 'int f(){checkCudaErrors(cudaDeviceSynchronize());return _ConvertSMVer2Cores(7,0);}', 'commit'),
    ('namespace_host', '', 'namespace caller{int f(){return _ConvertSMVer2Cores(10,7);}}', 'commit'),
    ('other_residual', '', 'int f(){return ftoi(1.5f)+_ConvertSMVer2Cores(7,0);}', 'retain'),
    ('foreign_overload', '', 'int _ConvertSMVer2Cores(long,long){return 9;}int f(){return _ConvertSMVer2Cores(7L,0L);}', 'refuse'),
    ('foreign_redecl', '', 'int _ConvertSMVer2Cores(int,int);int f(){return _ConvertSMVer2Cores(7,0);}', 'refuse'),
    ('qualified', '', 'int f(){return ::_ConvertSMVer2Cores(7,0);}', 'refuse'),
    ('indirect', '', 'int f(){auto p=&_ConvertSMVer2Cores;return p(7,0);}', 'refuse'),
    ('sizeof', '', 'int f(){return sizeof(_ConvertSMVer2Cores(7,0));}', 'refuse'),
    ('decltype', '', 'using T=decltype(_ConvertSMVer2Cores(7,0));T value;', 'refuse'),
    ('noexcept', '', 'bool f(){return noexcept(_ConvertSMVer2Cores(7,0));}', 'refuse'),
    ('global_init', '', 'int value=_ConvertSMVer2Cores(7,0);', 'refuse'),
    ('template', '', 'template<class T>int f(T a){return _ConvertSMVer2Cores(a,0);}int g(){return f(7);}', 'refuse'),
    ('zero_template', '', 'template<class T>int f(T a){return _ConvertSMVer2Cores(a,0);}', 'refuse'),
    ('lambda', '', 'int f(){auto x=[](){return _ConvertSMVer2Cores(7,0);};return x();}', 'refuse'),
    ('macro_wrapper', '#define CALL(a,b) _ConvertSMVer2Cores(a,b)\n', 'int f(){return CALL(7,0);}', 'refuse'),
    ('macro_argument', '#define MAJOR 7\n', 'int f(){return _ConvertSMVer2Cores(MAJOR,0);}', 'refuse'),
    ('stringize', '#define S0(x) #x\n#define S(x) S0(x)\n', 'const char* x=S(_ConvertSMVer2Cores);', 'refuse'),
    ('definition_macro', '#define SM renamed_SM\n', 'int f(){return _ConvertSMVer2Cores(7,0);}', 'refuse'),
    ('definition_printf_macro', '#define printf(...) (0)\n', 'int f(){return _ConvertSMVer2Cores(7,0);}', 'refuse'),
    ('foreign_printf', 'int printf(const char*,int,int,int);\n', 'int f(){return _ConvertSMVer2Cores(1,0);}', 'refuse'),
    ('fake_system_body', '#include <fake_stdio.h>\n', 'int f(){return _ConvertSMVer2Cores(1,0);}', 'refuse'),
    ('fake_system_asm', '#include <fake_stdio.h>\n', 'int f(){return _ConvertSMVer2Cores(1,0);}', 'refuse'),
    ('fake_system_pure', '#include <fake_stdio.h>\n', 'int f(){return _ConvertSMVer2Cores(1,0);}', 'refuse'),
    ('fake_system_noreturn', '#include <fake_stdio.h>\n', 'int f(){return _ConvertSMVer2Cores(1,0);}', 'refuse'),
    ('mutated_table', '', 'int f(){return _ConvertSMVer2Cores(7,0);}', 'refuse'),
    ('runtime_foreign_status', '', 'int f(){checkCudaErrors((cudaError_t)7);return _ConvertSMVer2Cores(7,0);}', 'refuse'),
    ('target_macro', '#define sampleSMVersionToCoresV130 caller_target\n', 'int f(){return _ConvertSMVer2Cores(7,0);}', 'refuse'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('clang', 'cuda-path', 'resource-dir', 'evidence-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--profile-host-math', type=Path,
                        default=ROOT/'frontend_compat/ascify-admitted-v1/host_math.h')
    parser.add_argument('--syntax-only-preparation', action='store_true')
    a = parser.parse_args()
    assert a.syntax_only_preparation or (a.binary and a.binary.is_file()), 'new binary required'
    out = a.evidence_dir.resolve()
    out.mkdir()
    helpers = ROOT / 'tests/rewrite/fixtures/nvidia_samples/Common'
    version134 = ROOT / 'tests/rewrite/fixtures/nvidia_samples_v13_4/Common/helper_cuda.h'
    original_helpers = inventory(helpers), sha(version134)
    rows = []

    def run(directory, argv):
        directory.mkdir()
        argv = list(map(str, argv))
        (directory / 'argv.json').write_text(json.dumps(argv, indent=2) + '\n')
        result = subprocess.run(argv, capture_output=True, timeout=120)
        (directory / 'stdout.log').write_bytes(result.stdout)
        (directory / 'stderr.log').write_bytes(result.stderr)
        (directory / 'process.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
        return result

    for version in (130, 134):
        target = '::ascify::sampleSMVersionToCoresV' + str(version)
        for name, before, body, kind in CASES:
            case = out / (str(version) + '_' + name)
            case.mkdir()
            inputs = case / 'inputs'
            inputs.mkdir()
            common = inputs / 'Common'
            shutil.copytree(helpers, common)
            if version == 134:
                shutil.copyfile(version134, common / 'helper_cuda.h')
            if name == 'mutated_table':
                p = common / 'helper_cuda.h'
                text = p.read_text()
                assert text.count('{0x70,  64}') == 1
                p.write_text(text.replace('{0x70,  64}', '{0x70,  63}', 1))
            # Each version rejects collision with its own emitted target.
            if name == 'target_macro':
                before = '#define sampleSMVersionToCoresV' + str(version) + ' caller_target\n'
            system_flags = []
            if name.startswith('fake_system_'):
                system = inputs/'System'
                system.mkdir()
                forms = {
                    'fake_system_body': 'extern "C" int printf(const char*,...){return 123;}\n',
                    'fake_system_asm': 'extern "C" int printf(const char*,...) __asm__("caller_printf");\n',
                    'fake_system_pure': 'extern "C" int printf(const char*,...) __attribute__((pure));\n',
                    'fake_system_noreturn': 'extern "C" int printf(const char*,...) __attribute__((noreturn));\n',
                }
                (system/'fake_stdio.h').write_text(forms[name])
                system_flags = ['-isystem', str(system)]
            source = inputs / 'input.cu'
            source.write_text(PREFIX + before + '#include <helper_cuda.h>\n' + body + '\n')
            frozen = inventory(inputs)
            flags = ['-x', 'cuda', '-std=c++17', '-I' + str(common), '-I' + str(inputs), *system_flags]
            syntax = run(case / 'original_syntax', [a.clang, '--cuda-host-only', '-nocudalib',
                '--cuda-path=' + str(a.cuda_path), '-resource-dir=' + str(a.resource_dir),
                '-Wno-unknown-cuda-version', '-include', a.profile_host_math,
                *flags, '-fsyntax-only', source])
            assert syntax.returncode == 0, (version, name, syntax.stderr.decode(errors='replace'))
            row = {'version': version, 'case': name, 'original_syntax_passed': True,
                   'expected': kind, 'new_frontend_run': False}
            if not a.syntax_only_preparation:
                output = case / 'published.cce'
                bundle = Path(str(output) + '.headers')
                if kind == 'refuse':
                    output.write_text('previous root\n')
                    bundle.mkdir()
                    (bundle / 'previous.txt').write_text('previous bundle\n')
                    prior = sha(output), inventory(bundle)
                destinations = [output] if kind != 'refuse' else [output, case / 'fresh.cce']
                for i, destination in enumerate(destinations):
                    result = run(case / ('convert_' + str(i)), [a.binary, source,
                        '--target-policy=dav-c310-vec', '--simt-math=fast', '--target-recipe=none',
                        '--default-preprocessor', '--cuda-path=' + str(a.cuda_path),
                        '--clang-resource-directory=' + str(a.resource_dir),
                        '--frontend-compat=ascify-admitted-v1', '--local-headers-recursive',
                        '-I' + str(common), '-I' + str(inputs), *system_flags,
                        '-o', destination, '--', *flags])
                    diagnostic = result.stderr.decode(errors='replace')
                    assert 'fatal error:' not in diagnostic, (name, diagnostic)
                    if kind == 'refuse':
                        assert result.returncode != 0, (name, 'unproved use admitted')
                        if name in ('definition_printf_macro', 'target_macro'):
                            macro = 'printf' if name == 'definition_printf_macro' else target.rsplit('::', 1)[-1]
                            assert ("CUDA sample-helper SM table boundary: input macro '" + macro +
                                    "' prevents the helper transaction") in diagnostic, (name, diagnostic)
                            assert 'version-preserving legacy SM table call(s)' not in diagnostic, (name, diagnostic)
                        assert (sha(output), inventory(bundle)) == prior, (name, 'previous output changed')
                        if destination != output:
                            assert not destination.exists() and not Path(str(destination)+'.headers').exists(), name
                    else:
                        assert result.returncode == 0, (name, diagnostic)
                        text = destination.read_text()
                        if kind == 'commit':
                            assert target in text and '#include <helper_cuda.h>' not in text, (name, text)
                            assert 'version-preserving legacy SM table call(s)' in diagnostic, (name, diagnostic)
                        else:
                            assert target not in text and '#include <helper_cuda.h>' in text, (name, text)
                            assert 'ftoi(1.5f)+_ConvertSMVer2Cores(7,0)' in text, (name, text)
                            assert "residual helper declaration 'ftoi'" in diagnostic, (name, diagnostic)
                            assert 'CUDA sample-helper SM table boundary:' not in diagnostic, (name, diagnostic)
                row['new_frontend_run'] = True
            assert inventory(inputs) == frozen, (name, 'source mutation')
            assert not list(case.glob('.ascify-local-closure*')), (name, 'staging leaked')
            row['passed'] = True
            rows.append(row)
    assert (inventory(helpers), sha(version134)) == original_helpers, 'frozen helper mutation'
    result = {'passed': True, 'total': len(rows), 'cases': rows,
              'scope': 'original syntax preparation only' if a.syntax_only_preparation else 'new frontend matrix only',
              'new_frontend_acceptance': not a.syntax_only_preparation,
              'native_product_built_by_this_test': False, 'device_executed': False,
              'original_source_parser_profile': str(a.profile_host_math),
              'original_source_parser_profile_sha256': sha(a.profile_host_math),
              'original_sample_successes': 0}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SM helper:', len(rows), result['scope'], 'PASS; no NPU')


if __name__ == '__main__':
    main()
