#!/usr/bin/env python3
"""Syntax-first host packed-main identity checks; never opens an NPU.

Physical content/inode replacement and forged parsed-byte checks additionally
run in the existing host_cxx_root_input_test C++ witness test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

if not __debug__:
    raise RuntimeError('Host packed checks require assertions')

PACK = ('#pragma pack(1)\n'
        'struct Packed { unsigned short tag; unsigned int length; };\n'
        '#pragma pack()\n')
MAIN = 'int main(){return sizeof(Packed)==6?0:1;}\n'
PREFIX = '#ifndef HOST_PACKED_RECORD\n#define HOST_PACKED_RECORD\n'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'clang', 'cuda-path', 'resource-dir', 'evidence-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    out = a.evidence_dir.resolve()
    out.mkdir()
    rows = []
    cases = [
        ('direct_cpp', True, False, PACK + MAIN, [], [], None),
        ('selected_header', True, True, PREFIX + PACK + '#endif\n', [], [], None),
        ('header_as_root', True, False, PACK + MAIN, [], [], '.h'),
        ('direct_float_field', False, False, PACK.replace('unsigned int length', 'float length') + MAIN, [], [], None),
        ('selected_float_field', False, True, PACK.replace('unsigned int length', 'float length'), [], [], None),
        ('shadow_witness_syntax', False, False, '#define __builtin_offsetof(...) 0\n' + PACK + MAIN, [], [], None),
        ('pack_option', False, False, PACK + MAIN, ['-fpack-struct=2'], [], None),
        ('module_option', False, False, PACK + MAIN, ['-fmodules'], [], None),
        ('ordinary_host_without_cuda_preprocessor_flag', True, False,
            '#if 0\n#error host must skip inactive conditionals\n#endif\n' + PACK + MAIN,
            [], ['no_default_flag'], None),
        ('host_amap', False, False, PACK + MAIN, [], ['amap'], None),
        ('caller_overlay_root', False, False, PACK + MAIN, [], ['overlay'], None),
        ('caller_overlay_header', False, True, PACK, [], ['overlay'], None),
        ('caller_remap', False, False, PACK + MAIN, [], ['remap'], None),
    ]

    def run(argv, directory):
        directory.mkdir()
        argv = list(map(str, argv))
        (directory / 'argv.json').write_text(json.dumps(argv) + '\n')
        result = subprocess.run(argv, capture_output=True, timeout=120)
        (directory / 'stdout.log').write_bytes(result.stdout)
        (directory / 'stderr.log').write_bytes(result.stderr)
        (directory / 'process.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
        return result

    for name, accept, header_case, text, flags, options, suffix in cases:
        case = out / name
        case.mkdir()
        inputs = case / 'inputs'
        inputs.mkdir()
        source = inputs / ('main' + (suffix or '.cpp'))
        header = inputs / 'record.h'
        if header_case:
            source.write_text('#include "record.h"\n' + MAIN)
            header.write_text(text)
        else:
            source.write_text(text)
        shadow = inputs / 'shadow.cpp'
        overlay = inputs / 'overlay.json'
        if 'overlay' in options:
            shadow.write_text(PACK)
            overlay.write_text(json.dumps({'version': 0, 'use-external-names': False,
                'roots': [{'type': 'file', 'name': str(header if header_case else source),
                           'external-contents': str(shadow)}]}))
        if 'remap' in options:
            shadow.write_text(PACK + MAIN)
        before = inventory(inputs)
        syntax = run([a.clang, '-x', 'c++', '-std=c++17', *flags,
            '-I' + str(inputs), '-fsyntax-only', source], case / 'source_syntax')
        assert syntax.returncode == 0, (name, syntax.stderr.decode(errors='replace'))
        output = case / 'published.cpp'
        bundle = Path(str(output) + '.headers')
        if not accept:
            output.write_text('unchanged previous root\n')
            bundle.mkdir()
            (bundle / 'previous.txt').write_text('unchanged previous bundle\n')
            prior = sha(output), inventory(bundle)
        destinations = [output] if accept else [output, case / 'fresh.cpp']
        for index, destination in enumerate(destinations):
            argv = [a.binary, source, '--input-language=c++',
                '--cuda-path=' + str(a.cuda_path),
                '--clang-resource-directory=' + str(a.resource_dir),
                '--target-policy=dav-c310-vec', '--simt-math=fast',
                '--frontend-compat=ascify-admitted-v1', '--local-headers-recursive',
                '-I' + str(inputs), '-o', destination]
            if 'no_default_flag' not in options:
                argv.append('--default-preprocessor')
            if 'amap' in options:
                argv.append('--amap')
            argv += ['--', '-x', 'c++', '-std=c++17', '-I' + str(inputs), *flags]
            if 'overlay' in options:
                argv += ['-ivfsoverlay', str(overlay)]
            if 'remap' in options:
                argv += ['-Xclang', '-remap-file', '-Xclang', str(source) + ';' + str(shadow)]
            converted = run(argv, case / ('convert_' + str(index)))
            assert inventory(inputs) == before, (name, 'original input mutation')
            diagnostic = converted.stderr.decode(errors='replace')
            if accept:
                assert converted.returncode == 0, (name, diagnostic)
                if header_case:
                    children = list(bundle.rglob('record.h.dpp'))
                    assert len(children) == 1, (name, children)
                    proof = children[0].read_text()
                    assert PACK not in destination.read_text(), (name, 'header was silently flattened')
                else:
                    proof = destination.read_text()
                assert PACK in proof and 'Ascify preserved packed record' in proof, name
                final = run([a.clang, '-x', 'c++', '-std=c++17', *flags,
                    '-I' + str(inputs), '-fsyntax-only', destination], case / 'generated_syntax')
                assert final.returncode == 0, (name, final.stderr.decode(errors='replace'))
            else:
                assert converted.returncode != 0, (name, 'unexpected admission')
                if 'overlay' in options:
                    assert 'VFS overlay' in diagnostic, (name, diagnostic)
                elif 'remap' in options or 'amap' in options:
                    assert 'host C++ input language conflicts' in diagnostic, (name, diagnostic)
                else:
                    assert 'packed-record' in diagnostic, (name, diagnostic)
                assert (sha(output), inventory(bundle)) == prior, (name, 'old outputs changed')
                if destination != output:
                    assert not destination.exists() and not Path(str(destination) + '.headers').exists(), name
            assert not list(case.glob('.ascify-local-closure*')), (name, 'staging leaked')
        rows.append({'name': name, 'passed': True, 'original_syntax_passed': True,
                     'expected_acceptance': accept})
    (out / 'summary.json').write_text(json.dumps({'passed': True, 'cases': rows,
        'total': len(rows), 'device_executed': False}, indent=2) + '\n')
    print('Host packed main/header identity:', len(rows), 'syntax-first cases PASS; no NPU')


if __name__ == '__main__':
    main()
