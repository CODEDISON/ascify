#!/usr/bin/env python3
"""Syntax-first contextual texture proof and transactional rejection tests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

if not __debug__:
    raise RuntimeError('Context tests require assertions')
ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / 'tests/rewrite/fixtures/nvidia_samples/Common'
HEADER = '#pragma once\n#include <cuda_runtime.h>\n#include <helper_cuda.h>\n#include <helper_functions.h>\n'
BODY = '''int main() {
  cudaArray *array = nullptr;
  cudaChannelFormatDesc format = cudaCreateChannelDesc<float>();
  checkCudaErrors(cudaMallocArray(&array, &format, 8, 4));
  checkCudaErrors(cudaFreeArray(array));
  return 0;
}
'''
SOURCE = '#include "context.cuh"\n' + BODY
PACK = '#pragma pack(1)\nstruct Packed { unsigned short a; unsigned int b; };\n#pragma pack()\n'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(path):
    return {str(p.relative_to(path)): sha(p) for p in path.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'cuda-path', 'resource-dir', 'clang', 'evidence-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--helper-root', type=Path, default=HELPERS)
    parser.add_argument('--ccec', type=Path)
    parser.add_argument('--target-arg', action='append', default=[])
    parser.add_argument('--target-include', action='append', default=[])
    a = parser.parse_args()
    out = a.evidence_dir.resolve()
    out.mkdir()
    helpers = a.helper_root.resolve()
    helpers_before = inventory(helpers)
    # label, source, header, expected success, conversion options, source valid
    cases = [
        ('transitive_texture', SOURCE, HEADER, True, [], True),
        ('transitive_layered_flag', SOURCE.replace(
            'cudaMallocArray(&array, &format, 8, 4)',
            'cudaMalloc3DArray(&array, &format, make_cudaExtent(8, 4, 2), cudaArrayLayered)'),
            HEADER, True, [], True),
        ('transitive_packed_texture', SOURCE.replace('return 0;', 'return sizeof(Packed)==6?0:1;'),
            HEADER + PACK, True, [], True),
        ('redefined_helper', SOURCE.replace(BODY,
            '#undef checkCudaErrors\n#define checkCudaErrors(x) ((void)(x))\n' + BODY),
            HEADER, False, [], True),
        ('wrapper_macro', SOURCE.replace(BODY,
            '#define WRAPPED(x) checkCudaErrors(x)\n' + BODY.replace('checkCudaErrors(', 'WRAPPED(')),
            HEADER, False, [], True),
        ('helper_condition', SOURCE + '\n#ifdef checkCudaErrors\nint helper_observed;\n#endif\n',
            HEADER, False, [], True),
        ('flag_condition', SOURCE + '\n#ifdef cudaArrayLayered\nint flag_observed;\n#endif\n',
            HEADER, False, [], True),
        ('texture_address', SOURCE + '\nauto texture_function = &cudaMallocArray;\n',
            HEADER, False, [], True),
        ('nested_real_entry', SOURCE, HEADER + '#include "extra.cuh"\n', False, [], True),
        ('repeated_direct_include', SOURCE.replace('#include "context.cuh"',
            '#include "context.cuh"\n#include "context.cuh"'), HEADER, False, [], True),
        ('include_level', SOURCE, HEADER + 'constexpr int observed_level = __INCLUDE_LEVEL__;\n',
            False, [], True),
        ('invalid_packed_field', SOURCE, HEADER + PACK.replace('unsigned int b', 'float b'),
            False, [], True),
        ('nonrecursive', SOURCE, HEADER, False, ['nonrecursive'], True),
        ('direct_only', SOURCE, HEADER, False, ['direct_only'], True),
        ('retained_preprocessor', SOURCE, HEADER, False, ['retained_preprocessor'], True),
        ('malformed_source', SOURCE + '\nvoid malformed( {\n', HEADER, False, [], False),
    ]
    independent = '#pragma once\n#include "context.cuh"\ninline int independent_sibling_value() { return 7; }\n'
    siblings = {
        'once_independent_sibling': {'sibling.cuh': independent},
        'once_two_independent_siblings': {'sibling.cuh': independent,
            'second.cuh': independent.replace('independent_sibling_value', 'second_sibling_value')},
        'once_packed_sibling': {'sibling.cuh': independent},
        'once_sibling_real_nested_entry': {'sibling.cuh': independent + '#include "extra.cuh"\n',
            'extra.cuh': 'constexpr int extra_value = 3;\n'},
        'once_sibling_include_level': {'sibling.cuh': independent + 'constexpr int observed_level = __INCLUDE_LEVEL__;\n'},
        'once_sibling_duplicate_root': {'sibling.cuh': independent},
        'once_sibling_macro_include': {'sibling.cuh': independent.replace('#include "context.cuh"',
            '#define CONTEXT_HEADER "context.cuh"\n#include CONTEXT_HEADER')},
        'once_sibling_angle_include': {'sibling.cuh': independent.replace('"context.cuh"', '<context.cuh>')},
        'once_sibling_not_direct_root': {'sibling.cuh': '#include "nested.cuh"\n', 'nested.cuh': independent},
    }
    for label, extra in siblings.items():
        prefix = '#include "context.cuh"\n#include "sibling.cuh"\n'
        if label == 'once_two_independent_siblings': prefix += '#include "second.cuh"\n'
        if label == 'once_sibling_duplicate_root': prefix += '#include "sibling.cuh"\n'
        cases.append((label, prefix + BODY, HEADER + (PACK if label == 'once_packed_sibling' else ''),
            label in ('once_independent_sibling', 'once_two_independent_siblings', 'once_packed_sibling'), [], True))
    rows = []

    def run(argv, directory):
        directory.mkdir()
        argv = list(map(str, argv))
        (directory / 'argv.json').write_text(json.dumps(argv) + '\n')
        result = subprocess.run(argv, capture_output=True)
        (directory / 'stdout.log').write_bytes(result.stdout)
        (directory / 'stderr.log').write_bytes(result.stderr)
        (directory / 'process.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
        return result

    seed_output = seed_bundle = None
    for label, source, header, accept, options, source_valid in cases:
        case = out / label
        case.mkdir()
        inputs = case / 'input'
        inputs.mkdir()
        root = inputs / 'root.cu'
        root.write_text(source)
        (inputs / 'context.cuh').write_text(header)
        if label == 'nested_real_entry':
            (inputs / 'extra.cuh').write_text('constexpr int extra_value = 3;\n')
        for name, content in siblings.get(label, {}).items():
            (inputs / name).write_text(content)
        initial = inventory(inputs)
        syntax = run([a.clang, '-x', 'cuda', '--cuda-host-only', '-nocudalib',
            '--cuda-path=' + str(a.cuda_path), '-resource-dir=' + str(a.resource_dir),
            '-std=c++17', '-I' + str(helpers), '-I' + str(inputs), '-fsyntax-only', root],
            case / 'source_syntax')
        assert (syntax.returncode == 0) == source_valid, (label, syntax.stderr.decode(errors='replace'))
        output = case / 'published.cce'
        bundle = Path(str(output) + '.headers')
        if not accept:
            # Preserve an actual managed output, not an unmanaged placeholder
            # whose publication refusal could hide an unsafe admission.
            shutil.copy2(seed_output, output)
            shutil.copytree(seed_bundle, bundle)
            prior = (sha(output), inventory(bundle))
        destinations = [output] if accept else [output, case / 'fresh.cce']
        for index, destination in enumerate(destinations):
            argv = [a.binary, root, '--cuda-path=' + str(a.cuda_path),
                '--clang-resource-directory=' + str(a.resource_dir),
                '--frontend-compat=ascify-admitted-v1', '--target-policy=dav-c310-vec',
                '--simt-math=fast', '-I' + str(helpers), '-I' + str(inputs), '-o', destination]
            if 'retained_preprocessor' not in options:
                argv.append('--default-preprocessor')
            if 'nonrecursive' not in options:
                argv.append('--local-headers' if 'direct_only' in options else '--local-headers-recursive')
            argv += ['--', '-std=c++17', '-I' + str(helpers), '-I' + str(inputs)]
            result = run(argv, case / ('convert_' + str(index)))
            diagnostic = result.stderr.decode(errors='replace')
            assert inventory(inputs) == initial, (label, 'input mutated')
            if accept:
                assert result.returncode == 0, (label, diagnostic)
                text = destination.read_text()
                marker = (bundle / '.ascify-local-closure').read_text()
                assert diagnostic.count('contextual helper discovery: evidence only') == 2, (label, diagnostic)
                assert 'joint helper transaction' in result.stdout.decode(), label
                assert 'context_proof=expanded-tokens-and-final-macros-v1' in marker, label
                for path in (root, inputs / 'context.cuh'):
                    assert 'context_input=' + str(path) + '\t' + sha(path) + '\n' in marker, label
                assert '#include "context.cuh"' not in text and '#include <helper_cuda.h>' not in text, label
                assert 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' in text, label
                if label.startswith('once_'):
                    assert 'independent_sibling_value' in text and '#include \"sibling.cuh\"' not in text, label
                    for name in siblings[label]:
                        path = inputs / name
                        assert 'context_inline=' + str(path) + '\t' in marker, label
                if label in ('transitive_packed_texture', 'once_packed_sibling'):
                    assert PACK in text and 'Ascify preserved packed' in text, label
                if a.ccec:
                    obj = case / 'target.o'
                    compiled = run([a.ccec, '-O2', '-std=c++17', '-x', 'dpp',
                        '--cce-aicore-arch=dav-c310-vec', *a.target_arg, '-c', destination, '-o', obj,
                        *sum((['-I', x] for x in a.target_include), [])], case / 'target_compile')
                    assert compiled.returncode == 0 and obj.read_bytes().startswith(b'\x7fELF'), (
                        label, compiled.stderr.decode(errors='replace'))
            else:
                assert result.returncode != 0, (label, 'unexpected acceptance')
                assert 'error:' in diagnostic or 'ERROR' in diagnostic, (label, diagnostic)
                assert (sha(output), inventory(bundle)) == prior, (label, 'old artifacts changed')
                if destination != output:
                    assert not destination.exists() and not Path(str(destination) + '.headers').exists(), label
            assert not list(case.glob('.ascify-local-closure*')), (label, 'staging leaked')
        if label == 'transitive_texture':
            seed_output, seed_bundle = output, bundle
        rows.append({'name': label, 'source_syntax_valid': source_valid,
                     'expected_acceptance': accept, 'passed': True})
    assert inventory(helpers) == helpers_before, 'helper fixture changed'
    (out / 'summary.json').write_text(json.dumps({'passed': True, 'cases': rows,
        'total': len(rows), 'device_executed': False}, indent=2) + '\n')
    print('contextual helper textures:', len(rows), 'cases PASS; no NPU')


if __name__ == '__main__':
    main()
