#!/usr/bin/env python3
"""Original CUDA syntax, real frontend and literal-MAX CPU oracle only.

No CCEC or CUDA/NPU program is executed. The CPU oracle consumes the original
observed macro and the literal replay written into the published source; it
does not claim a target preprocessor/dependency/ABI or complete Sample result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


SEMANTIC = r'''
#include <cstdint>
#include <cstdio>
#include <cstring>
static float value(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
static uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
static int lc=0,rc=0;
static float left(){++lc;return 3.0f;}
static float right(){++rc;return 3.0f;}
int main(){
 const float nan=value(0x7fc00001U),minus=value(0x80000000U),plus=0.0f;
 std::printf("%08x %08x %08x ",bits(MAX(nan,2.0f)),bits(MAX(2.0f,nan)),bits(MAX(minus,plus)));
 float answer=MAX(left(),right());
 std::printf("%08x %d %d %d\n",bits(answer),lc,rc,MAX(1|2,4));
}
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'clang', 'cuda-path', 'resource-dir', 'helper-root', 'evidence-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    assert __debug__, 'Assertions must remain enabled'
    for key in ('binary', 'clang', 'cuda_path', 'resource_dir', 'helper_root'):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    out = args.evidence_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec_path = Path(__file__).parent / 'fixtures/direct_image_provider_matrix.json'
    spec = json.loads(spec_path.read_text())
    cases = spec['cases']
    assert len(cases) == 15 and sum(c['expected_new_graph_or_legacy_acceptance'] for c in cases) == 5
    tools = [args.binary, args.clang, Path(__file__).resolve(), spec_path,
             args.resource_dir / 'include/__clang_cuda_runtime_wrapper.h',
             args.cuda_path / 'include/cuda_runtime.h', args.cuda_path / 'include/cuda_runtime_api.h']
    before = {str(p): sha(p) for p in tools}
    helper_before = inventory(args.helper_root)
    state = dict(schema='ascify.direct-image.frontend-cpu.v1', passed=False,
                 binary_sha256=sha(args.binary), cases=[], source_inputs=before,
                 original_sample_successes=0, compiled_target=False, device_executed=False,
                 target_preprocessor_verified=False)

    def command(argv, path):
        path.mkdir()
        (path / 'argv.json').write_text(json.dumps([str(x) for x in argv], indent=2) + '\n')
        result = subprocess.run(list(map(str, argv)), capture_output=True, timeout=180)
        (path / 'stdout.log').write_bytes(result.stdout)
        (path / 'stderr.log').write_bytes(result.stderr)
        (path / 'process.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
        return result

    seed_output = seed_bundle = None
    try:
        for case in cases:
            label = case['name']
            accepted = case['expected_new_graph_or_legacy_acceptance']
            directory = out / label
            directory.mkdir()
            inputs = directory / 'input'
            inputs.mkdir()
            source = inputs / 'root.cu'
            source.write_text(case['source'])
            helpers = args.helper_root
            setup = case['fixture_only_setup'] or {}
            if setup.get('copy_helpers_to_private_fixture'):
                helpers = inputs / 'changed_helpers'
                shutil.copytree(args.helper_root, helpers)
                changed = helpers / setup['mutate']
                changed.write_bytes(changed.read_bytes() + b'\n// private fixture mutation\n')
            extra = [x.replace('{helpers}', str(helpers)).replace('{case}', str(inputs))
                     for x in case['compiler_flags']]
            if setup.get('copy_exact_image_bytes_to_private_file'):
                remapped = inputs / 'remapped_image.h'
                remapped.write_bytes((helpers / 'helper_image.h').read_bytes())
                extra += ['-Xclang', '-remap-file', '-Xclang',
                          str(helpers / 'helper_image.h') + ';' + str(remapped)]
            snapshot = inventory(inputs)
            language = ['-x', 'cuda', '-std=c++17', '-fgpu-rdc',
                        '-I' + str(helpers), '-I' + str(inputs)]
            cccl = args.cuda_path / 'include/cccl'
            if (cccl / 'cuda/std/type_traits').is_file():
                language += ['-I' + str(cccl)]
            original = [args.clang, '--cuda-host-only', '-nocudalib',
                        '--cuda-path=' + str(args.cuda_path),
                        '-resource-dir=' + str(args.resource_dir),
                        '-Wno-unknown-cuda-version', *language, *extra]
            syntax = command([*original, '-fsyntax-only', source], directory / 'original_syntax')
            assert syntax.returncode == 0, (label, 'original CUDA syntax invalid', syntax.stderr)
            output = directory / 'published.cce'
            bundle = Path(str(output) + '.headers')
            if not accepted:
                assert seed_output is not None and seed_bundle is not None
                shutil.copy2(seed_output, output)
                shutil.copytree(seed_bundle, bundle)
                prior = sha(output), inventory(bundle)
            destinations = [output] if accepted else [output, directory / 'fresh.cce']
            for attempt, destination in enumerate(destinations):
                receipt = directory / ('receipt_' + str(attempt) + '.json')
                argv = [args.binary, source, '--cuda-path=' + str(args.cuda_path),
                        '--clang-resource-directory=' + str(args.resource_dir),
                        '--frontend-compat=ascify-admitted-v1', '--target-policy=dav-c310-vec',
                        '--target-recipe=none', '--simt-math=precise', '--default-preprocessor',
                        '--local-headers-recursive', '--migration-receipt=' + str(receipt),
                        '-o', destination, '--', *language, *extra]
                converted = command(argv, directory / ('convert_' + str(attempt)))
                assert inventory(inputs) == snapshot, (label, 'input mutated')
                diagnostic = converted.stderr.decode(errors='replace')
                record = json.loads(receipt.read_text())
                if not accepted:
                    assert converted.returncode != 0 and record['status'] == 'failed', (label, diagnostic)
                    assert 'fatal error:' not in diagnostic, (label, 'unrelated dependency failure', diagnostic)
                    assert (sha(output), inventory(bundle)) == prior, (label, 'managed publication changed')
                    if destination != output:
                        assert not destination.exists() and not Path(str(destination) + '.headers').exists()
                    continue
                assert converted.returncode == 0 and record['status'] == 'succeeded', (label, diagnostic)
                text = destination.read_text()
                assert '#include <helper_cuda.h>' not in text and 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' in text
                assert bundle.is_dir() and not bundle.is_symlink(), (label, 'published managed bundle absent')
                if label.startswith('direct_'):
                    assert '#include <helper_functions.h>' not in text
                    assert '#include <helper_image.h>' in text and '#include <helper_string.h>' in text
                    assert 'Ascify direct-image MAX replay v1:' in diagnostic
                    assert 'functions_root=0' in diagnostic
                    assert b'File included: helper_functions.h' not in converted.stdout
                    image_offset = text.index('#include <helper_image.h>')
                    definitions = re.findall(r'(?m)^#define MAX\([^\r\n]*$', text[:image_offset])
                    assert len(definitions) == 1, (label, 'one exact replay before Image', definitions)
                    observed = command([*original, '-E', '-dM', source], directory / 'original_macros')
                    assert observed.returncode == 0
                    old = re.findall(rb'(?m)^#define MAX\([^\r\n]*$', observed.stdout)
                    assert len(old) == 1
                    replay = definitions[0].encode()
                    assert re.sub(rb'\s+', b'', old[0]) == re.sub(rb'\s+', b'', replay), (label, 'macro token change')
                    outputs = []
                    for kind, line in [('original', old[0]), ('published_replay', replay)]:
                        host = directory / (kind + '.cpp')
                        host.write_bytes(line + b'\n' + SEMANTIC.encode())
                        executable = directory / kind
                        built = command([args.clang, '-x', 'c++', '-std=c++17', '-O0', host,
                                         '-o', executable], directory / (kind + '_compile'))
                        assert built.returncode == 0, (label, built.stderr)
                        executed = command([executable], directory / (kind + '_run'))
                        assert executed.returncode == 0
                        outputs.append(executed.stdout)
                    assert outputs[0] == outputs[1] == b'40000000 7fc00001 00000000 40400000 1 2 3\n', outputs
                else:
                    assert 'Ascify direct-image MAX replay v1:' not in diagnostic
                if seed_output is None:
                    seed_output, seed_bundle = output, bundle
            state['cases'].append(dict(name=label, passed=True, accepted=accepted,
                                       original_syntax_returncode=syntax.returncode, input_sha256=snapshot))
            (out / 'summary.json').write_text(json.dumps(state, indent=2) + '\n')
        assert inventory(args.helper_root) == helper_before, 'original helpers changed'
        assert all(sha(Path(p)) == value for p, value in before.items()), 'tools/SDK changed'
        state['passed'] = True
    finally:
        (out / 'summary.json').write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
