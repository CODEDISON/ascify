#!/usr/bin/env python3
"""Original-position host system-macro regressions; CPU only, never opens an NPU."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

HOST = ('#if defined(__CUDACC__) || defined(__CUDA__) || defined(__CUDA_ARCH__)\n'
        '#error host language contract changed\n#endif\n')
NULL_LEAF = ('#ifndef LEAF_NULL_H\n#define LEAF_NULL_H\n'
             'inline bool empty(void* value) { return value == NULL; }\n#endif\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    binary = Path(os.environ['ASCIFY_BINARY']).resolve(strict=True)
    cuda = Path(os.environ['ASCIFY_CUDA_PATH']).resolve(strict=True)
    resource = Path(os.environ['ASCIFY_CLANG_RESOURCE_DIRECTORY']).resolve(strict=True)
    cxx = os.environ.get('CXX', 'c++')
    temporary = tempfile.TemporaryDirectory(prefix='ascify-host-system-macro-')
    evidence = os.environ.get('ASCIFY_HOST_SYSTEM_MACRO_EVIDENCE')
    work = (Path(evidence) if evidence else Path(temporary.name) / 'cases').resolve()
    work.mkdir(parents=True, exist_ok=False)
    summary = {'passed': False, 'host_only': True, 'cases': [],
               'binary_sha256': sha(binary), 'original_sample_successes': 0}

    def save():
        (work / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    def invoke(directory, stage, argv):
        (directory / (stage + '.argv.json')).write_text(json.dumps(list(map(str, argv)), indent=2) + '\n')
        result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=120)
        (directory / (stage + '.stdout')).write_text(result.stdout)
        (directory / (stage + '.stderr')).write_text(result.stderr)
        return result

    def run(name, root_body, leaf, *, good, contextual=False, diagnostic='', provider=None, system=False):
        directory = work / name
        directory.mkdir()
        source, header, output = (directory / n for n in ('root.cpp', 'leaf.h', 'translated.cpp'))
        source.write_text(HOST + root_body)
        header.write_text(leaf)
        paths, flags = [source, header], []
        if provider is not None:
            fake = directory / 'fake'
            fake.mkdir()
            fake_header = fake / 'fake_std.h'
            fake_header.write_text(provider)
            paths.append(fake_header)
            flags = ['-isystem' if system else '-I', str(fake)]
        before = {str(p): sha(p) for p in paths}
        row = {'name': name, 'expected_admission': good, 'expected_context': contextual,
               'passed': False, 'source_sha256': before}
        summary['cases'].append(row)
        save()
        original = invoke(directory, 'original_syntax',
            [cxx, '-std=c++17', '-x', 'c++', '-fsyntax-only', *flags, source])
        row['original_syntax_rc'] = original.returncode
        assert original.returncode == 0, (name, original.stderr)
        converted = invoke(directory, 'conversion', [binary, source,
            '--cuda-path=' + str(cuda), '--clang-resource-directory=' + str(resource),
            '--local-headers-recursive', '--default-preprocessor',
            '--target-policy=dav-c310-vec', '--simt-math=fast', '--target-recipe=none',
            '-o', output, '--', '-std=c++17', '-x', 'c++', *flags])
        row['conversion_rc'] = converted.returncode
        assert (converted.returncode == 0) == good, (name, converted.stdout, converted.stderr)
        assert {str(p): sha(p) for p in paths} == before, (name, 'original input changed')
        bundle = Path(str(output) + '.headers')
        if not good:
            assert not output.exists() and not bundle.exists(), (name, 'failed output published')
            assert diagnostic in converted.stderr, (name, converted.stderr)
        else:
            text = output.read_text()
            marker = (bundle / '.ascify-local-closure').read_text()
            if contextual:
                assert 'context_proof=expanded-tokens-and-final-macros-v1' in marker, (name, marker)
                assert 'context_input=' + str(source) + '\t' + sha(source) in marker, (name, marker)
                assert '#include "leaf.h"' not in text, (name, text)
                assert not re.search(r'^\s*#\s*(?:define|undef)\s+(?:NULL|FLT_EPSILON)\b', text, re.M), text
            else:
                assert 'context_proof=' not in marker, (name, marker)
                assert len(list(bundle.rglob('leaf.h.dpp'))) == 1, (name, marker)
            for stage, input_path in (('original_cpu', source), ('translated_cpu', output)):
                program = directory / stage
                built = invoke(directory, stage + '_build',
                    [cxx, '-std=c++17', '-x', 'c++', *flags, input_path, '-o', program])
                assert built.returncode == 0, (name, built.stderr)
                executed = invoke(directory, stage + '_run', [program])
                assert executed.returncode == 0, (name, executed.stdout, executed.stderr)
            row['output_sha256'] = sha(output)
        row['passed'] = True
        save()
        print(name + ': PASS', flush=True)

    try:
        run('bare_null_parent_cstddef', '#include <cstddef>\n#include "leaf.h"\n'
            'int main() { return empty(nullptr) ? 0 : 1; }\n', NULL_LEAF,
            good=True, contextual=True)
        run('bare_float_parent_cfloat', '#include <cfloat>\n#include "leaf.h"\n'
            'int main() { return epsilon() > 0.0f ? 0 : 1; }\n',
            'inline float epsilon() { return FLT_EPSILON; }\n', good=True, contextual=True)
        run('self_contained_skipped_include', '#include <cstddef>\n#include "leaf.h"\n'
            'int main() { return empty(nullptr) && include_depth == 1 ? 0 : 1; }\n',
            '#ifndef SELF_H\n#define SELF_H\n#include <cstddef>\n'
            'constexpr unsigned include_depth = __INCLUDE_LEVEL__;\n'
            'inline bool empty(void* value) { return value == NULL; }\n#endif\n', good=True)
        run('required_context_include_level', '#include <cstddef>\n#include "leaf.h"\n'
            'int main() { return empty(nullptr) ? 0 : 1; }\n',
            'constexpr unsigned include_depth = __INCLUDE_LEVEL__;\n'
            'inline bool empty(void* value) { return value == NULL; }\n', good=False,
            diagnostic='expanded tokens or final macro state differ')
        run('required_context_pragma', '#include <cstddef>\n#include "leaf.h"\n'
            'int main() { return empty(nullptr) ? 0 : 1; }\n',
            '#pragma once\ninline bool empty(void* value) { return value == NULL; }\n',
            good=False, diagnostic='leaf without pragmas')
        run('required_context_repeated', '#include <cstddef>\n#include "leaf.h"\n'
            '#include "leaf.h"\nint main() { return empty(nullptr) ? 0 : 1; }\n',
            NULL_LEAF, good=False, diagnostic='one direct include occurrence')
        for name, provider, system in (
                ('caller_isystem_provider', '#define PARENT_TOKEN 23\n', True),
                ('late_pragma_system_promotion', '#pragma GCC system_header\n#define PARENT_TOKEN 23\n', False)):
            run(name, '#include <fake_std.h>\n#include "leaf.h"\n'
                'int main() { return token() == 23 ? 0 : 1; }\n',
                'inline int token() { return PARENT_TOKEN; }\n', good=False,
                provider=provider, system=system, diagnostic='PARENT_TOKEN')
        assert sha(binary) == summary['binary_sha256'], 'binary changed during regression'
        summary.update(passed=True, total=8)
        save()
    except Exception as error:
        summary['error'] = str(error)
        save()
        raise
    finally:
        temporary.cleanup()


if __name__ == '__main__':
    main()
