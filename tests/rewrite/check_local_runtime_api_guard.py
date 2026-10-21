#!/usr/bin/env python3
"""Real converter proof for Runtime includes in selected local-header guards."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


def hashes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--cuda-path', required=True)
    parser.add_argument('--resource-dir', required=True)
    args = parser.parse_args()
    binary = args.binary
    cuda = args.cuda_path
    resource = args.resource_dir
    include = '#include <cuda_runtime_api.h>\n'
    body = 'cudaError_t release(void *p) { return cudaFree(p); }\n'

    def guarded(text, name='LOCAL_RUNTIME_GUARD'):
        return f'#ifndef {name}\n#define {name}\n' + text + '#endif\n'

    cases = {
        'ordinary_guard': (guarded(include + body), True),
        'reserved_style_guard': (guarded(include + body, '_REDUCE_KERNEL_H_'), True),
        'comments': ('/* source license */\n' + guarded(
            '// first dependency\n' + include + body) + '// end comment\n', True),
        'quoted_sdk_include': (guarded(include.replace('<cuda_runtime_api.h>',
            '"cuda_runtime_api.h"') + body), True),
        'separate_nested_condition': (guarded(include + '#if 1\n' + body + '#endif\n'), True),
        'guard_and_nested_include': (guarded('#if 1\n' + include + '#endif\n' + body), False),
        'feature_condition_only': ('#if 1\n' + include + body + '#endif\n', False),
        'feature_ifndef_only': ('#ifndef FEATURE_SWITCH\n' + include + body + '#endif\n', False),
        'mismatched_define': ('#ifndef GUARD_A\n#define GUARD_B\n' + include + body + '#endif\n', False),
        'nonempty_define': ('#ifndef GUARD_A\n#define GUARD_A 1\n' + include + body + '#endif\n', False),
        'function_guard': ('#ifndef GUARD_A\n#define GUARD_A()\n' + include + body + '#endif\n', False),
        'outer_else': (guarded(include + body + '#else\nint fallback;\n'), False),
        'outer_elif': (guarded(include + body + '#elif 0\nint fallback;\n'), False),
        'tokens_after_guard': (guarded(include + body) + 'int outside;\n', False),
        'tokens_before_guard': ('int outside;\n' + guarded(include + body), False),
        'directive_after_guard': (guarded(include + body) + '#define AFTER 1\n', False),
        'guard_undefined': (guarded(include + '#undef LOCAL_RUNTIME_GUARD\n' + body), False),
        'guard_digraph_undefined': (guarded(include + '%:undef LOCAL_RUNTIME_GUARD\n' + body), False),
        'guard_redefined': (guarded(include + '#define LOCAL_RUNTIME_GUARD 1\n' + body), False),
        'guard_push': (guarded(include + '#pragma push_macro("LOCAL_RUNTIME_GUARD")\n' + body), False),
        'guard_pop': (guarded(include + '#pragma pop_macro("LOCAL_RUNTIME_GUARD")\n' + body), False),
        'guard_digraph_pop': (guarded(include + '%:pragma pop_macro("LOCAL_RUNTIME_GUARD")\n' + body), False),
        'guard_escaped_pop': (guarded(include + '#pragma pop_macro("LOCAL_RUNTIME_\\107UARD")\n' + body), False),
        'guard_pragma_operator': (guarded(include + '_Pragma("pop_macro(\\"LOCAL_RUNTIME_GUARD\\")")\n' + body), False),
        'guard_macro_pragma': (guarded(include + '#define RESTORE _Pragma("pop_macro(\\"LOCAL_RUNTIME_GUARD\\")")\nRESTORE\n' + body), False),
        'inactive_ms_pragma': (guarded(include + '#if 0\n__pragma(pop_macro("LOCAL_RUNTIME_GUARD"))\n#endif\n' + body), False),
        'inactive_macro_pragma': (guarded(include + '#if 0\n#define RESTORE _Pragma("pop_macro(\\"LOCAL_RUNTIME_GUARD\\")")\n#endif\n' + body), False),
        'macro_include': (guarded('#define HEADER <cuda_runtime_api.h>\n#include HEADER\n' + body), False),
    }
    with tempfile.TemporaryDirectory(prefix='ascify-local-runtime-guard-') as tmp:
        work = Path(tmp)
        inputs = work / 'inputs'
        inputs.mkdir()
        root = inputs / 'root.cu'
        leaf = inputs / 'leaf.cuh'
        output = work / 'converted.cce'
        bundle = Path(str(output) + '.headers')
        root.write_text('#include "leaf.cuh"\nint main() { return 0; }\n')

        def run(destination, selected=True, extra=()):
            argv = [binary, str(root), f'--cuda-path={cuda}',
                    f'--clang-resource-directory={resource}',
                    '--default-preprocessor', '-o', str(destination)]
            if selected:
                argv.append('--local-headers-recursive')
            argv += ['--', '-std=c++17', '-I' + str(inputs), *extra]
            before = hashes(inputs.rglob('*'))
            result = subprocess.run(argv, text=True, capture_output=True)
            assert hashes(inputs.rglob('*')) == before, 'input bytes changed'
            assert not list(work.glob('.ascify-local-closure*')), 'staging leaked'
            return result

        published = None
        for name, (text, accepted) in cases.items():
            leaf.write_text(text)
            result = run(output)
            if accepted:
                assert result.returncode == 0, (name, result.stdout, result.stderr)
                artifacts = [path for path in bundle.rglob('*') if path.is_file()
                             and path.name != '.ascify-local-closure']
                assert len(artifacts) == 1, (name, artifacts)
                emitted = artifacts[0].read_text()
                assert '#include <cuda_runtime_api.h>' not in emitted
                assert '#include "cuda_runtime_api.h"' not in emitted
                assert 'ascify/ascify_cuda_compat.hpp' in emitted, (name, emitted)
                guard = '_REDUCE_KERNEL_H_' if name == 'reserved_style_guard' else 'LOCAL_RUNTIME_GUARD'
                assert f'#ifndef {guard}' in emitted and f'#define {guard}' in emitted
                assert 'ascify::cudaFree(p)' in emitted, (name, emitted)
                published = hashes([output, *bundle.rglob('*')])
            else:
                assert result.returncode != 0, (name, 'unsafe guard accepted')
                assert 'CUDA runtime-API header boundary' in result.stderr, (name, result.stderr)
                assert hashes([output, *bundle.rglob('*')]) == published, name
                fresh = work / (name + '.cce')
                fresh_result = run(fresh)
                assert fresh_result.returncode != 0 and not fresh.exists(), name
                assert not Path(str(fresh) + '.headers').exists(), name
            print(name + ': ' + ('admitted' if accepted else 'rejected atomically'))

        # The exception is confined to an actual selected local header. A main
        # source with the same conditional must retain the existing refusal.
        root.write_text(guarded(include + body))
        result = run(work / 'main_guard.cce', selected=False)
        assert result.returncode != 0 and 'CUDA runtime-API header boundary' in result.stderr
        assert not (work / 'main_guard.cce').exists()
        print('main_source_guard: existing refusal preserved')

        # Exact physical SDK resolution remains mandatory even inside a valid
        # guard. This file is task-local; the real CUDA SDK is never changed.
        root.write_text('#include "leaf.cuh"\nint main() { return 0; }\n')
        leaf.write_text(guarded(include + body))
        shadow = inputs / 'cuda_runtime_api.h'
        shadow.write_text('// caller-owned shadow, not an SDK provider\n')
        result = run(work / 'shadow.cce')
        assert result.returncode != 0 and 'physical --cuda-path' in result.stderr
        assert not (work / 'shadow.cce').exists()
        assert not (work / 'shadow.cce.headers').exists()
        print('physical_sdk_shadow: rejected')

    print('local Runtime guard conversion boundaries passed; no device execution')


if __name__ == '__main__':
    main()
