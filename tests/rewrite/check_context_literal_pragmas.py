#!/usr/bin/env python3
"""Syntax-first once/unroll context replay and atomic negative boundaries."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / "tests/rewrite/fixtures/nvidia_samples/Common"


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def main():
    binary, cuda, resource = (os.environ[name] for name in
        ('ASCIFY_BINARY', 'ASCIFY_CUDA_PATH', 'ASCIFY_CLANG_RESOURCE_DIRECTORY'))
    clang = os.environ.get('CLANG', 'clang')
    fixtures = inventory(HELPERS)
    with tempfile.TemporaryDirectory(prefix='ascify-context-pragmas-') as tmp:
        work = Path(tmp)
        inputs = work / 'inputs'
        inputs.mkdir()
        root, child = inputs / 'root.cu', inputs / 'leaf.cuh'
        parent = '#include <helper_cuda.h>\n#include "leaf.cuh"\nint main() { checked(); return 0; }\n'
        body = 'inline void checked() { getLastCudaError("literal context"); }\n'
        loop = ('inline void checked() {\nPRAGMA\n'
                'for (int i=0; i<4; ++i) { getLastCudaError("loop context"); }\n}\n')
        output = work / 'published.cce'
        bundle = Path(str(output) + '.headers')

        def invoke(destination):
            before = inventory(inputs)
            syntax = subprocess.run([clang, '-x', 'cuda', '--cuda-host-only', '-nocudalib',
                '--cuda-path=' + cuda, '-resource-dir=' + resource, '-std=c++17',
                '-I' + str(HELPERS), '-I' + str(inputs), '-fsyntax-only', str(root)],
                capture_output=True, text=True)
            assert syntax.returncode == 0, syntax.stdout + syntax.stderr
            result = subprocess.run([binary, str(root), '--cuda-path=' + cuda,
                '--clang-resource-directory=' + resource, '--default-preprocessor',
                '--local-headers-recursive', '-I' + str(HELPERS), '-I' + str(inputs),
                '-o', str(destination), '--', '-std=c++17', '-I' + str(HELPERS), '-I' + str(inputs)],
                capture_output=True, text=True)
            assert before == inventory(inputs), 'original source changed'
            assert not list(work.glob('.ascify-local-closure*')), 'staging leaked'
            return result

        def success(name, text, retained=()):
            root.write_text(parent)
            child.write_text(text)
            result = invoke(output)
            assert result.returncode == 0, name + '\n' + result.stdout + result.stderr
            published = output.read_text()
            assert 'ASCIFY_NVIDIA_SAMPLE_GET_LAST_CUDA_ERROR' in published, name
            assert '#pragma once' not in published, name
            for fragment in retained:
                assert fragment in published, (name, fragment)
            receipt = (bundle / '.ascify-local-closure').read_text()
            assert 'context_pragma_events_sha256=' in receipt, name
            if '#pragma once' in text:
                assert 'context_once_line=' + str(child) + '\t' in receipt, name
            print(name + ': original syntax and complete context replay passed')

        success('once', '#pragma once\n' + body)
        success('unroll', loop.replace('PRAGMA', '#pragma unroll'), ('#pragma unroll',))
        success('once_unroll4', '#pragma once\n' + loop.replace('PRAGMA', '#pragma unroll 4'),
                ('#pragma unroll 4',))
        success('inactive_literal_unroll', '#if 0\n#pragma unroll 8\n#endif\n' + body,
                ('#pragma unroll 8',))
        frozen = (hashlib.sha256(output.read_bytes()).hexdigest(), inventory(bundle))

        def reject(name, text, parent_text=parent, reason=None):
            root.write_text(parent_text)
            child.write_text(text)
            for destination in (output, work / (name + '.cce')):
                result = invoke(destination)
                assert result.returncode > 0, name + ': unexpected acceptance'
                if reason:
                    assert reason in result.stderr, name + '\n' + result.stderr
                if destination != output:
                    assert not destination.exists() and not Path(str(destination) + '.headers').exists(), name
                assert (hashlib.sha256(output.read_bytes()).hexdigest(), inventory(bundle)) == frozen, name
            print(name + ': semantic rejection and old output preserved')

        reject('pragma_payload_changes', loop.replace('PRAGMA',
            '#if __INCLUDE_LEVEL__ == 0\n#pragma unroll 4\n#else\n#pragma unroll 8\n#endif'),
            reason='literal pragma payload, order or position differs')
        reject('new_active_unroll', loop.replace('PRAGMA',
            '#if __INCLUDE_LEVEL__ == 0\n#pragma unroll 4\n#endif'))
        reject('unknown_inactive_pragma', '#if 0\n#pragma pack(1)\n#endif\n' + body,
            reason='literal pragma replay is not proved')
        reject('macro_pragma', '#define DO_UNROLL _Pragma("unroll 4")\n' +
            loop.replace('PRAGMA', 'DO_UNROLL'))
        reject('count_macro', '#define UNROLL_COUNT 4\n' +
            loop.replace('PRAGMA', '#pragma unroll UNROLL_COUNT'))
        reject('duplicate_once', '#pragma once\n#pragma once\n' + body,
            reason='literal pragma replay is not proved')
        reject('root_line_spoof', '#pragma once\n' + body,
            '#line 1 "forged.cu"\n' + parent, reason='literal pragma replay is not proved')
        reject('child_line_spoof', '#line 1 "forged.cuh"\n#pragma once\n' + body,
            reason='literal pragma replay is not proved')
        reject('gnu_linemarker', '# 1 "forged.cuh"\n#pragma once\n' + body,
            reason='literal pragma replay is not proved')
        reject('duplicate_root_entry', '#pragma once\n' + body,
            parent.replace('#include "leaf.cuh"', '#include "leaf.cuh"\n#include "leaf.cuh"'),
            reason='one direct include occurrence')
        print('4 positive contexts and 10 syntax-valid atomic rejection pairs passed')
    assert inventory(HELPERS) == fixtures, 'frozen helper fixture changed'


if __name__ == '__main__':
    main()
