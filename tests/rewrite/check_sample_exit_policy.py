#!/usr/bin/env python3
"""Exercise frozen sample policy preservation with the actual CUDA frontend."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / 'tests/rewrite/fixtures/nvidia_samples/Common'
V134 = ROOT / 'tests/rewrite/fixtures/nvidia_samples_v13_4/Common/helper_cuda.h'
BODY = '''
void release(void* ptr) { checkCudaErrors(cudaFree(ptr)); }
int waived_status() { return EXIT_WAIVED; }
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--cuda-path', required=True)
    parser.add_argument('--resource-dir', required=True)
    args = parser.parse_args()
    if __debug__ is False:
        raise RuntimeError('Assertions must remain enabled')
    fixtures = [*COMMON.iterdir(), V134]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in fixtures}
    with tempfile.TemporaryDirectory(prefix='ascify-exit-policy-') as tmp:
        work = Path(tmp)
        cases = []
        for profile, header in [('v130', COMMON / 'helper_cuda.h'), ('v134', V134)]:
            helpers = work / profile / 'Common'
            shutil.copytree(COMMON, helpers)
            shutil.copyfile(header, helpers / 'helper_cuda.h')
            for order, includes in [
                ('direct', '#include <helper_cuda.h>\n'),
                ('functions_first', '#include <helper_functions.h>\n#include <helper_cuda.h>\n'),
                ('cuda_first', '#include <helper_cuda.h>\n#include <helper_functions.h>\n'),
            ]:
                cases.append((profile + '_' + order, helpers, includes + BODY, [], True))
        base = '#include <helper_cuda.h>\n'
        helpers = work / 'v130/Common'
        cases += [
            ('command_line_provider', helpers, base + BODY, ['-DEXIT_WAIVED=2'], False),
            ('user_provider', helpers, '#define EXIT_WAIVED 2\n' + base + BODY, [], False),
            ('changed_lifetime', helpers, base + '#undef EXIT_WAIVED\n#define EXIT_WAIVED 3\n' + BODY, [], False),
            ('forwarded_alias', helpers, base + '#define WAIVED EXIT_WAIVED\n' + BODY.replace('return EXIT_WAIVED', 'return WAIVED'), [], False),
        ]
        for label, helpers, text, defines, accepted in cases:
            source, output = work / (label + '.cu'), work / (label + '.cpp')
            source.write_text(text)
            result = subprocess.run([
                args.binary, str(source), '--target-policy=dav-c310-vec', '--simt-math=fast',
                '--default-preprocessor', '--cuda-path=' + args.cuda_path,
                '--clang-resource-directory=' + args.resource_dir, '-o', str(output), '--',
                '-x', 'cuda', '-std=c++17', '-I' + str(helpers), *defines,
            ], capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, (label, result.stderr[-5000:])
            generated = output.read_text()
            if accepted:
                assert '#include <helper_cuda.h>' not in generated, (label, result.stderr[-5000:])
                assert '#ifndef EXIT_WAIVED\n#define EXIT_WAIVED 2\n#endif' in generated, label
                assert 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(ascify::cudaFree(ptr))' in generated, label
                assert 'return EXIT_WAIVED;' in generated, label
                # The target preprocessor, rather than text matching alone,
                # must observe the original macro value and #if contract.
                probe = work / (label + '_probe.cpp')
                probe.write_text('#include "' + output.name + '"\n'
                                 '#if !defined(EXIT_WAIVED) || EXIT_WAIVED != 2\n'
                                 '#error "sample waived policy changed"\n#endif\n')
                compiled = subprocess.run([
                    os.environ.get('CXX', 'c++'), '-x', 'c++', '-std=c++17', '-fsyntax-only',
                    '-D__aicore__=', '-I' + str(ROOT / 'tests/rewrite/stubs'),
                    '-I' + str(ROOT / 'include'), '-I' + str(helpers), str(probe),
                ], capture_output=True, text=True, timeout=60)
                assert compiled.returncode == 0, (label, compiled.stderr[-5000:])
            else:
                assert '#include <helper_cuda.h>' in generated, (label, result.stderr[-5000:])
                assert 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' not in generated, label
                assert 'include kept' in result.stderr, (label, result.stderr[-5000:])
            print(label + (': policy preserved' if accepted else ': complete helper transaction retained'))
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in fixtures}
    print('Frozen sample EXIT_WAIVED: six preservation and four refusal cases passed')


if __name__ == '__main__':
    main()
