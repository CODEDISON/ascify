#!/usr/bin/env python3
"""Prove literal frozen-helper MAX expansion against the real preprocessor."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'cuda', 'resource', 'evidence'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--clang', type=Path)
    parser.add_argument('--samples', type=Path,
        default=Path(__file__).resolve().parents[2] / 'tests/rewrite/fixtures/nvidia_samples')
    args = parser.parse_args()
    if args.clang is None:
        args.clang = args.resource.resolve().parents[2] / 'bin/clang++'
    args.evidence.mkdir(parents=True, exist_ok=False)
    cases = [
        ('literal', 'MAX(1, 7)', '', True),
        ('locals', 'MAX(x, y)', '', True),
        ('nested_min', 'MAX(1, MIN(x, 10))', '', True),
        ('arithmetic', 'MAX(x + 2, y * 3)', '', True),
        ('assignment_precedence', 'MAX(x = 2, y)', '', True),
        ('conditional_precedence', 'MAX(x ? y : 3, 7)', '', True),
        ('repeated_evaluation', 'MAX(x++, y)', '', True),
        ('expanded_argument', 'MAX(SCALE(x), y)', '#define SCALE(v) ((v) * 3)\n', True),
        ('source_line', 'MAX(__LINE__, y)', '', True),
        ('counter_retained', 'MAX(__COUNTER__, y)', '', False),
        ('nested_counter_retained', 'MAX(COUNT, y)', '#define COUNT __COUNTER__\n', False),
        ('nested_max_retained', 'MAX(1, MAX(x, y))', '', False),
        ('alias_retained', 'PICK(x, y)', '#define PICK MAX\n', False),
        ('forward_retained', 'PICK(x, y)', '#define PICK(a,b) MAX(a,b)\n', False),
        ('self_macro_retained', 'MAX(USER_VALUE, y)', 'const int USER_VALUE=3;\n#define USER_VALUE USER_VALUE\n', False),
        ('redefined_body_retained', 'MAX(x, y)', '#undef MAX\n#define MAX(a,b) ((a)+(b))\n', False),
        ('complex_type_retained', 'MAX(sizeof(int), y)', '', False),
        ('pragma_retained', 'MAX(_Pragma("message \\\"max proof\\\"") x, y)', '', False),
    ]
    summary = {'passed': False, 'cases': [], 'scope': 'real CUDA frontend and exact expanded expression tokens; no device claim'}
    summary['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['binary_sha256'] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    for name, expression, prefix, admitted in cases:
        work = args.evidence / name
        work.mkdir()
        source, output = work / 'input.cu', work / 'output.cpp'
        source.write_text('#include <helper_cuda.h>\n#include <helper_functions.h>\n' + prefix +
            'int macro_value(int x, int y) { return ' + expression + '; }\n' +
            'int later_counter = __COUNTER__;\n' +
            'void checked() { checkCudaErrors(cudaFree(nullptr)); }\n')
        original = source.read_bytes()
        common = ['-x', 'cuda', '-std=c++17', '-I' + str(args.samples / 'Common'),
                  '--cuda-path=' + str(args.cuda), '-resource-dir=' + str(args.resource),
                  '--cuda-host-only', '-nocudalib', '-Wno-unknown-cuda-version']
        def run(label, argv):
            (work / (label + '.argv.json')).write_text(json.dumps(list(map(str, argv)), indent=2) + '\n')
            result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=90)
            (work / (label + '.stderr.log')).write_text(result.stderr)
            return result
        syntax = run('syntax', [args.clang, *common, '-fsyntax-only', source])
        assert syntax.returncode == 0, name + ': invalid source: ' + syntax.stderr
        expanded = run('preprocess', [args.clang, *common, '-E', '-P', source])
        assert expanded.returncode == 0, expanded.stderr
        conversion = run('convert', [args.binary, source, '--default-preprocessor',
            '--cuda-path=' + str(args.cuda), '--clang-resource-directory=' + str(args.resource),
            '-I' + str(args.samples / 'Common'), '-o', output, '--', '-x', 'cuda', '-std=c++17'])
        (work / 'convert.stdout.log').write_text(conversion.stdout)
        assert conversion.returncode == 0, name + ': ' + conversion.stderr
        generated = output.read_text()
        def body(text):
            matches = re.findall(r'int macro_value\(int x, int y\)\s*\{([^{}]*)\}', text)
            assert len(matches) == 1, name + ': expression region missing'
            return matches[0]
        source_body, target_body = body(expanded.stdout), body(generated)
        if admitted:
            assert re.sub(r'\s+', '', source_body) == re.sub(r'\s+', '', target_body), (name, source_body, target_body)
            assert '#include <helper_cuda.h>' not in generated, name
            assert 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' in generated, name
        else:
            assert '#include <helper_cuda.h>' in generated, name + ': helper incorrectly removed'
            assert expression in target_body, name + ': invocation not retained'
        assert source.read_bytes() == original
        row = {'case': name, 'admitted': admitted, 'passed': True,
               'source_sha256': hashlib.sha256(original).hexdigest(),
               'original_expanded_expression': source_body, 'generated_expression': target_body,
               'preprocess_stdout_sha256': hashlib.sha256(expanded.stdout.encode()).hexdigest()}
        summary['cases'].append(row)
        (args.evidence / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(name + ': exact expression preserved' if admitted else name + ': unsafe scope retained', flush=True)
    assert hashlib.sha256(args.binary.read_bytes()).hexdigest() == summary['binary_sha256']
    summary.update(passed=True, total=len(cases))
    (args.evidence / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('helper MAX:', len(cases), 'passed')


if __name__ == '__main__':
    main()
