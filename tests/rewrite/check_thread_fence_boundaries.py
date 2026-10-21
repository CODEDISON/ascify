#!/usr/bin/env python3
"""Syntax-first host status forwarding and device counter proofs; no NPU."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / 'tests/rewrite/fixtures/nvidia_samples/Common'
PREFIX = '#include <cuda_runtime.h>\n'
FORWARD = 'cudaError_t forward() { return cudaDeviceSynchronize(); }\n'
COUNTER = '__device__ unsigned int retirement = 0;\n'
KERNEL = '__global__ void kernel(unsigned *out) { out[0] = atomicInc(&retirement, 7U); }\n'


def cases():
    def status(name, declaration, call='forward()', accepted=False, body='', headers=None, recursive=False):
        code = PREFIX + '#include <helper_cuda.h>\n' + declaration
        code += ('void run() {\n  checkCudaErrors(cudaDeviceSynchronize());\n'
                 '  cudaError_t error;\n  ' + body + '\n  error = ' + call + ';\n'
                 '  checkCudaErrors(error);\n  checkCudaErrors(cudaDeviceSynchronize());\n}\n')
        return dict(name='status_' + name, code=code, accepted=accepted, kind='status',
                    headers=headers or {}, recursive=recursive)

    yield status('direct', FORWARD, accepted=True)
    yield status('renamed', FORWARD.replace('forward', 'relay'), 'relay()', True)
    yield status('parentheses', FORWARD.replace('cudaDeviceSynchronize()', '(cudaDeviceSynchronize())'), accepted=True)
    yield status('redecl', 'cudaError_t forward();\n' + FORWARD, accepted=True)
    yield status('symbol_copy', COUNTER + 'cudaError_t forward(unsigned value) { return cudaMemcpyToSymbol(retirement, &value, sizeof(value), 0, cudaMemcpyHostToDevice); }\n', 'forward(0U)', True)
    yield status('arguments', 'cudaError_t forward(void *a, const void *b, size_t n) { return cudaMemcpy(a,b,n,cudaMemcpyDeviceToDevice); }\n', 'forward(nullptr,nullptr,0)', True)
    header = '#ifndef PROJECT_FORWARD_CUH\n#define PROJECT_FORWARD_CUH\n' + PREFIX + FORWARD + '#endif\n'
    yield status('selected_header', '#include "forward.cuh"\n', accepted=True,
                 headers={'forward.cuh': header}, recursive=True)
    yield status('unselected_header', '#include "forward.cuh"\n',
                 headers={'forward.cuh': header})
    yield status('ignored_gnu_prefix', '__attribute__((project_future_marker)) ' + FORWARD)
    yield status('ignored_gnu_short_prefix', '__attribute((project_future_marker)) ' + FORWARD)
    yield status('ignored_cpp_prefix', '[[project_future::marker]] ' + FORWARD)
    yield status('ignored_gnu_trailing', FORWARD.replace('forward()', 'forward() __attribute__((project_future_marker))'))
    attribute_header = '#define PROJECT_ATTRIBUTE __attribute__((project_future_marker))\n'
    for name, prefix in [('external_attribute', 'PROJECT_ATTRIBUTE '),
                         ('attribute_interleaved_directive', 'PROJECT_ATTRIBUTE\n#if 1\n'),
                         ('attribute_inactive_semicolon', 'PROJECT_ATTRIBUTE\n#if 0\n;\n#endif\n')]:
        suffix = '#endif\n' if name == 'attribute_interleaved_directive' else ''
        yield status(name, '#include "attribute.cuh"\n' + prefix + FORWARD + suffix,
                     headers={'attribute.cuh': attribute_header})
    for name, declaration, call in [
        ('constant', 'cudaError_t forward() { return cudaSuccess; }\n', 'forward()'),
        ('cast', 'cudaError_t forward() { return static_cast<cudaError_t>(cudaDeviceSynchronize()); }\n', 'forward()'),
        ('conditional', 'cudaError_t forward(bool b) { return b ? cudaDeviceSynchronize() : cudaSuccess; }\n', 'forward(true)'),
        ('comma', 'cudaError_t forward() { return ((void)0,cudaDeviceSynchronize()); }\n', 'forward()'),
        ('second_statement', 'cudaError_t forward() { (void)0; return cudaDeviceSynchronize(); }\n', 'forward()'),
        ('external_only', 'cudaError_t forward();\n', 'forward()'),
        ('second_wrapper', FORWARD.replace('forward', 'inner') + 'cudaError_t forward(){return inner();}\n', 'forward()'),
        ('template', 'template<class T> cudaError_t forward(T) { return cudaDeviceSynchronize(); }\n', 'forward(0)'),
        ('default_argument', 'cudaError_t forward(int = 0) { return cudaDeviceSynchronize(); }\n', 'forward()'),
        ('variadic', 'cudaError_t forward(int, ...) { return cudaDeviceSynchronize(); }\n', 'forward(0)'),
        ('member', 'struct Object { static cudaError_t forward() { return cudaDeviceSynchronize(); } };\n', 'Object::forward()'),
        ('qualified', 'namespace project { ' + FORWARD + '}\n', 'project::forward()'),
        ('body_macro', '#define SYNC() cudaDeviceSynchronize()\n' + FORWARD.replace('cudaDeviceSynchronize()', 'SYNC()'), 'forward()'),
        ('decl_macro', '#define DECLARE ' + FORWARD.strip() + '\nDECLARE\n', 'forward()'),
        ('conditional_return', 'cudaError_t forward() {\n#if 1\n return cudaDeviceSynchronize();\n#else\n return cudaSuccess;\n#endif\n}\n', 'forward()'),
        ('digraph_return', 'cudaError_t forward() {\n%:if 1\n return cudaDeviceSynchronize();\n%:else\n return cudaSuccess;\n%:endif\n}\n', 'forward()'),
        ('pragma_operator', 'cudaError_t forward() { _Pragma("GCC diagnostic push") return cudaDeviceSynchronize(); }\n', 'forward()'),
        ('spliced_pragma', 'cudaError_t forward() { _Pra\\\ngma("GCC diagnostic push") return cudaDeviceSynchronize(); }\n', 'forward()'),
        ('explicit_attribute', '__attribute__((noinline)) ' + FORWARD, 'forward()'),
        ('weak_attribute', '__attribute__((weak)) ' + FORWARD, 'forward()'),
        ('pragma_weak', '#pragma weak forward\n' + FORWARD, 'forward()'),
        ('asm_label', 'cudaError_t forward() asm("project_forward");\n' + FORWARD, 'forward()'),
    ]:
        yield status(name, declaration, call)
    yield status('pointer_call', FORWARD, 'pointer()', body='auto pointer = &forward;')
    yield status('later_reference', FORWARD, body='cudaError_t &alias = error; (void)alias;')
    yield status('capture', FORWARD, body='auto observe = [&error] { return error; }; (void)observe;')
    yield status('address', FORWARD, body='auto pointer = &error; (void)pointer;')
    yield dict(name='status_direct_check_not_added', kind='status', accepted=False, headers={}, recursive=False,
               code=PREFIX + '#include <helper_cuda.h>\n' + FORWARD + 'void run(){checkCudaErrors(forward());}\n')

    def atomic(name, code, accepted=False, symbol=True, headers=None):
        return dict(name='counter_' + name, code=PREFIX + code, kind='counter', accepted=accepted,
                    symbol=symbol, headers=headers or {}, recursive=False)
    yield atomic('direct', COUNTER + KERNEL, True)
    yield atomic('renamed', (COUNTER + KERNEL).replace('retirement', 'tickets'), True)
    yield atomic('parentheses', COUNTER + KERNEL.replace('&retirement', '((&(retirement)))'), True)
    yield atomic('namespace', 'namespace project {\n' + COUNTER + KERNEL + '}\n', True)
    yield atomic('parameter_unchanged', '__global__ void kernel(unsigned *counter,unsigned *out) { out[0]=atomicInc(counter,7U); }\n', True, False)
    yield atomic('ignored_gnu_prefix', '__attribute__((project_future_marker)) ' + COUNTER + KERNEL)
    yield atomic('ignored_gnu_short_prefix', '__attribute((project_future_marker)) ' + COUNTER + KERNEL)
    yield atomic('ignored_cpp_prefix', '[[project_future::marker]] ' + COUNTER + KERNEL)
    yield atomic('ignored_gnu_trailing', COUNTER.replace('retirement =', 'retirement __attribute__((project_future_marker)) =') + KERNEL)
    for name, prefix in [('external_attribute', 'PROJECT_ATTRIBUTE '),
                         ('attribute_interleaved_directive', 'PROJECT_ATTRIBUTE\n#if 1\n'),
                         ('attribute_inactive_semicolon', 'PROJECT_ATTRIBUTE\n#if 0\n;\n#endif\n')]:
        suffix = '#endif\n' if name == 'attribute_interleaved_directive' else ''
        yield atomic(name, '#include "attribute.cuh"\n' + prefix + COUNTER + suffix + KERNEL,
                     headers={'attribute.cuh': attribute_header})
    for name, code in [
        ('nonzero', COUNTER.replace('= 0', '= 1') + KERNEL),
        ('implicit_zero', COUNTER.replace(' = 0', '') + KERNEL),
        ('list_zero', COUNTER.replace('= 0', '{0}') + KERNEL),
        ('macro_zero', '#define ZERO 0\n' + COUNTER.replace('= 0', '= ZERO') + KERNEL),
        ('macro_qualifier', '#define DEVICE __device__\n' + COUNTER.replace('__device__', 'DEVICE') + KERNEL),
        ('type_alias', 'using UInt = unsigned int;\n' + COUNTER.replace('unsigned int', 'UInt') + KERNEL),
        ('signed_cast', COUNTER.replace('unsigned int', 'int') + KERNEL.replace('&retirement', 'reinterpret_cast<unsigned*>(&retirement)')),
        ('wide_cast', COUNTER.replace('unsigned int', 'unsigned long long') + KERNEL.replace('&retirement', 'reinterpret_cast<unsigned*>(&retirement)')),
        ('float_cast', COUNTER.replace('unsigned int', 'float') + KERNEL.replace('&retirement', 'reinterpret_cast<unsigned*>(&retirement)')),
        ('array', '__device__ unsigned retirement[1] = {0};\n' + KERNEL.replace('&retirement', '&retirement[0]')),
        ('constant', COUNTER.replace('__device__', '__constant__') + KERNEL),
        ('volatile_cast', COUNTER.replace('unsigned int', 'volatile unsigned int') + KERNEL.replace('&retirement', 'const_cast<unsigned*>(&retirement)')),
        ('extern_only', 'extern __device__ unsigned int retirement;\n' + KERNEL),
        ('extern_redecl', 'extern __device__ unsigned int retirement;\n' + COUNTER + KERNEL),
        ('anonymous_namespace', 'namespace {\n' + COUNTER + KERNEL + '}\n'),
        ('template_symbol', 'template<class T> __device__ T retirement = 0;\n' + KERNEL.replace('&retirement', '&retirement<unsigned>')),
        ('inline_symbol', 'inline ' + COUNTER + KERNEL),
        ('static_member', 'struct Object { static __device__ unsigned retirement; };\n__device__ unsigned Object::retirement=0;\n' + KERNEL.replace('&retirement', '&Object::retirement')),
        ('local_alias', COUNTER + KERNEL.replace('out[0] = atomicInc(&retirement', 'unsigned *alias=&retirement; out[0] = atomicInc(alias')),
        ('address_arithmetic', COUNTER + KERNEL.replace('&retirement', '&retirement+0')),
        ('explicit_cast', COUNTER + KERNEL.replace('&retirement', 'static_cast<unsigned*>(&retirement)')),
        ('address_macro', '#define ADDRESS &retirement\n' + COUNTER + KERNEL.replace('&retirement', 'ADDRESS')),
        ('qualified_call', COUNTER + KERNEL.replace('atomicInc(', '::atomicInc(')),
        ('caller_callee_redecl', 'static __device__ inline unsigned atomicInc(unsigned*,unsigned);\n' + COUNTER + KERNEL),
        ('other_atomic', COUNTER + KERNEL.replace('atomicInc(', 'atomicAdd(')),
        ('device_helper', COUNTER + KERNEL.replace('__global__', '__device__')),
        ('local_shared', '__global__ void kernel(unsigned*out) { __shared__ unsigned retirement; out[0]=atomicInc(&retirement,7U); }\n'),
    ]:
        yield atomic(name, code)
    # Generated-name collision must reject publication, not drop the resolver
    # while keeping the newly admitted atomic address.
    offset = (PREFIX + COUNTER).index('retirement')
    yield atomic('resolver_collision', COUNTER + 'int ascify_generated_symbol_' + str(offset) + '_address;\n' + KERNEL)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('binary', 'cuda-path', 'resource-dir', 'clang', 'evidence-dir'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    assert not sys.flags.optimize
    if not args.clang.is_file():
        selected = shutil.which(str(args.clang))
        if selected is None: parser.error('matching --clang executable required')
        args.clang = Path(selected)
    args.binary, args.cuda_path, args.resource_dir, args.clang, work = (
        value.resolve() for value in (args.binary, args.cuda_path, args.resource_dir, args.clang, args.evidence_dir))
    work.mkdir(parents=True, exist_ok=False)
    frozen = {str(path): sha(path) for path in [args.binary, args.clang, Path(__file__), *HELPERS.iterdir()] if path.is_file()}
    summary = dict(schema='ascify.thread-fence-boundaries.v1', passed=False, cases=[],
                   frozen_inputs=frozen, device_executed=False, original_sample_successes=0)
    def save():
        (work / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    def run(directory, label, argv):
        argv = list(map(str, argv))
        (directory / (label + '.argv.json')).write_text(json.dumps(argv, indent=2) + '\n')
        result = subprocess.run(argv, text=True, capture_output=True, timeout=120)
        (directory / (label + '.stdout.log')).write_text(result.stdout)
        (directory / (label + '.stderr.log')).write_text(result.stderr)
        (directory / (label + '.returncode')).write_text(str(result.returncode) + '\n')
        return result
    save()
    try:
        version = run(work, 'clang_version', [args.clang, '--version'])
        assert version.returncode == 0 and re.search(r'\bclang version ' + re.escape(args.resource_dir.name.split('.')[0]) + r'(?:\.|\s)', version.stdout)
        for case in cases():
            directory = work / case['name']; directory.mkdir()
            source = directory / 'input.cu'; output = directory / 'output.cpp'; receipt = directory / 'receipt.json'
            source.write_text(case['code'])
            for name, text in case['headers'].items(): (directory / name).write_text(text)
            identities = {str(path): sha(path) for path in [source, *[directory / name for name in case['headers']]]}
            row = dict(name=case['name'], kind=case['kind'], expected_admitted=case['accepted'], passed=False, source_identity=identities)
            summary['cases'].append(row); save()
            syntax = run(directory, 'syntax', [args.clang, source, '-x', 'cuda', '-std=c++17', '-fsyntax-only', '--cuda-host-only', '-nocudalib', '--cuda-path=' + str(args.cuda_path), '-resource-dir=' + str(args.resource_dir), '-Wno-unknown-cuda-version', '-I' + str(HELPERS)])
            row['syntax_returncode'] = syntax.returncode; save()
            assert syntax.returncode == 0, (case['name'], 'invalid CUDA fixture', syntax.stderr)
            sentinel = 'UNCHANGED_OUTPUT_' + case['name'] + '\n'
            if not case['accepted']: output.write_text(sentinel)
            converted = run(directory, 'convert', [args.binary, source, '--target-policy=dav-c310-vec', '--simt-math=fast', '--target-recipe=none', '--default-preprocessor', '--cuda-path=' + str(args.cuda_path), '--clang-resource-directory=' + str(args.resource_dir), '--migration-receipt=' + str(receipt), '-o', output, *(['--local-headers-recursive'] if case['recursive'] else []), '--', '-x', 'cuda', '-std=c++17', '-I' + str(HELPERS), *(['-iquote' + str(directory)] if case['headers'] else [])])
            row['conversion_returncode'] = converted.returncode; save()
            if case['headers']:
                assert not re.search(r'file not found|no such file or directory', converted.stderr, re.IGNORECASE), (case['name'], 'fixture header lookup failed', converted.stderr)
            if converted.returncode:
                assert not case['accepted'] and output.read_text() == sentinel, (case['name'], converted.stderr)
                assert not Path(str(output) + '.headers').exists(), case['name']
                row['rejection'] = 'failed_without_publication'
            else:
                assert json.loads(receipt.read_text())['status'] == 'succeeded'
                text = output.read_text()
                if case['kind'] == 'status':
                    if case['accepted']:
                        assert '#include <helper_cuda.h>' not in text
                        assert text.count('ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(') == case['code'].count('checkCudaErrors('), (case['name'], text)
                        assert 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS(error)' in text
                    else:
                        assert '#include <helper_cuda.h>' in text and 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' not in text, (case['name'], text)
                        assert text.count('checkCudaErrors(') == case['code'].count('checkCudaErrors(')
                        diagnostic = ("unproven hash pragma 'weak' keeps all helper edits"
                                      if case['name'] == 'status_pragma_weak'
                                      else 'status domain not proven')
                        assert diagnostic in converted.stderr, (case['name'], converted.stderr)
                else:
                    assert text.count('ascify::atomic_inc_global(') == int(case['accepted']), (case['name'], text)
                    if case['accepted'] and case['symbol']:
                        assert text.count('ASCIFY_REGISTER_DEVICE_SYMBOL(') == 1, (case['name'], text)
                    if not case['accepted']:
                        operation = 'atomicAdd(' if case['name'] == 'counter_other_atomic' else 'atomicInc('
                        assert text.count(operation) == case['code'].count(operation), (case['name'], text)
                row['output_sha256'] = sha(output)
            assert {name: sha(Path(name)) for name in identities} == identities
            row['passed'] = True; save(); print(case['name'] + ': passed', flush=True)
        assert {name: sha(Path(name)) for name in frozen} == frozen
        summary.update(passed=True, total=len(summary['cases']))
    except Exception as error:
        summary['error'] = str(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
