#!/usr/bin/env python3
"""Bounded dynamic-kernel ABI tail completion and atomic source refusal checks.

Source validity uses the real SDK in Clang host mode, with optional native NVCC
-dc. No erroneous external device AST is consumed. No CCEC or device execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    path = Path(path)
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    fields = lambda st: (st.st_dev, st.st_ino, st.st_uid, st.st_gid, st.st_mode,
                         st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink)
    assert fields(before) == fields(after), ('input changed during hash', path)
    return [*fields(before), str(path.resolve(strict=True)), digest]


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['binary', 'cuda-path', 'resource-dir', 'clang']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--nvcc')
    parser.add_argument('--nvcc-host-cxx')
    parser.add_argument('--evidence-dir')
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled')
    assert len(CASES) == 6 and sum(row['boundary'] is None for row in CASES) == 1
    assert len({row['name'] for row in CASES}) == 6
    work = Path(args.evidence_dir or tempfile.mkdtemp(prefix='ascify-abi-tail-')).resolve()
    work.mkdir(parents=True, exist_ok=True)
    assert not list(work.iterdir()), 'Fresh evidence directory required'
    sdk = Path(args.cuda_path).resolve(strict=True)
    resource = Path(args.resource_dir).resolve(strict=True)
    binary = Path(args.binary).resolve(strict=True)
    clang = Path(shutil.which(args.clang) or args.clang).resolve(strict=True)
    nvcc = Path(shutil.which(args.nvcc) or args.nvcc).resolve(strict=True) if args.nvcc else None
    host = Path(shutil.which(args.nvcc_host_cxx) or args.nvcc_host_cxx).resolve(strict=True) if args.nvcc_host_cxx else None
    inputs = {binary, clang, Path(__file__).resolve()}
    roots = [sdk / 'include', sdk / 'nvvm/include', resource / 'include']
    for root in roots:
        assert root.is_dir()
        inputs.update(q for q in root.rglob('*') if q.is_file())
    if nvcc:
        inputs.add(nvcc)
    if host:
        inputs.add(host)
    initial_population = {q for root in roots for q in root.rglob('*') if q.is_file()}
    before = {str(q): identity(q) for q in sorted(inputs)}
    write(work / 'inputs.before.json', before)
    env = os.environ.copy()
    for key in ['ASCIFY_BINARY', 'LD_PRELOAD', 'LD_AUDIT', 'CPATH', 'CPLUS_INCLUDE_PATH',
                'C_INCLUDE_PATH', 'OBJC_INCLUDE_PATH', 'LIBRARY_PATH', 'PYTHONPATH',
                'PYTHONOPTIMIZE', 'CCC_OVERRIDE_OPTIONS', 'CLANG_CONFIG_FILE_USER_DIR',
                'CLANG_CONFIG_FILE_SYSTEM_DIR', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS',
                'CUDAHOSTCXX', 'CUDA_PATH']:
        env.pop(key, None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    rows = []
    summary = dict(passed=False, cases=6, rows=rows, optional_NVCC=bool(nvcc),
                   identity_scope='explicit SDK/resource trees and tools; not native dependency proof',
                   CCEC=0, NPU=0, sample_acceptance=0)

    def command(directory, name, argv):
        argv = list(map(str, argv))
        write(directory / (name + '.argv.json'), argv)
        started = time.monotonic()
        with (directory / (name + '.stdout.log')).open('xb') as stdout, \
                (directory / (name + '.stderr.log')).open('xb') as stderr:
            process = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr)
            timed_out = False
            try:
                code = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                timed_out, code = True, process.poll()
        result = dict(pid=process.pid, returncode=code, timed_out=timed_out,
                      process_running=code is None, elapsed=time.monotonic() - started)
        write(directory / (name + '.process.json'), result)
        assert not timed_out, ('timeout leaves process unsignalled', name, result)
        return code, (directory / (name + '.stderr.log')).read_text()

    try:
        write(work / 'cases.json', CASES)
        for item in CASES:
            directory = work / item['name']
            directory.mkdir(mode=0o700)
            source = directory / 'input.cu'
            source.write_text(item['source'])
            sdk_flags = ['-I' + str(sdk / 'include'), '-I' + str(sdk / 'include/cccl'),
                         '-I' + str(sdk / 'nvvm/include')]
            validity = [clang, '-x', 'cuda', '--cuda-path=' + str(sdk),
                        '--cuda-gpu-arch=sm_75', '-resource-dir=' + str(resource),
                        '-nocudalib', '-std=c++17', '-fgpu-rdc', *sdk_flags,
                        '--cuda-host-only', '-fsyntax-only', source]
            code, error = command(directory, 'Clang_host', validity)
            assert code == 0, (item['name'], 'source validity', error)
            if nvcc:
                native = [nvcc, '-std=c++17', '-arch=sm_75', '-dc', source,
                          '-o', directory / 'native.o']
                if host:
                    native += ['-ccbin', host]
                code, error = command(directory, 'NVCC', native)
                assert code == 0, (item['name'], 'native source validity', error)
            for mode in ['fresh'] if item['boundary'] is None else ['fresh', 'existing']:
                attempt = directory / mode
                attempt.mkdir(mode=0o700)
                output, receipt = attempt / 'source.cce', attempt / 'receipt.json'
                previous = None
                if mode == 'existing':
                    output.write_bytes(b'ASCIFY_ABI_TAIL_ATOMIC_SENTINEL\n')
                    output.chmod(0o640)
                    previous = identity(output)
                argv = [binary, source, '--target-policy=dav-c310-vec', '--simt-math=precise',
                        '--target-recipe=none', '--default-preprocessor',
                        '--frontend-compat=ascify-admitted-v1', '--cuda-path=' + str(sdk),
                        '--clang-resource-directory=' + str(resource), '--cuda-gpu-arch=sm_75',
                        '--migration-receipt=' + str(receipt), '-o', output, '--',
                        '-x', 'cuda', '-std=c++17', '-fgpu-rdc', *sdk_flags]
                code, error = command(attempt, 'convert', argv)
                row = dict(name=item['name'], mode=mode, returncode=code, passed=False)
                rows.append(row)
                write(work / ('progress-%02d.json' % len(rows)), summary)
                status = json.loads(receipt.read_text())['status']
                if item['boundary'] is None:
                    assert code == 0 and status == 'succeeded', error
                    assert 'Ascify dynamic kernel ABI tail v1: kernels=1' in error, error
                    emitted = output.read_text()
                    assert emitted.count(', unsigned int = 0u') == 1
                    assert 'unsigned n, unsigned int = 0u)' in emitted
                    assert '>>>(a,b,c,n)' in emitted.replace(' ', '').replace('\n', '')
                    assert '#include <ascify/dynamic_warp_groups_compat.hpp>' in emitted
                else:
                    expected = ('dynamic kernel ABI boundary: dynamic kernel ABI padding rejects '
                                + item['boundary'])
                    assert code == 1 and status == 'failed' and expected in error, error
                    if previous is None:
                        assert not output.exists()
                    else:
                        assert identity(output) == previous
                    allowed = {'convert.argv.json', 'convert.stdout.log', 'convert.stderr.log',
                               'convert.process.json', 'receipt.json'}
                    if previous is not None:
                        allowed.add('source.cce')
                    assert {q.name for q in attempt.iterdir()} == allowed, 'new bundle after refusal'
                row['passed'] = True
        assert len(rows) == 11 and all(row['passed'] for row in rows)
        summary['passed'] = True
    except Exception as error:
        summary['error'] = repr(error)
    finally:
        after = {str(q): identity(q) for q in sorted(inputs)}
        write(work / 'inputs.after.json', after)
        summary['inputs_unchanged'] = before == after
        summary['provider_population_unchanged'] = initial_population == {
            q for root in roots for q in root.rglob('*') if q.is_file()}
        summary['passed'] = (summary['passed'] and summary['inputs_unchanged'] and
                             summary['provider_population_unchanged'])
        write(work / 'summary.json', summary)
    if not summary['passed']:
        raise SystemExit(1)


CASES = [{'name': 'positive', 'source': '#include <cooperative_groups.h>\nnamespace cg=cooperative_groups;\n__global__ void kernel(int* a,int* b,int* c,unsigned n) {\n  auto g=cg::coalesced_threads();\n  if(g.thread_rank()==0)c[0]=int(g.size())+int(n);\n}\nvoid launch(int* a,int* b,int* c,unsigned n) {\n  kernel<<<1,32>>>(a,b,c,n);\n}\n', 'boundary': None}, {'name': 'ordinary5', 'source': '#include <cooperative_groups.h>\nnamespace cg=cooperative_groups;\n__global__ void kernel(int* a,int* b,int* c,unsigned n) {\n  auto g=cg::coalesced_threads();\n  if(g.thread_rank()==0)c[0]=int(g.size())+int(n);\n}\nvoid launch(int* a,int* b,int* c,unsigned n) {\n  kernel<<<1,32>>>(a,b,c,n);\n}\nint kernel(int*,int*,int*,unsigned,unsigned) {return 5;}\nint use_host(int* a,int* b,int* c,unsigned n) {return kernel(a,b,c,n,0u);}\n', 'boundary': 'caller same-name declarations or overloads'}, {'name': 'function_template5', 'source': '#include <cooperative_groups.h>\nnamespace cg=cooperative_groups;\n__global__ void kernel(int* a,int* b,int* c,unsigned n) {\n  auto g=cg::coalesced_threads();\n  if(g.thread_rank()==0)c[0]=int(g.size())+int(n);\n}\nvoid launch(int* a,int* b,int* c,unsigned n) {\n  kernel<<<1,32>>>(a,b,c,n);\n}\ntemplate<class T> int kernel(int*,int*,int*,unsigned,T) {return 5;}\nint use_host(int* a,int* b,int* c,unsigned n) {return kernel(a,b,c,n,0u);}\n', 'boundary': 'caller same-name declarations or overloads'}, {'name': 'nttp', 'source': '#include <cooperative_groups.h>\nnamespace cg=cooperative_groups;\n__global__ void kernel(int* a,int* b,int* c,unsigned n) {\n  auto g=cg::coalesced_threads();\n  if(g.thread_rank()==0)c[0]=int(g.size())+int(n);\n}\nvoid launch(int* a,int* b,int* c,unsigned n) {\n  kernel<<<1,32>>>(a,b,c,n);\n}\ntemplate<auto F> struct saved_kernel {};\nusing Saved=saved_kernel<kernel>;\nstatic_assert(sizeof(Saved)>0,"complete NTTP object");\n', 'boundary': 'kernel address escape, reflection, alias or unproved call'}, {'name': 'nttp_pack', 'source': '#include <cooperative_groups.h>\nnamespace cg=cooperative_groups;\n__global__ void kernel(int* a,int* b,int* c,unsigned n) {\n  auto g=cg::coalesced_threads();\n  if(g.thread_rank()==0)c[0]=int(g.size())+int(n);\n}\nvoid launch(int* a,int* b,int* c,unsigned n) {\n  kernel<<<1,32>>>(a,b,c,n);\n}\ntemplate<auto... F> struct saved_kernels {};\nusing Saved=saved_kernels<kernel,kernel>;\nstatic_assert(sizeof(Saved)>0,"complete NTTP pack object");\n', 'boundary': 'kernel address escape, reflection, alias or unproved call'}, {'name': 'file_scope_asm', 'source': '#include <cooperative_groups.h>\nnamespace cg=cooperative_groups;\n__global__ void kernel(int* a,int* b,int* c,unsigned n) {\n  auto g=cg::coalesced_threads();\n  if(g.thread_rank()==0)c[0]=int(g.size())+int(n);\n}\nvoid launch(int* a,int* b,int* c,unsigned n) {\n  kernel<<<1,32>>>(a,b,c,n);\n}\n\nasm(".text");\n', 'boundary': 'caller file-scope assembly symbol observations'}]

if __name__ == '__main__':
    main()
