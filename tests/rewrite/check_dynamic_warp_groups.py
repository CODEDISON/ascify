#!/usr/bin/env python3
"""Actual CUDA source validity and current-CI dynamic group boundary regression.

One combined positive and nine precise refusal cases. No device execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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
    assert len(CASES) == 10 and sum(row['boundary'] is None for row in CASES) == 1
    assert len({row['name'] for row in CASES}) == 10
    work = Path(args.evidence_dir or tempfile.mkdtemp(prefix='ascify-dynamic-')).resolve()
    work.mkdir(parents=True, exist_ok=True)
    assert not list(work.iterdir()), 'Fresh evidence directory required'
    sdk = Path(args.cuda_path).resolve(strict=True)
    binary = Path(args.binary).resolve(strict=True)
    resource = Path(args.resource_dir).resolve(strict=True)
    args.clang = str(Path(shutil.which(args.clang) or args.clang).resolve(strict=True))
    if args.nvcc:
        args.nvcc = str(Path(shutil.which(args.nvcc) or args.nvcc).resolve(strict=True))
    if args.nvcc_host_cxx:
        args.nvcc_host_cxx = str(Path(shutil.which(args.nvcc_host_cxx) or args.nvcc_host_cxx).resolve(strict=True))
    inputs = {binary, Path(__file__).resolve(), Path(args.clang)}
    for root in [sdk / 'include', sdk / 'nvvm/include', resource / 'include']:
        assert root.is_dir()
        inputs.update(q for q in root.rglob('*') if q.is_file())
    if args.nvcc:
        inputs.add(Path(args.nvcc).resolve(strict=True))
    if args.nvcc_host_cxx:
        inputs.add(Path(args.nvcc_host_cxx).resolve(strict=True))
    tree_roots = [sdk / 'include', sdk / 'nvvm/include', resource / 'include']
    initial_tree_files = {q for root in tree_roots for q in root.rglob('*') if q.is_file()}
    before = {str(q): identity(q) for q in sorted(inputs)}
    write(work / 'inputs.before.json', before)
    env = os.environ.copy()
    for key in ['LD_PRELOAD', 'LD_AUDIT', 'CPATH', 'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH',
                'OBJC_INCLUDE_PATH', 'LIBRARY_PATH', 'PYTHONPATH', 'PYTHONOPTIMIZE',
                'CCC_OVERRIDE_OPTIONS', 'CLANG_CONFIG_FILE_USER_DIR', 'CLANG_CONFIG_FILE_SYSTEM_DIR',
                'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'CUDAHOSTCXX', 'CUDA_PATH']:
        env.pop(key, None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    rows = []
    summary = dict(schema='ascify.dynamic-group-source-regression.v1', passed=False,
                   rows=rows, optional_NVCC=bool(args.nvcc), CCEC=0, NPU=0,
                   sample_acceptance=0, identity_scope='explicit SDK/resource trees and tools; not native build dependency proof')

    def command(directory, name, argv):
        argv = list(map(str, argv))
        write(directory / (name + '.argv.json'), argv)
        start = time.monotonic()
        child = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = child.communicate(timeout=180)
        except subprocess.TimeoutExpired as error:
            (directory / (name + '.stdout.log')).write_bytes(error.output or b'')
            (directory / (name + '.stderr.log')).write_bytes(error.stderr or b'')
            write(directory / (name + '.process.json'), dict(pid=child.pid, returncode=child.poll(),
                  timed_out=True, streams_partial=True, elapsed=time.monotonic() - start))
            raise
        (directory / (name + '.stdout.log')).write_bytes(stdout)
        (directory / (name + '.stderr.log')).write_bytes(stderr)
        write(directory / (name + '.process.json'), dict(pid=child.pid, returncode=child.returncode,
              elapsed=time.monotonic() - start))
        return child.returncode, stderr

    sdk_flags = ['-I' + str(sdk / 'include'), '-I' + str(sdk / 'include/cccl'),
                 '-I' + str(sdk / 'nvvm/include')]
    try:
        write(work / 'cases.json', CASES)
        for item in CASES:
            directory = work / item['name']
            directory.mkdir()
            for name, contents in item['files'].items():
                assert Path(name).name == name
                (directory / name).write_text(contents)
            source = directory / 'original.cu'
            source_before = {name: identity(directory / name) for name in item['files']}
            if args.nvcc:
                argv = [args.nvcc, '-std=c++17', '-arch=sm_75', '-dc']
                if args.nvcc_host_cxx:
                    argv += ['-ccbin', args.nvcc_host_cxx]
                rc, err = command(directory, 'NVCC', argv + [source, '-o', directory / 'original.o'])
                assert rc == 0, (item['name'], 'real NVCC source validity', err.decode())
            for view in ['host', 'device']:
                rc, err = command(directory, 'Clang_' + view, [args.clang, '-x', 'cuda',
                    '--cuda-' + view + '-only', '--cuda-path=' + str(sdk), '--cuda-gpu-arch=sm_75',
                    '-resource-dir=' + str(resource), '-nocudalib', '-std=c++17', '-fgpu-rdc',
                    *sdk_flags, '-Wno-unknown-cuda-version', '-fsyntax-only', source])
                assert rc == 0, (item['name'], view, 'real CUDA source validity', err.decode())
            for mode in (['fresh'] if item['boundary'] is None else ['fresh', 'existing']):
                output_directory = directory / mode
                output_directory.mkdir()
                output = output_directory / 'source.cce'
                receipt = output_directory / 'receipt.json'
                prior = None
                if mode == 'existing':
                    output.write_bytes(b'ASCIFY_DYNAMIC_ATOMIC_SENTINEL\n')
                    output.chmod(0o640)
                    prior = identity(output)
                argv = [binary, source, '--target-policy=dav-c310-vec', '--simt-math=precise',
                    '--target-recipe=none', '--default-preprocessor', '--print-stats',
                    '--frontend-compat=ascify-admitted-v1', '--cuda-path=' + str(sdk),
                    '--clang-resource-directory=' + str(resource), '--cuda-gpu-arch=sm_75',
                    '--migration-receipt=' + str(receipt), '-o', output, '--', '-x', 'cuda',
                    '-std=c++17', '-fgpu-rdc', *sdk_flags]
                rc, err = command(output_directory, 'CLI', argv)
                assert receipt.is_file(), (item['name'], 'missing migration receipt')
                status = json.loads(receipt.read_text())['status']
                if item['boundary'] is None:
                    assert rc == 0 and status == 'succeeded' and output.is_file(), err.decode()
                    assert b'values=2, factories=2, reductions=1, shuffles=0, grid_queries=2' in err
                    text = output.read_text()
                    key = '#include <ascify/dynamic_warp_groups_compat.hpp>'
                    assert re.search(r'namespace\s+cg\s*=\s*(?:::)?ascify_cg\s*;', text)
                    assert text.count(key) == 1
                    assert text.index(key) > text.index('#include <ascify/cooperative_groups_compat.hpp>')
                    assert text.index(key) > text.index('#include <ascify/ascify_cuda_compat.hpp>')
                else:
                    assert rc == 1 and status == 'failed' and item['boundary'].encode() in err, err.decode()
                    assert identity(output) == prior if mode == 'existing' else not output.exists()
                    allowed = {'CLI.argv.json', 'CLI.stdout.log', 'CLI.stderr.log', 'CLI.process.json', 'receipt.json'}
                    if mode == 'existing':
                        allowed.add('source.cce')
                    assert set(q.name for q in output_directory.iterdir()) == allowed, 'refusal published a new bundle'
                assert {name: identity(directory / name) for name in item['files']} == source_before
                assert identity(binary) == before[str(binary)]
                rows.append(dict(name=item['name'], mode=mode, boundary=item['boundary'], returncode=rc,
                                 receipt_status=status, passed=True))
                write(work / ('row_%02d.json' % len(rows)), rows[-1])
        assert len(rows) == 19
        summary['passed'] = True
    except BaseException as error:
        summary['error'] = repr(error)
        raise
    finally:
        after = {str(q): identity(q) for q in sorted(inputs)}
        write(work / 'inputs.after.json', after)
        final_tree_files = {q for root in tree_roots for q in root.rglob('*') if q.is_file()}
        summary['tree_population_unchanged'] = initial_tree_files == final_tree_files
        summary['all_bound_inputs_unchanged'] = before == after
        summary['passed'] = summary['passed'] and before == after and initial_tree_files == final_tree_files
        write(work / 'summary.json', summary)
        print('Dynamic source regression evidence:', work)
    return 0 if summary['passed'] else 1


# Exact source bytes from the already source-valid current-CI protocol follow.
CASES = [{'name': 'combined_static_binary_grid',
  'boundary': None,
  'files': {'original.cu': '#include <cuda_runtime.h>\n'
                           '#include <cooperative_groups.h>\n'
                           '#include <cooperative_groups/reduce.h>\n'
                           'namespace cg=cooperative_groups;\n'
                           '__global__ void k(int* out){cg::thread_block '
                           'cta=cg::this_thread_block(); cg::grid_group grid=cg::this_grid(); '
                           'cg::thread_block_tile<32,void> tile=cg::tiled_partition<32>(cta); int '
                           'v=out[0]; auto sub=cg::binary_partition(tile,v&1); '
                           'out[0]=cg::reduce(sub,v,cg::plus<int>())+int(grid.thread_rank()+grid.size()); '
                           'cg::sync(tile);}\n'}},
 {'name': 'written_static_method',
  'boundary': 'dynamic warp group boundary: static parent/CTA type is observed outside its exact '
              'literal local declaration',
  'files': {'original.cu': '#include <cuda_runtime.h>\n'
                           '#include <cooperative_groups.h>\n'
                           '#include <cooperative_groups/reduce.h>\n'
                           'namespace cg=cooperative_groups;\n'
                           '__global__ void k(int* out){cg::thread_block '
                           'cta=cg::this_thread_block(); cg::grid_group grid=cg::this_grid(); '
                           'cg::thread_block_tile<32,void> tile=cg::tiled_partition<32>(cta); '
                           'unsigned r=cg::thread_block::thread_rank(); int v=out[0]+int(r); auto '
                           'sub=cg::binary_partition(tile,v&1); '
                           'out[0]=cg::reduce(sub,v,cg::plus<int>())+int(grid.thread_rank()+grid.size()); '
                           'cg::sync(tile);}\n'}},
 {'name': 'written_explicit_sdk_conversion',
  'boundary': 'dynamic warp group boundary: partition requires an earlier same-function factory '
              'parent and effect-free builtin predicate/int label; static parents require SDK '
              'tile32 binary',
  'files': {'original.cu': '#include <cuda_runtime.h>\n'
                           '#include <cooperative_groups.h>\n'
                           '#include <cooperative_groups/reduce.h>\n'
                           'namespace cg=cooperative_groups;\n'
                           '__global__ void k(int* out){auto cta=cg::this_thread_block(); auto '
                           'grid=cg::this_grid(); cg::thread_block_tile<32> '
                           'tile=cg::tiled_partition<32>(cta).operator '
                           'cg::thread_block_tile<32>(); int v=out[0]; auto '
                           'sub=cg::binary_partition(tile,v&1); '
                           'out[0]=cg::reduce(sub,v,cg::plus<int>())+int(grid.thread_rank()+grid.size()); '
                           'cg::sync(tile);}\n'}},
 {'name': 'written_static_parent_alias',
  'boundary': 'dynamic warp group boundary: partition requires an earlier same-function factory '
              'parent and effect-free builtin predicate/int label; static parents require SDK '
              'tile32 binary',
  'files': {'original.cu': '#include <cuda_runtime.h>\n'
                           '#include <cooperative_groups.h>\n'
                           '#include <cooperative_groups/reduce.h>\n'
                           'namespace cg=cooperative_groups;\n'
                           '__global__ void k(int* out){cg::thread_block '
                           'cta=cg::this_thread_block(); cg::grid_group grid=cg::this_grid(); '
                           'cg::thread_block_tile<32,void> tile=cg::tiled_partition<32>(cta); '
                           'auto& alias=tile; int v=out[0]; auto '
                           'sub=cg::binary_partition(alias,v&1); '
                           'out[0]=cg::reduce(sub,v,cg::plus<int>())+int(grid.thread_rank()+grid.size()); '
                           'cg::sync(tile);}\n'}},
 {'name': 'foreign_layout_only',
  'boundary': 'dynamic warp group boundary: dynamic type alias, reflection, signature or '
              'representation is outside a local value declaration',
  'files': {'original.cu': '#include <cuda_runtime.h>\n'
                           '#include <cooperative_groups.h>\n'
                           '#include <cooperative_groups/reduce.h>\n'
                           'namespace cg=cooperative_groups;\n'
                           '#include "foreign_layout.cuh"\n'
                           '__global__ void k(unsigned* out){out[0]=foreign_layout;}\n',
            'foreign_layout.cuh': 'constexpr unsigned '
                                  'foreign_layout=sizeof(cg::coalesced_group);\n'}},
 {'name': 'standalone_greater_layout',
  'boundary': 'dynamic warp group boundary: dynamic type alias, reflection, signature or '
              'representation is outside a local value declaration',
  'files': {'original.cu': '#include <cuda_runtime.h>\n'
                           '#include <cooperative_groups.h>\n'
                           '#include <cooperative_groups/reduce.h>\n'
                           'namespace cg=cooperative_groups;\n'
                           '__global__ void k(int*out){out[0]=sizeof(cg::greater<int>);}\n'}},
 {'name': 'greater_macro_name_erasure',
  'boundary': "dynamic warp group boundary: new projection name 'greater' was captured by a macro",
  'files': {'original.cu': '#define greater uniform_reduce_group_domain\n'
                           '#include <cuda_runtime.h>\n'
                           '#include <cooperative_groups.h>\n'
                           '#include <cooperative_groups/reduce.h>\n'
                           'namespace cg=cooperative_groups;\n'
                           'static_assert(!std::is_base_of<std::false_type,cg::greater<int>>::value,"native '
                           'greater is not the target scratch-group trait");\n'
                           '__global__ void '
                           'k(int*out){out[0]=std::is_base_of<std::false_type,cg::greater<int>>::value;}\n'}},
 {'name': 'macro_parenthesized_factory',
  'boundary': 'dynamic warp group boundary: dynamic group must be initialized directly by a '
              'literal owned factory',
  'files': {'original.cu': '#include <cuda_runtime.h>\n'
                           '#include <cooperative_groups.h>\n'
                           '#include <cooperative_groups/reduce.h>\n'
                           'namespace cg=cooperative_groups;\n'
                           '#define MAKE cg::coalesced_threads\n'
                           '__global__ void k(int* out){auto active=(MAKE)(); '
                           'out[0]=active.size();}\n'}},
 {'name': 'caller_push_cacheline',
  'boundary': 'dynamic warp group boundary: dynamic caller macro-stack history is unproved',
  'files': {'original.cu': '#include <cuda_runtime.h>\n'
                           '#include <cooperative_groups.h>\n'
                           '#include <cooperative_groups/reduce.h>\n'
                           '#pragma push_macro("__GCC_DESTRUCTIVE_SIZE")\n'
                           'namespace cg=cooperative_groups;\n'
                           '__global__ void k(int* out){auto active=cg::coalesced_threads(); '
                           'out[0]=active.size();}\n'}},
 {'name': 'caller_pop_cacheline',
  'boundary': 'dynamic warp group boundary: dynamic caller macro-stack history is unproved',
  'files': {'original.cu': '#include <cuda_runtime.h>\n'
                           '#include <cooperative_groups.h>\n'
                           '#include <cooperative_groups/reduce.h>\n'
                           '#pragma pop_macro("__GCC_DESTRUCTIVE_SIZE")\n'
                           'namespace cg=cooperative_groups;\n'
                           '__global__ void k(int* out){auto active=cg::coalesced_threads(); '
                           'out[0]=active.size();}\n'}}]

if __name__ == '__main__':
    raise SystemExit(main())
