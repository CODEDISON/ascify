#!/usr/bin/env python3
"""Native translator/CPU checks for explicit host C++; never opens an NPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


HOST = '''#if defined(__CUDACC__) || defined(__CUDA__) || defined(__CUDA_ARCH__)
#error host input received CUDA language semantics
#endif
typedef struct { unsigned char x, y, z, w; } uchar4;
static_assert(sizeof(uchar4) == 4, "user type changed");
extern "C" unsigned host_pixel(uchar4 p) { return p.x + p.y + p.z + p.w; }
'''
CUDA = '''static_assert(__CUDA__ == 1, "default CUDA mode changed");
int default_cuda_input = 1;
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(argv):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, (argv, result.stdout, result.stderr)
    return result


def main():
    binary = str(Path(os.environ['ASCIFY_BINARY']).resolve(strict=True))
    cuda = Path(os.environ['ASCIFY_CUDA_PATH']).resolve(strict=True)
    resource = Path(os.environ['ASCIFY_CLANG_RESOURCE_DIRECTORY']).resolve(strict=True)
    cxx = os.environ.get('CXX', 'c++')
    base = [binary, '--cuda-path=' + str(cuda),
            '--clang-resource-directory=' + str(resource)]
    with tempfile.TemporaryDirectory(prefix='ascify-host-cxx-') as tmp:
        work = Path(tmp)
        count = 0

        def run(name, source_text=HOST, *, tool=(), flags=('-x', 'c++'),
                good=True, diagnostic='', suffix='.cpp', headers=None,
                compile_host=False, database=None, exact=False,
                runtime_overlay=False):
            nonlocal count
            directory = work / name
            directory.mkdir()
            source = directory / ('input' + suffix)
            source.write_text(source_text)
            output = directory / 'translated.cpp'
            paths = [source]
            for filename, text in (headers or {}).items():
                header = directory / filename
                header.parent.mkdir(parents=True, exist_ok=True)
                header.write_text(text)
                paths.append(header)
            overlay = None
            if runtime_overlay:
                shadow = directory / 'shadow_cuda_runtime.h'
                shadow.write_text('struct cudaArray;\n'
                                  'typedef unsigned long long cudaTextureObject_t;\n')
                overlay = directory / 'runtime_overlay.json'
                overlay.write_text(json.dumps({'version': 0, 'use-external-names': False, 'roots': [{
                    'type': 'file', 'name': str(cuda / 'include/cuda_runtime.h'),
                    'external-contents': str(shadow),
                }]}))
                paths.extend((shadow, overlay))
            # Prove __FILE__ at the real source identity, even though the
            # resulting C++ is later written to a different output path.
            if name == 'original_identity':
                source.write_text(HOST + 'static_assert(__builtin_strcmp(__FILE__, ' +
                                  json.dumps(str(source)) + ') == 0, "wrong source identity");\n')
            before = {str(p): digest(p) for p in paths}
            if database is None:
                source_argument = (os.path.relpath(source)
                                   if name == 'relative_explicit_runtime' else str(source))
                argv = base + [source_argument, '-o', str(output), *tool,
                               '--', '-std=c++17', *flags]
            else:
                entries = database(source, directory)
                db = directory / 'compile_commands.json'
                db.write_text(json.dumps(entries))
                before[str(db)] = digest(db)
                paths.append(db)
                argv = base + [str(source), '-p', str(directory), '-o', str(output), *tool]
                if flags is not None:
                    argv += ['--', *flags]
            if overlay is not None:
                argv += ['-ivfsoverlay', str(overlay)]
            result = subprocess.run(argv, text=True, capture_output=True, timeout=120)
            assert {str(p): digest(p) for p in paths} == before, (name, 'input changed')
            assert (result.returncode == 0) == good, (name, argv, result.stdout, result.stderr)
            if good:
                text = output.read_text()
                if exact:
                    assert text == source.read_text(), (name, text)
                if compile_host:
                    checked([cxx, '-std=c++17', '-x', 'c++', '-c', str(source),
                             '-o', str(directory / 'original.o')])
                    checked([cxx, '-std=c++17', '-x', 'c++', '-c', str(output),
                             '-o', str(directory / 'translated.o')])
            else:
                assert not output.exists(), (name, 'failed conversion published output')
                assert not Path(str(output) + '.headers').exists(), (name, 'failed closure published')
                if diagnostic:
                    assert diagnostic in result.stderr, (name, result.stderr)
            count += 1
            print(name + ': PASS')
            return source, output, result

        _, host_output, _ = run('custom_uchar4', exact=True, compile_host=True)
        run('tool_host_language', tool=('--input-language=c++',), flags=(), exact=True)
        run('tool_and_compiler_agree', tool=('--input-language=c++',), exact=True)
        run('joined_language', flags=('-xc++',), exact=True)
        run('language_alias', flags=('--language=c++',), exact=True)
        run('last_language_host', flags=('-x', 'cuda', '-x', 'c++'), exact=True)
        run('last_language_cuda', CUDA, flags=('-x', 'c++', '-x', 'cuda'))
        run('default_cuda_cpp', CUDA, flags=())
        run('explicit_cuda', CUDA, tool=('--input-language=cuda',), flags=())
        run('original_identity')
        run('ordinary_cpp_standard', 'template<class T> concept Integral = __is_integral(T);\n'
            'static_assert(Integral<int>);\n', flags=('-xc++', '-std=c++20'), exact=True)
        run('no_ast_bypass', 'int value = undeclared_symbol;\n', good=False,
            diagnostic='undeclared_symbol')
        run('host_conflicts_cuda', tool=('--input-language=c++',), flags=('-xcuda',),
            good=False, diagnostic='conflicts with compiler -x')
        run('cuda_conflicts_host', tool=('--input-language=cuda',), good=False,
            diagnostic='conflicts with compiler -x')
        run('cu_misclassified_host', '#ifdef __CUDACC__\n__global__ void k() {}\n#endif\n',
            suffix='.cu', good=False, diagnostic='refuses a .cu root')
        run('invalid_language', tool=('--input-language=c',), good=False,
            diagnostic='invalid --input-language')
        run('unsupported_x', flags=('-x', 'c'), good=False,
            diagnostic='unsupported compiler -x language')
        for name, flags in [
            ('host_cuda_host_only', ('-xc++', '--cuda-host-only')),
            ('host_cuda_arch', ('-xc++', '--cuda-gpu-arch=sm_80')),
            ('host_rdc', ('-xc++', '-fgpu-rdc')),
            ('host_cc1_override', ('-xc++', '-Xclang', '-fcuda-is-device')),
            ('host_wp_cuda_predefine', ('-xc++', '-Wp,-D__CUDA__=1')),
            ('host_xpreprocessor_cuda_predefine', ('-xc++', '-Xpreprocessor', '-D__CUDACC__=1')),
            ('host_cuda_predefine', ('-xc++', '-D__CUDACC__=1')),
        ]:
            run(name, flags=flags, good=False, diagnostic='host C++ input language conflicts')
        run('host_tool_arch', tool=('--cuda-gpu-arch=sm_80',), good=False,
            diagnostic='host C++ input language conflicts')
        run('host_tool_predefine', tool=('-D__CUDA_ARCH__=800',), good=False,
            diagnostic='host C++ input language conflicts')
        run('host_amap', tool=('--amap',), good=False,
            diagnostic='host C++ input language conflicts')

        # This must be transformed by the existing AST/Runtime API pipeline,
        # rather than passed through as an already-host C++ file.
        _, output, _ = run('explicit_runtime',
            '#include <cuda_runtime.h>\ncudaError_t release(void *p) { return cudaFree(p); }\n')
        translated = output.read_text()
        assert '#include <cuda_runtime.h>' not in translated, translated
        assert 'cudaFree(p)' in translated and 'ascify::cudaFree(p)' in translated, translated
        _, relative_output, _ = run('relative_explicit_runtime',
            '#include <cuda_runtime.h>\ncudaError_t release(void *p) { return cudaFree(p); }\n')
        assert 'ascify::cudaFree(p)' in relative_output.read_text()

        # A selected physical header is reparsed as a host-C++ main file.
        # Its proved whole-file guard is one structural level, not an
        # arbitrary preprocessing condition or a new source of SDK identity.
        texture_declarations = ('extern "C" void consume_texture(cudaArray*, '
                                'cudaTextureObject_t);\n')
        runtime_include = '#include <cuda_runtime.h>\n'
        texture_body = runtime_include + texture_declarations
        outer_prefix = '#ifndef HOST_TEXTURE_DECLS\n#define HOST_TEXTURE_DECLS\n'
        guarded_texture = outer_prefix + texture_body + '#endif\n'
        for name, header_text in (
                ('guarded_runtime_texture_header', guarded_texture),
                ('guarded_runtime_quote_header', guarded_texture.replace(
                    '<cuda_runtime.h>', '"cuda_runtime.h"'))):
            _, output, _ = run(name, '#include "detail/texture_decls.h"\n',
                tool=('--local-headers-recursive',),
                headers={'detail/texture_decls.h': header_text})
            emitted = list(Path(str(output) + '.headers').rglob('texture_decls.h.dpp'))
            assert len(emitted) == 1, (name, emitted)
            text = emitted[0].read_text()
            assert outer_prefix in text and '#endif' in text, (name, text)
            assert '#include <cuda_runtime.h>' not in text and '#include "cuda_runtime.h"' not in text
            assert 'ascify::cudaArray*' in text and 'ascify::cudaTextureObject_t' in text, (name, text)

        for name, header_text in (
                ('guarded_runtime_arbitrary_if', '#if 1\n' + texture_body + '#endif\n'),
                ('guarded_runtime_nested_if', outer_prefix + '#if 1\n' + texture_body + '#endif\n#endif\n'),
                ('guarded_runtime_redefined_guard', outer_prefix +
                    '#undef HOST_TEXTURE_DECLS\n#define HOST_TEXTURE_DECLS\n' + texture_body + '#endif\n'),
                ('guarded_runtime_digraph_mutation', outer_prefix +
                    '%:undef HOST_TEXTURE_DECLS\n#define HOST_TEXTURE_DECLS\n' + texture_body + '#endif\n'),
                ('guarded_runtime_macro_stack', outer_prefix +
                    '#pragma push_macro("HOST_TEXTURE_DECLS")\n' + texture_body +
                    '#pragma pop_macro("HOST_TEXTURE_DECLS")\n#endif\n'),
                ('guarded_runtime_tail_observer', guarded_texture +
                    '#ifdef HOST_TEXTURE_DECLS\nint observed_guard = 1;\n#endif\n')):
            run(name, '#include "detail/texture_decls.h"\n',
                tool=('--local-headers-recursive',),
                headers={'detail/texture_decls.h': header_text}, good=False,
                diagnostic='texture name is not owned by the admitted CUDA SDK')
        run('guarded_runtime_caller_main', guarded_texture, good=False,
            diagnostic='texture name is not owned by the admitted CUDA SDK')
        run('guarded_runtime_shadow_path', '#include "detail/texture_decls.h"\n',
            tool=('--local-headers-recursive',),
            headers={'detail/texture_decls.h': guarded_texture.replace(
                '<cuda_runtime.h>', '"cuda_runtime.h"'),
                'detail/cuda_runtime.h': 'struct cudaArray;\n'
                    'typedef unsigned long long cudaTextureObject_t;\n'}, good=False,
            diagnostic='texture name is not owned by the admitted CUDA SDK')
        run('guarded_runtime_vfs_override', '#include "detail/texture_decls.h"\n',
            tool=('--local-headers-recursive',),
            headers={'detail/texture_decls.h': guarded_texture}, runtime_overlay=True,
            good=False, diagnostic='Ascify cannot translate input selected by VFS overlay arguments')

        header = '#ifndef HOST_LEAF\n#define HOST_LEAF\n' + HOST + '#endif\n'
        _, output, _ = run('local_cuh_inherits_host', '#include "detail/leaf.cuh"\n',
            tool=('--local-headers-recursive',), headers={'detail/leaf.cuh': header})
        bundle = Path(str(output) + '.headers')
        emitted = list(bundle.rglob('leaf.cuh.dpp'))
        assert len(emitted) == 1 and 'typedef struct' in emitted[0].read_text()
        assert '#error host input received CUDA language semantics' in emitted[0].read_text()
        checked([cxx, '-std=c++17', '-x', 'c++', '-c', str(output),
                 '-o', str(output.with_suffix('.o'))])

        contextual = ('#ifdef __CUDACC__\n#error parent used CUDA mode\n#endif\n'
                      'constexpr int parent_value = 19;\n#include "leaf.cuh"\n')
        leaf = ('#ifdef __CUDACC__\n#error child used CUDA mode\n#endif\n'
                'int read_parent() { return parent_value; }\n')
        source, output, result = run('parent_context_inherits_host', contextual,
            tool=('--local-headers-recursive',), headers={'leaf.cuh': leaf})
        marker = Path(str(output) + '.headers/.ascify-local-closure').read_text()
        assert 'context_proof=expanded-tokens-and-final-macros-v1' in marker, marker
        assert 'context_input=' + str(source) + '\t' + digest(source) in marker
        assert '#include "leaf.cuh"' not in output.read_text()
        assert 'original parent source context' in result.stdout
        checked([cxx, '-std=c++17', '-x', 'c++', '-c', str(output),
                 '-o', str(output.with_suffix('.o'))])
        run('parent_context_bad_include_level', contextual,
            tool=('--local-headers-recursive',),
            headers={'leaf.cuh': 'constexpr int level = __INCLUDE_LEVEL__;\n' + leaf},
            good=False, diagnostic='expanded tokens or final macro state differ')

        def db(language, *, separator=False, duplicate=False):
            def entries(source, directory):
                command = [cxx, '-std=c++17', '-x', language]
                if separator:
                    command.append('--')
                command.append(str(source))
                entry = {'directory': str(directory), 'file': str(source), 'arguments': command}
                return [entry, dict(entry)] if duplicate else [entry]
            return entries

        run('compile_db_host', database=db('c++'), flags=None, exact=True)
        run('compile_db_separator', database=db('c++', separator=True), flags=None, exact=True)
        run('compile_db_conflict', database=db('cuda'), flags=None,
            tool=('--input-language=c++',), good=False, diagnostic='conflicts with compiler -x')
        run('compile_db_cli_override', database=db('cuda'), flags=('-xc++',), exact=True)
        run('compile_db_duplicate', database=db('c++', duplicate=True), flags=None,
            good=False, diagnostic='exactly one original TU compile command')
        run('extra_arg_before', flags=('-xc++',),
            tool=('--extra-arg-before=-xcuda',), exact=True)
        # CommonOptionsParser appends this after the input in a Fixed DB.
        # Do not reinterpret a trailing language switch as applying to that input.
        run('extra_arg_after_input', tool=('--extra-arg=-xcuda',), good=False,
            diagnostic='place compiler -x before')

        # Translate both original units, then compile and link their unchanged
        # extern-C ABI. This CPU executable contains no CUDA/Ascend Runtime call.
        run('mixed_host_caller',
            'typedef struct { unsigned char x, y, z, w; } uchar4;\n'
            'extern "C" unsigned host_pixel(uchar4);\n'
            'int main() { uchar4 p{1,2,3,4}; return host_pixel(p) == 10 ? 0 : 1; }\n',
            flags=('-xc++',), suffix='.cpp', exact=True)
        # A real CUDA TU uses the SDK vector declaration and has the same ABI.
        _, real_cuda_output, _ = run('mixed_real_cuda_caller',
            'extern "C" unsigned host_pixel(uchar4);\n'
            'int main() { uchar4 p{1,2,3,4}; return host_pixel(p) == 10 ? 0 : 1; }\n',
            flags=('-xcuda',), suffix='.cu')
        # Keep the SDK's definition for the CPU compile of the CUDA-origin TU;
        # do not force it into the independent host-origin TU.
        checked([cxx, '-std=c++17', '-x', 'c++', '-c', str(real_cuda_output),
                 '-include', str(cuda / 'include/vector_types.h'),
                 '-I' + str(cuda / 'include'), '-o', str(work / 'caller.o')])
        checked([cxx, '-std=c++17', '-x', 'c++', '-c', str(host_output),
                 '-o', str(work / 'pixel.o')])
        checked([cxx, str(work / 'caller.o'), str(work / 'pixel.o'), '-o', str(work / 'mixed')])
        checked([str(work / 'mixed')])
        print('mixed CUDA/C++ extern-C link and CPU value: PASS')
        print('host C++ translation cases:', count)


if __name__ == '__main__':
    main()
