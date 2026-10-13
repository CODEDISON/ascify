#!/usr/bin/env python3
"""Real frontend parent macro/declaration preservation and atomic refusals."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


def identity(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    binary = os.environ['ASCIFY_BINARY']
    cuda = os.environ['ASCIFY_CUDA_PATH']
    resource = os.environ['ASCIFY_CLANG_RESOURCE_DIRECTORY']
    with tempfile.TemporaryDirectory(prefix='ascify-parent-source-') as temp:
        work = Path(temp)
        inputs = work / 'inputs'
        inputs.mkdir()
        root = inputs / 'root.cu'
        child = inputs / 'leaf.cuh'
        output = work / 'converted.cce'
        bundle = Path(str(output) + '.headers')
        marker = bundle / '.ascify-local-closure'

        def convert(dst=output, recursive=True, standard=True):
            args = [binary, str(root), f'--cuda-path={cuda}',
                    f'--clang-resource-directory={resource}', '-o', str(dst)]
            if recursive:
                args.append('--local-headers-recursive')
            if standard:
                args.append('--default-preprocessor')
            args += ['--', '-std=c++17', '-I' + str(inputs)]
            before = identity(inputs.iterdir())
            result = subprocess.run(args, text=True, capture_output=True)
            assert identity(inputs.iterdir()) == before, 'inputs modified'
            assert not list(work.glob('.ascify-local-closure*')), 'staging leaked'
            return result

        def success(label, parent, leaf, needles=()):
            root.write_text(parent)
            child.write_text(leaf)
            result = convert()
            assert result.returncode == 0, label + ': ' + result.stdout + result.stderr
            generated = output.read_text()
            assert '#include "leaf.cuh"' not in generated, generated
            assert 'original parent source context' in result.stdout, result.stdout
            for needle in needles:
                assert needle in generated, (label, needle, generated)
            proof = marker.read_text()
            assert 'context_proof=expanded-tokens-and-final-macros-v1' in proof
            assert f'context_inline={child}\t' in proof
            for path in (root, child):
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                assert f'context_input={path}\t{sha}\n' in proof
            print(label + ': original parent token/macro proof and source SHA preserved')
            return generated

        parent = ('#define WEIGHT 0.70710678118654752440f;\n'
                  'const unsigned LOG_BANKS = 4;\n'
                  '#include "leaf.cuh"\n')
        leaf = ('__global__ void scale(float *data) {\n'
                '  float value = data[threadIdx.x] * WEIGHT;\n'
                '  data[threadIdx.x + (threadIdx.x >> LOG_BANKS)] = value;\n}\n')
        success('macro_semicolon_and_parent_value', parent, leaf,
                ('#define WEIGHT 0.70710678118654752440f;', 'const unsigned LOG_BANKS = 4;', '>> LOG_BANKS'))
        success('constant_only', 'constexpr int offset = 11;\n#include "leaf.cuh"\n',
                '__global__ void add(int *data) { data[threadIdx.x] += offset; }\n',
                ('constexpr int offset = 11;', '+= offset'))
        success('header_macro_decl_reference', 'constexpr int offset = 23;\n#include "leaf.cuh"\n',
                '#define READ_PARENT offset\n__global__ void add(int *data) { data[threadIdx.x] += READ_PARENT; }\n',
                ('constexpr int offset = 23;', '#define READ_PARENT offset'))
        success('record_signature', 'struct Value { int x; };\n#include "leaf.cuh"\n',
                '__device__ int read(Value const &v) { return v.x; }\n', ('struct Value', 'Value const &v'))
        success('typedef_signature', 'struct Value { int x; };\ntypedef Value Alias;\n#include "leaf.cuh"\n',
                '__device__ int read(Alias const &v) { return v.x; }\n', ('typedef Value Alias;', 'Alias const &v'))
        success('alias_template_instantiation',
                'template<class T> struct Value { T x; };\nusing Alias = Value<int>;\n#include "leaf.cuh"\n',
                'template<class T> __device__ T read(Value<T> const &v) { return v.x; }\n'
                '__global__ void add(Alias *v) { v[threadIdx.x].x = read(v[threadIdx.x]); }\n',
                ('using Alias = Value<int>;', 'Value<T> const &v'))
        success('function_macro', '#define COMBINE(x,y) ((x) * 3 + (y))\n#include "leaf.cuh"\n',
                '__global__ void add(int *p) { p[threadIdx.x] = COMBINE(p[threadIdx.x], 5); }\n',
                ('#define COMBINE(x,y) ((x) * 3 + (y))',))
        # Real Samples also obtain signature types from an earlier sibling,
        # rather than from text physically written in the root TU. Keep that
        # sibling in the closure; inline only the dependent leaf in place.
        sibling = inputs / 'types.cuh'
        sibling_parent = '#include "types.cuh"\n#include "leaf.cuh"\n'
        for label, definitions, use, needles in [
            ('sibling_record', 'struct Value { int x; };\n',
             '__device__ int read(Value const &v) { return v.x; }\n', ('Value const &v',)),
            ('sibling_typedef', 'struct Value { int x; };\ntypedef Value Alias;\n',
             '__device__ int read(Alias const &v) { return v.x; }\n', ('Alias const &v',)),
            ('sibling_template', 'template<class T> struct Value { T x; };\n',
             '__device__ int read(Value<int> const &v) { return v.x; }\n', ('Value<int> const &v',)),
            ('sibling_constant', 'constexpr int bias = 7;\n',
             '__global__ void add(int *p) { p[threadIdx.x] += bias; }\n', ('+= bias',)),
            ('sibling_macro', '#define SIBLING_BIAS 13\n',
             '__global__ void add(int *p) { p[threadIdx.x] += SIBLING_BIAS; }\n', ('+= SIBLING_BIAS',)),
        ]:
            sibling.write_text(definitions)
            generated = success(label, sibling_parent, use, needles)
            proof = marker.read_text()
            assert f'context_input={sibling}\t{hashlib.sha256(sibling.read_bytes()).hexdigest()}\n' in proof
            assert f'context_inline={sibling}\t' not in proof
        first_root = root
        for value in [3, 19]:
            root = inputs / f"parent_{value}.cu"
            success(f'redefined_parent_{value}',
                    f'#define BIAS 1\n#undef BIAS\n#define BIAS {value}\n#include "leaf.cuh"\n',
                    '__global__ void add(int *p) { p[threadIdx.x] += BIAS; }\n',
                    (f'#define BIAS {value}',))
        root = first_root
        # Preserve line/file/counter meaning across child and parent, including
        # a macro defined in the child and expanded after return to the root.
        success('source_builtins', '#line 200 "logical-parent.cu"\n' + parent +
                'const char *file_after = DEFERRED_FILE;\nconstexpr int count_after = __COUNTER__;\n',
                '#define DEFERRED_FILE __FILE__\nconst char *file_here = __FILE__;\n'
                'constexpr int line_here = __LINE__;\nconstexpr int count_here = __COUNTER__;\n' + leaf)
        published = identity([output, marker])

        def reject(label, child_text, parent_text=parent, reason='contextual', standard=True):
            root.write_text(parent_text)
            child.write_text(child_text)
            result = convert(standard=standard)
            assert result.returncode > 0, label + ': unsafe context accepted'
            assert reason in result.stderr, label + ': ' + result.stderr
            assert identity([output, marker]) == published, label + ': prior output changed'
            fresh = work / (label + '.cce')
            result = convert(fresh, standard=standard)
            assert result.returncode > 0 and not fresh.exists(), label
            assert not Path(str(fresh) + '.headers').exists(), label
            print(label + ': rejected atomically on fresh output and retry')

        reject('nonstandard_preprocessing', leaf, reason='require standard preprocessing', standard=False)
        guarded = '#ifndef LEAF_GUARD\n#define LEAF_GUARD\n' + leaf + '#endif\n'
        reject('multiple_includes', guarded, parent + '#include "leaf.cuh"\n', 'one direct include occurrence')
        reject('include_level', 'constexpr int level = __INCLUDE_LEVEL__;\n' + leaf,
               reason='expanded tokens or final macro state differ')
        reject('macro_final_state', '#if __INCLUDE_LEVEL__ == 1\n#define END_STATE 1\n'
               '#else\n#define END_STATE 2\n#endif\n' + leaf,
               reason='expanded tokens or final macro state differ')
        reject('counter_state', '#if __INCLUDE_LEVEL__ == 1\n#if __COUNTER__ == 0\n#endif\n#endif\n' + leaf,
               reason='expanded tokens or final macro state differ')
        reject('pragma', '#pragma pack(push, 1)\n' + leaf + '#pragma pack(pop)\n',
               reason='leaf without pragmas')
        (inputs / 'nested.h').write_text('constexpr int nested = 1;\n')
        reject('nested_include', '#include "nested.h"\n' + leaf, reason='leaf without pragmas')
        sibling.write_text('struct Value { int x; };\n')
        sibling_leaf = '__device__ int read(Value const &v) { return v.x; }\n'
        reject('sibling_include_level', 'constexpr int level = __INCLUDE_LEVEL__;\n' + sibling_leaf,
               parent_text=sibling_parent, reason='expanded tokens or final macro state differ')
        reject('sibling_pragma', '#pragma pack(push, 1)\n' + sibling_leaf + '#pragma pack(pop)\n',
               parent_text=sibling_parent, reason='leaf without pragmas')
        root.write_text(parent)
        child.write_text(leaf)
        plain = work / 'nonrecursive.cce'
        result = convert(plain, recursive=False)
        assert result.returncode == 0, result.stderr
        assert '#include "leaf.cuh"' in plain.read_text()
        assert not Path(str(plain) + '.headers').exists()
        # An independent leaf retains ordinary staged closure rather than
        # entering the source-context path simply because it is local.
        root.write_text('#include "leaf.cuh"\n')
        child.write_text('__global__ void clear(int *p) { p[threadIdx.x] = 0; }\n')
        normal = work / 'independent.cce'
        result = convert(normal)
        assert result.returncode == 0, result.stderr
        assert 'Proven contextual' not in result.stdout
        assert 'context_inline=' not in Path(str(normal) + '.headers/.ascify-local-closure').read_text()
        print('nonrecursive behavior and independent staged header unchanged')
        # A direct header that includes the same provider itself is not a leaf
        # requiring sibling inheritance. It must use ordinary staged closure.
        for label, definitions, use in [
            ('nonleaf_record_typedef_template',
             'struct Value { int x; };\ntypedef Value Alias;\n'
             'template<class T> struct Box { T x; };\n',
             '__device__ int read(Value const &v, Alias const &a, Box<int> const &b) '
             '{ return v.x + a.x + b.x; }\n'),
            ('nonleaf_constant', 'constexpr int bias = 7;\n',
             '__global__ void add(int *p) { p[threadIdx.x] += bias; }\n'),
            ('nonleaf_macro', '#define SIBLING_BIAS 13\n',
             '__global__ void add(int *p) { p[threadIdx.x] += SIBLING_BIAS; }\n'),
        ]:
            sibling.write_text('#ifndef TYPES_GUARD\n#define TYPES_GUARD\n' + definitions + '#endif\n')
            root.write_text(sibling_parent)
            child.write_text('#include "types.cuh"\n' + use)
            destination = work / (label + '.cce')
            result = convert(destination)
            assert result.returncode == 0, label + ': ' + result.stdout + result.stderr
            proof = Path(str(destination) + '.headers/.ascify-local-closure').read_text()
            assert 'context_inline=' not in proof, (label, proof)
            assert 'original parent source context' not in result.stdout, (label, result.stdout)
            staged = list(Path(str(destination) + '.headers').rglob('leaf.cuh.dpp'))
            assert len(staged) == 1, (label, staged)
            assert 'types.cuh.dpp' in staged[0].read_text(), (label, staged[0].read_text())
            assert len(list(Path(str(destination) + '.headers').rglob('types.cuh.dpp'))) == 1
            print(label + ': self-contained non-leaf uses ordinary staged closure')
        # The first own quoted include may occur after a sibling macro use.
        # This header obtains the macro independently through an angle include;
        # deciding leaf eligibility in MacroExpands would misclassify it.
        sibling.write_text('#ifndef TYPES_GUARD\n#define TYPES_GUARD\n'
                           '#define SIBLING_BIAS 13\n#endif\n')
        (inputs / 'late.h').write_text('constexpr int late_value = 1;\n')
        root.write_text(sibling_parent)
        child.write_text('#include <types.cuh>\n'
                         '__global__ void late(int *p) { p[threadIdx.x] += SIBLING_BIAS; }\n'
                         '#include "late.h"\n')
        destination = work / 'nonleaf_macro_before_late_include.cce'
        result = convert(destination)
        assert result.returncode == 0, result.stdout + result.stderr
        proof = Path(str(destination) + '.headers/.ascify-local-closure').read_text()
        assert 'context_inline=' not in proof, proof
        staged = list(Path(str(destination) + '.headers').rglob('leaf.cuh.dpp'))
        assert len(staged) == 1 and 'late.h.dpp' in staged[0].read_text()
        assert '#include <types.cuh>' in staged[0].read_text()
        print('nonleaf_macro_before_late_include: complete preprocessing determines leaf eligibility')
        print('parent source context: 15 contextual successes, 4 ordinary non-leaf successes, '
              '9 atomic refusal pairs, 2 mode boundaries passed')


if __name__ == '__main__':
    main()
