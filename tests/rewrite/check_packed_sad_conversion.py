#!/usr/bin/env python3
"""Real CUDA source proof for one literal unsigned packed-byte SAD instruction."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

INSTRUCTION = 'vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;'
ASM = 'asm("' + INSTRUCTION + '" : "=r"(value) : "r"(a), "r"(b), "r"(c))'
LOWERED = '(value) = ::ascify::packed_u8_sad4_add((a), (b), (c))'
REGION = re.compile(r'/\*sad_begin\*/(.*?)/\*sad_end\*/', re.S)


@dataclass
class Case:
    name: str
    code: str
    expected: str
    refused: bool = False
    local_header: bool = False


def case(name, statement=ASM, expected=None, owner='__device__',
         local='unsigned int value;', prefix='', suffix='',
         parameters='unsigned int a, unsigned int b, unsigned int c'):
    return Case(name, prefix + '\n' + owner + ' void sample(' + parameters + ') {\n' +
                local + '\n/*sad_begin*/' + statement + '/*sad_end*/;\n' + suffix + '\n}\n',
                statement if expected is None else expected)


def cases(clang_major):
    yield case('device_parameters', expected=LOWERED)
    yield case('global_parameters', owner='__global__', expected=LOWERED)
    yield case('private_local_inputs', parameters='',
               local='unsigned value; unsigned a=0,b=1,c=0xffffffffu;', expected=LOWERED)
    yield case('const_local_inputs', parameters='',
               local='unsigned value; const unsigned a=1,b=2,c=3;', expected=LOWERED)
    yield case('unsigned_typedef', prefix='using word=unsigned int;',
               local='word value;', parameters='word a,word b,word c', expected=LOWERED)
    yield case('same_decoded_literals', statement=ASM.replace('"'+INSTRUCTION+'"',
               '"vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;"'), expected=LOWERED)
    yield case('parenthesized_values', statement=ASM.replace('(value)', '((value))').replace('(a)', '((a))'),
               expected='((value)) = ::ascify::packed_u8_sad4_add(((a)), (b), (c))')
    yield case('unsigned_constants', statement=ASM.replace('"r"(a), "r"(b), "r"(c)',
               '"r"(0u), "r"(4294967295u), "r"(1u)'),
               expected='(value) = ::ascify::packed_u8_sad4_add((0u), (4294967295u), (1u))')
    yield case('same_output_input', local='unsigned value=0xffffffffu;',
               statement=ASM.replace('"r"(c)', '"r"(value)'),
               expected=LOWERED.replace('(c)', '(value)'))
    yield case('unbraced_if_else', parameters='unsigned a,unsigned b,unsigned c,bool flag',
               statement='if(flag) '+ASM+'; else value=0u',
               expected='if(flag) '+LOWERED+'; else value=0u')
    yield case('two_statements', statement=ASM+'; '+ASM, expected=LOWERED+'; '+LOWERED)
    yield Case('original_usad4_wrapper', '__device__ unsigned int __usad4(unsigned int A,unsigned int B,unsigned int C=0){unsigned int result;\n/*sad_begin*/'+
               ASM.replace('(value)', '(result)').replace('(a)', '(A)').replace('(b)', '(B)').replace('(c)', '(C)').replace('"'+INSTRUCTION+'"',
               '"vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;"')+'/*sad_end*/;return result;}\n',
               '(result) = ::ascify::packed_u8_sad4_add((A), (B), (C))')
    header = case('selected_local_header', expected=LOWERED)
    header.local_header = True
    yield header
    yield case('signed_output', local='int value;')
    yield case('wide_output', local='unsigned long long value;')
    yield case('volatile_output', local='volatile unsigned value;')
    yield case('reference_output', local='unsigned storage; unsigned& value=storage;')
    yield case('parameter_output', local='', parameters='unsigned a,unsigned b,unsigned c,unsigned value')
    yield case('static_output', local='static unsigned value;')
    yield case('shared_output', local='__shared__ unsigned value;')
    yield case('attributed_output', local='unsigned value __attribute__((unused));')
    yield case('global_output', prefix='__device__ unsigned value;', local='')
    yield case('member_output', local='struct S{unsigned value;};S object;', statement=ASM.replace('(value)', '(object.value)'))
    yield case('subscript_output', local='unsigned values[1];', statement=ASM.replace('(value)', '(values[0])'))
    yield case('side_effect_output', local='unsigned values[2];int index=0;', statement=ASM.replace('(value)', '(values[index++])'))
    yield case('signed_input', parameters='int a,unsigned b,unsigned c')
    yield case('wide_input', parameters='unsigned long long a,unsigned b,unsigned c')
    yield case('volatile_input', parameters='volatile unsigned a,unsigned b,unsigned c')
    yield case('reference_input', parameters='unsigned& a,unsigned b,unsigned c')
    yield case('static_input', parameters='unsigned b,unsigned c', local='unsigned value;static unsigned a;')
    yield case('shared_input', parameters='unsigned b,unsigned c', local='unsigned value;__shared__ unsigned a;')
    yield case('attributed_input', parameters='unsigned a __attribute__((unused)),unsigned b,unsigned c')
    yield case('global_input', parameters='unsigned b,unsigned c', prefix='__device__ unsigned a;')
    yield case('side_effect_input', statement=ASM.replace('"r"(a)', '"r"(a++)'))
    yield case('binary_input', statement=ASM.replace('"r"(a)', '"r"(a+b)'))
    yield case('cast_input', statement=ASM.replace('"r"(a)', '"r"(static_cast<unsigned>(a))'))
    yield case('call_input', prefix='__device__ unsigned take(unsigned x){return x;}',
               statement=ASM.replace('"r"(a)', '"r"(take(a))'))
    yield case('dereference_input', parameters='unsigned* a,unsigned b,unsigned c', statement=ASM.replace('"r"(a)', '"r"(*a)'))
    yield case('signed_literal_input', statement=ASM.replace('"r"(a)', '"r"(1)'))
    yield case('volatile_asm', statement=ASM.replace('asm(', 'asm volatile('))
    yield case('readwrite_constraint', local='unsigned value=0;', statement=ASM.replace('"=r"', '"+r"'))
    yield case('earlyclobber_constraint', statement=ASM.replace('"=r"', '"=&r"'))
    yield case('named_output', statement=ASM.replace('"=r"', '[out] "=r"'))
    yield case('named_input', statement=ASM.replace('"r"(a)', '[lhs] "r"(a)'))
    yield case('matching_constraint', local='unsigned value=0;', statement=ASM.replace('"r"(a)', '"0"(a)'))
    yield case('memory_clobber', statement=ASM[:-1]+' : "memory")')
    yield case('goto_asm', statement=ASM.replace('asm(', 'asm goto(')[:-1]+' : : target)', suffix='target:;')
    yield case('signed_opcode', statement=ASM.replace('vabsdiff4.u32.u32.u32', 'vabsdiff4.s32.s32.s32'))
    yield case('other_opcode', statement=ASM.replace('vabsdiff4', 'vadd4'))
    yield case('no_add', statement=ASM.replace('.add ', ' '))
    yield case('saturation', statement=ASM.replace('.add ', '.sat.add '))
    yield case('byte_selector', statement=ASM.replace('%1,', '%1.b3210,'))
    yield case('another_instruction', statement=ASM.replace('%3;', '%3; add.u32 %0, %0, 1;'))
    yield case('different_spacing', statement=ASM.replace('%0, %1', '%0,%1'))
    yield case('host_function', owner='')
    yield case('host_device_function', owner='__host__ __device__')
    yield case('template_function', owner='template<class T> __device__')
    yield Case('template_class_method', 'template<class T>struct Box{__device__ void sample(unsigned a,unsigned b,unsigned c){unsigned value;\n/*sad_begin*/'+ASM+'/*sad_end*/;}};\n', ASM)
    yield Case('lambda', '__device__ void sample(unsigned a,unsigned b,unsigned c){auto f=[] __device__ (unsigned a,unsigned b,unsigned c){unsigned value;\n/*sad_begin*/'+ASM+'/*sad_end*/;};f(a,b,c);}\n', ASM)
    yield case('macro_template', prefix='#define PROJECT_ASM_TEXT "'+INSTRUCTION+'"', statement=ASM.replace('"'+INSTRUCTION+'"', 'PROJECT_ASM_TEXT'))
    yield case('macro_concat', prefix='#define PROJECT_TAIL " %0, %1, %2, %3;"', statement=ASM.replace('"'+INSTRUCTION+'"', '"vabsdiff4.u32.u32.u32.add" PROJECT_TAIL'))
    yield case('macro_output_constraint', prefix='#define PROJECT_OUTPUT "=r"', statement=ASM.replace('"=r"', 'PROJECT_OUTPUT'))
    yield case('macro_input_constraint', prefix='#define PROJECT_INPUT "r"', statement=ASM.replace('"r"(a)', 'PROJECT_INPUT(a)'))
    yield case('macro_input', prefix='#define PROJECT_VALUE a', statement=ASM.replace('"r"(a)', '"r"(PROJECT_VALUE)'))
    yield case('macro_punctuation', prefix='#define PROJECT_COLON :', statement=ASM.replace(' : ', ' PROJECT_COLON '))
    yield case('macro_whole_statement', prefix='#define PROJECT_ASM '+ASM, statement='PROJECT_ASM')
    conditional = ASM.replace('"'+INSTRUCTION+'"', '\n#if 1\n"'+INSTRUCTION+'"\n#else\n"vadd4.u32.u32.u32.add %0, %1, %2, %3;"\n#endif\n')
    yield case('conditional_instruction_literal', statement=conditional)
    yield case('digraph_conditional_literal', statement=conditional.replace('#if', '%:if').replace('#else', '%:else').replace('#endif', '%:endif'))
    yield case('pragma_operator', statement=ASM.replace('asm(', 'asm(_Pragma("clang diagnostic push") '), suffix='_Pragma("clang diagnostic pop")')
    spliced_pragma = '_Pra' + chr(92) + '\n' + 'gma("clang diagnostic push") '
    yield case('line_spliced_pragma_operator', statement=ASM.replace('asm(', 'asm('+spliced_pragma), suffix='_Pragma("clang diagnostic pop")')
    yield case('inactive_pragma_operator', statement=ASM.replace('asm(', 'asm(\n#if 0\n__pragma(pop_macro("PROJECT_X"))\n#endif\n'))
    if clang_major >= 21:
        yield case('constexpr_template', prefix='#include <string_view>', statement=ASM.replace('"'+INSTRUCTION+'"', '(std::string_view("'+INSTRUCTION+'"))'))
        yield case('constexpr_input_constraint', prefix='#include <string_view>', statement=ASM.replace('"r"(a)', '(std::string_view("r"))(a)'))
        yield case('constexpr_output_constraint', prefix='#include <string_view>', statement=ASM.replace('"=r"', '(std::string_view("=r"))'))
        yield case('dependent_input_constraint', owner='template<class T> __device__', statement=ASM.replace('"r"(a)', '(T::constraint())(a)'))
    for name, prefix in [
        ('helper_macro', '#define packed_u8_sad4_add(...) 17u'),
        ('helper_undefined_macro', '#define packed_u8_sad4_add(...) 17u\n#undef packed_u8_sad4_add'),
        ('namespace_macro', '#define ascify project'),
        ('namespace_declaration', 'namespace ascify{}'),
    ]:
        row=case(name,prefix=prefix);row.refused=True;yield row

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--cuda-path', required=True, type=Path)
    parser.add_argument('--resource-dir', required=True, type=Path)
    parser.add_argument('--clang', type=Path)
    parser.add_argument('--evidence-dir', type=Path)
    parser.add_argument('--source-only', action='store_true',
                        help='Validate inputs only; does not claim rewrite or target success')
    args = parser.parse_args()
    binary, cuda, resource = args.binary.resolve(), args.cuda_path.resolve(), args.resource_dir.resolve()
    clang = args.clang.resolve() if args.clang else None
    if clang is None:
        choices = [binary.parent / 'clang++', resource.parents[2] / 'bin/clang++']
        on_path = shutil.which('clang++')
        if on_path:
            choices.append(Path(on_path))
        clang = next((p.resolve() for p in choices if p.is_file()), None)
    if clang is None or not clang.is_file() or not binary.is_file():
        parser.error('supply a real binary and its matching --clang')
    if not (cuda / 'include/cuda_runtime.h').is_file() or not (resource / 'include/__clang_cuda_runtime_wrapper.h').is_file():
        parser.error('supply real CUDA and complete resource headers')
    version = subprocess.run([str(clang), '--version'], text=True, capture_output=True, check=True).stdout
    major = re.search(r'\bclang version (\d+)(?:\.|\s)', version)
    if major is None or major.group(1) != resource.name.split('.')[0]:
        parser.error('clang version must match resource directory')
    with tempfile.TemporaryDirectory(prefix='ascify-packed-sad-') as temporary:
        work = args.evidence_dir.resolve() if args.evidence_dir else Path(temporary)
        if args.evidence_dir:
            work.mkdir(parents=True, exist_ok=False)
        summary = {'schema': 'ascify.packed-sad-conversion.v1', 'passed': False,
                   'binary': str(binary), 'binary_sha256': sha(binary),
                   'clang': str(clang), 'clang_sha256': sha(clang), 'clang_version': version,
                   'source_only': args.source_only, 'cases': [],
                   'scope': 'real CUDA source syntax and frontend boundaries; no target build or device execution'}
        def save():
            (work / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        def run(directory, label, command):
            (directory / (label + '.argv.json')).write_text(json.dumps(command, indent=2) + '\n')
            result = subprocess.run(command, text=True, capture_output=True, timeout=60)
            (directory / (label + '.stdout.log')).write_text(result.stdout)
            (directory / (label + '.stderr.log')).write_text(result.stderr)
            return result
        save()
        try:
            for item in cases(int(major.group(1))):
                directory = work / item.name
                directory.mkdir()
                source, output, receipt = (directory / name for name in ('input.cu', 'output.cpp', 'receipt.json'))
                header = directory / 'leaf.cuh'
                header_identity = None
                if item.local_header:
                    header.write_text('#ifndef PROJECT_SAD_HEADER\n#define PROJECT_SAD_HEADER\n' + item.code + '#endif\n')
                    header_identity = sha(header)
                    source.write_text('#include <cuda_runtime.h>\n#include "leaf.cuh"\n')
                    original = header.read_text()
                else:
                    source.write_text('#include <cuda_runtime.h>\n' + item.code)
                    original = source.read_text()
                regions = REGION.findall(original)
                if len(regions) != 1:
                    raise AssertionError('one statement region required')
                row = {'case': item.name, 'input_sha256': sha(source), 'passed': False,
                       'expected_rewrite': regions[0] != item.expected,
                       'expected_publication_refusal': item.refused}
                summary['cases'].append(row)
                syntax = run(directory, 'syntax', [str(clang), str(source), '-x', 'cuda',
                    '-std=c++17', '-fsyntax-only', '--cuda-host-only', '-nocudalib',
                    '--cuda-path=' + str(cuda), '-resource-dir=' + str(resource), '-Wno-unknown-cuda-version'])
                row['syntax_returncode'] = syntax.returncode
                save()
                if syntax.returncode:
                    raise AssertionError(item.name + ': invalid CUDA source: ' + syntax.stderr)
                if not args.source_only:
                    conversion = run(directory, 'conversion', [str(binary), str(source),
                        '--target-policy=dav-c310-vec', '--simt-math=fast', '--target-recipe=none',
                        '--default-preprocessor', '--cuda-path=' + str(cuda),
                        '--clang-resource-directory=' + str(resource), '--migration-receipt=' + str(receipt),
                        *( ['--local-headers-recursive'] if item.local_header else [] ),
                        '-o', str(output), '--', '-x', 'cuda', '-std=c++17', '-I' + str(directory)])
                    row['conversion_returncode'] = conversion.returncode
                    save()
                    if item.refused:
                        if conversion.returncode == 0 or output.exists():
                            raise AssertionError(item.name + ': unsafe output was published')
                        if 'Ascify cannot publish CUDA compatibility output' not in conversion.stderr:
                            raise AssertionError(item.name + ': unexpected rejection: ' + conversion.stderr)
                    else:
                        if conversion.returncode or json.loads(receipt.read_text())['status'] != 'succeeded':
                            raise AssertionError(item.name + ': conversion failed: ' + conversion.stderr)
                        artifacts = [output]
                        if item.local_header:
                            artifacts += [p for p in Path(str(output) + '.headers').rglob('*') if p.is_file() and p.name != '.ascify-local-closure']
                        actual = '\n'.join(p.read_text() for p in artifacts)
                        if item.local_header:
                            assert len(artifacts) == 2, 'exactly one translated local header required'
                            row['local_header_sha256'] = header_identity
                            row['generated_header_sha256'] = sha(artifacts[1])
                        if REGION.findall(actual) != [item.expected]:
                            raise AssertionError(item.name + ': statement mismatch: ' + repr(REGION.findall(actual)))
                        if actual.count('::ascify::packed_u8_sad4_add(') != item.expected.count('::ascify::packed_u8_sad4_add('):
                            raise AssertionError(item.name + ': unexpected helper invocation')
                        row['output_sha256'] = sha(output)
                    row['receipt_sha256'] = sha(receipt)
                if header_identity is not None and sha(header) != header_identity:
                    raise AssertionError('input local header changed')
                if sha(source) != row['input_sha256']:
                    raise AssertionError('input source changed')
                row['passed'] = True
                save()
                print(item.name + ': source valid' + ('; rewrite boundary verified' if not args.source_only else ''), flush=True)
            if sha(binary) != summary['binary_sha256'] or sha(clang) != summary['clang_sha256']:
                raise AssertionError('frontend changed during validation')
            summary.update(passed=True, total=len(summary['cases']))
            save()
        except Exception as error:
            summary['failure'] = str(error)
            save()
            raise
        print('CUDA packed-byte SAD: ' + str(summary['total']) + ' cases passed' + (' (source only)' if args.source_only else ''))


if __name__ == '__main__':
    main()
