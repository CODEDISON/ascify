#!/usr/bin/env python3
"""Real syntax/conversion checks; optional CCEC object metadata only, never NPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CUDA = '#include <cuda_runtime.h>\n'


def fixtures():
    def row(name, body, count=1, accepted=True, retained=()):
        return dict(name=name, source=CUDA + body, count=count, accepted=accepted, retained=retained)
    yield row('signed_short', '__global__ void step(short*out){__shared__ short a[16];a[0]=7;out[0]=a[0];}\n')
    yield row('unsigned_short', '__global__ void step(unsigned short*out){__shared__ unsigned short a[16];a[0]=65535;out[0]=a[0];}\n')
    yield row('extent_size_stride', '#define H 32\n#define S 34\n__global__ void step(short*out){__shared__ short a[H*S];static_assert(sizeof(a)==2176,"size");a[34]=3;out[0]=a[34];}\n', retained=('a[H*S]', 'sizeof(a)==2176', 'a[34]=3'))
    yield row('word_pair_helper', '''
__device__ void pairs(unsigned int*p){p[0]=p[0]+1;}
__device__ void rows(short*p,int stride){p[stride]=p[0];}
__global__ void sharedWordPairs(short*out){
 __shared__ short tile[32*34];
 unsigned row=threadIdx.x&31u, col=(threadIdx.y&3u)*8u;
 short*p=tile+col*34+row*2;
 ((int*)p)[0]=1; pairs((unsigned int*)(tile+row*34+col));
 rows(p,34); out[0]=tile[0];
}
''', retained=('short tile[32*34]', 'tile+col*34+row*2', '(tile+row*34+col)', 'rows(p,34)'))
    yield row('signed_word_helper', '__device__ void pairs(int*p){p[0]+=1;}\n__global__ void step(short*out){__shared__ short a[16];pairs((int*)a);out[0]=a[0];}\n')
    yield row('exact_overload_ignores_uncalled_observer', '''
__device__ void transport(short*p,unsigned long*out){out[0]=(unsigned long)p;}
__device__ void transport(unsigned int*p){p[0]+=1;}
__global__ void step(short*out){__shared__ short a[16];transport((unsigned int*)a);out[0]=a[0];}
''')
    yield row('independent_pointer_updates', '__global__ void step(short*out){__shared__ short a[16];a[0]=7;short*p=nullptr;p=a;p+=2;p-=2;p++;p--;out[0]=p[0];}\n')
    yield row('closed_helper_postincrement', '__device__ void write(short*q){q[0]=3;}\n__global__ void step(short*out){__shared__ short a[16];short*p=a;write(p++);out[0]=a[0];}\n')
    yield row('closed_helper_assignment', '__device__ void write(short*q){q[0]=3;}\n__global__ void step(short*out){__shared__ short a[16];short*p=nullptr;write(p=a);out[0]=a[0];}\n')
    yield row('explicit8_before', '__global__ void step(short*out){__shared__ __align__(8) short a[16];out[0]=a[0];}\n', count=0, retained=('ASCIFY_ALIGN(8)',))
    yield row('explicit16_after', '__global__ void step(short*out){__align__(16) __shared__ short a[16];out[0]=a[0];}\n', count=0, retained=('ASCIFY_ALIGN(16)',))
    yield row('explicit2_unchanged', '__global__ void step(short*out){__shared__ __align__(2) short a[16];out[0]=a[0];}\n', count=0, retained=('ASCIFY_ALIGN(2)',))
    yield row('float_unchanged', '__global__ void step(float*out){__shared__ float a[16];out[0]=a[0];}\n', count=0)
    yield row('record_unchanged', 'struct R{short value;};__global__ void step(short*out){__shared__ R a[16];out[0]=a[0].value;}\n', count=0)
    yield row('dynamic_unchanged', '__global__ void step(short*out){extern __shared__ short a[];out[0]=a[0];}\n', count=0)
    yield row('caller_alias_not_expanded', '#define MY_SHARED __shared__\n__global__ void step(short*out){MY_SHARED short a[16];out[0]=a[0];}\n', count=0)
    yield row('caller_declaration_not_expanded', '#define SHARED_ARRAY(name) __shared__ short name[16]\n__global__ void step(short*out){SHARED_ARRAY(a);out[0]=a[0];}\n', count=0)
    yield row('caller_type_not_expanded', '#define ELEMENT short\n__global__ void step(short*out){__shared__ ELEMENT a[16];out[0]=a[0];}\n', count=0)
    negatives = [
        ('alignof_object', '__global__ void step(unsigned*out){__shared__ short a[16];out[0]=__alignof__(a);}\n'),
        ('alignof_alias', '__global__ void step(unsigned*out){__shared__ short a[16];short*p=a;out[0]=__alignof__(*p);}\n'),
        ('integer_address', '__global__ void step(unsigned long*out){__shared__ short a[16];out[0]=(unsigned long)a;}\n'),
        ('cross_object_difference', '__global__ void step(long*out){__shared__ short a[16];__shared__ short b[16];out[0]=a-b;}\n'),
        ('opaque_escape', '__device__ void sink(short*);__global__ void step(){__shared__ short a[16];sink(a);}\n'),
        ('global_escape', '__device__ short* leaked;__global__ void step(){__shared__ short a[16];leaked=a;}\n'),
        ('return_from_helper', '__device__ short* pass(short*p){return p;}__global__ void step(){__shared__ short a[16];pass(a);}\n'),
        ('helper_observes_address', '__device__ void inspect(short*p,unsigned long*out){out[0]=(unsigned long)p;}__global__ void step(unsigned long*out){__shared__ short a[16];inspect(a,out);}\n'),
        ('multiple_declarators', '__global__ void step(short*out){__shared__ short a[16],b[16];out[0]=a[0]+b[0];}\n'),
        ('function_template', 'template<int N> __global__ void step(short*out){__shared__ short a[N];out[0]=a[0];}\ntemplate __global__ void step<16>(short*);\n'),
        ('numeric_pointer_compare', '__global__ void step(bool*out){__shared__ short a[16];out[0]=(a==nullptr);}\n'),
        ('qualifier_redefinition', '#undef __shared__\n#define __shared__ __attribute__((shared))\n__global__ void step(short*out){__shared__ short a[16];out[0]=a[0];}\n'),
        ('exact_overload_rejects_called_observer', '__device__ void transport(short*p){p[0]+=1;}\n__device__ void transport(unsigned int*p,unsigned long*out){out[0]=(unsigned long)p;}\n__global__ void step(unsigned long*out){__shared__ short a[16];transport((unsigned int*)a,out);}\n'),
        ('empty_layout_macro', '#define LAYOUT\n__global__ void step(short*out){__shared__ LAYOUT short a[16];out[0]=a[0];}\n'),
        ('nested_assignment_global_escape', '__device__ short* leaked;__global__ void step(){__shared__ short a[16];short*p=nullptr;leaked=(p=a);}\n'),
        ('nested_assignment_numeric_address', '__global__ void step(unsigned long*out){__shared__ short a[16];short*p=nullptr;out[0]=(unsigned long)(p=a);}\n'),
        ('postincrement_helper_observation', '__device__ void inspect(short*q,unsigned long*out){out[0]=(unsigned long)q;}\n__global__ void step(unsigned long*out){__shared__ short a[16];short*p=a;inspect(p++,out);}\n'),
        ('preincrement_helper_observation', '__device__ void inspect(short*q,unsigned long*out){out[0]=(unsigned long)q;}\n__global__ void step(unsigned long*out){__shared__ short a[16];short*p=a;inspect(++p,out);}\n'),
        ('pointer_holder_helper_observation', '__device__ void inspect(short**q,unsigned long*out){out[0]=(unsigned long)*q;}\n__global__ void step(unsigned long*out){__shared__ short a[16];short*p=a;inspect(&p,out);}\n'),
        ('scalar_assignment_address_escape', '__device__ short* leaked;__global__ void step(){__shared__ short a[16];leaked=&(a[0]=1);}\n'),
        ('scalar_preincrement_address_escape', '__device__ short* leaked;__global__ void step(){__shared__ short a[16];a[0]=0;leaked=&++a[0];}\n'),
        ('GNU_statement_expression_escape', '__device__ short* leaked;__global__ void step(){__shared__ short a[16];short*p=a;leaked=({p;});}\n'),
    ]
    for name, body in negatives:
        yield row(name, body, accepted=False)

    # Preserve the original 40 fixtures above. These eight distinguish a
    # discarded unbraced for-body store from consumed or escaping expressions.
    yield row('unbraced_for_word_store', '''
__global__ void step(int*input,short*out){
 __shared__ short a[32*34]; short*p=a;
 for(int i=0;i<8;++i) ((int*)p)[i*17]=input[i];
 out[0]=a[0];
}
''', retained=('for(int i=0;i<8;++i)', '((int*)p)[i*17]=input[i]'))
    yield row('braced_for_word_store', '''
__global__ void step(int*input,short*out){
 __shared__ short a[32*34]; short*p=a;
 for(int i=0;i<8;++i){ ((int*)p)[i*17]=input[i]; }
 out[0]=a[0];
}
''', retained=('for(int i=0;i<8;++i)', '((int*)p)[i*17]=input[i]'))
    yield row('unbraced_for_unsigned_helper_store', '''
__device__ void words(unsigned int*q){for(int i=0;i<4;++i)q[i]=i;}
__global__ void step(short*out){
 __shared__ short a[16]; words((unsigned int*)a); out[0]=a[0];
}
''', retained=('for(int i=0;i<4;++i)q[i]=i', 'words((unsigned int*)a)'))
    yield row('for_init_element_store_outside_body',
              '__global__ void step(short*out){__shared__ short a[16];for(a[0]=1;false;){}out[0]=a[0];}\n',
              accepted=False)
    yield row('for_condition_element_store_outside_body',
              '__global__ void step(short*out){__shared__ short a[16];for(;(a[0]=1);){}out[0]=a[0];}\n',
              accepted=False)
    yield row('for_increment_element_store_outside_body',
              '__global__ void step(short*out){__shared__ short a[16];for(int i=0;i<8;a[0]=1){++i;}out[0]=a[0];}\n',
              accepted=False)
    yield row('for_body_scalar_assignment_address_escape',
              '__device__ short* leaked;__global__ void step(){__shared__ short a[16];for(int i=0;i<8;++i)leaked=&(a[0]=1);}\n',
              accepted=False)
    yield row('for_body_GNU_statement_expression_escape',
              '__device__ short* leaked;__global__ void step(){__shared__ short a[16];short*p=a;for(int i=0;i<8;++i)leaked=({p;});}\n',
              accepted=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(path):
    return {str(p.relative_to(path)): sha(p) for p in path.rglob('*') if p.is_file()}


def elf_sections(data):
    assert data[:6] == b'\x7fELF\x02\x01', 'expected ELF64 little endian'
    offset = struct.unpack_from('<Q', data, 40)[0]
    entry, count, names_index = struct.unpack_from('<HHH', data, 58)
    assert entry == 64 and 0 < count < 65535 and names_index < count
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, offset + entry*i) for i in range(count)]
    names_header = headers[names_index]
    names = data[names_header[4]:names_header[4]+names_header[5]]
    for h in headers:
        end = names.find(b'\0', h[0])
        assert end >= 0
        name = names[h[0]:end].decode()
        yield name, h[5], h[8], data[h[4]:h[4]+h[5]] if h[1] != 8 else b''


def ub_metadata(path, kernel, expected_bytes):
    found = []
    pending = [(path.name, path.read_bytes())]
    while pending:
        owner, data = pending.pop()
        for name, size, alignment, payload in elf_sections(data):
            if name.startswith('.ubuf.') and kernel in name:
                found.append(dict(owner=owner, section=name, bytes=size, alignment=alignment))
            if name in ('.aicore_binary', '__aicore_rel_binary') and payload.startswith(b'\x7fELF'):
                pending.append((owner + ':' + name, payload))
    assert found and all(row['bytes'] == expected_bytes and row['alignment'] >= 4 for row in found), found
    return found


def main():
    if not __debug__:
        raise RuntimeError('assertions are required')
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'clang', 'cuda-path', 'resource-dir', 'evidence-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dct-source', type=Path)
    parser.add_argument('--dct-common', type=Path)
    parser.add_argument('--ccec', type=Path)
    parser.add_argument('--cann', type=Path)
    parser.add_argument('--ccec-include', type=Path, action='append', default=[])
    a = parser.parse_args()
    assert bool(a.dct_source) == bool(a.dct_common)
    assert not a.ccec or (a.cann and a.ccec_include), 'explicit CANN/CCEC includes required'
    out = a.evidence_dir.resolve(); out.mkdir()
    base_env=os.environ.copy()
    for key in ('LD_PRELOAD','LD_AUDIT','CPATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','OBJC_INCLUDE_PATH','LIBRARY_PATH'):
        base_env.pop(key,None)
    frozen = {str(p.resolve()): sha(p.resolve()) for p in (a.binary, a.clang, Path(__file__))}
    if a.ccec: frozen[str(a.ccec.resolve())] = sha(a.ccec.resolve())
    rows = []
    def run(argv, folder, env=None):
        folder.mkdir()
        argv = list(map(str, argv)); (folder/'argv.json').write_text(json.dumps(argv, indent=2)+'\n')
        result = subprocess.run(argv, capture_output=True, timeout=180, env=base_env if env is None else env)
        (folder/'stdout.txt').write_bytes(result.stdout); (folder/'stderr.txt').write_bytes(result.stderr)
        (folder/'process.json').write_text(json.dumps({'returncode': result.returncode})+'\n')
        return result
    def convert(source, case, flags):
        return [a.binary, source, '--target-policy=dav-c310-vec', '--simt-math=fast',
            '--target-recipe=none', '--default-preprocessor', '--frontend-compat=ascify-admitted-v1',
            '--local-headers-recursive', '--cuda-path='+str(a.cuda_path),
            '--clang-resource-directory='+str(a.resource_dir), '-I'+str(source.parent),
            '-o', case/'output.cce', '--', *flags]
    for row in fixtures():
        case = out/row['name']; case.mkdir(); inputs=case/'inputs'; inputs.mkdir()
        source=inputs/'input.cu'; source.write_text(row['source']); before=inventory(inputs)
        flags=['-x','cuda','-std=c++17','-I'+str(inputs)]
        syntax=run([a.clang,'--cuda-host-only','-nocudalib','--cuda-path='+str(a.cuda_path),
            '-resource-dir='+str(a.resource_dir),'-Wno-unknown-cuda-version',*flags,'-fsyntax-only',source],case/'syntax')
        assert syntax.returncode == 0, (row['name'], syntax.stderr.decode())
        output=case/'output.cce'; bundle=Path(str(output)+'.headers')
        if not row['accepted']:
            output.write_text('previous source\n');bundle.mkdir();(bundle/'previous.txt').write_text('previous closure\n')
            prior=(sha(output),inventory(bundle))
        result=run(convert(source,case,flags),case/'conversion')
        assert (result.returncode == 0) == row['accepted'], (row['name'],result.stderr.decode())
        if row['accepted']:
            text=output.read_text()
            assert text.count('__ubuf__ ASCIFY_ALIGN(4)') == row['count'], (row['name'],text)
            for retained in row['retained']: assert retained in text, (row['name'],retained)
        else:
            assert 'shared word alignment boundary:' in result.stderr.decode(), (row['name'],result.stderr.decode())
            assert (sha(output),inventory(bundle)) == prior, row['name']
            argv=convert(source,case,flags);argv[argv.index('-o')+1]=case/'fresh.cce'
            fresh=run(argv,case/'fresh_conversion')
            assert fresh.returncode != 0 and not (case/'fresh.cce').exists() and not (case/'fresh.cce.headers').exists()
        assert inventory(inputs)==before, 'fixture input changed'
        assert not list(case.glob('.ascify-local-closure*'))
        item=dict(name=row['name'],passed=True,source_syntax_passed=True,expected_min4_arrays=row['count'] if row['accepted'] else 0)
        if a.ccec and row['name']=='word_pair_helper':
            env=base_env.copy();env.update(ASCEND_HOME_PATH=str(a.cann),ASCEND_OPP_PATH=str(a.cann/'opp'),
                LD_LIBRARY_PATH=str(a.cann/'lib64')+':'+str(a.cann/'aarch64-linux/lib64'))
            argv=[a.ccec,'-x','dpp','--cce-aicore-arch=dav-c310-vec','-std=c++17','-O2','-funsigned-char','-c',
                '-I'+str(a.binary.resolve().parent.parent/'include'),'-I'+str(case),
                *['-I'+str(p) for p in a.ccec_include],output,'-o',case/'output.o']
            result=run(argv,case/'ccec_object_only',env)
            assert result.returncode==0,result.stderr.decode()
            item['actual_ELF_UB']=ub_metadata(case/'output.o','sharedWordPairs',2176)
        rows.append(item)
    if a.dct_source:
        case=out/'original_dct';case.mkdir()
        source=a.dct_source.resolve();common=a.dct_common.resolve()
        original_files={str(p):sha(p) for root in (source.parent,common) for p in root.rglob('*') if p.is_file()}
        flags=['-x','cuda','-std=c++17','-I'+str(source.parent),'-I'+str(common),'-I'+str(a.cuda_path/'include/cccl'),'-fgpu-rdc']
        syntax=run([a.clang,'--cuda-host-only','-nocudalib','--cuda-path='+str(a.cuda_path),
            '-resource-dir='+str(a.resource_dir),'-Wno-unknown-cuda-version',*flags,'-fsyntax-only',source],case/'syntax')
        assert syntax.returncode==0,syntax.stderr.decode()
        argv=convert(source,case,flags);argv.insert(argv.index('-o'),'-I'+str(common))
        result=run(argv,case/'conversion');assert result.returncode==0,result.stderr.decode()
        assert (case/'output.cce').read_text().count('__ubuf__ ASCIFY_ALIGN(4)')==2
        assert all(sha(Path(path))==digest for path,digest in original_files.items())
        item=dict(name='original_dct',passed=True,source_syntax_passed=True,expected_min4_arrays=2,source_unchanged=True)
        if a.ccec:
            env=base_env.copy();env.update(ASCEND_HOME_PATH=str(a.cann),ASCEND_OPP_PATH=str(a.cann/'opp'),
                LD_LIBRARY_PATH=str(a.cann/'lib64')+':'+str(a.cann/'aarch64-linux/lib64'))
            argv=[a.ccec,'-x','dpp','--cce-aicore-arch=dav-c310-vec','-std=c++17','-O2','-funsigned-char','-c',
                '-I'+str(a.binary.resolve().parent.parent/'include'),'-I'+str(case),
                '-I'+str(source.parent),'-I'+str(common),*['-I'+str(p) for p in a.ccec_include],
                case/'output.cce','-o',case/'output.o']
            result=run(argv,case/'ccec_object_only',env);assert result.returncode==0,result.stderr.decode()
            item['actual_ELF_UB_DCT']=ub_metadata(case/'output.o','CUDAkernelShortDCT',2176)
            item['actual_ELF_UB_IDCT']=ub_metadata(case/'output.o','CUDAkernelShortIDCT',2176)
            assert all(sha(Path(path))==digest for path,digest in original_files.items())
        rows.append(item)
    assert all(sha(Path(path))==digest for path,digest in frozen.items())
    (out/'summary.json').write_text(json.dumps({'passed':True,'total':len(rows),'cases':rows,
        'tool_identity':frozen,'device_queries':0,'device_executed':False,'native_linked':False,
        'CCEC_metadata_checked':bool(a.ccec),'original_sample_successes':0},indent=2)+'\n')
    print('Shared word alignment:',len(rows),'syntax-first cases PASS; no NPU')


if __name__ == '__main__':
    main()
