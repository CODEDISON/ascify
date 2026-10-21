#!/usr/bin/env python3
"""Original-syntax-first packed record preservation and atomic refusals."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

if not __debug__:
    raise RuntimeError('Packed record tests require assertions')
ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT/'tests/rewrite/fixtures/nvidia_samples/Common'
BODY = '''typedef char Byte;
typedef unsigned short Word;
typedef unsigned int DWord;
typedef struct { Word tag; DWord length; DWord reserved; DWord offset; } FileRecord;
'''
OPEN = '#pragma pack(1)\n'
RESET = '#pragma pack()\n'
CONDITION = 'defined(TEST_BINARY_FORMAT_A) || defined(TEST_BINARY_FORMAT_B)'
PREFIX = '#if '+CONDITION+'\n#pragma pack(push)\n#endif\n\n'
FOOTER = '#if '+CONDITION+'\n#pragma pack(pop)\n#else\n#pragma pack()\n#endif\n'
HELPER = '#include <helper_cuda.h>\n'
TAIL = 'inline void checked(){ getLastCudaError("packed"); }\nint main(){checked();return sizeof(FileRecord)==14?0:1;}\n'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(path):
    return {str(p.relative_to(path)): sha(p) for p in path.rglob('*') if p.is_file()}


def main():
    p=argparse.ArgumentParser()
    for name in ('binary','cuda-path','resource-dir','evidence-dir','clang'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ccec',type=Path)
    p.add_argument('--helper-root',type=Path,default=HELPERS)
    p.add_argument('--target-arg',action='append',default=[])
    p.add_argument('--target-include',action='append',default=[])
    a=p.parse_args(); out=a.evidence_dir.resolve();out.mkdir()
    helpers=a.helper_root.resolve()
    fixture=inventory(helpers)
    cases=[
        ('plain', True, HELPER+OPEN+BODY+RESET+TAIL, None, []),
        ('conditional', True, HELPER+PREFIX+OPEN+BODY+FOOTER+TAIL, None, []),
        ('conditional_no_gap', True, HELPER+PREFIX.rstrip('\n')+'\n'+OPEN+BODY+FOOTER+TAIL, None, []),
        ('conditional_two_gaps', True, HELPER+PREFIX+'\n'+OPEN+BODY+FOOTER+TAIL, None, []),
        ('conditional_whitespace_crlf', True, HELPER+PREFIX.replace('\n','\r\n')+' \t\r\n'+OPEN+BODY+FOOTER+TAIL, None, []),
        ('context_once', True, HELPER+'#include "record.cuh"\nint main(){checked();return sizeof(FileRecord)==14?0:1;}\n',
            '#pragma once\n'+PREFIX+OPEN+BODY+FOOTER+'inline void checked(){getLastCudaError("context packed");}\n', []),
        ('direct_char', True, HELPER+OPEN+'struct FileRecord { char x; unsigned char y; signed char z; };\n'+RESET+TAIL, None, []),
        ('unmatched', False, HELPER+OPEN+BODY+TAIL, None, []),
        ('nested', False, HELPER+OPEN+BODY+OPEN+RESET+TAIL, None, []),
        ('push_pop', False, HELPER+'#pragma pack(push,1)\n#pragma pack(pop)\n'+OPEN+BODY+RESET+TAIL, None, []),
        ('unknown_alignment', False, HELPER+'#pragma pack(2)\n'+OPEN+BODY+RESET+TAIL, None, []),
        ('cross_file', False, HELPER+OPEN+BODY+'#include "record.cuh"\n'+TAIL, RESET, []),
        ('include_inside', False, HELPER+OPEN+BODY+'#include "record.cuh"\n'+RESET+TAIL, 'typedef int Extra;\n', []),
        ('macro_type', False, HELPER+'#define FIELD_TYPE unsigned int\n'+OPEN+BODY.replace('DWord length','FIELD_TYPE length')+RESET+TAIL, None, []),
        ('foreign_alias', False, HELPER+'typedef unsigned int External;\n'+OPEN+BODY.replace('DWord length','External length')+RESET+TAIL, None, []),
        ('float_field', False, HELPER+OPEN+BODY.replace('DWord length','float length')+RESET+TAIL, None, []),
        ('pointer_field', False, HELPER+OPEN+BODY.replace('DWord length','DWord *length')+RESET+TAIL, None, []),
        ('array_field', False, HELPER+OPEN+BODY.replace('DWord length','DWord length[2]')+RESET+TAIL, None, []),
        ('bitfield', False, HELPER+OPEN+BODY.replace('DWord length','DWord length:3')+RESET+TAIL, None, []),
        ('initializer', False, HELPER+OPEN+BODY.replace('DWord length','DWord length=1')+RESET+TAIL, None, []),
        ('ignored_attribute', False, HELPER+OPEN+BODY.replace('typedef struct {','typedef struct __attribute__((unknown_record_attribute)) {')+RESET+TAIL, None, []),
        ('dependent_template', False, HELPER+OPEN+'template<class T> struct Candidate { T x; };\ntypedef Candidate<unsigned> FileRecord;\n'+RESET+TAIL, None, []),
        ('mapped_legacy_name', False, HELPER+OPEN+BODY.replace('DWord length','DWord blockIdx')+RESET+TAIL, None, []),
        ('inactive_unowned_pack', False, HELPER+'#if 0\n#pragma pack(8)\n#endif\n'+OPEN+BODY+RESET+TAIL, None, []),
        ('footer_defined', False, HELPER+'#define TEST_BINARY_FORMAT_A 1\n'+PREFIX+OPEN+BODY+FOOTER+TAIL, None, []),
        ('prefix_condition_differs', False, HELPER+PREFIX.replace('TEST_BINARY_FORMAT_A','OTHER_FORMAT')+OPEN+BODY+FOOTER+TAIL, None, []),
        ('entry_pack_option', False, HELPER+OPEN+BODY+RESET+TAIL, None, ['-fpack-struct=2']),
        ('module_option', False, HELPER+OPEN+BODY+RESET+TAIL, None, ['-fmodules']),
        ('retain_excluded_standalone', False, OPEN+BODY+RESET+'int main(){return 0;}\n', None, []),
    ]
    # Standalone inputs prove the new generated syntax cannot be erased by
    # macros even without the ordinary CUDA compatibility-header conflict gate.
    for syntax in ('static_assert','__builtin_offsetof','sizeof','alignof','static_cast','decltype','nullptr'):
        macro = '#define '+syntax+('(... ) 1\n' if syntax!='nullptr' else ' 0\n')
        cases.append(('shadow_'+syntax,False,macro+OPEN+BODY+RESET+'int main(){return 0;}\n',None,[]))
    rows=[]
    def run(argv,directory):
        directory.mkdir();(directory/'argv.json').write_text(json.dumps([str(x) for x in argv])+'\n')
        r=subprocess.run([str(x) for x in argv],capture_output=True)
        (directory/'stdout.log').write_bytes(r.stdout);(directory/'stderr.log').write_bytes(r.stderr)
        (directory/'process.json').write_text(json.dumps({'returncode':r.returncode})+'\n');return r
    for name,success,text,header,extra in cases:
        case=out/name;case.mkdir();inputs=case/'input';inputs.mkdir()
        src=inputs/'input.cu';src.write_text(text)
        if header is not None: (inputs/'record.cuh').write_text(header)
        initial=inventory(inputs)
        syntax=run([a.clang,'-x','cuda','--cuda-host-only','-nocudalib','--cuda-path='+str(a.cuda_path),
            '-resource-dir='+str(a.resource_dir),'-std=c++17','-I'+str(helpers),'-I'+str(inputs),*extra,'-fsyntax-only',src],case/'source_syntax')
        assert syntax.returncode==0,(name,syntax.stderr.decode(errors='replace'))
        output=case/'published.cce';bundle=Path(str(output)+'.headers')
        if not success:
            bundle.mkdir();output.write_text('preserve old output\n')
            (bundle/'old.txt').write_text('preserve old bundle\n')
            prior=(sha(output),inventory(bundle))
        destinations=(output,) if success else (output,case/'fresh.cce')
        for index,destination in enumerate(destinations):
            command=[a.binary,src,'--cuda-path='+str(a.cuda_path),'--clang-resource-directory='+str(a.resource_dir),
                '--frontend-compat=ascify-admitted-v1','--target-policy=dav-c310-vec','--simt-math=fast',
                '--default-preprocessor','--local-headers-recursive','-I'+str(helpers),'-I'+str(inputs),'-o',destination,
                '--','-std=c++17','-I'+str(helpers),'-I'+str(inputs),*extra]
            if name=='retain_excluded_standalone':
                command.remove('--default-preprocessor')
            result=run(command,case/('convert_'+str(index)))
            assert inventory(inputs)==initial,(name,'original input mutation')
            if success:
                assert result.returncode==0,(name,result.stderr.decode(errors='replace'))
                generated=destination.read_bytes().decode()
                assert BODY in generated or name=='direct_char',name
                assert 'Ascify preserved packed' in generated and 'Ascify packed layout witness syntax' in generated,name
                if name.startswith('conditional') or name=='context_once':
                    prefix = (PREFIX.rstrip('\n')+'\n' if name=='conditional_no_gap' else
                              PREFIX+'\n' if name=='conditional_two_gaps' else
                              PREFIX.replace('\n','\r\n')+' \t\r\n' if name=='conditional_whitespace_crlf' else PREFIX)
                    assert prefix in generated and FOOTER in generated,name
                    assert 'requires the original inactive configuration' in generated,name
                if a.ccec:
                    obj=case/'target.o'
                    target=run([a.ccec,'-O2','-std=c++17','-x','dpp','--cce-aicore-arch=dav-c310-vec',
                        *a.target_arg,'-c',destination,'-o',obj,*sum((['-I',x] for x in a.target_include),[])],case/'target_compile')
                    assert target.returncode==0 and obj.read_bytes().startswith(b'\x7fELF'),(name,target.stderr.decode(errors='replace'))
            else:
                assert result.returncode!=0,(name,'unexpected admission')
                assert 'packed-record' in result.stderr.decode(errors='replace'),(name,'did not reach packed proof',result.stderr.decode(errors='replace'))
                assert (sha(output),inventory(bundle))==prior,(name,'old output changed')
                if destination!=output: assert not destination.exists() and not Path(str(destination)+'.headers').exists()
            assert not list(case.glob('.ascify-local-closure*')),(name,'staging leaked')
        rows.append({'name':name,'source_syntax_passed':True,'expected_acceptance':success,'passed':True})
    assert inventory(helpers)==fixture
    (out/'summary.json').write_text(json.dumps({'passed':True,'cases':rows,'total':len(rows),'device_executed':False},indent=2)+'\n')
    print('packed record preservation:',len(rows),'syntax-first cases PASS; no NPU')


if __name__=='__main__': main()
