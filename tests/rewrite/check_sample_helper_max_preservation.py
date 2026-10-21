#!/usr/bin/env python3
"""Physical MAX ownership, both include orders, CPU semantics and atomic refusal.

Every admitted case compiles the complete translated TU with real CCEC. The
host semantic checks compile only the MAX definition obtained by preprocessing
the original and translated TUs; they do not execute CUDA or NPU work.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess


HELPER = '#include <helper_cuda.h>\n'
FUNCTIONS = '#include <helper_functions.h>\n'
GT = '#ifndef MAX\n#define MAX(a, b) (a > b ? a : b)\n#endif\n'
LT = '#ifndef MAX\n#define MAX(a, b) ((a < b) ? b : a)\n#endif\n'
BODY = '''
int main() {
  cudaChannelFormatDesc format = cudaCreateChannelDesc<float>();
  cudaArray_t array = nullptr;
  checkCudaErrors(cudaMallocArray(&array, &format, 2, 2));
  return int(MAX(1.0f, 2.0f));
}
'''
SEMANTIC = r'''
#include <cstdint>
#include <cstdio>
#include <cstring>
static float value(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
static uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
static int lc=0,rc=0;
static float left(){++lc;return 3.0f;}
static float right(){++rc;return 3.0f;}
int main(){
  const float nan=value(0x7fc00001U), minus=value(0x80000000U), plus=0.0f;
  std::printf("%08x %08x %08x ",bits(MAX(nan,2.0f)),bits(MAX(2.0f,nan)),bits(MAX(minus,plus)));
  const float answer=MAX(left(),right());
  std::printf("%08x %d %d\n",bits(answer),lc,rc);
}
'''
EXPECTED = {
    'gt': b'40000000 7fc00001 00000000 40400000 1 2\n',
    'lt': b'7fc00001 40000000 80000000 40400000 2 1\n',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def run(argv, directory):
    directory.mkdir()
    argv = list(map(str, argv))
    (directory/'argv.json').write_text(json.dumps(argv)+'\n')
    result = subprocess.run(argv, capture_output=True, timeout=180)
    (directory/'stdout.log').write_bytes(result.stdout)
    (directory/'stderr.log').write_bytes(result.stderr)
    (directory/'process.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
    return result


def macro_line(data, expected_tables):
    # Pinned Clang -dM emits sorted unique macro names. CCEC's DPP driver
    # concatenates two such tables (host/device); validate each, not just a set.
    tables = [[]]
    previous = None
    for line in data.decode().splitlines():
        match = re.match(r'^#define ([A-Za-z_][A-Za-z_0-9]*)(?:\(|\s|$)', line)
        require(match is not None, 'only macro definitions in -dM output')
        name = match.group(1)
        if previous is not None and name <= previous:
            tables.append([])
        tables[-1].append(line)
        previous = name
    require(len(tables) == expected_tables and all(tables),
            'exact expected preprocessor table count')
    definitions = []
    for table in tables:
        lines = [line for line in table if re.match(r'^#define MAX\(', line)]
        require(len(lines) == 1, 'exactly one final MAX definition per table')
        definitions.append(lines[0])
    require(all(re.sub(r'\s+', '', line) == re.sub(r'\s+', '', definitions[0])
                for line in definitions), 'all target MAX definitions agree')
    return definitions[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary','clang','cuda-path','resource-dir','ccec','helper-root','evidence-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--target-include',action='append',default=[])
    parser.add_argument('--target-arg',action='append',default=[])
    a = parser.parse_args()
    out = a.evidence_dir.resolve()
    require(not out.exists(),'fresh evidence directory')
    require(a.target_include,'real installed/CANN target include closure')
    out.mkdir()
    helpers = a.helper_root.resolve()
    helpers_before = inventory(helpers)
    tool_paths = (a.binary,a.clang,a.ccec,Path(__file__).resolve())
    tool_identity = {str(p):sha(p) for p in tool_paths}
    header = '#ifndef OWNED_FALLBACK_HEADER\n#define OWNED_FALLBACK_HEADER\n'+LT+'#endif\n'
    # name, complete root, optional physical selected header, expected owner,
    # conversion extras, original syntax validity. No sample-name matching.
    cases = [
        ('helper_first_main',HELPER+LT+BODY,None,'gt',[],True),
        ('helper_first_header',HELPER+'#include "local.h"\n'+BODY,header,'gt',[],True),
        ('caller_first_main',LT+HELPER+FUNCTIONS+BODY,None,'lt',[],True),
        ('caller_first_header','#include "local.h"\n'+HELPER+FUNCTIONS+BODY,header,'lt',[],True),
        ('caller_parameter_names',LT.replace('(a, b)','(left, right)').replace('a < b','left < right').replace('? b : a','? right : left')+HELPER+BODY,None,'lt',[],True),
        ('caller_gt_grouped',GT.replace('(a > b ? a : b)','((a > b) ? a : b)')+HELPER+FUNCTIONS+BODY,None,'gt',[],True),
        ('helper_first_two_fallbacks',HELPER+LT+GT+BODY,None,'gt',[],True),
        ('header_direct_use',HELPER+'#include "local.h"\n'+BODY,header+'inline float local(float x,float y){return MAX(x,y);}\n','gt',[],True),
        ('host_cxx_helper_first','#include <cuda_runtime.h>\n'+HELPER+LT+BODY,None,'gt',['host_cxx'],True),
        ('host_cxx_caller_first','#include <cuda_runtime.h>\n'+LT+HELPER+FUNCTIONS+BODY,None,'lt',['host_cxx'],True),
        ('direct_call_arguments',HELPER+LT+'static float left(){return 3.0f;} static float right(){return 3.0f;}\n'+BODY.replace('MAX(1.0f, 2.0f)','MAX(left(), right())'),None,'gt',[],True),
        ('active_undef',HELPER+LT+'#undef MAX\n'+GT+BODY,None,None,[],True),
        ('prior_conflict_sticky','#define MAX(a,b) ((a)+(b))\n#undef MAX\n'+LT+HELPER+BODY,None,None,[],True),
        ('redefinition',LT+HELPER+'#define MAX(a,b) ((a)+(b))\n'+BODY,None,None,[],True),
        ('macro_stack',HELPER+LT+'#pragma push_macro("MAX")\n#pragma pop_macro("MAX")\n'+BODY,None,None,[],True),
        ('if_observer',HELPER+LT+'#if MAX(1,2)\nint observed;\n#endif\n'+BODY,None,None,[],True),
        ('ifdef_observer',HELPER+LT+'#ifdef MAX\nint observed;\n#endif\n'+BODY,None,None,[],True),
        ('defined_observer',HELPER+LT+'#if defined(MAX)\nint observed;\n#endif\n'+BODY,None,None,[],True),
        ('alias',HELPER+LT+'#define PICK MAX\n'+BODY.replace('MAX(1.0f','PICK(1.0f'),None,None,[],True),
        ('address',HELPER+LT+'float (MAX)(float,float); auto address=&MAX;\n'+BODY,None,None,[],True),
        ('wrapper_observer',HELPER+LT+'#define OBSERVE(x) ((void)sizeof(#x),(x))\n'+BODY.replace('int(MAX(1.0f, 2.0f))','int(OBSERVE(MAX(1.0f, 2.0f)))'),None,None,[],True),
        ('header_wrapper_observer',HELPER+'#include "local.h"\n'+BODY,header+'#define OBSERVE(x) ((void)sizeof(#x),(x))\ninline float local(){return OBSERVE(MAX(1.0f,2.0f));}\n',None,[],True),
        ('argument_macro',HELPER+LT+'#define VALUE 1.0f\n'+BODY.replace('MAX(1.0f','MAX(VALUE'),None,None,[],True),
        ('inactive_raw',HELPER+LT+'#if 0\nint unseen=MAX(1,2);\n#endif\n'+BODY,None,None,[],True),
        ('fallback_else',HELPER+LT.replace('#endif','#else\nint conditional_side_effect;\n#endif')+BODY,None,None,[],True),
        ('fallback_extra_body',HELPER+LT.replace('#endif','int extra_body;\n#endif')+BODY,None,None,[],True),
        ('fallback_comment',HELPER+LT.replace('#define MAX','#define /*not-owned*/ MAX')+BODY,None,None,[],True),
        ('fallback_splice',HELPER+LT.replace('a < b','a < \\\nb')+BODY,None,None,[],True),
        ('fallback_variadic',HELPER+LT.replace('(a, b)','(a, b, ...)')+BODY,None,None,[],True),
        ('fallback_extra_parameter',HELPER+LT.replace('(a, b)','(a, b, c)')+BODY,None,None,[],True),
        ('fallback_paste',HELPER+LT.replace('((a < b) ? b : a)','a ## b')+BODY,None,None,[],True),
        ('fallback_stringify',HELPER+LT.replace('((a < b) ? b : a)','#a')+BODY,None,None,[],True),
        ('fallback_object_spacing',HELPER+LT.replace('MAX(a, b)','MAX (a, b)')+BODY,None,None,[],True),
        ('fallback_elif',HELPER+LT.replace('#endif','#elif 0\nint other_branch;\n#endif')+BODY,None,None,[],True),
        ('angle_header',HELPER+'#include <local.h>\n'+BODY,header,None,[],True),
        ('unguarded_caller',LT.replace('#ifndef MAX\n','').replace('#endif\n','')+HELPER+BODY,None,None,[],True),
        ('preinclude_observer','#ifdef MAX\nint before_helper;\n#endif\n'+HELPER+LT+BODY,None,None,[],True),
        ('wrong_shape',LT.replace('((a < b) ? b : a)','((a)+(b))')+HELPER+BODY,None,None,[],True),
        ('mapped_parameter',LT.replace('(a, b)','(cudaError_t, b)').replace('a < b','cudaError_t < b').replace(': a',': cudaError_t')+HELPER+BODY,None,None,[],True),
        ('command_line_provider',HELPER+LT+BODY,None,None,['-DMAX(a,b)=((a)<(b)?(b):(a))'],True),
        ('helper_reserved_parameter',HELPER+LT.replace('(a, b)','(ASCIFY_ALIGN, b)').replace('a < b','ASCIFY_ALIGN < b').replace(': a',': ASCIFY_ALIGN')+BODY,None,None,[],True),
        ('caller_reserved_parameter',LT.replace('(a, b)','(ASCIFY_ALIGN, b)').replace('a < b','ASCIFY_ALIGN < b').replace(': a',': ASCIFY_ALIGN')+HELPER+BODY,None,None,[],True),
        ('retained_preprocessor',HELPER+LT+BODY,None,None,['retain'],True),
        ('altered_helper',HELPER+LT+BODY,None,None,['alter_helper'],True),
        ('host_macro_runtime','#define SDK_RUNTIME_HEADER <cuda_runtime.h>\n#include SDK_RUNTIME_HEADER\n'+HELPER+LT+BODY,None,None,['host_cxx'],True),
        ('host_transitive_runtime','#include <local.h>\n'+HELPER+LT+BODY,'#include <cuda_runtime.h>\n',None,['host_cxx'],True),
        ('host_conditional_runtime','#if 1\n#include <cuda_runtime.h>\n#endif\n'+HELPER+LT+BODY,None,None,['host_cxx'],True),
        ('host_shadow_runtime','#include <cuda_runtime.h>\n'+HELPER+LT+BODY,None,None,['host_cxx','shadow_runtime'],True),
        ('host_skipped_literal_then_macro','#define __CUDA_RUNTIME_H__\n#include <cuda_runtime.h>\n#undef __CUDA_RUNTIME_H__\n#define SDK_RUNTIME_HEADER <cuda_runtime.h>\n#include SDK_RUNTIME_HEADER\n'+HELPER+LT+BODY,None,None,['host_cxx'],True),
        ('host_texture_reflection','#include <cuda_runtime.h>\n'+HELPER+LT+BODY.replace('return int(MAX(1.0f, 2.0f));','return int(MAX(1.0f, 2.0f)) + int(sizeof(cudaChannelFormatDesc));'),None,None,['host_cxx'],True),
        ('host_texture_address','#include <cuda_runtime.h>\n'+HELPER+LT+'auto escaped_malloc_array = &cudaMallocArray;\n'+BODY,None,None,['host_cxx'],True),
    ]
    rows = []
    seed_output = seed_bundle = None
    for label,source,local,owner,extra,valid in cases:
        case = out/label
        case.mkdir()
        inputs = case/'input'
        inputs.mkdir()
        host_cxx = 'host_cxx' in extra
        root = inputs/('root.cpp' if host_cxx else 'root.cu')
        root.write_text(source)
        if local is not None:(inputs/'local.h').write_text(local)
        if 'shadow_runtime' in extra:(inputs/'cuda_runtime.h').write_text('#include_next <cuda_runtime.h>\n')
        chosen_helpers = helpers
        if 'alter_helper' in extra:
            chosen_helpers = inputs/'changed_helpers'
            shutil.copytree(helpers,chosen_helpers)
            p=chosen_helpers/'helper_cuda.h';p.write_bytes(p.read_bytes()+b'\n// altered identity\n')
        before = inventory(inputs)
        compiler_extra = [x for x in extra if x.startswith('-D')]
        language = ['-x','c++','-I'+str(a.cuda_path/'include')] if host_cxx else [
            '-x','cuda','--cuda-host-only','-nocudalib','--cuda-path='+str(a.cuda_path)]
        original = [a.clang,*language,'-resource-dir='+str(a.resource_dir),'-std=c++17',
                    '-I'+str(chosen_helpers),'-I'+str(inputs),*compiler_extra]
        syntax = run([*original,'-fsyntax-only',root],case/'source_syntax')
        require((syntax.returncode==0)==valid,(label,'original syntax',syntax.stderr.decode(errors='replace')))
        output=case/'published.cce';bundle=Path(str(output)+'.headers')
        if owner is None:
            require(seed_output is not None,'positive managed output seed missing')
            shutil.copy2(seed_output,output);shutil.copytree(seed_bundle,bundle)
            prior = sha(output),inventory(bundle)
        for index,destination in enumerate([output] if owner else [output,case/'fresh.cce']):
            argv = [a.binary,root,'--cuda-path='+str(a.cuda_path),'--clang-resource-directory='+str(a.resource_dir),
                    '--frontend-compat=ascify-admitted-v1','--target-policy=dav-c310-vec','--simt-math=fast',
                    '--local-headers-recursive','-I'+str(chosen_helpers),'-I'+str(inputs),'-o',destination]
            if 'retain' not in extra:argv.append('--default-preprocessor')
            if host_cxx:argv.append('--input-language=c++')
            argv += ['--','-std=c++17','-I'+str(chosen_helpers),'-I'+str(inputs),*compiler_extra]
            if host_cxx:argv += ['-x','c++','-I'+str(a.cuda_path/'include')]
            result=run(argv,case/('convert_'+str(index)))
            require(inventory(inputs)==before,(label,'source changed'))
            diagnostic=result.stderr.decode(errors='replace')
            if owner:
                require(result.returncode==0,(label,diagnostic))
                body=destination.read_text()
                require('#include <helper_cuda.h>' not in body and 'ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS' in body,(label,'helper transaction'))
                target=[a.ccec,'-O2','-std=c++17','-x','dpp','--cce-aicore-arch=dav-c310-vec',*a.target_arg,
                        *sum((['-I',x] for x in [*a.target_include,str(chosen_helpers)]),[])]
                obj=case/'target.o'
                compiled=run([*target,'-c',destination,'-o',obj],case/'target_compile')
                require(compiled.returncode==0 and obj.read_bytes()[:4]==b'\x7fELF' and struct.unpack_from('<H',obj.read_bytes(),18)[0]==183,(label,'real CCEC target',compiled.stderr.decode(errors='replace')))
                old_pp=run([*original,'-E','-dM',root],case/'original_macros')
                new_pp=run([*target,'-E','-dM',destination],case/'translated_macros')
                require(old_pp.returncode==new_pp.returncode==0,(label,'macro preprocess'))
                original_max,translated_max=macro_line(old_pp.stdout,1),macro_line(new_pp.stdout,2)
                require(re.sub(r'\s+','',original_max)==re.sub(r'\s+','',translated_max),(label,'final macro token contract'))
                for kind,line in [('original',original_max),('translated',translated_max)]:
                    host=case/(kind+'_semantics.cpp');host.write_text(line+'\n'+SEMANTIC)
                    exe=case/(kind+'_semantics')
                    built=run([a.clang,'-x','c++','-std=c++17','-O0',host,'-o',exe],case/(kind+'_host_compile'))
                    require(built.returncode==0,(label,'host semantic compile'))
                    observed=run([exe],case/(kind+'_host_run'))
                    require(observed.returncode==0 and observed.stdout==EXPECTED[owner],(label,kind,'NaN/tie/evaluation count semantics',observed.stdout))
            else:
                require(result.returncode!=0 and ('error:' in diagnostic or 'ERROR' in diagnostic),(label,'unexpected admission',diagnostic))
                require('fatal error:' not in diagnostic,(label,'unrelated missing dependency'))
                require((sha(output),inventory(bundle))==prior,(label,'old output/bundle changed'))
                if destination!=output:require(not destination.exists() and not Path(str(destination)+'.headers').exists(),(label,'fresh publication'))
            require(not list(case.glob('.ascify-local-closure*')),(label,'staging leaked'))
        if seed_output is None and bundle.is_dir():seed_output,seed_bundle=output,bundle
        rows.append({'name':label,'source_syntax_valid':valid,'expected_acceptance':owner is not None,
                     'active_macro_owner':owner,'passed':True,'input_sha256':before})
    require(inventory(helpers)==helpers_before,'frozen fixture modified')
    require(all(sha(Path(p))==h for p,h in tool_identity.items()),'tools changed')
    (out/'summary.json').write_text(json.dumps({'passed':True,'cases':rows,'total':len(rows),
        'tools':tool_identity,'helpers':helpers_before,'device_executed':False,'original_sample_successes':0},indent=2)+'\n')
    print('MAX macro preservation:',len(rows),'cases PASS; no NPU')


if __name__=='__main__':
    main()
