#!/usr/bin/env python3
"""Check host math values and 62 converter identity/domain cases."""
import os
import hashlib
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "frontend_compat/ascify-admitted-v1/host_math.h"


def run(args, *, accepted=True, source=None):
    result = subprocess.run(args, input=source, text=True, capture_output=True)
    if (result.returncode == 0) != accepted:
        raise AssertionError(
            f"unexpected RC {result.returncode}: {shlex.join(map(str, args))}\n"
            f"{result.stdout}\n{result.stderr}")
    return result


def main():
    cxx = shlex.split(os.environ.get("CXX", "c++"))
    with tempfile.TemporaryDirectory(prefix="ascify-host-math-") as temp:
        work = Path(temp)
        source = r'''
#include <cmath>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <utility>
#include "HEADER"
template<class A, class B, class = void> struct admitted : std::false_type {};
template<class A, class B> struct admitted<A, B,
    std::void_t<decltype(::max(std::declval<A>(), std::declval<B>()))>>
    : std::true_type {};
static_assert(admitted<float,float>::value);
static_assert(!admitted<double,double>::value);
static_assert(!admitted<float,double>::value);
static_assert(admitted<int,int>::value);
static_assert(!admitted<unsigned,unsigned>::value);
static_assert(!admitted<int,unsigned>::value);
static_assert(!admitted<float,int>::value);
template<class A, class B, class = void> struct min_admitted : std::false_type {};
template<class A, class B> struct min_admitted<A, B,
    std::void_t<decltype(::min(std::declval<A>(), std::declval<B>()))>>
    : std::true_type {};
static_assert(min_admitted<int,int>::value);
static_assert(!min_admitted<unsigned,unsigned>::value);
static_assert(!min_admitted<float,float>::value);
static_assert(!min_admitted<double,double>::value);
static_assert(min_admitted<int,unsigned>::value);
static_assert(min_admitted<unsigned,int>::value);
static_assert(std::is_same<decltype(::min(0,0u)), unsigned int>::value);
static_assert(std::is_same<decltype(::min(0u,0)), unsigned int>::value);
static_assert(!min_admitted<long,unsigned>::value);
static_assert(!min_admitted<unsigned,long>::value);
static_assert(!min_admitted<short,unsigned>::value);
static_assert(!min_admitted<unsigned,short>::value);
static_assert(!min_admitted<int,unsigned long>::value);
static_assert(!min_admitted<unsigned long,int>::value);
static_assert(!min_admitted<bool,unsigned>::value);
static_assert(!min_admitted<unsigned,bool>::value);
int calls = 0;
int once_int(int value) { ++calls; return value; }
unsigned once_uint(unsigned value) { ++calls; return value; }
float once(float value) { ++calls; return value; }
int main() {
  float values[] = {0.0f, -0.0f, 1.0f, -2.0f,
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::quiet_NaN()};
  for (float a : values) for (float b : values) {
    const float actual = ::max(a,b), expected = std::fmax(a,b);
    if (std::isnan(expected)) { if (!std::isnan(actual)) return 1; }
    else if (actual != expected) return 2;
    // C/C++ fmax does not promise a sign ordering for equal signed zeros.
  }
  if (::max(once(1.0f), once(2.0f)) != 2.0f || calls != 2) return 3;
  const int integers[] = {0, 1, -1, std::numeric_limits<int>::min(),
                         std::numeric_limits<int>::max()};
  for (int a : integers) for (int b : integers)
    if (::min(a,b) != (a < b ? a : b) || ::max(a,b) != (a > b ? a : b)) return 4;
  calls = 0;
  if (::min(once_int(1), once_int(2)) != 1 || calls != 2) return 5;
  calls = 0;
  if (::max(once_int(1), once_int(2)) != 2 || calls != 2) return 6;
  const unsigned unsigneds[] = {0u, 1u,
      static_cast<unsigned>(std::numeric_limits<int>::max()),
      static_cast<unsigned>(std::numeric_limits<int>::max()) + 1u,
      std::numeric_limits<unsigned>::max()};
  for (int a : integers) for (unsigned b : unsigneds) {
    const unsigned ua = static_cast<unsigned>(a);
    const unsigned expected = ua < b ? ua : b;
    if (::min(a,b) != expected || ::min(b,a) != expected) return 7;
  }
  calls = 0;
  if (::min(once_int(-1), once_uint(1u)) != 1u || calls != 2) return 8;
  calls = 0;
  if (::min(once_uint(std::numeric_limits<unsigned>::max()), once_int(-1)) !=
      std::numeric_limits<unsigned>::max() || calls != 2) return 9;
}
'''.replace("HEADER", str(HEADER))
        executable = work / "host-check"
        run(cxx + ["-std=c++17", "-x", "c++", "-", "-o", str(executable)],
            source=source)
        run([str(executable)])
        user = ('float max(float a,float b) { return a-b; }\n'
                'int min(int a,int b) { return a+b; }\n'
                'unsigned min(int a,unsigned b) { return b+17u; }\n'
                'unsigned min(unsigned a,int b) { return a+19u; }\n'
                f'#include "{HEADER}"\n'
                'int main() { return max(5.f,2.f)==3.f && min(5,2)==7 && '
                'min(-1,2u)==19u && min(2u,-1)==21u ? 0 : 1; }\n')
        run(cxx + ["-std=c++17", "-x", "c++", "-", "-o", str(executable)],
            source=user)
        run([str(executable)])
        # Compile the actual frozen target helper, with the repository's host
        # SDK stubs. This checks its finite signature and conversions without
        # using the parser shim as an oracle for emitted target behavior.
        target = r'''
#define __aicore__
#include <ascify/ascify_cuda_compat.hpp>
#include <limits>
#include <type_traits>
static_assert(std::is_same<decltype(&ascify::cuda_host_min_unsigned),
    unsigned int (*)(unsigned int,unsigned int)>::value);
static_assert(std::is_same<decltype(ascify::cuda_host_min_unsigned(0,0u)),
    unsigned int>::value);
static_assert(std::is_same<decltype(ascify::cuda_host_min_unsigned(0u,0)),
    unsigned int>::value);
int si_calls=0,ui_calls=0;
int once_int(int x){++si_calls;return x;}
unsigned once_uint(unsigned x){++ui_calls;return x;}
int main(){
  const int si[]={std::numeric_limits<int>::min(),-1,0,1,std::numeric_limits<int>::max()};
  const unsigned ui[]={0u,1u,static_cast<unsigned>(std::numeric_limits<int>::max()),
      static_cast<unsigned>(std::numeric_limits<int>::max())+1u,
      std::numeric_limits<unsigned>::max()};
  for(int a:si)for(unsigned b:ui){
    const unsigned converted=static_cast<unsigned>(a);
    const unsigned expected=converted<b?converted:b;
    if(ascify::cuda_host_min_unsigned(a,b)!=expected ||
       ascify::cuda_host_min_unsigned(b,a)!=expected)return 1;
  }
  if(ascify::cuda_host_min_unsigned(once_int(-1),once_uint(1u))!=1u ||
     si_calls!=1 || ui_calls!=1)return 2;
  si_calls=ui_calls=0;
  if(ascify::cuda_host_min_unsigned(once_uint(std::numeric_limits<unsigned>::max()),once_int(-1))!=
      std::numeric_limits<unsigned>::max() || si_calls!=1 || ui_calls!=1)return 3;
}
'''
        run(cxx + ["-std=c++17", "-I" + str(ROOT / "tests/rewrite/stubs"),
                   "-I" + str(ROOT / "include"), "-x", "c++", "-", "-o", str(executable)],
            source=target)
        run([str(executable)])
        print("host math: exact argument domains, 49 floating/50 signed/50 mixed integer pairs, "
              "single evaluation and user overload passed; actual target helper: 50 mixed pairs passed")

        binary = os.environ.get("ASCIFY_BINARY")
        if not binary:
            print("real converter checks not run: ASCIFY_BINARY not supplied")
            return
        cuda = os.environ["ASCIFY_CUDA_PATH"]
        resource = os.environ["ASCIFY_CLANG_RESOURCE_DIRECTORY"]
        cases = {
            "int_min": ("int choose(int a,int b) { return min(a,b); }", True),
            "min_initializer": ("int choose(int a,int b) { "
                                "int result = ::min(a,b); return result; }", True),
            "min_custom": ("int min(int a,int b) { return a+b; }\n"
                           "int choose(int a,int b) { return min(a,b); }", True),
            "min_unsigned": ("unsigned choose(unsigned a,unsigned b) { return min(a,b); }", False),
            "min_mixed": ("unsigned choose(int a,unsigned b) { return min(a,b); }", True),
            "min_mixed_reverse": ("unsigned choose(unsigned a,int b) { return min(a,b); }", True),
            "min_mixed_qualified": ("unsigned choose(int a,unsigned b) { return ::min(a,b); }", True),
            "min_mixed_custom": ("unsigned min(int a,unsigned b) { return b+17u; }\n"
                                 "unsigned choose(int a,unsigned b) { return min(a,b); }", True),
            "min_mixed_custom_reverse": ("unsigned min(unsigned a,int b) { return a+19u; }\n"
                                         "unsigned choose(unsigned a,int b) { return min(a,b); }", True),
            "min_mixed_side_effects": ("int once_int(); unsigned once_uint();\n"
                                       "unsigned choose() { return min(once_int(),once_uint()); }", True),
            "min_mixed_negative": ("unsigned choose() { return min(-1,1u); }", True),
            "min_mixed_long": ("unsigned choose(long a,unsigned b) { return min(a,b); }", False),
            "min_mixed_short": ("unsigned choose(short a,unsigned b) { return min(a,b); }", False),
            "min_mixed_macro": ("#define SELECT(a,b) min(a,b)\n"
                                "unsigned choose(int a,unsigned b) { return SELECT(a,b); }", False),
            "min_mixed_builtin_macro": ("#define __builtin_elementwise_min(a,b) ((a)+(b))\n"
                                        "unsigned choose(unsigned a,int b) { return min(a,b); }", False),
            "min_mixed_output_macro": ("#define cuda_host_min_unsigned(a,b) ((a)+(b))\n"
                                       "unsigned choose(int a,unsigned b) { return min(a,b); }", False),
            "min_mixed_namespace_conflict": ("namespace ascify { int caller_owned; }\n"
                                             "unsigned choose(int a,unsigned b) { return min(a,b); }", False),
            "min_mixed_address": ("unsigned choose(int a,unsigned b) { "
                                  "auto p = &min<int,unsigned>; return p(a,b); }", False),
            "min_mixed_template": ("template<class T> unsigned choose(T a,unsigned b) { return min(a,b); }\n"
                                   "unsigned instantiate() { return choose(-1,1u); }", False),
            "min_mixed_namespace_using": ("namespace named { using ::min; }\n"
                                          "unsigned choose(int a,unsigned b) { return named::min(a,b); }", False),
            "min_mixed_global_initializer": ("unsigned result = min(-1,1u);", False),
            "min_mixed_lambda": ("unsigned choose(int a,unsigned b) { "
                                 "return [=] { return min(a,b); }(); }", False),
            "min_float": ("float choose(float a,float b) { return min(a,b); }", False),
            "min_macro": ("#define SELECT(a,b) min(a,b)\n"
                          "int choose(int a,int b) { return SELECT(a,b); }", False),
            "min_builtin_macro": ("#define __builtin_elementwise_min(a,b) ((a)+(b))\n"
                                  "int choose(int a,int b) { return min(a,b); }", False),
            "float": ("float choose(float a,float b) { return max(a,b); }", True),
            "qualified": ("float choose(float a,float b) { return ::max(a,b); }", True),
            "initializer": ("float choose(float a,float b) { "
                            "float result = max(a,b); return result; }", True),
            "namespace_builtin": ("namespace own { float __builtin_fmaxf(float a,float b) { return a-b; } "
                                  "float choose(float a,float b) { return max(a,b); } }", True),
            "member_builtin": ("struct Own { float __builtin_fmaxf(float a,float b) { return a-b; } "
                               "float choose(float a,float b) { return max(a,b); } };", True),
            "min_member_builtin": ("struct Own { int __builtin_elementwise_min(int a,int b) { return a+b; } "
                                   "int choose(int a,int b) { return min(a,b); } };", True),
            "custom": ("float max(float a,float b) { return a-b; }\n"
                       "float choose(float a,float b) { return max(a,b); }", True),
            "double": ("double choose(double a,double b) { return max(a,b); }", False),
            "mixed": ("float choose(float a,int b) { return max(a,b); }", False),
            "macro": ("#define SELECT(a,b) max(a,b)\n"
                      "float choose(float a,float b) { return SELECT(a,b); }", False),
            "address": ("float choose(float a,float b) { "
                        "auto p = &max<float,float>; return p(a,b); }", False),
            "template": ("template<class T> T choose(T a,T b) { return max(a,b); }\n"
                         "float instantiate() { return choose(1.f,2.f); }", False),
            "namespace_using": ("namespace named { using ::max; }\n"
                                "float choose(float a,float b) { return named::max(a,b); }", False),
            "builtin_macro": ("#define __builtin_fmaxf(a,b) ((a)+(b))\n"
                              "float choose(float a,float b) { return max(a,b); }", False),
            "global_initializer": ("float result = max(1.f,2.f);", False),
            "lambda": ("float choose(float a,float b) { "
                       "return [=] { return max(a,b); }(); }", False),
        }
        cases.update({
            "max_int": ("int choose(int a,int b) { return max(a,b); }", True),
            "max_int_initializer": ("int choose(int a,int b) { int result = ::max(a,b); return result; }", True),
            "max_int_custom": ("int max(int a,int b) { return a-b; }\nint choose(int a,int b) { return max(a,b); }", True),
            "max_int_member_builtin": ("struct Own { int __builtin_elementwise_max(int a,int b) { return a-b; } int choose(int a,int b) { return max(a,b); } };", True),
            "max_int_unsigned": ("unsigned choose(unsigned a,unsigned b) { return max(a,b); }", False),
            "max_int_mixed": ("unsigned choose(int a,unsigned b) { return max(a,b); }", False),
            "max_int_macro": ("#define SELECT(a,b) max(a,b)\nint choose(int a,int b) { return SELECT(a,b); }", False),
            "max_int_builtin_macro": ("#define __builtin_elementwise_max(a,b) ((a)+(b))\nint choose(int a,int b) { return max(a,b); }", False),
            "max_int_address": ("int choose(int a,int b) { auto p = &max<int,int>; return p(a,b); }", False),
            "max_int_template": ("template<class T> T choose(T a,T b) { return max(a,b); }\nint instantiate() { return choose(1,2); }", False),
            "max_int_namespace_using": ("namespace named { using ::max; }\nint choose(int a,int b) { return named::max(a,b); }", False),
            "max_int_global_initializer": ("int result = max(1,2);", False),
            "max_int_lambda": ("int choose(int a,int b) { return [=] { return max(a,b); }(); }", False),
        })
        for name, (text, accepted) in cases.items():
            input_path, output = work / (name + ".cu"), work / (name + ".cce")
            input_path.write_text(text + "\n")
            command = [binary, str(input_path), "--frontend-compat=ascify-admitted-v1",
                       f"--cuda-path={cuda}", f"--clang-resource-directory={resource}",
                       "-o", str(output), "--", "-std=c++17"]
            result = run(command, accepted=accepted)
            if accepted:
                generated = output.read_text()
                expected_builtin = ("ascify::cuda_host_min_unsigned" if name.startswith("min_mixed")
                                    else "__builtin_elementwise_min" if name == "int_min" or name.startswith("min_")
                                    else "__builtin_elementwise_max" if name.startswith("max_int")
                                    else "__builtin_fmaxf")
                custom = {"custom", "min_custom", "max_int_custom",
                          "min_mixed_custom", "min_mixed_custom_reverse"}
                assert (expected_builtin in generated) == (name not in custom), generated
                assert "host_math.h" not in generated and "host_float_max_enabled" not in generated
                assert "host_mixed_uint_min_enabled" not in generated
                if name not in custom:
                    arguments = ("(once_int(),once_uint())" if name == "min_mixed_side_effects"
                                 else "(-1,1u)" if name == "min_mixed_negative" else "(a,b)")
                    assert "::" + expected_builtin + arguments in generated, generated
                    assert "::::" not in generated, generated
                    if name.startswith("min_mixed"):
                        assert "#include <ascify/ascify_cuda_compat.hpp>" in generated, generated
                        assert "__builtin_elementwise_min" not in generated, generated
            elif name in {"macro", "address", "template", "namespace_using", "builtin_macro",
                          "global_initializer", "lambda", "min_macro", "min_builtin_macro",
                          "max_int_macro", "max_int_builtin_macro", "max_int_address",
                          "max_int_template", "max_int_namespace_using",
                          "max_int_global_initializer", "max_int_lambda",
                          "min_mixed_macro", "min_mixed_builtin_macro", "min_mixed_address",
                          "min_mixed_template", "min_mixed_namespace_using",
                          "min_mixed_global_initializer", "min_mixed_lambda"}:
                which = "min" if name.startswith("min_") else "max"
                assert f"Ascify frontend host {which} requires" in result.stderr, result.stderr
            if not accepted:
                assert not output.exists(), "failed conversion published output"
                sentinel = "EXISTING_UNCHANGED_" + name + "\n"
                output.write_text(sentinel)
                run(command, accepted=False)
                assert output.read_text() == sentinel, "failed conversion replaced existing output"
            print(f"real converter: {name} expected {'success' if accepted else 'rejection'} passed")
        # An SDK-shaped header outside the authenticated product profile must
        # keep its own function identity, even for the exact mixed signatures.
        foreign = work / "foreign-sdk" / "crt"
        foreign.mkdir(parents=True)
        foreign_header = foreign / "math_functions.hpp"
        foreign_text = ("#pragma once\n"
                        "inline unsigned min(int a,unsigned b){return b+17u;}\n"
                        "inline unsigned min(unsigned a,int b){return a+19u;}\n")
        foreign_header.write_text(foreign_text)
        input_path = work / "nonfrozen-sdk.cu"
        input_path.write_text('#include "foreign-sdk/crt/math_functions.hpp"\n'
                              'unsigned choose(int a,unsigned b){return min(a,b)+min(b,a);}\n')
        output = work / "nonfrozen-sdk.cce"
        run([binary, str(input_path), "--frontend-compat=ascify-admitted-v1",
             f"--cuda-path={cuda}", f"--clang-resource-directory={resource}",
             "-o", str(output), "--", "-std=c++17"])
        generated = output.read_text()
        assert "min(a,b)+min(b,a)" in generated and "__builtin_elementwise_min" not in generated
        assert foreign_header.read_text() == foreign_text
        print("real converter: nonfrozen SDK-shaped declarations kept their own identities")
        # CPU parser/byte-identity unit environment only: this is not a real
        # SDK, target runtime, or complete native dependency acceptance test.
        # Authenticate the exact topheader while isolating its dependencies.
        native = work / "native-input" / "include"
        isolated = work / "identity-unit" / "include"
        shutil.copytree(ROOT / "include/ascify", native / "ascify")
        shutil.copytree(ROOT / "include/ascify", isolated / "ascify")
        native_compat = native / "ascify/ascify_cuda_compat.hpp"
        compat = isolated / "ascify/ascify_cuda_compat.hpp"
        frozen = compat.read_bytes()
        assert len(frozen) == 132328
        assert hashlib.sha256(frozen).hexdigest() == (
            "ba71595ff2049892202a81d918f521a6f619608671a3ebe90181e35fa5a1f011")
        # The symbol-registration macro is itself a protected output spelling.
        # Do not change the product trust boundary to admit that transitive
        # macro as caller input. Only this parser fixture replaces that leaf.
        symbol_stub = isolated / "ascify/symbol_compat.hpp"
        symbol_stub.write_text("#pragma once\n// CPU identity-only symbol dependency stub.\n")
        platform = work / "identity-platform-system-stub"
        platform.mkdir()
        (platform / "ascify_test_platform.hpp").write_text(
            "#pragma once\n// CPU parser platform qualifier stub, not a CANN header.\n"
            "#define __aicore__\n")
        # Mark parser stubs as system dependencies from first entry. This is
        # explicit fixture setup, not a product or installation trust change.
        # The caller negative below retains the original ordinary -I tree.
        immutable = {p: p.read_bytes() for directory in (native, isolated, platform)
                     for p in directory.rglob("*") if p.is_file() and p != compat}
        providers = {
            "frozen": (frozen, True),
            "same_name_caller": (b"#pragma once\nnamespace ascify { int caller_owned; }\n", False),
            "one_byte_changed": (frozen.replace(b"Internal host target", b"internal host target", 1), False),
            # Preserve the old complete native dependency + caller qualifier
            # input verbatim, with its correct expected result: refusal.
            "caller_aicore_native_tree": (frozen, False),
        }
        assert len(providers["one_byte_changed"][0]) == len(frozen)
        assert providers["one_byte_changed"][0] != frozen
        for provider, (content, accepted) in providers.items():
            compat.write_bytes(content)
            caller = provider == "caller_aicore_native_tree"
            prefix = ('#define __aicore__\n' if caller else
                      '#include <ascify_test_platform.hpp>\n')
            source_path = work / ("provider-" + provider + ".cpp")
            source_path.write_text(prefix + '#include <ascify/ascify_cuda_compat.hpp>\n'
                                   'unsigned choose(int a,unsigned b){return min(a,b);}\n')
            include_args = (["-I" + str(native), "-I" + str(ROOT / "tests/rewrite/stubs")]
                            if caller else
                            ["-isystem", str(platform), "-isystem", str(isolated),
                             "-isystem", str(ROOT / "tests/rewrite/stubs")])
            for mode in (["fresh"] if accepted else ["fresh", "existing"]):
                output = work / ("provider-" + provider + "-" + mode + ".cpp")
                sentinel = "EXISTING_UNCHANGED_" + provider + "\n"
                if mode == "existing":
                    output.write_text(sentinel)
                result = run([binary, str(source_path), "--frontend-compat=ascify-admitted-v1",
                              f"--cuda-path={cuda}", f"--clang-resource-directory={resource}",
                              "-o", str(output), "--", "-x", "c++", "-std=c++17",
                              *include_args], accepted=accepted)
                if accepted:
                    generated = output.read_text()
                    assert "::ascify::cuda_host_min_unsigned(a,b)" in generated, generated
                    assert generated.count("#include <ascify/ascify_cuda_compat.hpp>") == 1, generated
                    assert "unrecognized header" not in result.stderr, result.stderr
                    assert "untrusted input macro" not in result.stderr, result.stderr
                else:
                    assert "Ascify cannot publish CUDA compatibility output" in result.stderr, result.stderr
                    if caller:
                        assert "untrusted input macro '__aicore__'" in result.stderr, result.stderr
                        assert "reserved output macro 'ASCIFY_REGISTER_DEVICE_SYMBOL'" in result.stderr, result.stderr
                    else:
                        assert "include spelling resolves to an unrecognized header" in result.stderr, result.stderr
                    assert (not output.exists()) if mode == "fresh" else output.read_text() == sentinel
                assert compat.read_bytes() == content
                assert native_compat.read_bytes() == frozen
                assert all(path.read_bytes() == data for path, data in immutable.items())
            print(f"real converter: frozen compat provider {provider} passed")
        compat.write_bytes(frozen)
        # The default profile must not silently acquire the new host overload.
        for name in ("float", "min_mixed", "min_mixed_reverse"):
            input_path = work / (name + ".cu")
            run([binary, str(input_path), f"--cuda-path={cuda}",
                 f"--clang-resource-directory={resource}", "-o", str(work / ("default-" + name + ".cce")),
                 "--", "-std=c++17"], accepted=False)
        print("default frontend remains unchanged")


if __name__ == "__main__":
    main()
