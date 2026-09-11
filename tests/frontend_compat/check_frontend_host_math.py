#!/usr/bin/env python3
"""Check the opt-in host float max domain, then the real converter if supplied."""
import os
from pathlib import Path
import shlex
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
    with tempfile.TemporaryDirectory(prefix="ascify-host-float-max-") as temp:
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
static_assert(!admitted<int,int>::value);
static_assert(!admitted<float,int>::value);
int calls = 0;
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
}
'''.replace("HEADER", str(HEADER))
        executable = work / "host-check"
        run(cxx + ["-std=c++17", "-x", "c++", "-", "-o", str(executable)],
            source=source)
        run([str(executable)])
        user = ('float max(float a,float b) { return a-b; }\n'
                f'#include "{HEADER}"\n'
                'int main() { return max(5.f,2.f)==3.f ? 0 : 1; }\n')
        run(cxx + ["-std=c++17", "-x", "c++", "-", "-o", str(executable)],
            source=user)
        run([str(executable)])
        print("host float max: exact argument domain, 49 floating pairs, "
              "single evaluation and user overload passed")

        binary = os.environ.get("ASCIFY_BINARY")
        if not binary:
            print("real converter checks not run: ASCIFY_BINARY not supplied")
            return
        cuda = os.environ["ASCIFY_CUDA_PATH"]
        resource = os.environ["ASCIFY_CLANG_RESOURCE_DIRECTORY"]
        cases = {
            "float": ("float choose(float a,float b) { return max(a,b); }", True),
            "qualified": ("float choose(float a,float b) { return ::max(a,b); }", True),
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
        }
        for name, (text, accepted) in cases.items():
            input_path, output = work / (name + ".cu"), work / (name + ".cce")
            input_path.write_text(text + "\n")
            command = [binary, str(input_path), "--frontend-compat=ascify-admitted-v1",
                       f"--cuda-path={cuda}", f"--clang-resource-directory={resource}",
                       "-o", str(output), "--", "-std=c++17"]
            result = run(command, accepted=accepted)
            if accepted:
                generated = output.read_text()
                assert ("__builtin_fmaxf" in generated) == (name != "custom"), generated
                assert "host_math.h" not in generated and "host_float_max_enabled" not in generated
            elif name in {"macro", "address", "template", "namespace_using", "builtin_macro"}:
                assert "Ascify frontend host max requires" in result.stderr, result.stderr
            if not accepted:
                assert not output.exists(), "failed conversion published output"
            print(f"real converter: {name} expected {'success' if accepted else 'rejection'} passed")
        # The default profile must not silently acquire the new host overload.
        input_path = work / "float.cu"
        run([binary, str(input_path), f"--cuda-path={cuda}",
             f"--clang-resource-directory={resource}", "-o", str(work / "default.cce"),
             "--", "-std=c++17"], accepted=False)
        print("default frontend remains unchanged")


if __name__ == "__main__":
    main()
