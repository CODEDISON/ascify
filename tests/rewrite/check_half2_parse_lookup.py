#!/usr/bin/env python3
"""Check parser-only half2 ADL without changing unrelated template operators."""
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "tests/support/cuda/cuda_fp16.h"
CLANG = shlex.split(os.environ.get("ASCIFY_TEST_CLANGXX", "clang++"))
TEMPLATES = """
template<class T> T probe_add(T a, T b) { return a + b; }
template<class T> T probe_multiply(T a, T b) { return a * b; }
half2 probe_half2_add(half2 a, half2 b) { return a + b; }
half2 probe_half2_multiply(half2 a, half2 b) { return a * b; }
"""


def nodes(tree):
    yield tree
    for child in tree.get("inner", []):
        yield from nodes(child)


def parse(source, accepted=True):
    command = CLANG + ["-std=c++17", "-fsyntax-only", "-x", "c++", "-",
                       "-Xclang", "-ast-dump=json"]
    result = subprocess.run(command, input=source, text=True, capture_output=True)
    if (result.returncode == 0) != accepted:
        raise AssertionError(f"unexpected RC {result.returncode}: {result.stderr}")
    return json.loads(result.stdout) if accepted else None


def dependent_operators_are_builtin(tree):
    templates = {n.get("name"): n for n in nodes(tree)
                 if n.get("kind") == "FunctionTemplateDecl"}
    for name, opcode in [("probe_add", "+"), ("probe_multiply", "*")]:
        expressions = list(nodes(templates[name]))
        if any(n.get("kind") == "CXXOperatorCallExpr" for n in expressions):
            return False
        if not any(n.get("kind") == "BinaryOperator" and n.get("opcode") == opcode
                   for n in expressions):
            return False
    return True


def main():
    include = '#include "' + str(HEADER) + '"\n'
    positive = parse(include + TEMPLATES)
    assert dependent_operators_are_builtin(positive), (
        "half2 profile polluted ordinary lookup in unrelated dependent templates")
    functions = {n.get("name"): n for n in nodes(positive)
                 if n.get("kind") == "FunctionDecl" and
                 n.get("name", "").startswith("probe_half2_")}
    for name, operation in [("probe_half2_add", "operator+"),
                            ("probe_half2_multiply", "operator*")]:
        expressions = list(nodes(functions[name]))
        assert any(n.get("kind") == "CXXOperatorCallExpr" for n in expressions)
        assert any(n.get("referencedDecl", {}).get("name") == operation
                   for n in expressions), "concrete half2 ADL operator not resolved"
    # This legal redeclaration reproduces the original pollution: making the
    # same operators visible to ordinary lookup changes the template AST.
    polluted = parse(include + "half2 operator+(half2,half2);\n"
                     "half2 operator*(half2,half2);\n" + TEMPLATES)
    assert not dependent_operators_are_builtin(polluted), (
        "negative control did not detect ordinary-lookup pollution")
    # This input profile admits infix half2 use, not explicit global lookup of
    # the complete CUDA SDK's operator namespace.
    parse(include + "half2 rejected(half2 a,half2 b){return ::operator+(a,b);}",
          accepted=False)
    print("half2 parser lookup: dependent builtins, concrete ADL, pollution "
          "negative control and qualified-call boundary passed")


if __name__ == "__main__":
    main()
