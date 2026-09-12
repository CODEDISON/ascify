#!/usr/bin/env python3
"""Check explicit template status domains; native mode uses the real frontend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/rewrite/fixtures/sample_helper_template_domain.cu"
HELPERS = ROOT / "tests/rewrite/fixtures/nvidia_samples/Common"
MESSAGE = "Ascify: dependent helper status is only proven for the emitted template type domain"
PENDING = "proved dependent template status domain for 'template_memory'; guard pending helper transaction: "
PRAGMA_BOUNDARY = "unresolved trusted-system macro-generated pragma keeps all helper edits"


def run(command, success=True):
    result = subprocess.run(command, text=True, capture_output=True)
    assert (result.returncode == 0) == success, (
        command, result.returncode, result.stdout, result.stderr)
    return result


def host_guard_contract(work):
    # Exercise the emitted language boundary, including an alias, a class with
    # CUDA-like ADL overloads, and an unobserved fundamental type. This is not
    # a substitute for the native transformation tests below.
    header = work / "guard.hpp"
    header.write_text('template<class T> void memory(T*) {\n'
                      'static_assert(__is_same(T, float) || __is_same(T, int), "' +
                      MESSAGE + '");\n}\n')
    for name, argument, good in [
        ("int", "int", True), ("float_alias", "Alias", True),
        ("char", "char", False), ("cv_float", "const float", False),
        ("class_adl", "project::Value", False),
    ]:
        source = work / (name + ".cpp")
        source.write_text('#include "guard.hpp"\nusing Alias = float;\n'
                          'namespace project { struct Value {}; '
                          'int cudaFree(Value*) { return 0; } }\n'
                          'int main() { memory(static_cast<' + argument +
                          '*>(nullptr)); }\n')
        result = run([os.environ.get("CXX", "c++"), "-std=c++17",
                      "-fsyntax-only", str(source)], good)
        if not good:
            assert MESSAGE in result.stderr, result.stderr
    print("host template guard: 2 admitted types/aliases, 3 rejected domains")

    # Establish the actual AST precondition: the pattern's outer check is
    # unresolved, whereas each scalar specialization resolves both calls.
    source = work / "dependent-lookup.cpp"
    source.write_text('enum Status { ok }; Status cudaFree(void*);\n'
                      'template<class E> void check(E,const char*,const char*,int);\n'
                      '#define checkCudaErrors(x) check((x),#x,__FILE__,__LINE__)\n'
                      'template<class T> void memory(T* p) { checkCudaErrors(cudaFree(p)); }\n'
                      'void use() { memory<int>(nullptr); memory<float>(nullptr); }\n')
    result = run([os.environ.get("CLANG", "clang"), "-std=c++17",
                  "-fsyntax-only", "-Xclang", "-ast-dump=json", str(source)])
    ast = json.loads(result.stdout)
    template = next(node for node in ast["inner"]
                    if node.get("kind") == "FunctionTemplateDecl" and
                    node.get("name") == "memory")

    def descendants(node, kind):
        return ([node] if node.get("kind") == kind else []) + [
            found for child in node.get("inner", [])
            for found in descendants(child, kind)]

    functions = [node for node in template["inner"]
                 if node.get("kind") == "FunctionDecl"]
    assert len(functions) == 3
    primary_calls = descendants(functions[0], "CallExpr")
    assert len(primary_calls) == 2
    assert primary_calls[0]["inner"][0]["kind"] == "UnresolvedLookupExpr"
    for specialization in functions[1:]:
        calls = descendants(specialization, "CallExpr")
        assert len(calls) == 2
        for call, expected in zip(calls, ["check", "cudaFree"]):
            references = descendants(call["inner"][0], "DeclRefExpr")
            assert len(references) == 1
            assert references[0]["referencedDecl"]["name"] == expected
    print("host Clang AST: unresolved primary; both scalar check/runtime callees resolved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary")
    parser.add_argument("--cuda-path")
    parser.add_argument("--resource-dir")
    parser.add_argument("--expect-retained-closure", action="store_true",
                        help="AST-planning-only: require the existing system pragma boundary; "
                             "does not validate helper/guard publication")
    args = parser.parse_args()
    assert bool(args.binary) == bool(args.cuda_path) == bool(args.resource_dir)
    assert not args.expect_retained_closure or args.binary
    before = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="ascify-template-helper-") as temp:
        work = Path(temp)
        host_guard_contract(work)
        if not args.binary:
            print("native template helper transformation gate not run (no binary supplied)")
            return
        for mode in range(13):
            output = work / (str(mode) + ".cpp")
            conflicting_macro = {6: "__is_same", 7: "static_assert", 8: "T"}.get(mode)
            result = run([
                args.binary, str(FIXTURE), "--target-policy=dav-c310-vec",
                "--simt-math=fast", "--default-preprocessor",
                "--cuda-path=" + args.cuda_path,
                "--clang-resource-directory=" + args.resource_dir,
                "-o", str(output), "--", "-x", "cuda", "-std=c++17",
                "-I" + str(HELPERS), "-DASCIFY_TEMPLATE_CASE=" + str(mode),
            ], success=conflicting_macro is None)
            if conflicting_macro is not None:
                # These tokens also belong to the frozen compat surface. Its
                # existing whole-output rejection is stronger than retaining
                # the helper transaction, and must not be mistaken for an
                # arbitrary frontend failure or an admitted template domain.
                assert "Ascify cannot publish CUDA compatibility output because active input macro '" + conflicting_macro + "' collides with the frozen compat header" in result.stderr, result.stderr
                assert "status domain not proven" in result.stderr, result.stderr
                assert "all helper edits kept" in result.stderr, result.stderr
                assert PENDING not in result.stderr, result.stderr
                assert "committed 1 explicit template" not in result.stderr
                assert not output.exists()
                if args.expect_retained_closure:
                    assert PRAGMA_BOUNDARY in result.stderr, result.stderr
                scope = "AST-planning-only " if args.expect_retained_closure else "template "
                print(f"native {scope}case {mode}: type proof rejected; "
                      f"no output published after '{conflicting_macro}' macro collision")
                continue
            text = output.read_text()
            expected_guard = 'static_assert(__is_same(T, float)'
            if mode == 0:
                expected_guard += ' || __is_same(T, int)'
            expected_guard += ', "' + MESSAGE + '");'
            if args.expect_retained_closure:
                # The Mac system headers currently trip a separate, existing
                # pragma provenance boundary. Check the real AST proof without
                # relaxing that boundary or calling retained output a success.
                assert PRAGMA_BOUNDARY in result.stderr, result.stderr
                assert "#include <helper_cuda.h>" in text, result.stderr
                assert MESSAGE not in text, text
                assert "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS" not in text, text
                assert "::ascify::sampleFindCudaDevice" not in text, text
                assert "include kept" in result.stderr, result.stderr
                assert "committed 1 explicit template" not in result.stderr
                if mode <= 1:
                    planned = [line.split(PENDING, 1)[1] for line in
                               result.stderr.splitlines() if PENDING in line]
                    assert planned == [expected_guard], result.stderr
                else:
                    assert PENDING not in result.stderr, result.stderr
                    assert "status domain not proven" in result.stderr, result.stderr
                print(f"native AST-planning-only case {mode}: "
                      f"{'exact type guard proved' if mode <= 1 else 'type proof rejected'}; "
                      "helper and guard retained by system pragma boundary")
                continue
            if mode <= 1:
                assert "#include <helper_cuda.h>" not in text, result.stderr
                assert MESSAGE in text, text
                assert "__is_same(T, float)" in text, text
                assert ("__is_same(T, int)" in text) == (mode == 0), text
                assert "::ascify::sampleFindCudaDevice" in text, text
                assert "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS" in text, text
                assert "committed 1 explicit template type-domain guard(s)" in result.stderr
                # Include the untouched generated source in a separate TU.
                # In-domain reuse compiles; a new char instantiation must fail
                # explicitly instead of relying on today's source call set.
                for argument, good in [("float", True), ("int", mode == 0),
                                       ("char", False)]:
                    harness = work / (str(mode) + "-" + argument + ".cpp")
                    harness.write_text('#include "' + output.name + '"\n'
                                       'void additional_use() { template_memory<' +
                                       argument + '>(nullptr,nullptr,0,0,nullptr); }\n')
                    compiled = run([
                        os.environ.get("CXX", "c++"), "-std=c++17",
                        "-fsyntax-only", "-D__aicore__=",
                        "-DASCIFY_TEMPLATE_CASE=" + str(mode),
                        "-I" + str(ROOT / "tests/rewrite/stubs"),
                        "-I" + str(ROOT / "include"), "-I" + str(HELPERS),
                        str(harness),
                    ], good)
                    if not good:
                        assert MESSAGE in compiled.stderr, compiled.stderr
                print(f"native template case {mode}: atomic helper+guard; future char rejected")
            else:
                assert "#include <helper_cuda.h>" in text, result.stderr
                assert MESSAGE not in text, text
                assert "ASCIFY_NVIDIA_SAMPLE_CHECK_CUDA_ERRORS" not in text, text
                assert "::ascify::sampleFindCudaDevice" not in text, text
                assert "include kept" in result.stderr, result.stderr
                assert "status domain not proven" in result.stderr, result.stderr
                print(f"native template case {mode}: transaction retained without guard")
        assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == before
        if args.expect_retained_closure:
            print("native AST-planning-only: 2 positive proofs and 11 negative proofs passed; "
                  "helper publication and target validation remain pending")
        else:
            print("native template helper gate: 2 positive and 11 negative cases passed")


if __name__ == "__main__":
    main()
