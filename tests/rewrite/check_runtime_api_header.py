#!/usr/bin/env python3
"""Verify physical SDK include admission and retained SDK-surface refusal."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "tests/rewrite/fixtures/runtime_api_header/direct.cu"
INCLUDE = "#include <cuda_runtime_api.h>\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cuda-path", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--target-compiler")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--target-arg", action="append", default=[])
    args = parser.parse_args()
    if args.evidence_dir:
        args.evidence_dir.mkdir(parents=True, exist_ok=False)
    def record(label, stage, argv, result, source):
        if not args.evidence_dir:
            return
        stem = args.evidence_dir / (label + "." + stage)
        stem.with_suffix(stem.suffix + ".json").write_text(json.dumps({
            "argv": argv, "returncode": result.returncode,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "scope": ("frontend conversion" if stage == "frontend" else
                      "CCEC -O2 -c ET_REL object generation; no link/device execution" if stage == "target_object" else
                      "CCEC syntax only; no device execution"),
        }, indent=2) + "\n")
        stem.with_suffix(stem.suffix + ".stdout.log").write_text(result.stdout)
        stem.with_suffix(stem.suffix + ".stderr.log").write_text(result.stderr)
        (args.evidence_dir / (label + (".cu" if stage == "frontend" else ".dpp"))).write_bytes(source.read_bytes())
    cases = {
        "angle": (DIRECT.read_text(), True, ""),
        "quote": (DIRECT.read_text().replace("<cuda_runtime_api.h>", '"cuda_runtime_api.h"'), True, ""),
        "macro_include": ('#define HEADER <cuda_runtime_api.h>\n#include HEADER\n', False, "direct literal"),
        "conditional_include": ('#if 1\n' + INCLUDE + '#endif\n', False, "unconditional direct"),
        "version_if": (INCLUDE + '#if CUDART_VERSION >= 13000\nint version = 1;\n#else\nint version = 0;\n#endif\n', False, "removed SDK macro"),
        "guard_ifdef": (INCLUDE + '#ifdef __CUDA_RUNTIME_API_H__\nint selected = 1;\n#endif\n', False, "removed SDK macro"),
        "guard_ifndef": (INCLUDE + '#ifndef __CUDA_RUNTIME_API_H__\nint selected = 0;\n#endif\n', False, "removed SDK macro"),
        "version_defined": (INCLUDE + '#if defined(CUDART_VERSION)\nint selected = 1;\n#endif\n', False, "removed SDK macro"),
        "inactive_version": (INCLUDE + '#if 0\nint version = CUDART_VERSION;\n#endif\n', False, "removed SDK macro"),
        "version_undef": (INCLUDE + '#undef CUDART_VERSION\n', False, "removed SDK macro"),
        "version_paste": (INCLUDE + '#define CAT_I(a,b) a##b\n#define CAT(a,b) CAT_I(a,b)\nint version = CAT(CUDART_,VERSION);\n', False, "runtime-API header boundary"),
        "pragma_stack": (INCLUDE + '#pragma push_macro("CUDART_VERSION")\n#pragma pop_macro("CUDART_VERSION")\n', False, "pragma/replacement string"),
        "compiler_condition": (INCLUDE + '#ifdef __CUDACC__\nint selected = 1;\n#endif\n', False, "CUDA compiler condition"),
        "compiler_predefine": (INCLUDE + '#if defined(__CUDA__)\nint selected = 1;\n#endif\n', False, "CUDA compiler condition"),
        "compiler_arch": (INCLUDE + '#ifdef __CUDA_ARCH__\nint selected = 1;\n#endif\n', False, "CUDA compiler condition"),
        "strict_undef": (INCLUDE + '#undef __STRICT_ANSI__\n', False, "CUDA compiler condition"),
        "strict_rebind": (INCLUDE + '#define __STRICT_ANSI__ 1\n', False, "CUDA compiler condition"),
        "strict_line_spoof": ('#line 1 "<built-in>"\n#undef __STRICT_ANSI__\n#define __STRICT_ANSI__ 1\n' + INCLUDE, False, "CUDA compiler condition"),
        "vector_layout": (INCLUDE + 'static_assert(sizeof(float3) == 12, "CUDA ABI");\n', False, "unproved SDK record ABI"),
        "retained_header": (INCLUDE + '#include "observer.h"\n', False, "removed SDK macro"),
        "isystem_observer": (INCLUDE + '#include <sys_observer.h>\n', False, "removed SDK macro"),
        "unexpanded_paste": (INCLUDE + '#define CAT_I(a,b) a##b\n#define LATER CAT_I(CUDART_,VERSION)\n', False, "unproved retained macro dependency"),
        "inactive_paste": (INCLUDE + '#if 0\n#define LATER_INACTIVE(a,b) a##b\n#endif\n', False, "unproved retained token paste"),
        "native_unexpanded_paste": (INCLUDE + '#define GET_VERSION __CONCAT(CUDART_, VERSION)\n', False, "unproved retained macro dependency"),
        "native_nested_paste": (INCLUDE + '#define CONCAT_ALIAS __CONCAT\n#define GET_VERSION CONCAT_ALIAS(CUDART_, VERSION)\n', False, "unproved retained macro dependency"),
        "native_future_version": (INCLUDE + '#define FUTURE_VERSION __CONCAT(CUDART_, VERSION)\ntemplate<class T> int version() { return FUTURE_VERSION; }\n', False, "runtime-API header boundary"),
        "qualifier_sdk_argument": (INCLUDE + 'struct __align__(1 << (CUDART_VERSION % 3)) Value { int field; };\n', False, "removed SDK macro"),
        "isystem_false_native_guard": (INCLUDE + '#include <false_native_guard.h>\n', False, "CUDA compiler condition"),
        "pragma_replacement_string": (INCLUDE + '#define SAVE _Pragma("push_macro(\\\"CUDART_VERSION\\\")")\nSAVE\n', False, "unproved escaped pragma string"),
        "pragma_encoded_name": (INCLUDE + r'#pragma push_macro("CUDART_\126ERSION")' + '\n', False, "unproved escaped pragma string"),
        "pragma_encoded_replacement": (INCLUDE + r'#define SAVE _Pragma("push_macro(\"CUDART_\\126ERSION\")")' + '\nSAVE\n', False, "unproved escaped pragma string"),
        "guard_runtime": (INCLUDE + '#ifdef __CUDA_RUNTIME_H__\nint branch = 1;\n#endif\n', False, "removed SDK macro"),
        "guard_driver": (INCLUDE + '#ifdef __DRIVER_TYPES_H__\nint branch = 1;\n#endif\n', False, "removed SDK macro"),
        "guard_vector": (INCLUDE + '#ifdef __VECTOR_TYPES_H__\nint branch = 1;\n#endif\n', False, "removed SDK macro"),
        "vector_alias_pointer": (INCLUDE + 'using Triplet = float3; Triplet* values;\n', False, "unproved SDK record ABI"),
        "vector_pointer": (INCLUDE + 'float3* values;\n', False, "unproved SDK record ABI"),
        "vector_array": (INCLUDE + 'float3 values[2];\n', False, "unproved SDK record ABI"),
        "vector_alignment": (INCLUDE + 'static_assert(alignof(float3) == 4, "CUDA ABI");\n', False, "unproved SDK record ABI"),
        "vector_class_member": (INCLUDE + 'struct Container { float3 value; };\n', False, "unproved SDK record ABI"),
        "vector_typedef": (INCLUDE + 'typedef float3 Triplet; Triplet values[2];\n', False, "unproved SDK record ABI"),
        "vector_template": (INCLUDE + 'template<class T> struct Container { T value; }; Container<float3> value;\n', False, "unproved retained dependent type observation"),
        "vector_decltype": (INCLUDE + 'using Triplet = decltype(float3{}); Triplet value;\n', False, "unproved SDK record ABI"),
        "vector_decltype_member": (INCLUDE + 'using Scalar = decltype(float3{}.x); Scalar value;\n', False, "unproved SDK record ABI"),
        "vector_sizeof_pointer": (INCLUDE + 'static_assert(sizeof(float3*) == sizeof(void*), "pointer");\n', False, "unproved SDK record ABI"),
        "dim3_trait": (INCLUDE + '#include <type_traits>\nbool observed = std::is_default_constructible<dim3>::value;\n', False, "unproved SDK record ABI"),
        "dim3_builtin_trait": (INCLUDE + 'bool observed = __is_constructible(dim3);\n', False, "unproved SDK record ABI"),
        "dim3_sizeof": (INCLUDE + 'unsigned observed = sizeof(dim3);\n', False, "unproved SDK record ABI"),
        "runtime_record_deduced_reflection": (INCLUDE + 'template<class T> constexpr auto bytes(const T&) { return sizeof(T); }\nstatic_assert(bytes(cudaDeviceProp{}) > 512, "SDK size");\n', False, "unproved retained dependent type observation"),
        "runtime_record_explicit_reflection": (INCLUDE + 'template<class T> constexpr auto bytes() { return sizeof(T); }\nstatic_assert(bytes<cudaDeviceProp>() > 512, "SDK size");\n', False, "unproved retained dependent type observation"),
        "runtime_record_lambda_reflection": (INCLUDE + 'constexpr auto bytes = [](auto value) { return sizeof(value); };\nstatic_assert(bytes(cudaDeviceProp{}) > 512, "SDK size");\n', False, "unproved retained dependent type observation"),
        "future_dependent_function": (INCLUDE + 'template<class T> constexpr auto bytes(const T&) { return sizeof(T); }\n', False, "unproved retained dependent type observation"),
        "future_dependent_lambda": (INCLUDE + 'constexpr auto bytes = [](auto value) { return sizeof(value); };\n', False, "unproved retained dependent type observation"),
        "future_dependent_traits": (INCLUDE + '#include <type_traits>\ntemplate<class T> constexpr bool observed() { return std::is_default_constructible<T>::value; }\n', False, "unproved retained dependent type observation"),
        "isystem_dependent_function": (INCLUDE + '#include <future_observer.h>\n', False, "unproved retained dependent type observation"),
        "runtime_record_offsetof": (INCLUDE + 'static_assert(__builtin_offsetof(cudaDeviceProp, multiProcessorCount) > 256, "SDK layout");\n', False, "unproved SDK record ABI"),
        "runtime_record_alias_offsetof": (INCLUDE + 'using Device = cudaDeviceProp;\nstatic_assert(__builtin_offsetof(Device, multiProcessorCount) > 256, "SDK layout");\n', False, "unproved SDK record ABI"),
        "runtime_record_enclosed_sizeof": (INCLUDE + 'struct Container { cudaDeviceProp value; };\nstatic_assert(sizeof(Container) > 512, "SDK layout");\n', False, "unproved SDK record ABI"),
        "runtime_record_base_sizeof": (INCLUDE + 'struct Container : cudaDeviceProp {};\nstatic_assert(sizeof(Container) > 512, "SDK layout");\n', False, "unproved SDK record ABI"),
        "runtime_record_alignment_operand": (INCLUDE + 'struct alignas(cudaDeviceProp) Aligned { char value; };\nstatic_assert(alignof(Aligned) > 4, "SDK alignment");\n', False, "unproved SDK record ABI"),
        "runtime_record_sfinae": (INCLUDE + 'template<class T> constexpr auto has_warp(const T& value) -> decltype((void)value.warpSize, 1) { return 1; }\nconstexpr int has_warp(...) { return 0; }\nstatic_assert(has_warp(cudaDeviceProp{}) == 1, "SDK field");\n', False, "unproved retained dependent type observation"),
        "runtime_record_pointer_sfinae": (INCLUDE + 'template<class T> constexpr auto has_warp(T* value) -> decltype((void)value->warpSize, 1) { return 1; }\nconstexpr int has_warp(...) { return 0; }\nstatic_assert(has_warp((cudaDeviceProp*)nullptr) == 1, "SDK field");\n', False, "unproved retained dependent type observation"),
        "runtime_record_variable_alignment_operand": (INCLUDE + 'alignas(cudaDeviceProp) char value;\nstatic_assert(alignof(value) > 4, "SDK alignment");\n', False, "unproved SDK record ABI"),
        "future_dependent_member": (INCLUDE + 'template<class T> constexpr auto member(const T& value) -> decltype(value.warpSize) { return value.warpSize; }\n', False, "unproved retained dependent type observation"),
        "runtime_enum_trait": (INCLUDE + '#include <type_traits>\nstatic_assert(std::is_enum<cudaError_t>::value, "SDK enum");\n', False, "unproved SDK enum reflection"),
        "runtime_enum_builtin_trait": (INCLUDE + 'static_assert(__is_enum(cudaError_t), "SDK enum");\n', False, "unproved SDK enum reflection"),
        "runtime_enum_builtin_classify": (INCLUDE + 'static_assert(__builtin_classify_type(cudaSuccess) != __builtin_classify_type(0), "SDK enum category");\n', False, "unproved compiler SDK inquiry"),
        "runtime_record_builtin_object_size": (INCLUDE + 'cudaDeviceProp properties;\nstatic_assert(__builtin_object_size(&properties, 0) > 512, "SDK object size");\n', False, "unproved compiler SDK inquiry"),
        "runtime_record_member_builtin_object_size": (INCLUDE + 'cudaDeviceProp properties;\nstatic_assert(__builtin_object_size(&properties.name[0], 0) > 512, "enclosing SDK object size");\n', False, "unproved compiler SDK inquiry"),
        "runtime_record_builtin_dynamic_object_size": (INCLUDE + 'unsigned observed(cudaDeviceProp* properties) { return __builtin_dynamic_object_size(properties, 0); }\n', False, "unproved compiler SDK inquiry"),
        "runtime_record_constexpr_pointer_object_size": (INCLUDE + 'cudaDeviceProp properties; constexpr void* address = &properties;\nstatic_assert(__builtin_object_size(address, 0) > 512, "erased SDK object size");\n', False, "unproved compiler SDK inquiry"),
        "runtime_record_const_pointer_object_size": (INCLUDE + 'cudaDeviceProp properties; void* const address = &properties;\nstatic_assert(__builtin_object_size(address, 0) > 512, "erased SDK object size");\n', False, "unproved compiler SDK inquiry"),
        "runtime_record_accessor_object_size": (INCLUDE + 'cudaDeviceProp properties; constexpr const void* address() { return &properties; }\nstatic_assert(__builtin_object_size(address(), 0) > 512, "erased SDK object size");\n', False, "unproved compiler SDK inquiry"),
        "runtime_record_lambda_object_size": (INCLUDE + 'cudaDeviceProp properties; constexpr auto address = []() { return static_cast<const void*>(&properties); };\nstatic_assert(__builtin_object_size(address(), 0) > 512, "erased SDK object size");\n', False, "unproved compiler SDK inquiry"),
        "future_builtin_object_size": (INCLUDE + '#define FUTURE_SIZE(value) __builtin_object_size(value, 0)\n', False, "unproved retained compiler SDK inquiry"),
        "runtime_enum_numeric_cast": (INCLUDE + 'static_assert(static_cast<int>(cudaErrorInvalidValue) == 1, "SDK ordinal");\n', False, "unproved SDK enum numeric observation"),
        "runtime_enum_numeric_construction": (INCLUDE + 'static_assert(static_cast<cudaError_t>(1) == cudaErrorInvalidValue, "SDK ordinal construction");\n', False, "unproved SDK enum numeric observation"),
        "runtime_enum_arithmetic": (INCLUDE + 'static_assert(cudaErrorInvalidValue + 0 == 1, "SDK ordinal");\n', False, "unproved SDK enum numeric observation"),
        "runtime_enum_numeric_assignment": (INCLUDE + 'int ordinal = cudaErrorInvalidValue;\n', False, "unproved SDK enum numeric observation"),
        "runtime_enum_ordering": (INCLUDE + 'static_assert(cudaErrorMemoryAllocation < cudaErrorInvalidDevice, "SDK ordinal ordering");\n', False, "unproved SDK enum numeric observation"),
        "runtime_enum_switch": (INCLUDE + 'constexpr bool ordinal() { switch(cudaErrorInvalidValue) { case 1: return true; default: return false; } }\nstatic_assert(ordinal(), "SDK ordinal switch");\n', False, "unproved SDK enum numeric observation"),
        "runtime_enum_array_bound": (INCLUDE + 'char values[cudaErrorMemoryAllocation];\nstatic_assert(sizeof(values) == 2, "SDK ordinal bound");\n', False, "unproved SDK enum numeric observation"),
        "runtime_enum_alignment": (INCLUDE + 'alignas(cudaErrorMemoryAllocation) char value;\n', False, "unproved SDK enum numeric observation"),
        "runtime_enum_bitfield_width": (INCLUDE + 'struct Bits { unsigned value : cudaErrorMemoryAllocation; };\n', False, "unproved SDK enum numeric observation"),
        "runtime_enum_enumerator_initializer": (INCLUDE + 'enum Own { Value = cudaErrorMemoryAllocation };\nstatic_assert(Value == 2, "SDK ordinal initializer");\n', False, "unproved SDK enum numeric observation"),
        "runtime_enum_overload": (INCLUDE + 'constexpr int pick(cudaError_t) { return 1; }\nconstexpr int pick(int) { return 2; }\nstatic_assert(pick(cudaSuccess) == 1, "SDK overload");\n', False, "unproved caller SDK enum function interface"),
        "runtime_enum_parameter": (INCLUDE + 'bool failed(cudaError_t value) { return value != cudaSuccess; }\n', False, "unproved caller SDK enum function interface"),
        "runtime_enum_template_return": (INCLUDE + 'template<class T> cudaError_t status() { return cudaSuccess; }\n', False, "unproved caller SDK enum function interface"),
        "runtime_enum_conversion": (INCLUDE + 'struct Error { constexpr operator cudaError_t() const { return cudaSuccess; } constexpr operator int() const { return 7; } };\nstatic_assert(static_cast<cudaError_t>(Error{}) == cudaSuccess, "SDK conversion selection");\n', False, "unproved caller SDK enum function interface"),
        "runtime_enum_member_return": (INCLUDE + 'struct Status { cudaError_t release(void* value) { return cudaFree(value); } };\n', True, ""),
        "runtime_enum_function_pointer_interface": (INCLUDE + 'using EF = int(*)(cudaError_t); using IF = int(*)(int);\nconstexpr int pick(EF) { return 1; }\nconstexpr int pick(IF) { return 2; }\nstatic_assert(pick((EF)nullptr) == 1, "SDK interface");\n', False, "unproved caller SDK enum function interface"),
        "runtime_enum_member_pointer_interface": (INCLUDE + 'struct User { cudaError_t status; int value; };\nusing EP = cudaError_t User::*; using IP = int User::*;\nconstexpr int pick(EP) { return 1; }\nconstexpr int pick(IP) { return 2; }\nstatic_assert(pick((EP)nullptr) == 1, "SDK interface");\n', False, "unproved caller SDK enum function interface"),
        "runtime_properties_value_init": (INCLUDE + 'int properties() { cudaDeviceProp value{}; return value.multiProcessorCount; }\n', True, ""),
        "runtime_func_attributes_value_init": (INCLUDE + 'int attributes() { cudaFuncAttributes value{}; return static_cast<int>(value.sharedSizeBytes); }\n', True, ""),
        "runtime_properties_nonempty_init": (INCLUDE + 'cudaDeviceProp value{{0}};\n', False, "unproved SDK aggregate initializer"),
        "runtime_attributes_nonempty_init": (INCLUDE + 'cudaFuncAttributes value{1};\n', False, "unproved SDK aggregate initializer"),
        "plain_compiler_inquiries": (INCLUDE + 'static_assert(__builtin_classify_type(0) == 1, "integer category");\nchar value[32];\nstatic_assert(__builtin_object_size(value, 0) == 32, "plain object size");\n', True, ""),
        "plain_constant_pointer_inquiry": (INCLUDE + 'char value[32]; constexpr void* address = &value;\nstatic_assert(__builtin_object_size(address, 0) == 32, "plain erased object size");\n', True, ""),
        "runtime_symbolic_error_observations": (INCLUDE + 'static_assert(cudaErrorMemoryAllocation != cudaErrorInvalidDevice, "distinct errors");\nstatic_assert(static_cast<bool>(cudaErrorInvalidValue), "error truth");\n', True, ""),
        "runtime_same_enum_cast": (INCLUDE + 'static_assert(static_cast<cudaError_t>(cudaErrorInvalidValue) == cudaErrorInvalidValue, "same error domain");\n', True, ""),
        "runtime_error_comparison": (INCLUDE + 'bool configure() { return cudaSetDevice(0) == cudaSuccess; }\n', True, ""),
        "runtime_status_return": (INCLUDE + 'cudaError_t release(void* value) { return cudaFree(value); }\n', True, ""),
        "plain_conversion_operators": (INCLUDE + 'struct Plain { constexpr operator int() const { return 7; } constexpr operator bool() const { return true; } };\nstatic_assert(static_cast<int>(Plain{}) == 7, "integer conversion");\nstatic_assert(static_cast<bool>(Plain{}), "boolean conversion");\n', True, ""),
        "gnu_language_mode": (INCLUDE + 'int value = 1;\n', False, "CUDA compiler condition"),
        "sdk_header_inquiry": (INCLUDE + '#if __has_include(<cuda_runtime_api.h>)\nconstexpr int present = 1;\n#else\nconstexpr int present = 0;\n#endif\nstatic_assert(present == 1, "SDK header");\n', False, "unproved retained compiler inquiry"),
        "compiler_feature_inquiry": (INCLUDE + '#if __has_feature(cuda)\nint value = 1;\n#endif\n', False, "unproved retained compiler inquiry"),
        "retained_record_collision": (INCLUDE + 'namespace own { struct cudaDeviceProp { int value; }; }\nown::cudaDeviceProp value;\n', False, "conflicting retained runtime type identity"),
        "retained_alias_collision": (INCLUDE + 'namespace own { using cudaFuncAttributes = int; }\nown::cudaFuncAttributes value;\n', False, "conflicting retained runtime type identity"),
        "native_standard_templates": (INCLUDE + '#include <type_traits>\nstatic_assert(std::is_integral<int>::value, "integral");\nstatic_assert(std::is_floating_point<float>::value, "floating");\nstruct PlainRecord { int value; };\nstatic_assert(std::is_standard_layout<PlainRecord>::value, "layout");\n', True, ""),
        "unsupported_managed": (INCLUDE + 'cudaError_t managed(float** value) { return cudaMallocManaged(value, sizeof(float)); }\n', True, ""),
    }
    fixture_hash = hashlib.sha256(DIRECT.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="ascify-runtime-api-header-") as temp:
        work = Path(temp)
        (work / "observer.h").write_text('#if CUDART_VERSION >= 13000\ninline int observed() { return 1; }\n#else\ninline int observed() { return 0; }\n#endif\n')
        system_observer = work / "system_observer"
        system_observer.mkdir()
        (system_observer / "sys_observer.h").write_bytes((work / "observer.h").read_bytes())
        (system_observer / "false_native_guard.h").write_text(
            '#if !defined(__STRICT_ANSI__) && defined(_GLIBCXX_USE_FLOAT128) && !defined(__CUDACC__)\nint selected = 1;\n#endif\n')
        (system_observer / "future_observer.h").write_text(
            'template<class T> constexpr auto future_bytes(const T&) { return sizeof(T); }\n')
        shadow_bootstrap = work / "shadow_bootstrap"
        shadow_bootstrap.mkdir()
        (shadow_bootstrap / "string.h").write_text(
            '#ifndef CALLER_SHADOW_STRING_H\n#define CALLER_SHADOW_STRING_H\n'
            '#include_next <string.h>\n#if defined(__CUDACC__)\n'
            'constexpr int bootstrap_observed() { return 1; }\n#else\n'
            'constexpr int bootstrap_observed() { return 0; }\n#endif\n#endif\n')
        cases["bootstrap_shadow_observer"] = (
            INCLUDE + '#include <string.h>\nstatic_assert(bootstrap_observed() == 1, "compiler branch");\n',
            False, "CUDA compiler condition")
        spoof = work / "spoof"
        spoof.mkdir()
        (spoof / "cuda_runtime_api.h").write_text('inline int user_owned() { return 7; }\n')
        cases["spoof"] = (INCLUDE + 'int selected = user_owned();\n', False, "physical --cuda-path")
        sdk_api = Path(args.cuda_path).resolve() / "include/cuda_runtime_api.h"
        shadow = work / "overridden_runtime_api.h"
        shadow.write_bytes(sdk_api.read_bytes() + b"\n")
        overlay = work / "overlay.json"
        overlay.write_text(json.dumps({"version": 0, "use-external-names": False,
                                      "roots": [{"type": "file", "name": str(sdk_api),
                                                 "external-contents": str(shadow)}]}))
        cases["vfs_override"] = (INCLUDE + 'int selected = 1;\n', False, "Ascify cannot translate input selected by VFS overlay arguments")
        cases["vfs_other_header"] = (INCLUDE + 'int selected = 1;\n', False, "Ascify cannot translate input selected by VFS overlay arguments")
        cases["sdk_api_remap"] = (INCLUDE + 'int selected = 1;\n', False, "physical --cuda-path")
        cases["sdk_dependency_remap"] = (INCLUDE + 'int selected = 1;\n', False, "physical SDK dependency contents")
        copied_resources = work / "copied_resources"
        shutil.copytree(args.resource_dir, copied_resources)
        resource_observer = copied_resources / "include/caller_sdk_observer.h"
        resource_observer.write_text(
            '#if CUDART_VERSION >= 13000\nconstexpr int resource_observed() { return 1; }\n#else\nconstexpr int resource_observed() { return 0; }\n#endif\n')
        observer_include = '#include "' + str(resource_observer) + '"\n'
        resource_source = INCLUDE + observer_include + 'static_assert(resource_observed() == 1, "SDK branch");\n'
        cases["resource_observer"] = (resource_source, False, "removed SDK macro")
        cases["resource_preinclude_observer"] = (INCLUDE + 'static_assert(resource_observed() == 1, "SDK branch");\n', False, "removed SDK macro")
        cases["copied_resources_positive"] = (DIRECT.read_text(), True, "")
        sdk_mirror = work / "sdk_mirror"
        sdk_mirror.mkdir()
        shutil.copytree(Path(args.cuda_path) / "include", sdk_mirror / "include")
        for name in ("nvvm", "bin", "version.json", "version.txt"):
            original = Path(args.cuda_path).resolve() / name
            if original.is_dir():
                (sdk_mirror / name).symlink_to(original, target_is_directory=True)
            elif original.is_file():
                shutil.copy2(original, sdk_mirror / name)
        sdk_observer = sdk_mirror / "include/caller_observer.h"
        sdk_observer.write_text(
            '#if CUDART_VERSION >= 13000\nconstexpr int sdk_observed() { return 1; }\n#else\n'
            'constexpr int sdk_observed() { return 0; }\n#endif\n')
        cases["sdk_root_caller_main"] = (INCLUDE + '#if CUDART_VERSION >= 13000\nint version = 1;\n#endif\n', False, "removed SDK macro")
        cases["sdk_root_caller_header"] = (INCLUDE + '#include "' + str(sdk_observer) + '"\nstatic_assert(sdk_observed() == 1, "SDK branch");\n', False, "removed SDK macro")
        cases["sdk_mirror_positive"] = (DIRECT.read_text(), True, "")
        dependency = Path(args.cuda_path).resolve() / "include/vector_types.h"
        dependency_shadow = work / "overridden_vector_types.h"
        dependency_shadow.write_bytes(dependency.read_bytes() + b"\n")
        other_overlay = work / "other_overlay.json"
        other_overlay.write_text(json.dumps({"version": 0, "roots": [{
            "type": "file", "name": str(work / "unreferenced.h"),
            "external-contents": str(shadow)}]}))
        unknown = set(args.case) - set(cases)
        if unknown:
            parser.error("unknown case: " + ", ".join(sorted(unknown)))
        for label, (text, accepted, reason) in cases.items():
            if args.case and label not in args.case:
                continue
            source = (sdk_mirror / "include" if label == "sdk_root_caller_main" else work) / (label + ".cu")
            output = work / (label + ".dpp")
            source.write_text(text)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            resource = copied_resources if label.startswith("resource_") or label == "copied_resources_positive" else args.resource_dir
            sdk = sdk_mirror if label.startswith("sdk_root_") or label == "sdk_mirror_positive" else args.cuda_path
            argv = [args.binary, str(source), "--default-preprocessor",
                    "--target-policy=dav-c310-vec", "--simt-math=fast",
                    "--cuda-path=" + str(sdk),
                    "--clang-resource-directory=" + str(resource),
                    "-o", str(output), "--", "-x", "cuda", "-std=c++17",
                    "-I" + str(work)]
            if label == "spoof":
                argv.append("-I" + str(spoof))
            if label == "vfs_override":
                argv.extend(["-ivfsoverlay", str(overlay)])
            if label == "vfs_other_header":
                argv.extend(["-ivfsoverlay", str(other_overlay)])
            if label == "sdk_api_remap":
                argv.extend(["-Xclang", "-remap-file", "-Xclang", str(sdk_api) + ";" + str(shadow)])
            if label == "sdk_dependency_remap":
                argv.extend(["-Xclang", "-remap-file", "-Xclang", str(dependency) + ";" + str(dependency_shadow)])
            if label == "resource_preinclude_observer":
                argv.extend(["-include", str(resource_observer)])
            if label == "bootstrap_shadow_observer":
                argv.append("-I" + str(shadow_bootstrap))
            if label == "gnu_language_mode":
                argv.append("-std=gnu++17")
            if label in ("isystem_observer", "isystem_false_native_guard", "isystem_dependent_function"):
                argv.extend(["-isystem", str(system_observer)])
            result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
            record(label, "frontend", argv, result, source)
            detail = "\n".join(line for line in result.stderr.splitlines() if "error:" in line)
            if (result.returncode == 0) != accepted:
                raise AssertionError(label + ": unexpected frontend admission\n" + detail)
            if reason and reason not in result.stderr:
                raise AssertionError(label + ": missing refusal reason\n" + detail)
            if hashlib.sha256(source.read_bytes()).hexdigest() != before:
                raise AssertionError(label + ": source bytes changed")
            if accepted:
                emitted = output.read_text()
                if "cuda_runtime_api.h" in emitted or "ascify/ascify_cuda_compat.hpp" not in emitted:
                    raise AssertionError(label + ": incomplete include adaptation")
                if label in ("angle", "quote", "copied_resources_positive", "sdk_mirror_positive") and 'dim3 one = dim3(1U, 1U, 1U)' not in emitted:
                    raise AssertionError(label + ": default CUDA dim3 changed")
                if label == "unsupported_managed" and "cudaMallocManaged" not in emitted:
                    raise AssertionError(label + ": unsupported managed API was hidden")
                if args.target_compiler:
                    cmd = [args.target_compiler, "-x", "dpp", "-std=c++17", "-fsyntax-only",
                           "-I" + str(ROOT / "include"), *args.target_arg, str(output)]
                    target = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    record(label, "target", cmd, target, output)
                    target_ok = label != "unsupported_managed"
                    if (target.returncode == 0) != target_ok:
                        raise AssertionError(label + ": unexpected target admission\n" + target.stderr[-5000:])
                    if not target_ok and "cudaMallocManaged" not in target.stderr:
                        raise AssertionError(label + ": target refusal did not name unsupported API")
                    if target_ok:
                        target_object = work / (label + ".o")
                        object_cmd = [args.target_compiler, "-x", "dpp", "-std=c++17",
                                      "-O2", "-c", "-I" + str(ROOT / "include"),
                                      *args.target_arg, str(output), "-o", str(target_object)]
                        compiled = subprocess.run(object_cmd, capture_output=True, text=True, timeout=60)
                        record(label, "target_object", object_cmd, compiled, output)
                        if compiled.returncode != 0:
                            raise AssertionError(label + ": CCEC object compilation failed\n" + compiled.stderr[-5000:])
                        data = target_object.read_bytes()
                        byte_order = "little" if len(data) > 5 and data[5] == 1 else "big"
                        if len(data) < 18 or data[:4] != b"\x7fELF" or int.from_bytes(data[16:18], byte_order) != 1:
                            raise AssertionError(label + ": CCEC did not emit an ET_REL object")
                        if args.evidence_dir:
                            (args.evidence_dir / (label + ".o")).write_bytes(data)
                            metadata = args.evidence_dir / (label + ".target_object.json")
                            value = json.loads(metadata.read_text())
                            value.update(object_bytes=len(data), elf_type="ET_REL",
                                         object_sha256=hashlib.sha256(data).hexdigest())
                            metadata.write_text(json.dumps(value, indent=2) + "\n")
            print(label + (": accepted" if accepted else ": refused"))
    if hashlib.sha256(DIRECT.read_bytes()).hexdigest() != fixture_hash:
        raise AssertionError("fixture bytes changed")
    print("runtime-API header boundaries passed; target checks " + ("executed" if args.target_compiler else "not requested"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
