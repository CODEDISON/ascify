#ifndef ASCIFY_CUDA_VEC3_LAYOUT_HPP
#define ASCIFY_CUDA_VEC3_LAYOUT_HPP

#include <stddef.h>
#include <type_traits>

// These records deliberately live in the CUDA records' original ADL namespace.
// They do not replace, inherit from, or implicitly convert to native DPP vectors.
struct alignas(4) ascify_cuda_float3 { float x, y, z; };
struct alignas(4) ascify_cuda_uint3 { unsigned int x, y, z; };
struct alignas(4) ascify_cuda_int3 { int x, y, z; };

static_assert(sizeof(float) == 4 && sizeof(unsigned int) == 4 && sizeof(int) == 4,
              "Ascify CUDA vec3 requires 32-bit component types");
static_assert(sizeof(ascify_cuda_float3) == 12 && alignof(ascify_cuda_float3) == 4 &&
              offsetof(ascify_cuda_float3, x) == 0 && offsetof(ascify_cuda_float3, y) == 4 &&
              offsetof(ascify_cuda_float3, z) == 8 && sizeof(ascify_cuda_float3[2]) == 24,
              "Ascify CUDA float3 layout must be 12 bytes, aligned to 4");
static_assert(sizeof(ascify_cuda_uint3) == 12 && alignof(ascify_cuda_uint3) == 4 &&
              offsetof(ascify_cuda_uint3, x) == 0 && offsetof(ascify_cuda_uint3, y) == 4 &&
              offsetof(ascify_cuda_uint3, z) == 8 && sizeof(ascify_cuda_uint3[2]) == 24,
              "Ascify CUDA uint3 layout must be 12 bytes, aligned to 4");
static_assert(sizeof(ascify_cuda_int3) == 12 && alignof(ascify_cuda_int3) == 4 &&
              offsetof(ascify_cuda_int3, x) == 0 && offsetof(ascify_cuda_int3, y) == 4 &&
              offsetof(ascify_cuda_int3, z) == 8 && sizeof(ascify_cuda_int3[2]) == 24,
              "Ascify CUDA int3 layout must be 12 bytes, aligned to 4");
static_assert(std::is_standard_layout<ascify_cuda_float3>::value &&
              std::is_trivially_copyable<ascify_cuda_float3>::value &&
              std::is_aggregate<ascify_cuda_float3>::value &&
              std::is_standard_layout<ascify_cuda_uint3>::value &&
              std::is_trivially_copyable<ascify_cuda_uint3>::value &&
              std::is_aggregate<ascify_cuda_uint3>::value &&
              std::is_standard_layout<ascify_cuda_int3>::value &&
              std::is_trivially_copyable<ascify_cuda_int3>::value &&
              std::is_aggregate<ascify_cuda_int3>::value,
              "Ascify CUDA vec3 records must remain plain aggregates");

// This target qualifier matches CANN 9.1's pure-SIMT make_* implementation.
// A real host-C++ translation unit needs no target headers or CUDA predefines.
#if defined(__NPU_COMPILER_INTERNAL_PURE_SIMT__)
#define ASCIFY_CUDA_VEC3_FUNCTION __aicore__ inline
#elif defined(__NPU_ARCH__)
#error "Ascify CUDA vec3 factories require the reviewed pure-SIMT compiler mode"
#else
#define ASCIFY_CUDA_VEC3_FUNCTION inline
#endif

ASCIFY_CUDA_VEC3_FUNCTION ascify_cuda_float3 ascify_make_cuda_float3(float x, float y, float z) {
  ascify_cuda_float3 result;
  result.x = x; result.y = y; result.z = z;
  return result;
}
ASCIFY_CUDA_VEC3_FUNCTION ascify_cuda_uint3 ascify_make_cuda_uint3(unsigned int x, unsigned int y, unsigned int z) {
  ascify_cuda_uint3 result;
  result.x = x; result.y = y; result.z = z;
  return result;
}
ASCIFY_CUDA_VEC3_FUNCTION ascify_cuda_int3 ascify_make_cuda_int3(int x, int y, int z) {
  ascify_cuda_int3 result;
  result.x = x; result.y = y; result.z = z;
  return result;
}

#undef ASCIFY_CUDA_VEC3_FUNCTION
#endif
