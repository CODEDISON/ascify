#ifndef ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_H_
#define ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_H_

#include <acl/acl.h>
#include <cstdint>

#ifndef ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_TYPES_READY
#include <simt_api/asc_fp16.h>
#endif

// Version 1 fixes the launch signatures, selection contract, and the four
// separate device-registration artifacts. Rebuilding both caller and runtime
// against another CANN SDK is still required.
#define ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION 1

extern "C" aclError ascify950_softmax_reg_recompute_launch_v1(
    aclrtStream stream, const half* input, half* output, int64_t rows,
    int columns, int block_threads, int grid_cap);

// Early v1 libraries support only the original wide Softmax domain. The
// optional symbol is a weak reference in PIC ELF callers. Darwin and
// non-PIC Linux callers resolve it by name.
extern "C"
#if defined(__ELF__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((weak))
#endif
int ascify950_softmax_supports_short_v1(
    aclrtStream stream, const half* input, half* output, int64_t rows,
    int columns);

extern "C" aclError ascify950_rmsnorm_reg_cached_launch_v1(
    aclrtStream stream, const half* input, half* output, const half* weight,
    float* inverse_rms, int64_t rows, int64_t columns, double epsilon);

// Plain row-batch intentionally has no weight parameter, so affine calls
// cannot cross this ABI boundary.
extern "C" aclError ascify950_rmsnorm_reg_plain_rowbatch_launch_v1(
    aclrtStream stream, const half* input, half* output, float* inverse_rms,
    int64_t rows, int64_t columns, double epsilon);

extern "C" aclError ascify950_layernorm_reg_cached_launch_v1(
    aclrtStream stream, const half* input, half* output, float* mean,
    float* inverse_variance, int64_t rows, int64_t columns, double epsilon);

#endif  // ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_ABI_H_
