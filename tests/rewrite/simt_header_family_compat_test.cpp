#define __aicore__
#define __global__

#include <ascify/ascify_cuda_compat.hpp>

#include <cassert>
#include <cstdint>

#if defined(ASCIFY_TEST_EXPECT_PUBLIC_85)
#if !defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85) || \
    defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3)
#error "public 8.5 test did not select exactly the aggregate SIMT header"
#endif

ASCIFY_GLOBAL void HeaderFamilyKernel() {}
#else
#if !defined(ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3) || \
    defined(ASCIFY_SIMT_HEADER_FAMILY_PUBLIC_85)
#error "legacy test did not select exactly the split SIMT headers"
#endif
#if !defined(ASCIFY_SIMT_LEGACY_HAS_VECTOR_CONSTRUCTORS)
#error "legacy test did not include the native vector constructor surface"
#endif
#endif

int main() {
  assert(static_cast<unsigned int>(ascify::cudaDevAttrWarpSize) == 202U);
  assert(static_cast<unsigned int>(
             ascify::cudaDevAttrMaxThreadsPerVectorCore) == 203U);
  assert(static_cast<unsigned int>(
             ascify::cudaDevAttrLocalMemoryPerVectorCore) == 204U);
  assert(static_cast<unsigned int>(
             ascify::cudaDevAttrMaxThreadsPerBlock) == 209U);

  assert(ascify::warp_reduce_add(1.0f) == 1.0f);
  assert(ascify::warp_reduce_add(int32_t{2}) == 2);
  assert(ascify::warp_reduce_add(uint32_t{3}) == 3U);
  assert(ascify::warp_reduce_max(4.0f) == 4.0f);
  assert(ascify::warp_reduce_min(5.0f) == 5.0f);
  assert(ascify::shfl_sync(UINT32_MAX, 6, 0, 32) == 6);
  assert(ascify::shfl_up_sync(UINT32_MAX, 7, 1, 16) == 7);
  assert(ascify::shfl_down_sync(UINT32_MAX, 8, 1, 8) == 8);
  assert(ascify::shfl_xor_sync(UINT32_MAX, 9, 1, 4) == 9);
  assert(ascify::fdividef(10.0f, 2.0f) == 5.0f);
  ascify::syncwarp();
  ascify::syncthreads();
  ascify::threadfence_block();
  ascify::threadfence();

#if defined(ASCIFY_TEST_EXPECT_PUBLIC_85)
  assert(ascify::cudaEventRecord(nullptr) == ACL_ERROR_FEATURE_UNSUPPORTED);
  assert(AscendC::Simt::ascify_test_public_warp_reduce_add_calls == 3);
  assert(AscendC::Simt::ascify_test_public_warp_reduce_max_calls == 1);
  assert(AscendC::Simt::ascify_test_public_warp_reduce_min_calls == 1);
  assert(AscendC::Simt::ascify_test_public_warp_shuffle_calls == 4);
  assert(AscendC::Simt::ascify_test_public_thread_barrier_calls == 1);
  assert(AscendC::Simt::ascify_test_public_thread_fence_calls == 2);
#else
  // Existing full-mask 64-bit operations must not instantiate the narrower
  // partial-mask helper. Native stubs preserve all bits, including high bits.
  assert(ascify::shfl_down_sync(UINT32_MAX, int64_t{-0x100000001LL}, 1, 32)
         == -0x100000001LL);
  assert(ascify::shfl_down_sync(UINT32_MAX, uint64_t{0x8000000100000001ULL}, 1, 32)
         == 0x8000000100000001ULL);
  const float4 vector = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
  assert(vector.x == 1.0f && vector.y == 2.0f &&
         vector.z == 3.0f && vector.w == 4.0f);
  assert(ascify_test_legacy_warp_reduce_add_calls == 3);
  assert(ascify_test_legacy_warp_reduce_max_calls == 1);
  assert(ascify_test_legacy_warp_reduce_min_calls == 1);
  assert(ascify_test_legacy_warp_shuffle_calls == 6);
  assert(ascify_test_legacy_thread_barrier_calls == 1);
  assert(ascify_test_legacy_thread_fence_block_calls == 1);
  assert(ascify_test_legacy_thread_fence_calls == 1);
  assert(ascify_test_legacy_fdivide_calls == 1);
#endif
  return 0;
}
