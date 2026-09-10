#include <ascify/target/dav_c310/rowwise_simd_selectors_v1.hpp>
#include "rmsnorm_cached_capacity_v1.hpp"

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace simd = ascify::target::dav_c310::rowwise_simd_v1;

template<typename T>
T* Address(uintptr_t value) {
  return reinterpret_cast<T*>(value);
}

int main() {
  static_assert(simd::kHalfStorageBytes == 2U,
                "selector v1 requires two-byte fp16 storage");
  void* stream = Address<void>(0x1000U);
  const void* input = Address<void>(0x100000U);
  void* output = Address<void>(0x200000U);
  void* inverse_rms = Address<void>(0x300000U);
  const void* weight = Address<void>(0x400000U);
  void* mean = Address<void>(0x500000U);
  void* inverse_variance = Address<void>(0x600000U);

  assert(simd::IsSoftmaxSimdDomain(
      stream, input, output, 8, 4096));
  assert(simd::IsSoftmaxSimdDomain(
      stream, input, input, 8, 4096));
  assert(!simd::IsSoftmaxSimdDomain(
      stream, input, Address<void>(0x100002U), 8, 4096));
  assert(!simd::IsSoftmaxSimdDomain(
      stream, input, output, 8, 2048));
  assert(!simd::IsSoftmaxSimdDomain(
      stream, input, output, 8, 4096, 256, 0));

  assert(simd::IsRmsNormPlainRowBatchSimdDomain(
      stream, input, output, nullptr, inverse_rms,
      8, 1536, 1.0e-5));
  assert(simd::IsRmsNormCachedSimdDomain(
      stream, input, output, nullptr, inverse_rms,
      8, 1536, 1.0e-5));
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kPlainRowBatch);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, inverse_rms,
             8, 1544, 1.0e-5)
         == simd::RmsNormRoute::kCached);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, weight, inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kCached);
  assert(simd::SelectRmsNormRoute(
             stream, input, input, nullptr, inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kPlainRowBatch);

  // Both adapter variants share the extended cache domain, including partial
  // vector tails. The former 8192 boundary remains cached and has the same UB.
  static_assert(ascify950_rmsnorm_cached_capacity_v1::Legacy::
                        kMaximumAffineUbBytes == 65600,
                "legacy cached footprint changed");
  for (const void* gamma : {static_cast<const void*>(nullptr), weight}) {
    for (int64_t columns : {8192, 8200, 12288, 13312, 16376, 16384}) {
      assert(simd::SelectRmsNormRoute(
                 stream, input, output, gamma, inverse_rms,
                 8, columns, 1.0e-5)
             == simd::RmsNormRoute::kCached);
      assert(simd::SelectRmsNormRoute(
                 stream, input, input, gamma, inverse_rms,
                 8, columns, 1.0e-5)
             == simd::RmsNormRoute::kSimt);
    }
    for (int64_t columns : {8193, 12289, 16383, 16385, 16392, 32768}) {
      assert(simd::SelectRmsNormRoute(
                 stream, input, output, gamma, inverse_rms,
                 8, columns, 1.0e-5)
             == simd::RmsNormRoute::kSimt);
    }
  }

  // The original 33 include these exact 2^31 and >2^31 element geometries.
  const void* large_input = Address<void>(0x10000000000ULL);
  void* large_output = Address<void>(0x20000000000ULL);
  void* large_inverse = Address<void>(0x30000000000ULL);
  const void* large_weight = Address<void>(0x40000000000ULL);
  for (const void* gamma : {static_cast<const void*>(nullptr), large_weight}) {
    for (int64_t large_rows : {131072, 488281}) {
      assert(simd::SelectRmsNormRoute(
                 stream, large_input, large_output, gamma, large_inverse,
                 large_rows, 16384, 1.0e-5)
             == simd::RmsNormRoute::kCached);
    }
  }

  assert(simd::SelectRmsNormRoute(
             stream, input, Address<void>(0x100010U), nullptr, inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kSimt);

  assert(simd::IsLayerNormCachedHybridDomain(
      stream, input, output, mean, inverse_variance,
      8, 1536, 1.0e-5));
  assert(simd::IsLayerNormCachedHybridDomain(
      stream, input, input, mean, inverse_variance,
      8, 1536, 1.0e-5));
  assert(!simd::IsLayerNormCachedHybridDomain(
      stream, input, Address<void>(0x100010U), mean, inverse_variance,
      8, 1536, 1.0e-5));
  assert(!simd::IsLayerNormCachedHybridDomain(
      stream, input, output, Address<void>(0x100000U), inverse_variance,
      8, 1536, 1.0e-5));
  assert(!simd::IsLayerNormCachedHybridDomain(
      stream, input, output, mean, mean, 8, 1536, 1.0e-5));
  assert(!simd::IsLayerNormCachedHybridDomain(
      stream, input, output, mean, inverse_variance,
      8, 1537, 1.0e-5));
  assert(!simd::IsLayerNormCachedHybridDomain(
      stream, input, output, mean, inverse_variance,
      8, 1536, -1.0));
  assert(simd::SelectRmsNormRoute(
             stream, Address<void>(0x100002U), output, nullptr, inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kSimt);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, Address<void>(0x100000U),
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kSimt);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, Address<void>(0x100000U), inverse_rms,
             8, 1536, 1.0e-5)
         == simd::RmsNormRoute::kSimt);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, inverse_rms,
             8, 1536, -1.0)
         == simd::RmsNormRoute::kSimt);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, inverse_rms,
             8, 1536, 1.0e-5, 256, 0)
         == simd::RmsNormRoute::kSimt);
  assert(simd::SelectRmsNormRoute(
             stream, input, output, nullptr, inverse_rms,
             std::numeric_limits<int64_t>::max(), 1536, 1.0e-5)
         == simd::RmsNormRoute::kSimt);

  simd::ByteSpan overflow_span{};
  assert(!simd::TryMakeMatrixByteSpan(
      Address<void>(0x1000U), std::numeric_limits<int64_t>::max(), 1,
      simd::kHalfStorageBytes, &overflow_span));

  return 0;
}
