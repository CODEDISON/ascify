#ifndef ASCIFY950_RMSNORM_CACHED_CAPACITY_V1_HPP_
#define ASCIFY950_RMSNORM_CACHED_CAPACITY_V1_HPP_

#include <ascify/target/dav_c310/rowwise_simd_selectors_v1.hpp>

#include <cstdint>

namespace ascify950_rmsnorm_cached_capacity_v1 {

constexpr uint32_t kLegacyMaxColumns = 8192;
constexpr uint32_t kMaxColumns = 16384;
constexpr uint32_t kPipelineDepth = 1;
constexpr uint32_t kScalarSlotBytes = 32;
constexpr uint32_t kScalarSlotCount = 2;
constexpr uint32_t kConservativeUbBytes = 196608;

static_assert(kMaxColumns == ascify::target::dav_c310::rowwise_simd_v1::
                                kRmsNormCachedMaximumColumns,
              "cached runtime and selector must share the column limit");
static_assert(kPipelineDepth == 1,
              "cached RMSNorm capacity assumes a single pipeline slot");

template <uint32_t kColumns>
struct Capacity {
  static_assert(kColumns > 0 && kColumns <= kMaxColumns && kColumns % 16 == 0,
                "cached capacity must contain aligned fp16 rows");
  static constexpr uint32_t kRowBytes = kColumns * sizeof(uint16_t);
  // Input/output queues, retained weight, normalized row, inverse and sum.
  static constexpr uint32_t kMaximumAffineUbBytes =
      (2 * kPipelineDepth + 2) * kRowBytes
      + kScalarSlotCount * kScalarSlotBytes;
  static_assert(kMaximumAffineUbBytes <= kConservativeUbBytes,
                "cached RMSNorm exceeds conservative UB budget");
};

using Legacy = Capacity<kLegacyMaxColumns>;
using Wide = Capacity<kMaxColumns>;
static_assert(Legacy::kRowBytes == 16384 &&
                  Legacy::kMaximumAffineUbBytes == 65600,
              "existing cached shapes must retain their original UB capacity");
static_assert(Wide::kRowBytes == 32768 &&
                  Wide::kMaximumAffineUbBytes == 131136,
              "unexpected wide cached RMSNorm UB capacity");

}  // namespace ascify950_rmsnorm_cached_capacity_v1

#endif  // ASCIFY950_RMSNORM_CACHED_CAPACITY_V1_HPP_
