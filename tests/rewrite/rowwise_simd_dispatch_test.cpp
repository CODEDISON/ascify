#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>
#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace simd = ascify::target::dav_c310::rowwise_simd_v1;

namespace {

int softmax_calls = 0;
int cached_calls = 0;
int row_batch_calls = 0;
int layernorm_calls = 0;
aclError softmax_status = ACL_SUCCESS;
aclError cached_status = ACL_SUCCESS;
aclError row_batch_status = ACL_SUCCESS;
aclError layernorm_status = ACL_SUCCESS;

template<typename T>
T* Address(uintptr_t value) {
  return reinterpret_cast<T*>(value);
}

struct Load {
  using ascify_target_direct_load_tag = void;
  using ascify_target_adapter_owner_type = Load;
  using ascify_target_storage_type = half;
  using ascify_target_compute_type = float;

  const half* data;
  int64_t stride;

  const half* ascify_target_data() const { return data; }
  int64_t ascify_target_row_stride() const { return stride; }
};

struct PlainStore {
  using ascify_target_direct_store_tag = void;
  using ascify_target_adapter_owner_type = PlainStore;
  using ascify_target_storage_type = half;
  using ascify_target_compute_type = float;
  static constexpr bool ascify_target_store_is_affine = false;

  half* data;
  int64_t stride;

  half* ascify_target_data() const { return data; }
  int64_t ascify_target_row_stride() const { return stride; }
};

struct AffineStore {
  using ascify_target_direct_store_tag = void;
  using ascify_target_adapter_owner_type = AffineStore;
  using ascify_target_storage_type = half;
  using ascify_target_compute_type = float;
  static constexpr bool ascify_target_store_is_affine = true;

  half* data;
  const half* weight;
  int64_t stride;

  half* ascify_target_data() const { return data; }
  const half* ascify_target_weight() const { return weight; }
  int64_t ascify_target_row_stride() const { return stride; }
};

}  // namespace

extern "C" aclError ascify950_softmax_reg_recompute_launch_v1(
    aclrtStream stream, const half* input, half* output, int64_t rows,
    int columns, int block_threads, int grid_cap) {
  if (!simd::IsSoftmaxSimdDomain(
          stream, input, output, rows, columns, block_threads, grid_cap)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  ++softmax_calls;
  return softmax_status;
}

extern "C" aclError ascify950_rmsnorm_reg_cached_launch_v1(
    aclrtStream stream, const half* input, half* output, const half* weight,
    float* inverse_rms, int64_t rows, int64_t columns, double epsilon) {
  if (!simd::IsRmsNormCachedSimdDomain(
          stream, input, output, weight, inverse_rms, rows, columns, epsilon,
          0, 0)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  ++cached_calls;
  return cached_status;
}

extern "C" aclError ascify950_rmsnorm_reg_plain_rowbatch_launch_v1(
    aclrtStream stream, const half* input, half* output, float* inverse_rms,
    int64_t rows, int64_t columns, double epsilon) {
  if (!simd::IsRmsNormPlainRowBatchSimdDomain(
          stream, input, output, nullptr, inverse_rms, rows, columns, epsilon,
          0, 0)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  ++row_batch_calls;
  return row_batch_status;
}

extern "C" aclError ascify950_layernorm_reg_cached_launch_v1(
    aclrtStream stream, const half* input, half* output, float* mean,
    float* inverse_variance, int64_t rows, int64_t columns, double epsilon) {
  if (!simd::IsLayerNormCachedHybridDomain(
          stream, input, output, mean, inverse_variance, rows, columns,
          epsilon, 0, 0)) {
    return ACL_ERROR_RT_PARAM_INVALID;
  }
  ++layernorm_calls;
  return layernorm_status;
}

int main() {
  aclrtStream stream = Address<void>(0x1000U);
  const half* input = Address<half>(0x100000U);
  half* output = Address<half>(0x200000U);
  float* inverse_rms = Address<float>(0x300000U);
  const half* weight = Address<half>(0x400000U);
  float* mean = Address<float>(0x500000U);
  float* inverse_variance = Address<float>(0x600000U);

  softmax_status = 73;
  static_assert(std::is_same<
                    simd::SimdTryResult,
                    ascify::target::dav_c310::SimdTryResult>::value,
                "the unversioned namespace must export SimdTryResult");
  static_assert(std::is_same<
                    simd::HybridTryResult,
                    ascify::target::dav_c310::HybridTryResult>::value,
                "the unversioned namespace must export HybridTryResult");

  ascify::target::dav_c310::SimdTryResult result =
      simd::TrySoftmaxSimd(stream, input, output, 8, 4096);
  assert(result.handled && result.status == 73 && softmax_calls == 1);
  result = simd::TrySoftmaxSimd(stream, input, output, 8, 2048);
  assert(!result.handled && result.status == ACL_SUCCESS);
  assert(softmax_calls == 1);

  // Direct ABI calls use the same v1 domain predicates as the facade.
  assert(ascify950_softmax_reg_recompute_launch_v1(
             stream, input, Address<half>(0x100002U), 8, 4096, 0, 0)
         == ACL_ERROR_RT_PARAM_INVALID);
  assert(softmax_calls == 1);

  row_batch_status = 91;
  result = simd::TryRmsNormSimd(
      stream, input, output, nullptr, 8, 1536, 1.0e-5, inverse_rms);
  assert(result.handled && result.status == 91);
  assert(row_batch_calls == 1 && cached_calls == 0);

  cached_status = 92;
  result = simd::TryRmsNormSimd(
      stream, input, output, nullptr, 8, 1544, 1.0e-5, inverse_rms);
  assert(result.handled && result.status == 92);
  assert(row_batch_calls == 1 && cached_calls == 1);

  result = simd::TryRmsNormSimd(
      stream, input, output, weight, 8, 1536, 1.0e-5, inverse_rms);
  assert(result.handled && result.status == 92);
  assert(row_batch_calls == 1 && cached_calls == 2);

  assert(ascify950_rmsnorm_reg_cached_launch_v1(
             stream, input, Address<half>(0x100000U), weight, inverse_rms,
             8, 1536, 1.0e-5)
         == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify950_rmsnorm_reg_cached_launch_v1(
             stream, input, output, weight, Address<float>(0x100000U),
             8, 1536, 1.0e-5)
         == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify950_rmsnorm_reg_cached_launch_v1(
             stream, input, output, Address<half>(0x100000U), inverse_rms,
             8, 1536, 1.0e-5)
         == ACL_ERROR_RT_PARAM_INVALID);
  assert(cached_calls == 2);

  assert(ascify950_rmsnorm_reg_plain_rowbatch_launch_v1(
             stream, input, output, Address<float>(0x100000U),
             8, 1536, 1.0e-5)
         == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify950_rmsnorm_reg_plain_rowbatch_launch_v1(
             stream, input, Address<half>(0x100010U), inverse_rms,
             8, 1536, 1.0e-5)
         == ACL_ERROR_RT_PARAM_INVALID);
  assert(ascify950_rmsnorm_reg_plain_rowbatch_launch_v1(
             stream, input, output, inverse_rms,
             std::numeric_limits<int64_t>::max(), 16, 1.0e-5)
         == ACL_ERROR_RT_PARAM_INVALID);
  assert(row_batch_calls == 1);

  // A selected ABI error remains handled; dispatch does not try another SIMD
  // route and therefore cannot hide or duplicate the failed launch.
  row_batch_status = ACL_ERROR_RT_PARAM_INVALID;
  result = simd::TryRmsNormSimd(
      stream, input, output, nullptr, 8, 1536, 1.0e-5, inverse_rms);
  assert(result.handled && result.status == ACL_ERROR_RT_PARAM_INVALID);
  assert(row_batch_calls == 2 && cached_calls == 2);

  result = simd::TryRmsNormSimd(
      stream, input, Address<half>(0x100010U), nullptr,
      8, 1536, 1.0e-5, inverse_rms);
  assert(!result.handled && result.status == ACL_SUCCESS);
  assert(row_batch_calls == 2 && cached_calls == 2);

  float compute = 0.0F;
  const Load softmax_load{input, 4096};
  const PlainStore softmax_store{output, 4096};
  softmax_status = ACL_SUCCESS;
  result = simd::RowwiseHybridFacadeV1::TrySoftmaxHybrid(
      stream, softmax_load, softmax_store, 8, 4096, &compute);
  assert(result.handled && result.status == ACL_SUCCESS);
  assert(softmax_calls == 2);

  const PlainStore bad_stride{output, 4095};
  result = simd::RowwiseHybridFacadeV1::TrySoftmaxHybrid(
      stream, softmax_load, bad_stride, 8, 4096, &compute);
  assert(!result.handled && softmax_calls == 2);

  const Load rms_load{input, 1536};
  const PlainStore rms_store{output, 1536};
  row_batch_status = ACL_SUCCESS;
  result = simd::RowwiseHybridFacadeV1::TryRmsNormHybrid(
      stream, rms_load, rms_store, 8, 1536, 1.0e-5,
      inverse_rms, &compute);
  assert(result.handled && result.status == ACL_SUCCESS);
  assert(row_batch_calls == 3 && cached_calls == 2);

  const AffineStore affine_store{output, weight, 1536};
  result = simd::RowwiseHybridFacadeV1::TryRmsNormHybrid(
      stream, rms_load, affine_store, 8, 1536, 1.0e-5,
      inverse_rms, &compute);
  assert(result.handled && result.status == 92);
  assert(row_batch_calls == 3 && cached_calls == 3);

  static_assert(
      simd::IsRegisteredHybridRecipe<simd::LayerNormHybridRecipeV1>::value,
      "LayerNorm must use the common recipe registry");
  const Load layernorm_load{input, 1536};
  const PlainStore layernorm_store{output, 1536};
  layernorm_status = 93;
  result = simd::RowwiseHybridFacadeV1::TryLayerNormHybrid(
      stream, layernorm_load, layernorm_store, 8, 1536, 1.0e-5,
      mean, inverse_variance, &compute);
  assert(result.handled && result.status == 93);
  assert(layernorm_calls == 1);

  result = simd::RowwiseHybridFacadeV1::TryLayerNormHybrid(
      stream, layernorm_load, layernorm_store, 8, 1537, 1.0e-5,
      mean, inverse_variance, &compute);
  assert(!result.handled && layernorm_calls == 1);

  // The widened column domain must reach the existing cached ABI through
  // both typed adapters, including a partial vector tail. Other contracts
  // still reject before any ABI call.
  const int row_batch_calls_before_wide = row_batch_calls;
  for (int64_t columns : {8192, 8200, 12288, 13312, 16376, 16384}) {
    const Load wide_load{input, columns};
    const PlainStore wide_plain{output, columns};
    const AffineStore wide_affine{output, weight, columns};
    int calls_before = cached_calls;
    result = simd::RowwiseHybridFacadeV1::TryRmsNormHybrid(
        stream, wide_load, wide_plain, 8, columns, 1.0e-5,
        inverse_rms, &compute);
    assert(result.handled && result.status == cached_status);
    ++calls_before;
    assert(cached_calls == calls_before);
    result = simd::RowwiseHybridFacadeV1::TryRmsNormHybrid(
        stream, wide_load, wide_affine, 8, columns, 1.0e-5,
        inverse_rms, &compute);
    assert(result.handled && result.status == cached_status);
    ++calls_before;
    assert(cached_calls == calls_before);

    const PlainStore wide_bad_stride{output, columns - 1};
    result = simd::RowwiseHybridFacadeV1::TryRmsNormHybrid(
        stream, wide_load, wide_bad_stride, 8, columns, 1.0e-5,
        inverse_rms, &compute);
    assert(!result.handled && cached_calls == calls_before);
    const AffineStore wide_missing_weight{output, nullptr, columns};
    result = simd::RowwiseHybridFacadeV1::TryRmsNormHybrid(
        stream, wide_load, wide_missing_weight, 8, columns, 1.0e-5,
        inverse_rms, &compute);
    assert(!result.handled && cached_calls == calls_before);
  }
  for (int64_t columns : {8193, 16383, 16385, 16392, 32768}) {
    const int calls_before = cached_calls;
    result = simd::TryRmsNormHybrid(
        stream, input, output, weight, 8, columns, 1.0e-5, inverse_rms);
    assert(!result.handled && cached_calls == calls_before);
  }
  assert(row_batch_calls == row_batch_calls_before_wide);

  return 0;
}
