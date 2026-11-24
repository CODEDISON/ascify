// Versioned dav-c310 row-wise hybrid recipes. A selector miss is explicitly
// not handled; once a mixed SIMD+SIMT route owns a call, its status is final.
#ifndef ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_
#define ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_

#include <ascify/target/dav_c310/rowwise_simd_abi.h>
#include <ascify/target/dav_c310/rowwise_hybrid_registry_v1.hpp>
#include <ascify/target/dav_c310/rowwise_simd_selectors_v1.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>
#if defined(__APPLE__) || (defined(__linux__) && !defined(__PIC__))
#include <dlfcn.h>
#endif

namespace ascify {
namespace target {
namespace dav_c310 {
namespace rowwise_simd_v1 {

static_assert(ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION == 1,
              "rowwise SIMD recipes require ABI version 1");
static_assert(kSelectorAbiVersion == 1,
              "rowwise SIMD recipes require selector version 1");

struct SimdTryResult {
  bool handled;
  aclError status;
};

using HybridTryResult = SimdTryResult;

inline SimdTryResult NotHandled() {
  return {false, ACL_SUCCESS};
}

inline SimdTryResult TrySoftmaxSimd(
    aclrtStream stream, const half* input, half* output, int64_t rows,
    int64_t columns) {
  if (!IsSoftmaxSimdDomain(
          stream, input, output, rows, columns, 0, 0)) {
    return NotHandled();
  }
  if (columns < kSoftmaxMinimumColumns) {
#if defined(__APPLE__) || (defined(__linux__) && !defined(__PIC__))
    // A non-PIC ELF linker may resolve an absent weak function permanently
    // to zero when the caller is built against an early v1 library.
    static const auto supports_short =
        reinterpret_cast<decltype(&ascify950_softmax_supports_short_v1)>(
            ::dlsym(RTLD_DEFAULT, "ascify950_softmax_supports_short_v1"));
#elif defined(__ELF__) && (defined(__GNUC__) || defined(__clang__))
    const auto supports_short = &ascify950_softmax_supports_short_v1;
#else
    constexpr decltype(&ascify950_softmax_supports_short_v1) supports_short = nullptr;
#endif
    if (!supports_short ||
        !supports_short(stream, input, output, rows, static_cast<int>(columns))) {
      return NotHandled();
    }
  }
  return {true, ascify950_softmax_reg_recompute_launch_v1(
                    stream, input, output, rows,
                    static_cast<int>(columns), 0, 0)};
}

inline SimdTryResult TryRmsNormSimd(
    aclrtStream stream, const half* input, half* output, const half* weight,
    int64_t rows, int64_t columns, double epsilon, float* inverse_rms) {
  switch (SelectRmsNormRoute(
      stream, input, output, weight, inverse_rms, rows, columns, epsilon,
      0, 0)) {
    case RmsNormRoute::kPlainRowBatch:
      return {true, ascify950_rmsnorm_reg_plain_rowbatch_launch_v1(
                        stream, input, output, inverse_rms, rows, columns,
                        epsilon)};
    case RmsNormRoute::kCached:
      return {true, ascify950_rmsnorm_reg_cached_launch_v1(
                        stream, input, output, weight, inverse_rms, rows,
                        columns, epsilon)};
    case RmsNormRoute::kSimt:
      return NotHandled();
  }
  return NotHandled();
}

inline SimdTryResult TryLayerNormSimd(
    aclrtStream stream, const half* input, half* output, int64_t rows,
    int64_t columns, double epsilon, float* mean, float* inverse_variance) {
  if (!IsLayerNormCachedHybridDomain(
          stream, input, output, mean, inverse_variance, rows, columns,
          epsilon, 0, 0)) {
    return NotHandled();
  }
  return {true, ascify950_layernorm_reg_cached_launch_v1(
                    stream, input, output, mean, inverse_variance, rows,
                    columns, epsilon)};
}

inline HybridTryResult TrySoftmaxHybrid(
    aclrtStream stream, const half* input, half* output, int64_t rows,
    int64_t columns) {
  return TrySoftmaxSimd(stream, input, output, rows, columns);
}

inline HybridTryResult TryRmsNormHybrid(
    aclrtStream stream, const half* input, half* output, const half* weight,
    int64_t rows, int64_t columns, double epsilon, float* inverse_rms) {
  return TryRmsNormSimd(
      stream, input, output, weight, rows, columns, epsilon, inverse_rms);
}

inline HybridTryResult TryLayerNormHybrid(
    aclrtStream stream, const half* input, half* output, int64_t rows,
    int64_t columns, double epsilon, float* mean, float* inverse_variance) {
  return TryLayerNormSimd(
      stream, input, output, rows, columns, epsilon, mean, inverse_variance);
}

namespace adapter_detail {

template<typename...>
using VoidT = void;

template<typename Pointer>
struct IsHalfInputPointer
    : std::integral_constant<
          bool,
          std::is_convertible<
              typename std::remove_reference<Pointer>::type,
              const half*>::value> {};

template<typename Pointer>
struct IsHalfOutputPointer
    : std::integral_constant<
          bool,
          std::is_convertible<
              typename std::remove_reference<Pointer>::type,
              half*>::value> {};

template<typename T, typename = void>
struct HasMarkedLoadAccess : std::false_type {};

template<typename T>
struct HasMarkedLoadAccess<
    T, VoidT<typename T::ascify_target_direct_load_tag,
             typename T::ascify_target_adapter_owner_type,
             typename T::ascify_target_storage_type,
             typename T::ascify_target_compute_type,
             decltype(std::declval<const T&>().ascify_target_data()),
             decltype(std::declval<const T&>().ascify_target_row_stride())>>
    : std::integral_constant<
          bool,
          std::is_same<T, typename T::ascify_target_adapter_owner_type>::value
              && std::is_same<
                     typename std::remove_cv<
                         typename T::ascify_target_storage_type>::type,
                     half>::value
              && std::is_same<
                     typename std::remove_cv<
                         typename T::ascify_target_compute_type>::type,
                     float>::value
              && IsHalfInputPointer<decltype(
                     std::declval<const T&>().ascify_target_data())>::value> {};

template<typename T, typename = void>
struct HasMarkedStoreAccess : std::false_type {};

template<typename T>
struct HasMarkedStoreAccess<
    T, VoidT<typename T::ascify_target_direct_store_tag,
             typename T::ascify_target_adapter_owner_type,
             typename T::ascify_target_storage_type,
             typename T::ascify_target_compute_type,
             typename std::integral_constant<
                 bool, static_cast<bool>(
                           T::ascify_target_store_is_affine)>::type,
             decltype(std::declval<const T&>().ascify_target_data()),
             decltype(std::declval<const T&>().ascify_target_row_stride())>>
    : std::integral_constant<
          bool,
          std::is_same<T, typename T::ascify_target_adapter_owner_type>::value
              && std::is_same<
                     typename std::remove_cv<
                         typename T::ascify_target_storage_type>::type,
                     half>::value
              && std::is_same<
                     typename std::remove_cv<
                         typename T::ascify_target_compute_type>::type,
                     float>::value
              && IsHalfOutputPointer<decltype(
                     std::declval<const T&>().ascify_target_data())>::value> {};

template<typename T, typename = void>
struct HasMarkedAffineStoreAccess : std::false_type {};

template<typename T>
struct HasMarkedAffineStoreAccess<
    T, VoidT<typename T::ascify_target_direct_store_tag,
             typename T::ascify_target_adapter_owner_type,
             decltype(std::declval<const T&>().ascify_target_weight())>>
    : std::integral_constant<
          bool,
          std::is_same<T, typename T::ascify_target_adapter_owner_type>::value
              && IsHalfInputPointer<decltype(
                     std::declval<const T&>().ascify_target_weight())>::value> {};

template<typename T, typename Enable = void>
struct LoadAccess {
  static constexpr bool kValid = false;
};

template<typename T>
struct LoadAccess<
    T, typename std::enable_if<HasMarkedLoadAccess<T>::value>::type> {
  static constexpr bool kValid = true;
  static const half* Data(const T& value) {
    return value.ascify_target_data();
  }
  static int64_t Stride(const T& value) {
    return static_cast<int64_t>(value.ascify_target_row_stride());
  }
};

template<typename T, typename Enable = void>
struct StoreAccess {
  static constexpr bool kValid = false;
  static constexpr bool kAffine = false;
};

template<typename T>
struct StoreAccess<
    T, typename std::enable_if<HasMarkedStoreAccess<T>::value>::type> {
  static constexpr bool kAffine =
      static_cast<bool>(T::ascify_target_store_is_affine);
  static constexpr bool kValid =
      !kAffine || HasMarkedAffineStoreAccess<T>::value;
  static half* Data(const T& value) { return value.ascify_target_data(); }
  static int64_t Stride(const T& value) {
    return static_cast<int64_t>(value.ascify_target_row_stride());
  }
  static const half* Weight(const T& value) {
    return WeightImpl(
        value,
        std::integral_constant<
            bool, kAffine && HasMarkedAffineStoreAccess<T>::value>{});
  }

 private:
  static const half* WeightImpl(const T& value, std::true_type) {
    return value.ascify_target_weight();
  }
  static const half* WeightImpl(const T&, std::false_type) { return nullptr; }
};

template<typename T>
using Bare = typename std::remove_cv<
    typename std::remove_reference<T>::type>::type;

template<typename T, typename = void>
struct MarkedComputeType {
  using type = void;
};

template<typename T>
struct MarkedComputeType<T, VoidT<typename T::ascify_target_compute_type>> {
  using type = typename T::ascify_target_compute_type;
};

template<typename Compute, typename Load, typename Store>
struct HasSupportedComputeContract
    : std::integral_constant<
          bool,
          std::is_same<Compute, float>::value
              && std::is_same<
                     Compute,
                     typename MarkedComputeType<Bare<Load>>::type>::value
              && std::is_same<
                     Compute,
                     typename MarkedComputeType<Bare<Store>>::type>::value> {};

}  // namespace adapter_detail

template<typename Load, typename Store, typename Compute>
inline SimdTryResult TrySoftmaxSimd(
    aclrtStream stream, const Load& load, const Store& store, int64_t rows,
    int64_t columns, Compute*) {
  using LoadAccess = adapter_detail::LoadAccess<
      adapter_detail::Bare<Load>>;
  using StoreAccess = adapter_detail::StoreAccess<
      adapter_detail::Bare<Store>>;
  if constexpr (!LoadAccess::kValid || !StoreAccess::kValid
                || StoreAccess::kAffine
                || !adapter_detail::HasSupportedComputeContract<
                       Compute, Load, Store>::value) {
    return NotHandled();
  } else {
    if (LoadAccess::Stride(load) != columns
        || StoreAccess::Stride(store) != columns) {
      return NotHandled();
    }
    return TrySoftmaxSimd(stream, LoadAccess::Data(load),
                          StoreAccess::Data(store), rows, columns);
  }
}

template<typename Load, typename Store, typename Inverse, typename Compute>
inline SimdTryResult TryRmsNormSimd(
    aclrtStream stream, const Load& load, const Store& store, int64_t rows,
    int64_t columns, double epsilon, Inverse* inverse_rms, Compute*) {
  using LoadAccess = adapter_detail::LoadAccess<
      adapter_detail::Bare<Load>>;
  using StoreAccess = adapter_detail::StoreAccess<
      adapter_detail::Bare<Store>>;
  if constexpr (!LoadAccess::kValid || !StoreAccess::kValid
                || !std::is_same<Inverse, float>::value
                || !adapter_detail::HasSupportedComputeContract<
                       Compute, Load, Store>::value) {
    return NotHandled();
  } else {
    if (LoadAccess::Stride(load) != columns
        || StoreAccess::Stride(store) != columns) {
      return NotHandled();
    }
    const half* weight = StoreAccess::Weight(store);
    if constexpr (StoreAccess::kAffine) {
      if (weight == nullptr) {
        return NotHandled();
      }
    }
    return TryRmsNormSimd(
        stream, LoadAccess::Data(load), StoreAccess::Data(store), weight,
        rows, columns, epsilon, inverse_rms);
  }
}

template<typename Load, typename Store, typename Compute>
inline HybridTryResult TrySoftmaxHybrid(
    aclrtStream stream, const Load& load, const Store& store, int64_t rows,
    int64_t columns, Compute* compute) {
  return TrySoftmaxSimd(stream, load, store, rows, columns, compute);
}

template<typename Load, typename Store, typename Inverse, typename Compute>
inline HybridTryResult TryRmsNormHybrid(
    aclrtStream stream, const Load& load, const Store& store, int64_t rows,
    int64_t columns, double epsilon, Inverse* inverse_rms,
    Compute* compute) {
  return TryRmsNormSimd(
      stream, load, store, rows, columns, epsilon, inverse_rms, compute);
}

template<typename Load, typename Store, typename Mean, typename Inverse,
         typename Compute>
inline SimdTryResult TryLayerNormSimd(
    aclrtStream stream, const Load& load, const Store& store, int64_t rows,
    int64_t columns, double epsilon, Mean* mean, Inverse* inverse_variance,
    Compute*) {
  using LoadAccess = adapter_detail::LoadAccess<
      adapter_detail::Bare<Load>>;
  using StoreAccess = adapter_detail::StoreAccess<
      adapter_detail::Bare<Store>>;
  if constexpr (!LoadAccess::kValid || !StoreAccess::kValid
                || StoreAccess::kAffine
                || !std::is_same<Mean, float>::value
                || !std::is_same<Inverse, float>::value
                || !adapter_detail::HasSupportedComputeContract<
                       Compute, Load, Store>::value) {
    return NotHandled();
  } else {
    if (LoadAccess::Stride(load) != columns
        || StoreAccess::Stride(store) != columns) {
      return NotHandled();
    }
    return TryLayerNormSimd(
        stream, LoadAccess::Data(load), StoreAccess::Data(store), rows,
        columns, epsilon, mean, inverse_variance);
  }
}

template<typename Load, typename Store, typename Mean, typename Inverse,
         typename Compute>
inline HybridTryResult TryLayerNormHybrid(
    aclrtStream stream, const Load& load, const Store& store, int64_t rows,
    int64_t columns, double epsilon, Mean* mean, Inverse* inverse_variance,
    Compute* compute) {
  return TryLayerNormSimd(
      stream, load, store, rows, columns, epsilon, mean, inverse_variance,
      compute);
}

struct SoftmaxHybridRecipeV1 final {
  inline static constexpr bool kRegistered = true;
  inline static constexpr HybridStageDescriptor kStages[] = {
      {HybridEngine::kMte, HybridStageKind::kLoad},
      {HybridEngine::kSimd, HybridStageKind::kReduceMaximum},
      {HybridEngine::kSimd, HybridStageKind::kExponentiate},
      {HybridEngine::kSimd, HybridStageKind::kReduceSum},
      {HybridEngine::kSimd, HybridStageKind::kNormalize},
      {HybridEngine::kSimt, HybridStageKind::kOutputAdapt},
      {HybridEngine::kMte, HybridStageKind::kStore},
  };

  static constexpr HybridRecipeDescriptor Descriptor() {
    return {"softmax.fp16-fp32.hybrid.v1", kStages,
            sizeof(kStages) / sizeof(kStages[0])};
  }

  template<typename... Arguments>
  static HybridTryResult Try(Arguments&&... arguments) {
    return rowwise_simd_v1::TrySoftmaxHybrid(
        std::forward<Arguments>(arguments)...);
  }
};

struct RmsNormHybridRecipeV1 final {
  inline static constexpr bool kRegistered = true;
  inline static constexpr HybridStageDescriptor kStages[] = {
      {HybridEngine::kMte, HybridStageKind::kLoad},
      {HybridEngine::kSimd, HybridStageKind::kSquare},
      {HybridEngine::kSimd, HybridStageKind::kReduceSum},
      {HybridEngine::kSimd, HybridStageKind::kMean},
      {HybridEngine::kSimd, HybridStageKind::kReciprocalSqrt},
      {HybridEngine::kSimd, HybridStageKind::kNormalize},
      {HybridEngine::kSimd, HybridStageKind::kAffine},
      {HybridEngine::kSimt, HybridStageKind::kOutputAdapt},
      {HybridEngine::kMte, HybridStageKind::kStore},
  };

  static constexpr HybridRecipeDescriptor Descriptor() {
    return {"rmsnorm.fp16-fp32.hybrid.v1", kStages,
            sizeof(kStages) / sizeof(kStages[0])};
  }

  template<typename... Arguments>
  static HybridTryResult Try(Arguments&&... arguments) {
    return rowwise_simd_v1::TryRmsNormHybrid(
        std::forward<Arguments>(arguments)...);
  }
};

struct LayerNormHybridRecipeV1 final {
  inline static constexpr bool kRegistered = true;
  inline static constexpr HybridStageDescriptor kStages[] = {
      {HybridEngine::kMte, HybridStageKind::kLoad},
      {HybridEngine::kSimd, HybridStageKind::kMean},
      {HybridEngine::kSimd, HybridStageKind::kSquare},
      {HybridEngine::kSimd, HybridStageKind::kReduceSum},
      {HybridEngine::kSimd, HybridStageKind::kReciprocalSqrt},
      {HybridEngine::kSimt, HybridStageKind::kNormalize},
      {HybridEngine::kSimt, HybridStageKind::kOutputAdapt},
      {HybridEngine::kMte, HybridStageKind::kStore},
  };

  static constexpr HybridRecipeDescriptor Descriptor() {
    return {"layernorm.fp16-fp32.hybrid.v1", kStages,
            sizeof(kStages) / sizeof(kStages[0])};
  }

  template<typename... Arguments>
  static HybridTryResult Try(Arguments&&... arguments) {
    return rowwise_simd_v1::TryLayerNormHybrid(
        std::forward<Arguments>(arguments)...);
  }
};

// Generated code enters through this closed class rather than a namespace
// free function.  Input code cannot add an ordinary overload to the class;
// the frontend separately rejects declarations in this protected namespace,
// including explicit member-template specializations.
struct RowwiseSimdFacadeV1 final {
  template<typename Load, typename Store, typename Compute>
  static SimdTryResult TrySoftmaxSimd(
      aclrtStream stream, const Load& load, const Store& store, int64_t rows,
      int64_t columns, Compute* compute) {
    return rowwise_simd_v1::TrySoftmaxSimd(
        stream, load, store, rows, columns, compute);
  }

  template<typename Load, typename Store, typename Inverse, typename Compute>
  static SimdTryResult TryRmsNormSimd(
      aclrtStream stream, const Load& load, const Store& store, int64_t rows,
      int64_t columns, double epsilon, Inverse* inverse_rms,
      Compute* compute) {
    return rowwise_simd_v1::TryRmsNormSimd(
        stream, load, store, rows, columns, epsilon, inverse_rms, compute);
  }

  template<typename Load, typename Store, typename Mean, typename Inverse,
           typename Compute>
  static SimdTryResult TryLayerNormSimd(
      aclrtStream stream, const Load& load, const Store& store, int64_t rows,
      int64_t columns, double epsilon, Mean* mean, Inverse* inverse_variance,
      Compute* compute) {
    return rowwise_simd_v1::TryLayerNormSimd(
        stream, load, store, rows, columns, epsilon, mean, inverse_variance,
        compute);
  }
};

// This is the canonical generated-code facade.  The legacy Simd spelling is
// retained as an ABI-v1 source alias, but a selected call now enters an
// intra-kernel SIMD+SIMT implementation rather than an all-SIMD operator.
struct RowwiseHybridFacadeV1 final {
  template<typename Recipe, typename... Arguments>
  static HybridTryResult Try(Arguments&&... arguments) {
    return DispatchRegisteredHybrid<Recipe>(
        std::forward<Arguments>(arguments)...);
  }

  template<typename Load, typename Store, typename Compute>
  static HybridTryResult TrySoftmaxHybrid(
      aclrtStream stream, const Load& load, const Store& store, int64_t rows,
      int64_t columns, Compute* compute) {
    return Try<SoftmaxHybridRecipeV1>(
        stream, load, store, rows, columns, compute);
  }

  template<typename Load, typename Store, typename Inverse, typename Compute>
  static HybridTryResult TryRmsNormHybrid(
      aclrtStream stream, const Load& load, const Store& store, int64_t rows,
      int64_t columns, double epsilon, Inverse* inverse_rms,
      Compute* compute) {
    return Try<RmsNormHybridRecipeV1>(
        stream, load, store, rows, columns, epsilon, inverse_rms, compute);
  }

  template<typename Load, typename Store, typename Mean, typename Inverse,
           typename Compute>
  static HybridTryResult TryLayerNormHybrid(
      aclrtStream stream, const Load& load, const Store& store, int64_t rows,
      int64_t columns, double epsilon, Mean* mean, Inverse* inverse_variance,
      Compute* compute) {
    return Try<LayerNormHybridRecipeV1>(
        stream, load, store, rows, columns, epsilon, mean, inverse_variance,
        compute);
  }
};

}  // namespace rowwise_simd_v1

using rowwise_simd_v1::SimdTryResult;
using rowwise_simd_v1::HybridTryResult;

}  // namespace dav_c310
}  // namespace target
}  // namespace ascify

#endif  // ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_RECIPES_HPP_
