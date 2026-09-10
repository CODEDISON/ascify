#ifndef ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_SELECTORS_V1_HPP_
#define ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_SELECTORS_V1_HPP_

#include <cmath>
#include <cstdint>
#include <limits>

namespace ascify {
namespace target {
namespace dav_c310 {
namespace rowwise_simd_v1 {

constexpr int kSelectorAbiVersion = 1;
constexpr uintptr_t kHalfStorageBytes = sizeof(uint16_t);
constexpr int64_t kSoftmaxMinimumColumns = 4096;
constexpr int64_t kSoftmaxMaximumColumns = 256000;
constexpr int64_t kRmsNormCachedMaximumColumns = 16384;
constexpr int64_t kRmsNormPlainRowBatchMaximumColumns = 3072;
constexpr int64_t kLayerNormCachedMaximumColumns = 8192;

enum class RmsNormRoute {
  kSimt,
  kPlainRowBatch,
  kCached,
};

struct ByteSpan {
  uintptr_t begin;
  uintptr_t end;
};

struct RmsNormSpans {
  ByteSpan input;
  ByteSpan output;
  ByteSpan inverse_rms;
  ByteSpan weight;
  bool has_weight;
};

struct LayerNormSpans {
  ByteSpan input;
  ByteSpan output;
  ByteSpan mean;
  ByteSpan inverse_variance;
};

inline bool IsAligned(const void* pointer, uintptr_t alignment) {
  return pointer != nullptr && alignment != 0U
      && (alignment & (alignment - 1U)) == 0U
      && (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1U)) == 0U;
}

inline bool HasValidContiguousExtent(int64_t rows, int64_t columns) {
  return rows > 0 && columns > 0
      && rows <= std::numeric_limits<int64_t>::max() / columns;
}

inline bool TryMakeByteSpan(
    const void* pointer, uint64_t element_count, uintptr_t element_bytes,
    ByteSpan* span) {
  if (pointer == nullptr || span == nullptr || element_count == 0U
      || element_bytes == 0U) {
    return false;
  }
  const uintptr_t maximum = std::numeric_limits<uintptr_t>::max();
  if (element_count > maximum / element_bytes) {
    return false;
  }
  const uintptr_t byte_count =
      static_cast<uintptr_t>(element_count) * element_bytes;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(pointer);
  if (byte_count > maximum - begin) {
    return false;
  }
  span->begin = begin;
  span->end = begin + byte_count;
  return true;
}

inline bool TryMakeMatrixByteSpan(
    const void* pointer, int64_t rows, int64_t columns,
    uintptr_t element_bytes, ByteSpan* span) {
  if (!HasValidContiguousExtent(rows, columns)) {
    return false;
  }
  const uint64_t element_count =
      static_cast<uint64_t>(rows) * static_cast<uint64_t>(columns);
  return TryMakeByteSpan(pointer, element_count, element_bytes, span);
}

inline bool SpansOverlap(const ByteSpan& left, const ByteSpan& right) {
  return left.begin < right.end && right.begin < left.end;
}

inline bool SpansAreExact(const ByteSpan& left, const ByteSpan& right) {
  return left.begin == right.begin && left.end == right.end;
}

inline bool SpansAreDisjointOrExact(
    const ByteSpan& left, const ByteSpan& right) {
  return !SpansOverlap(left, right) || SpansAreExact(left, right);
}

inline bool IsSoftmaxSimdDomain(
    const void* stream, const void* input, const void* output, int64_t rows,
    int64_t columns, int block_threads = 0, int grid_cap = 0) {
  ByteSpan input_span{};
  ByteSpan output_span{};
  return stream != nullptr
      && columns >= kSoftmaxMinimumColumns
      && columns <= kSoftmaxMaximumColumns
      && block_threads == 0 && grid_cap == 0
      && TryMakeMatrixByteSpan(
             input, rows, columns, kHalfStorageBytes, &input_span)
      && TryMakeMatrixByteSpan(
             output, rows, columns, kHalfStorageBytes, &output_span)
      && SpansAreDisjointOrExact(input_span, output_span);
}

inline bool TryMakeRmsNormSpans(
    const void* input, const void* output, const void* weight,
    const void* inverse_rms, int64_t rows, int64_t columns,
    RmsNormSpans* spans) {
  if (spans == nullptr
      || !TryMakeMatrixByteSpan(
             input, rows, columns, kHalfStorageBytes, &spans->input)
      || !TryMakeMatrixByteSpan(
             output, rows, columns, kHalfStorageBytes, &spans->output)
      || !TryMakeByteSpan(
             inverse_rms, static_cast<uint64_t>(rows), sizeof(float),
             &spans->inverse_rms)) {
    return false;
  }
  spans->has_weight = weight != nullptr;
  spans->weight = ByteSpan{0U, 0U};
  return !spans->has_weight
      || TryMakeByteSpan(
             weight, static_cast<uint64_t>(columns), kHalfStorageBytes,
             &spans->weight);
}

inline bool HasNoRmsNormAuxAliasing(const RmsNormSpans& spans) {
  if (SpansOverlap(spans.inverse_rms, spans.input)
      || SpansOverlap(spans.inverse_rms, spans.output)) {
    return false;
  }
  return !spans.has_weight
      || (!SpansOverlap(spans.weight, spans.input)
          && !SpansOverlap(spans.weight, spans.output)
          && !SpansOverlap(spans.weight, spans.inverse_rms));
}

inline bool HasCommonRmsNormSimdContract(
    const void* stream, const void* input, const void* output,
    const void* weight, const void* inverse_rms, int64_t rows,
    int64_t columns, double epsilon, int block_threads, int grid_cap,
    RmsNormSpans* spans) {
  if (stream == nullptr || columns <= 0
      || block_threads != 0 || grid_cap != 0
      || !(epsilon > 0.0) || !std::isfinite(epsilon)
      || !IsAligned(input, 16U) || !IsAligned(output, 16U)
      || !IsAligned(inverse_rms, 4U)
      || (weight != nullptr && !IsAligned(weight, 16U))
      || !TryMakeRmsNormSpans(
             input, output, weight, inverse_rms, rows, columns, spans)
      || !HasNoRmsNormAuxAliasing(*spans)) {
    return false;
  }
  const float narrow_epsilon = static_cast<float>(epsilon);
  return narrow_epsilon > 0.0F && std::isfinite(narrow_epsilon);
}

inline bool IsRmsNormCachedSimdDomain(
    const void* stream, const void* input, const void* output,
    const void* weight, const void* inverse_rms, int64_t rows,
    int64_t columns, double epsilon, int block_threads = 0,
    int grid_cap = 0) {
  RmsNormSpans spans{};
  return columns <= kRmsNormCachedMaximumColumns && columns % 8 == 0
      && HasCommonRmsNormSimdContract(
             stream, input, output, weight, inverse_rms, rows, columns,
             epsilon, block_threads, grid_cap, &spans)
      && !SpansOverlap(spans.input, spans.output);
}

inline bool IsRmsNormPlainRowBatchSimdDomain(
    const void* stream, const void* input, const void* output,
    const void* weight, const void* inverse_rms, int64_t rows,
    int64_t columns, double epsilon, int block_threads = 0,
    int grid_cap = 0) {
  RmsNormSpans spans{};
  return weight == nullptr
      && columns <= kRmsNormPlainRowBatchMaximumColumns
      && columns % 16 == 0
      && HasCommonRmsNormSimdContract(
             stream, input, output, weight, inverse_rms, rows, columns,
             epsilon, block_threads, grid_cap, &spans)
      && SpansAreDisjointOrExact(spans.input, spans.output);
}

// The experimental recipe gives the safe plain row-batch domain ownership
// before the cached domain. Calls outside both proven domains remain SIMT.
inline RmsNormRoute SelectRmsNormRoute(
    const void* stream, const void* input, const void* output,
    const void* weight, const void* inverse_rms, int64_t rows,
    int64_t columns, double epsilon, int block_threads = 0,
    int grid_cap = 0) {
  if (IsRmsNormPlainRowBatchSimdDomain(
          stream, input, output, weight, inverse_rms, rows, columns, epsilon,
          block_threads, grid_cap)) {
    return RmsNormRoute::kPlainRowBatch;
  }
  if (IsRmsNormCachedSimdDomain(
          stream, input, output, weight, inverse_rms, rows, columns, epsilon,
          block_threads, grid_cap)) {
    return RmsNormRoute::kCached;
  }
  return RmsNormRoute::kSimt;
}

inline bool TryMakeLayerNormSpans(
    const void* input, const void* output, const void* mean,
    const void* inverse_variance, int64_t rows, int64_t columns,
    LayerNormSpans* spans) {
  return spans != nullptr
      && TryMakeMatrixByteSpan(
             input, rows, columns, kHalfStorageBytes, &spans->input)
      && TryMakeMatrixByteSpan(
             output, rows, columns, kHalfStorageBytes, &spans->output)
      && TryMakeByteSpan(
             mean, static_cast<uint64_t>(rows), sizeof(float), &spans->mean)
      && TryMakeByteSpan(
             inverse_variance, static_cast<uint64_t>(rows), sizeof(float),
             &spans->inverse_variance);
}

inline bool IsLayerNormCachedHybridDomain(
    const void* stream, const void* input, const void* output,
    const void* mean, const void* inverse_variance, int64_t rows,
    int64_t columns, double epsilon, int block_threads = 0,
    int grid_cap = 0) {
  LayerNormSpans spans{};
  if (stream == nullptr || columns <= 0
      || columns > kLayerNormCachedMaximumColumns || columns % 8 != 0
      || block_threads != 0 || grid_cap != 0
      || !(epsilon > 0.0) || !std::isfinite(epsilon)
      || !IsAligned(input, 16U) || !IsAligned(output, 16U)
      || !IsAligned(mean, 4U) || !IsAligned(inverse_variance, 4U)
      || !TryMakeLayerNormSpans(
             input, output, mean, inverse_variance, rows, columns, &spans)
      || !SpansAreDisjointOrExact(spans.input, spans.output)
      || SpansOverlap(spans.mean, spans.input)
      || SpansOverlap(spans.mean, spans.output)
      || SpansOverlap(spans.inverse_variance, spans.input)
      || SpansOverlap(spans.inverse_variance, spans.output)
      || SpansOverlap(spans.mean, spans.inverse_variance)) {
    return false;
  }
  const float narrow_epsilon = static_cast<float>(epsilon);
  return narrow_epsilon > 0.0F && std::isfinite(narrow_epsilon);
}

}  // namespace rowwise_simd_v1
}  // namespace dav_c310
}  // namespace target
}  // namespace ascify

#endif  // ASCIFY_TARGET_DAV_C310_ROWWISE_SIMD_SELECTORS_V1_HPP_
