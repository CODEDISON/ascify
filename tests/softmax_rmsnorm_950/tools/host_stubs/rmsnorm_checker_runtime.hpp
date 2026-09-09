// Host test double for ACL and CUDA only. The checker and store adapter under
// test are compiled from their actual source bytes, unchanged.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#define __device__
#if defined(__ARM_FP16_FORMAT_IEEE)
// GCC 11 on DT exposes IEEE half storage as __fp16, while rejecting the
// _Float16 C++ spelling even though it defines __FLT16_* feature macros.
using half = __fp16;
#else
using half = _Float16;
#endif
static_assert(sizeof(half) == sizeof(uint16_t), "host test needs IEEE binary16 storage");
using aclError = int;
using aclrtStream = void*;
constexpr int ACL_SUCCESS = 0;
constexpr int ACL_ERROR_RT_PARAM_INVALID = 1;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int ACL_MEMCPY_DEVICE_TO_HOST = 2;
inline int aclrtMemcpy(void* dst, size_t capacity, const void* src, size_t size, int kind) {
  if (size > capacity) return 1;
  // Fail only after some successful chunk readback, to exercise partial results.
  static int readbacks = 0;
  if (kind == ACL_MEMCPY_DEVICE_TO_HOST && std::getenv("FAIL_READBACK") && ++readbacks == 12) {
    return 71;
  }
  std::memcpy(dst, src, size);
  return 0;
}
inline int aclrtMemset(void* dst, size_t capacity, int value, size_t size) {
  if (size > capacity) return 1;
  std::memset(dst, value, size);
  return 0;
}
inline int aclrtSynchronizeStream(aclrtStream) { return 0; }

namespace oneflow { namespace cuda { namespace layer_norm {
template<typename T, int N> struct Packet { T value[N]; };
template<typename T, int N> using PackType = Packet<T, N>;
template<typename T, int N> union Pack { T elem[N]; PackType<T, N> storage; };
template<typename T, typename C> struct DirectLoad {
  DirectLoad(const T* data, int64_t cols) : data(data), cols(cols) {}
  const T* data;
  int64_t cols;
};
} namespace rms_norm {
template<typename Load, typename Store, typename Compute>
aclError LaunchRmsNorm(aclrtStream, Load load, Store store, int64_t rows, int64_t cols,
                       double epsilon, Compute* inverse) {
  for (int64_t row = 0; row < rows; ++row) {
    float sum = 0;
    for (int64_t col = 0; col < cols; ++col) {
      const float value = float(load.data[row * cols + col]);
      sum += value * value;
    }
    inverse[row] = 1.0f / std::sqrt(sum / float(cols) + float(epsilon));
    for (int64_t col = 0; col < cols; ++col) {
      const float value = float(load.data[row * cols + col]) * inverse[row];
      store.template store<1>(&value, row, col);
    }
  }
  const std::string corruption = std::getenv("CORRUPTION") ? std::getenv("CORRUPTION") : "";
  if (corruption == "output") store.dst[rows * cols - 1] = half(123);
  if (corruption == "input") const_cast<half*>(load.data)[rows * cols - 1] = half(123);
  if (corruption == "inverse") inverse[rows - 1] = 123.0f;
  if (corruption == "nan") store.dst[rows * cols - 1] = std::numeric_limits<float>::quiet_NaN();
  if (corruption == "weight" && store.weight) const_cast<half*>(store.weight)[cols - 1] = half(123);
  if (corruption == "guard") reinterpret_cast<uint8_t*>(store.dst)[rows * cols * sizeof(half)] = 0;
  if (corruption == "unwritten") {
    const uint16_t marker = 0xffff;
    std::memcpy(store.dst + rows * cols - 1, &marker, sizeof(marker));
  }
  return 0;
}
}}}

namespace ascify950 {
using InputPatternCode = int;
inline const char* GetArg(int argc, char** argv, const char* key, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return fallback;
}
struct Case {
  std::string op = "rms_norm", dtype = "fp16", tier = "full", case_id, input_pattern = "random";
  int64_t idx = 0, rows = 0, cols = 0;
  bool run_check = true, affine = false;
  double eps = 1e-5;
};
inline std::vector<Case> ReadCases(const std::string& path) {
  std::ifstream input(path);
  std::vector<Case> result;
  std::string line;
  while (std::getline(input, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    Case c;
    if (row >> c.idx >> c.rows >> c.cols >> c.affine) result.push_back(c);
  }
  return result;
}
inline bool TierEnabled(const std::string&, const std::string&) { return true; }
inline bool CheckedTensorSize(int64_t rows, int64_t cols, size_t width, size_t* elements,
                              size_t* bytes, std::string* reason) {
  if (rows <= 0 || cols <= 0 || uint64_t(rows) > SIZE_MAX / uint64_t(cols) / width) {
    *reason = "invalid shape";
    return false;
  }
  *elements = size_t(rows) * cols;
  *bytes = *elements * width;
  return true;
}
inline InputPatternCode PatternCode(const std::string&) { return 0; }
inline float HostInputValue(InputPatternCode, uint64_t linear, int64_t row,
                            int64_t col, int64_t, uint64_t seed) {
  return float((linear * 7 + uint64_t(row * 3 + col) + seed % 31) % 64) / 16 - 2;
}
inline uint64_t SplitMix64Host(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
inline uint16_t FloatToHalfBits(float value) {
  const half h = half(value);
  uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}
inline float HalfBitsToFloat(uint16_t bits) {
  half h;
  std::memcpy(&h, &bits, sizeof(h));
  return float(h);
}
class DeviceBuffer {
 public:
  ~DeviceBuffer() { std::free(pointer); }
  aclError Allocate(size_t size) { bytes = size; pointer = std::malloc(size); return pointer ? 0 : 1; }
  void* Get() { return pointer; }
  size_t Bytes() { return bytes; }
 private:
  void* pointer = nullptr;
  size_t bytes = 0;
};
class AclSession {
 public:
  aclError Init(int) { return 0; }
  aclrtStream Stream() { return nullptr; }
};
inline std::string AclErrorReason(const char* operation, aclError error) {
  return std::string(operation) + ":" + std::to_string(error);
}
struct AccuracyRecord {
  std::string run_id, op, dtype, math_mode, variant, status, reason;
  Case test_case;
  int device_id = -1;
  double max_abs_error = 0, max_scaled_rel_error = 0, max_aux_abs_error = 0,
         max_aux_scaled_rel_error = 0;
  uint64_t nonfinite_count = 0, guard_mismatch_count = 0, canary_mismatch_count = 0;
};
class AccuracyCsvWriter {
 public:
  explicit AccuracyCsvWriter(const std::string& path) : output(path) {}
  void Write(const AccuracyRecord& r) {
    output << std::setprecision(17) << r.test_case.idx << ',' << r.variant << ',' << r.status << ','
           << r.max_abs_error << ',' << r.max_scaled_rel_error << ',' << r.max_aux_abs_error << ','
           << r.max_aux_scaled_rel_error << ',' << r.nonfinite_count << ','
           << r.guard_mismatch_count << ',' << r.canary_mismatch_count << ',' << r.reason << '\n';
  }
 private:
  std::ofstream output;
};
}
