#pragma once

// Parsing-only CUDA FP16 declarations, never a numerical implementation.
// Generated code uses the target half/half2 types and real CANN intrinsics.
#if defined(__CUDACC__) || defined(__CUDA__)
#define ASCIFY_PARSE_HALF_HD __host__ __device__
#else
#define ASCIFY_PARSE_HALF_HD
#endif

struct half {
  unsigned short storage;
  ASCIFY_PARSE_HALF_HD half() = default;
  ASCIFY_PARSE_HALF_HD half(float);
  ASCIFY_PARSE_HALF_HD operator float() const;
};

using __half = half;

struct alignas(4) half2 {
  half x;
  half y;
  ASCIFY_PARSE_HALF_HD half2() = default;
  ASCIFY_PARSE_HALF_HD half2(half, half);
};

using __half2 = half2;

ASCIFY_PARSE_HALF_HD half2 operator+(half2, half2);
ASCIFY_PARSE_HALF_HD half2 operator*(half2, half2);
ASCIFY_PARSE_HALF_HD half2 __hadd2(half2, half2);
ASCIFY_PARSE_HALF_HD half2 __hmul2(half2, half2);
ASCIFY_PARSE_HALF_HD half2 __hfma2(half2, half2, half2);
ASCIFY_PARSE_HALF_HD half2 __float2half2_rn(float);
ASCIFY_PARSE_HALF_HD half2 __floats2half2_rn(float, float);
ASCIFY_PARSE_HALF_HD float __low2float(half2);
ASCIFY_PARSE_HALF_HD float __high2float(half2);

#undef ASCIFY_PARSE_HALF_HD
