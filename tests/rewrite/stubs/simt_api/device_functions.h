#ifndef ASCIFY_TEST_STUB_DEVICE_FUNCTIONS_H
#define ASCIFY_TEST_STUB_DEVICE_FUNCTIONS_H

#ifndef ASCIFY_TEST_SIMT_BUILTINS
#define ASCIFY_TEST_SIMT_BUILTINS
struct ascify_test_simt_dim3 { unsigned int x, y, z; };
inline ascify_test_simt_dim3 threadIdx{0, 0, 0};
inline ascify_test_simt_dim3 blockDim{1, 1, 1};
#endif

inline float __fdividef(float numerator, float denominator) {
  return numerator / denominator;
}

#endif
