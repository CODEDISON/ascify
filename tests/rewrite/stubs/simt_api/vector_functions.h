#ifndef ASCIFY_TEST_STUB_VECTOR_FUNCTIONS_H
#define ASCIFY_TEST_STUB_VECTOR_FUNCTIONS_H

// CPU model types only. Real CCEC layout is independently asserted by the
// installed helper and positive target-compilation matrix.
struct alignas(16) float4 {
  float x;
  float y;
  float z;
  float w;
};
struct alignas(4) uchar4 { unsigned char x, y, z, w; };

inline uchar4 make_uchar4(unsigned char x, unsigned char y,
                         unsigned char z, unsigned char w) {
  return {x, y, z, w};
}

inline float4 make_float4(float x, float y, float z, float w) {
  return {x, y, z, w};
}

#endif
