#include <ascify/cuda_vec3_layout.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include <initializer_list>

static void require(bool condition) { if (!condition) abort(); }
struct Nested { char tag; ascify_cuda_float3 value; char tail; };
struct Ray { ascify_cuda_float3 origin, direction; };
static_assert(sizeof(Nested) == 20 && alignof(Nested) == 4 &&
              offsetof(Nested, value) == 4 && offsetof(Nested, tail) == 16, "nested layout");
static_assert(sizeof(Ray) == 24 && alignof(Ray) == 4 && offsetof(Ray, direction) == 12, "ray layout");

template<class T> static void representation() {
  // This is a representation test, not frontend admission of arrays or GM.
  const uint32_t words[] = {0x80000000u, 0x3f800000u, 0xffffffffu,
                            0x00000001u, 0x7f7fffffu, 0x01234567u,
                            0x89abcdefu, 0u, 0x7fc12345u};
  T values[3];
  memcpy(values, words, sizeof(words));
  require(reinterpret_cast<const unsigned char*>(&values[1]) -
          reinterpret_cast<const unsigned char*>(&values[0]) == 12);
  for (unsigned index = 0; index < 3; ++index) {
    T copy = values[index];
    T assigned;
    assigned = copy;
    require(memcmp(&assigned, words + index * 3, 12) == 0);
    const T& reference = assigned;
    require(memcmp(&reference.z, words + index * 3 + 2, 4) == 0);
  }
  for (unsigned offset : {4u, 8u}) {
    alignas(16) unsigned char storage[64];
    memset(storage, 0xa5, sizeof(storage));
    T* value = new (storage + offset) T;
    memcpy(value, words, 12);
    require(memcmp(value, words, 12) == 0);
    for (unsigned index = 0; index < sizeof(storage); ++index)
      if (index < offset || index >= offset + 12) require(storage[index] == 0xa5);
    value->~T();
  }
}

static int evaluations = 0;
static float component(float value) { ++evaluations; return value; }
int main() {
  representation<ascify_cuda_float3>();
  representation<ascify_cuda_uint3>();
  representation<ascify_cuda_int3>();
  const auto f = ascify_make_cuda_float3(component(1.0f), component(-0.0f), component(-3.5f));
  const auto u = ascify_make_cuda_uint3(0u, 0xffffffffu, 0x80000000u);
  const auto i = ascify_make_cuda_int3(-2147483647 - 1, 2147483647, -1);
  require(evaluations == 3 && f.x == 1.0f && f.z == -3.5f);
  uint32_t negativeZero = 0;
  memcpy(&negativeZero, &f.y, 4);
  require(negativeZero == 0x80000000u);
  require(u.x == 0 && u.y == 0xffffffffu && u.z == 0x80000000u);
  require(i.x == (-2147483647 - 1) && i.y == 2147483647 && i.z == -1);
  puts("cuda vec3 host layout: three records, 12/4, offsets 0/4/8, stride12, nested20, ray24, copies and factories PASS");
}
