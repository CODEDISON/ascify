#include <cuda_runtime.h>
#include <string.h>

struct CopyPacket { uint4 x, y, z, w; };

__device__ CopyPacket private_pack(unsigned int seed) {
  unsigned int words[16];
  for (unsigned int i = 0; i < 16; ++i)
    words[i] = seed * 1664525U + i * 1013904223U;
  CopyPacket packet;
  memcpy(&packet, words, sizeof(packet));
  return packet;
}

__global__ void private_copy_result(CopyPacket *out, unsigned int *status) {
  unsigned char bytes[64];
  for (unsigned int i = 0; i < 64; ++i)
    bytes[i] = static_cast<unsigned char>(i * 37U + threadIdx.x);
  CopyPacket packet;
  void *returned = memcpy(&packet, bytes, sizeof(packet));
  out[threadIdx.x] = packet;
  status[threadIdx.x] = returned == &packet;
}

namespace application {
namespace ascify {
template<size_t> __device__ void *device_memcpy_private(void *, const void *);
}
__device__ CopyPacket namespace_capture() {
  unsigned int words[16] = {};
  CopyPacket packet;
  memcpy(&packet, words, sizeof(packet));
  return packet;
}
}
