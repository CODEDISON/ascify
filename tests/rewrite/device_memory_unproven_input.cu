#include <cuda_runtime.h>
#include <string.h>

struct CopyPacket { uint4 x, y, z, w; };
#define PRIVATE_COPY_MACRO(dst, src) memcpy(dst, src, sizeof(*(dst)))

__global__ void unproven_copy(unsigned int *global, unsigned int count) {
  unsigned int words[16] = {};
  unsigned int short_words[8] = {};
  __shared__ unsigned int shared_words[16];
  __shared__ CopyPacket shared_packet;
  CopyPacket packet;
  memcpy(&packet, global, sizeof(packet));
  memcpy(&packet, shared_words, sizeof(packet));
  memcpy(&shared_packet, words, sizeof(shared_packet));
  memcpy(&packet, short_words, sizeof(packet));
  memcpy(&packet, words, count);
  unsigned int *alias = words;
  memcpy(&packet, alias, sizeof(packet));
  PRIVATE_COPY_MACRO(&packet, words);
}

template<typename T>
__device__ void dependent_copy(T *out) {
  unsigned int words[16] = {};
  T packet;
  memcpy(&packet, words, sizeof(packet));
  *out = packet;
}
__global__ void instantiate_copy(CopyPacket *out) { dependent_copy(out); }

namespace user {
// Keep the object's associated namespace here so ADL does not additionally
// select the global CUDA memcpy and turn the negative fixture into an error.
struct CopyPacket { uint4 x, y, z, w; };
// Even an exact forwarding body does not establish SDK declaration identity.
static __device__ inline void *memcpy(void *dst, const void *src, size_t bytes) {
  return __builtin_memcpy(dst, src, bytes);
}
__device__ void namesake() {
  unsigned int words[16] = {};
  CopyPacket packet;
  memcpy(&packet, words, sizeof(packet));
}
}

void host_copy() {
  unsigned int words[16] = {};
  CopyPacket packet;
  memcpy(&packet, words, sizeof(packet));
}
