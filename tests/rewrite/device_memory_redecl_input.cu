#include <cuda_runtime.h>
#include <string.h>

// Redeclare the selected device wrapper, not the separate host libc overload.
// This legal redeclaration must invalidate its all-system provenance proof.
static __device__ void *memcpy(void *, const void *, size_t);
struct CopyPacket { uint4 x, y, z, w; };
__device__ CopyPacket untrusted_redeclaration() {
  unsigned int words[16] = {};
  CopyPacket packet;
  memcpy(&packet, words, sizeof(packet));
  return packet;
}
