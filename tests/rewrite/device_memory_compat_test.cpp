#define __aicore__
#include "ascify/device_memory_compat.hpp"

#include <assert.h>
#include <stdint.h>
#include <string.h>

struct alignas(16) Packet { uint32_t words[16]; };
struct GuardedPacket { unsigned char before[16]; Packet packet; unsigned char after[16]; };

int main() {
  for (unsigned int seed = 0; seed < 256; ++seed) {
    unsigned char source[64];
    for (unsigned int i = 0; i < 64; ++i)
      source[i] = static_cast<unsigned char>(seed + i * 37U);
    GuardedPacket destination;
    memset(&destination, 0xa5, sizeof(destination));
    assert(ascify::device_memcpy_private<64>(&destination.packet, source) == &destination.packet);
    assert(memcmp(&destination.packet, source, sizeof(source)) == 0);
    for (unsigned char byte : destination.before) assert(byte == 0xa5);
    for (unsigned char byte : destination.after) assert(byte == 0xa5);
  }
  uint32_t source_words[16];
  for (unsigned int i = 0; i < 16; ++i) source_words[i] = 0xff800000U + i * 1234567U;
  Packet packet;
  assert(ascify::device_memcpy_private<64>(&packet, source_words) == &packet);
  assert(memcmp(&packet, source_words, sizeof(packet)) == 0);
}
