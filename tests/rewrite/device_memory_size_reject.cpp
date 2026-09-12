#define __aicore__
#include "ascify/device_memory_compat.hpp"
struct WrongSize { unsigned char data[32]; };
int main() {
  unsigned char source[64] = {};
  WrongSize destination;
  ascify::device_memcpy_private<64>(&destination, source);
}
