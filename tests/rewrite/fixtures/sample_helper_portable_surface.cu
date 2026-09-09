#include <helper_cuda.h>

uint32_t portableHelperSurface(const char* left, const char* right) {
  const std::string owned(left);
  return static_cast<uint32_t>(
      strlen(owned.c_str()) + (STRCASECMP(left, right) == 0 ? 1 : 0));
}
