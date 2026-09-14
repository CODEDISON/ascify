// Synthetic host contract test. No NPU access or device correctness claim.
// Isolate the symbol adapter from the separately tested lifecycle manager.
#define ASCIFY_ASCIFY_CUDA_COMPAT_HPP
#define ASCIFY_SIMT_HEADER_FAMILY_LEGACY_BETA3 1
#include <ascify/symbol_compat.hpp>

#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <limits>

namespace {
struct Recorded {
  int ready_calls = 0;
  int size_calls = 0;
  int copy_calls = 0;
  aclError ready_status = ACL_SUCCESS;
  aclError size_status = ACL_SUCCESS;
  aclError copy_status = ACL_SUCCESS;
  std::size_t size = 0;
  const void *size_token = nullptr;
  const void *copy_token = nullptr;
  const void *source = nullptr;
  std::size_t count = 0;
  std::size_t offset = 0;
  aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_HOST;
} recorded;

void reset(std::size_t size) {
  recorded = {};
  recorded.size = size;
}
}  // namespace

namespace ascify {
inline aclError cudaRuntimeEnsureReady() {
  ++recorded.ready_calls;
  return recorded.ready_status;
}
}  // namespace ascify

aclError aclrtGetSymbolSize(const void *symbol, std::size_t *size) {
  ++recorded.size_calls;
  recorded.size_token = symbol;
  if (recorded.size_status == ACL_SUCCESS) *size = recorded.size;
  return recorded.size_status;
}

aclError aclrtMemcpyToSymbol(const void *symbol, const void *source,
                            std::size_t count, std::size_t offset,
                            aclrtMemcpyKind kind) {
  ++recorded.copy_calls;
  recorded.copy_token = symbol;
  recorded.source = source;
  recorded.count = count;
  recorded.offset = offset;
  recorded.kind = kind;
  return recorded.copy_status;
}

int main() {
  float scalar = 17.0f;
  float array[8] = {};
  unsigned table[3][31] = {};
  const float source[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  reset(sizeof(array));
  assert(ascify::cudaMemcpyToSymbol(array, source, sizeof(source)) == ACL_SUCCESS);
  assert(recorded.size_token == &array && recorded.copy_token == &array);
  assert(recorded.source == source && recorded.count == sizeof(source));
  assert(recorded.offset == 0 && recorded.kind == ACL_MEMCPY_HOST_TO_DEVICE);
  assert(array[0] == 0 && array[7] == 0);  // The shim did not write host memory.

  reset(sizeof(scalar));
  assert(ascify::cudaMemcpyToSymbol(scalar, source, sizeof(scalar)) == ACL_SUCCESS);
  assert(recorded.copy_token == &scalar && scalar == 17.0f);

  reset(sizeof(table));
  assert(ascify::cudaMemcpyToSymbol(table, source, sizeof(float),
                                  31 * sizeof(unsigned),
                                  ACL_MEMCPY_DEVICE_TO_DEVICE) == ACL_SUCCESS);
  assert(recorded.copy_token == &table && recorded.size_token == &table);
  assert(recorded.offset == 31 * sizeof(unsigned));
  assert(recorded.kind == ACL_MEMCPY_DEVICE_TO_DEVICE);

  // The runtime-reported size, not sizeof(T), controls the checked range.
  reset(7);
  assert(ascify::cudaMemcpyToSymbol(array, source, 4, 4) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(recorded.size_calls == 1 && recorded.copy_calls == 0);
  reset(sizeof(array));
  assert(ascify::cudaMemcpyToSymbol(array, source, 1, sizeof(array)) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(recorded.copy_calls == 0);
  reset(sizeof(array));
  assert(ascify::cudaMemcpyToSymbol(array, source, 2,
                                  std::numeric_limits<std::size_t>::max()) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(recorded.copy_calls == 0);

  for (auto kind : {ACL_MEMCPY_HOST_TO_HOST, ACL_MEMCPY_DEVICE_TO_HOST,
                    static_cast<aclrtMemcpyKind>(99)}) {
    reset(sizeof(array));
    assert(ascify::cudaMemcpyToSymbol(array, source, 1, 0, kind) ==
           ACL_ERROR_RT_PARAM_INVALID);
    assert(recorded.ready_calls == 0 && recorded.copy_calls == 0);
  }
  reset(sizeof(array));
  assert(ascify::cudaMemcpyToSymbol(array, nullptr, 1) ==
         ACL_ERROR_RT_PARAM_INVALID);
  assert(recorded.ready_calls == 0 && recorded.size_calls == 0);

  reset(sizeof(array));
  recorded.ready_status = 98701;
  assert(ascify::cudaMemcpyToSymbol(array, source, 1) == 98701);
  assert(recorded.size_calls == 0 && recorded.copy_calls == 0);
  reset(sizeof(array));
  recorded.size_status = 98702;
  assert(ascify::cudaMemcpyToSymbol(array, source, 1) == 98702);
  assert(recorded.copy_calls == 0);
  reset(sizeof(array));
  recorded.copy_status = 98703;
  assert(ascify::cudaMemcpyToSymbol(array, source, 1) == 98703);
  assert(recorded.copy_calls == 1);

  // Zero-byte copies do not bypass invalid-symbol errors or SDK status.
  reset(sizeof(array));
  recorded.size_status = 98704;
  assert(ascify::cudaMemcpyToSymbol(array, nullptr, 0) == 98704);
  assert(recorded.copy_calls == 0);
  reset(sizeof(array));
  recorded.copy_status = 98705;
  assert(ascify::cudaMemcpyToSymbol(array, nullptr, 0, sizeof(array)) == 98705);
  assert(recorded.copy_calls == 1 && recorded.count == 0);

  // Each call obtains current lifecycle and registration state again.
  reset(sizeof(array));
  assert(ascify::cudaMemcpyToSymbol(array, source, 1) == ACL_SUCCESS);
  recorded.size_status = 98706;
  assert(ascify::cudaMemcpyToSymbol(array, source, 1) == 98706);
  assert(recorded.ready_calls == 2 && recorded.size_calls == 2);
  assert(recorded.copy_calls == 1);

  std::puts("synthetic symbol compatibility host contracts passed (no NPU)");
}
