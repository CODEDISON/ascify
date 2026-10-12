#ifndef ASCIFY_TEST_STUB_KERNEL_SIMT_INTF_H
#define ASCIFY_TEST_STUB_KERNEL_SIMT_INTF_H

#ifndef ASCIFY_TEST_SIMT_BUILTINS
#define ASCIFY_TEST_SIMT_BUILTINS
struct ascify_test_simt_dim3 { unsigned int x, y, z; };
inline ascify_test_simt_dim3 threadIdx{0, 0, 0};
inline ascify_test_simt_dim3 blockDim{1, 1, 1};
#endif

#include <stdint.h>

namespace AscendC {
namespace Simt {

inline int ascify_test_public_warp_reduce_add_calls = 0;
inline int ascify_test_public_warp_reduce_max_calls = 0;
inline int ascify_test_public_warp_reduce_min_calls = 0;
inline int ascify_test_public_warp_shuffle_calls = 0;
inline int ascify_test_public_thread_barrier_calls = 0;
inline int ascify_test_public_thread_fence_calls = 0;

template<typename T>
__aicore__ inline T WarpShflSync(T value, int32_t, int32_t = 32) {
  ++ascify_test_public_warp_shuffle_calls;
  return value;
}

template<typename T>
__aicore__ inline T WarpShflUpSync(T value, uint32_t, int32_t) {
  ++ascify_test_public_warp_shuffle_calls;
  return value;
}

template<typename T>
__aicore__ inline T WarpShflDownSync(T value, uint32_t, int32_t) {
  ++ascify_test_public_warp_shuffle_calls;
  return value;
}

template<typename T>
__aicore__ inline T WarpShflXorSync(T value, int32_t, int32_t) {
  ++ascify_test_public_warp_shuffle_calls;
  return value;
}

template<typename T>
__aicore__ inline T WarpReduceAddSync(T value) {
  ++ascify_test_public_warp_reduce_add_calls;
  return value;
}

template<typename T>
__aicore__ inline T WarpReduceMaxSync(T value) {
  ++ascify_test_public_warp_reduce_max_calls;
  return value;
}

template<typename T>
__aicore__ inline T WarpReduceMinSync(T value) {
  ++ascify_test_public_warp_reduce_min_calls;
  return value;
}

__aicore__ inline void ThreadBarrier() {
  ++ascify_test_public_thread_barrier_calls;
}

__aicore__ inline void ThreadFence() {
  ++ascify_test_public_thread_fence_calls;
}

}  // namespace Simt
}  // namespace AscendC

#endif
