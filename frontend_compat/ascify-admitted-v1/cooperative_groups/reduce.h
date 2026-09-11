#ifndef ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_REDUCE_H_
#define ASCIFY_FRONTEND_COMPAT_ADMITTED_V1_REDUCE_H_
#include <cooperative_groups.h>
#if defined(__CUDACC__) || defined(__CUDA__)
#define ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_ __device__
#else
#define ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
#endif
namespace cooperative_groups {
template <typename T> struct plus {
  static_assert(std::is_same<T, int>::value || std::is_same<T, float>::value,
                "Ascify cooperative plus supports only int and float");
  ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_ T operator()(T left, T right) const;
};
// The target implements a register-only all-reduction for a complete tile32.
// No arbitrary functor, block reduction, or multi-warp scratch is admitted.
template <unsigned int Size, typename ParentT, typename T>
ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
typename std::enable_if<Size == 32 && (std::is_same<T, int>::value ||
                                    std::is_same<T, float>::value), T>::type
reduce(const thread_block_tile<Size, ParentT>& group, T value, plus<T> operation);
}  // namespace cooperative_groups
#undef ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
#endif
