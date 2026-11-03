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
  static_assert(std::is_same<T, int>::value || std::is_same<T, float>::value ||
                std::is_same<T, double>::value,
                "Ascify cooperative plus supports int, float and exact binary64 double");
  ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_ T operator()(T left, T right) const;
};
// The target implements a register-only all-reduction for a complete tile32.
// No arbitrary functor or generic multi-warp tile is admitted here.
template <unsigned int Size, typename ParentT, typename T>
ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
typename std::enable_if<Size == 32 && (std::is_same<T, int>::value ||
                                    std::is_same<T, float>::value ||
                                    std::is_same<T, double>::value), T>::type
reduce(const thread_block_tile<Size, ParentT>& group, T value, plus<T> operation);
// This overload is published only when the mandatory whole-TU scratch proof
// succeeds. It does not admit a generic multi-warp thread_block_tile.
template <unsigned Size, unsigned BlockSize, typename T>
ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
typename std::enable_if<std::is_same<T, int>::value ||
                        std::is_same<T, float>::value ||
                        std::is_same<T, double>::value, T>::type
reduce(const scratch_thread_block_tile<Size, BlockSize>& group, T value, plus<T> operation);

// These declarations provide inventory for the mandatory exact int/owned
// plus-or-max gate. They grant neither a generic functor nor a sync capability.
template<class T> struct greater {
  ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_ T operator()(T,T) const;
};
template<class Op> ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
int reduce(const coalesced_group&,int,Op);
}  // namespace cooperative_groups
#undef ASCIFY_FRONTEND_COMPAT_REDUCE_DEVICE_
#endif
