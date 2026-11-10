#ifndef ASCIFY_DYNAMIC_WARP_GROUPS_COMPAT_HPP
#define ASCIFY_DYNAMIC_WARP_GROUPS_COMPAT_HPP

// WIP, deliberately not included by the public facade or translator. A source
// proof must admit only the closed SDK-origin value and operation domain before
// this backend can be published. This header grants no grid/partial-warp sync.
// Cooperative grid-wide synchronization is backed by the device-global
// sense-reversing barrier defined in ascify_cuda_compat.hpp (declared at
// the true global ::ascify scope so the merged-TU include order resolves).
namespace ascify {
__SIMT_DEVICE_FUNCTIONS_DECL__ void grid_barrier_sync(unsigned blocks);
}

namespace ascify_cg {
namespace dynamic_detail {
__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned count(unsigned mask) {
  unsigned result = 0;
  for (; mask != 0; mask &= mask - 1) ++result;
  return result;
}
__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned lane() {
  return ::cooperative_groups::thread_block::thread_rank() % 32;
}
__SIMT_DEVICE_FUNCTIONS_DECL__ inline unsigned member(unsigned mask, unsigned rank) {
  for (unsigned bit = 0; bit != 32; ++bit) {
    if ((mask & (1u << bit)) != 0) {
      if (rank == 0) return bit;
      --rank;
    }
  }
  ::ascify::detail::reject_device_contract();
  return 0;
}
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void require_members_active(unsigned mask) {
  // Register exchange needs all members at this instruction; additional live
  // lanes may independently operate on a disjoint labeled/binary group.
  const unsigned active = asc_activemask();
  if (mask == 0 || (mask & (1u << lane())) == 0 || (active & mask) != mask)
    ::ascify::detail::reject_device_contract();
}
}  // namespace dynamic_detail

class coalesced_group final {
  unsigned mask_;
  unsigned meta_rank_;
  unsigned meta_size_;
  __SIMT_DEVICE_FUNCTIONS_DECL__ explicit coalesced_group(
      unsigned mask, unsigned meta_rank = 0, unsigned meta_size = 1)
      : mask_(mask), meta_rank_(meta_rank), meta_size_(meta_size) {
    if (mask_ == 0 || (mask_ & (1u << dynamic_detail::lane())) == 0)
      ::ascify::detail::reject_device_contract();
  }
  friend __SIMT_DEVICE_FUNCTIONS_DECL__ coalesced_group coalesced_threads();
  friend __SIMT_DEVICE_FUNCTIONS_DECL__ coalesced_group binary_partition(
      const coalesced_group&, bool);
  friend __SIMT_DEVICE_FUNCTIONS_DECL__ coalesced_group labeled_partition(
      const coalesced_group&, int);
  template <unsigned Size, typename Parent>
  friend __SIMT_DEVICE_FUNCTIONS_DECL__ coalesced_group binary_partition(
      const thread_block_tile<Size, Parent>&, bool);
 public:
  __SIMT_DEVICE_FUNCTIONS_DECL__ unsigned size() const {
    return dynamic_detail::count(mask_);
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ unsigned num_threads() const { return size(); }
  __SIMT_DEVICE_FUNCTIONS_DECL__ unsigned thread_rank() const {
    const unsigned bit = dynamic_detail::lane();
    return dynamic_detail::count(mask_ & ((1u << bit) - 1u));
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ unsigned meta_group_rank() const { return meta_rank_; }
  __SIMT_DEVICE_FUNCTIONS_DECL__ unsigned meta_group_size() const { return meta_size_; }
  template <typename T> __SIMT_DEVICE_FUNCTIONS_DECL__
  typename std::enable_if<std::is_same<T, int>::value ||
                         std::is_same<T, unsigned>::value, T>::type
  shfl(T value, int source_rank) const {
    dynamic_detail::require_members_active(mask_);
    if (source_rank < 0 || static_cast<unsigned>(source_rank) >= size())
      ::ascify::detail::reject_device_contract();
    return asc_shfl(value, dynamic_detail::member(mask_, unsigned(source_rank)), 32);
  }
};

__SIMT_DEVICE_FUNCTIONS_DECL__ inline coalesced_group coalesced_threads() {
  return coalesced_group(asc_activemask());
}
__SIMT_DEVICE_FUNCTIONS_DECL__ inline coalesced_group binary_partition(
    const coalesced_group& parent, bool predicate) {
  dynamic_detail::require_members_active(parent.mask_);
  const unsigned yes = asc_ballot(predicate) & parent.mask_;
  const bool homogeneous = yes == 0 || yes == parent.mask_;
  const unsigned selected = predicate ? yes : parent.mask_ & ~yes;
  return coalesced_group(selected, homogeneous ? 0 : unsigned(predicate),
                         homogeneous ? 1 : 2);
}
template <unsigned Size, typename Parent>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline coalesced_group binary_partition(
    const thread_block_tile<Size, Parent>&, bool predicate) {
  static_assert(Size == 32 && (std::is_same<Parent, void>::value ||
                              std::is_same<Parent, thread_block>::value),
                "dynamic binary partition requires one complete SDK block tile32");
  dynamic_detail::require_members_active(0xffffffffu);
  const unsigned yes = asc_ballot(predicate);
  const bool homogeneous = yes == 0 || yes == 0xffffffffu;
  return coalesced_group(predicate ? yes : ~yes,
                         homogeneous ? 0 : unsigned(predicate), homogeneous ? 1 : 2);
}

// Partition by int label using only the verified 32-bit register primitives.
// Each parent group has one mask/size, so every member executes every key
// fetch. First physical leaders order the resulting groups, as in CUDA13.4
// details/partitioning.h::_labeled_partition_dispatch.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline coalesced_group labeled_partition(
    const coalesced_group& parent, int label) {
  dynamic_detail::require_members_active(parent.mask_);
  unsigned selected = 0;
  const unsigned count = parent.size();
  for (unsigned rank = 0; rank < count; ++rank) {
    const int candidate = parent.shfl(label, int(rank));
    if (candidate == label)
      selected |= 1u << dynamic_detail::member(parent.mask_, rank);
  }
  const unsigned leader = dynamic_detail::member(selected, 0);
  dynamic_detail::require_members_active(parent.mask_);
  const unsigned leaders = asc_ballot(dynamic_detail::lane() == leader) & parent.mask_;
  return coalesced_group(selected,
      dynamic_detail::count(leaders & ((1u << leader) - 1u)),
      dynamic_detail::count(leaders));
}

template <typename T> struct greater {
  static_assert(std::is_same<T, int>::value,
                "dynamic reduction max currently admits only CUDA int");
  __SIMT_DEVICE_FUNCTIONS_DECL__ T operator()(T left, T right) const {
    return left > right ? left : right;
  }
};
namespace dynamic_detail {
// The frozen 950 CANN target uses two's-complement int32 registers. Prove the
// byte representation at compilation, then use defined unsigned arithmetic
// across a different reduction tree. No existing plus<> path is changed.
static_assert(sizeof(int)==4 && sizeof(unsigned)==4 &&
              std::is_same<int32_t,int>::value &&
              std::is_same<uint32_t,unsigned>::value &&
              __builtin_bit_cast(unsigned,-1)==0xffffffffu,
              "dynamic integer reduction requires the bound two's-complement int32 ABI");
struct sum_int32 {
  __SIMT_DEVICE_FUNCTIONS_DECL__ int operator()(int left,int right) const {
    const unsigned bits=static_cast<unsigned>(left)+static_cast<unsigned>(right);
    int value;
    __builtin_memcpy(&value,&bits,sizeof(value));
    return value;
  }
};
template <typename Op>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline int reduce_int(
    const coalesced_group& group, int value, Op operation) {
  const unsigned rank = group.thread_rank();
  const unsigned count = group.size();
  for (unsigned offset = 1; offset < count; offset <<= 1) {
    const bool combine = rank % (offset * 2) == 0 && rank + offset < count;
    // Every member executes each transport, including the ragged tree tail.
    const unsigned source = combine ? rank + offset : rank;
    const int peer = group.shfl(value, int(source));
    if (combine) value = operation(value, peer);
  }
  return group.shfl(value, 0);
}
}  // namespace dynamic_detail
__SIMT_DEVICE_FUNCTIONS_DECL__ inline int reduce(
    const coalesced_group& group, int value, plus<int>) {
  return dynamic_detail::reduce_int(group, value, dynamic_detail::sum_int32{});
}
__SIMT_DEVICE_FUNCTIONS_DECL__ inline int reduce(
    const coalesced_group& group, int value, greater<int> operation) {
  return dynamic_detail::reduce_int(group, value, operation);
}


// Grid coordinates stay read-only; synchronization is the documented
// co-residency barrier above (admission contract: the cooperative occupancy
// advice sizes grids that can be simultaneously resident).
class grid_group final {
  __SIMT_DEVICE_FUNCTIONS_DECL__ explicit grid_group() {}
  friend __SIMT_DEVICE_FUNCTIONS_DECL__ grid_group this_grid();
  __SIMT_DEVICE_FUNCTIONS_DECL__ static void check_dimensions() {
    if (blockDim.y != 1 || blockDim.z != 1 || gridDim.y != 1 || gridDim.z != 1)
      ::ascify::detail::reject_device_contract();
  }
 public:
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned long long size() {
    check_dimensions();
    return static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned long long thread_rank() {
    check_dimensions();
    return static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ void sync() const {
    check_dimensions();
    // CUDA grid-wide sync: every thread of every block must reach this call.
    // The block rendezvous brackets the single ticket taken by thread 0, so
    // all lanes of a block observe barrier completion together.
    ::ascify_cg::this_thread_block().sync();
    if (threadIdx.x == 0)
      ::ascify::grid_barrier_sync(static_cast<unsigned>(gridDim.x));
    ::ascify_cg::this_thread_block().sync();
  }
};
__SIMT_DEVICE_FUNCTIONS_DECL__ inline grid_group this_grid() { return grid_group(); }
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void sync(const grid_group& group) {
  group.sync();
}
}  // namespace ascify_cg
#endif
