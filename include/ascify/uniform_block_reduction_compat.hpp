#ifndef ASCIFY_UNIFORM_BLOCK_REDUCTION_COMPAT_HPP
#define ASCIFY_UNIFORM_BLOCK_REDUCTION_COMPAT_HPP

// Included after the ordinary cooperative-groups facade. These distinct types
// are emitted only after UniformBlockReduction proves whole-block participation
// and unique scratch ownership. They never implement generic subgroup sync.
namespace ascify_cg {
template <typename T> struct uniform_scalar_domain {
  static constexpr bool value = __is_same(T, int) || __is_same(T, float) || __is_same(T, double);
};
template <unsigned BlockSize> struct block_tile_memory {
  static_assert(BlockSize >= 32 && BlockSize <= 1024 && BlockSize % 32 == 0,
                "uniform reduction requires 32..1024 complete block threads");
  volatile int warp_i[BlockSize / 32];
  volatile int result_i[BlockSize / 32];
  volatile float warp_f[BlockSize / 32];
  volatile float result_f[BlockSize / 32];
  // Binary64 crosses shared storage only as uint64 bits. No floating64
  // instruction, float narrowing, or aliasing through the int/float slots.
  volatile uint64_t warp_d[BlockSize / 32];
  volatile uint64_t result_d[BlockSize / 32];
};

template <unsigned BlockSize> struct scratch_thread_block {
  __ubuf__ block_tile_memory<BlockSize>* storage;
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned int thread_rank() {
    return thread_block::thread_rank();
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned int size() {
    return thread_block::size();
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static void sync() { thread_block::sync(); }
};

template <unsigned BlockSize>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline scratch_thread_block<BlockSize>
this_thread_block(__ubuf__ block_tile_memory<BlockSize>& storage) {
  if (thread_block::size() != BlockSize)
    ::ascify::detail::reject_device_contract();
  return {&storage};
}

template <unsigned BlockSize>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void
sync(const scratch_thread_block<BlockSize>& group) { group.sync(); }

template <unsigned Size, unsigned BlockSize> struct scratch_thread_block_tile {
  static_assert(Size == 32 || Size == 64 || Size == 128 || Size == 256 || Size == 512,
                "uniform reduction admits tile sizes 32, 64, 128, 256, 512");
  static_assert(BlockSize % Size == 0, "tiles must exactly partition their block");
  __ubuf__ block_tile_memory<BlockSize>* storage;
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned int thread_rank() {
    return thread_block::thread_rank() % Size;
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static constexpr unsigned int size() { return Size; }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static unsigned int meta_group_rank() {
    return thread_block::thread_rank() / Size;
  }
  __SIMT_DEVICE_FUNCTIONS_DECL__ static constexpr unsigned int meta_group_size() {
    return BlockSize / Size;
  }
};

template <unsigned Size, unsigned BlockSize>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline scratch_thread_block_tile<Size, BlockSize>
tiled_partition(const scratch_thread_block<BlockSize>& parent) {
  return {parent.storage};
}

template <typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline T uniform_reduction_shuffle_down(
    T value, unsigned offset) {
  if constexpr (std::is_same<T, double>::value)
    return ::ascify::detail::scalar64_shfl_down(UINT32_MAX, value, offset, 32);
  else return asc_shfl_down(value, offset, 32);
}

// An emitted guard for the unique two-type pure reduce-forwarder. This
// trait never describes an arbitrary group or grants layout observations.
template <typename Group> struct uniform_reduce_group_domain : std::false_type {};
template <typename Group> struct uniform_reduce_group_domain<const Group>
    : uniform_reduce_group_domain<Group> {};
template <typename Parent>
struct uniform_reduce_group_domain<thread_block_tile<32, Parent>>
    : std::integral_constant<bool,std::is_same<Parent,void>::value ||
                                  std::is_same<Parent,thread_block>::value> {};
template <unsigned Size, unsigned BlockSize>
struct uniform_reduce_group_domain<scratch_thread_block_tile<Size,BlockSize>>
    : std::integral_constant<bool,(Size==32 || Size==64 || Size==128 ||
                                  Size==256 || Size==512) &&
                                 BlockSize>=32 && BlockSize<=1024 && BlockSize%Size==0> {};

template <unsigned Size, unsigned BlockSize, typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline
typename std::enable_if<std::is_same<T, int>::value ||
                        std::is_same<T, float>::value ||
                        std::is_same<T, double>::value, T>::type
reduce(const scratch_thread_block_tile<Size, BlockSize>& group, T value, plus<T> operation) {
  if (thread_block::size() != BlockSize || asc_activemask() != 0xffffffffU)
    ::ascify::detail::reject_device_contract();
  const unsigned rank = thread_block::thread_rank();
  const unsigned lane = rank % 32, warp = rank / 32, tile = rank / Size;
  constexpr unsigned warps_per_tile = Size / 32;
  for (unsigned offset = 16; offset != 0; offset >>= 1)
    value = operation(value, uniform_reduction_shuffle_down(value, offset));
  if (lane == 0) {
    if constexpr (std::is_same<T, int>::value) group.storage->warp_i[warp] = value;
    else if constexpr (std::is_same<T, float>::value) group.storage->warp_f[warp] = value;
    else group.storage->warp_d[warp] = ::ascify::detail::scalar64_to_bits(value);
  }
  thread_block::sync();
  if (warp % warps_per_tile == 0) {
    value = T(0);
    if (lane < warps_per_tile) {
      if constexpr (std::is_same<T, int>::value)
        value = group.storage->warp_i[tile * warps_per_tile + lane];
      else if constexpr (std::is_same<T, float>::value)
        value = group.storage->warp_f[tile * warps_per_tile + lane];
      else value = ::ascify::detail::scalar64_from_bits<double>(
          group.storage->warp_d[tile * warps_per_tile + lane]);
    }
    // Reduce only real partials. Adding synthetic +0 would change an all--0
    // result, and Size==32 must not perform an extra identity operation.
    for (unsigned offset = warps_per_tile / 2; offset != 0; offset >>= 1) {
      const T source = uniform_reduction_shuffle_down(value, offset);
      if (lane < offset) value = operation(value, source);
    }
    if (lane == 0) {
      if constexpr (std::is_same<T, int>::value) group.storage->result_i[tile] = value;
      else if constexpr (std::is_same<T, float>::value) group.storage->result_f[tile] = value;
      else group.storage->result_d[tile] = ::ascify::detail::scalar64_to_bits(value);
    }
  }
  thread_block::sync();
  T result;
  if constexpr (std::is_same<T, int>::value) result = group.storage->result_i[tile];
  else if constexpr (std::is_same<T, float>::value) result = group.storage->result_f[tile];
  else result = ::ascify::detail::scalar64_from_bits<double>(group.storage->result_d[tile]);
  thread_block::sync(); // Finish every reader before a following call reuses slots.
  return result;
}
}  // namespace ascify_cg
#endif
