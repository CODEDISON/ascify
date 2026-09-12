#ifndef ASCIFY_UNIFORM_BLOCK_REDUCTION_COMPAT_HPP
#define ASCIFY_UNIFORM_BLOCK_REDUCTION_COMPAT_HPP

// Included after the ordinary cooperative-groups facade. These distinct types
// are emitted only after UniformBlockReduction proves whole-block participation
// and unique scratch ownership. They never implement generic subgroup sync.
namespace ascify_cg {
template <typename T> struct uniform_scalar_domain {
  static constexpr bool value = __is_same(T, int) || __is_same(T, float);
};
template <unsigned BlockSize> struct block_tile_memory {
  static_assert(BlockSize >= 32 && BlockSize <= 1024 && BlockSize % 32 == 0,
                "uniform reduction requires 32..1024 complete block threads");
  volatile int warp_i[BlockSize / 32];
  volatile int result_i[BlockSize / 32];
  volatile float warp_f[BlockSize / 32];
  volatile float result_f[BlockSize / 32];
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

template <unsigned Size, unsigned BlockSize, typename T>
__SIMT_DEVICE_FUNCTIONS_DECL__ inline
typename std::enable_if<std::is_same<T, int>::value ||
                        std::is_same<T, float>::value, T>::type
reduce(const scratch_thread_block_tile<Size, BlockSize>& group, T value, plus<T> operation) {
  if (thread_block::size() != BlockSize || asc_activemask() != 0xffffffffU)
    ::ascify::detail::reject_device_contract();
  const unsigned rank = thread_block::thread_rank();
  const unsigned lane = rank % 32, warp = rank / 32, tile = rank / Size;
  constexpr unsigned warps_per_tile = Size / 32;
  for (unsigned offset = 16; offset != 0; offset >>= 1)
    value = operation(value, asc_shfl_down(value, offset, 32));
  if (lane == 0) {
    if constexpr (std::is_same<T, int>::value) group.storage->warp_i[warp] = value;
    else group.storage->warp_f[warp] = value;
  }
  thread_block::sync();
  if (warp % warps_per_tile == 0) {
    value = T(0);
    if (lane < warps_per_tile) {
      if constexpr (std::is_same<T, int>::value)
        value = group.storage->warp_i[tile * warps_per_tile + lane];
      else value = group.storage->warp_f[tile * warps_per_tile + lane];
    }
    // Reduce only real partials. Adding synthetic +0 would change an all--0
    // result, and Size==32 must not perform an extra identity operation.
    for (unsigned offset = warps_per_tile / 2; offset != 0; offset >>= 1) {
      const T source = asc_shfl_down(value, offset, 32);
      if (lane < offset) value = operation(value, source);
    }
    if (lane == 0) {
      if constexpr (std::is_same<T, int>::value) group.storage->result_i[tile] = value;
      else group.storage->result_f[tile] = value;
    }
  }
  thread_block::sync();
  T result;
  if constexpr (std::is_same<T, int>::value) result = group.storage->result_i[tile];
  else result = group.storage->result_f[tile];
  thread_block::sync(); // Finish every reader before a following call reuses slots.
  return result;
}
}  // namespace ascify_cg
#endif
