// Executes the actual software adapter with a scheduling model. This verifies
// arithmetic and scratch reuse; it is not evidence about an NPU memory model.
#include <array>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace model {
struct Barrier {
  std::mutex mutex;
  std::condition_variable condition;
  unsigned count = 0, generation = 0, participants;
  explicit Barrier(unsigned participants) : participants(participants) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    const unsigned previous = generation;
    if (++count == participants) {
      count = 0;
      ++generation;
      condition.notify_all();
    } else condition.wait(lock, [&] { return generation != previous; });
  }
};
struct Warp {
  Barrier barrier{32};
  std::array<uint32_t, 32> values{};
};
thread_local unsigned rank, block_size;
thread_local Warp* warp;
thread_local Barrier* block_barrier;
template <typename T> T down(T value, unsigned delta, unsigned width) {
  assert(width == 32);
  const unsigned lane = rank % 32;
  std::memcpy(&warp->values[lane], &value, sizeof(T));
  warp->barrier.wait();
  T result;
  const unsigned source = lane + delta < 32 ? lane + delta : lane;
  std::memcpy(&result, &warp->values[source], sizeof(T));
  warp->barrier.wait();
  return result;
}
}  // namespace model

#define __SIMT_DEVICE_FUNCTIONS_DECL__
#define __ubuf__
namespace ascify { namespace detail {
inline void reject_device_contract() { std::abort(); }
} }
namespace ascify_cg {
struct thread_block {
  static unsigned thread_rank() { return model::rank; }
  static unsigned size() { return model::block_size; }
  static void sync() { model::block_barrier->wait(); }
};
template <typename T> struct plus { T operator()(T a, T b) const { return a + b; } };
}  // namespace ascify_cg
unsigned asc_activemask() { return UINT32_MAX; }
template <typename T> T asc_shfl_down(T value, unsigned delta, unsigned width) {
  return model::down(value, delta, width);
}
#include <ascify/uniform_block_reduction_compat.hpp>

static uint32_t bits(float value) {
  uint32_t output;
  std::memcpy(&output, &value, sizeof(output));
  return output;
}
static float tree(std::array<float, 32> input, unsigned leaves = 32) {
  for (unsigned stride = leaves / 2; stride; stride /= 2)
    for (unsigned i = 0; i < stride; ++i) {
      volatile float rounded = input[i] + input[i + stride];
      input[i] = rounded;
    }
  return input[0];
}
static int integer_input(unsigned rank, unsigned round) {
  return int(rank * 7 + round * 11) - 900;
}
static float float_input(unsigned rank, unsigned round) {
  if (round == 0) return -0.0f;
  if (round == 1) return rank % 2 ? INFINITY : -INFINITY;
  const float cancel[] = {16777216.f, 1.f, -16777216.f, 0.25f, -0.f, 0.f, -2.f, 0.125f};
  return cancel[rank % 8] + float(round * (rank / 32 + 1));
}

template <unsigned BlockSize, unsigned Group> static void run() {
  constexpr unsigned rounds = 4;
  ascify_cg::block_tile_memory<BlockSize> scratch{};
  model::Barrier barrier(BlockSize);
  model::Warp warps[BlockSize / 32];
  std::vector<int> integer_results(BlockSize * rounds);
  std::vector<float> float_results(BlockSize * rounds);
  std::vector<unsigned> ranks(BlockSize), group_ranks(BlockSize), group_sizes(BlockSize);
  std::vector<std::thread> threads;
  for (unsigned rank = 0; rank < BlockSize; ++rank) threads.emplace_back([&, rank] {
    const unsigned x = rank % 8, y = (rank / 8) % 4, z = rank / 32;
    model::rank = x + 8 * (y + 4 * z);
    model::block_size = BlockSize;
    model::warp = &warps[rank / 32];
    model::block_barrier = &barrier;
    const auto block = ascify_cg::this_thread_block(scratch);
    const auto tile = ascify_cg::tiled_partition<Group>(block);
    static_assert(std::is_same<decltype(tile.thread_rank()), unsigned int>::value, "CUDA rank type");
    ranks[rank] = tile.thread_rank();
    group_ranks[rank] = tile.meta_group_rank();
    group_sizes[rank] = tile.meta_group_size();
    for (unsigned round = 0; round < rounds; ++round) {
      integer_results[round * BlockSize + rank] =
          ascify_cg::reduce(tile, integer_input(rank, round), ascify_cg::plus<int>());
      float_results[round * BlockSize + rank] =
          ascify_cg::reduce(tile, float_input(rank, round), ascify_cg::plus<float>());
    }
  });
  for (auto& thread : threads) thread.join();
  unsigned errors = 0;
  double max_error = 0;
  for (unsigned round = 0; round < rounds; ++round) {
    for (unsigned group = 0; group < BlockSize / Group; ++group) {
      int64_t integer_sum = 0;
      double real_sum = 0;
      std::array<float, 32> partials{};
      for (unsigned warp = 0; warp < Group / 32; ++warp) {
        std::array<float, 32> values{};
        for (unsigned lane = 0; lane < 32; ++lane) {
          const unsigned rank = group * Group + warp * 32 + lane;
          integer_sum += integer_input(rank, round);
          values[lane] = float_input(rank, round);
          real_sum += double(values[lane]);
        }
        partials[warp] = tree(values);
      }
      const float expected = tree(partials, Group / 32);
      max_error = std::fmax(max_error, std::fabs(double(expected) - real_sum));
      for (unsigned lane = 0; lane < Group; ++lane) {
        const unsigned rank = group * Group + lane;
        errors += integer_results[round * BlockSize + rank] != integer_sum;
        const float actual = float_results[round * BlockSize + rank];
        errors += std::isnan(expected) ? !std::isnan(actual) : bits(actual) != bits(expected);
        errors += ranks[rank] != lane || group_ranks[rank] != group ||
                  group_sizes[rank] != BlockSize / Group;
      }
    }
  }
  std::printf("uniform host B=%u G=%u rounds=%u scalar_results=%u max_double_error=%.9g errors=%u\n",
              BlockSize, Group, rounds, 2 * BlockSize * rounds, max_error, errors);
  assert(errors == 0);
}
int main() {
  run<32, 32>();
  run<128, 32>();
  run<128, 64>();
  run<128, 128>();
}
