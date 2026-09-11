// Host scheduling/model test, not evidence of NPU convergence or synchronization.
#include <ascify/cooperative_groups_compat.hpp>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace model {
struct Barrier {
  std::mutex mutex;
  std::condition_variable ready;
  unsigned arrived = 0, generation = 0;
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    const unsigned previous = generation;
    if (++arrived == 32) {
      arrived = 0;
      ++generation;
      ready.notify_all();
    } else {
      ready.wait(lock, [&] { return generation != previous; });
    }
  }
};
struct Warp {
  Barrier barrier;
  uint32_t slots[32] = {};
  unsigned block_size, warp_rank;
  Warp(unsigned size, unsigned rank) : block_size(size), warp_rank(rank) {}
};
thread_local Warp* current;
thread_local unsigned lane;
template <typename T> T shuffle(T value, unsigned source) {
  static_assert(sizeof(T) == sizeof(uint32_t), "only 32-bit registers are modeled");
  std::memcpy(&current->slots[lane], &value, sizeof(value));
  current->barrier.wait(); // Every source value has been published.
  T result;
  std::memcpy(&result, &current->slots[source], sizeof(result));
  current->barrier.wait(); // No slot can be reused until all readers finish.
  return result;
}
}  // namespace model

unsigned int asc_activemask() { return UINT32_MAX; }
namespace cooperative_groups {
unsigned int thread_block::thread_rank() { return model::current->warp_rank * 32 + model::lane; }
unsigned int thread_block::size() { return model::current->block_size; }
template <unsigned int Size, typename ParentT>
unsigned long long thread_block_tile<Size, ParentT>::thread_rank() { return model::lane; }
template <unsigned int Size, typename ParentT> template <typename T>
T thread_block_tile<Size, ParentT>::shfl_down(T value, unsigned int delta) const {
  const unsigned source = model::lane + delta;
  return model::shuffle(value, source < Size ? source : model::lane);
}
template <unsigned int Size, typename ParentT> template <typename T>
T thread_block_tile<Size, ParentT>::shfl(T value, int source) const {
  assert(source >= 0 && static_cast<unsigned>(source) < Size);
  return model::shuffle(value, static_cast<unsigned>(source));
}
template <unsigned int Size>
thread_block_tile<Size, thread_block> tiled_partition(const thread_block&) { return {}; }
}  // namespace cooperative_groups

static uint32_t bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}
static float tree_sum(const float (&input)[32]) {
  float work[32];
  std::memcpy(work, input, sizeof(work));
  // One scalar tree oracle, not 32 lane-wise replicas of the implementation.
  for (unsigned stride = 16; stride != 0; stride /= 2)
    for (unsigned i = 0; i < stride; ++i) {
      volatile float rounded = work[i] + work[i + stride];
      work[i] = rounded;
    }
  return work[0];
}
static void run(unsigned fixture, unsigned block_size, unsigned warp_rank) {
  model::Warp warp(block_size, warp_rank);
  int input_i[32], typed_i[32], erased_i[32];
  float input_f[32], typed_f[32], erased_f[32];
  int64_t integer_sum = 0;
  double mathematical_sum = 0;
  for (unsigned i = 0; i < 32; ++i) {
    input_i[i] = static_cast<int>(fixture * 1000) +
                 (i % 2 ? 1 : -1) * (100 + static_cast<int>(i) * 13);
    input_f[i] = (i % 2 ? 1.0f : -1.0f) * (i + 1) * 0.125f;
    if (fixture == 1) {
      input_f[i] = i == 0 ? 16777216.0f : i == 16 ? 1.0f :
                   i == 8 ? -16777216.0f : i % 2 ? 0.125f : -0.125f;
    }
    integer_sum += input_i[i];
    mathematical_sum += static_cast<double>(input_f[i]);
  }
  const float golden = tree_sum(input_f);
  std::vector<std::thread> workers;
  for (unsigned lane = 0; lane < 32; ++lane) workers.emplace_back([&, lane] {
    model::current = &warp;
    model::lane = lane;
    const auto block = ascify_cg::this_thread_block();
    const auto typed = ascify_cg::tiled_partition<32>(block);
    const ascify_cg::thread_block_tile<32> erased = typed;
    assert(block.thread_rank() == warp_rank * 32 + lane && block.size() == block_size);
    assert(typed.thread_rank() == lane && erased.thread_rank() == lane);
    assert(typed.size() == 32 && erased.size() == 32);
    assert(typed.meta_group_rank() == warp_rank && erased.meta_group_rank() == warp_rank);
    assert(typed.meta_group_size() == block_size / 32 && erased.meta_group_size() == block_size / 32);
    typed_i[lane] = ascify_cg::reduce(typed, input_i[lane], ascify_cg::plus<int>());
    erased_i[lane] = ascify_cg::reduce(erased, input_i[lane], ascify_cg::plus<int>());
    typed_f[lane] = ascify_cg::reduce(typed, input_f[lane], ascify_cg::plus<float>());
    erased_f[lane] = ascify_cg::reduce(erased, input_f[lane], ascify_cg::plus<float>());
  });
  for (auto& worker : workers) worker.join();
  for (unsigned lane = 0; lane < 32; ++lane) {
    assert(typed_i[lane] == integer_sum && erased_i[lane] == integer_sum);
    assert(bits(typed_f[lane]) == bits(golden) && bits(erased_f[lane]) == bits(golden));
    assert(bits(typed_f[lane]) == bits(typed_f[0]) && bits(erased_f[lane]) == bits(typed_f[0]));
  }
  std::printf("cg_reduce host block_size=%u warp_rank=%u lanes=32 tree=%.9g "
              "double_sum=%.17g abs_error=%.17g all_lane_bits_equal=1\n",
              block_size, warp_rank, static_cast<double>(golden), mathematical_sum,
              std::fabs(static_cast<double>(typed_f[0]) - mathematical_sum));
}
int main() {
  run(0, 64, 1);
  run(1, 128, 2);
}
