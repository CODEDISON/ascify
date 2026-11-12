// Ascify admission header for CUDA's <cuda/barrier>: block-scope
// arrive/wait barrier over the admitted pure-SIMT generic atomics.
//
// Contracts this header must honor (simpleAWBarrier spelling):
//   __shared__ cuda::barrier<cuda::thread_scope_block> b;
//   init(&b, blockDim.x);      // thread 0, followed by a block-wide sync
//   auto token = b.arrive();   // every participating thread, once per phase
//   b.wait(std::move(token));
//
// The state lives inside the __shared__ object; pure SIMT admits the generic
// atomics on the shared address space, and the rendezvous is the proven
// sense-reversing pattern of ascify grid_barrier_sync: the phase-completing
// arrival resets the counter BEFORE flipping the parity, and every peer can
// arrive at the next phase only after observing the flip, so the reset never
// races a later arrival. Release/acquire ordering comes from the block-scope
// device fence on both sides of the rendezvous.
#ifndef ASCIFY_CUDA_BARRIER_COMPAT_HPP
#define ASCIFY_CUDA_BARRIER_COMPAT_HPP

#include <utility>

#include "ascify_cuda_compat.hpp"

namespace cuda {

// libcudacxx scope tags; this admission surface implements the block scope.
enum thread_scope {
  thread_scope_block = 0,
};

template <thread_scope Scope>
class barrier;

// Move-only phase token (cuda::barrier_arrival_token spelling).
class barrier_arrival_token {
  unsigned int phase_;
  explicit barrier_arrival_token(unsigned int phase) : phase_(phase) {}
  template <thread_scope> friend class barrier;

 public:
  barrier_arrival_token() : phase_(0u) {}
  barrier_arrival_token(barrier_arrival_token&& other) : phase_(other.phase_) {}
  barrier_arrival_token& operator=(barrier_arrival_token&& other) {
    phase_ = other.phase_;
    return *this;
  }
  barrier_arrival_token(const barrier_arrival_token&) = delete;
  barrier_arrival_token& operator=(const barrier_arrival_token&) = delete;
};

template <>
class barrier<thread_scope_block> {
  unsigned int expected_;  // Arrivals per phase; set once by init.
  unsigned int arrived_;   // Arrivals of the current phase.
  unsigned int sense_;     // Phase parity; flips when a phase completes.

 public:
  __SIMT_DEVICE_FUNCTIONS_DECL__ inline void block_scope_init(
      unsigned int expected) {
    expected_ = expected;
    arrived_ = 0u;
    sense_ = 0u;
  }

  __SIMT_DEVICE_FUNCTIONS_DECL__ inline barrier_arrival_token arrive() {
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS)
    // Release: writes issued before the arrival are visible to phase peers.
    ::ascify::threadfence_block();
    const unsigned int local_sense =
        ::ascify::atomic_add_generic(&sense_, 0u);
    const unsigned int ticket =
        ::ascify::atomic_add_generic(&arrived_, 1u);
    if (ticket + 1u == expected_) {
      // The phase completes here: reset before the flip; peers arrive at
      // the next phase only after they observed the flip.
      ::ascify::atomic_exch_generic(&arrived_, 0u);
      ::ascify::atomic_exch_generic(&sense_, local_sense ^ 1u);
    }
    return barrier_arrival_token(local_sense);
#else
    return barrier_arrival_token(0u);  // non-target parse environments only
#endif
  }

  __SIMT_DEVICE_FUNCTIONS_DECL__ inline void wait(
      barrier_arrival_token&& token) {
#if defined(ASCIFY_SIMT_LEGACY_HAS_GM_ATOMICS)
    // Spin until the phase parity moves past the token snapshot.
    while (::ascify::atomic_add_generic(&sense_, 0u) == token.phase_) {
    }
    // Acquire: peer phase writes are visible to this thread afterwards.
    ::ascify::threadfence_block();
#else
    (void)token;  // non-target parse environments only
#endif
  }

  __SIMT_DEVICE_FUNCTIONS_DECL__ inline void arrive_and_wait() {
    wait(arrive());
  }
};

// The samples' spelling init(&barrier, count): ADL resolves to this overload
// because barrier<thread_scope_block> lives in namespace cuda.
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void init(
    barrier<thread_scope_block>* b, unsigned int expected) {
  b->block_scope_init(expected);
}
__SIMT_DEVICE_FUNCTIONS_DECL__ inline void init(
    barrier<thread_scope_block>* b, int expected) {
  b->block_scope_init(static_cast<unsigned int>(expected));
}

}  // namespace cuda

#endif  // ASCIFY_CUDA_BARRIER_COMPAT_HPP
