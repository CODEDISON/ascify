# Cooperative reduction and already converged warp masks

Status: implemented for the domains below; actual compiler and device results
are recorded separately for each DT build.

The original twenty CUDA Samples remain the counting unit. Adding a working
non-FP64 reduction fixture does not establish that the original Reduction
translation unit converts, compiles, or executes. That unit also instantiates
multi-warp groups and double. Its source and instantiations are unchanged.
This decision extends the cooperative-group boundary recorded in ADR 0021.

## Source and target contract

The opt-in `ascify-admitted-v1` profile now admits the exact
`cooperative_groups/reduce.h` include from its authenticated profile directory.
Source declarations expose `plus<int>`, `plus<float>`, and reduction over a
32-thread tile. They are parsing declarations; device arithmetic resides in
`ascify/cooperative_groups_compat.hpp`. Unsupported reduction groups and
operations have no matching overload; unsupported tile sizes give an explicit
multi-warp synchronization diagnostic. Aliased paths, modified headers, and
local header shadows remain rejected by the include provenance check.

The tile facade preserves CUDA's static unsigned-int rank and size. Tiles
created from a thread block have static metadata accessors. Conversion into
`thread_block_tile<32, void>` captures its creation metadata, whose accessors
are non-static as in CUDA. The native thread block already declares static
unsigned-int rank and size, so its alias preserves those source signatures.
Three-dimensional block rank and size are checked independently in the device
fixture. Register shuffles admit exact int, unsigned int, and float types; the
existing uint4 XOR shuffle remains four separate 32-bit transfers.

A supported reduction performs a float/int down-shuffle tree and broadcasts
lane zero's result to every lane. The broadcast makes every float result use
the same addition order even under cancellation. It does not promise sequential
FP32 summation or exact real arithmetic. All 32 lanes of the tile must currently
participate; an incomplete active warp traps. Another warp in the block may
bypass this operation. No block barrier, block memory fence, or shared scratch
is used for this single-warp collective. Integer inputs must avoid signed
addition overflow, as with the source integer arithmetic.

## Masked warp operations

For the verified legacy SIMT headers, public `ascify::ballot_sync` and the
partial-mask path of `ascify::shfl_down_sync` use a shared guard. Every active
lane must provide the same mask, equal to the instruction's complete active
mask. An unmasked native ballot combines local failures before trapping. This
is a deliberately smaller domain than CUDA's masked rendezvous: lanes named
by a mask must already be converged; disjoint masks in one active cohort are
rejected. The adapter does not synchronize lanes arriving through different
branches and does not introduce a block barrier.

The down-shuffle partial path admits int32, uint32, and float, with widths
1, 2, 4, 8, 16, and 32. It preserves physical source addressing. A source lane
outside the participating mask has no prescribed CUDA result; the adapter
never substitutes zero. Existing full-mask shuffle paths, including native
int64/uint64, remain intact. Ordinary full-mask widths 2 through 32 retain the
previous path. The public CANN 8.5 family retains its previous shuffle domain
and rejects the new ballot API when instantiated.

The final reduction step in the frozen sample's `reduce7` passes a prefix
ballot mask into a conditional warp reduction. Its fixed offsets can address
an unselected source when fewer than 32 lanes remain. Accepting that mask does
not make the source algorithm defined or establish a numerical pass. Its
shape and arithmetic are not modified to force a passing result.

## Multi-warp boundary and further implementation

The installed SDK has native block barriers, shared storage, scalar shuffle,
ballot, and active-mask primitives. Those primitives are sufficient building
blocks for software algorithms; their existence does not provide a generic
multi-warp cooperative-group implementation. The native tile implementation
admits sizes up to 32, and its tile synchronization is a memory fence rather
than a collective execution barrier. Substituting a block barrier for a tile
barrier would deadlock callers where another tile does not participate.

Generic multi-warp reduction therefore remains rejected. A possible separate
adapter must prove whole-block uniform participation at conversion time and
then use shared partial sums plus real block barriers and scratch reuse
protection. Such proof must cover the enclosing control flow, early returns,
all tile instances, and any forwarding function. A source name or a documented
caller obligation is insufficient. Until this proof and its divergent-call,
early-return, and altered-wrapper negatives are implemented and tested, the
profile does not admit that domain. FP64 is outside this change.

## Evidence and validation

SDK inspection used DT CANN 9.1.0 under
`aarch64-linux/asc/{include,impl}/simt_api`. The cooperative header declares
thread-block sync/rank/size at lines 84–90 and tile sizes/register operations at
178–211. Its implementation uses a block execution barrier for thread-block
sync, while tile sync at 394–398 calls a block memory fence. Warp headers and
implementation expose `asc_ballot`, `asc_activemask`, and native shuffle. These
source facts establish API availability, not successful device execution.

`check_frontend_compat.sh` checks authenticated profile bytes, exact includes,
positive int/float reduction declarations and rejected block/multi-warp uses.
`check_cooperative_groups_compat.sh` checks target signatures and existing
uint4 behavior, and calls the real reduce facade using a 32-thread host
shuffle model with read/write barriers. `check_simt_compat.sh` retains both SDK
families and checks
that full-mask int64/uint64 calls avoid the new partial helper.
`check_masked_warp_compat.sh` models masks and source lanes, including rejection
and nonzero unspecified-source values; its host stubs do not prove hardware
convergence. Raw macro-collision identifiers include the new masked header.

`cooperative_groups_reduce_probe.cce` runs full and alternate-warp participation,
typed and erased tiles, signed integer inputs, float cancellation, and mixed
magnitudes. It checks each output lane against one host tree per warp, reports
an independent double mathematical sum and its error, and verifies metadata
using an independently calculated 3D output index. A separate process tests
partial tile participation. `masked_warp_device_probe.cce` uses the public
mapped wrappers for full/prefix/sparse/single-lane masks, six widths and seven
deltas, and separately counts undefined source-lane cases without inventing
expected values. Rejection modes run in separate processes after context traps.
The initial DT reduction probe exposed that `__builtin_trap` lowers to an
unsupported host `abort` call when a rejection branch survives optimization.
The shared `device_contract_reject.hpp` now calls the SDK's direct
`__asc_simt_vf::__trap` for 3510 device builds. It preserves caller `assert` and
`ascendc_assert` macros and the SDK's seven assertion-helper macros around
the SDK include; neither `NDEBUG` nor
`ASCENDC_DUMP=0` disables this direct call. CPU debug uses the host trap because
the SDK debug branch is a no-op. Legacy mask and width guards use the same
helper; the public 8.5 paths retain their previous behavior.

An isolated DT CANN 9.1.0 experiment compiled this SDK call with ordinary O2,
linked through CCEC, checked all 32 positive outputs, and ran the rejection in
a separate process on locked device 1. Positive synchronization returned 0;
negative synchronization returned exactly 507035 without timing out. Source,
compiler, SDK, and included main-header SHA identities were unchanged. This
establishes the chosen error mechanism; it does not replace the complete
CG/mask numerical probes. Their negative modes require exactly 507035, so
unrelated allocation or device initialization errors cannot count as passes.
All target tests require actual compile, link, launch, synchronization and
readback evidence before a device pass can be reported.
