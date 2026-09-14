# Converged masked warp candidate

`include/ascify/masked_warp_compat.hpp` provides the partial-mask paths for
public `ascify::ballot_sync` and `ascify::shfl_down_sync`
only when **every currently active lane supplies the same mask, equal to that
instruction's active lanes**. An unmasked native ballot combines all local
validation failures before any lane traps. No block barrier, shared scratch,
lane compaction, or waiting for another subgroup is used.

This domain excludes a valid CUDA collective whose named lanes have not yet
converged, and concurrent disjoint CUDA masks inside one active cohort. It is
not a general implementation of CUDA masked rendezvous. A runtime mask is
admitted only when the collective check succeeds. Shuffle types are restricted
to `int32_t`, `uint32_t`, and `float`. The public down-shuffle retains its
existing full-mask native path, including int64/uint64; partial masks do not
extend those types. Public CANN 8.5 keeps the original shuffle domain and
rejects the new ballot API when instantiated.

Shuffle forwards `value`, `delta`, and `width` to the native primitive. Widths
1, 2, 4, 8, 16, and 32 are admitted. No value is prescribed for a source lane
outside the mask. In particular, the helper does not replace that value with
zero, mask `delta`, or alter width boundaries.

## SDK evidence

These are CANN 9.1.0 SDK source observations, not device validation. Paths are
relative to the SDK's `aarch64-linux/asc` directory:

- `include/simt_api/device_warp_functions.h:26,28,50–58` declares unmasked
  ballot, active-mask, and shuffle operations.
- `impl/simt_api/device_warp_functions_impl.h:49–57,109–132` forwards them to
  `__ballot`, `__activemask`, and `__shfl_down`.
- `include/simt_api/cooperative_groups.h:178–188` admits tile size 1;
  `impl/simt_api/cooperative_groups_impl.h:438–445` forwards its shuffle with
  `width=Size`, including width 1.

SDK source establishes these calls exist. The device test below is required to
check the compiler's active-mask behavior inside conditional execution.

## Verification

Run the host guard model with:

```sh
bash tests/rewrite/check_masked_warp_compat.sh
```

It checks collective rejection (including a mismatched mask or width in another
lane), physical source addressing, width boundaries, preservation of a nonzero
unspecified-source result, and compile-time rejection of FP64 and 64-bit integer
types. Host stubs do **not** establish device convergence or synchronization.

Compile `tests/rewrite/masked_warp_device_probe.cce` with the legacy SIMT
compiler configuration for your CANN 9.1.0 installation, the real SDK headers,
and the repository `include/` directory on the include path. This probe is
separate from the host-only release checks. Its executable uses logical device
0; configure device visibility for the intended device before launching it.
Run the executable without arguments first. All positive and negative calls
use the public mapped wrappers, including full-mask and partial-mask dispatch. It checks eight full,
prefix, sparse, and single-lane masks,
six widths, seven deltas, and all three admitted types across 42 blocks. Each
active observation must report the selected mask; inactive lanes must leave
their output untouched. Shuffle values are compared only when CUDA defines the
source; an unselected source is counted and excluded from numerical assertions.

Run each negative mode in a **fresh process**, because an intentional device
trap can invalidate the context:

```text
inconsistent-mask
unconverged-mask
subset-mask
disjoint-masks
zero-mask
invalid-width
```

Each negative run must report synchronization status 507035 from the verified
SDK device trap. A kernel that returns successfully, or an unrelated runtime
error, is a failure. The test itself does not change the
main compatibility header, converter routing, or hardware settings.

## Reduction boundary

A converged partial-mask collective does not by itself prove that a reduction
algorithm reads only selected lanes. For example, a width-32 down-shuffle with
delta 16 can read an unselected source when the final group has fewer than 32
members. The CUDA result for that operation is undefined. Passing this helper's
tests therefore does not establish numerical correctness for every reduction
that uses it. Validate the caller's source-lane domain separately; the helper
does not repair an invalid reduction algorithm.
