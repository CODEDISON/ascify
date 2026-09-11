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

These are source observations from DT's installed
`/usr/local/Ascend/cann-9.1.0/aarch64-linux/asc`, not device validation:

- `include/simt_api/device_warp_functions.h:26,28,50–58` declares unmasked
  ballot, active-mask, and shuffle operations.
- `impl/simt_api/device_warp_functions_impl.h:49–57,109–132` forwards them to
  `__ballot`, `__activemask`, and `__shfl_down`.
- `include/simt_api/cooperative_groups.h:178–188` admits tile size 1;
  `impl/simt_api/cooperative_groups_impl.h:438–445` forwards its shuffle with
  `width=Size`, including width 1.

SDK source establishes these calls exist. The DT test below is required to
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

Compile `tests/rewrite/masked_warp_device_probe.cce` using the project's current
DT compiler/launch recipe and its real SDK headers. Run the executable without
arguments first. All positive and negative calls use the public mapped
wrappers, including full-mask and partial-mask dispatch. It checks eight full,
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

Each negative run must report a non-success synchronization status; a kernel
that returns successfully is a failure. The test itself does not change the
main compatibility header, converter routing, or hardware settings.

## Reduction boundary

The frozen Reduction source SHA-256 is
`c9ac9a9a424726522dd616fa24d58e6f0919ac95dd746a233d5fb857bffffde9`.
Its final reduce7 stage passes the ballot from line 499 into the branch at
lines 500–504. This is a candidate for an already converged prefix group, to be
checked on DT. However, its `warpReduceSum` at lines 75–80 always begins with
delta 16 at width 32. When the final group has fewer than 32 members, it can read
an unselected source. The CUDA result for that operation is undefined. Passing
this helper's tests therefore does not establish numerical correctness for
that Reduction path. Algorithm changes or a narrower acceptance boundary need
separate review; this helper does not silently repair the source.
