# Converged masked warp candidate

`include/ascify/masked_warp_compat.hpp` provides the partial-mask paths for
public `ascify::ballot_sync`, `ascify::shfl_down_sync`, and `ascify::shfl_up_sync`
only when **every currently active lane supplies the same mask, equal to that
instruction's active lanes**. An unmasked native ballot combines all local
validation failures before any lane traps. No block barrier, shared scratch,
lane compaction, or waiting for another subgroup is used.

This domain excludes a valid CUDA collective whose named lanes have not yet
converged, and concurrent disjoint CUDA masks inside one active cohort. It is
not a general implementation of CUDA masked rendezvous. A runtime mask is
admitted only when the collective check succeeds. Shuffle types are restricted
to `int32_t`, `uint32_t`, and `float`. Both public shuffles retain their
existing full-mask native path, including int64/uint64; partial masks do not
extend those types. Public CANN 8.5 keeps the original shuffle domain and
rejects the new ballot API when instantiated.

Down-shuffle forwards `value`, `delta`, and `width` to the native primitive.
Its widths are 1, 2, 4, 8, 16, and 32. The new up-shuffle admits only widths
2, 4, 8, 16, and 32; width 1 remains rejected as in the earlier public up API.
Up-shuffle preserves physical source lane addressing and segment boundaries.
When `delta >= width`, CUDA specifies the caller's own value. These lanes pass
native delta 0, because CANN packs the delta directly into its instruction
control word without masking it. All active lanes still execute the native
shuffle, so a lane with a large delta remains available as another lane's
source. No value is prescribed for an inactive source, and no sparse-lane
compaction or replacement with zero is performed. The existing full-mask
paths are unchanged; the new normalization applies to the partial-mask up
path only. The physical control-word finding also exposes an existing risk
in the unchanged full-mask up path for large deltas. This candidate does not
fix that path or claim its entire legal CUDA delta domain is validated. Its
full-mask host regression checks only earlier small-delta routing and types.
Neither shuffle adds a memory barrier or a masked rendezvous.

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
- `device_warp_functions.h:40–48` and its implementation at 84–107 declare
  up-shuffle and forward it to `__shfl_up`. The physical header hashes are
  `d3fc428852429e9b8ae73edf89aacd088eb6954d23e1efe41c78f29b3298288a`
  and `dbbd22168206cc9d8827b9c91d342f3204653eacb09a1d39e28e4452082fdebd`.
- The CANN compiler header `__clang_cce_simt_dav3510_intrinsic.h:635–696`
  (SHA256 `79b42a2ef9dc79b005ad7ce31e33ba4df0e94109752db3fc1fac22fd5f041157`)
  packs `(32-width)<<16 | clamp<<8 | delta`, with clamp 0 for up-shuffle.
  This is why unbounded deltas are normalized without removing participants.

The CANN `asc_shfl_up` API card describes independent width segments and
returning the caller's own value at their lower boundary. Its width wording
excludes 1, while the CG header admits tile 1. Up-shuffle keeps the conservative
existing width domain until that discrepancy is validated independently.

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
types. The up model additionally checks bit transport for signed zero,
infinities, NaN payloads, subnormals and integer extrema, large and per-lane
deltas, all-lane native participation, and single argument evaluation. Public
wrapper tests retain full-mask 64-bit paths and reject partial-mask 64-bit,
FP64, width 1, and unsupported public-8.5 paths. Host stubs do **not** establish
device convergence or synchronization.

`tests/frontend_compat/masked_warp_input.cu` includes the three actual CUDA
up-shuffle overloads; the binary-enabled release check requires exactly three
mapped calls and no residual CUDA up-shuffle. This check still requires the
new converter's real frontend. The up implementation requires a separate
CCEC/device validation before any original `shfl_scan` success can be claimed.

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

## Pending up-shuffle device probe

`masked_up_device_probe.cce` is a separate, source-only probe for the new path.
It must be compiled and executed under the same real-provider, exclusive-device,
and health checks as the down probe before target support is accepted. It does
not establish a device result merely by being present in the repository.

Every positive up-probe call goes through the public `ascify::shfl_up_sync`
wrapper with a non-full mask, selecting the new legacy helper. It never uses
the full-mask route. Its positive run covers 45 blocks, eight partial masks,
widths 2/4/8/16/32,
and nine deltas (0, 1, width-1, width, width+1, 31, 32, 255, UINT_MAX), for
three block shapes: 256, 16x16, and 8x8x4. In the UINT_MAX case, odd lanes use
delta 1 while even lanes keep UINT_MAX, ensuring a large-delta lane remains
available as a source. All three admitted types are checked against a separate
CPU physical-lane oracle. Inactive outputs retain canaries; undefined inactive
sources are counted without assigning them a value. The device probe uses
exactly representable finite float values; NaN payload and special bit-pattern
transport is currently a host-model check only.

The up probe adds `width-one` and `peer-invalid-width` to the six negative modes
listed above. Each must run separately with the same exact device-trap status
check. Block shapes exercise linear thread ordering; this probe does not
validate the original `shfl_scan` execution gate or its complete CPU oracle.

## Reduction boundary

A converged partial-mask collective does not by itself prove that a reduction
algorithm reads only selected lanes. For example, a width-32 down-shuffle with
delta 16 can read an unselected source when the final group has fewer than 32
members. The CUDA result for that operation is undefined. Passing this helper's
tests therefore does not establish numerical correctness for every reduction
that uses it. Validate the caller's source-lane domain separately; the helper
does not repair an invalid reduction algorithm.
