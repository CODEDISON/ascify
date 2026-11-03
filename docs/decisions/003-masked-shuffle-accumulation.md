# Masked down-shuffle accumulation

Status: isolated implementation draft; no new operator acceptance.

An unchanged CUDA Samples v13.0 reduction run with int, kernel7, 256 threads,
16,777,216 elements and its original warmup plus100 iterations produced
2143424895 instead of the original CPU result2139353471 on the verified950
product. The campaign stopped at that first numerical failure. The kernel9
int/float/double configurations passed separately; these do not establish
the remaining kernel families.

The final kernel7 warp reduction has eight participants but starts with
down-shuffle offsets16 and8. CUDA does not define values read from inactive
source lanes, and the950 AscShflDown contract describes uninitialized register
values for those reads. This is a source-contract defect and a candidate
explanation for the observed failure, not a measured attribution of every
incorrect output bit.

For an independently proved pure add/down-shuffle helper, the proposed target
completion omits additions from inactive source lanes inside width32. A new
delta wrapper selects a valid self register for such unused reads. Every
participating lane still executes its shuffle; existing convergence checks
remain. The ordinary shuffle functions and their mask semantics are unchanged.

Defined source-neighbor reads and out-of-width self additions retain their
original operation and rounding order for every active lane. Only the source
undefined case receives the new completion. Consecutive low lanes then form
the intended prefix reduction; arbitrary sparse masks do not acquire a
mathematical all-members reduction guarantee. Signed-overflow inputs remain
outside the defined arithmetic domain. Binary64 uses the existing verified
integer RNE backend, and skipped additions never introduce synthetic positive
zero into an all-negative-zero group.

The source proof must authenticate the actual SDK callee, exact same scalar
variable, width32, immutable effect-free metadata, the pure helper loop,
all concrete template occurrences and protected emitted names. Edits compose
with existing replacements through bounded insertions; no unknown edit may
be dropped. Int/float/double are the only scalar domain. No SDK-layout,
reflection, inquiry, rendezvous or function-name exception follows.

Before acceptance, validate source-valid paired cases, all defined-lane
results, prefix sums, negative zero, source boundaries and target resources;
then rerun the unchanged original target with actual input/output evidence.
Keep the failed product, source bytes, original workload and numerical gates.

Sources: [CUDA13.0 shuffle contract](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#warp-shuffle-functions),
[reported source issue398](https://github.com/NVIDIA/cuda-samples/issues/398).
