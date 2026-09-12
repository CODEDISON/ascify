# Uniform block scratch reduction

Status: converter candidate compiled and tested with native macOS LLVM 23;
current DT target compilation and device validation remain pending.

## Decision

Add a distinct scratch reduction domain to the admitted frontend profile. The
source spelling `block_tile_memory<B>` with `auto block =
this_thread_block(scratch)` and `auto tile = tiled_partition<G>(block)` is
projected into scratch-specific block and tile types. Ordinary
`thread_block_tile<Size>` still requires Size 32. A generic tile reduction is
never replaced by a block barrier.

This projection changes a parsing type identity. Its use therefore requires a
mandatory whole-translation-unit proof before publication. Source code may not
observe, alias, specialize, overload on, or explicitly name the projected
types. The narrow accepted use consists of a single shared scratch declaration,
one auto block, one auto tile, scalar metadata calls, and top-level assignments
through one unique, unconstrained two-type forwarding template whose only
statement returns the authenticated `cg::reduce(group, value, cg::plus<T>())`.
Names of the kernel and forwarding function are immaterial.
This initial domain requires C++17. C++20 constraints and all forwarding
function/redeclaration attributes other than the CUDA device attribute are
excluded, as are parameter and template-parameter attributes. This prevents
the parsing type projection from changing a constraint or attribute-based
overload decision.
In addition, a scratch TU may not add declarations to `cooperative_groups`,
including through nested or inline namespaces and redeclarations. Such
declarations can participate in ADL for the original tile type and disappear
from overload resolution for the projected type. Checking only the forwarding
function's ordinary lookup scope is insufficient. The source namespace alias
`namespace cg = cooperative_groups` remains accepted.
Source declarations in the target `ascify_cg` namespace are likewise rejected
throughout the TU, including forced input headers. Otherwise a source-only
preinclude could inject a new target ADL candidate or specialize the scalar
domain guard. The entire forwarding primary template is checked for explicit
specializations, including specializations not chosen for a projected tile.
Source using-directives are also excluded: importing an overload from another
namespace can create the same discrepancy through ordinary unqualified lookup.

Each concrete kernel instantiation is checked. A kernel template without a
concrete instantiation is rejected. Kernel templates have exactly one named
type parameter and two unsigned integer size parameters, without defaults or
packs. A publication transaction inserts a scalar domain assertion into the
template body so future target instantiations cannot introduce a user class or
FP64 through the scalar parameter. The target storage and tile templates
independently enforce legal sizes and exact partitioning, and the device entry
checks actual block size. These constraints continue to apply beyond the
instantiations observed during conversion.

The initial proof rejects loops, constexpr branches, early returns, jumps,
inline assembly, indirect calls, unproved device helpers, nontrivial local
class lifetimes, and unproved constructors. Every scratch reference must belong
to the admitted declarations and calls. The shared object cannot escape,
change owner, or be reused by another tile variable. Multiple top-level
reductions may reuse the same scratch in the same sequence across the block.
The corresponding complete-block barriers make that sequence safe.

Authenticated profile file paths, every relevant function redeclaration, and
specialization ownership are checked. A header marked with `#pragma GCC
system_header` does not gain trust. Scratch kernels must be literal main-file
definitions. The whole TU is visited, while raw main-file auditing rejects
inactive or macro-generated storage-type uses and preprocessor directives
inside a proven kernel or forwarder. Initially the main TU admits only its two
authenticated cooperative-group includes; other preprocessor directives and
input dependencies remain outside this proof. Loaded dependency headers are
also scanned for inactive scratch type uses. Standard preprocessing is mandatory.
The publication layer stages all template assertions and rejects conflicting
edits before committing any of them.
Assertions are inserted after the complete opening-brace token, including the
legal two-byte `<%` spelling, rather than assuming a one-byte brace.

## Software algorithm

The adapter admits signed int32 and float32, block capacities 32..1024 in
complete warps, and tile sizes 32, 64, 128, 256, and 512 that divide the block.
Separate typed shared arrays avoid aliasing and object-lifetime casts.

Each warp computes a down-shuffle tree and publishes its lane-zero partial.
A block barrier publishes all partials. One leader warp per tile reduces only
the actual `G / 32` partials, publishes one result, and all block threads cross
a second barrier before reading their tile's result. A third barrier completes
every reader before scratch reuse. Every lane receives the same stored result.
All warps continue to execute the block barriers even when only leader warps
perform the second arithmetic stage.

The second arithmetic stage must not pad its reduction tree with artificial
positive zeros: doing so changes an all-negative-zero result. For G 32, this
stage performs no addition. For larger groups, all leader-warp lanes execute
each shuffle, but only valid pairs of real partials execute the addition.
Host and device oracles use exactly the real source leaves. NaNs are checked
by classification and results within a tile must have identical bits; signed
zero and infinities are explicitly covered. Floating-point tree order can
differ from sequential or exact summation. Integer inputs must avoid signed
overflow in intermediate additions.

## Scope and evidence

This extends ADR 0023 only for the separately proven scratch domain. The old
isolated DT prototype established that shared storage and barriers are usable;
its padded-zero arithmetic and handwritten caller did not prove this product
path. Current target compilation, linking, device execution, and converter
proof tests are separate requirements.

`uniform_block_reduction_host_test.cpp` invokes the actual adapter with a host
scheduler and checks two tiles sharing one block, all lanes, float and integer
reuse, negative zero, nonfinite inputs, and independent scalar sums.
`check_uniform_block_reduction.py` invokes the real converter for positive
renamed kernels and unsafe mutations, checking both new-output rejection and
preservation of a prior output. `uniform_block_reduction_probe.cce` exercises
the product adapter on the device; the CUDA conversion fixture independently
checks frontend admission and generated-code compilation.

The final macOS LLVM 23 candidate compiled all 18 translator modules and linked
successfully. Its real frontend gate accepted three sources, including renamed
identifiers and digraph braces (six concrete instantiations and twelve
collectives each), and rejected twenty-eight unsafe/preprocessor cases.
Three byte-identical controls previously accepted by older candidates (source
ADL overload, explicit function specialization, and preincluded target namespace)
were independently rerun and rejected without publishing or replacing output.
The host adapter test checked 3,328 scalar results
with zero errors. A separate temporary-header mutation restoring padded zeros
failed that same test, establishing that the signed-zero check detects the old
bug. These logs live in `cuda20_non_fp64_followup`; they are frontend and host
model evidence, not DT CCEC or device passes.

The frozen original Reduction sample still contains FP64 instantiations and
its multi-warp kernel has input loops and a `SharedMemory<T>` helper. Those
loops and that helper are outside the initial proof. A successful non-FP64
fixture does not establish that the original sample converts, compiles, or
runs. Supporting the original input loops needs an additional termination and
participation proof; removing the loops or FP64 source instantiations does not
count as an original-sample pass. TileVectorAdd remains a separate programming
model boundary. No CUDA189 expansion or PR test is part of this change.
