# CUDA compatibility boundaries

This reference describes the implemented source and target contracts, including
their limits. Experimental entries require target compilation and device
validation before use. See the [conversion reference](conversion-reference.md)
for command-line options.

Ascify preserves unsupported constructs or diagnoses them at the relevant stage.
Successful source generation alone does not establish target compilation,
linking, execution, or numerical correctness.

## Runtime lifecycle and ownership

The installed [CUDA compatibility header](../include/ascify/ascify_cuda_compat.hpp)
preserves CUDA-shaped calls while managing ACL's explicit lifecycle:

- The first admitted runtime operation attempts initialization once. Successful
  initialization is owned by Ascify; an already initialized ACL runtime is
  borrowed. Other initialization failures are cached and returned.
- The selected device is tracked per host thread, with logical zero as the
  default. Context-dependent operations restore the thread's binding when
  necessary. Explicit reset permits later rebinding.
- Cleanup resets owned bindings in reverse order and finalizes only an
  initialization owned by Ascify. The allocation-free registry tracks up to
  64 distinct bindings; exhaustion rolls back the new binding and returns an
  error rather than losing cleanup ownership.
- Exit callbacks are registered after initialization and again after device
  binding, including failed binding attempts that may create SDK state.
  Cleanup is idempotent. Registration failures roll back immediately; reset,
  finalize, and binding errors take precedence over the allocation error.

The manager is a shared C++17 inline object across translated translation units.
Multiple independent shared-object copies and unconverted global constructors
remain integration boundaries. A raw kernel launch as the very first CUDA
operation has no runtime call to initialize this manager; launch-first programs
are not covered by this lifecycle contract. Process-exit errors cannot be
returned to the caller; explicit reset is needed when cleanup status matters.

The [lifecycle test](../tests/rewrite/runtime_lifecycle_compat_test.cpp) exercises
ownership, rebinding, callback ordering, rollback, and normal process exit.

## Host and target API adaptations

| Surface | Contract and limit |
|---|---|
| Allocation, copies, and streams | Preserve admitted CUDA call shapes through typed ACL adapters and propagate errors; this is not the entire CUDA Runtime API |
| Vector constructors | Use the target's vector types and constructors; the reviewed CUDA `float3`/`uint3` record ABI is rejected before publication as described below |
| Default `dim3` | Only proven, direct, host-local declarations of CUDA's uninitialized type gain explicit `(1,1,1)` initialization; global, device, array, macro, and user-defined cases remain unchanged |
| Event recording | Explicit streams are preserved; the omitted-stream form queries the current default stream only on the admitted legacy/ACL 1.17+ surface, otherwise returns unsupported |
| Stream flags | Default creation uses the runtime adapter; nonblocking creation returns unsupported, and unknown bits return invalid-parameter status |
| Device properties | Report the actual SoC name and vector-core count without changing the current binding; publish output only after successful queries; do not invent CUDA compute capability or SM counts |
| Host math profile | Admit exact `max(float, float)` with `fmaxf` semantics and `min(int, int)`; mixed-type and double max are outside this admission |
| Half2 | Proven device-local construction and arithmetic use target constructors/intrinsics, preserving separate half multiplication and addition rounding; unsupported macros, effects, and template contexts do not receive a broad rewrite |

Parsing-only declarations in test support are not device arithmetic
implementations. The legacy RMSNorm block-affine route additionally preserves
two FP16 rounding points and confines packed weight multiplication to its
aligned cached path (`rows <= 256`, `4096 <= columns <= 8192`); warp, tail,
and streaming paths retain scalar stores. These implementation limits do not
authorize a type-only half2 transformation in arbitrary source.

Registered-symbol copies retain the original symbol object and use the SDK's
symbol lookup and transfer APIs, including overflow-safe offset/extent checks.
They require legacy SIMT headers and SDK declarations of both
`aclrtGetSymbolSize` and `aclrtMemcpyToSymbol`; an ACL version number alone is
insufficient. Without these APIs, including the compatibility headers remains
valid, but calling a symbol copy is rejected at compile time. This transfer
path remains experimental; validate generated code with the target toolchain.
Host-memory substitutes and fabricated registrations are not supported fallbacks.

### CUDA float3 and uint3

The frozen CUDA 13.4.1 CUB `util_type.cuh` and its CCCL
`cuda/std/__tuple_dir/vector_types.h` and
`thrust/type_traits/is_trivially_relocatable.h` dependencies contain dormant
`CubVector<...,3>`, vector tuple-interface, and relocation-trait definitions even in scalar
`BlockReduce<float>` programs. Their internal
declarations are exempt only when the parsed and physical bytes match the
reviewed provider, its real include ancestry reaches the reviewed physical
`cub/cub.cuh` under the explicitly configured CUDA root, and that literal
main-file include has a successfully queued mapping to `acl_cub/aclcub.hpp`.
A CUB/tuple provider parsed through an independently retained include, a copied
CUB entry, arbitrary system-header status, or a CUB path/name grants no
exemption. Caller uses of
the SDK records, including their inherited `CubVector<...,3>` ABI, remain
subject to the boundary.
After an unconditional mapped umbrella, Ascify can remove a redundant direct
literal `cub/util_type.cuh` include when Clang reports that exact directive as
skipped. The previously parsed subheader must match the reviewed physical SDK
file and bytes, and its actual include ancestry must reach that earlier
unconditionally mapped umbrella. Removal is limited to an unconditional
main-file directive on one physical line, with only whitespace or comments
after the filename; multiline block comments and extra directive tokens are
retained. This preserves the source no-op without inventing an
`aclcub/util_type.cuh` target header.
Independent or earlier subheader includes, copied providers, conditional,
macro, transitive, continued, and other unproved include forms receive no
removal authority. They retain the existing conversion boundaries, including
target missing-header failures where applicable. This does not add CUB
device algorithms or permit caller vec3 types. The
[redundant-include matrix](../tests/rewrite/check_cub_redundant_include.py)
checks the admitted directives, retained forms, and ABI rejection with output
preservation; target compilation and device execution remain separate checks.
This removal proves only that the skipped directive contributes no source
tokens. CUB/CCCL preprocessor observations, including `__has_include` queries
and version-macro conditions such as `defined(CUB_VERSION)`, are outside the
supported domain. Umbrella replacement can change their results with or
without a redundant subheader include. Ascify does not currently translate or
fully diagnose these observations; successful target compilation does not
establish their equivalence.

The frozen `cub/block/radix_rank_sort_operations.cuh`,
`thrust/detail/functional/actor.h`, `cuda/std/__memory/compressed_pair.h`,
`cub/device/dispatch/dispatch_transform.cuh`, `cuda/__iterator/zip_iterator.h`,
and `cub/detail/binary_search_helpers.cuh`
have dependent tuple `get` lookups that list vector overloads even in scalar
templates. Only when both the exact physical/parsed use site and candidate
tuple provider are proved to belong to the actually mapped CUB closure, the
boundary audits the return type rather than unselected parameter types.
The exact removed `cuda/std/__mdspan/submdspan_helper.h` provider's
tuple `get` using-declarations receive the same candidate-provider and return
checks; importing this overload set does not select a CUDA vector overload.
Concrete caller types, resolved function references, and other lookups retain
the full audit.

The reviewed CUDA 13.4.1 `float3` and `uint3` are three-field records with
size 12, alignment 4, field offsets 0/4/8, and array stride 12. Native DPP
types with the same spellings on the reviewed CANN 9.1 target are extended
vectors with size and alignment 16. Leaving those spellings unchanged cannot
preserve the CUDA record ABI.

Ascify diagnoses observed uses of those reviewed records with
`CUDA vec3 ABI boundary` and publishes no translated output. This includes
pointer/array declarations, aliases, layout observations, helper results with
deduced types, function signatures, and dormant template patterns. Existing
complete output is preserved on failure. Ordinary scalar types, `dim3` scalar
construction, vector2/vector4, and distinct user records named `float3` or
`uint3` do not match this rejection rule.

This is a bounded rejection profile, not an automatic adapter or a guarantee
for every SDK. Recognition requires the parsed `vector_types.h` buffer to
match the reviewed CUDA 13.4.1 bytes, the canonical global record identity,
and a complete record definition. Diagnostics report the actual AST layout;
a source `#pragma pack` that changes it is also rejected. Exact reviewed SDK
and Clang 23 provider
buffers are exempt from use auditing so implicit SDK declarations alone do
not reject scalar programs. Other system headers, `-isystem`, `#pragma
system_header`, and presumed `#line` filenames do not grant exemptions.
The actual include ancestry of the SDK definition must also use the reviewed
Clang CUDA runtime wrapper, when one is present. The parsed builtin-variable
provider declarations must likewise use their reviewed bytes. An unknown
wrapper or builtin-variable provider disables this bounded rejection profile
and preserves existing conversion behavior;
it does not certify vec3 ABI compatibility. Direct frozen SDK record includes
without a runtime wrapper remain within the observed-use boundary.
The hash check requires LLVM 13 or newer. Unknown or modified SDK headers
remain outside this rule and receive no vec3 rewrite authority; successful
generation on those headers does not prove their target ABI.

An explicit three-field storage adapter needs separate source provenance,
type identity, overload/ADL, template, and ABI escape proofs before it can be
automatically substituted. Native vector consumers and `uint3`/`dim3`
conversions need their own explicit semantics. The
[real frontend matrix](../tests/rewrite/check_vec3_abi_boundary.py) checks
rejection, legal controls, and output preservation; this safety rejection
does not count as a conversion or object success-rate gain.

Relevant checks: [target ABI](../tests/rewrite/check_target_abi_compat.sh),
[device properties](../tests/rewrite/check_device_properties.sh),
[host math](../tests/frontend_compat/check_frontend_host_math.py),
[half2](../tests/rewrite/check_half2_compat.py), and
[symbol copies](../tests/rewrite/check_symbol_compat.sh).

## Frontend admission and cooperative groups

`--frontend-compat=none` leaves normal CUDA include resolution in place.
`ascify-admitted-v1` explicitly selects the checked-in parsing profile.
Its manifest, exact bytes, and directory layout are verified, and admitted
includes must resolve to those verified files. Local shadows, altered profiles,
unlisted files, and unadmitted cooperative-group subheaders are rejected.
An installed or relocated executable cannot replace a missing/damaged installed
profile with an unrelated source-tree copy.

The current profile includes thread-block operations, a restricted 32-thread
tile, typed reductions, and a distinct scratch-reduction projection. Target
implementations live in the compatibility headers; parsing declarations alone
do not provide those operations.

| Operation | Admitted domain |
|---|---|
| Thread-block synchronization | The exact block operation uses a real block barrier; generic group synchronization is not substituted with a block barrier |
| Ordinary tiles | Size 32, partitioned from a thread block; preserve typed/erased metadata and rank/size signatures; tile synchronization is not a general collective barrier |
| Register shuffles | Exact int32, uint32, and float types; `uint4` XOR shuffle transfers all four 32-bit components |
| Tile reduction | `plus<int>` and `plus<float>` across 32 participating lanes; a down-shuffle tree followed by broadcast gives every lane the same result bits |
| Legacy masked ballot/down-shuffle | Every active lane supplies the same mask, equal to the complete currently active cohort; partial down-shuffle supports int32/uint32/float and widths 1, 2, 4, 8, 16, 32 |

Masked operations do not implement CUDA's rendezvous for lanes arriving from
different branches. Invalid cohorts trap. Physical source-lane addressing is
preserved; results from a source outside the participating mask are unspecified
and are not replaced with zero. Existing full-mask int64/uint64 shuffle paths
retain their separate domain. Other SDK header families do not inherit these
legacy extensions merely because a similarly named primitive exists.

Single-warp reduction requires every lane in that tile to participate; another
warp may bypass it. Floating-point tree order is not sequential summation, and
signed integer inputs must avoid intermediate overflow.

See [frontend-profile checks](../tests/frontend_compat/check_frontend_compat.sh),
[cooperative-group tests](../tests/rewrite/check_cooperative_groups_compat.sh), and
[mask tests](../tests/rewrite/check_masked_warp_compat.sh).

## Uniform block scratch reduction

This experimental adapter requires target compilation and device validation
before use. It does not enable generic multi-warp tiles.

The narrow C++17 form uses one shared `block_tile_memory<B>`, an `auto` block,
an `auto` tile, and one verified forwarding template for typed reduction.
Standard preprocessing and a whole-translation-unit proof are mandatory because
the parser projects these uses into different internal types. Source cannot
observe or specialize those types, introduce conflicting ADL/overloads or
namespace declarations, or let scratch escape. Loops, conditional participation,
early returns, indirect calls, unproved helpers, and unproved constructors are
rejected. All template instances and future scalar instantiations remain guarded.

The adapter admits int32/float32, block capacities 32..1024 in complete warps,
and tile sizes 32, 64, 128, 256, or 512 that exactly divide the block. Actual block
size is checked. Three complete-block barriers protect publication of warp
partials, result reads, and scratch reuse; all block threads must follow the
same reduction sequence. The second reduction stage uses only real partials,
without padding with positive zero, to preserve all-negative-zero inputs.
Every lane receives its tile's same stored result. Floating-point ordering and
signed-overflow limits still apply.

The [AST gate](../tests/frontend_compat/check_uniform_block_reduction.py) covers
accepted inputs and unsafe mutations. The
[host adapter model](../tests/rewrite/uniform_block_reduction_host_test.cpp) checks
all-lane results, reuse, and special values. These checks do not replace the
separate [device probe](../tests/rewrite/uniform_block_reduction_probe.cce).

## Device memory and atomics

Global atomic rewriting requires a direct call to an admitted CUDA system
declaration in the current kernel, a directly spelled main-file call, and an
address rooted at that kernel's pointer parameter. Supported operations use
matching signed or unsigned 32-bit integer types; increment/decrement require
unsigned values. Local aliases, helper parameters, device-global variables,
shared/private storage, macros, float, and 64-bit operands do not receive a
global wrapper. Unsupported target header families reject wrapper instantiation.
Shared atomic lowering remains unadmitted; a global-memory cast or non-atomic
read/modify/write cannot substitute for it.

Private object copying is experimental and requires target/device validation.
It accepts only a trusted `memcpy` declaration or verified forwarding
builtin wrapper, standard preprocessing, and two distinct, thread-private,
non-volatile, trivially copyable 64-byte objects in a non-template device
function. The source is a local array and the length is `sizeof(destination)`.
Parameters, aliases, casts, macros, shared objects, and dynamic lengths remain
outside the proof. The generated byte-copy helper returns the destination and
asserts both target object sizes; it adds no allocation, synchronization, or
address-space cast. This is not a general device `memcpy` implementation.

Device FP64, including the double `CUDART_INF` and `CUDART_NAN` constants, is
unsupported. Their mappings do not establish valid double precision values or
NaN semantics.

Relevant checks: [global atomics](../tests/rewrite/global_atomic_compat_test.cpp),
[atomic rejection](../tests/rewrite/global_atomic_fail_closed_test.cpp),
[private-copy host contract](../tests/rewrite/check_device_memory_compat.sh), and
[private-copy conversion](../tests/rewrite/check_device_memory_translation.sh).

For source-level NVIDIA helper handling, see [CUDA Samples helpers](sample-helpers.md).
For proof-gated whole-operator acceleration, see the
[row-wise Hybrid guide](rowwise-simd-conversion.md).
