# CUDA compatibility boundaries

The admitted frontend profile's physical `cooperative_groups.h` can establish
the origin of its exact CUDA device-qualifier macro chain and fixed CUDA control
condition when a Runtime header is removed. Each include instance must retain
the product bytes and authenticated macro context. Independent caller uses of
the same SDK macro names remain rejected. With ordinary CUDA preprocessing,
the exact forced product `host_math.h` can also establish its four literal CUDA
control conditions. The include must come from the compiler predefines, with
unchanged physical bytes, macro expansion history and include-guard ownership.
Caller guard manipulation, macro-stack operations, conditional-retention mode,
PCH, modules and external AST inputs cannot use this admission.

The full Runtime type audit may retain dependent observations in original,
uninstantiated primary and partial template patterns in that authenticated
host-math header and the main product CG header. Concrete instantiations,
explicit specializations, caller templates, SDK reflection and ordinary body
checks remain audited. Within the authenticated CG pattern, the direct
`uint4` type argument of its dependent `std::is_same<T, uint4>` selector may
retain its identity after the native template, SDK record, declarations and
exact argument location are proved. The SDK's own one-level `typedef struct
uint4 uint4` spelling also requires every typedef redeclaration and its direct
record source type to belong to the authenticated SDK; caller aliases and
longer typedef chains are excluded. This grants no `uint4` layout, arithmetic,
caller type-query or instantiated-interface support. The exception lasts only
for that template-id's TypeLoc traversal. `cooperative_groups/reduce.h` is outside this additional
full Runtime rule. This does not establish device execution or complete-project
support.

A member template copied into an implicitly instantiated product CG class can
still have uninstantiated parameters. A one-step Clang member-template origin
and class-pattern identity proof may retain those parameters and that exact
templated function declaration. All redeclarations must belong to the frozen
CG profile. This scope excludes the member template's concrete specializations
and excludes explicitly instantiated or explicitly specialized classes; their
arguments, interfaces, bodies and SDK observations retain the ordinary audit.

An adjacent local Runtime status assignment may also call a visible ordinary
host function whose only statement returns an already admitted Runtime API
status unchanged. Every declaration must belong to the physical main input or
a selected local-header closure; the complete function range must be free of
macros, directives and opaque pragma operators. Templates, indirect calls,
default arguments, alternate result expressions and semantic/linkage
attributes are not admitted. This proves the status domain only: the original
API arguments and success/error values remain intact, and the existing full
helper transaction, alias/capture audit and closure publication gates remain.
The source check includes leading declaration tokens even when Clang ignores
an attribute and omits it from the AST range. GNU/MS/C++ attribute spellings
are rejected, as are externally defined macros in that prefix. The conservative
prefix scan rejects a directive after pending declaration syntax; inactive
branch punctuation cannot become a trusted declaration boundary.

The global `atomicInc` mapping can additionally prove a directly addressed
file-scope device counter with direct builtin `unsigned int` type, 32 bits,
one complete definition and literal zero initialization. Its device qualifier
must come from the actual configured CUDA SDK, and its declaration and address
must have physical macro-free source evidence. Casts, aliases, arrays, shared
or constant storage, qualifiers, redeclarations and variable templates remain
outside this additional rule. A same-module symbol resolver must commit before
output publication. This does not add new atomic/fence Runtime code or establish
cross-block execution or repeated-reset correctness on a target device.

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

The complete original CUDA Samples v13.0 `simpleMPI` target passed
[two-rank original-target validation](validation/simple-mpi-20261105.md) using
the existing runtime lifecycle and copy adapters. The unchanged MPI host TU is
compiled and linked against a real MPI implementation. The validated scope is
one host and two processes on the same default logical device.

## Host and target API adaptations

| Surface | Contract and limit |
|---|---|
| Allocation, copies, and streams | Preserve admitted CUDA call shapes through typed ACL adapters and propagate errors; this is not the entire CUDA Runtime API |
| Vector constructors | Use the target's vector types and constructors; the reviewed CUDA `float3`/`uint3` record ABI is rejected before publication as described below |
| Default `dim3` | Only proven, direct, host-local declarations of CUDA's uninitialized type gain explicit `(1,1,1)` initialization; global, device, array, macro, and user-defined cases remain unchanged |
| Grid-constant value parameters | A bounded const int-array record may lose `__grid_constant__` only after every use and declaration proves value-only behavior; address, reference, copy and reflection observations are rejected; see the [parameter contract](grid-constant-parameters.md) |
| Event recording | Explicit streams are preserved; the omitted-stream form queries the current default stream only on the admitted legacy/ACL 1.17+ surface, otherwise returns unsupported |
| Event queries and host functions | On the reviewed legacy/ACL 1.17.0 surface, use real Ex-event completion state and ordered callbacks on explicit streams; see the [runtime contract](runtime-profiling-and-host-functions.md) |
| Profiling | An explicit `ASCIFY_PROFILE_DIR` opts into owned CANN collection; link `msprofiler`; unsupported configurations return an error, as described in the [runtime contract](runtime-profiling-and-host-functions.md) |
| Stream flags | Default creation uses the runtime adapter; nonblocking creation is admitted on the reviewed legacy SIMT / ACL 1.17.0 surface and remains unsupported elsewhere; unknown bits return invalid-parameter status |
| Stream priorities | Query the native range and create named nonblocking streams at a clamped native priority on the reviewed legacy SIMT / ACL 1.17.0 surface; see the [priority contract](stream-priorities.md) |
| Device properties | Report the actual SoC name and vector-core count without changing the current binding; publish output only after successful queries; do not invent CUDA compute capability or SM counts |
| Host math profile | Admit exact `max(float, float)` with `fmaxf` semantics, `min(int, int)` / `max(int, int)`, and `min(int, unsigned int)` / `min(unsigned int, int)` returning unsigned int; other mixed types and double max remain outside this admission |
| Half2 | Proven device-local construction and arithmetic use target constructors/intrinsics, preserving separate half multiplication and addition rounding; unsupported macros, effects, and template contexts do not receive a broad rewrite |
| Packed-byte SAD | Experimental lowering of one proved literal `vabsdiff4.u32.u32.u32.add` instruction to unsigned integer arithmetic; target compilation and device validation are separate requirements |

The two mixed integer `min` orders follow CUDA's unsigned comparison domain:
convert the signed operand to `unsigned int`, then select the unsigned minimum.
Thus `min(-1, 1u)` returns `1u`, while `min(-1, UINT_MAX)` returns `UINT_MAX`.
Only the callee of an authenticated direct host call is rewritten to
`ascify::cuda_host_min_unsigned`, whose two value parameters have unsigned int
type; the argument expressions remain intact and are
evaluated once. The exact product header, physical owner, macro and host-context
checks apply to these overloads as they do to the existing host math calls.

Ordinary host templates that only exchange two pointer values can receive a
closed type-domain proof. The rule checks a single type parameter, two
references and the complete three-statement copy/assignment body by AST; it
does not depend on the function or file name. Every original use and concrete
instance must close over directly spelled builtin pointers. The emitted
`static_assert` restricts later primary-body instantiations to that proved type
set. An entirely unreferenced template can retain its body with an empty-domain
guard; signature/address/using/extern observations must still pass the separate
reference inventory. This contract covers the current translation unit and
does not infer a type domain from another source file.

Caller aliases, erased or escaped pointer origins, unproved allocators,
redefinitions and macro-controlled patterns receive no such proof. CUDA
allocation evidence requires the authenticated physical SDK declarations and
exact signature. Normal SDK record, interface and reflection checks remain
active. The guard cannot distinguish typedef spellings of one identical C++
type; original-source provenance is checked separately. See the
[host pointer tests](../tests/rewrite/check_host_pointer_swap.py).

The packed-byte SAD rule accepts the exact decoded instruction string
`vabsdiff4.u32.u32.u32.add %0, %1, %2, %3;`, one unnamed `=r` output and three
unnamed `r` inputs, without volatility, goto or clobbers. The owner must be a
non-template device/global function. The output must be a writable private
automatic unsigned-int32 local; inputs are unsigned-int32 private local values,
parameters or unsigned integer literals. Attributed/register variables, references,
volatile values, side effects, casts, calls, raw directives, pragma operators,
escaped identifier spellings and macro-expanded statement tokens
remain outside the rule. Adjacent
ordinary string literals may form the exact string, including in a separately
proved selected local header. Function names do not determine admission.

For inputs A, B and C, the helper adds C to the four absolute differences between
corresponding unsigned bytes of A and B, keeping the low 32 bits. It uses shifts,
masks and unsigned arithmetic, with a static 32-bit type check. This follows the
default byte selection and secondary accumulation in the [PTX video instruction
contract](https://docs.nvidia.com/cuda/parallel-thread-execution/#simd-video-instructions-vadd4-vsub4-vavrg4-vabsdiff4-vmin4-vmax4).
Signed variants, explicit selectors, saturation and multi-instruction assembly
remain unchanged. Host arithmetic tests establish the helper's integer result;
they do not establish CCEC compilation, device execution or sample correctness.

On CANN 9.1, an ordinary ACL stream has no implicit synchronization with the
default stream, matching `cudaStreamNonBlocking`. Ascify calls `aclrtCreateStream`
through its existing lifecycle adapter; it does not forward CUDA bit 0 to ACL's
unrelated fast-launch option. This does not emulate CUDA legacy-default-stream
implicit ordering for blocking streams. Programs requiring that ordering remain
outside the supported contract. See the [CANN 9.1 stream
contract](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0011.html).

For ordinary `cudaMemcpy`, every device-memory direction first waits for the
current default stream, then performs the immediate ACL copy; synchronization
failures prevent the copy and are returned unchanged. H2H does not wait for
device work. D2D retains the adapter's existing blocking behavior: CUDA may
queue D2D without a host wait, but doing that here would also require a lifetime
change because `aclrtFree` does not wait for outstanding memory users. This
conservative D2D path is not a performance or host-asynchrony equivalence claim.
Independent nonblocking streams are not synchronized. The legacy cross-stream
limitation above still applies; symbol copies and other memory entry points
have their own contracts. See [CUDA copy synchronization](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html),
[CANN default streams](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/programug/acldevg/runtime_doc_dev_0013.html),
and [CANN immediate free](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_0100.html).

Parsing-only declarations in test support are not device arithmetic
implementations. The legacy RMSNorm block-affine route additionally preserves
two FP16 rounding points and confines packed weight multiplication to its
aligned cached path (`rows <= 256`, `4096 <= columns <= 8192`); warp, tail,
and streaming paths retain scalar stores. These implementation limits do not
authorize a type-only half2 transformation in arbitrary source.

Symbol copies retain the original device storage and check offset and extent
without integer overflow. For an admitted complete file-scope CUDA device or
constant definition, the converter emits a small address kernel in the same
translation unit and registers its host resolver. The kernel obtains the
actual device address; real allocation, launch, synchronization, copy and free
operations propagate their errors. Device addresses are resolved afresh for
each transfer, so reset or context changes cannot reuse a cached address.

The definition must have a literal main-file name and complete, nonvolatile,
trivially copyable storage without const subobjects. Templates, inline
variables, anonymous namespaces, extern-only declarations, references and
unproved storage layouts do not receive a generated resolver. Generated names
and published macros are checked for collisions. Other symbol tokens retain
the native SDK lookup and its errors; no host-memory substitute is used.

`cudaMemcpyToSymbol` admits host-to-device and device-to-device transfers;
`cudaMemcpyFromSymbol` admits device-to-host and device-to-device transfers.
The current contract requires legacy SIMT headers and SDK declarations of
`aclrtGetSymbolSize` and the corresponding symbol transfer API. An ACL version
number alone is insufficient. Missing APIs reject a transfer at compile time
without making inclusion of the compatibility headers invalid. Validate the
generated module with the target toolchain. See the [source boundaries](../tests/rewrite/check_symbol_conversion.py)
and [transfer contracts](../tests/rewrite/check_symbol_compat.sh).

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

Texture projection may retain the uninstantiated SFINAE utilities in the two
exact product CG headers. Each physical inclusion must preserve the CUDA
controls, the verified SDK device-attribute expansion chain, and the global
`cooperative_groups` namespace. Foreign macro expansions or control changes
keep the texture audit closed. This exception does not admit caller templates,
skip concrete template instances or SDK layout observations. The separate
full Runtime rule described above additionally tracks original template
patterns versus instantiations; it admits only the main CG header, not
the reduction header.

The current profile includes thread-block operations, a restricted 32-thread
tile, typed reductions, and a distinct scratch-reduction projection. Target
implementations live in the compatibility headers; parsing declarations alone
do not provide those operations.

| Operation | Admitted domain |
|---|---|
| Thread-block synchronization | The exact block operation uses a real block barrier; generic group synchronization is not substituted with a block barrier |
| Ordinary tiles | Size 32, partitioned from a thread block; preserve typed/erased metadata and rank/size signatures; member/free synchronization requires all 32 lanes at the same call; typed and erased tiles share this restriction |
| Register shuffles | Exact int32, uint32, and float types; `uint4` XOR shuffle transfers all four 32-bit components |
| Tile reduction | `plus<int>`, `plus<float>` and exact `plus<double>` across 32 participating lanes; a down-shuffle tree followed by broadcast gives every lane the same result bits. Double uses the integer binary64 RNE backend. |
| Legacy masked ballot/down-shuffle | Every active lane supplies the same mask, equal to the complete currently active cohort; partial down-shuffle supports int32/uint32/float and widths 1, 2, 4, 8, 16, 32 |

Masked operations do not implement CUDA's rendezvous for lanes arriving from
different branches. Invalid cohorts trap. Physical source-lane addressing is
preserved; results from a source outside the participating mask are unspecified
and are not replaced with zero. Existing full-mask int64/uint64 shuffle paths
retain their separate domain. Other SDK header families do not inherit these
legacy extensions merely because a similarly named primitive exists.

Tile synchronization orders shared/global memory publication and consumption
within one complete warp. On the verified 950 legacy SDK, the native tile sync
provides only a memory fence, so this facade uses a block-scope fence, a full-warp
ballot rendezvous, and a second fence. Another warp may bypass the call or use a
different iteration count. A partial active mask traps; lanes arriving from
different divergent branches and multi-warp tiles are not admitted. Device
producer/consumer tests cover both memory spaces, repeated reuse, bypassing
warps, and typed/erased member/free forms. The available target disassembler
does not decode the SIMT instructions; no instruction-level observation is
claimed.

Single-warp reduction requires every lane in that tile to participate; another
warp may bypass it. Floating-point tree order is not sequential summation, and
signed integer inputs must avoid intermediate overflow.

See [frontend-profile checks](../tests/frontend_compat/check_frontend_compat.sh),
[cooperative-group tests](../tests/rewrite/check_cooperative_groups_compat.sh), and
[mask tests](../tests/rewrite/check_masked_warp_compat.sh).

## Dynamic register groups

The bounded coalesced, partition, query, integer broadcast and reduction source
domain is described in the [dynamic group contract](dynamic-warp-groups-source.md).
The complete original CUDA Samples v13.0 `binaryPartitionCG` target passed
[original-target NPU validation](validation/binary-partition-cg-20261103.md).
Eligible kernel signatures receive a proved [launch ABI tail completion](dynamic-kernel-abi-tail.md).
Other group algorithms, architecture-selected regions and launch shapes retain
their separate conversion and device-validation boundaries.

## Uniform block scratch reduction

This bounded adapter passed the [complete original reduction validation](validation/reduction-20261103.md).
Other inputs still require target compilation and device validation. It does
not enable generic multi-warp tiles.

The narrow C++17 form uses one shared `block_tile_memory<B>`, an `auto` block,
an `auto` tile, and one verified forwarding template for typed reduction.
Standard preprocessing and a whole-translation-unit proof are mandatory because
the parser projects these uses into different internal types. Source cannot
observe or specialize those types, introduce conflicting ADL/overloads or
namespace declarations, or let scratch escape. A closed grid-stride or zero-based
builtin induction loop may remain outside every collective and block barrier.
Its induction, stride and invariant bound cannot form aliases or be mutated by
the body beyond the canonical step; preformed pointer/reference aliases and
reference-typed bounds are refused. This preserves source loops and their
arithmetic/nontermination preconditions. Conditional participation, early
returns, indirect calls, unproved helpers, and unproved constructors remain
rejected. All template instances and future scalar instantiations remain guarded.

The adapter admits int32/float32 and exact binary64 double, block capacities
32..1024 in complete warps,
and tile sizes 32, 64, 128, 256, or 512 that exactly divide the block. Actual block
size is checked. Three complete-block barriers protect publication of warp
partials, result reads, and scratch reuse; all block threads must follow the
same reduction sequence. The second reduction stage uses only real partials,
without padding with positive zero, to preserve all-negative-zero inputs.
Every lane receives its tile's same stored result. Floating-point ordering and
signed-overflow limits still apply. Double partials/results use separate volatile
uint64 slots; the declared arrays total `B` payload bytes for block capacity
`B`. Actual target layout, dynamic shared memory and total compiler resources
require independent verification.

A bounded empty shared-pointer wrapper may retain its selected conversion to
`extern __shared__ int[]` for int/float or a separate double array for double.
The proof checks the actual constructor/conversion and const interface; it does
not exempt an entire class or an unselected method from the existing audits.
Target compilation must independently validate the resulting `__ubuf__`
address spaces, alignment, dynamic launch bytes and total resource use.

Raw preprocessing additionally permits a complete, sole empty outer guard,
one directly resolved native `/usr/include/stdio.h`, and one exact inactive
`__CUDA_ARCH__ >= 800` integer specialization with its SDK and AST witnesses.
The source condition is retained. An inserted target `#error` rejects a target
compilation that defines `__CUDA_ARCH__`; caller controls, other alternatives
and hidden scratch references remain refused. See the
[decision record](decisions/002-bounded-binary64-cg-reduction.md).

The [AST gate](../tests/frontend_compat/check_uniform_block_reduction.py) covers
accepted inputs and unsafe mutations. The versioned
[extended matrix](../tests/frontend_compat/check_uniform_extended_domain.py)
adds 47 source-syntax-checked cases (12 admitted, 35 refused), including loop,
shared conversion, raw control and atomic-publication boundaries. The
[host adapter model](../tests/rewrite/uniform_block_reduction_host_test.cpp) checks
all-lane results, reuse, and special values. These checks do not replace the
separate [device probe](../tests/rewrite/uniform_block_reduction_probe.cce).

## Device memory and atomics

For `dav-c310-vec`, a fixed-size, directly spelled `__shared__` array of builtin
16-bit `short` or `unsigned short` can receive `ASCIFY_ALIGN(4)` on its `__ubuf__`
object. The AST proof closes local pointer aliases and direct source device
helpers before changing that qualifier. Element types, array extents, `sizeof`,
initializers and index strides remain unchanged. Existing explicit alignments
keep their original value; this rule does not raise an explicit 1/2-byte or
unknown alignment. Macro-spelled declarations, dynamic/extern arrays, records,
templates and multiple declarators receive no new alignment admission.
Object alignment reflection, numeric address/layout observations and pointer
escapes reject the new transformation. A local-header candidate and its helper
bodies require the existing complete source-context transaction.

This object-placement rule does not repair an unaligned element offset or
establish device correctness by itself. The
[syntax-first checks](../tests/rewrite/check_shared_word_alignment.py) include
an optional CCEC object-only check of actual UB section size and alignment;
no check in that script launches a device program.

Global atomic rewriting requires a direct call to an admitted CUDA system
declaration in the current kernel, a directly spelled main-file call, and an
address rooted at that kernel's pointer parameter. Supported operations use
matching signed or unsigned 32-bit integer types; increment/decrement require
unsigned values. Local aliases, helper parameters, device-global variables,
shared/private storage, macros, float, and 64-bit operands do not receive a
global wrapper. Unsupported target header families reject wrapper instantiation.
A separate generic `atomicAdd` path accepts matching signed or unsigned 32-bit
operands rooted at the current pure-device, non-template helper's pointer
parameter. It uses the target's public generic atomic operation for the actual
GM or UB address space, without a global-memory cast. Direct shared-variable
expressions, local aliases, other atomic families, host/device helpers,
qualified or macro calls, floating-point and 64-bit operands remain outside
this proof. The reviewed legacy pure-SIMT header surface is required; other
header families fail wrapper instantiation. See the
[conversion boundaries](../tests/rewrite/check_generic_atomic_conversion.py)
and [target rejection contract](../tests/rewrite/generic_atomic_fail_closed_test.cpp).

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

### Host standard-math calls

The unqualified `exp`, `log`, and `sqrt` spellings of resolved standard-library
calls remain unchanged inside proven host-only, non-template functions. This
includes ordinary automatic-local initializers and preserves the selected
float/double overload, argument conversions, and evaluation count. The proof
requires trusted system declarations and a direct, non-macro main-file callee
token. It emits no replacement expression.

This narrow preservation rule does not change CUDA device/global or combined
host/device math mapping. Explicitly qualified math already retains its source
spelling. CUDA-only `rsqrt`, macro-generated callees, templates, lambdas,
static/TLS/global initializers, and project overloads remain outside this rule;
their pre-existing translation behavior is unchanged. In particular, the old
float narrowing remains an unfixed limitation for those host contexts; passing
a boundary test does not establish semantic correctness or target support for
them. Regression:
[`check_host_standard_math.py`](../tests/rewrite/check_host_standard_math.py).

### Real memory and restricted kernel thread limits

On the `dav-c310-vec` legacy SIMT family, `cudaGetDeviceProperties` exposes
`totalGlobalMem` from CANN's actual total-global-memory device query,
`maxThreadsPerBlock` from `ACL_DEV_ATTR_MAX_THREADS_PER_BLOCK`, and
`maxGridSize[0..2]` from `ACL_DEV_ATTR_MAX_GRID_DIM_X/Y/Z`. These are device
limits; they do not claim a particular kernel can use the maximum or bypass
the total-block-count limit. All four SIMT limit queries must succeed with a
positive value representable by `int`; otherwise the entire property call
fails explicitly and leaves the caller's object unchanged. This also means
older runtimes/devices without these queries cannot provide this extended
property surface. No CUDA SM version metadata is added.

Property queries support Runtime logical Device 0, including before any
`cudaSetDevice` call. They do not require or change the caller's current device
or context. CANN's `aclrtGetSocName` always describes Device 0; requests for a
different device therefore return `ACL_ERROR_FEATURE_UNSUPPORTED` without
writing the output, even if that other device is currently bound. Binding it
temporarily would not make the name query describe it. This boundary follows
the [CANN 9.1 name-query contract](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_0048.html).

A restricted `cudaFuncGetAttributes` conversion can publish
`maxThreadsPerBlock` when the CUDA SDK call names a concrete, defined,
non-template global kernel at translation-unit scope in the input or owned
local-header closure; namespace kernels are retained. The
compiler bound is 1024 without a launch attribute, or a literal single-argument
`__launch_bounds__(N)` in 1..2048. The runtime result is the smaller of that
per-kernel bound and the active device's real thread limit. CCEC's pure-SIMT
contract defines register allocation from this bound; a device-wide limit alone
is not kernel metadata.

The kernel reference must retain its physical source token through any macro
argument expansion and spell the canonical qualified kernel name (optionally
with `&` or parentheses); macro bodies and namespace aliases cannot select the
kernel. The kernel must have a single definition with no forward declarations,
in a physical file without
conditional preprocessing, and its signature may use only the actual SDK
`__global__` and `__launch_bounds__` macros. This conservative boundary also
retains kernels declared in guarded headers. Kernel definition and include
selection must remain identical under source and target preprocessing;
conditional selection in parent include files is outside this support boundary
and is not fully audited automatically. The attribute output
must be one automatic local object declared using the direct SDK record name,
with no written initializer or an empty `{}` initializer, used only by that
query and by value reads of
`maxThreadsPerBlock`, including parenthesized and lvalue conditional selections
whose result is read by value (such as a `MIN` macro). The complete declaration statement and query call must be
outside conditional preprocessing and contain no macro invocations or
preprocessor directives. Nonempty or copy initialization, copies, aliases, captures,
escapes, layout observations through `offsetof`, other field observations and
unproved maximum-field uses refuse publication.
Only this field is written after all runtime queries succeed. This is not full
`cudaFuncAttributes` support and does not supply static shared-memory metadata.
The legacy shared-size adapter is unchanged.

## Device dynamic parallelism (experimental)

Device-side `<<<...>>>` launches inside `__global__`/`__device__` functions
are queued for host replay: the call is rewritten to an atomic slot claim in
a device-global task ring, and `cudaDeviceSynchronize` drains the ring and
replays every entry through typed per-kernel launch glue emitted beside the
kernel declarations. Device-side stream creation/destruction, error queries,
`printf` and device `cudaMemcpyAsync` route to queue-faithful stubs (a memcpy
task replays as a device-to-device copy). PTX inline asm (`bfind.u32`) is
rewritten to the equivalent software bit scan. `cdpSimpleQuicksort` is the
validated reference.

## CUDA graphs (experimental)

The target has no native graph API; the compat layer records kernel, memcpy,
memset, memalloc, memfree and host-callback nodes and replays them in
topological order on `cudaGraphLaunch`. Manual kernel nodes snapshot their
`kernelParams` at add time through a typed `<<<>>>` bridge, preserving CUDA
copy-at-add semantics. Stream capture diverts the launch shims and records
the same node kinds. Memalloc nodes assign addresses synchronously and
reproduce virtual-address reuse across graph relaunches. Concurrency between
independent branches is not reproduced (sequential replay); `simpleCudaGraphs`
and `graphMemoryNodes` are the validated references.

## Cross-device access

Peer detection/enable and device-to-device copies run natively. Kernel-arg
pointers owned by a different device are refreshed into local mirrors at
launch time (the device cannot dereference peer global memory on this
target), covering regular and pool allocations. Cross-device stream-event
waits are host-synchronized because the native edge never advances.
`simpleP2P` and `streamOrderedAllocationP2P` are the validated references.

## Block-scope barriers and warp votes

`cuda::barrier<cuda::thread_scope_block>` has an admitted parse provider and
a sense-reversing rendezvous implementation on the admitted pure-SIMT
atomics; coalesced-group `ballot` is a masked warp vote. `simpleAWBarrier`
and the CG warp-vote samples are the validated references.
