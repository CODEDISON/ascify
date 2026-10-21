# Bounded texture conversion

The texture adapter is under validation. Earlier scalar engineering evidence
does not establish acceptance of a later candidate. The RGBA8 extension and
descriptor ABI v2 described below have not yet completed their new host,
target or device gates; they add no original CUDA Sample success.

## Experimental RGBA8 extension

An authenticated SDK `uchar4` channel factory, or an explicit
`{8,8,8,8,cudaChannelFormatKindUnsigned}` description, selects four unsigned
byte channels. Array, layered-2D array and adapter-owned Pitch2D resources use
four bytes per texel. NormalizedFloat reads return native `float4`, with every
channel including alpha divided by 255, using Point or Linear filtering.
ElementType reads return native `uchar4` with exact byte values and require
Point filtering. No color-space conversion, channel reordering or alpha
replacement occurs. A Pitch2D view does not change the allocation's other
views. Return-type/read-mode mismatches report a sticky error.

The frontend proves canonical SDK identity, all redeclarations, global owner,
four component fields and their exact layout. Aliases of that same SDK type
are admitted; copied SDK headers, caller records, redeclarations and structural
lookalikes gain no authority. This rule does not grant general record-layout
reflection. `cudaCreateChannelDesc<float4>`, U8 scalar, signed/16-bit channels,
3D, mipmaps, surfaces and sRGB remain outside this extension. Native vector
template support requires the admitted pure-SIMT header/compiler domain;
ordinary host C++ and public-8.5 scalar includes receive no synthetic vectors.

RGBA interpolation has a separate coordinate and arithmetic contract. The
planned `2^-21` absolute-error gate compares against double interpolation at
the **same FP32-transformed coordinates**; it is not a bound on normalized
coordinate rounding or on selecting a different texel. Point normalization is
checked to one ULP of the correctly rounded reference, with exact 0/1 endpoints.
Arbitrary CUDA hardware quantization, tie handling and bitwise equivalence are
not promised. Original image tolerances must remain unchanged.

For a negative normalized-wrap coordinate, FP32 subtraction can round its
wrapped fraction to 1. For example, the exact fraction for `-2^-149` is
`1-2^-149`: Point should select the final texel, while rounding to 1 and wrapping
again would select the first. RGBA rejects this unresolved case before a texel
load in both precise and fast modes. Legacy F32/U32 coordinate behavior is
preserved. Negative zero selects texel zero; clamp bounds large finite inputs
before integer conversion. NaN/Inf and transformed values outside the bounded
coordinate domain are rejected. CUDA equivalence claims require an input domain
whose coordinate transformation and texel selection have been independently
established; merely passing a software interpolation tolerance is insufficient.

## Descriptor ABI v2: rebuild all consumers

The private descriptor remains 72 bytes, aligned to 8, with its sticky-error
pointer at offset64 in a 96-byte allocation. Its channel and filter fields now
carry nonzero versioned tags and the former padding at offset60 is an explicit
version. These tags apply to **all F32, U32 and RGBA textures**. New samplers
validate them before loading texels; old float and U32 samplers reject v2 tags
through their existing guards. Old zero-padding v1 descriptors are rejected by
new samplers. Such rejection is a compatibility safeguard, not permission to
mix installations.

Rebuild every translated TU, linked library and raw-descriptor probe that uses
texture helpers against one complete installation. Recreate texture handles;
do not carry live descriptors between versions. Bind the full installed helper
inventory as well as the converter binary when collecting acceptance evidence.

## Unsigned 32-bit element extension

The additional channel is the exact CUDA `unsigned int` type, with a 32-bit
single-component unsigned channel description. Array and layered-array storage,
and an existing adapter-owned `cudaMallocPitch` allocation, retain the same
four-byte element width and byte-pitch checks as float storage. An unsigned
texture accepts only element-type reads and point filtering; normalized integer
reads, linear filtering, smaller integer channels and vector channels remain
outside this extension. Float and unsigned fetch types are checked against the
descriptor, including calls selected by the fast-math policy. The integer
payload is loaded and returned as `unsigned int`, with no float intermediate.

Coordinate normalization, wrap/clamp and lifetime rules are unchanged. A type
mismatch or invalid coordinate sets the existing sticky error; an unsigned
failure returns zero only as a placeholder, and the blocking error status must
still be checked. The private host/device descriptor is now 72 bytes with the
error-address field at byte 64. Old installed headers or raw-descriptor probes
must not be mixed with the new product; each probe must bind the new ABI.

This removes one source-level rejection in `stereoDisparity`, but does not yet
make that original sample executable: its pitch2D resource uses ordinary
`cudaMalloc`, which is not an adapter-owned pitched allocation. That ownership
extension, full target compilation and original CPU verification remain
separate gates. No sample source, algorithm, input or oracle is modified.

## Accepted surface

The frontend requires physical CUDA SDK declarations and direct literal references. The SDK `cudaArrayLayered` macro is materialized as its exact `0x01` literal, retaining its int prvalue type; unproved enclosing macro observers are rejected. Texture calls inside `checkCudaErrors` require the authenticated frozen helper and its successfully committed rewrite transaction. It maps single-component float32 or unsigned-int32 arrays, layered 2D arrays, owned pitched allocations, and corresponding `tex2D`/`tex2DLayered` calls. Descriptor fields retain their CUDA meanings; descriptor binary layout is private. User declarations, indirect function calls, macro-generated names, general layout/reflection, and dependent texture fetches are rejected. Literal zero initialization and a narrowly proved native `memset` of exactly the same whole descriptor are admitted.

The public APIs include array/pitch allocation and free, 2D row copies and memset, copies to an array, array/pointer `cudaMemcpy3D`, texture creation/destruction, and the associated descriptors. Width/height/layer extents are positive and at most 2^20. Pitches and pointer offsets are bytes; an array endpoint makes the copy extent width count four-byte texels. Bounds and multiplication overflow are checked. Arrays support float32, unsigned-int32 and the experimental RGBA8 format, with no flags except the layered flag for layered allocation. Resource views, mipmaps, normalized integer reads outside RGBA8, sRGB, mirror/border addressing, and other unsupported descriptor settings return an explicit error.

When the explicit frontend compatibility profile is enabled, the texture record audit recognizes its exact product-owned, forced host-math include. Only that inclusion's unchanged dependent template declarations receive the utility exception; actual macro expansions in it disable the exception. Every entered include instance is audited separately, so a caller that undefines the include guard and includes the same physical header again does not inherit the forced inclusion's exception. Caller templates, copied or shadow headers, and concrete texture-layout observations remain subject to the same audit.

The texture audit also proves a narrow class of scalar non-type-template calls, including recursive `template<int n>` functions with a fixed `float` return and fixed scalar arguments. A call must name exactly one defined free function template: all template parameters are explicit non-pack `int` values without defaults or constraints, and every actual argument has the same canonical builtin arithmetic type as its fixed formal parameter. Explicit template arguments are integer literals or that template's own integer parameters combined with unary signs and `+`/`-`. Such scalar arguments cannot add ADL candidates. Dependent `+`, `-`, and `*` expressions are admitted only when both recursively proved operand types match, with ordinary integer promotion. The unresolved name is admitted only as that call's direct callee. This is not general dependent-template support: function addresses, overloaded candidate sets, dependent returns, record arguments and other unproved expressions remain rejected. Function bodies, concrete instantiations, explicit types and reflection operands still undergo the full audit.

## Sampling and numerical limits

Finite scalar `int` template call chains can additionally qualify for [bounded attribute-only device template inlining](bounded-device-template-inlining.md). That optional rule preserves language linkage and the existing texture/reflection gates; it is not general template support or a runtime acceptance claim.

Point filtering uses floor coordinates. Normalized wrap/clamp and unnormalized clamp are supported; unnormalized wrap is converted to clamp as specified by CUDA. Linear filtering applies the half-texel offset. Default/precise conversion accepts only weights exactly on the 1/256 grid and reports other fractional coordinates. With both `--target-policy=dav-c310-vec` and the explicit `--simt-math=fast`, the converter selects separate `tex2DFast` / `tex2DLayeredFast` wrappers that use unquantized FP32 bilinear weights for general fractional coordinates. Point filtering and the descriptor/addressing/lifetime rules are identical in both policies. Nonfinite coordinates, out-of-range layers, and a nonfinite texel actually loaded by either sampler set a sticky GM error. A blocking public synchronization or copy reports that error. A failed copy leaves the host destination unchanged when failure is detected before copying.

Fast interpolation deliberately approximates CUDA hardware filtering: CUDA stores weights with eight fractional bits, whereas this mode uses FP32 weights without guessing the hardware tie rule. It is suitable only when the application accepts a numerical tolerance; original Sample tolerances are never changed. For the same four finite texels and real arithmetic, changing weights by `da` and `db` changes the interpolated value by at most `(abs(da) + abs(db)) * (max(texels) - min(texels))`. This local bound excludes coordinate arithmetic differences and FP32 rounding/overflow, and is not a universal CUDA error guarantee.

Interpolation uses float arithmetic over finite participating texels. The adapter does not promise bitwise equivalence to all hardware texture interpolation or an undocumented FMA/rounding rule. Engineering probes compare independently listed values; original Samples must separately pass their complete, unchanged CPU acceptance criteria. Hardware texture-cache performance and unrestricted asynchronous equivalence are not claimed.

## Ownership and lifetime

Allocations, device descriptors and sticky words use real ACL memory. The registry is process-wide across translation units and records device/context ownership. API operations propagate actual allocation, copy, synchronization and free errors. Copies preserve row padding and expose backing updates to later fetches. Non-host-to-host copies and row memset wait for the real default stream before issuing immediate ACL operations, even before a texture object is created; synchronization errors are returned before writing destination data.

Freeing an array or an owned pitched allocation synchronizes pending device work, invalidates every referring device descriptor, and frees the backing GM allocation. It does not silently destroy the separately owned texture objects or defer the backing free. A retired registry slot remains reserved while its texture objects exist. A later destroy frees that object's descriptor and recycles the retired slot after the final reference. Using a texture after its backing has been freed is outside the valid CUDA resource precondition; this adapter detects it before reading backing memory and reports a sticky error. No claim is made that CUDA itself defines such a fetch.

If any descriptor invalidation fails, the backing allocation is retained and the ACL error is returned. If backing free fails, the allocation record remains live. Some descriptors may already have been invalidated in either failure case; no rollback is promised. A successful device reset discards corresponding live and retired registry entries only after the real reset succeeds. Concurrent reset while other threads use the device is outside the CUDA reset precondition.

The registry currently bounds simultaneously retained allocations and texture objects to 128 each; capacity exhaustion is an explicit allocation error. Blocking APIs may synchronize the entire device while texture objects exist. Invalid arbitrary raw device tokens remain outside the valid handle precondition.

## Validation status

The records below describe predecessor engineering builds. They are not a
validation claim for the new descriptor ABI v2 or RGBA8 implementation. The new
gates include actual-helper CPU arithmetic/lifecycle tests, exact old/new ABI
consumer bodies, protected-page rejection checks, original-source syntax,
atomic publication, native CCEC layout/return compilation and later independent
installed-helper device readback. CPU stubs are not ACL/NPU evidence. Full
original Sample builds and numerical acceptance remain separate requirements.

The native v2 registry probe passed real two-TU compilation, library/ELF checks and NPU execution: 32 listed fetch values, 36 array readbacks, padding, nine initial host rejections, three released-storage rejections, three nonfinite-texel rejections, 130 retirement cycles, sticky propagation and two resets. Earlier v1 evidence remains preserved with its superseded lifetime restriction.

Native product v9 built, installed and passed CTest. Its actual frontend matrix passed all 57 cases: 17 admitted and 40 rejected with normal failure receipts. This includes the independent record/reflection gate for cuda_runtime.h, exact frozen helper dependencies, macro observers and type aliases. Earlier failing roles remain preserved. A separate native default-stream probe passed 73,728 CPU values across six copy/memset paths and three rounds without any texture objects; it also verified that an unrelated nonblocking stream remained pending until explicitly released. No original texture Sample has yet been accepted.

The first combined original-target build exposed an additional frontend-profile ownership rejection before backend compilation. Its correction adds a separate profile-enabled matrix using recursive local-header projection, including caller header and macro counterexamples. That correction's native build and execution results are pending.

Sources: [CUDA Runtime memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html), [CUDA texture objects](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TEXTURE__OBJECT.html), and [CUDA 13.0.1 Programming Guide](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-programming-guide/index.html). Actual CANN signatures and engineering evidence are archived by the sample-conversion task; they do not replace public CUDA numerical requirements.
