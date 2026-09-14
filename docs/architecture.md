# Architecture

Ascify is a Clang-based CUDA-to-Ascend source translator. It analyzes source,
rewrites the constructs it can support, and publishes generated source for a
separate target build. The converter, generated-code compatibility headers,
and device runtime have different responsibilities.

## Components and consumers

| Component | Responsibility |
|---|---|
| `src/main.cpp`, `src/ArgParse.cpp` | Command-line handling, Clang resource discovery, and CUDA parsing setup |
| `src/AscifyAction.cpp`, `src/CUDA2DPP*.cpp` | Preprocessor/AST analysis, API maps, and coordinated source edits |
| `src/LocalHeader.cpp` | Local dependency discovery and transactional root/header publication |
| `src/DeviceMemoryLowering.cpp`, `src/UniformBlockReduction.cpp` | Separate proofs for private object copies and uniform block reductions |
| `src/DavC310TargetRecipe.cpp`, `src/StaticKernelProof.h` | Row-wise semantic recognition and verified launch-wrapper placement |
| `include/ascify/`, `acl_cub/` | Compatibility interfaces consumed by generated Ascend code |
| `frontend_compat/` | Versioned, optional parsing profiles consumed by the converter |
| `runtime/dav_3510/rowwise/` | Separately compiled implementations used by selected Hybrid calls |

For example, a CUDA `cudaMalloc` call becomes `ascify::cudaMalloc`, provided by
`<ascify/ascify_cuda_compat.hpp>`. Compile generated code with
`-I<install-prefix>/include`. These are public target headers, not the
translator implementation. Clang's private parsing headers live under
`libexec/ascify/clang/<resource-version>/include`; installed executables discover
them relative to their configured installation layout.

## How conversion decisions are made

Simple mappings handle compatible names. Differences in signatures or runtime
behavior use typed compatibility functions. Transformations that depend on a
call's types, address space, control flow, or provenance require an AST proof.
This preserves language identity and avoids global replacements such as
changing every `double` to `float`.

Clang's parsed declarations and the observed include/preprocessor state are
part of that proof. A matching filename or spelling alone is insufficient for
SDK/helper admission. The optional frontend profile is checked against its
manifest and compiled-in identities before parsing; it is not a general CUDA
header shim. See [compatibility boundaries](compatibility.md) and
[CUDA Samples helpers](sample-helpers.md).

Proof-dependent edits are staged together. A failed proof must not publish a
partial helper replacement or half of a managed header bundle. Conflicting
edits and failed filesystem restoration are reported. Depending on the feature,
an unsupported construct is retained, rejected during conversion, or rejected
by its target compatibility interface. These outcomes do not establish that
the resulting application can execute. The
[local-header contract](local-header-closure.md) describes publication limits.

## Target policy and Hybrid recipes

The defaults are `--target-policy=portable`, `--simt-math=precise`, and
`--target-recipe=none`. Target-specific fast rewrites require the explicit
`dav-c310-vec` policy and `fast` math mode. Canonical warp-reduction and reducer
rewrites additionally prove their operation, types, ownership, and effects;
unproved functors keep the callable fallback.

The separate `dav-3510-rowwise-simd-v1` recipe adds an external runtime dependency
only when requested. It recognizes supported Softmax, RMSNorm, and LayerNorm
data flows and concrete adapters. `StaticKernelProof` binds each result to its
canonical kernel, analyzed definition, source provenance, and semantic family.
Launch-wrapper analysis must consume one matching family and preserve required
geometry and error handling. This internal proof representation is not a public
lowering IR or an arbitrary CUDA-to-SIMD partitioner.

A recognized call reaches a runtime selector before launch. A selector miss
executes the retained whole-SIMT launch. A selected call owns execution and
returns its status, including errors; it must never launch the fallback after
a selected error. One selected kernel contains both SIMD and SIMT stages.
The stage registry checks execution composition; the frontend proof and runtime
selector separately establish operator semantics and the input domain.

The runtime uses versioned C entry points and four separate shared libraries.
ABI, selector, dynamic UB capacity, and generated headers must agree. The
[row-wise guide](rowwise-simd-conversion.md) defines activation, shapes,
rounding, launch ownership, and build/link requirements.

## Extending the translator

Start with the smallest supported input domain and its explicit rejection
boundary. Add the API adapter or semantic analyzer, then cover valid inputs and
nearby invalid cases. A new Hybrid family also needs a stage descriptor,
selector, versioned runtime entry, and target implementation. Names, input
hashes, and deposited output files cannot substitute for semantic recognition.

Useful checks include [SIMT contracts](../tests/rewrite/check_simt_compat.sh),
[canonical reductions](../tests/rewrite/check_canonical_reducer_rewrite.sh),
[kernel proof invariants](../tests/rewrite/check_static_kernel_proof_m0.sh), and
[Hybrid dispatch contracts](../tests/rewrite/check_rowwise_simd_recipes.sh).
The [contribution guide](../CONTRIBUTING.md) gives the common test entry points.
Host models, actual conversion, target compilation, device correctness, and
performance are separate validation levels; see the
[release process](release-process.md).
