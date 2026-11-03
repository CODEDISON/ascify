# Changelog

User-visible changes to Ascify are recorded here. Unreleased entries describe
development features; consult the [compatibility reference](docs/compatibility.md)
for supported domains and experimental paths.

## Unreleased

- Revalidate the complete original CUDA Samples v13.0 cooperative-groups and
  occupancy targets on one Ascify product. See the [correctness record](docs/validation/vector-cg-occupancy-20261103.md)
  for the actual target, numeric checks and remaining coverage boundaries.

- Preserve once-protected contextual providers across independent selected root
  siblings, with the same two-parse skipped-edge and strict joint proof.
- Bind temporary discovery-root and original proof-root include edges through
  the identical parsed source bytes, retaining all nested-edge checks.

### Added

- Per-inclusion provenance for the admitted product CG qualifier macro chain
  at the Runtime-header boundary.
- Authenticated forced host-math conditions and original, uninstantiated
  primary/partial utility patterns in the main product CG and host-math headers
  at the full Runtime boundary, including the CG pattern's authenticated
  `uint4` lane-type identity argument, including its one-level physical SDK
  typedef spelling. Concrete instances and caller observations
  retain their checks; this does not admit a complete thread-fence project.
- Remaining member-template patterns in implicitly instantiated CG classes
  require exact member/class origin proofs. Their concrete specializations
  and caller observations remain audited.
- Adjacent helper status checks can recognize a visible, single-return host
  wrapper around an already admitted Runtime status API. The complete helper
  transaction and local-header publication checks remain required.
- Experimental `atomicInc` address proof for a uniquely defined UInt32 device
  counter initialized to zero, with a committed same-module symbol resolver.
  Other object/address forms keep their existing behavior; target validation
  and cross-block publication/reset checks remain pending.
- Recursive inherited-helper discovery can collect original context before
  checking helper-wrapped texture calls. Discovery emits no artifact; the
  contextual joint conversion retains all macro, texture, layout and atomic
  publication checks.

- Experimental preservation of literal packed fixed-integer record regions,
  with configuration guards, target layout witnesses and final edit protection.
  Packed pointers, arrays, templates and arbitrary packing state remain outside
  this proof; target compilation and complete sample validation are required.
  Inactive push-prefix recognition preserves intervening blank lines. Plain-char
  signedness must match the source toolchain and is checked by target witnesses.
  Selected child headers are proved during discovery; their guards and witnesses
  are emitted by the translation that owns their source, avoiding cross-file
  replacement conflicts while retaining complete-closure publication checks.

- Contextual local headers can retain literal unroll pragmas and normalize a
  single pragma once after physical include-entry proof. Ordered payload and
  position replay, unchanged inputs and atomic helper checks remain required.


- Experimental lowering of one literal unsigned packed-byte SAD accumulate
  instruction, with private uint32 operands and modulo-32-bit arithmetic.
  Other inline assembly remains outside this rewrite; target validation is required.

- Proposed bounded Pitch2D views over recorded ordinary cudaMalloc allocations,
  with requested-size, ownership and lifetime checks. This path awaits target
  validation and adds no original sample acceptance by itself.

- Bounded synchronous `cudaMemcpy2DToArray` rectangles for the registered F32,
  U32 and RGBA8 array formats, with explicit H2D/D2D, whole-texel offsets and
  preserved row padding. New candidate and original Sample validation are
  separate from predecessor scalar-copy evidence.
- Experimental preservation of bounded, physically proved `MAX` fallbacks in
  the frozen Samples-helper transaction. Both include orders retain the exact
  original macro body and evaluations; other observations and macro epochs
  remain rejected. See [the helper contract](docs/sample-helpers.md).
- Experimental authenticated RGBA8 texture channels: normalized `float4`
  Point/Linear reads and element-type `uchar4` Point reads on bounded 2D and
  layered resources. Numerical, coordinate and validation limits are recorded
  in [the texture contract](docs/texture-compatibility.md); original Sample
  support is not implied by the source extension.
- Experimental authenticated CUDA float3/uint3/int3 lowering to explicit 12-byte
  scalar-local records. Arrays, GM, external ABI, templates and native vec3
  interoperability remain guarded. See [the bounded contract](docs/cuda-vec3-local-layout.md).
- Explicit host C++ input mode and compiler `-x c++` selection, retaining AST
  translation and local-header proofs without implicit CUDA headers or CUDA
  language predefines. The default without a language remains CUDA; conflicting
  device options fail. See [the input contract](docs/host-cxx-input.md).
- Texture projection can retain uninstantiated utilities from the exact selected
  product cooperative-groups profile after physical-input, CUDA-control and
  device-attribute macro proofs. Concrete instances, caller templates and SDK
  layout observations retain their existing checks.

- Contextual helper providers and repeated local includes proven physically
  skipped in both original parses, preserving token/macro and atomic helper
  proofs. Real nested entries, re-entry and opaque pragmas remain rejected.
- Explicit host C++ input mode and compiler `-x c++` selection, retaining AST
  translation and local-header proofs without implicit CUDA headers or CUDA
  language predefines. The default without a language remains CUDA; conflicting
  device options fail. See [the input contract](docs/host-cxx-input.md).
- Direct physical CUDA Runtime includes inside proved whole-file guards of
  selected local headers. Nested conditions, guard macro-stack operations and
  conditional root-source includes remain outside this admission rule.
- Experimental single-component unsigned-int32 texture point reads with exact
  element bits and sticky rejection errors. Signed, normalized-read and linear
  integer sampling remain unsupported; target validation is still required.
- Transactional configuration guards for proved inactive Samples-helper branches;
  enabling an unsupported retained branch in the target produces an explicit error.
- Generic 32-bit integer `atomicAdd` for proven device-helper parameters,
  preserving the target GM or UB address space.
- Real device-memory properties and proven per-kernel thread limits for
  nonescaping max-only attribute queries.
- Device-symbol address kernels and checked transfers for admitted definitions,
  including downloads and fresh resolution after device reset.
- Explicit CANN profiling sessions, ordered host functions, and real event-state
  queries on the reviewed runtime surface.
- Bounded host diagnostics that display CUDA SM versions as unavailable on
  Ascend without inventing numeric device capabilities.
- Exact expression preservation for admitted frozen Samples `MAX` invocations
  when converting the surrounding helper closure.
- Bounded conversion of the nonvolatile CUDA lane-ID instruction, with
  three-dimensional block indexing and conservative assembly boundaries.

- Optional JSON migration receipts with global and per-input conversion status.
- Recursive local-header conversion with checked dependency identity and
  transactional publication of generated source and headers.
- An opt-in, versioned frontend profile for admitted cooperative-group operations.
- Scoped CUDA Runtime, device, and NVIDIA Samples helper compatibility adapters.
- Explicit row-wise SIMD+SIMT Hybrid conversion for recognized Softmax, RMSNorm,
  and LayerNorm patterns, with versioned runtime libraries and SIMT fallback.
- Repository checks, host tests, and Linux CI for native LLVM builds, installation,
  and the public CLI.
- English and Chinese setup guides, API compatibility documentation, and an FP32
  vector-add example.

### Changed

- Private texture descriptor ABI v2 rejects incompatible channel/filter/version
  tags before texel reads. Rebuild all texture-using TUs, libraries and probes
  with one complete helper installation and recreate handles, including existing
  float32/unsigned-int32 consumers. Do not mix old and new descriptor helpers.
- CLI help, diagnostics, and dependencies consistently describe CUDA-to-Ascend
  conversion. Inactive legacy switches and unimplemented generators were removed.
- Standalone CMake builds respect the selected compiler and use target-scoped
  settings. Build and run wrappers support help without environment setup.
- Installation separates public compatibility headers from private Clang parsing
  resources and includes dependency license notices.
- Explicit invalid resource paths report an error; installed binaries resolve
  their resources from the installation layout.
- Compatibility rewrites check source semantics, types, and provenance before
  replacing supported patterns; unsupported domains retain explicit boundaries.

### Fixed

- Retain Samples helper transactions when an unproved enclosing macro can
  observe a materialized `MAX` spelling, including stringization whitespace.
- Preserve standard host `exp`, `log`, and `sqrt` overloads in proven direct
  calls so CPU reference calculations keep their original precision.
- Preserve admitted local-header dependencies on root macros and declarations
  through checked source-context expansion.
- Extend proven CUDA integer argument conversion preservation to `__clz(int)`.
- Preserve proven CUDA `__ffs(int)` integral argument conversions when the
  target provides additional integer-width overloads.
- Recognize proven adjacent local Runtime status assignments in NVIDIA
  Samples error checks, retaining unsupported control-flow and alias cases.
- Recognize proven NVIDIA Samples device selection in automatic local-variable
  initializers, preserving helper source, scope, and transaction checks.
- Support nonblocking stream creation on the reviewed legacy SIMT / ACL 1.17.0
  surface without forwarding CUDA flag bits to unrelated ACL stream options.
- Remove proven redundant, skipped CUDA 13.4.1 `cub/util_type.cuh` includes
  after an authenticated mapped CUB umbrella, preserving caller vec3 ABI
  rejection and unsupported standalone subheader boundaries. CUB/CCCL
  preprocessor observations remain outside the supported domain.
- Preserve include replacement text until Clang consumes the edit.
- Reject unknown command-line options beginning with the short help spelling.
- Order owned runtime cleanup after initialization and device binding.
- Propagate selected Hybrid launch failures without launching the SIMT fallback.
- Match Hybrid launch memory capacity to the runtime's shared allocations.

- Authenticate a direct literal physical CUDA Runtime umbrella include in host C++ before early texture declarations; retain all SDK dependency and texture use checks. Validate final MAX definitions separately for both CCEC preprocessor tables.
