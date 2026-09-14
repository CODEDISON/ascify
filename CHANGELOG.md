# Changelog

## Unreleased

### Added

- Repository checks for product identity, generated artifacts, and local
  documentation links; native build/install CI and executable CLI contracts.
- Open-source landing pages in English and Chinese, an FP32 introductory
  example, and a dedicated conversion reference.

- Host-only GitHub CI, contribution templates, editor settings, documentation
  navigation, and commit-bound validation summaries for phase-one closeout.
  The release process separates host checks, real translation, target execution,
  and performance acceptance; this engineering change adds no new device result.

- Single-warp int/float cooperative reductions with typed tile metadata and
  identical result bits across lanes, plus legacy CANN ballot/down-shuffle
  adapters for masks that match the currently active lanes. Generic multi-warp
  synchronization and device FP64 remain explicit boundaries (ADR 0023).
- Context-preserving conversion of directly included CUDA leaf headers that
  inherit NVIDIA sample-helper macros, with expanded-token and macro-state
  checks before joint conversion and transactional publication (ADR 0022).
- Narrow CUDA Samples adapters for host float max, half2 operations, full-warp
  tile/vote operations, partial device properties, and registered symbol
  copies, with explicit source and target boundaries documented in ADR 0021.
- English and Chinese user guides with simple steps for Linux setup, clean
  GitHub clone, build, install, first CUDA source conversion, result checks,
  mode selection, and common problems.
- Opt-in `--migration-receipt=<path>` output with deterministic JSON,
  per-input source-conversion stages, explicit contract selections, and
  fail-closed atomic publication.
- Lazy CUDA-runtime-compatible ACL lifecycle, memory, device, stream, and
  event adapters, including ordered process-exit cleanup and parameter-root
  GM atomic compatibility.
- Opt-in `--frontend-compat=ascify-admitted-v1` handling for the narrow
  admitted CUDA frontend surface and cooperative thread-block synchronization;
  a closed SHA-256 manifest, compiled-in exact byte identities and
  resolved-include provenance reject damaged profiles, local shadows and
  unsupported cooperative-group subheaders.
- Transactional recursive translation of quoted local-header closure with
  weak-canonical artifact/input isolation and checked publish/rollback.
- Narrow target-ABI adapters for target-owned vector constructors, host-local
  default `dim3`, default-stream events, and fail-closed stream flags.
- A proof-gated, all-or-nothing NVIDIA CUDA Samples helper closure for
  `checkCudaErrors` and `getLastCudaError`, including active-macro,
  status-domain, raw-EOF, external-preprocessor, and reserved-macro gates.
- Proven direct calls to NVIDIA's official `findCudaDevice` definition, using
  an Ascify-owned, fail-closed single-visible-device/logical-zero contract
  instead of CUDA GFLOPS ranking; paired versioned source identities freeze
  every active helper dependency, a transitive direct-call closure rejects
  user redeclarations of libc/runtime dependencies, and untrusted preinclude
  macros or later `system_header` self-promotion fail closed, while qualified,
  macro, alias, indirect, external, and altered-definition uses remain
  unmodified.
- Exact b7c `helper_functions.h -> helper_image.h` closures may provide the
  observed `EXIT_WAIVED=2` and `MAX(a,b)=((a>b)?a:b)` policy dependencies used
  by frozen helper consumers. Admission is bound to frozen content, exact
  bodies, and first physical-file identity; command-line, project-header,
  same-name, changed-body, pragma-promotion, re-entry, and line-spoofed origins
  remain fail-closed.
- Opt-in `dav-c310-vec + fast` AST-gated row-wise recipes for FP16 Softmax,
  RMSNorm, and LayerNorm, including affine RMSNorm.
- Versioned dav-c310 target headers, thin CUDA-to-ACL compatibility helpers,
  fail-close mutation coverage, and 950PR correctness/performance gates.
- Explicit `--target-recipe=dav-3510-rowwise-simd-v1` AST-gated dispatch for
  proved Softmax, RMSNorm, and LayerNorm wrappers. The option is disabled by
  default and requires `--target-policy=dav-c310-vec --simt-math=fast`.
- Version 1 row-wise hybrid ABI and separately built dav-3510 target support for
  Softmax recompute, RMSNorm cached, RMSNorm plain row-batch, and LayerNorm
  cached routes. The canonical facade returns `HybridTryResult`; the four C
  entries and DSO linker names use explicit `_v1` suffixes, and the libraries use
  `VERSION 1.0.0` / `SOVERSION 1` SONAMEs.
- An operator-independent compile-time mixed-stage recipe registry and generic
  `RowwiseHybridFacadeV1::Try<Recipe>` entry. LayerNorm is the third semantic
  family to exercise the registry end to end, with name-independent AST proof,
  fail-closed mutations, a selector, runtime DSO, and device harness.
- Versioned OneFlow conversion fixtures with source hashes and third-party
  licensing metadata.
- One host-only release-check entry point and CTest integration.

### Changed

- Public help and diagnostics now describe CUDA-to-Ascend translation. Removed
  inactive legacy launch switches and unimplemented script/document generators;
  actual statistics CSV output remains available. `--versions` reports the
  linked LLVM build instead of an unverified target support range.
- Standalone CMake builds respect the selected host compiler and use
  target-scoped settings, explicit source lists, relocatable install paths,
  and Python 3.9+ for tests. Build/run scripts provide help before requiring
  environment configuration. Installed distributions include license notices.
- Clang resource discovery uses the explicit resource directory or the
  matching build/install layout. An explicitly invalid resource directory
  reports an error instead of silently searching other installations.

- Device/SIMT mappings now preserve unsupported semantics conservatively and
  use AST context for device scalar-double lowering.
- Explicit row-wise Hybrid calls retain the original whole-SIMT launch for
  selector misses. A selected launch owns the call and executes row-wise math
  through a registered SIMD-to-SIMT stage split inside the same kernel. The
  current SIMT regions cover contiguous output staging and, for LayerNorm,
  centered element-wise normalization; a selected call returns its status
  without a second launch, including on error.
- Explicit SIMD semantic proof no longer trusts `ascify.semantic.*`
  annotations for primitive semantics or wrapper status types; it checks the
  complete raw FP32 primitive-specialization body and requires every textual
  conditional branch in that body to contain only the same exact primitive
  return. Reserved semantic-callee and injected-name macros fail closed.
  Generation reports use the v3 structural dispatch/ownership/fallback
  validator, require every recipe call to have a later direct-scope fallback
  in a distinct wrapper, require input/output launch counts to match, and
  ignore marker text in comments, literals, preprocessor directives, and
  unproved conditional-preprocessor regions. The publisher rejects raw
  invalid UTF-8, NUL bytes, UTF-8 BOMs, and trigraph spellings, handles standard and Clang
  whitespace-extended line splices, and normalizes digraph tokens before
  structural scope analysis. Its C/C++ lexical pass also preserves numeric
  digit separators and treats only CR/LF as physical line endings.
  Reports expose canonical `try_*_hybrid_count` fields and retain the v3
  `try_*_simd_count` fields as compatibility aliases.
- Explicit row-wise conversion rejects non-system GNU/MS assembler, including
  skipped-source tokens, and protects the closed facade, launch ABI, injected
  header guards, and ABI version from source-level redefinition. Injected
  headers preserve conventional outer-guard state even for valued defines.
- Softmax SIMD launch geometry now queries the current device, vector-core
  count, and maximum threads per vector core on every call, propagating each
  runtime error without a fixed fallback or cross-device cache.
- Each `_launch_v1` DSO entry repeats its v1 selector-domain check and rejects
  an invalid direct ABI call before launching a kernel. The 950PR harness uses
  `ROWWISE_SIMD_RUNTIME_DIR` for both versioned link lookup and process-local
  runtime loading.
- All conversion, build, benchmark, and evidence outputs are confined to one
  ignored `.work/softmax_rmsnorm_950` tree.

### Fixed

- The include-rewrite buffer now remains alive until the replacement has
  consumed it; it previously exposed a dangling `StringRef` after leaving an
  inactive legacy branch.
- Removed the obsolete checked-in generated vector-add output. Examples now
  generate into an ignored work directory using the current translator.

- Mixed-kernel launches now pass the full dynamic UB capacity required by
  their `TPipe` allocations: 163,904 bytes for Softmax recompute, 65,600 bytes
  for the original RMSNorm cached domain (131,136 bytes for the 16K-column
  extension), 147,520 bytes for RMSNorm plain row-batch, and 32,832
  bytes for LayerNorm cached. A second launch argument of `0` or `nullptr` is
  invalid; both forms are covered by the host static gate and selected-route
  device correctness tests.
