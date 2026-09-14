# Source conversion reference

This reference describes the `ascify-clang` command-line interface and the
semantic boundaries of generated source. For setup and a first conversion, use
the [English tutorial](user-guide.en.md) or [中文教程](user-guide.zh-CN.md).
Target compilation and device results are recorded separately in the
[validation matrix](validation-matrix.md).

## Build options

| CMake option | Default | Purpose |
|---|---|---|
| `ASCIFY_CLANG_TESTS` | `OFF` | Build Ascify and register release checks |
| `ASCIFY_CLANG_TESTS_ONLY` | `OFF` | Register host checks without LLVM/Clang |
| `ASCIFY_INSTALL_CLANG_HEADERS` | `ON` | Install matching Clang resource headers |
| `ASCIFY_CLANG_RESOURCE_DIRECTORY` | Detected | Override the matching Clang resource directory |

`./build.sh --help` describes build paths, compilers, linker, and job settings.
`./run.sh --help` describes the conversion wrapper. The wrapper requires
`CUDA_PATH`; `CLANG_RESOURCE_DIRECTORY` optionally overrides default resource
discovery. The release suite uses the separate
`ASCIFY_CUDA_PATH` and `ASCIFY_CLANG_RESOURCE_DIRECTORY` environment variables.

## Usage

Typical invocation passes ascify flags first, then **`--`**, then normal Clang flags (omit `--` if there are no extra Clang arguments):

```bash
ascify-clang [ascify-options] -- [clang-options] <inputs>
```

Minimal ingredients for CUDA sources:

- **`--cuda-path=<dir>`** — root of the CUDA installation (headers, `nvvm`, etc.).
- **`--clang-resource-directory=<dir>`** — parent of Clang’s parsing `include/` tree (contains `__clang_cuda_runtime_wrapper.h`). Ascify installs these private resources in `<install-prefix>/libexec/ascify/clang/<major>` and discovers that location automatically. An external LLVM build or installation can instead supply its own `lib/clang/<version>` resource directory. This setting is separate from the public compatibility include path used to compile generated source.

Example (adjust paths and Clang major version):

```bash
./ascify_install/bin/ascify-clang examples/vector_add.cu \
  --cuda-path=/path/to/user-owned/cuda \
  --clang-resource-directory="$PWD/ascify_install/libexec/ascify/clang/23" \
  -o /tmp/vector_add.cpp
```

The example assumes Clang 23 and the default Ascify install prefix. Substitute
your build's resource version, or omit the explicit override to exercise installed
resource discovery. Write to a file with **`-o`**, a directory tree with **`-o-dir`**,
or inspect only with **`-examine`** (combines `-no-output` and `-print-stats`).

### Useful ascify options

| Flag | Role |
|------|------|
| `-o`, `-o-dir` | Output file or directory |
| `-inplace` | Rewrite source in place (optional backup unless `-no-backup`) |
| `-cuda-gpu-arch=sm_XX` | GPU arch for CUDA compilation (repeatable) |
| `-print-stats` / `-print-stats-csv` | Translation statistics |
| `-local-headers` / `-local-headers-recursive` | Process quoted local includes |
| `--frontend-compat=none\|ascify-admitted-v1` | Keep strict CUDA frontend semantics or opt in to the narrow, versioned compatibility profile; default `none` |
| `--target-policy=portable\|dav-c310-vec` | Select conservative output or opt in to validated dav-c310 SIMT rewrites |
| `--simt-math=precise\|fast` | Keep precise semantics or enable guarded fast-SIMT transformations |
| `--target-recipe=none\|dav-3510-rowwise-simd-v1` | Explicitly enable the versioned, externally linked row-wise SIMD+SIMT hybrid dispatch; default `none` |
| `--migration-receipt=<path>` | Atomically write an opt-in, deterministic JSON result for this source-conversion invocation |
| `--no-lower-device-double-params` | Preserve by-value scalar `double` parameters on CUDA `__global__` functions |
| `-versions` | Print translator and LLVM build identity; target validation is recorded separately in the validation matrix |

Full list: run **`ascify-clang --help`**.

### Machine-readable migration receipt

Use `--migration-receipt=<path>` when an experiment runner needs a stable
source-conversion result instead of parsing diagnostics:

```bash
ascify-clang input.cu \
  --target-policy=dav-c310-vec \
  --target-recipe=none \
  --migration-receipt=/tmp/input.receipt.json \
  -o /tmp/input.cu.dpp
```

The target directory must already exist, and the receipt path must differ from
the input, translated output, and statistics output. Ascify first publishes an
`in_progress` receipt, then atomically replaces it with `succeeded` or
`failed`. The JSON records the selected frontend, local-header, target, and
output contracts plus a stable per-input stage result. It does not claim target
compilation, linking, device execution, numerical correctness, or performance.
Full compiler diagnostics remain on standard error.

### Frontend compatibility and local-header boundaries

`--frontend-compat=ascify-admitted-v1` is a closed parser profile, not a broad
CUDA-header shim. Ascify verifies its versioned manifest, required admission
and poison headers, compiled-in exact byte identities, published SHA-256 values,
and exact directory layout before parsing. During preprocessing,
`<cooperative_groups.h>` must resolve to
that verified profile file. Quoted local shadows and every unadmitted
`cooperative_groups/*` subheader fail with no translated output. The default
`--frontend-compat=none` leaves the CUDA include search unchanged.

`--local-headers` and `--local-headers-recursive` apply only to eligible quoted
user headers. They do not admit angle-bracket SDK helpers such as
`<helper_cuda.h>`. Converted headers and the root output are staged as one
managed two-artifact transaction. Existing-parent symlinks are resolved before
alias checks; unmanaged destinations, input aliases and restore failures fail
closed. The transaction protects normal publish failures but is not a durable
crash-recovery journal. See
[the local-header closure contract](local-header-closure.md).

### NVIDIA CUDA Samples helper closure

Ascify separately recognizes the proven NVIDIA CUDA Samples
`helper_cuda.h` included by a main translation unit. It can close the two
error-handling macros `checkCudaErrors(expr)` and
`getLastCudaError(message)` only when the resolved header, active macro
definitions, direct source tokens, and CUDA Runtime status domain all match
the admitted contract. It can also replace a direct host call to the official
`findCudaDevice(int, const char **)` definition after proving its canonical
signature, active device-selection AST, unqualified non-macro call token, and
the version-paired SHA-256 identities of the actual `helper_cuda.h` and
`helper_string.h` definition files. The hashes cover Clang's parsed buffers,
every macro that affects either frozen header must be defined by that profile,
a file that entered preprocessing as a system header, or the compiler itself,
and every resolved direct call transitively reached from the replaced helper
must stay inside the frozen profile or that same system-file set. Return-value
use is preserved because only the callee token changes. This proof gate is
automatic when such an include is encountered; it is not enabled by the
frontend-profile or local-header flags. The frozen SHA-256 proof requires LLVM
13 or newer; on an older LLVM build this one helper rewrite stays disabled and
fails closed while the rest of Ascify remains available.

This proof trusts the configured Ascify/Clang resources, CUDA SDK, sysroot, and
system-header graph; an untrusted `-isystem` directory enlarges that trust and
is outside the guarantee. Main source, `-D`, ordinary preincludes, VFS helper
remaps, forged `#line` filenames, and ordinary headers that later apply
`#pragma clang/GCC system_header` (including re-entry under a new `FileID`)
remain fail-closed inputs.
The macro replacements evaluate the status expression once, preserve
expression/file/line diagnostics, and implement `getLastCudaError` with
consume-and-reset semantics.

This is one all-or-nothing source transaction. A duplicate or indirect
include, residual preprocessor use in any local header, altered active macro,
reserved output-macro collision, unsupported status domain, external helper
declaration use, or incomplete raw-file audit keeps the original include and
helper calls. The owned device helper admits only one visible logical device:
no selector or one exact `--device=0` binds logical zero; any other selector,
device count, query error, or bind error fails explicitly. It does not emulate
CUDA's GFLOPS ranking. Occupancy, Driver/VMM, compression, and
compressible-allocation helpers remain unsupported. See
[ADR-0015](decisions/0015-admit-only-proven-nvidia-sample-helper-closure.md)
and
[ADR-0016](decisions/0016-proof-gate-single-device-sample-selection.md).

`portable` and `precise` are the defaults. The 950PR SIMT policy used in the
[recorded validation](validation-matrix.md) is explicitly opt-in:

```bash
ascify-clang input.cuh \
  --target-policy=dav-c310-vec \
  --simt-math=fast \
  --cuda-path=/path/to/cuda \
  --clang-resource-directory=/path/to/llvm/lib/clang/23 \
  -o output.cuh
```

Under that policy, Ascify may replace a semantically proven full-mask,
32-lane add-reduction loop with the native warp reduction helper, and may tag
pure binary sum/max/min functors for `aclcub::BlockReduce`. Partial masks,
sub-warps, side effects, dependent calls, unsupported data types, and
unproven functors retain the compatibility fallback.

### Explicit row-wise SIMD+SIMT hybrid dispatch

The dav-3510 row-wise hybrid path has an additional explicit opt-in:

```bash
ascify-clang input.cuh \
  --target-policy=dav-c310-vec \
  --simt-math=fast \
  --target-recipe=dav-3510-rowwise-simd-v1 \
  --cuda-path=/path/to/cuda \
  --clang-resource-directory=/path/to/llvm/lib/clang/23 \
  -o output.cuh
```

`--target-recipe=none` is the default and does not add the externally linked
row-wise target ABI to generated code. The explicit recipe is rejected unless
both `dav-c310-vec` and `fast` are selected.

In explicit mode, Ascify inserts the closed `RowwiseHybridFacadeV1` entry only
when it proves all of the following in the main-file AST:

- an exact packed row-major FP16-to-FP32 load and FP32-to-FP16 direct or
  per-column affine store;
- exact-owner adapter metadata, so derived classes cannot inherit the proof;
- the complete Softmax, RMSNorm, or LayerNorm primitive/data-flow graph, including
  resolved `exp`/divide/`rsqrt` semantics;
- a supported shape/type contract at runtime.

Explicit hybrid mode does not accept `ascify.semantic.*` annotations as the
semantic or status-type proof. It inspects the FP32 helper specialization,
reads the complete source body rather than only the preprocessor-selected AST
branch, and requires every textual conditional branch in that body to contain
only the corresponding `exp`, divide, or `rsqrt` return. It also verifies the
wrapper return type and rejects identity or side-effecting helpers. Reserved
macros, pre-instrumented adapter markers, input-defined dispatch declarations,
and input-defined launch-ABI symbols fail closed. Feature macros that affect a
proved source branch must match between conversion and CCE build. The recorded
release evidence aligns both `OF_*_USE_FAST_MATH` values; unrelated toolkit
version macros may differ only when they select the same validated branch. The
v3 report records the converter arguments and frontend/generator digests.
Non-system GNU/MS assembler is rejected, including assembler tokens in skipped
dependencies, so source cannot alias a protected `_launch_v1` symbol. Injected
headers are macro-shielded; a conventional main-file include guard is
temporarily undefined only around the injected include and restored before the
translated body.

The generated main CCE calls
`RowwiseHybridFacadeV1::TrySoftmaxHybrid` or
`RowwiseHybridFacadeV1::TryRmsNormHybrid` or
`RowwiseHybridFacadeV1::TryLayerNormHybrid` and retains its original
translated SIMT launch. The outer `handled` branch is a pre-launch support gate: a
selector miss continues into that whole-SIMT launch, while a selected call is
owned by one target kernel that uses both execution styles. It is not an
all-SIMD versus all-SIMT operator choice. A selected error is returned and
never launches the fallback a second time.

Inside a selected Softmax kernel, SIMD performs row maximum,
exponentiation/sum reduction, and normalization division. Inside selected
RMSNorm kernels, SIMD performs square/sum reduction, square root, division,
normalization, and the optional affine multiply. For LayerNorm, SIMD performs
mean, centered variance, and reciprocal square root; SIMT applies centered
normalization and output-local indexing. In Softmax and RMSNorm, SIMT performs
contiguous normalized-value staging from UB to the output-local tile. None of
the current recipes provides mask, gather, scatter, stride, or other
non-contiguous adapter lowering.

Mixed kernels require an explicit dynamic-UB launch capacity in the second
launch argument, large enough for the complete `TPipe` allocation. The
current budgets are 163,904 bytes for Softmax recompute, 65,600 bytes for
RMSNorm cached, 147,520 bytes for RMSNorm plain row-batch, and 32,832 bytes for
LayerNorm cached. `0` and `nullptr` are invalid; the host static gate rejects
both forms. Recorded selected-route device checks cover these budgets; their
tested candidate and evidence scope are recorded in the
[validation matrix](validation-matrix.md).

The Softmax runtime queries the current device, vector-core count, and maximum
threads per vector core on every selected call. Query failures are propagated
unchanged; the runtime does not substitute a fixed core count or reuse a
process-global result from another device.

The facade reports that decision as `HybridTryResult`, an alias of the legacy
`SimdTryResult { bool handled; aclError status; }` type. ABI v1 exports four C
symbols: `ascify950_softmax_reg_recompute_launch_v1`,
`ascify950_rmsnorm_reg_cached_launch_v1`,
`ascify950_rmsnorm_reg_plain_rowbatch_launch_v1`, and
`ascify950_layernorm_reg_cached_launch_v1`. Their respective DSO linker names
are `libascify950_softmax_reg_recompute_v1.so`,
`libascify950_rmsnorm_reg_cached_v1.so`,
`libascify950_rmsnorm_reg_plain_rowbatch_v1.so`, and
`libascify950_layernorm_reg_cached_v1.so`. Their SONAMEs append `.1` to those
linker names (`VERSION 1.0.0`, `SOVERSION 1`). Each public DSO entry
repeats the corresponding v1 selector-domain check before launching a kernel,
so a direct ABI call cannot bypass the shape, alignment, extent, or aliasing
guard.

The mixed target-support device implementations are built separately from
`runtime/dav_3510/rowwise/` and linked or deployed with the generated main CCE.
Source conversion alone therefore does not establish CANN build, device
correctness, hybrid routing, or performance. RMSNorm affine also requires
converting the caller file that defines its store adapter together with
`layer_norm.cuh` and `rms_norm.cuh`; an unknown external store is never
inferred from field names or layout.

The intra-kernel stage change invalidates predecessor build, correctness, and
performance attribution. Current claims require regenerated Hybrid-facade
output and freshly built, checked, and measured target-support binaries.

For the existing 950PR harness, set `ROWWISE_SIMD_RUNTIME_DIR` to the
unchanged CMake build/install `lib` directory containing all four versioned
`.so` linker-name files, all four `.so.1` SONAME files, and their
implementations. The harness verifies each ELF SONAME and that its linker and
SONAME paths resolve to the same file. The build script links the Softmax check
ELF only to the Softmax DSO and the RMSNorm check ELF only to the two RMSNorm
DSOs; the dedicated LayerNorm check exposes only its LayerNorm launch symbol.
The script rejects any extra or missing row-wise `_launch_v1` dynamic symbol.
The smoke runner prepends the same directory to its process-local
`LD_LIBRARY_PATH`.

See [the explicit row-wise SIMD+SIMT conversion guide](rowwise-simd-conversion.md)
for the ABI, build/link steps, selector domains, verification gates, and
conversion/performance reporting definitions. The explicit activation and AST
proof are recorded in
[ADR-0007](decisions/0007-explicit-ast-gated-rowwise-simd-dispatch.md),
and the same-kernel stage composition is recorded in
[ADR-0008](decisions/0008-compose-rowwise-simd-and-simt-stages-in-one-kernel.md).
The common recipe registry and third-family LayerNorm extension are recorded
in
[ADR-0009](decisions/0009-register-rowwise-hybrid-recipes-and-add-layernorm.md).

## Tests

Run the host gate with the dependencies listed in the contribution guide:

```bash
ASCIFY_BINARY= sh tests/run_release_checks.sh
```

It checks rewrite contracts and runs the host Python suites. To re-translate and
compare all golden fixtures, also set `ASCIFY_BINARY`, `ASCIFY_CUDA_PATH`, and
`ASCIFY_CLANG_RESOURCE_DIRECTORY`; see [CONTRIBUTING.md](../CONTRIBUTING.md).
The binary-enabled gate also converts the complete OneFlow fixtures through
the generation wrapper, validates the v3 dispatch/ownership/fallback report,
rejects pre-instrumented input, and rejects annotated identity-helper and
non-status-wrapper mutations. It also corrupts each side of a conditional
primitive in turn and verifies that an invalid inactive branch cannot authorize
hybrid conversion. Report acceptance structurally requires every recipe call to
form the generated ownership contract in a distinct wrapper with a later
direct-scope SIMT launch; the generated and input launch counts must match.

The shared 950PR harness and its legacy in-header row-wise recipe replay are
documented in
[tests/softmax_rmsnorm_950/README.md](../tests/softmax_rmsnorm_950/README.md).
That older replay is useful as a SIMT baseline but does not enable the new
`--target-recipe` or build the four hybrid target-support DSOs. Use the
explicit conversion guide above for the v1 SIMD+SIMT workflow.

