# Source conversion reference

This reference describes the `ascify-clang` command-line interface and the
semantic boundaries of generated source. For setup and a first conversion, use
the [English tutorial](user-guide.en.md) or [中文教程](user-guide.zh-CN.md).
Supported APIs and limitations are described in the
[compatibility reference](compatibility.md).

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

Pass Ascify options and input paths before **`--`**, followed by extra Clang flags
(omit `--` if there are no extra Clang arguments):

```text
ascify-clang [ascify-options] <inputs> -- [clang-options]
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
| `-versions` | Print translator and LLVM build identity; target support depends on the compatibility surface and runtime |

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

Ascify recognizes a restricted `helper_cuda.h` dependency and can replace
`checkCudaErrors`, `getLastCudaError`, and direct calls to the admitted
`findCudaDevice` implementation. The device helper requires one visible logical
device and selects logical zero. Recognition is automatic and depends on source
identity, active macro definitions, and supported call types. It does not enable
general CUDA Samples or library compatibility.

See [CUDA Samples helper compatibility](sample-helpers.md) for supported include
forms, status domains, frozen dependencies, and rejection behavior.

### Target-specific SIMT rewrites

`portable` and `precise` are the defaults. Enable the target-specific fast policy
explicitly:

```bash
ascify-clang input.cuh \
  --target-policy=dav-c310-vec \
  --simt-math=fast \
  --cuda-path=/path/to/cuda \
  -o output.cuh
```

Under that policy, Ascify may replace a semantically proven full-mask,
32-lane add-reduction loop with the native warp reduction helper, and may tag
pure binary sum/max/min functors for `aclcub::BlockReduce`. Partial masks,
sub-warps, side effects, dependent calls, unsupported data types, and
unproven functors retain the compatibility fallback.

### Explicit row-wise SIMD+SIMT Hybrid dispatch

Enable the versioned recipe together with the target and math policy:

```bash
ascify-clang input.cuh \
  --target-policy=dav-c310-vec \
  --simt-math=fast \
  --target-recipe=dav-3510-rowwise-simd-v1 \
  --cuda-path=/path/to/cuda \
  -o output.cuh
```

Ascify recognizes supported FP16 Softmax, RMSNorm, and LayerNorm data flows,
including admitted load/store adapters. Recognition checks the implementation
and its source context; function names or annotations alone do not qualify.
The generated call uses the Hybrid runtime when its selector accepts the inputs,
and keeps the whole-SIMT launch for selector misses. A selected launch returns
its status, including errors, without launching a second fallback.

Hybrid output requires separately built runtime libraries. Use the
[row-wise conversion guide](rowwise-simd-conversion.md) for supported source
patterns, shapes, ABI, build/link commands, and testing. The
[architecture guide](architecture.md) explains how recognition and runtime
selection fit together.

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
