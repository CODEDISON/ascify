# Explicit row-wise SIMD+SIMT conversion

Ascify can add a versioned dav-3510 same-kernel SIMD+SIMT dispatch to
semantically proved Softmax, RMSNorm, and LayerNorm wrappers while retaining the
translated whole-SIMT launch as the selector-miss fallback. This path is
disabled by default and has an explicit source, ABI, build, and measurement
boundary.

## Convert a source file

Select all three required options:

```bash
./build/ascify-clang input.cuh \
  --target-policy=dav-c310-vec \
  --simt-math=fast \
  --target-recipe=dav-3510-rowwise-simd-v1 \
  --cuda-path=/path/to/user-owned/cuda \
  --clang-resource-directory=/path/to/llvm/lib/clang/23 \
  -o output.cuh \
  -- -I/path/to/source/includes -std=c++17
```

`--target-recipe=none` is the default. It leaves the new externally linked
hybrid path disabled and preserves ordinary conversion behavior. The explicit
recipe is rejected unless `--target-policy=dav-c310-vec` and
`--simt-math=fast` are both selected.

The converter policy and recipe names describe different boundaries:

- `dav-c310-vec` selects the existing target-aware SIMT rewrite policy;
- `dav-3510-rowwise-simd-v1` selects the 950PR/dav-3510 row-wise hybrid support
  ABI and selector. The recipe spelling is retained for compatibility.

Source conversion needs Ascify, CUDA parsing headers, and Clang resource
headers. It does not need a CANN installation or a device. Building and
running the converted result does.

## Generated main-CCE contract

When the AST proof succeeds, Ascify inserts:

```cpp
#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>
```

and a call through the closed `RowwiseHybridFacadeV1` at each proved direct
launch wrapper. The target call returns `rowwise_simd_v1::HybridTryResult`,
which is also re-exported as
`ascify::target::dav_c310::HybridTryResult`:

```cpp
struct SimdTryResult {
  bool handled;
  aclError status;
};
using HybridTryResult = SimdTryResult;
```

`HybridTryResult` is an alias of the legacy `SimdTryResult` type.
The old `RowwiseSimdFacadeV1::Try*Simd` entry remains only as a compatibility
alias for already generated review artifacts; new conversion output uses the
Hybrid facade.

The generated ownership pattern is:

```cpp
const auto result = ::ascify::target::dav_c310::rowwise_simd_v1::RowwiseHybridFacadeV1::TrySoftmaxHybrid(...);
if (result.handled) {
  return result.status;
}

// Original translated SIMT launch remains here.
```

The same rule applies to `TryRmsNormHybrid` and `TryLayerNormHybrid`. A selector miss returns
`handled == false` and continues into the original whole-SIMT launch.
`handled == true` means that one mixed target kernel owns the call, even when
`status` is an error. The outer branch is therefore pre-launch support gating,
not an all-SIMD versus all-SIMT operator choice. The caller returns a selected
error and must not launch the fallback afterward.

Softmax insertion is at the proved direct-wrapper entry and is guarded to the
Softmax algorithm choice. RMSNorm and LayerNorm first execute their proved
geometry and error-return block, then try the hybrid target immediately before
the unique original launch. Dispatchers, LogSoftmax paths, and unproved
wrappers receive no target call.

The common facade also exposes `Try<Recipe>(...)`. A recipe registers a typed
launch and a compile-time list of MTE, SIMD, and SIMT stages. The registry
requires an MTE load, one or more SIMD stages, one SIMD-to-SIMT handoff whose
SIMT region ends in output adaptation, and an MTE store. It rejects an interior
MTE stage, SIMT before SIMD, SIMD after the handoff, or work after output
adaptation. Operator semantics and runtime domains are still
proved separately; registration is not an assertion that arbitrary CUDA is
safe to partition.

## Build and link the target support

The generated main CCE contains dispatch code, not the mixed target-support
device implementations. Build the versioned support separately with the
dav-3510 CANN toolchain:

```bash
export ASCEND_HOME_PATH=/path/to/cann

cmake -S runtime/dav_3510/rowwise \
  -B build/rowwise-simd-v1 \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/rowwise-simd-v1
cmake --install build/rowwise-simd-v1 \
  --prefix /path/to/ascify-rowwise-simd-v1
```

Configure the CANN/ASC package lookup as required by the local installation.
The runtime build emits four separate shared libraries so each device launch
keeps its own registration domain and all four can be linked without merging
their device registrations:

| Route | ABI-v1 C symbol | Linker-name file | SONAME |
|---|---|---|---|
| Softmax recompute | `ascify950_softmax_reg_recompute_launch_v1` | `libascify950_softmax_reg_recompute_v1.so` | `libascify950_softmax_reg_recompute_v1.so.1` |
| RMSNorm cached | `ascify950_rmsnorm_reg_cached_launch_v1` | `libascify950_rmsnorm_reg_cached_v1.so` | `libascify950_rmsnorm_reg_cached_v1.so.1` |
| RMSNorm plain row-batch | `ascify950_rmsnorm_reg_plain_rowbatch_launch_v1` | `libascify950_rmsnorm_reg_plain_rowbatch_v1.so` | `libascify950_rmsnorm_reg_plain_rowbatch_v1.so.1` |
| LayerNorm cached | `ascify950_layernorm_reg_cached_launch_v1` | `libascify950_layernorm_reg_cached_v1.so` | `libascify950_layernorm_reg_cached_v1.so.1` |

CMake sets `VERSION 1.0.0` and `SOVERSION 1` for every DSO. A build directory
therefore contains the `.so` linker name, the `.so.1` SONAME link, and the
`.so.1.0.0` implementation file.

Compile the generated main CCE against the installed public headers and link
or deploy the runtime libraries required by its possible routes. Main CCE and
runtime must use `ASCIFY_DAV_C310_ROWWISE_SIMD_ABI_VERSION == 1` and a
compatible CANN SDK. A successful source rewrite alone does not prove this
build/link contract.

The repository's 950PR harness accepts the runtime library directory
explicitly. Point `ROWWISE_SIMD_RUNTIME_DIR` at the unchanged CMake
build/install `lib` directory. It must contain all four `.so` linker-name
files and their four `.so.1` SONAME files listed above; copying only the
linker-name files is not a deployable runtime:

```bash
ROWWISE_SIMD_RUNTIME_DIR=build/rowwise-simd-v1/lib \
  tests/softmax_rmsnorm_950/scripts/build.sh check production-fast

ROWWISE_SIMD_RUNTIME_DIR=build/rowwise-simd-v1/lib \
  CHECK_ONLY=1 tests/softmax_rmsnorm_950/scripts/run_smoke.sh
```

The build script adds `-L${ROWWISE_SIMD_RUNTIME_DIR}` and uses an
operator-specific symbol contract. The Softmax and RMSNorm checks use their
corresponding routes, while `layernorm_hybrid_check` links only the LayerNorm
entry at the dynamic-symbol boundary. The script rejects an extra or missing
row-wise `_launch_v1` symbol. The run script prepends the same directory to its
process-local `LD_LIBRARY_PATH`. Both scripts inspect each ELF SONAME and
require the `.so` and `.so.1` paths to resolve to the same implementation;
neither script modifies the system CANN installation or the caller's shell
environment.

## Supported source boundary

The AST matcher proves structure and semantics; it does not trigger from a
filename, namespace, operator name, or source hash.

The explicit hybrid proof does not trust `ascify.semantic.*` annotations as a
substitute for inspectable math semantics or a compatible wrapper status
type. FP32 template specializations and wrapper return types are checked
directly. Primitive checking reads the complete raw function body, including
branches excluded from the current Clang AST, and requires every textual
conditional branch in that body to contain only the exact corresponding
`exp`, divide, or `rsqrt` return. A mismatched inactive branch, identity
helper, side-effecting helper, or annotated non-status wrapper leaves the
source on the SIMT path. Reserved macros, pre-instrumented adapter markers,
input-defined dispatch declarations, and input-defined launch-ABI symbols
reject explicit conversion. Feature macros affecting a proved branch must
agree between conversion and CCE build. The release evidence aligns the
`OF_*_USE_FAST_MATH` macros; toolkit-version macros may differ only when they
select the same validated branch. Converter arguments are bound into the v3
report, while the package manifest records the build-side configuration.
Non-system GNU/MS assembler is rejected as well, including raw assembler
tokens in skipped dependencies, because it can otherwise alias a protected
launch symbol outside the ordinary declaration checks.

The v3 publication scanner also models the relevant source translation
phases before it accepts structural evidence. It requires valid UTF-8 and
rejects NUL bytes, a leading UTF-8 BOM, and every raw C/C++ trigraph spelling,
removes both standard and Clang whitespace-extended
backslash-newline splices before comment handling, blanks comments and
literals, preserves C++ digit-separated preprocessing numbers, splits physical
lines only at CR/LF, normalizes digraph tokens, and excludes unproved
preprocessing branches. This prevents inactive, macro-body, or comment-spliced
text from being counted as a live dispatch/fallback contract even when extra
compiler flags are supplied.

The generated target include is inserted at the start of the main file under
macro shielding. A conventional outer include guard is recognized from raw
tokens, temporarily undefined around the injected include, and restored before
the original `#ifndef` / `#define` body. This keeps the body active even when a
target dependency happens to use the same guard spelling; valued guard defines
are preserved unchanged.

| Operator | Supported source form | Excluded examples |
|---|---|---|
| Softmax | Forward FP16 storage, FP32 compute, contiguous row-major direct load/store, complete max-exp-sum-divide flow | LogSoftmax, backward, affine store, mask, bias, clamp, shifted/non-contiguous adapters |
| RMSNorm plain | Forward FP16 storage, FP32 compute, contiguous direct load/store, complete square-sum-mean-epsilon-rsqrt flow, float inverse-RMS output | Backward, unsupported types, non-contiguous adapters, unproved inverse output |
| RMSNorm affine | Plain contract plus a proved same-column FP16 weight multiply in the concrete caller store | Unknown external store, shifted weight, extra transformation or side effect |
| LayerNorm | Forward FP16 storage, FP32 compute, contiguous direct load/store, proved Welford mean/M2/count, centered variance-epsilon-rsqrt normalization, and distinct float mean/inverse-variance outputs | Backward, affine output, wrong variance source, uncentered normalization, mutable row coordinates, missing or non-leader auxiliary output |

Wrapper insertion counts are topology checks, not independently converted
operator counts. Affine RMSNorm requires converting the caller that owns the
store adapter in addition to `layer_norm.cuh` and `rms_norm.cuh`.

A source form outside the proof remains on the ordinary translated path. It is
not assumed equivalent because its name resembles Softmax or RMSNorm.

For a selected Softmax route, the runtime queries the current device, vector
core count, and maximum threads per vector core in the same failure order as
the translated launch geometry path. Runtime errors are returned unchanged;
there is no fixed core-count fallback and no cross-device cache.

## Selected-kernel stage ownership

After the selector accepts a call, one `__aicore__` operator kernel invokes
both SIMD and SIMT vector functions. The implemented division of work is:

| Operator | SIMD-owned stages | SIMT-owned stages |
|---|---|---|
| Softmax | row maximum, exponentiation, sum reduction, normalization division | contiguous normalized-value staging from UB to the output-local tile, including element indexing and tail copy |
| RMSNorm plain | square accumulation, sum reduction, square root, division, normalization | contiguous normalized-value staging from UB to the output-local tile, including element indexing and tail copy |
| RMSNorm affine | square accumulation, sum reduction, square root, division, normalization, per-column affine multiply | contiguous normalized-value staging from UB to the output-local tile, including element indexing and tail copy |
| LayerNorm | mean, centered-square accumulation, variance reduction, reciprocal square root | centered normalization and output-local indexing, including tail-safe staging |

The shell still owns the ordinary GM-to-UB and UB-to-GM pipeline. Current SIMT
regions operate only on contiguous UB elements: Softmax/RMSNorm stage an
already normalized tile, while LayerNorm combines element-wise normalization
with output indexing. This is not a general adapter-lowering facility. Mask,
gather, scatter, stride, shifted access, and other non-contiguous forms are
neither implemented nor claimed by version 1.

The mixed-kernel launch's second argument is the dynamic UB capacity and must
cover the complete `TPipe` allocation shared by those stages. Version 1 uses
the following fixed budgets:

| Route | Dynamic UB launch capacity |
|---|---:|
| Softmax recompute | 163,904 bytes |
| RMSNorm cached | 65,600 bytes |
| RMSNorm plain row-batch | 147,520 bytes |
| LayerNorm cached | 32,832 bytes |

A `0` or `nullptr` second argument is invalid. Any change to a kernel's
`TPipe` buffers must update and revalidate the corresponding launch
capacity. The host static gate rejects both forms, and the selected-route
device correctness suite exercises the current launch budgets.

This stage composition is specified by
[ADR-0008](decisions/0008-compose-rowwise-simd-and-simt-stages-in-one-kernel.md);
the operator-independent registry and LayerNorm extension are specified by
[ADR-0009](decisions/0009-register-rowwise-hybrid-recipes-and-add-layernorm.md).

## Version 1 runtime selector boundary

Source recognition is broader than runtime hybrid selection. Version 1
applies these additional checks:

| Route | Required runtime domain |
|---|---|
| Softmax recompute | Non-null stream/input/output; positive non-overflowing extent; `4096 <= columns <= 256000`; input/output byte spans are disjoint or exactly in-place |
| RMSNorm plain row-batch | No affine weight; positive extent; finite positive epsilon; `columns <= 3072` and divisible by 16; required pointer alignment and auxiliary non-aliasing; input/output disjoint or exactly in-place |
| RMSNorm cached | Plain or affine; positive extent; finite positive epsilon; `columns <= 16384` and divisible by 8; required pointer alignment and auxiliary non-aliasing; input/output strictly non-overlapping |
| LayerNorm cached | Positive extent; finite positive epsilon; `columns <= 8192` and divisible by 8; 16-byte input/output alignment; aligned, distinct mean/inverse-variance outputs; input/output disjoint or exactly in-place |

The RMSNorm selector checks the plain row-batch route before the cached route.
Any call outside these domains returns `handled == false` and uses the retained
whole-SIMT launch. Selector thresholds belong to recipe/ABI version 1 and must
not be silently changed without new correctness and performance evidence.

The cached RMSNorm extension to 16384 columns retains pipeline depth 1.
Calls through 8192 columns keep their original 16384-byte row buffers and
65600-byte launch capacity; wider calls use 32768-byte row buffers and a
131136-byte launch capacity. The shared capacity assertions cover both the
queue allocation and launch budget. Row-batch selection, FP32 reduction,
plain FP16 rounding, two-step affine FP16 rounding, and the fixed grid policy
are unchanged. This extension requires fresh target compilation, full
correctness and performance evidence before promotion; host selector and
capacity checks alone do not establish device correctness or speed.

The public `_launch_v1` entry in each DSO repeats the same route predicate
before it launches a kernel. This second check protects the ABI boundary even
when a caller invokes a launch symbol directly. A rejected direct ABI call
returns `ACL_ERROR_RT_PARAM_INVALID` without a kernel launch; it is not a
facade selector miss. If the facade already returned `handled == true`, that
status is final and the retained SIMT path is not launched.

## Verification gates

Treat conversion, build, correctness, routing, and performance as separate
gates:

1. Run the host rewrite contracts and negative option tests. The row-wise hybrid
   host gate compiles direct-ABI negative cases and checks that each runtime
   entry reuses its corresponding v1 selector predicate. It also rejects
   semantic annotations attached to an identity helper or a non-status
   wrapper, and corrupts both sides of a conditional primitive in turn to
   verify that an invalid inactive branch is rejected.
2. Convert the full Softmax/RMSNorm/LayerNorm fixtures and verify dispatch placement,
   adapter markers, `handled/status` ownership, and retained fallback. The
   wrapper report uses `ascify.rowwise-cce-report.v3` as a structural
   integrity check, not as a second AST/CFG proof. Comments (including
   continued `//` lines), strings, preprocessor directives, unproved
   conditional regions, and incomplete calls do not count as a dispatch.
   Every remaining recipe call must have the exact generated guard and
   ownership contract in a distinct function body and a later direct-scope
   SIMT launch in that same wrapper. Pre-instrumented source is rejected, and
   generated/input kernel-launch counts must match. The report also records
   exact and normalized converter argv, compiler arguments, frontend SHA256,
   generator SHA256, and a deterministic conversion ID. Canonical
   `try_*_hybrid_count` fields count Hybrid-facade calls; the v3
   `try_*_simd_count` spellings remain equal-valued compatibility aliases.
3. Build and simultaneously link the four target-support libraries and the
   generated main CCE with the same ABI/CANN environment; inspect the exported
   `_launch_v1` symbols and `.so.1` ELF SONAMEs.
4. Run device correctness across both hybrid-selected and selector-miss
   shapes, including plain and affine RMSNorm plus LayerNorm boundary and
   in-place cases. For LayerNorm, run both the direct-ABI oracle and the
   generated-wrapper oracle so the converted adapter, injected generic
   facade, selector, ABI, and runtime are covered as one device path. The
   generated top-level dispatch gate covers its proved warp range through
   1024 columns; the direct ABI oracle additionally covers the v1 runtime
   ceiling through 8192 columns.
5. Measure only binaries that passed correctness, recording device state and
   the complete sampling protocol.

The shared device harness is described in
[`tests/softmax_rmsnorm_950/README.md`](../tests/softmax_rmsnorm_950/README.md).
Its `run_910_conversion_v3.sh` path is the legacy in-header recipe baseline;
it does not pass `--target-recipe` or build the four v1 hybrid target-support
DSOs. For this
explicit path, generate with `tools/generate_rowwise_cce.py`, build the target
support above, and then use the harness with `ROWWISE_SIMD_RUNTIME_DIR`.

## Reporting conversion and performance

Use separate denominators:

| Metric | Definition |
|---|---|
| Structural recognition rate | AST-proved in-scope operator implementation units / predefined in-scope units |
| Rewrite success rate | Units with hybrid dispatch inserted and SIMT fallback retained / recognized units |
| End-to-end qualified conversion rate | Units completing rewrite, support build/link, main-CCE build, and device correctness / in-scope units |
| Hybrid route coverage | Correctness-passing runtime cases selecting the mixed target kernel / all correctness-passing in-scope runtime cases |

An operator implementation unit is the counting unit for conversion rate;
wrapper count is reported separately. A selector miss may still belong to a
successfully converted operator, so hybrid route coverage is not conversion
rate. Legacy ABI, report, and filename spellings may retain `simd`; those names
do not imply an all-SIMD selected implementation.

Every performance result must identify the source and shape corpora, precision,
converter/runtime/build identities, CANN version, device, warmup/sample policy,
and aggregation. State the baseline explicitly. For example, `1.23x` is 123%
of baseline and a 23% increase, not a 123% increase. Do not carry a predecessor
payload's ratio onto a newly generated or rebuilt binary before its own build,
correctness, and measurement gates pass.

The intra-kernel change defined by ADR-0008 invalidates predecessor build,
correctness, and performance attribution. Current results require regenerated
Hybrid-facade output and freshly built, checked, and measured target-support
binaries from the same frozen source state.
