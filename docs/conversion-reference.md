# Source conversion reference

This reference describes the `ascify-clang` command-line interface and the
semantic boundaries of generated source. For setup and a first conversion, use
the [English tutorial](user-guide.en.md) or [中文教程](user-guide.zh-CN.md).
Supported APIs and limitations are described in the
[compatibility reference](compatibility.md).

## CUDA SM display diagnostics

An ordinary host `printf` statement that only displays the SDK-owned
`cudaDeviceProp.major` and `.minor` fields as `%d.%d` can print `N/A (Ascend)`
instead. Ascend has no CUDA SM version. The fields remain unsupported for
capability checks, workload selection, arithmetic and other computation.
This conversion intentionally changes the displayed metadata; it never
supplies a fictional numeric compute capability.

The bounded rule requires the physical Linux libc `printf` declaration, an
ignored return value, a literal format containing `SM `, `Compute ` or `compute capability`,
and a single adjacent version pair from the same nonvolatile automatic local
property object. Only plain `%d`, `%s` and `%%` formats are admitted. Other
arguments must be simple expressions with no side effects. Macros, templates,
device functions, pointers, references, external property objects, positional
or width formats and used return values retain the source.

The same display rule can recognize two ordinary integer locals populated by
the SDK's major/minor attribute queries. Each local must have exactly one query
and one display use, with both queries preceding the display in the same block
and using the same simple device expression. Queries remain in the program;
the Runtime adapter reports those unavailable attributes as unsupported without
writing invented values. Any additional read, write, alias or capture retains
the original display.

Literal-zero assignments to the two property fields can be removed with the
display when every use of that entire local property object is the display,
one of those assignments, or an SDK property output query. Record copies,
member addresses, captures and numeric observations retain the assignments.
The display and dead assignments are published in one transaction.

## CUDA lane ID instruction

The frontend converts the exact nonvolatile inline PTX instruction
`asm("mov.u32 %0, %%laneid;" : "=r"(value))` when `value` is an automatic
`unsigned int` local of the current, non-template CUDA device or global
function. It writes the output once through an internal compatibility helper.
The helper uses the block's three-dimensional linear thread index modulo 32;
it preserves lane numbering across rows, planes, partial warps, and blocks.

This rule requires one unnamed output, the exact `=r` constraint, and no inputs,
clobbers, `goto`, or macro-expanded tokens in the statement. Other PTX strings,
volatile assembly, other output types, references, arbitrary output lvalues,
templates, lambdas, and host or host/device functions remain unchanged. The
generated helper uses the existing compatibility-header ownership and macro
collision checks. This rule does not translate general PTX assembly.

## CUDA runtime-API includes

A direct, unconditional literal `#include <cuda_runtime_api.h>` (or its quoted
form) can use the same existing Runtime adapters as `cuda_runtime.h`. It must
resolve to the physical `include/cuda_runtime_api.h` in explicit `--cuda-path`;
macro includes, conditional includes, same-named user headers and virtual file
overrides are rejected. The output uses ACL and Ascify compatibility declarations.
This adds an include spelling; it does not expand supported Runtime APIs.
Caller source and extra caller headers remain audited even when placed under
the CUDA include directory. SDK ownership requires the actual runtime-API or
compiler-bootstrap dependency chain as well as physical input identity.
Its dependency roles are limited to verified Runtime and forced-wrapper
relative paths; unknown SDK child files remain retained audit inputs.
The three `nv/target` and `nv/detail` bootstrap dependencies may resolve to
their physical `cccl/nv` copies through an SDK CCCL include path when those
copies have identical bytes. Their include ancestry and parsed bytes are
still checked; a modified copy or a caller shadow is rejected.

The projection rejects observations of CUDA SDK macros such as `CUDART_VERSION`
and header guards, including conditions, inactive branches, macro replacement
lists and pragma macro-stack operations in retained headers. User token pasting
is rejected because unused paste macros can construct SDK names later. Native
standard-library implementation paste macros remain available internally;
unexpanded caller replacements that depend on them are rejected, including
nested aliases. An arbitrary `-isystem` path grants no exemption. CUDA compiler
conditions such as `__CUDACC__` are also rejected. One physical native-library
condition is admitted by a boolean proof: in strict C++17 or later, the initial
`!defined(__STRICT_ANSI__)` conjunct is false in both source and target builds,
independently of CUDA or float128 macro values. Modifying that compiler control
or using GNU language mode invalidates the proof.

On the verified Linux source and target builds, the product's physical
`stdarg.h` entered from native `/usr/include/stdio.h` also admits its exact
`defined(__MVS__) && __has_include_next(<stdarg.h>)` condition: `__MVS__` is
absent, so the inquiry cannot change that branch. Caller or command-line
definitions and undefinitions of `__MVS__` invalidate this proof. A modified
resource copy and other inquiries remain rejected.
This does not promise that
the entire CUDA and target C++ compilation modes are equivalent. Existing supported qualifier
and API macro rewrites are admitted only in ordinary code. A source that uses
an unadapted SDK record (including CUDA vector structs with a different target
ABI) is rejected; `dim3` and existing explicit Runtime type adapters retain their
separately documented boundaries. Unsupported APIs, including managed allocation,
remain visible and fail target compilation instead of gaining a placeholder.
Those adapters do not admit SDK record reflection or template-type arguments:
`sizeof`, `alignof`, RTTI, class traits and template arguments can observe an
unproved target ABI or constructor property even for `dim3`.
SDK enum reflection is also rejected: an error-code adapter may use a native
integer rather than preserve the source enum category.
Compiler builtin classification and object-size inquiries also audit their
original SDK operands, including a member pointer whose containing object
has an unproved SDK layout. Direct native objects and constant aliases with
proved native origins retain their behavior. Mutable, externally supplied,
pointer-valued field and pointer-returning accessor origins are rejected when
their containing object cannot be proved. Caller macro replacements containing
these builtin inquiries are rejected because their future operands are unknown.
Empty `cudaDeviceProp{}` and `cudaFuncAttributes{}` initializers retain the
published zero-initialization contract without exposing SDK-only implicit
subobjects. Nonempty positional aggregate lists are rejected because their
SDK field order has no adapter proof.
Caller function parameters containing an SDK enum and template returns of an
SDK enum are rejected because their overload and type-selection domains are
unproved. SDK enum conversion operators are also rejected: converting an enum
adapter to a native integer can merge distinct conversion interfaces. Ordinary
Runtime calls, error comparisons and non-template status returns retain the
explicit adapter contract, including ordinary member-function status returns.
Numeric SDK enum casts and constructions, arithmetic, ordering, switch conditions,
array bounds, alignment values, bit-field widths, enumerator initializers and
assignments to native integers are rejected. Symbolic equality/inequality is
admitted only within the supported error-code or memory-copy-direction domains,
whose mapped symbols have distinct target values, or between two references to
the same mapped enum constant. Other SDK enum comparisons are rejected because
distinct source attributes can share a target adapter value. Ordinary device
attribute API calls and error-code truth retain their adapter contracts;
the target's native error ordinals are not CUDA ordinals.
The audit also visits instantiated function templates and generic lambdas.
Caller dependent types are rejected even before instantiation; a future caller
could supply an SDK record with a different target contract. This admission
covers nondependent caller code, including ordinary explicit Runtime adapter
functions. Native standard-library template implementation bodies do not grant
this exemption to caller templates. Alignment operands and enclosing record
layout observations are also audited.
Retained declarations that reuse a mapped Runtime type name are rejected;
their caller-owned namespace does not establish the SDK type identity.

An explicit Clang resource directory changes header lookup. Its extra files
remain retained inputs subject to this audit. A compiler-input exception
requires matching bytes from the executable's own installed resources (or its
exact build-tree resources) and the actual include chain rooted in the
predefined CUDA runtime wrapper. A caller header or preinclude cannot inherit
that exception from its location. Escaped caller pragma strings are rejected
because they can conceal SDK macro names from raw-token inspection.
Caller compiler inquiries such as `__has_include` and `__has_feature` are also
rejected; SDK header availability and CUDA features can change after projection.

`builtin_types.h`, `vector_types.h` and `cuda/cmath` do not have whole-header
compatibility mappings. In particular, CUDA vector struct layouts and libcu++'s
`cuda::` utilities cannot be inferred from similarly named target types or the
standard `<cmath>` header. Generated source must compile against the target
SDK without adding NVIDIA include directories.

## Integer intrinsic argument conversions

For a proven CUDA `__ffs(int)` or `__clz(int)` call, Ascify preserves an implicit integral
argument conversion by spelling it as `static_cast<int>(argument)`. Native
SIMT also overloads these functions for `long` and `long long`; selecting an overload
from the original operand type could change the bit width or make an unsigned
call ambiguous. The explicit conversion retains CUDA's resolved parameter type
and evaluates the argument once.

This rule requires a direct call in a non-template device or global function,
the exact SDK/compiler-owned declaration chain, and an ordinary integer
expression written in the current source file. Macro-expanded callees or
argument-expression node boundaries, arguments containing calls, floating-point
conversions, dependent code and caller-owned overloads remain unchanged. An
operator-only macro token inside an otherwise ordinary expression retains its
original spelling and may be enclosed by the cast. An already-`int` argument
needs no edit. The rule does not add a general intrinsic-name mapping or adapt
`__ffsll` or `__clzll`. CUDA frontend/host configurations declaring `__clz` as
`unsigned int(unsigned int)` do not match this rule's exact `int(int)` contract.

## Input filesystem overlays

Ascify rejects `-ivfsoverlay` before preprocessing. Its current LibTooling entry
does not install the driver's overlay into the supplied file manager; accepting
the option would silently parse different input bytes. This rejection applies
to all conversions. Explicit remapped SDK contents also cannot qualify for a
physical runtime-API header projection.

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

A literal main-file invocation of the frozen `helper_cuda.h` `MAX` macro can
be materialized as its original conditional expression. The converter uses
Clang's expanded argument tokens and retains the original argument precedence
and repeated evaluation. It does not replace the macro with a function or the
different `helper_image.h` definition. Stateful counter expansion, residual
active macros, unsupported tokens, indirect invocations and conflicting source
edits retain the complete helper transaction. Other helper include, identity
and macro-observation checks still apply.

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
