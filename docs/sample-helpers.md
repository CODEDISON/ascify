# NVIDIA CUDA Samples helper compatibility

Ascify can replace a limited part of NVIDIA CUDA Samples' `helper_cuda.h` after proving its source
identity, active definitions, and uses. Recognition is automatic; there is no helper-enable flag. The
frontend profile does not enable or widen it. This closes a source dependency; it does not establish
that an entire sample compiles for Ascend, runs correctly, or meets a performance target.

## Supported operations

| Source operation | Generated behavior |
|---|---|
| `checkCudaErrors(expr)` | Evaluate `expr` once; on failure print expression, file, line, numeric ACL status and available error text, then exit with failure. |
| `getLastCudaError(message)` | Call Ascify's consuming `cudaGetLastError()` once, clearing a pending lifecycle error or querying ACL's consuming thread error; report failure and exit. |
| Direct host `findCudaDevice(argc, argv)` | Query the visible logical-device count, require exactly one, bind logical device zero and return zero. |

An ordinary status argument must resolve to a direct call from trusted system declarations, appear in
the explicit Runtime-status allowlist, and map to an `ascify::` Runtime implementation returning
`aclError`. Driver, library, occupancy, arbitrary integer-returning functions and project overloads are
rejected. Admission of a status domain does not establish support for every argument or runtime state;
for example, a symbol-transfer error remains an error. The current allowlist is in
`isAdmittedCudaRuntimeStatusCall` in [AscifyAction.cpp](../src/AscifyAction.cpp).

Device selection admits no selector or one exact `--device=0`. Other recognized selector spellings or
values, duplicates, invalid argument storage, null arguments after `argv[0]`, zero/multiple visible
devices, and query/bind errors fail explicitly. Unrelated command-line arguments are ignored. CUDA
GFLOPS ranking, CUDA SM property emulation, and general multi-device selection are outside this
contract.

## Source identity and call proof

The error macros require the recognized NVIDIA helper structure and exact active token contracts:
`check((p), #p, __FILE__, __LINE__)` and `__getLastCudaError(p, __FILE__, __LINE__)`. Parameter spelling
may differ; inactive official-looking text cannot authorize a changed active definition. Same-name
project headers, macro aliases, and indirect macro invocations gain no admission.

Device selection additionally requires the reviewed, frozen helper-file profile. Basename, byte identity
and SHA-256 are checked against the actual parsed buffers; the current identities live in
`frozenNvidiaSampleHelperFileRole` in the source, with the [vendored positive
fixture](../tests/rewrite/fixtures/nvidia_samples/Common). Frozen SHA-256 admission requires LLVM 13 or
newer. The full active `findCudaDevice` definition, its transitive calls, redeclarations, source-backed
attributes and macro dependencies must retain trusted provenance. Changed bodies, user interposition,
aliases and symbol redirection are rejected. The call must use an unqualified, non-macro
`findCudaDevice` token in a main-file host function. Qualified calls, function pointers, indirect calls,
using aliases, device calls and external-header uses outside the proven leaf context retain the
dependency. Only the callee token changes, preserving both arguments and return-value use.

Trust in system files is recorded at their first physical-file entry. Later `system_header` pragmas,
re-entry, forged `#line` filenames, remapped provider files and command-line definitions cannot acquire
that provenance. The configured converter, Clang resources, CUDA SDK, sysroot and system include paths
remain a trusted configuration boundary; adding an untrusted `-isystem` directory enlarges it.

## Frozen macro providers and observable state

Both official include orders are supported only with the proven dependency graph:

- **Functions first:** exact `helper_functions.h` enters exact `helper_image.h`;
  the image provider supplies `EXIT_WAIVED=2` to the frozen CUDA/string helpers
  and the exact two-argument `MAX` body to the CUDA helper's `#ifndef` dependency.
- **CUDA helper first:** the frozen CUDA helper's `MAX` may expand inside the exact
  image helper reached from the exact functions root. The root must be a direct,
  literal, unconditional main-file include; ancestry is checked per include instance.
  Removing the CUDA helper then leaves the equivalent image-provider definition.

Provider roles do not grant general function, declaration or macro trust. Alternative providers, changed
bodies, conditional/transitive roots, wrong ancestry, redefinitions and user/command-line policy macros
retain the complete helper transaction. A main-file `#ifndef MAX` fallback is admitted only after that
retained provider's proven internal use; it must contain just `#define MAX(a,b) (a > b ? a : b)` and `#endif`
(parameter names may differ). Extra logic, changed definitions or other observations fail.

Unproved preprocessor observations of the removed include guard, policy macros or error macros
prevent closure, including indirect names and unresolved skipped branches. Pragmas use a closed
allowlist; unknown or unresolved forms and macro-stack operations on helper or reserved output names
retain the transaction. `_Pragma`/`__pragma` requires a confirmed diagnostic operation in an initially
trusted system file. The [helper tests](../tests/rewrite/check_sample_helper_compat.sh) cover exact
accepted forms, macro-state observations, spoofed spellings and unsafe provider mutations.

## Include replacement and publication

There must be exactly one removable, direct main-file include using the literal `helper_cuda.h`
spelling. Duplicate, relative-path, macro-expanded or transitive includes are rejected. At that position
Ascify preserves these portable includes, in order: `<stdint.h>`, `<stdio.h>`, `<stdlib.h>`,
`<string.h>`, `<helper_string.h>`. It inserts the authenticated `<ascify/ascify_cuda_compat.hpp>` when
needed; a later compat include does not establish that its declarations were available earlier.

Every helper macro, device-selection rewrite, template guard and include change is staged together.
Declaration uses, unresolved template lookup, raw tokens through EOF, retained headers, macro state and
edit conflicts must all pass before commit. Ordinary proof failure retains the original helper include
and calls; conversion output may consequently retain an unsupported dependency. Reserved compat macros,
backing identifiers, namespace collisions or a counterfeit compat header can fail translation outright
when another CUDA mapping requires that header. `ASCIFY_*` is reserved for generated output;
synthesizing those identifiers by token pasting is outside the accepted source contract. Occupancy,
Driver/VMM, compression and compressible-allocation helpers remain unsupported.

## Dependent status calls and local-header context

A dependent status call has a separate bounded proof: one named, non-pack type parameter in a main-file
host function template, with at least one instantiated body. Every observed specialization must be an
implicit instantiation or explicit instantiation definition for exact `int` or `float`, and every status
call must resolve to the proven helper and Runtime domain in every instance. The output receives a
`static_assert` permitting only the observed type set. Other scalars, cv-qualified types, class ADL,
explicit specializations, extern instantiations, missing bodies, member/multi-parameter templates,
nested-lambda owners and macros changing the guard's insertion tokens fail the proof. Guard and helper
publication remain one transaction. This feature is experimental; validate the generated code with
your target toolchain.

With `--local-headers-recursive`, a selected leaf using a parent's helper may be converted jointly. This
requires `--default-preprocessor` (or its skip-excluded synonym), no `--amap`, and
`--target-recipe=none`. Each candidate needs one direct literal include on its own line, one file entry,
a final newline, no nested quoted include and no pragma. Ascify virtually expands the original leaf at
its include position and requires identical expanded tokens and final macro state, including actual
builtin expansions and the counter. The complete helper transaction must also succeed. Original paths,
bytes, include edges and proof digests are recorded and rechecked before publication. Failure publishes
neither partial output nor a partial bundle and preserves previous output. Inputs remain unchanged. See
the [local-header closure contract](local-header-closure.md).

## Maintenance and verification

The runtime helpers are in [ascify_cuda_compat.hpp](../include/ascify/ascify_cuda_compat.hpp); context
handling is in [LocalHeader.cpp](../src/LocalHeader.cpp). Updating a frozen provider requires reviewing
the upstream diff and updating identity checks plus positive and mutation-negative fixtures together.
The relevant gates are [helper compatibility](../tests/rewrite/check_sample_helper_compat.sh), [template
domains](../tests/rewrite/check_sample_helper_template_domain.py) and [inherited
context](../tests/rewrite/check_inherited_sample_helper_context.py). Host tests do not establish target
compilation, linking, device correctness or successful process exit. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for validation methods.
