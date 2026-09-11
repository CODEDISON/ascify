# Preserve CUDA sample semantics in narrow adapters

Status: implemented; target validation is recorded separately per build.

The original twenty CUDA Samples expose several gaps between source parsing,
target source generation, and CCEC object compilation. These stages remain
separate results. An emitted translation or relocatable object does not prove
that a sample can link, execute, or produce correct results.

The opt-in frontend profile admits host `max(float, float)` with CUDA's
`fmaxf` semantics, exact host `min(int, int)`, and the verified register
operations of a 32-thread tile.
Host double/mixed-type max, tile synchronization, other tile sizes, and the
unimplemented cooperative reduction header remain outside this admission.
Target tile adapters preserve source return types instead of exposing a
wider native type that would change overload resolution. A `uint4` shuffle
performs four independent native 32-bit register transfers, preserving every
component rather than narrowing the value.

CUDA half2 is a class; the target half2 is a native vector. Proven device-local
two-component construction uses the native constructor function. Half2
multiply and add retain separate half rounding through explicit intrinsics;
unsupported macro, side-effect, or template contexts must not be rewritten
into a different computation. Parsing declarations are not arithmetic
implementations and must never enter a device numerical test as substitutes.

The device-property adapter exposes the current device's real SoC name and
vector-core count. It has no `major`, `minor`, or CUDA core-count emulation.
It does not change the caller's device binding, and writes the output only
after all required queries succeed. Full-warp any/all normalize the predicate
before integer max/min reduction; partial masks retain the existing rejection.

Symbol copies retain the original symbol object reference, declaration, and
compiler registration. The admitted SDK performs symbol lookup and transfer;
the adapter checks the byte offset and extent without overflow and propagates
runtime errors. A host-memory substitute, a fabricated registration table, or
a success return for an unsupported copy is not an acceptable implementation.
The DT CANN 9.1.0 probe compiled and linked but its first symbol-size lookup
returned 107043 (invalid device symbol). Symbol transfer is therefore
experimental and blocked at runtime in that tested compiler/runtime path;
partial, device-to-device, cross-file, and reset cases were not reached.
This is evidence of a toolchain/runtime boundary, not a hardware verdict.

Helper removal remains a transaction: every helper call, active macro, retained
header dependency, and raw source observation must be proven before any helper
edit is committed. A main-file `#ifndef MAX` fallback is admitted only after
an exact retained `helper_functions.h` / `helper_image.h` provider has already
executed the frozen internal MAX use. The active MAX must still have the exact
frozen `helper_cuda.h` body. The accepted fallback contains exactly `#ifndef`,
a two-argument `#define MAX(a,b) (a > b ? a : b)`, and `#endif`, on their own
logical lines with no additional tokens; parameter names may differ. Removing
the CUDA helper leaves MAX defined by the retained provider, so the fallback
stays inactive and its text is preserved. Only its two MAX name tokens pass
the raw-token audit. Changed definitions, missing providers, extra conditional
logic, macro expansion/stringization, and observations of other guards retain
the original helper boundary. Newly admitted property-query and symbol-copy
status calls use the existing trusted-declaration proof and ACL error domain;
this does not assert successful symbol registration or device execution.

Recursive local headers are still parsed as separate translation units. A
child that depends on a parent helper macro therefore has an explicit context
boundary. Injecting a helper header without proving the include-time
preprocessor state would change the input program. A future fix needs either
one root AST with edits across files or proven include-edge states and an
atomic helper-closure transaction. The negative regression checks that this
boundary publishes no partial output and preserves previous output on retry.

The frozen twenty-source roster remains unchanged. DT is the development and
test target for this change; PR validation is a separate later task. Generated
and compiled test artifacts are removed after retaining commands, diagnostics,
exit codes, identities, and result tables.
