# Preserve CUDA sample semantics in narrow adapters

Status: implemented; target validation is recorded separately per build.

The original twenty CUDA Samples expose several gaps between source parsing,
target source generation, and CCEC object compilation. These stages remain
separate results. An emitted translation or relocatable object does not prove
that a sample can link, execute, or produce correct results.

The opt-in frontend profile admits host `max(float, float)` with CUDA's
`fmaxf` semantics and the verified register operations of a 32-thread tile.
Host double/mixed-type max, tile synchronization, other tile sizes, and the
unimplemented cooperative reduction header remain outside this admission.
Target tile adapters preserve source return types instead of exposing a
wider native type that would change overload resolution.

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
