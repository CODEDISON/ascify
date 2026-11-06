# Owned target for aggregate device helpers

Status: implementation candidate; target and device validation pending.

CCEC cannot compile some CUDA device helpers that return a large trivial
record through its ordinary device-call ABI. Forced inlining removes that
unsupported boundary. The original source can enable relocatable device code,
so a single parsed translation unit alone cannot establish that no external
device caller observes the helper symbol.

Ascify admits this lowering through an explicit owned-executable mode. The tool
parses the original physical CUDA root and selected headers, preserves its RDC
option, binds the source and native providers, then builds a new fixed target
with exactly one generated device object and no user device archives. This
construction supplies the target boundary. Source names and a caller manifest
supply no such authority.

The analysis records the helper's exact type layout, definition, direct call
and reference graph, shared-memory uses, source bytes and macro observations.
Each selected header proves and consumes its root facts before the output
transaction can commit. Helper address escape, assembly observation, ambiguous
macro provenance and unproved source-location observations are rejected.

A separate register-only nominal-width32 adapter permits a contiguous final
warp prefix. Its source proof admits only the closed register operations it
implements. The adapter checks the active membership and exposes neither a
collective barrier nor a general reduction. The caller's launch dimensions and
algorithm are retained.

CANN dependencies, source and alias directory identities, generated files and
the private link directory are checked around each child stage. The complete
build artifacts join the existing atomic output transaction. A compiler failure
keeps private diagnostic evidence and publishes no completed executable bundle.
This mode compiles and links the target; it does not execute it.

The ordinary translation path retains its existing admission rules. Existing
cooperative-group, runtime and aggregate source boundaries remain independent.
