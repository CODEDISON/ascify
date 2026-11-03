# Bounded binary64 cooperative reduction

## Status

Accepted source-proof design. Target resource and device acceptance are separate gates.

## Context

The complete reduction source uses int, float and double instantiations of the
same kernels. Its ordinary tile32 helper forwards a typed reduction; its
multi-warp kernel adds shared storage, grid-stride loops, an empty dynamic-shared
pointer wrapper and an inactive integer optimization. Parsing declarations or
preserving the source template alone cannot establish the target arithmetic,
participation, storage ABI or inactive-source behavior.

## Decision

Keep binary64 values as eight-byte values. Use the existing integer RNE adder
at each source addition and reduction tree edge. Ordinary tile32 reduction uses
five typed down-shuffles and one broadcast. Uniform scratch reduction stores
binary64 bits in separate uint64 partial/result arrays, preserves the existing
int/float paths and adds no synthetic positive-zero terms to the second stage.
The fixed RNE and NaN policy remain those of the scalar64 backend.

Admit only the authenticated SDK plus/reduce declarations and the separately
proved groups. A pure two-type forwarding function obtains explicit scalar and
group guards; reflection, competing lookup, address escape and unproved future
instances remain outside the contract. SDK coordinate reads require their real
property/getter/builtin identities. An address NTTP requires a physical written
integer-literal witness for each concrete substitution. A source-valid explicit
instantiation with no such argument witness remains unsupported.

Keep the original loops and selected shared-pointer conversion. Prove unique
scratch ownership and the collective call sequence independently of loop
participation. Reject loop-local barriers/collectives, holder aliases and bound
mutation. This does not make an originally overflowing, out-of-bounds or
nonterminating execution valid. Inspect the selected conversion and retain the
existing audit of unselected methods and the surrounding class.

Permit only a structurally complete outer guard with one empty main-file
definition, a witnessed native stdio include, and one exactly recognized inactive
integer specialization. Check the real SDK qualifiers/intrinsic and the related
active primary declaration. Check incoming controls and complete macro history.
Retain the original architecture condition and specialization. Emit a target
preprocessor error before that condition if `__CUDA_ARCH__` is defined, so a new
compiler flag cannot silently select an unproved branch.

Both scalar and uniform guards must survive a shared insertion offset. Compose
only the exact zero-length insertion already committed by the scalar proof,
with the uniform guard first. Preserve all other replacements and refuse
unknown overlaps transactionally.

## Alternatives

Narrowing double through float loses source values and reduction order. Reusing
int/float scratch slots through pointer aliasing adds unproved layout and
aliasing behavior. Treating every SDK callee, constant expression or conditional
as trusted would bypass the participation and physical-source proofs. Dropping
the inactive specialization or defining a CUDA architecture for CANN would
change original source choices. None of those approaches is admitted.

## Verification consequences

Run the versioned 47-case matrix with the final installed converter, including
original CUDA syntax checks, preservation of existing outputs, and absence of
new bundles on refusal. The historical double-plus and namespace future-double refusals are
intentionally replaced by positive fixtures; other refusal boundaries remain.
Retain failed predecessor evidence and bind every run to its actual product.

Rebuild the complete original two-TU target, with host C++ and CUDA input modes
preserved, all int/float/double instances and original launch arguments. Actual
CCEC compilation, symbols, static/dynamic shared resources and loaded providers
must be checked independently. A host model, a diagnostic mixed-object frontend,
or successful source conversion does not establish NPU correctness or full
sample acceptance.
