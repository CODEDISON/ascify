# Quoted local-header closure

`--local-headers` and `--local-headers-recursive` convert user-owned quoted
headers into a managed artifact bundle beside the root output.  The feature is
for source closure, not for widening Ascify's CUDA API support.

## Admission boundary

- Existing CUDA include-map entries are handled first.
- Only literal quoted includes resolved by Clang inside the main source
  directory or an explicit Ascify `-I` root are eligible.
- Angle includes, toolchain headers and unresolved optional edges are not
  converted.  In particular, this feature does not admit `<helper_cuda.h>`.
- `--local-headers` selects only direct root edges.  The recursive form follows
  newly selected quoted-header edges to a fixed point.
- Canonical paths deduplicate aliases, shared children and header cycles.
  Resolved selected, excluded and root dependencies are retained in one
  canonical identity set even when an edge itself is not converted.
  Absolute spellings, root escape, a header edge back to the root input, strict
  recursive macro includes and `.dpp` input roots fail closed.

## Artifact contract

For root output `<output>`, converted headers are staged below
`<output>.headers/` and published only after every selected node and redirected
edge validates.  The bundle carries an `.ascify-local-closure` marker.  Ascify
will replace only a bundle with that marker; a file, symlink or unmanaged
directory at the destination is rejected.

The root and bundle are staged in the destination filesystem.  A publish
preflight weakly canonicalizes the root output, bundle and both transaction
backup names.  This resolves every existing parent-directory symlink before
comparing them with canonical input identities.  A root/backup that aliases an
input, or a bundle/backup that aliases or contains an input, is rejected before
publication.  Explicit `--inplace` is the only exception: the root output may
equal the root input, but no other artifact/input alias is admitted.

Every publish, move-aside, rollback and backup-cleanup rename/remove is checked.
A publish failure restores the previous managed root/bundle pair.  A first
rollback operation failure is reported and retried once; a persistent restore
failure is reported as a rollback failure, never as a successful transaction.
Likewise, backup cleanup failure makes the command fail explicitly.  This is
not a crash-recovery journal: abrupt process or host termination between the
two final renames can still leave an artifact requiring operator inspection.

`--no-output` performs analysis without publishing root or header artifacts.
Multiple inputs that would collide under `-o-dir` are rejected before any
conversion starts.

## Parent source context

A local header can use a macro or declaration defined by the root before its
include. In recursive mode, Ascify detects root-defined macro expansions and
references to root declarations, records, typedefs and template types in selected
direct headers. These dependencies can be preserved by converting the header
at its original include position. Ascify does not manufacture a standalone
macro environment or copy initializer values.

A selected direct leaf may also reference a declaration or macro from an earlier
selected sibling. Only this added sibling rule requires the consumer to have
no quoted includes of its own; otherwise it uses ordinary recursive closure
parsing. This avoids mistaking a self-contained header's own included provider
for an inherited sibling dependency. Macro observations are deferred until the
original preprocessing pass completes, so later quoted includes are considered.
The original root-defined dependency rule remains separate from this sibling
rule. A non-leaf that genuinely needs a root definition is rejected unless it
satisfies the physically skipped helper-context rule below.

This path requires standard preprocessing (`--default-preprocessor`), no
`--amap`, and no target recipe. A candidate must have exactly one direct literal
include on its own line, one file entry, a final newline and no pragma
(including raw pragma introducers in inactive branches). Nested quoted includes
normally remain outside the boundary. A virtual expansion uses `#line` to retain source
locations, then a second parse must produce the same expanded token digest and
final macro-state digest as the original complete translation unit. Builtins
such as the include level and counter can make that proof fail. Input bytes,
canonical path bindings and include edges are bound to the proof and rechecked
before atomic publication. Unsupported contexts leave previous outputs intact.

When a selected header also depends on a recognized sample-helper macro, its
existing joint helper transaction remains mandatory. Pure parent source
dependencies do not require a helper include. See the [parent source context
regression](../tests/rewrite/check_parent_source_context.py) and [helper context
regression](../tests/rewrite/check_inherited_sample_helper_context.py). This proof
preserves source context; it does not establish target API support or device
correctness.

For a joint helper transaction, a selected direct local header may itself
provide the recognized helper include. It is expanded at its original position;
the helper is still accepted only by the existing direct-include provenance and
atomic transaction proof in the joint parse.

A nested quoted include in this context is admitted only when Clang reports
`FileSkipped` in both discovery and original-context parses. The resolved file
must have exactly one actual physical entry, unchanged bytes, and the same
physical identity and path binding. Every additional incoming edge to an
inlined provider must be such a skipped nested edge; repeated direct root
includes are still rejected. Each skipped directive must occupy one literal
physical line without continuation, macro include or extra tokens. Only those
directives are replaced by spaces in the virtual input, retaining newlines and
avoiding a new relative lookup from the root. Token and final macro equivalence,
input revalidation and the helper transaction remain mandatory. A pre-defined
guard, a real nested entry, re-entry, opaque pragma or unproved skip does not
qualify. The marker records the original skipped edges. See the
[frontend regression](../tests/rewrite/check_context_skipped_includes.py).

## Literal pragmas in contextual headers

A contextual header may contain one literal `#pragma once`, plus literal
`#pragma unroll` directives with an optional decimal count. Its complete raw
input, including inactive branches, must contain no other pragmas, macro pragma
operators, continued directives or macro count. Every selected input must be
ordinary physical preprocessing, without modules, PCH, precompiled preambles or
an external AST source. User `#line` and GNU line directives anywhere in entered
physical inputs prevent this new admission because they could spoof pragma
position evidence. Existing pragma-free context rules are unaffected.

The unique original entry and every skipped re-entry must pass the existing
include proof in both original parses. Only the exact once line is blanked in
the virtual input, preserving line boundaries; original files stay unchanged.
All non-once active pragma events have an ordered digest of their complete
physical line and presumed source location. This digest must match discovery,
original and joint parsing, alongside expanded tokens, final macros and input
identity. The payload digest is necessary because annotation tokens do not
expose an unroll count. The published marker identifies only normalized once
lines from actual inlined headers. Unknown pragmas remain rejected.

[Focused cases](../tests/rewrite/check_context_literal_pragmas.py) validate
original CUDA syntax before conversion and check atomic rejection with both
fresh and pre-existing output. This rule supplies a context prerequisite; it
does not by itself admit packed records or claim a complete Sample passed.

## Runtime API includes inside a header guard

A selected local header may directly include the physical configured
`cuda_runtime_api.h` inside one conventional whole-file `#ifndef` / empty
`#define` guard. This exception applies only while translating an actual
selected header, not to conditional includes in the root translation unit.
The guard remains unchanged in the output. The complete raw-token structure
must end at its matching `#endif`, with no outer `else`/`elif`, guard mutation,
or tokens outside the guard. Macro-stack pragmas and `_Pragma`/`__pragma`
operators in the selected header are conservatively refused, including
inactive source and macro replacement lists. An additional conditional around the Runtime
include, a macro include, or a shadow SDK path still fails admission.

This rule reuses the existing physical SDK identity and retained-source
audits. It does not replay parent declarations/macros, authorize arbitrary
conditional SDK inclusion, or prove a whole Sample can build or run. The
[focused converter check](../tests/rewrite/check_local_runtime_api_guard.py)
covers guarded positives, structural/conditional refusals, SDK shadows and
preservation of previously published output on failure.

## Verification boundary

The host model exercises path selection, canonicalization, parent-directory
symlink aliases, selected and excluded dependency identities, cycles, shared
children, direct/recursive depth, checked rollback retry and managed-bundle
replacement.  Static checks verify the C++ callback/context plumbing.  When a
real Ascify binary is supplied, the release gate repeats selected/excluded
parent-symlink and unmanaged-bundle negatives.  On Linux it also interposes a
one-shot filesystem failure to exercise root publish failure and checked
rollback-rename retry, then verifies byte-identical inputs/old outputs and no
transaction-backup leak.  These tests prove publication behavior only; they do
not establish target compilation, device correctness or crash recovery.

## Preserved packed integer records (experimental)

A separate proof can preserve literal `#pragma pack(1)` regions followed by a
same-file literal `#pragma pack()`. The input must start with ordinary default
packing options, without serialized AST inputs, PCH, modules or preambles.
Unknown pragma failures remain failures. Includes, file transitions, nested
regions and unowned raw pack directives, even in inactive source, reject the
new proof. Macro pragma construction and cleaned tokens are excluded from a
file that supplies an admitted region.

The region contains only plain global structs and typedefs of 8/16/32-bit
integer scalars. Macro-derived declarations, pointers, arrays, initializers,
bitfields, methods, base classes, dependent records and source attributes are
excluded. The original AST must show one-byte packing and contiguous field
layout. All original record and pragma bytes remain unchanged. Target
`static_assert` witnesses check record size/alignment, every field offset,
width and signedness, including plain `char`. Both original insertion-time
macro checks and immediately adjacent target guards protect generated witness
syntax from macro replacement.

A common conditional footer is allowed: a disjunction of `defined(NAME)` terms
selects an inactive `pack(pop)` branch and an active `pack()` branch. Those names
must all be undefined at the actual source points. An optional preceding
conditional `pack(push)` must use the identical condition. The original branches
remain intact; an error guard immediately before `pack(1)` prevents target
headers or flags from activating the other configuration. No platform or sample
name grants permission.

The final replacement set, including helper edits, token rewrites and generated
includes, is checked against the complete protected interval and its insertion
boundaries. Only the proof-owned guard and post-region witnesses may be added.
Physical path aliases cannot bypass this check. Publication is transactional.
When such a header also requires inherited context, only region-owned raw pack
lines join the literal once/unroll context proof; complete ordered pragma,
token, macro, input and skipped-edge checks still apply.

The [focused matrix](../tests/rewrite/check_packed_record_conversion.py) validates
original source syntax before every conversion, preserves old outputs on
rejection, and can compile all positive outputs with actual CCEC. This source
feature is still awaiting those checks and complete original DCT verification;
it is not a new numeric Sample result.

## Packed integer record regions

With the default preprocessor, literal `#pragma pack(1)` regions can preserve
closed, same-file records containing only proved 8-, 16- or 32-bit integer
fields. Layout witnesses retain each source size, alignment, offset and integer
signedness. Pointers, arrays, bitfields, templates, nested packing and arbitrary
packing state remain outside this path.

An optional inactive `pack(push)` prefix and `pack(pop)` footer must use the
same literal disjunction of undefined names. Blank lines between the prefix
and opener retain their original bytes; directives, macro histories and final
edits remain subject to the existing proof. No inactive pack directive is
accepted merely because it is near a supported region.

Target compilation must retain the source's plain-char signedness. For the
verified AArch64 CUDA compiler configurations whose plain `char` is unsigned,
CCEC requires explicit `-funsigned-char`; a mismatched target mode fails the
generated signedness witness. This is a configuration-specific build option,
not a global reinterpretation of `char` or removal of source assertions.
