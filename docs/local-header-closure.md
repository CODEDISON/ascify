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

This path requires standard preprocessing (`--default-preprocessor`), no
`--amap`, and no target recipe. A candidate must have exactly one direct literal
include on its own line, one file entry, a final newline, no nested quoted
include and no pragma. A virtual expansion uses `#line` to retain source
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
