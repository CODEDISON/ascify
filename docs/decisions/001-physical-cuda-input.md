# ADR-001: Parse physical CUDA inputs without rewriting them

Status: Accepted for this candidate; release validation is tracked separately.

## Context

Parsing a copied CUDA source in a temporary directory loses its native quoted
include parent. Adding the original directory as `-I` does not restore that
precedence: a profile header or an explicit `-iquote` can win first. Adding a
global root `-iquote` is also incorrect, because it changes the fallback search
order of nested headers. Both differences have executable preprocessing
counterexamples.

## Decision

For ordinary CUDA parsing with LLVM 13 or later, run RefactoringTool at the
original input spelling using `run()`. Freeze the canonical physical target's
UID, size, bytes and digest, and verify the parsed main file against that
snapshot before and after the action. The lexical path must still name the
same physical input. Reject an overridden main file or a changed binding.

Apply replacements to the frozen bytes only after parsing succeeds. Publish to
the requested output or the existing closure staging file. An explicit
`--inplace` request still authorizes final replacement of the original file.
`--no-output` performs analysis without publishing; `--save-temps` retains a
separate converted debug copy.

Keep Host C++, contextual reparses and selected kernel-policy-header parsing
on their existing paths. The snapshot implementation is shared with the host
identity helper, but this CUDA action receives no host, profiler, SDK or virtual
input permissions.

The local-header ownership plan currently stores a canonical root identity.
Until it also models a lexical root, reject an aliased closure root explicitly
rather than silently searching the target's directory. Ordinary CUDA parsing
without a closure retains its complete absolute lexical source spelling.
Closure roots also require the unnormalized parent's physical identity to
match the canonical root parent: equal source-file targets alone do not
establish equal quote-parent lookup through directory aliases and `..`.

## Consequences

Root and nested quoted includes follow the parser's own lookup rules. The
change adds no global include directory and grants no additional type, macro
or SDK permissions. Physical source and header bytes remain unchanged during
parsing. Legacy LLVM versions retain their existing temporary-input path.

Validation must cover root versus explicit quote directories, nested-header
fallback, source aliases, explicit inplace, no output, debug copies, the
original provider-copy counterexample, and the unchanged frontend matrices.
Successful parsing or native product tests alone do not establish CCEC or NPU
support for an original sample.
