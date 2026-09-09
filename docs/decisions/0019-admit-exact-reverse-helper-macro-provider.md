# ADR-0019: Admit the exact reverse helper macro provider

## Status

Accepted

## Date

2026-08-24

## Context

The proof-gated CUDA Samples helper transaction removes one direct
`helper_cuda.h` include only when every dependent helper call, macro use, and
declaration can be replaced atomically. NVIDIA cuda-samples v13.3 contains two
direct include orders for its frozen Common headers.

When `helper_functions.h` precedes `helper_cuda.h`, the exact
`helper_image.h` defines `MAX` and `EXIT_WAIVED` before the exact CUDA helper
consumes them. ADR-0017 admits that exact provider graph. In the reverse order,
used by the official `scalarProd` sample, `helper_cuda.h` first defines
`MAX(a,b)` and the later exact `helper_functions.h` enters exact
`helper_image.h`. The image helper expands the active CUDA-helper `MAX` once
inside its own frozen body. The previous policy treated that expansion like a
dependency from an arbitrary user header and rolled back the whole helper
transaction.

Removing `helper_cuda.h` from this reverse graph makes the unchanged exact
image helper publish its own `MAX` before the same internal use. The two
frozen bodies differ only in redundant parentheses and have the same
two-argument conditional semantics. This is an include-order provider
substitution problem, not an operator-specific `scalarProd` rule.

## Decision

Admit exactly that reverse-provider substitution during the helper
transaction. The admission requires all of the following:

- the active macro is `MAX`, defined by the exact frozen v13.3
  `helper_cuda.h`, with its exact parameterized token body;
- the expansion location is the exact frozen `helper_image.h`, its
  non-overridden physical identity was recorded on first file entry, and this
  particular image FileID was entered directly from the proven functions
  root;
- the exact frozen `helper_functions.h` root was entered; and
- that root came from a direct, literal, unconditional main-file include, so
  removing `helper_cuda.h` cannot make the replacement provider disappear.

The admission is checked on the current image FileID, not cached on the
image's physical identity. Seeing the same exact file once below the proven
root therefore cannot lend ancestry to a later include instance reached from
the main file, a conditional path, or another wrapper.

Preprocessor conditional depth is tracked only to prove the last condition.
The image/functions roles remain macro-provider roles; they do not enter the
call, redeclaration, attribute, or status-domain trust masks. Any expansion in
a local or user header, a conditional or transitive provider root, an altered
file, a changed macro body, a remapped file, or a command-line/redefined
policy macro keeps the existing all-or-nothing rollback.

The transaction also treats the macros removed with exact `helper_cuda.h` as
observable state: its include guard, `EXIT_WAIVED`, `MAX`, and the three error
macros. Every evaluated main/external `ifdef`, `ifndef`, `defined`, `undef`,
`elifdef`, and `elifndef` callback for those names keeps the include and every
helper edit, including macro-generated names that a raw spelling scan cannot
see. A skipped external `elifdef`/`elifndef` fails closed as a class because
its callback exposes only an uncleaned raw range.

Hash pragmas use a direct-spelling allowlist. Direct literal macro-stack
operations are admitted only for an ASCII identifier outside both the removed
helper surface and Ascify's reserved compatibility surface. An observable,
macro-produced, escaped, or otherwise unresolved stack target rolls back the
transaction. Exactly one optional trailing semicolon on a literal stack pragma
is admitted, as used by Clang's CUDA texture header. Direct `once`, system-header, diagnostic, numeric unroll, and
fixed CUDA warning-control forms are admitted; other hash pragma classes fail
closed, including annotation handlers whose argument lexing could reveal a
helper macro through an unrelated alias. LF, CRLF, and bare-CR line splices
are normalized. Trigraph and block-comment spellings fail closed instead of
duplicating Clang's preprocessing phases.

An additional fixed form is admitted only in files already trusted as system
headers on their first entry: `#pragma GCC visibility push(default)` and its
matching syntax `#pragma GCC visibility pop`. libstdc++ 11 uses these around
portable standard declarations. Ordinary user headers and later
`system_header` promotion do not acquire this admission; hidden/protected,
macro-produced, or extra-token visibility forms remain outside it. DT's
2026-09-09 frontend run exposed these standard-header forms and the CUDA
macro-stack semicolons as false rejections of the helper transaction.

For non-hash `_Pragma`/`__pragma`, only a diagnostic pragma applied in a file
that was already a system header on its first entry is admitted. A generic
callback creates a pending obligation, and only Clang's specialized diagnostic
push/pop/mapping callback can discharge it. An untrusted location, unknown
pragma class, or obligation still pending at the next pragma or end of the
translation unit rolls back the transaction. Exact v13.3 helper
timer/image/functions files remain macro-graph roles only.

## Alternatives Considered

### Allow helper macros in every CUDA Samples Common header

This would make a filename or directory act as trust and could remove
`helper_cuda.h` while an unrelated header still depends on its definitions.

### Special-case the scalarProd source path

That would hide the include-order contract and would not generalize to another
sample using the same official helper graph.

### Keep fail-closed behavior for the reverse order

This is safe but rejects an exact upstream dependency graph whose
counterfactual provider can be proven without expanding runtime or target ABI
surface.

## Consequences

- Exact reverse-order CUDA Samples can complete the same atomic helper
  transaction as the already admitted forward order.
- User-header and conditional/transitive provider dependencies remain
  fail-closed.
- The change only closes host-helper provenance. It does not by itself prove
  whole-TU target compilation, device correctness, performance, or generic
  Hybrid SIMD+SIMT support for a new kernel family.
- Release gates cover both official include orders plus altered consumer,
  missing-root, conditional/transitive/wrong-ancestry roots, macro-state
  observation, and external-user-header negatives.
