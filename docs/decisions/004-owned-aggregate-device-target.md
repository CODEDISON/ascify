# Owned target for aggregate device helpers

Status: accepted for the bounded target and original-sample validation below.

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

An implicit block-barrier call requires the normalized NVPTX compiler builtin
identity, an implicit body-free `void()` declaration across all redeclarations,
and the literal source tokens `__syncthreads ( )`. The helper must also pass the
existing shared-access, single-call and top-level caller proof. Clang's implicit
`SC_Static` is admitted only for a direct literal
`__shared__ unsigned int name[number]` declaration with authenticated qualifier
macro provenance and no written or injected `static` or `extern`. The original
array type, size, initializer and attribute checks remain.

Straight declaration setup before that call may consume a CUDA execution
coordinate only through the existing complete property, variable, resource,
getter and NVVM-return proof. It skips the semantic getter only for that exact
proved property expression. Explicit getter calls retain their original checks.

Native-input classification is memoized only within one CompilerInstance, keyed
by FileID and system-header status. Every touched file and its actual include
ancestors are captured and revalidated, and the unchanged classifier is replayed
before facts or edits can commit. This memo supplies no new native or SDK role.

A separate register-only nominal-width32 adapter permits a contiguous final
warp prefix. Its source proof admits only the closed register operations it
implements. The adapter checks the active membership and exposes neither a
collective barrier nor a general reduction. The caller's launch dimensions and
algorithm are retained.

The original compiler arguments after `--` are captured before the tooling
compilation database can discard positional linker inputs. The owned mode uses
the Clang option table to reject extra inputs and linker or archive options;
option operands such as include paths keep their normal grammar.

CANN dependencies, source and alias directory identities, generated files and
the private link directory are checked around each child stage. Each checkpoint
still reads and hashes complete file contents. Buffered whole-file reads replace
per-byte construction; full hashing, canonical paths, physical UIDs and generated
file checks remain, with no metadata-only shortcut or persistent content cache.
The complete build artifacts join the existing atomic output transaction. A
compiler failure keeps private diagnostic evidence and publishes no completed
executable bundle. This mode compiles and links the target; it does not execute it.

The ordinary translation path retains its existing admission rules. Existing
cooperative-group, runtime and aggregate source boundaries remain independent.

The unchanged original `shfl_scan` target passed its complete O2 build and link
with RDC, exactly one generated device object and zero user device archives.
Its original 120-thread rows block and all four D2H outputs passed the external
4,212,992-value exact-integer oracle on Ascend950DT with CANN 9.1.0. A separate
raw archive-input refusal returned before conversion and preserved the existing
Gold publication. The same translator passed 307 software-regression cases, 30 additional checks,
two literal/macro barrier cases and the finite source/argument and UID-integrity
checks. One host-unit link initially failed because its output path was also a
log directory; a link-only continuation preserved the compiled object and ran
the two previously unexecuted UID tests.
