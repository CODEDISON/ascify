# ADR-0020: Replace only the CUDA Samples helper layer

## Status

Accepted

## Date

2026-08-25

## Context

The proof-gated CUDA Samples transaction used to erase one exact
`helper_cuda.h` directive after rewriting admitted helper calls. That header
also directly includes portable standard headers and exact
`helper_string.h`. Removing the directive can therefore remove declarations,
types, string macros, and inline helpers even when no AST reference points to
their originating include. Examples include `uint32_t`, `std::string`,
`strlen`, and `STRCASECMP`.

The replacement may also need `ascify_cuda_compat.hpp`. That header introduces
the reserved `ASCIFY_*` output macro surface and namespace `ascify`. A
same-spelling counterfeit header, a pre-existing reserved macro, a top-level
declaration named `ascify`, a macro-stack restoration, or a retained header
that observes a newly published macro can make the generated source differ or
fail to compile. A path basename or a successful original parse is not proof
that the generated include is safe.

Finally, an uninstantiated or implicitly instantiated dependent call in a
retained header can refer to frozen `findCudaDevice` through an
`UnresolvedLookupExpr`. Auditing only resolved `DeclRefExpr` nodes can remove
the declaration while leaving that dependent lookup behind.

## Decision

At the original direct include location, replace exact `helper_cuda.h` with
its portable direct include surface in original order:

```cpp
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_string.h>
```

Append the exact Ascify compatibility header at that location only when an
admitted helper rewrite or another CUDA mapping needs it. This preserves
transitive portable declarations and macro state while replacing only the
CUDA-specific helper implementation. A later exact compat include does not
prove declarations were visible at the rewritten call, so the transaction
still inserts the earlier guarded include.

The compatibility preflight is independent of the SIMD recipe:

- the parsed compat buffer must be the frozen 43,500-byte file with SHA-256
  `c1f87bd416aa4f389171593bd5e47894999d05407e5935d3151fbcd2aa6438c9`;
  an overridden, remapped, or counterfeit same-spelling header is rejected;
- input definitions of the compat guard, control/output macros, internal
  define/undef macros, backing helper names, `ascify`, or
  `sampleFindCudaDevice` invalidate helper publication; if an independent raw
  CUDA mapping already requires compat, translation fails instead of
  publishing a known-broken output;
- a top-level input declaration named `ascify`, including an `extern "C"`
  declaration, is rejected whenever a new compat namespace must be inserted;
- main and retained included files are raw-lexed with Clang spelling cleanup
  for direct observations of published compat macro tokens; and
- `UnresolvedLookupExpr` declaration sets are audited with the same frozen
  helper-declaration rule as resolved references.

`ASCIFY_*` remains a reserved generated-output namespace. Inputs that synthesize
those names indirectly with token pasting are outside the accepted source
contract; direct definitions, observations, and macro-stack operations are
machine-rejected.

All helper call rewrites, the portable-surface replacement, and any compat
include remain one staged transaction. No helper edit is committed until the
preprocessor, AST, raw-token, collision, and range proofs complete.

## Alternatives Considered

### Copy individual declarations inferred from the AST

This misses macro state, uninstantiated templates, overload lookup sets, and
future uses of the portable inline helper surface.

### Keep erasing the full helper include and enumerate more exceptions

That treats portable transitive dependencies as accidental and grows an
open-ended negative list. Preserving the exact direct portable surface is a
smaller counterfactual change.

### Trust the compat include spelling or filesystem path

The preprocessor may parse a remapped buffer for the same physical path, and a
project may provide a same-spelling header. Only the actual parsed frozen
buffer is an admitted identity.

## Consequences

- CUDA Samples that use portable helper_string or standard-header surface can
  keep compiling after the CUDA-specific helper layer is replaced.
- Dependent external template lookups and compat output collisions now fail
  closed instead of producing a latent or immediate generated-source failure.
- The contract is intentionally conservative for pragmas, retained published
  macro tokens, and the reserved `ASCIFY_*` namespace.
- These proofs establish source transformation and target-build eligibility;
  they do not establish device execution, correctness, or performance.

## Phase 1 integration, 2026-09-09

This helper-only change is integrated on the `ac3dced8` product baseline.
The reviewed implementation and fixtures come from the `df5c7ee` helper
candidate lineage; its unrelated command-line, main-program, migration-receipt,
and documentation removals are not imported. The runtime compatibility header
is unchanged and matches the identity specified above.

The first validation scope is the original presentation's 20 CUDA Samples
translation units at `b7c5481c`; repository-wide coverage belongs to phase 2.
The reverse direct-header graph occurs in ScalarProd and BlackScholes. A
successful helper transaction still does not admit missing device-property
APIs, Driver/VMM/compression behavior, constant-symbol upload, or runtime helper
calls whose use is in a separately translated external header. In particular,
this change does not promise full SimpleTemplates or fastWalshTransform
conversion/compilation.

Local host-only release checks passed on this baseline. Translation fixtures
and the original 20 conversion/compilation regressions require the freshly
built DT binary; PR is reserved for final validation. No new device execution
or performance result is claimed by this integration.
