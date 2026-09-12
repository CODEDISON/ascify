# ADR-0025: Guard instantiated helper status domains

- Status: Implemented candidate; real AST planning checked, atomic publication and DT validation pending
- Date: 2026-09-11
- Scope: Dependent error-check calls in a single-type-parameter host function template

## Context

The frozen SimpleTemplates source instantiates its host function for `float`
and `int`. Four `checkCudaErrors` expressions call `cudaMemcpy` or `cudaFree`
with a dependent `T*`. Clang retains unresolved lookup in the template pattern,
so the existing proof, which requires resolved direct calls, retains the entire
helper transaction. The later object compiler first reports `findCudaDevice`
undeclared because that transaction also retained the helper include and calls.

A matching unresolved name is insufficient evidence. A future class type can
introduce a same-name function through argument-dependent lookup (ADL). Seeing
only the current scalar instantiations is also insufficient to promise all
future types. Removing a source branch or forcing a CUDA declaration into lookup
would silently change that program.

## Decision

Keep the existing proof for nondependent calls. A dependent call may instead
use a bounded instantiation proof with an explicit output constraint:

1. The enclosing function must be an ordinary translation-unit host function
   template with exactly one named, non-pack type parameter. Its body braces
   and the parameter name must have direct source locations in the main file.
2. At least one instantiated definition must exist. Every known specialization
   must be an implicit instantiation or explicit instantiation definition for
   exactly `int` or `float`. Uninstantiated templates, other types, explicit
   specializations, and external instantiation declarations are rejected.
3. For every observed specialization and every recorded error-check macro in
   that template, locate exactly one call at the pattern's original source
   location. It must resolve to the frozen helper's `check` declaration, with
   frozen/system redeclaration provenance. Its status argument must pass the
   existing runtime API mapping and system declaration provenance proof.
   A project overload or arbitrary status function keeps the transaction.
4. Insert a `static_assert` at the opening body brace that permits only the
   exact observed type set. For the two scalar instances this is
   `__is_same(T, float) || __is_same(T, int)`. A future class, unobserved scalar,
   or cv-qualified type fails with an explicit Ascify template-domain message.
   A canonical alias of an admitted type remains admitted.
5. Check macro definitions at that exact insertion point for the parameter,
   `static_assert`, `__is_same`, `int`, and `float`. A later `#undef` does not
   excuse a macro that would change the inserted guard's meaning.

Stage the guard together with every helper macro rewrite, device-selection
rewrite, and include replacement. A failed proof, macro audit, overlapping
rewrite, or other helper conflict publishes none of that transaction, including
the guard. Existing staged output publication rules remain in force.

This makes the type restriction part of the generated program rather than
silently deleting unsupported template instances. There is no sample-name
match and no change to the source instantiations, arguments, runtime calls, or
their evaluation count. The guarded domain justifies the previously unresolved
source ADL using the exact instantiated ASTs in that domain.

## Limits

This is not arbitrary dependent-expression support. Function/member templates
with more parameters, type packs, class-valued substitutions, nested lambda
owners, missing instantiation bodies, and unprovable call or source identities
retain the original helper layer. Unsupported future types fail explicitly.
It does not create CUDA SM major/minor properties, add `_ConvertSMVer2Cores`,
resolve helper expansions in external headers, or implement device FP64.

## Verification

`tests/rewrite/check_sample_helper_template_domain.py` first checks the guard
with the host compiler and confirms Clang's unresolved-pattern/resolved-scalar
AST behavior. Without a supplied converter it explicitly reports that native
transformation validation was not run.

With the real converter, CUDA parsing root, and Clang resource root supplied,
the gate converts two positive and eleven negative variants of
`sample_helper_template_domain.cu`. Positive output must commit the helper and
guard together; a separate translation unit includes the untouched generated
source and proves that an additional `float` call compiles while `char` fails
at the guard. Negative variants cover no observed instantiation, an unadmitted
scalar, class ADL, a project overload for an admitted scalar, macros active at
the insertion point and undefined later, macro-provided braces, explicit/extern
specializations, and mixed custom status calls. Rejected transactions must
retain their helper include/calls and contain no output guard. The three
guard-token macro cases also violate the existing frozen compat identifier
surface: they must fail with that exact collision diagnostic and publish no
output, in addition to rejecting the template-domain proof.

The final source must also pass the existing helper gate and the frozen
original CUDA20 conversion/object suite. Host compiler checks alone do not
establish native converter or target object success.

For a host whose system headers hit the existing macro-generated pragma
boundary, `--expect-retained-closure` is a separate **AST-planning-only** mode.
It requires that boundary and no helper/guard publication in all 13 variants
(retained output for ten variants, explicit macro rejection with no output for
three). The two positive variants must report the exact planned type guard;
the eleven negative variants must report no successful template proof. This
mode does not relax pragma provenance, exercise atomic publication, or replace
the default native gate and target validation. The proof diagnostic explicitly
marks a guard as pending the helper transaction; only the existing commit
diagnostic indicates publication.
