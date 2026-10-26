# Restricted scalar64 conversion

This route targets Ascend950DT (`__NPU_ARCH__ == 3510`) with the verified CANN
9.1 legacy SIMT headers and `__builtin_bit_cast`. Target compilation rejects
other providers. It requires `--default-preprocessor`; existing SDK/provider,
macro, physical-source and transactional-publication checks still apply.

| Operation | Admitted domain | Contract |
|---|---|---|
| Raw `__shfl_down_sync` / CG tile32 `shfl_down` | Exact builtin `long`, `unsigned long`, `long long`, `unsigned long long` when 64 bits; `double` | Width32. Mask equals the already converged active cohort; no rendezvous is added. Delta >=32 is passed as native delta0 while all lanes still shuffle. |
| Raw `__shfl_sync` / CG tile32 `shfl` | Same builtin64 domain | Width32, full active warp, source lane0..31. |
| Evaluated device builtin `a+b` | `double`; guarded shared `T` in `{int,float,double}` | Integer-only binary64 RNE addition. Float/int retain their builtin operation and source grouping. |
| Discarded statement `a+=b` | Same arithmetic domain | Proved nonvolatile scalar lvalue, effect-free address/RHS. Reference binding evaluates its address once; the old value is read and stored once after RHS evaluation. |

`double` values remain eight bytes in typed pointers, local values, helper
arguments and results. New binary64 arithmetic never converts through float.
A participating `__global__` scalar by-value double parameter is refused,
including when `--no-lower-device-double-params` is present. The unproved scalar
kernel ABI cannot overlap the older double-to-float parameter rule.

Closed template proof requires a single named type parameter, optional fixed
builtin integral NTTPs, builtin/pointer/reference signatures, known concrete
instances and no address escape, competing lookup or explicit specialization.
An injected `static_assert` closes the future builtin dtype domain. Only the
exact guarded `T`-derived types, outer declaration signature TypeLoc and proved
raw SDK shuffle call identities receive dependent-type facts. No general
function-type, dependent-member, void-NTTP or SDK-record exception is added.

The first version refuses class/enum/ADL conversions, volatile/atomic values,
side-effectful addresses/RHS, value-used `+=`, macro operators/operands,
preprocessor alternatives inside a participating body, constexpr/consteval,
dual host/device or exception-specified functions, and operators observed in
sizeof/decltype/noexcept/traits/static assertions/constant initializers.
Runtime binary64 conversions, comparisons, negation, subtraction, multiply and
divide in participating arithmetic functions remain unsupported. Plain local
and parameter references, pointer dereferences and indexed arrays are the
initial lvalue domain; array indices use effect-free local/parameter integer
expressions.

The binary64 backend fixes RNE and the SoftFloat3e `8086-SSE` NaN payload policy.
It does not implement exception flags or a floating-point environment. Its
arithmetic body matches the separately checked engineering add closure; see
`THIRD_PARTY_NOTICES.md` for the pinned source and license. Engineering probes
are supporting evidence, not acceptance of these new product dispatchers.

CG `plus/reduce`, uniform scratch, wider tiles, up/xor binary64 shuffle,
threadFenceReduction's float NTTP/visibility problem and other arithmetic
operations retain their earlier boundaries. This change does not claim full
reduction<double> or complete CUDA Samples support.

## Validation gates

1. Review the isolated product diff, source fingerprints and test fixtures.
2. Build the actual frontend, run release checks, and use the new syntax-first
   `tests/rewrite/check_scalar64_lowering.py` with the actual binary/CUDA SDK.
   Its 16 admitted/retained cases execute the generated C++ against host SDK
   doubles, compare raw bits with literal independent expectations, and check
   float/int order and shuffle call counts. Forty-one source-valid refusals plus
   five target compilation and five host collective refusal cases cover the
   boundary. No host-model result establishes NPU correctness.
3. Compile actual generated payloads with CANN9.1 CCEC. In particular prove the
   typed GM/shared-reference `+=` ABI and inspect the object/assembly: binary64
   addition must use the integer backend, and shuffle must transport uint64.
4. Separately review and serially execute new wrapper probes under the project
   device guardian. Recheck all input/output guard words and original raw-bit
   expectations. Prior384/768-word private probe results do not replace this.
5. Rebuild complete original sample targets and run their unchanged original
   oracle. Remaining CG/scratch/other dtype failures stay failures.

The implementation and regressions were prepared for review before compilation;
the review bundle records what was actually run and which gates remain pending.
