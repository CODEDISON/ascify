# Restricted scalar64 conversion

This route targets Ascend950DT (`__NPU_ARCH__ == 3510`) with the verified CANN
9.1 legacy SIMT headers and `__builtin_bit_cast`. New binary64 arithmetic and
explicit scalar64 helpers reject unverified targets. Raw wrappers retain the
pre-existing native integer64 paths of other providers; double always enters
the restricted helper and cannot fall through to a native float overload.
It requires `--default-preprocessor`; existing SDK/provider,
macro, physical-source and transactional-publication checks still apply.
Reachable builtin atomic/complex/vector `+` and `+=` with double components
are refused before route activation. These types never enter a scalar helper.
Host functions, copying without a new addition, and other element dtypes retain
their existing behavior.

| Operation | Admitted domain | Contract |
|---|---|---|
| Raw `__shfl_down_sync` / CG tile32 `shfl_down` | Exact builtin `long`, `unsigned long`, `long long`, `unsigned long long` when 64 bits; `double` | Width32. Mask equals the already converged active cohort; no rendezvous is added. Delta >=32 is passed as native delta0 while all lanes still shuffle. |
| Raw `__shfl_sync` / CG tile32 `shfl` | Same builtin64 domain | Width32, full active warp, source lane0..31. |
| Evaluated device builtin `a+b` | `double`; guarded shared `T` in `{int,float,double}` | Integer-only binary64 RNE addition. Float/int retain their builtin operation and source grouping. |
| Discarded statement `a+=b` | Same arithmetic domain | Proved nonvolatile scalar lvalue, effect-free address/RHS. Reference binding evaluates its address once; the old value is read and stored once after RHS evaluation. |
| CG `plus<double>` / `reduce` | Full tile32, or the separately proved uniform scratch groups | Preserve binary64 bits and the stated reduction tree using the integer RNE helper. No conversion through float. |

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
raw SDK shuffle call identities receive dependent-type facts. A separate two-type-parameter exception admits only a pure, one-return forwarding
function whose selected callee and group belong to the exact SDK reduction
projection; a group-domain guard closes future instances. No general
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
expressions. Authenticated CUDA execution-coordinate reads are accepted only
through their real SDK builtin variable, owned property and compiler getter.
An address NTTP requires the actual template parameter/substitution identity and
a direct written integer-literal argument witness for every concrete instance.
Macro arguments, `sizeof`, user constexpr expressions and unknown dependent
forwarding are refused. An explicit-instantiation-only address NTTP without that
written-argument witness remains a conservative refusal, even when its CUDA
source is valid.

The binary64 backend fixes RNE and the SoftFloat3e `8086-SSE` NaN payload policy.
It does not implement exception flags or a floating-point environment. Its
arithmetic body matches the separately checked engineering add closure; see
`THIRD_PARTY_NOTICES.md` for the pinned source and license. Engineering probes
are supporting evidence, not acceptance of these new product dispatchers.

The [uniform reduction contract](compatibility.md#uniform-block-scratch-reduction)
and [decision record](decisions/002-bounded-binary64-cg-reduction.md) describe the
separate scratch and inactive-source proof. Wider ordinary tiles, up/xor
binary64 shuffle, threadFenceReduction's float NTTP/visibility problem and other
arithmetic operations retain their separate boundaries. These source contracts
do not establish full `reduction<double>` or complete CUDA Samples acceptance.

## Validation gates

1. Review the isolated product diff, source fingerprints and test fixtures.
2. Build the actual frontend, run release checks, and use the new syntax-first
   `tests/rewrite/check_scalar64_lowering.py` with the actual binary/CUDA SDK.
   Its original 16 admitted/retained cases execute the generated C++ against host SDK
   doubles, compare raw bits with literal independent expectations, and check
   float/int order and shuffle call counts. Forty-one source-valid refusals plus
   seven target compilation and five host collective refusal cases cover the
   boundary. No host-model result establishes NPU correctness.
   The original `cg_double_plus_exact` refusal is intentionally replaced by a
   positive exact-reduction fixture: five down-shuffles and one broadcast, six
   typed transports in total. Other original57 fixture domains remain intact.
   Fifteen supplementary cases add
   ten wrapped-double/constant-observation refusals and five host/copy/other
   dtype controls, for72 frontend cases (21 admitted/retained,51 refused).
   The constinit case explicitly selects C++20.
   Where present, the selected SDK's `include/cccl` directory is supplied to
   both original CUDA syntax and conversion, with its type_traits header hash
   recorded and checked unchanged. SDK source and syntax requirements remain
   intact.
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
