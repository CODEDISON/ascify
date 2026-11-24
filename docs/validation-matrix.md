# Target validation

## CUDA Samples on 950DT

On 24 November 2026 (UTC), the same installed translator generated target
source for 38 unmodified CUDA Samples v13.0 vector examples. CANN 9.1.0
built 31 executables. Of those, 29 met their recorded criteria on four
consecutive runs. The sample names, build adaptations and failures are
listed in the [English README](../README.md#cuda-samples-on-ascend950dt)
and [Chinese README](../README.zh-CN.md#ascend950dt-上的-cuda-samples-支持情况).

The CUDA Samples source revision was `3f1c5096`. The installed translator
had SHA-256 `b24012d79874b79bd3d15821ffac0139ad6fd65a09e9dccc2fe73571dd91d0de`.
The earlier 38-sample campaign accumulated results across multiple
translator revisions. Its executables and their numerical records describe
those builds. For example, the archived `cdpSimpleQuicksort` executable passed
on the same machine where the newly built program reported an out-of-order pair.

## OneFlow FP16 Hybrid on 950PR

### Translation

The test uses the [pinned OneFlow fixtures](../tests/fixtures/oneflow/README.md)
and the [FP16 parsing profile](../tests/softmax_rmsnorm_950/scripts/README.fp16_input_profile.md).
The RMSNorm fixture includes a tail-row fix. Its affine case also uses the
repository's store adapter. The production path uses fast math and the
`dav-3510-rowwise-simd-v1` recipe. For the three operator headers, the
[row-wise generator](../tools/generate_rowwise_cce.py) ran with
`--mode=simd-simt --require-recipe` and checked the dispatch counts below.

| Input | November rebuild of public revision `610b517` | November translator corresponding to `e199603`, same input and flags |
|---|---|---|
| Softmax | 3 Hybrid dispatch sites | Translation stopped at a dependent occupancy query (fixture line 551). |
| LayerNorm dependency | 1 Hybrid dispatch site | Translation stopped at a dependent occupancy query (line 106). |
| RMSNorm | 3 Hybrid dispatch sites | Translation stopped because the kernel thread bound was unproved (lines 341, 764 and 885). |
| RMSNorm affine store adapter | Translated, 0 dispatch sites | Generated output matched the historical SHA-256. |

Each prepared input matched its September SHA-256. Rebuilding `610b517`
from source on 950DT reproduced all four historical generated files
byte for byte. Those hashes also matched the headers used for the fresh
950PR build. Running the newer translator with the older Clang resource
headers produced the same three occupancy errors. The newer translator
did not reach a runnable Hybrid target for either benchmark. See the
[September translator](https://github.com/CODEDISON/ascify/tree/610b5172616772ea37fcdfa43e9e0a00e611740a) and
[November translator](https://github.com/CODEDISON/ascify/tree/e199603dbf0c542cbc7a1442659897d25cd16929).

<details>
<summary>SHA-256 of the regenerated Hybrid headers</summary>

| Generated header | SHA-256 |
|---|---|
| `softmax.cuh` | `a949f2ac801d6575985cdf13c86fd970c0e76fad0e4e89c2d463022af8ed5f2c` |
| `layer_norm.cuh` | `e6b472540bb6f905284ac673eb62bcb40bb7f34d331762df5204dd5a17bdc7fb` |
| `rms_norm.cuh` | `b50b1989fbbb5d54b59a5dbd2cbf480fa6e1d16088cd68c15c63d78310f445d4` |
| `rmsnorm_affine_store.cuh` | `26d5d0adceb30270f438d87b5572cc958498959eef3862c775dfe5e9c4938789` |

</details>

### 950PR measurement protocol

The original November run rebuilt the four Hybrid runtime libraries against CANN
`9.1.0-beta.3`. All four matched their September SHA-256 hashes. Both dates
used a 950PR on logical device 2. The PCI address differed between dates,
and the driver changed from `25.7.rc1.6` to `25.7.0`.

Each full Softmax timing run uses the same 1,000 case indices in their frozen order,
five warm-ups per case, 20 ACL event timings and one dispatch per timing.
The A800 reference is the frozen per-case median from an A800-SXM4-80GB with
CUDA 12.9. Each index matches the operation, matrix dimensions and data
volume on the two devices. The score for a case is
`100 × A800 median latency / 950PR median latency`. A score of 100% means
equal median latency. The overall score is the arithmetic mean of the 1,000
case scores, with repeated shapes counted at their original indices. Case
scores above 100% contribute their full value. Accuracy uses a separate
input list.

For Softmax, a case is a contiguous FP16 matrix. Each row is normalized
independently across its columns. The timing list contains 627 distinct
geometries. The four workload labels describe synthetic operator inputs.
For example, `2,688,000 × 1,500` represents 2,688,000 independent Softmax
reductions over 1,500 values each. Inferring a model's batch size, head
count or sequence length from these dimensions requires a model configuration.
The reported latencies cover the operator dispatch.

### November reproduction of the September runtime

| FP16 benchmark | September | November run 1 | November run 2 | Independent accuracy cases passed |
|---|---:|---:|---:|---:|
| Softmax | 92.17% | 44.40% | 45.13% | 1,000 / 1,000 |
| RMSNorm plain | 82.45% | 81.53% | — | 500 / 500 |
| RMSNorm affine | 82.36% | 80.86% | — | 500 / 500 |

Both RMSNorm variants cleared the original 80% mean-performance threshold.
Softmax missed that threshold on two full runs of the same rebuilt
executable. The median November-to-September 950PR latency ratio across the
1,000 indices was 2.26 in run 1 and 2.22 in run 2. Maximum absolute errors
in the original November accuracy runs were 0.00024418 for Softmax and
0.001953125 for RMSNorm. There were zero nonfinite values and zero guard or
canary mismatches. The separate September suite of 66 large RMSNorm accuracy
cases was outside the scope of this reproduction.

### Softmax performance by workload

The original runtime gave the following results at the two test dates.

| Workload | Cases | September | November 1 | November 2 |
|---|---:|---:|---:|---:|
| `attn_prefill` | 450 | 48.10% | 42.60% | 43.96% |
| `attn_decode` | 200 | 111.22% | 41.65% | 42.55% |
| `lm_head_vocab` | 250 | 168.51% | 58.57% | 58.36% |
| `stress` | 100 | 61.57% | 22.53% | 22.52% |
| **All cases** | **1,000** | **92.17%** | **44.40%** | **45.13%** |

In the first November run, vocabulary Softmax cases contributed 27.49
percentage points of the 47.78-point fall from September. Attention decode
contributed 13.91 points. Each contribution is the group's decline in
score multiplied by its fraction of the 1,000 cases. The two groups account
for about 87% of the decline.

Latency also depended on where a shape appeared in the run. In the first
November run, the two `6144 × 512` cases at indices 200 and 517 took
0.0763 ms and 0.2443 ms. The corresponding September medians were 0.0759 ms
and 0.0758 ms. In a separate 11-case check, the archived and freshly built
executables both ran case 900 in about 6.85 ms, close to September's
6.82 ms. The full November run took 18.55 ms for that case.

An independent short run of the original runtime reproduced three September
decode latencies on the November board. The `3072 × 4000` case took 0.4254 ms
in the short run and 0.4246 ms in September. The second
full November run took 1.504 ms for that case. The other two decode cases,
`640 × 1500` and `384 × 1000`, also came within 3% of September in the short
run. The one-second driver samples during the short run recorded no thermal
or software electrical-design-point frequency limitation.

### November comparison of the original and updated builds

The three full runs below use the same generated OneFlow header from
`610b517`, production-fast math, case order and A800 reference. The
updated executables were compiled against the complete public headers of
revision `392330b` and linked to its Softmax runtime. The three other Hybrid
libraries matched their original hashes. A separate transitional build
combined the updated row-wise headers with the historical general CUDA
compatibility header and scored 54.97% across the full list.

| Workload | Cases | Original control | Updated 1 | Updated 2 |
|---|---:|---:|---:|---:|
| `attn_prefill` | 450 | 40.97% | 46.72% | 43.48% |
| `attn_decode` | 200 | 95.54% | 43.47% | 46.90% |
| `lm_head_vocab` | 250 | 65.15% | 63.87% | 66.52% |
| `stress` | 100 | 22.14% | 21.77% | 21.25% |
| **All cases** | **1,000** | **56.04%** | **47.86%** | **47.70%** |

The updated runs finished 8.18 and 8.34 percentage points below the
contemporaneous original control, and both missed the 80% target. The
200 decode cases contributed declines of 10.41 and 9.73 points on their own.
Each large prefill case still has the same weight as one small decode case.

During these three full runs, the board reached 99 °C. The driver's thermal
frequency-limit bit was set in 523 of 900 one-second samples for the original
control, 435 of 559 for updated run 1, and 539 of 605 for updated run 2.
The monitor recorded 400 MHz at high utilization. Every one-second sample
during the vocabulary group recorded the thermal bit in all three runs. The decode groups
completed in a few seconds, too quickly to assign a throttle reason to each
case. The September run used a different driver and PCI address and has no
matching reason trace. These measurements establish thermal throttling in
the November runs. Its exact contribution to the September-to-November score
change is unknown. See the [950 driver frequency-limit definitions](https://github.com/hicann/driver/blob/999c25d7406900c7c55e747bbc59a8817f8d5650/docs/zh/Ascend_950_DCMI_API/Ascend_950_DCMI_API/dcmiv2_get_device_info.md).

| ACL event measurement | Original control | Updated 1 | Updated 2 |
|---|---:|---:|---:|
| All 20,000 timed dispatches, total seconds | 567.56 | 316.01 | 346.30 |
| Prefill, index 214, `2,688,000 × 1,500`, median ms | 215.90 | 88.70 | 88.74 |
| Decode, index 517, `6,144 × 512`, median ms | 0.07744 | 0.18874 | 0.18790 |

The total is the sum of each case's mean event latency multiplied by its 20
timed dispatches. It excludes allocation, initialization, the validation
launch and warm-ups. The updated runs spent 44.32% and 38.98% less time
inside the timed intervals. Case 214 illustrates how a large matrix can
save substantial time while case 517 loses relative performance in the
equally weighted A800 score.

### Paired timing on the same 950PR

Twelve inputs were selected using only their frozen A800 timings. Each of
the four workloads supplied its 25th, 50th and 75th percentile case. For
each case, the original and updated executables ran twice each in ABBA or
BAAB order, with the first executable alternating between cases. Every
invocation used the original five warm-ups and 20 timed dispatches.

Against the rebuilt September executable, the update was faster on 10 of
the 12 inputs. The geometric mean speedup was **1.56×**. The corresponding
three-case means were 3.75× for prefill, 1.45× for decode, 1.17× for
vocabulary and 0.92× for stress. No other process or thermal-limit bit was
observed on device 2 in the 63 telemetry samples. The full 1,000-case score
above remains the performance acceptance measure. In a supplementary repeat
using the public parent revision `566f96d` as the control, another process
appeared on the device in 31 of 63 samples.

### Updated Softmax runtime

For admitted short rows the runtime batches several rows in on-chip unified
buffer (UB) and reuses FP32 exponentials. At widths of 1,000, 1,500 and 4,000,
the UB row has padding and reductions include only valid columns. For
4,096–32,768 columns, a row remains in UB through normalization. Wider
rows use two global-memory passes. The first combines tile maxima and
exponential sums and the second normalizes the original input. SIMD performs
the row arithmetic and SIMT stages contiguous output values.

Applying the selector to the frozen timing geometries with aligned,
disjoint benchmark buffers gives 295 aligned short-row cases, 78 padded
short-row cases, 274 UB-resident cases and 221 streaming cases. The other
132 retain the translated SIMT path. These counts describe selection from
the recorded dimensions. For an unaligned wide-row stride, the kernel
assigns contiguous groups of rows to the same vector core so that adjacent
cores do not share a 32-byte output block. A misaligned wide-row output base
uses one vector core. A caller with new headers and an older v1 library
uses the generated SIMT path for short rows. The [row-wise guide](rowwise-simd-conversion.md#version-1-runtime-selector-boundary)
lists the selector and library contracts.

In a separate accuracy run on idle 950PR device 1, the shipping client
forwarded one successful call into the shipping Hybrid library for each
of `7 × 1000`, `5 × 4096` and `3 × 65024`. The `9 × 333` case completed
through the generated fallback with zero Hybrid calls. All four passed
the original numerical and guard checks. The shipping library's device
ELF contains both the SIMD vector functions and the SIMT output adapter.
The trace recorded host launch calls. The source shows the in-kernel
sequence from SIMD arithmetic to SIMT output staging.

### Updated Softmax accuracy and limits

| Test suite | Result | What the cases check |
|---|---:|---|
| Frozen correctness list | 1,000 / 1,000 | 100 distinct geometries, FP64 reference on the quantized FP16 inputs. Maximum absolute error 0.00024418, maximum scaled relative error 0.00048821, maximum row-sum error 0.00038719. |
| Large-row-count matrices | 4 / 4 | 32,769 or 32,770 rows at widths 1,024, 1,536, 2,048 and 3,072. |
| Padded short rows | 18 / 18 | Widths 1,000, 1,500 and 4,000, each at 1, 3, 5, 17, 113 and 32,769 rows. |
| Cross-version client and library | 1,000 / 1,000 in each pairing | The updated client with the original v1 Softmax library, and the original client with the updated v1 library, each passed the frozen accuracy list on 950PR. |
| Public selector and boundary cases | 27 / 27 | Includes 18 exact in-place cases checked element by element against the disjoint FP16 output, alongside the FP64 reference and guard checks. |
| Additional boundary stress cases | 23 / 23 in each of four rounds | Exact in-place checks at row and vector-core boundaries, including padded short rows, odd wide rows and two-byte base offsets. |

The frozen list reported zero nonfinite values and zero guard or canary
mismatches. The updated Softmax runtime library used for these measurements
has SHA-256 `3a4b7c68d8f36a38af53d43521a0c4895052c9e7a87dcf4f96669be797f6c05b`.
The [public boundary list](../tests/softmax_rmsnorm_950/shapes/rowwise_simd_v1_correctness.csv)
and [checker](../tests/softmax_rmsnorm_950/softmax_check.cce) give the inputs
and assertions.

The original and updated builds each passed 139 of 140 additional boundary
cases. Both failed the row-sum check on the same `3 × 256000` dominant-input
case. Each row has one logit of +8 and 255,999 logits of −8. Rounding each
ideal output to the nearest FP16 value gives 0.97216796875 for the largest
probability and 1.1920928955078125 × 10⁻⁷ for each other entry. Those FP16
values sum to 1.0026854276657104. Both builds measured a row-sum error of
0.0026854276657104492, above the 0.002 limit. Both passed the element-wise
error and guard checks. This case remains a failure under the published
row-sum criterion and is outside the frozen 1,000-case accuracy list.

The [harness](../tests/softmax_rmsnorm_950/README.md) provides the
check and timing programs. Select the production-fast build and point
`ROWWISE_SIMD_RUNTIME_DIR` to the four runtime libraries. Reproducing
the full percentages requires the frozen benchmark case lists and matched
A800 timings, which are held separately from the public source tree.
