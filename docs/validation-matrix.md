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

### 950PR benchmark results

The November run rebuilt the four Hybrid runtime libraries against CANN
`9.1.0-beta.3`. All four libraries matched their September SHA-256 hashes.
September and November used a 950PR, logical device 2. The PCI address differed
between the two dates, and the driver changed from `25.7.rc1.6` to `25.7.0`.

Each group has 1,000 indexed timing cases. For each case, the November run
used five warm-ups, 20 timed samples and one inner repetition. The score
is the arithmetic mean of the individual ratios
`100 × A800 median latency / 950PR median latency`. The A800 medians came
from September measurements on an A800-SXM4-80GB with CUDA 12.9. The two
devices used matching geometries and data volumes. Correctness was checked
on separate inputs.

| FP16 benchmark | September | November rebuild of `610b517` | Additional full timing run | Independent correctness cases passed |
|---|---:|---:|---:|---:|
| Softmax | 92.17% | 44.40% | 45.13% | 1,000 / 1,000 |
| RMSNorm plain | 82.45% | 81.53% | — | 500 / 500 |
| RMSNorm affine | 82.36% | 80.86% | — | 500 / 500 |

Both RMSNorm variants cleared the original 80% mean-performance threshold.
Softmax missed that threshold on two independent full runs of all 1,000
cases with the same rebuilt executable, device, input order and timing
settings. The median of the 1,000 individual November-to-September 950PR
latency ratios was 2.26 for the first run and 2.22 for the second.

The first full run also slowed down for a repeated shape. The two
`6144 × 512` cases at indices 200 and 517 took 0.0763 ms and 0.2443 ms.
In September, the same two cases took 0.0759 ms and 0.0758 ms. In a
separate 11-case check, both the archived and freshly built executables
ran case 900 in about 6.85 ms, close to September's 6.82 ms. The first
full November run took 18.55 ms for that case. The cause is unresolved.

The Softmax timing list contained 627 distinct geometries. Its separate
correctness list covered 100 distinct geometries across 1,000 cases. The
maximum absolute error was approximately 0.00024418 for Softmax and
0.001953125 for RMSNorm. Both correctness suites had zero nonfinite
values and zero guard or canary mismatches. The separate September suite
of 66 large RMSNorm correctness cases was outside the scope of this run.

The [row-wise guide](rowwise-simd-conversion.md) documents how to generate
the headers and build the four runtime libraries. The [harness](../tests/softmax_rmsnorm_950/README.md)
provides the check and timing programs. Select the production-fast build and
set `ROWWISE_SIMD_RUNTIME_DIR` to the directory containing all four libraries.
Reproducing the percentages also requires
the frozen benchmark case lists and matched A800 timings, which are held
separately from the public source tree.
