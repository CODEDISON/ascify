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
