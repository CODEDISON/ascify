# Ascify documentation

[Project overview](../README.md) · [中文项目介绍](../README.zh-CN.md)

## Get started

Follow the [English user guide](user-guide.en.md) or [中文使用手册](user-guide.zh-CN.md)
to prepare LLVM and CUDA, build Ascify, convert the FP32 example, and inspect its
output and JSON receipt.

## Build and use Ascify

| Task | Guide |
|---|---|
| Look up CLI options and source-conversion contracts | [Conversion reference](conversion-reference.md) |
| Convert and publish local header dependencies | [Local-header closure](local-header-closure.md) |
| Build and validate the row-wise SIMD+SIMT path | [Row-wise conversion](rowwise-simd-conversion.md) |
| Inspect measured coverage and performance | [Validation matrix](validation-matrix.md) |

## Contribute and release

| Task | Guide |
|---|---|
| Propose a fix, run checks, and prepare a pull request | [Contributing](../CONTRIBUTING.md) |
| Understand build, install, and validation requirements | [Release process](release-process.md) |
| Understand a semantic or ABI decision | [Architecture decisions](decisions/) |
| Understand product scope and repository checks | [CUDA-to-Ascend product contract](decisions/0029-cuda-to-ascend-product-contract.md) |
| Read source and dependency notices | [Third-party notices](../THIRD_PARTY_NOTICES.md) |

## Validation records

The [validation matrix](validation-matrix.md) binds results to tested commits,
input populations, platforms, and evidence identities. It separates source
conversion from target compilation, execution, correctness, and performance.
Historical results do not establish acceptance for a newer binary.

The [Softmax/RMSNorm tuning record](softmax-rmsnorm-950pr-tuning-report.md)
preserves experiment context. Generalization experiments have their own pinned
inputs and denominators; reusable implementation changes return to this repository.
