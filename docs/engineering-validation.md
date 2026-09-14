# Engineering verification

This local record covers the engineering changes in
`593ed58106fc149a842dc105f5585574235780da`, based on the pre-normalization
candidate `721ca1fa45568337b4865ce62f331ec45eeaae60`.
Verification ran on macOS arm64 with LLVM/Clang 23.1.0 on 2026-09-13.
It does not establish a new CUDA Samples conversion rate or device acceptance.

## Completed checks

| Check | Result and scope |
|---|---|
| Standard CMake Release build | All 18 translation units compiled and linked |
| CTest | 2/2 suites passed: host release contracts and actual CLI checks |
| Build entry points and resource selection | 10 tests passed, including paths with spaces, compiler precedence, invalid resources, and directory impersonation |
| Repository contract | Source/product scope, local documentation links, required metadata, artifacts, and symlink boundaries checked |
| Installation | Public compatibility headers, private parsing resources, admitted frontend profile, and third-party licenses installed |
| Installed frontend probes | 7 checks passed: angled/quoted header conversion, resource/profile relocation, explicit invalid/empty resources, missing resources, and missing profile |

The final built and installed binary share SHA-256:

```text
a622bc5c262e02860873b2498120516cd691a7a4016f097f057a8b2772b006b1
```

The 56 recorded build input hashes remained unchanged through build, CTest, and
installation. The host CTest run also received deliberately invalid inherited
`ASCIFY_*` paths; explicit CMake test settings correctly isolated it from that
stale environment. CLI checks reject all 12 removed controls and invalid help
prefixes, while preserving standard help and both version spellings.

Installed frontend probes use a small CUDA kernel definition and runtime memory
calls, with successful global and per-input migration receipts. They preserve
the input header's delimiter style, use the installed profile, and retain prior
output on rejected configurations. Relocation includes a prefix containing
spaces. External shared-library dependencies remain required; this is a test of
resource/profile discovery, not a standalone binary distribution claim.

## Open validation gates

The full FP32 vector-add installation check returned nonzero on this host:
the CUDA launch expression reports `cudaConfigureCall` as undeclared. The parsing
environment uses CUDA 12.9 headers/libdevice with the repository's recorded
parser stubs; it is not an official macOS CUDA Toolkit. This reproduces the
previous local launch-parsing boundary and does not establish a DT regression.
The complete example remains a required check on the supported integration
environment.

The full release suite with the final binary and CUDA parsing root was also
attempted. Its host checks passed, then the first real sample-helper closure
check failed because the expected `#include <stdint.h>` was absent from the
published output. Later translated fixtures were not reached. This is retained
as a failed gate, consistent with the previously open local helper-publication
boundary; it is not counted as a full release-suite pass.

The GitHub workflow contains separate Ubuntu host-contract and LLVM native
build/install/CLI jobs. Its native job does not install CUDA or run source
conversion. Inspect the pull request's actual check runs for cloud CI status;
the local results above do not substitute for them.

No new target compilation, linking, NPU execution, or performance measurements
were run. The pre-existing candidate's DT integration and complete translated
fixture gates remain open. Historical Softmax/RMSNorm acceptance and CUDA
Samples counts retain their original commits in the
[validation matrix](validation-matrix.md).

## Preserved boundaries

Public compatibility headers, target runtimes, admitted profile contents, and
frozen operator fixtures are unchanged from `721ca1f`. Shared frontend and CLI
code changed, so equality of those preserved directories does not transfer
older hardware acceptance to this candidate.

The audit also identified existing numeric mapping debt: the double
`CUDART_INF`/`CUDART_NAN` entries place `UNSUPPORTED` in the API-section field,
and `CUDART_NAN` names an infinity replacement. This cleanup leaves that table
unchanged. Fixing it requires separate semantic validation, including the
phase-one infinity-family proof; device FP64 remains unsupported.

Use the [release process](release-process.md) to bind subsequent integration
results to an exact source and toolchain. Raw logs, build input hashes, failed
attempts, and installation probe receipts are retained in evidence collection
`ascify_engineering_20260913`.
