# Build an owned CUDA translation target

The `--standalone-device-executable` mode builds one new CANN executable from
one physical CUDA root and its recursive header closure. Use it with
`--default-preprocessor`, `--local-headers-recursive`,
`--target-policy=dav-c310-vec`, `--target-recipe=none`, and an output source path.
`--standalone-cann-root` selects the existing CANN installation; its default is
`/usr/local/Ascend/cann-9.1.0`.

The mode preserves source parsing options, including RDC. It owns a fixed
single-device-object build and publishes source, headers, target and build
receipts together. Extra compiler inputs and linker or archive arguments after
`--` are rejected before tooling can discard them. Parent CMake targets and user
device archives lie outside this mode. Generated programs are never run by the
translator.

Within this bound target, a closed large trivial-record helper may be forced
inline after its physical source, layout and complete reference facts match.
A separate nominal-width32 register view supports proved register operations
with a final contiguous partial warp. It retains the original block size and
provides no barrier or reduction API. The separate shared-helper proof retains
its own literal compiler-barrier, storage and caller-control conditions.

Consult the [decision](decisions/004-owned-aggregate-device-target.md) for the
source, macro and publication conditions. The unchanged original CUDA Samples
v13.0 `shfl_scan` target compiled and linked at O2, with RDC and its original
120-thread rows block preserved. One execution on Ascend950DT with CANN 9.1.0
passed the external exact-integer oracle for all 4,212,992 values in its four
D2H outputs. The same translator passed 307 software-regression cases, 30 additional checks,
the finite source/argument refusals and two host UID-integrity units.
Other original-sample acceptances retain their own product and target evidence.
