# Build an owned CUDA translation target

The implementation candidate adds `--standalone-device-executable` to build one
new CANN executable from one physical CUDA root and its recursive header closure.
Use it with `--default-preprocessor`, `--local-headers-recursive`,
`--target-policy=dav-c310-vec`, `--target-recipe=none`, and an output source path.
`--standalone-cann-root` selects the existing CANN installation; its default is
`/usr/local/Ascend/cann-9.1.0`.

The mode preserves source parsing options, including RDC. It owns a fixed
single-device-object build and publishes source, headers, target and build
receipts together. Parent CMake targets and user device archives lie outside
this mode. Generated programs are never run by the translator.

Within this bound target, a closed large trivial-record helper may be forced
inline after its physical source, layout and complete reference facts match.
A separate nominal-width32 register view supports proved register operations
with a final contiguous partial warp. It retains the original block size and
provides no barrier or reduction API.

Consult the [decision](decisions/004-owned-aggregate-device-target.md) for the
source, macro and publication conditions. Compilation and numerical validation
of the current implementation candidate are pending. Existing original-sample
acceptances keep their own product and target evidence.
