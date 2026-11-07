# Stream priorities

Ascify adapts authenticated host calls to `cudaDeviceGetStreamPriorityRange`
and `cudaStreamCreateWithPriority` on the reviewed legacy SIMT header family
with ACL 1.17.0, as supplied by CANN 9.1.

The range query returns actual native endpoints. Smaller values have higher
priority; the reviewed native range lies within 0 through 7. Either output
pointer may be null. Native errors or an unsupported range leave both caller
outputs unchanged. The adapter does not invent negative CUDA priorities.

Creation accepts `cudaStreamNonBlocking`. The signed requested priority is
clamped to the queried range before one native stream creation call. CUDA flag
1 maps to native flags 0: ACL bit 0 means fast launch, whereas ordinary CANN 9.1
streams already have the nonblocking ordering required here. Unknown flag bits
return invalid parameter; `cudaStreamDefault` returns unsupported because this
entry does not implement CUDA legacy default-stream ordering. Other header or
ACL versions return unsupported.

A successful creation establishes the requested native priority, subject to
clamping. It does not establish preemption, a scheduling deadline, or that a
high-priority workload must finish faster. Timing measurements must be treated
separately from functional correctness.

The frontend requires the exact physical CUDA SDK declarations and call types.
It preserves argument evaluation and the existing source, status-domain, and
header provenance checks.
