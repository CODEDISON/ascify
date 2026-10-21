# ADR-001: Preserve requested allocation bounds for texture views

Status: Proposed; implementation awaits independent review and validation.
Date: 2026-10-21

The original stereoDisparity project allocates its packed images with ordinary
cudaMalloc before creating Pitch2D textures. The existing adapter only knows
storage created by its pitched and array allocation APIs. A caller-provided
pitch cannot prove either the size or ownership of an unknown pointer.

Record successful ordinary allocations in a fixed host registry: exact base,
requested byte count, current device and current context. Readiness has already
established the device; capture the context from the existing ACL query. A
failed query or a full 128-entry table leaves that allocation untracked, without
changing the underlying allocation return status. Untracked storage cannot
back a new texture. This does not admit arbitrary ACL, host or interior pointers.

Texture creation checks four-byte alignment and pitch, checked last-row span,
channel/filter restrictions, and matching current ownership. Every descriptor
keeps its own width, height and pitch, allowing different valid views over the
same allocation. Its host texture record remains pinned by live descriptors
after a successful free; address reuse creates a new record. A freed view cannot
silently become a view of a later allocation at the same address.

Use the allocation-registry lock before the texture-registry lock. Creation
holds the allocation lock through descriptor publication. Ordinary free holds
it while the existing callback invalidates all descriptors and frees storage,
then removes provenance only on success. A failed free retains the owner;
existing conservative descriptor invalidation behavior is preserved. Successful
device reset clears both registries while holding the allocation lock. Texture
callbacks never reacquire that lock. This orders the supported host operations;
it does not authorize foreign raw frees or resets, concurrent caller mutation
of resource descriptors, or unsynchronized device use after free.

Alternatives considered: a runtime address-range query may report rounded
allocator bounds rather than the original requested extent, and cannot alone
bind existing descriptors to an allocation lifetime. Registering a callback
only on first texture creation loses allocations already made. Inferring size
from width or pitch would allow an invalid resource to establish its own proof.

The registry adds bounded host state and an optional context query to successful
ordinary allocations. Ordinary allocation capacity is unchanged; only texture
admission has the 128-owner bound. Descriptor ABI stays unchanged. Host models
must cover capacity, query/allocation/free/reset failure, wrong ownership,
multiple views, address reuse and create/free ordering. Real CCEC/ACL execution
and the unchanged complete stereo sample remain separate acceptance gates.
