# Changelog

User-visible changes to Ascify are recorded here. Unreleased entries describe
development features; consult the [compatibility reference](docs/compatibility.md)
for supported domains and experimental paths.

## Unreleased

### Added

- Optional JSON migration receipts with global and per-input conversion status.
- Recursive local-header conversion with checked dependency identity and
  transactional publication of generated source and headers.
- An opt-in, versioned frontend profile for admitted cooperative-group operations.
- Scoped CUDA Runtime, device, and NVIDIA Samples helper compatibility adapters.
- Explicit row-wise SIMD+SIMT Hybrid conversion for recognized Softmax, RMSNorm,
  and LayerNorm patterns, with versioned runtime libraries and SIMT fallback.
- Repository checks, host tests, and Linux CI for native LLVM builds, installation,
  and the public CLI.
- English and Chinese setup guides, API compatibility documentation, and an FP32
  vector-add example.

### Changed

- CLI help, diagnostics, and dependencies consistently describe CUDA-to-Ascend
  conversion. Inactive legacy switches and unimplemented generators were removed.
- Standalone CMake builds respect the selected compiler and use target-scoped
  settings. Build and run wrappers support help without environment setup.
- Installation separates public compatibility headers from private Clang parsing
  resources and includes dependency license notices.
- Explicit invalid resource paths report an error; installed binaries resolve
  their resources from the installation layout.
- Compatibility rewrites check source semantics, types, and provenance before
  replacing supported patterns; unsupported domains retain explicit boundaries.

### Fixed

- Preserve include replacement text until Clang consumes the edit.
- Reject unknown command-line options beginning with the short help spelling.
- Order owned runtime cleanup after initialization and device binding.
- Propagate selected Hybrid launch failures without launching the SIMT fallback.
- Match Hybrid launch memory capacity to the runtime's shared allocations.
