## Change

Describe the problem, resulting behavior, and affected conversion or runtime contracts.

## Validation

Record commands and outcomes. Identify checks that were not run and why.

- Host-only checks:
- Actual source conversion, if relevant:
- Target compilation, linking, or device checks, if relevant:

The host CI job covers CPU fixtures and static contracts; the native job builds
and installs the translator and checks its CLI. Record actual CUDA conversion,
device correctness, and performance evidence separately when needed.

## Evidence and compatibility

For changes to a measured path, identify the source commit, input/profile hashes,
target/toolchain, recipe, and evidence reference. State whether an existing result
still applies or requires a new validation batch. Link the relevant ADR or support
matrix update when a supported domain or public contract changes.
