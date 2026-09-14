# Release process

## Stable entry points and repository layout

Use `build.sh` for the native translator, `cmake --install` for installation,
and `tests/run_release_checks.sh` or CTest for the existing release checks.
The [documentation index](README.md) links to setup tutorials, conversion reference,
and target runtime instructions.

| Path | Responsibility |
|---|---|
| `src/` | Clang frontend analysis and rewrite rules |
| `include/ascify/` | Compatibility surface, target recipes, and versioned ABI |
| `frontend_compat/` | Authenticated, versioned parsing profiles |
| `runtime/` | Target-specific runtime implementations |
| `tests/`, `examples/` | Fixed fixtures, test entry points, and small examples |
| `docs/` | User documentation, validation summaries, and decisions |
| `tools/`, `tests/engineering/` | Repository, CLI, and installation checks |
| `build/`, `ascify_install/`, `.work/` | Ignored local build, install, and experiment outputs |

Keep proof-bound source and fixture paths stable. Broad formatting, source
relocation, dependency upgrades, or rewrite changes need their own review and
verification. The current cleanup retains internal `CUDA2DPP` names and `.dpp`
output suffixes while aligning the public product with CUDA-to-Ascend; see
[ADR-0029](decisions/0029-cuda-to-ascend-product-contract.md).

## Validation levels

1. **Repository and host contracts:** Run `python3 tools/check_repository.py --root .`
   and the release suite with `ASCIFY_BINARY` empty. The repository check enforces
   product scope, document links, metadata, and generated-artifact boundaries.
   Host checks exclude real translation and NPU execution. Python 3.9+ and the
   declared host compilers must be available.
2. **Native build, installation, and actual translation:** Build and install the
   candidate, run `tests/engineering/test_cli_surface.py --binary PATH`, and run
   `tests/engineering/check_install.py --binary PATH --cuda-path PATH` through
   Python. The installed check must exercise default resource discovery and
   convert the included FP32 example. Run the full release suite with the binary,
   CUDA parsing root, and Clang resource directory set. Report failed or skipped
   fixtures explicitly. DT is the phase-one integration environment; older 910C
   evidence retains its original identity.
3. **Target compilation and correctness:** Compile generated code on the
   named target, then link and run the required independent oracles. Keep
   compilation and execution results separate.
4. **Performance acceptance:** Bind the exact candidate, generated code,
   runtime, compiler, baseline, case manifest, and timing scope. Use the
   [phase-one criterion](validation-matrix.md) on 950PR. A host CI success
   cannot fill a missing device result.

Documentation-only edits require repository and host checks plus a diff proving
that measured code and inputs have not changed. Build, packaging, and CLI edits
also require their native build, install, or command-line checks. These changes
do not by themselves require new timings.
Changes to shared translation or runtime code require verification of their
affected paths before attributing hardware results to the new candidate.

## Release record

Use an immutable source commit as the release-candidate identifier. When a
product release is published, choose a product version and tag together with
its completed validation record. Runtime ABI versions and parsing-profile
versions are independent; do not bump them merely to name a product release.
`CHANGELOG.md` entries remain Unreleased until publication.

The release record must identify:

- candidate commit, build/toolchain identities, and any dirty-source status;
- target policy, recipe, frontend/input profile, dependency and case manifests;
- each validation level's status, population, platform, and raw evidence hashes;
- unavailable evidence, known limitations, and the relationship to older results;
- the published commit/tag and CI run only once they actually exist.

Keep small sanitized summaries in Git. Preserve raw logs and artifacts in the
experiment evidence store; do not commit machine addresses, credentials, device
locks, generated code, or binary bundles. Retain original tested SHAs when
documenting newer commits. Code equality in selected directories is useful
provenance, but does not prove a rebuilt binary has identical behavior.

## Phase-one closeout and phase-two handoff

Phase one closes the two-operator performance result and the engineering
delivery. The fixed20 boundary table is retained. Full CUDA Samples coverage
belongs to phase two in a separate workspace, with its own pinned baseline,
inputs, exclusions, and stage-by-stage results. Reusable converter fixes still
return to the Ascify repository through reviewed changes.

The `721ca1f` candidate has pending DT integration/target validation. Preserve
that status during repository cleanup. A release containing it cannot inherit
the complete `32b33a3` hardware acceptance solely from the unchanged row-wise
runtime directories.

## Packaging and public metadata

Include the executable, supported compatibility headers, frontend profile, and
matching Clang resources in the installed layout. Public headers for generated
source belong under `include/ascify/` and `include/acl_cub/`; private parsing
resources belong under `libexec/ascify/clang/<major>/include`. Verify that the
installed executable discovers its matching private resource directory without
adding it to the target compiler's public include path. Include `LICENSE` and
`THIRD_PARTY_NOTICES.md` in package documentation and retain individual upstream
headers. A source or binary package must state which LLVM/Clang and target runtime
dependencies it expects; do not imply that the translator bundles CANN.

Before publication, verify that documentation examples use installed paths and
ignored output directories, help lists only implemented options, and build
identity does not claim an untested target compatibility range. Record the exact
CI run and installed-example result for the release candidate. Remote branch
protection and release settings are repository administration state; committed
workflow files alone do not prove those settings are enabled.
