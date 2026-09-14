# Release process

This checklist covers source and binary releases. Development setup and test
commands are in [Contributing](../CONTRIBUTING.md).

## Validate the candidate

Use an immutable source commit and a clean build. Run the checks relevant to
the change and record their results with the release:

| Check | Required scope |
|---|---|
| Repository and host tests | Metadata, documentation links, CLI/build contracts, and host compatibility tests |
| Native build and installation | Compile the translator, test its CLI, install it, and verify resource discovery |
| Source conversion | Run the release suite with an actual translator and CUDA parsing dependencies; convert the installed FP32 example |
| Target correctness | For affected translation/runtime paths, compile and link generated code and check results on the declared target |
| Performance | For performance changes or published claims, compare the exact candidate against a declared baseline and case population |

GitHub Actions runs host contracts and an LLVM 23 native build/install/CLI job.
It does not install CUDA or run source conversion or device tests. Record failed
and skipped checks explicitly. Build success and nonempty generated source do
not establish a working migrated application.

Documentation-only changes need repository checks and a diff confirming that
code and fixtures are unchanged. Re-run relevant tests when commands, test
entry points, or build configuration change.

## Check the package

The installed layout separates public target interfaces from converter resources:

| Path under the installation prefix | Contents |
|---|---|
| `bin/ascify-clang` | Translator executable |
| `include/ascify/`, `include/acl_cub/` | Headers used to compile generated source |
| `libexec/ascify/clang/<major>/include/` | Matching private Clang parsing headers |
| `libexec/ascify/frontend-compat/` | Versioned frontend profiles |
| `share/licenses/ascify/` | License and third-party notices |

Check default resource discovery after installation. Exercise a relocated
prefix when distributing relocatable packages. State external LLVM/Clang shared
library requirements and target dependencies; the translator does not bundle
CANN. Hybrid runtime libraries are built and deployed separately as described
in the [row-wise guide](rowwise-simd-conversion.md).

Retain all source and dependency licenses. Exclude build directories, generated
examples, logs, credentials, machine configuration, and experiment archives.
Verify that help and documentation list only implemented options.

## Publish

Choose a product version and tag the validated commit. Parsing-profile and
runtime ABI versions are independent; change them only when their contracts
change. Move the relevant Unreleased entries in [CHANGELOG.md](../CHANGELOG.md)
under the published version.

Release notes should state user-visible changes, dependency requirements, known
limitations, and the checks completed for that version. Link actual CI runs and
any published benchmark artifacts supporting a claim. Keep raw development
journals and local test records outside the source tree.
