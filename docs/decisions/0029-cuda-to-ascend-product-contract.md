# ADR-0029: Make CUDA-to-Ascend product scope an engineering contract

- Status: Accepted
- Date: 2026-09-13

## Context

Ascify inherited a translator skeleton and later added Ascend-specific conversion,
compatibility, and runtime support. Some inherited command-line options had no
implementation, build settings still referred to another target SDK, and the
version output advertised a target compatibility range without a matching test
record. A checked-in translated example had also drifted from the converter.
These surfaces made it difficult for a new contributor to distinguish supported
product behavior from historical implementation scaffolding.

## Decision

The public product is a CUDA C/C++ to Ascend source translator. Remove unused
backend controls and artifact-generator options, keep actual conversion statistics
and receipts, and identify the translator/LLVM build separately from target
validation. The default build and installed resource lookup must work without
another accelerator SDK. `build.sh --help` and `run.sh --help` describe their
supported settings without requiring a configured toolchain.

Install public compatibility headers for generated source under `include/ascify/`
and `include/acl_cub/`. Install the translator's private Clang parsing headers under
`libexec/ascify/clang/<major>/include`, and discover them relative to the executable
using the configured installation layout. The resource-directory option identifies
the parent of this private `include/` directory. This separates target compilation
with `-I<install-prefix>/include` from the converter's frontend resources and avoids
mixing LLVM parsing headers into the public Ascify header surface.

Keep existing internal `CUDA2DPP` names and `.dpp` suffixes as compatibility names.
Retain upstream copyrights, MIT permission headers, third-party source attribution,
and frozen CUDA fixtures. Branding cleanup is not a license rewrite or permission
to edit byte-bound test inputs. Record source provenance in
[third-party notices](../../THIRD_PARTY_NOTICES.md), including its known limits.

Store the FP32 CUDA example as input and regenerate its output with each tested
candidate. Put conversion outputs in `.work/`. Test the actual installed binary,
its default resource lookup, receipt, and converted source; a host-only fixture
pass does not establish that installation or conversion works.

`tools/check_repository.py --root PATH` enforces the repository contract.
`tests/engineering/test_repository_contract.py` exercises its rejection cases.
`tests/engineering/test_cli_surface.py` verifies the implemented public CLI, and
`tests/engineering/check_install.py` verifies installed conversion. The existing
`tests/run_release_checks.sh` remains the common host and translation gate.

The English and Chinese READMEs are project entry points. Detailed setup stays in
the bilingual user guides, command/semantic details in the conversion reference,
and measured results in the validation matrix. Documentation examples and links
are included in repository checks.

## Alternatives considered

- **Rename every inherited internal symbol and source file.** Rejected because
  it increases review scope and disrupts established paths without improving
  the user's conversion interface.
- **Hide dead options only in documentation.** Rejected because the binary would
  still accept commands that produce no useful result.
- **Treat a host CI pass as release acceptance.** Rejected because it does not
  validate native linking, installation, CUDA parsing, or target execution.

## Consequences

Previously accepted options with no implementation now fail as unknown options.
Existing supported conversion modes, migration receipt contracts, target recipe
names, runtime ABIs, and frozen inputs remain explicit compatibility boundaries.
Build, CLI, and source-translation changes receive native validation. Historical
performance stays attributed to its tested commit until affected device paths
are revalidated; see [ADR-0028](0028-separate-release-engineering-from-validation-claims.md).
