# Separate release engineering from validation claims

Status: accepted for local phase-one engineering; publication pending.

Date: 2026-09-13

## Context

Phase-one Softmax and RMSNorm acceptance, full CUDA Samples exploration, and
newer fixed20 compatibility work have different tested commits and populations.
The repository already has build/install entry points, host and translator
checks, ABI versions, and semantic decisions, but lacks automated host CI and
a central account of those validation boundaries.

## Decision

Reuse the existing build, test, and directory structure. Add host-only CI,
contribution templates, documentation navigation, and an immutable baseline
summary. Keep hardware evidence attached to its tested commit. Record newer
candidate validation as pending where it has not run.

Phase two generalization gets a separate experiment workspace and frozen
source-file/project inventories. Source generation, object compilation,
linking, execution, and correctness are distinct outcomes. Engineering cleanup
alone does not update any conversion or performance percentage.

## Consequences

Contributors can run the same host checks locally and in CI without NPU access.
Maintainers can distinguish a passing CI job from a validated target release.
Existing source/fixture identities remain intact. A larger source reorganization
or new universal-support claim requires a separate change and evidence.
