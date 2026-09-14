---
name: Bug report
about: Report a reproducible build, conversion, or target execution problem
title: ""
labels: ""
assignees: ""
---

## Failure stage

Identify the stage: host build, source conversion, target compilation, link,
device correctness, or performance comparison.

## Reproduction

Provide the smallest input and exact command that reproduces the problem,
including the real exit code and relevant diagnostic. Describe the expected and
observed behavior.

## Environment and identity

- Ascify source commit and binary SHA-256:
- Host OS, architecture, and LLVM/Clang version:
- CUDA parsing toolkit and frontend compatibility profile:
- Target policy, recipe, and math mode:
- Target hardware and CANN/compiler version, if applicable:

## Evidence

Link a small log or evidence summary. For a performance report, include the input
list identity, reference platform, timing protocol, and aggregation formula.
Remove credentials and private connection details before posting.
