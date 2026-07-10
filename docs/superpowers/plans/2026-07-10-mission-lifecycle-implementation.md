# Mission Lifecycle Workflow Implementation Plan

**Goal:** Compose launch, orbital operations, digital-twin screening, deorbit, and reentry into one
suite-owned, verifiable mission lifecycle product.

**Architecture:** Add an `astro_mission` orchestration package that calls existing suite product
boundaries rather than duplicating their physics. A lifecycle scenario references launch, twin, and
reentry scenarios plus typed orbit/deorbit phase configuration. The runner enforces insertion,
mass, epoch, state, frame, propellant, and entry-interface gates and returns embedded suite products,
an artifact manifest, continuity report, integrated margin report, and explicit provenance.

**Done condition:** `astro run-mission-lifecycle examples/lifecycle/leo_round_trip.yaml` writes a
single JSON lifecycle result, text summary, and optional phase artifact directory. The checked case
runs launch through ground-reaching reentry with all continuity checks passing, no failed mission
margins, and no native backend objects. Focused and full lint, type, test, CLI, packaging, build, and
artifact-parsing gates pass.

**Mutable files:** New `src/astro_mission`, `tests/astro_mission`, `examples/lifecycle`, and lifecycle
docs; targeted integration edits to `astro_twin.runner`, `astro_cli.main`, package configuration,
README, validation/release/current-state docs, and focused tests.

**Fixed boundaries:** Existing launch, orbit, twin, and reentry commands and schemas remain backward
compatible. The existing low-performance launch examples keep their target-miss behavior. The
lifecycle reference gets a dedicated insertion-qualified local launch fixture. External backends
remain adapters and all public products remain Astro Suite models.

**Validation:** Phase unit tests, fail-closed continuity cases, end-to-end runner and CLI tests,
artifact manifest parsing, public example execution, `ruff`, strict `mypy`, full `pytest`, packaging,
`git diff --check`, and wheel/sdist build.

## Work Packages

- [ ] Add lifecycle phase configuration, continuity, margin, manifest, and result models.
- [ ] Add YAML/JSON IO, artifact-bundle writing, and summary formatting.
- [ ] Add twin trajectory override support without changing existing behavior.
- [ ] Implement launch-to-orbit, operations, deorbit, interface selection, and reentry orchestration.
- [ ] Add integrated continuity and mission-margin assessment.
- [ ] Add `run-mission-lifecycle` CLI workflow.
- [ ] Add insertion-qualified launch, aligned twin/reentry templates, and lifecycle reference case.
- [ ] Add focused tests and public documentation.
- [ ] Run public artifacts and full release gates; review and publish.
