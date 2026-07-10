# Reentry Suite Implementation Plan

**Goal:** Add a complete suite-owned reentry modeling and simulation workflow covering ballistic,
prescribed-bank lifting, and target-tracking guided entry.

**Architecture:** Add an `astro_reentry` package with typed Pydantic scenarios and results, a
deterministic spherical-Earth 3-DOF local propagator, atmospheric and Sutton-Graves-style heating
models, guidance laws, requirement-margin assessment, local guidance optimization, orbit-state
handoff, backend dispatch, and CLI/IO surfaces. Public products remain Astro Suite models with
explicit model provenance. External high-fidelity engines can be added behind the dispatch boundary
without changing the public contract.

**Done condition:** Ballistic and lifting examples run through the public CLI; guided entry targets
a declared landing site; results include kinematics, loads, aerothermal history, integrated heat
load, events, peaks, target miss, and margins; local optimization emits a tuned scenario and result;
orbit-state handoff is covered; focused and full tests, lint, strict typing, package build, and public
CLI checks pass.

**Mutable files:** New `src/astro_reentry`, `tests/astro_reentry`, and `examples/reentry` trees;
targeted integration edits to `src/astro_cli/main.py`, `pyproject.toml`, `README.md`,
`docs/reentry.md`, `docs/validation-matrix.md`, and `docs/current-state.md`.

**Fixed boundaries:** Existing orbit, launch, OD, assistant, backend, and digital-twin behavior must
remain compatible. Optional backends remain non-required release gates. The local model does not
claim CFD fidelity, certified TPS design, 6-DOF flight qualification, or operational impact
prediction.

**Validation:** Focused `astro_reentry` and CLI tests after each slice, then `ruff`, strict `mypy`,
the complete non-live pytest suite, representative CLI artifact inspection, `git diff --check`, and
wheel/sdist build.

**Stop conditions:** Complete when all done conditions and gates pass. Stop only for a destructive
operation, paid or sensitive external runtime, or a scientific choice that cannot be represented by
a conservative documented assumption.

## Work Packages

- [x] Add scenario, guidance, atmosphere, aerothermal, event, sample, peak, margin, and result models.
- [x] Implement YAML/JSON IO and human-readable summaries.
- [x] Implement local spherical 3-DOF ballistic and lifting propagation with event termination.
- [x] Implement prescribed-bank and target-tracking guidance with crossrange steering.
- [x] Implement target miss, requirement margins, peak extraction, and explicit provenance.
- [x] Implement local guidance optimization and tuned-scenario output.
- [x] Implement orbit-state handoff and backend dispatch boundaries.
- [x] Add `simulate-reentry`, `optimize-reentry`, and handoff CLI commands.
- [x] Add ballistic capsule and guided lifting-body examples.
- [x] Add focused tests, public docs, validation matrix coverage, and current-state evidence.
- [x] Run full verification and review claim boundaries.
