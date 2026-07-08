# Astro Suite Current State

Date: 2026-07-08 00:00 PDT

## Canonical Workspace

Use `/Users/emerygunselman/Code/astro` for this roadmap thread.

The sibling checkout at `/Users/emerygunselman/Documents/astro` is stale for this work. It remains on
`codex/orbit-fd-od-mvp` at `ecccfa6`, has `main` at `dc5e253`, and contains an untracked
`docs/research/2026-06-20-verifiable-ai-space-workflows.md` file. Do not edit or merge from that
checkout unless the user explicitly asks to sync the old workspace.

Integrated Code checkout state before this final state update:

- Branch: `main`
- Latest integrated release-evidence commit before this state update: `91e213a`
- Required local release gates: passed.
- Optional backend smoke refresh: passed on the current checkout for Orekit, RocketPy,
  Dymos/OpenMDAO, TudatPy, and JAX.

## North Star

Astro Suite is a Python flight-dynamics suite with suite-owned product boundaries for orbital
simulation, flight dynamics, orbit determination, launch/ascent, optional operational backends, and
research workflows. External engines are adapters; public outputs stay in Astro Suite models with
explicit provenance and claim boundaries.

## Current Roadmap Decision

The previous suite-owned roadmap pass and Verifiable OD Workflow Pack are integrated on `main` and
pushed to `origin/main`. The new active roadmap scope is the Integrated Digital Twin: a
single-spacecraft mission screening product that combines orbit geometry, power, thermal, ADCS,
coverage, link budget, mass, and design-margin evidence in one suite-owned workflow.

Implemented and verified in the current pass:

- Local deterministic propagation, covariance, events, maneuvers, conjunction, attitude, OD,
  measurement IO, DSN-style bridge/calibration products, launch/ascent, launch tuning/reporting, and
  research Monte Carlo workflows.
- Orekit, RocketPy, Dymos/OpenMDAO, TudatPy, and JAX runtime gates and adapter boundaries.
- Required local release checklist and packaging gate.
- Optional live backend campaign ledger with clear environment and claim boundaries.
- Verifiable OD Workflow Pack slices: `examples/workflows/local_od/manifest.yaml`,
  `examples/workflows/local_od/golden_prompts.yaml`, `astro ask --report-output`,
  `astro ask --report-summary-output`,
  workflow-report metrics for measurement count, TDM line count, estimate convergence, iteration
  count, RMS, Jacobian rank, residual count, and max residual, plus golden prompt regression checks
  across every supported local OD scenario.
- OD workflow pack branch release-readiness pass: focused assistant tests, real CLI execution with
  trace/JSON/text report artifacts, `ruff`, `mypy`, full `pytest`, packaging tests,
  `git diff --check`, and `python -m build` passed locally on `codex/od-workflow-pack`.
- Integrated Digital Twin planning artifacts: `docs/superpowers/specs/2026-07-08-integrated-digital-twin-design.md`
  and `docs/superpowers/plans/2026-07-08-integrated-digital-twin-implementation.md`.

Post-MVP / external-campaign items:

- Production-grade covariance certification through external drag, SRP, and third-body dynamics.
- Flight-qualified actuator/sensor ACS modeling beyond deterministic screening products.
- Native RocketPy multi-motor/staged-separation execution if a validated upstream API becomes
  available.
- Full high-fidelity multistage Dymos ascent design optimization beyond the current suite product
  boundary.
- Official standards-grade DSN ODF/TNF decoding, official station calibration solving, and deeper
  astrometry/CCSDS authority.
- Operational-grade differentiable OD services beyond the current JAX research products.

## Active Work Registry

| ID | Status | Lane | Owner | Scope | Acceptance |
| --- | --- | --- | --- | --- | --- |
| roadmap-finish-state | done | decide/verify | steward | `docs/current-state.md`, release and live-ledger docs | State file records canonical workspace, release readiness, optional smoke evidence, and remaining post-MVP boundaries. |
| verifiable-od-workflow-pack | done | productize/verify | steward | `src/astro_assistant`, `examples/workflows/local_od`, assistant docs and validation matrix | Local OD workflow has a manifest, golden prompt fixtures, JSON and text report outputs, focused tests, real CLI execution evidence, and is pushed on `main`. |
| integrated-digital-twin | planned | productize/verify | steward | `docs/superpowers/specs`, `docs/superpowers/plans`, future `src/astro_twin` | Design and implementation plan define a single-spacecraft v1 with orbit geometry, power, thermal, ADCS, coverage, link budget, mass, and design margins. |

## Next Best Paths

1. Review and approve the integrated digital twin design/implementation plan.
2. Implement `astro_twin` v1 on a feature branch using the planned TDD task sequence.
3. Tag a release candidate from `/Users/emerygunselman/Code/astro` once the user provides the
   desired release/tag policy.
