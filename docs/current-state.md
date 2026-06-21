# Astro Suite Current State

Date: 2026-06-20 17:50 PDT

## Canonical Workspace

Use `/Users/emerygunselman/Code/astro` for this roadmap thread.

The sibling checkout at `/Users/emerygunselman/Documents/astro` is stale for this work. It remains on
`codex/orbit-fd-od-mvp` at `ecccfa6`, has `main` at `dc5e253`, and contains an untracked
`docs/research/2026-06-20-verifiable-ai-space-workflows.md` file. Do not edit or merge from that
checkout unless the user explicitly asks to sync the old workspace.

Integrated Code checkout state before this final state update:

- Branch: `main`
- Latest integrated release-evidence commit before this state update: `870cb13`
- Required local release gates: passed.
- Optional backend smoke refresh: passed on the current checkout for Orekit, RocketPy,
  Dymos/OpenMDAO, TudatPy, and JAX.

## North Star

Astro Suite is a Python flight-dynamics suite with suite-owned product boundaries for orbital
simulation, flight dynamics, orbit determination, launch/ascent, optional operational backends, and
research workflows. External engines are adapters; public outputs stay in Astro Suite models with
explicit provenance and claim boundaries.

## Current Roadmap Decision

The current suite-owned roadmap pass is release-ready for the implemented product surfaces. The
remaining roadmap text is no longer a local implementation backlog for this pass; it is a set of
explicit non-claims and future external validation campaigns.

Implemented and verified in the current pass:

- Local deterministic propagation, covariance, events, maneuvers, conjunction, attitude, OD,
  measurement IO, DSN-style bridge/calibration products, launch/ascent, launch tuning/reporting, and
  research Monte Carlo workflows.
- Orekit, RocketPy, Dymos/OpenMDAO, TudatPy, and JAX runtime gates and adapter boundaries.
- Required local release checklist and packaging gate.
- Optional live backend campaign ledger with clear environment and claim boundaries.

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

## Next Best Paths

1. Tag or push a release candidate from `/Users/emerygunselman/Code/astro` once the user provides
   the desired remote/tag policy.
2. Run a fresh optional live-backend campaign only when those claims need promotion for a specific
   machine or release candidate.
3. Start a new roadmap cycle only after choosing one external-campaign scope, such as official DSN
   fixtures, production covariance validation criteria, or native RocketPy staged-flight support.
