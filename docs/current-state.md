# Astro Suite Current State

Date: 2026-07-08 00:00 PDT

## Canonical Workspace

Use `/Users/emerygunselman/Code/astro` for this roadmap thread.

The sibling checkout at `/Users/emerygunselman/Documents/astro` is stale for this work. It remains on
`codex/orbit-fd-od-mvp` at `ecccfa6`, has `main` at `dc5e253`, and contains an untracked
`docs/research/2026-06-20-verifiable-ai-space-workflows.md` file. Do not edit or merge from that
checkout unless the user explicitly asks to sync the old workspace.

Integrated Code checkout state before the digital-twin implementation branch:

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
pushed to `origin/main`. The stacked implementation branches make the digital twin roadmap slice
review-ready: `codex/digital-twin-plan` adds the single-spacecraft integrated twin, and
`codex/constellation-twin-design` extends it with a multi-spacecraft constellation screening product
that aggregates fleet access, revisit, link-budget, data-volume, and margin evidence over embedded
member twin results.

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
- Integrated Digital Twin implementation surface on `codex/digital-twin-plan`:
  `src/astro_twin` for suite-owned models, IO, geometry, power, thermal, ADCS, coverage,
  link-budget, margin aggregation, and runner orchestration; `astro run-twin` in
  `src/astro_cli/main.py`; `examples/twin/leo_observer.yaml`; `tests/astro_twin`; and
  `docs/digital-twin.md`.
- Integrated Digital Twin verification on `codex/digital-twin-plan`:
  `astro run-twin examples/twin/leo_observer.yaml --output /tmp/astro-twin-result.json --summary-output /tmp/astro-twin-summary.txt`
  wrote JSON and text artifacts with 11 samples, 1 access window, worst link margin 30.280 dB,
  limiting mass-margin warning 0.037, and explicit design-screening warnings;
  `python -m pytest tests/astro_twin -q` passed with 15 tests; `python -m ruff check .` passed;
  `python -m mypy` passed with no issues in 78 source files; `python -m pytest -q` passed with
  606 tests and 11 optional-backend skips; `git diff --check` was clean;
  `python -m pytest tests/test_packaging.py -q` passed with 3 tests; and `python -m build`
  produced `astro_suite-0.1.0.tar.gz` and `astro_suite-0.1.0-py3-none-any.whl`.
- Constellation Digital Twin planning artifacts:
  `docs/superpowers/specs/2026-07-08-constellation-digital-twin-design.md` and
  `docs/superpowers/plans/2026-07-08-constellation-digital-twin-implementation.md`.
- Constellation Digital Twin implementation surface on `codex/constellation-twin-design`:
  `src/astro_twin/constellation_models.py`, `src/astro_twin/constellation_io.py`,
  `src/astro_twin/constellation.py`, `astro run-constellation-twin` in `src/astro_cli/main.py`,
  `examples/twin/constellation_leo_observers.yaml`,
  `examples/twin/leo_observer_plane_a.yaml`, `examples/twin/leo_observer_plane_b.yaml`,
  `examples/scenarios/leo_two_body_phase_minus_4deg.yaml`,
  `tests/astro_twin/test_constellation_models.py`,
  `tests/astro_twin/test_constellation_io.py`,
  `tests/astro_twin/test_constellation_aggregation.py`,
  `tests/astro_twin/test_constellation_runner.py`, and constellation CLI tests in
  `tests/astro_cli/test_cli.py`.
- Constellation Digital Twin verification on `codex/constellation-twin-design`:
  `astro run-constellation-twin examples/twin/constellation_leo_observers.yaml --output /tmp/astro-constellation-twin.json --summary-output /tmp/astro-constellation-twin.txt`
  wrote JSON and text artifacts for `leo-observers` with 2 members, a 0.0 to 600.0 second analysis
  window, 1 `equator-eci` fleet access summary, 300.0 seconds total fleet access, 0.5 coverage
  fraction, 300.0 second longest gap, max simultaneous spacecraft 2, 1080.000 Mbit total data
  volume, member data volumes of 480.0 Mbit for `plane-a` and 600.0 Mbit for `plane-b`, and a
  limiting `fleet_longest_gap_s_equator-eci` margin with status `warn` and margin 0.000. The
  product keeps explicit design-screening warnings and does not claim operational constellation
  coverage authority.
- Constellation Digital Twin final gates on `codex/constellation-twin-design`:
  `python -m pytest tests/astro_twin/test_constellation_models.py tests/astro_twin/test_constellation_io.py tests/astro_twin/test_constellation_aggregation.py tests/astro_twin/test_constellation_runner.py tests/astro_cli/test_cli.py::test_run_constellation_twin_command_writes_json_and_summary -q`
  passed with `24 passed in 0.86s`; `python -m ruff check .` passed; `python -m mypy` passed with no
  issues in 81 source files; `python -m pytest -q` passed with
  `632 passed, 11 skipped in 6.87s`; `git diff --check` was clean;
  `python -m pytest tests/test_packaging.py -q` passed with `3 passed in 0.03s`; and
  `python -m build` produced `astro_suite-0.1.0.tar.gz` and
  `astro_suite-0.1.0-py3-none-any.whl`.

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
| integrated-digital-twin | review-ready | productize/verify | steward | `src/astro_twin`, `astro run-twin`, `examples/twin`, `tests/astro_twin`, `docs/digital-twin.md` | Single-spacecraft v1 writes suite-owned orbit geometry, power, thermal, ADCS, coverage, link budget, mass, and design-margin evidence from one scenario; required local gates pass with explicit design-screening claim boundaries. |
| constellation-digital-twin | review-ready | productize/verify | steward | `src/astro_twin/constellation*`, `astro run-constellation-twin`, `examples/twin/constellation_leo_observers.yaml`, constellation tests, `docs/digital-twin.md` | Constellation v1 embeds member `DigitalTwinResult`s and writes suite-owned fleet access, revisit, link, data-volume, and margin evidence for a checked two-member reference; required local gates pass with explicit design-screening and non-operational claim boundaries. |

## Next Best Paths

1. Review and merge `codex/digital-twin-plan`, then the stacked
   `codex/constellation-twin-design`, into `main` if the integrated and constellation twin scopes
   are accepted.
2. Tag a release candidate from `/Users/emerygunselman/Code/astro` once the user provides the
   desired release/tag policy.
3. Choose the next roadmap expansion deliberately: higher-fidelity subsystem models, constellation
   coverage-map/sensor-FOV scope, or one external-campaign validation scope. Do not treat all three
   as hidden implementation backlog.
