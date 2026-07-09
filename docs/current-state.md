# Astro Suite Current State

Date: 2026-07-09 00:00 PDT

## Canonical Workspace

Use `/Users/emerygunselman/Code/astro` for this roadmap thread.

The sibling checkout at `/Users/emerygunselman/Documents/astro` is stale for this work. It remains on
`codex/orbit-fd-od-mvp` at `ecccfa6`, has `main` at `dc5e253`, and contains an untracked
`docs/research/2026-06-20-verifiable-ai-space-workflows.md` file. Do not edit or merge from that
checkout unless the user explicitly asks to sync the old workspace.

Integrated Code checkout state for the external-validation refresh:

- Base branch: `main`
- Working branch: `codex/orekit-validation-refresh`
- Latest integrated commit before this state update: `8111390`
- Required local release gates: passed in the subsystem-fidelity merge pass.
- Optional backend smoke refresh on 2026-07-09: Orekit, RocketPy, Dymos/OpenMDAO, and JAX
  were available on the current machine. TudatPy was not installed in the base environment, and
  the previously recorded isolated Tudat environment at `/tmp/astro-tudat-live-env` no longer
  existed.
- Bounded live validation refresh on `codex/orekit-validation-refresh`: Orekit high-fidelity
  covariance, drag, SRP, and Sun/Moon third-body trajectory products executed with the explicit
  Homebrew OpenJDK and `~/.orekit/orekit-data.zip` environment. This is machine-scoped optional
  backend evidence, not a production covariance certification.

## North Star

Astro Suite is a Python flight-dynamics suite with suite-owned product boundaries for orbital
simulation, flight dynamics, orbit determination, launch/ascent, optional operational backends, and
research workflows. External engines are adapters; public outputs stay in Astro Suite models with
explicit provenance and claim boundaries.

## Current Roadmap Decision

The previous suite-owned roadmap pass, Verifiable OD Workflow Pack, integrated digital twin,
constellation digital twin, constellation coverage-map stack, and subsystem fidelity pack are
integrated on `main` and pushed to `origin/main`. The current digital twin surface includes
deterministic subsystem fidelity evidence for scheduled power loads, battery efficiency/energy,
thermal heat balance, ADCS slew/utilization margins, and itemized mass-budget rollups.

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
- Constellation Coverage Maps v1 planning artifacts:
  `docs/superpowers/specs/2026-07-09-constellation-coverage-maps-design.md` and
  `docs/superpowers/plans/2026-07-09-constellation-coverage-maps-implementation.md`.
- Constellation Coverage Maps v1 implementation surface on `codex/coverage-maps-v1`:
  `ConstellationCoverageSensorConfig`, `ConstellationCoverageTargetConfig`,
  `ConstellationCoverageMapConfig`, `CoverageMapSummary`, and `CoverageMapTargetSummary` in
  `src/astro_twin/constellation_models.py`; `aggregate_coverage_maps` and coverage-map fleet
  margins in `src/astro_twin/constellation.py`; summary formatting in
  `src/astro_twin/constellation_io.py`; the checked two-target `equatorial-targets` map in
  `examples/twin/constellation_leo_observers.yaml`; and focused model, aggregation, runner, IO, and
  CLI tests.
- Constellation Coverage Maps v1 verification on `codex/coverage-maps-v1`:
  `astro run-constellation-twin examples/twin/constellation_leo_observers.yaml --output /tmp/astro-constellation-coverage-map.json --summary-output /tmp/astro-constellation-coverage-map.txt`
  wrote JSON and text artifacts for `leo-observers` with 1 coverage map, 2 configured targets, 2
  covered targets, 0.250 mean coverage fraction, 0.200 minimum target coverage fraction, 480.0
  second maximum target gap, max simultaneous spacecraft 2, target fractions of 0.200 for
  `prime-meridian` and 0.300 for `east-equator`, and limiting
  `coverage_map_min_fraction_equatorial-targets` margin with status `warn` and margin 0.000. The
  focused constellation and CLI slice passed with `29 passed in 0.86s`; `python -m ruff check .`
  passed; `python -m mypy` passed with no issues in 81 source files; the twin test slice passed with
  `43 passed in 0.32s`; `python -m pytest -q` passed with `637 passed, 11 skipped in 7.09s`;
  `git diff --check` was clean; the packaging test slice passed with `3 passed in 0.03s`; and
  `python -m build` produced
  `astro_suite-0.1.0.tar.gz` and `astro_suite-0.1.0-py3-none-any.whl`. The product keeps explicit
  design-screening warnings and does not claim operational coverage authority or certified sensor
  performance.
- Subsystem Fidelity Pack planning artifacts:
  `docs/superpowers/specs/2026-07-09-subsystem-fidelity-pack-design.md` and
  `docs/superpowers/plans/2026-07-09-subsystem-fidelity-pack-implementation.md`.
- Subsystem Fidelity Pack implementation surface merged from `codex/subsystem-fidelity-pack`:
  `PowerLoadSchedule`, `MassBudgetItemConfig`, and `MassBudgetSummary` in
  `src/astro_twin/models.py`; scheduled-load and battery-efficiency logic in
  `src/astro_twin/power.py`; albedo/planet-IR/mode-heat balance logic in
  `src/astro_twin/thermal.py`; slew-rate and actuator-utilization evidence in
  `src/astro_twin/adcs.py`; `build_mass_budget_summary` in `src/astro_twin/mass.py`; subsystem
  margins in `src/astro_twin/margins.py`; runner wiring in `src/astro_twin/runner.py`; richer
  assumptions in `examples/twin/leo_observer.yaml`; and focused model, power, thermal, ADCS, mass,
  margin, runner, and CLI tests.
- Subsystem Fidelity Pack verification on `codex/subsystem-fidelity-pack`:
  `astro run-twin examples/twin/leo_observer.yaml --output /tmp/astro-subsystem-fidelity-twin.json --summary-output /tmp/astro-subsystem-fidelity-twin.txt`
  wrote JSON and text artifacts with 11 samples, minimum battery SOC 0.850, first scheduled-load
  sample at 120.0 seconds with 35.0 W scheduled load, 1042.686 Wh battery energy, bus heat-balance
  evidence, ADCS slew-rate margin 0.150 deg/s, actuator utilization 0.375, itemized mass base
  128.0 kg, contingency 12.6 kg, total 140.6 kg, dry-plus-payload reference 145.0 kg, and limiting
  `mass_budget_rollup_margin_kg` status `warn` with margin 4.400 kg. `python -m pytest
  tests/astro_twin -q` passed with `48 passed in 0.60s`; the twin CLI test passed with
  `1 passed in 0.86s`; `python -m ruff check .` passed; `python -m mypy` passed with no issues in
  82 source files; `python -m pytest -q` passed with `642 passed, 11 skipped in 7.00s`;
  `git diff --check` was clean; the packaging test slice passed with `3 passed in 0.02s`; and
  `python -m build` produced `astro_suite-0.1.0.tar.gz` and
  `astro_suite-0.1.0-py3-none-any.whl`. The product keeps explicit design-screening warnings and
  does not claim EPS certification, thermal certification, mass-properties authority, or
  flight-qualified GNC simulation. PR #6 merged this scope into `main` at merge commit `729a7d4`.
- External validation refresh on `codex/orekit-validation-refresh`:
  `JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home PATH="/opt/homebrew/opt/openjdk/bin:$PATH" astro orekit-smoke`
  reported Orekit JPype 13.1.5.0 available; `astro rocketpy-smoke`, `astro dymos-smoke`, and
  `astro jax-smoke` also reported available runtimes; `astro tudat-smoke` reported TudatPy not
  installed, and `conda run -p /tmp/astro-tudat-live-env ...` reported the historical Tudat
  environment path no longer exists. The Orekit validation refresh passed
  `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_covariance_history_returns_suite_product tests/astro_backends/test_orekit_propagation.py::test_live_orekit_high_fidelity_covariance_records_force_models -q`
  with `2 passed in 9.22s` and wrote `/tmp/astro-orekit-validation-high-fidelity-covariance-20260709.json`,
  `/tmp/astro-orekit-validation-drag-20260709.json`, `/tmp/astro-orekit-validation-srp-20260709.json`,
  and `/tmp/astro-orekit-validation-third-body-20260709.json`. The high-fidelity covariance product
  contains 11 samples, 11 covariance samples, `orekit_finite_difference_state_transition`, white
  acceleration process noise, and transition force models for J2, drag, SRP, Sun, and Moon.

Post-MVP / external-campaign items:

- Production-grade covariance certification through external drag, SRP, and third-body dynamics.
- Flight-qualified actuator/sensor ACS modeling beyond deterministic screening products.
- Operational constellation coverage authority, scheduling, or certified sensor performance beyond
  deterministic target-grid screening.
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
| integrated-digital-twin | done | productize/verify | steward | `src/astro_twin`, `astro run-twin`, `examples/twin`, `tests/astro_twin`, `docs/digital-twin.md` | Single-spacecraft v1 writes suite-owned orbit geometry, power, thermal, ADCS, coverage, link budget, mass, and design-margin evidence from one scenario; required local gates pass with explicit design-screening claim boundaries and the stack is merged on `main`. |
| constellation-digital-twin | done | productize/verify | steward | `src/astro_twin/constellation*`, `astro run-constellation-twin`, `examples/twin/constellation_leo_observers.yaml`, constellation tests, `docs/digital-twin.md` | Constellation v1 embeds member `DigitalTwinResult`s and writes suite-owned fleet access, revisit, link, data-volume, and margin evidence for a checked two-member reference; required local gates pass with explicit design-screening and non-operational claim boundaries and the stack is merged on `main`. |
| constellation-coverage-maps | done | productize/verify | steward | `src/astro_twin/constellation*`, `examples/twin/constellation_leo_observers.yaml`, `docs/digital-twin.md`, constellation tests | Coverage Maps v1 writes deterministic target-grid sensor coverage summaries and fleet margins from member geometry samples; required local gates pass with explicit design-screening and non-operational claim boundaries and the stack is merged on `main`. |
| subsystem-fidelity-pack | done | productize/verify | steward | `src/astro_twin`, `examples/twin/leo_observer.yaml`, `docs/digital-twin.md`, subsystem tests | Adds scheduled power loads, battery energy/efficiency, thermal heat-balance evidence, ADCS slew/utilization margins, and itemized mass-budget rollups; required local gates pass with explicit design-screening and non-certification claim boundaries and the scope is merged on `main`. |
| external-validation-refresh | done | verify | steward | `docs/validation/live-backend-campaigns.md`, optional runtime smoke checks, Orekit live covariance gate | Refreshes machine-scoped optional validation evidence: Orekit high-fidelity covariance and drag/SRP/third-body products pass on this machine; Tudat-specific current release claims are not refreshed because the historical isolated Tudat environment is absent. |

## Next Best Paths

1. Merge the external-validation refresh if the branch review is clean, then tag a release candidate
   from `/Users/emerygunselman/Code/astro` once the user provides the desired release/tag policy.
2. Recreate an isolated TudatPy environment only if a release claim specifically needs a fresh
   Tudat-vs-local comparison or native variational covariance refresh.
3. Choose the next roadmap expansion deliberately: mission design/reporting polish, a bounded
   operational-readiness criteria pass, or one new external-campaign validation scope. Do not treat
   all three as hidden implementation backlog.
