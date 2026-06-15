# Validation Matrix

Date: 2026-06-15

This matrix lists the current validation surfaces for Astro Suite. The suite treats local
deterministic workflows as always-on regression references and optional backends as smoke-gated
adapters.

## Always-On Gates

| Area | Command | Expected Result |
| --- | --- | --- |
| Test suite | `python -m pytest -q` | All non-live tests pass; Orekit live test remains skipped unless explicitly enabled. |
| Lint | `python -m ruff check .` | No lint findings. |
| Types | `python -m mypy` | Strict type checking passes. |
| Scenario validation | `astro validate examples/scenarios/leo_two_body.yaml` | Valid scenario message for `leo-two-body`. |
| Local propagation | `astro propagate examples/scenarios/leo_two_body.yaml --backend local --output /tmp/astro-local-trajectory.json` | Writes a `Trajectory` with `backend = "local"` and 11 samples. |
| Ephemeris export | `astro export-trajectory /tmp/astro-local-trajectory.json --format csv --output /tmp/astro-local-trajectory.csv` | Writes CSV samples with epoch, position, and velocity. |
| Local OD ingest | `astro estimate-measurements examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --output /tmp/astro-local-estimate.json` | Writes `EstimateResult` from explicit measurements. |
| Optical measurement synthesis | `astro synth-measurements examples/scenarios/leo_two_station_angles.yaml --backend local --output /tmp/astro-angle-measurements.json` | Writes right-ascension and declination records in degrees. |
| Local launch report | `astro report-tuned-launch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --iterations 2 --orbit-duration-s 600 --orbit-step-s 60 --output /tmp/astro-launch-report.json` | Writes tuned launch report with insertion and short-arc assessments. |
| Local research propagation | `astro research-propagate examples/scenarios/leo_two_body.yaml --backend local --cases 2 --position-sigma-km 0.01 --velocity-sigma-km-s 0.000001 --seed 7 --output /tmp/astro-research.json` | Writes `MonteCarloResult` with deterministic seeded cases. |

## Optional Backend Gates

| Backend | Command | Available Behavior | Unavailable Behavior |
| --- | --- | --- | --- |
| Orekit | `astro orekit-smoke` | Exits 0 with wrapper/JVM/frame/time availability. | Exits 1 with structured JSON explaining missing `orekit-jpype`, Java, or data-context failure. |
| Orekit propagation | `astro propagate examples/scenarios/leo_two_body.yaml --backend orekit --output /tmp/astro-orekit-trajectory.json` | Writes suite `Trajectory` with `backend = "orekit"`. | Exits 2 with `UnsupportedBackendError` diagnostics. |
| RocketPy | `astro rocketpy-smoke` | Exits 0 when RocketPy imports and required classes exist. | Exits 1 with structured JSON explaining missing package/import failure. |
| Dymos/OpenMDAO | `astro dymos-smoke` | Exits 0 when Dymos and OpenMDAO APIs import. | Exits 1 with structured JSON explaining missing package/import failure. |
| TudatPy | `astro tudat-smoke` | Exits 0 when TudatPy imports. | Exits 1 with structured JSON; install path is platform-specific and not assumed to be pip-only. |
| JAX/JAXLIB | `astro jax-smoke` | Exits 0 when JAX and `jax.numpy` import; `astro research-propagate --backend jax` can run two-body seeded ensembles. | Exits 1 with structured JSON explaining missing `astro-suite[research]` runtime. |

## Reference Tolerances

| Comparison | Current Gate |
| --- | --- |
| Local two-body determinism | Unit tests compare sample count and deterministic products. |
| Orekit two-body vs local | `orekit_live` test compares final LEO state with `abs(position) <= 1 km` and `abs(velocity) <= 1e-3 km/s` when `ASTRO_RUN_OREKIT_LIVE=1`. |
| Launch handoff | Tests confirm `LaunchTrajectory.insertion_state` converts into a normal orbital `Scenario` and propagates locally. |
| OD explicit measurements | Tests validate JSON, CSV, TDM range/range-rate ingest/export, inertial angle generation, and local least-squares convergence. |
| Research Monte Carlo | Tests confirm seeded repeatability, local Monte Carlo provenance, and built-in JAX two-body runner parity against the local reference for zero dispersion. |

## Backend Boundary Rule

No optional backend may return native engine objects through public products. All backends must map
into suite-owned `Trajectory`, `EstimateResult`, `LaunchTrajectory`, `LaunchPitchTuningResult`, or
`MonteCarloResult` models and include backend provenance in product metadata when available.
