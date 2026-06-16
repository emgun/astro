# Validation Matrix

Date: 2026-06-15

This matrix lists the current validation surfaces for Astro Suite. The suite treats local
deterministic workflows as always-on regression references and optional backends as smoke-gated
adapters.

## Always-On Gates

| Area | Command | Expected Result |
| --- | --- | --- |
| Test suite | `python -m pytest -q` | All non-live tests pass; Orekit live tests remain skipped unless explicitly enabled. |
| Lint | `python -m ruff check .` | No lint findings. |
| Types | `python -m mypy` | Strict type checking passes. |
| Scenario validation | `astro validate examples/scenarios/leo_two_body.yaml` | Valid scenario message for `leo-two-body`. |
| Local propagation | `astro propagate examples/scenarios/leo_two_body.yaml --backend local --output /tmp/astro-local-trajectory.json` | Writes a `Trajectory` with `backend = "local"` and 11 samples. |
| Local MEO/GEO propagation | `astro propagate examples/scenarios/meo_two_body.yaml --backend local --output /tmp/astro-meo.json` and `astro propagate examples/scenarios/geo_two_body.yaml --backend local --output /tmp/astro-geo.json` | Writes medium-Earth and geosynchronous-radius two-body trajectories. |
| Local finite burn | `astro propagate examples/scenarios/leo_finite_burn.yaml --backend local --output /tmp/astro-finite-burn.json` | Writes maneuver start/end events, thrust-vector mass-flow provenance metadata, and sample mass history. |
| Local covariance | `astro propagate examples/scenarios/leo_covariance.yaml --backend local --output /tmp/astro-covariance.json` | Writes covariance history samples using finite-difference state transition and optional white-acceleration process noise. |
| Ephemeris export | `astro export-trajectory /tmp/astro-local-trajectory.json --format csv --output /tmp/astro-local-trajectory.csv` | Writes CSV samples with epoch, position, and velocity. |
| Local OD ingest | `astro estimate-measurements examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --output /tmp/astro-local-estimate.json` | Writes `EstimateResult` from explicit measurements. |
| Optical measurement synthesis | `astro synth-measurements examples/scenarios/leo_two_station_angles.yaml --backend local --output /tmp/astro-angle-measurements.json`, `astro synth-measurements examples/scenarios/leo_two_station_topocentric.yaml --backend local --output /tmp/astro-topocentric-measurements.json`, and `astro synth-measurements examples/scenarios/leo_geodetic_topocentric.yaml --backend local --output /tmp/astro-geodetic-topocentric-measurements.json` | Writes inertial RA/Dec, ECI-station local-horizon az/el, and WGS-84 geodetic-station az/el records in degrees. |
| Local launch report | `astro report-tuned-launch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --iterations 2 --orbit-duration-s 600 --orbit-step-s 60 --output /tmp/astro-launch-report.json` | Writes tuned launch report with insertion and short-arc assessments. |
| RocketPy-configured launch schema | `python -m pytest tests/astro_launch/test_launch_io.py::test_load_rocketpy_configured_launch_scenario tests/astro_launch/test_launch_io.py::test_load_rocketpy_single_stage_launch_scenario -q` | Loads the two-stage RocketPy configuration fixture and the single-stage direct-simulation fixture with typed RocketPy vehicle/motor/flight configuration. |
| Local research propagation | `astro research-propagate examples/scenarios/leo_two_body.yaml --backend local --cases 2 --position-sigma-km 0.01 --velocity-sigma-km-s 0.000001 --seed 7 --output /tmp/astro-research.json` | Writes `MonteCarloResult` with deterministic seeded cases. |

## Optional Backend Gates

| Backend | Command | Available Behavior | Unavailable Behavior |
| --- | --- | --- | --- |
| Orekit | `astro orekit-smoke` | Exits 0 with wrapper/JVM/frame/time availability. | Exits 1 with structured JSON explaining missing `orekit-jpype`, Java, or data-context failure. |
| Orekit propagation | `astro propagate examples/scenarios/leo_two_body.yaml --backend orekit --output /tmp/astro-orekit-trajectory.json`, `astro propagate examples/scenarios/leo_j2.yaml --backend orekit --output /tmp/astro-orekit-j2.json`, `astro propagate examples/scenarios/leo_orekit_high_fidelity.yaml --backend orekit --output /tmp/astro-orekit-high-fidelity.json`, `astro propagate examples/scenarios/leo_orekit_drag.yaml --backend orekit --output /tmp/astro-orekit-drag.json`, `astro propagate examples/scenarios/leo_orekit_srp.yaml --backend orekit --output /tmp/astro-orekit-srp.json`, and `astro propagate examples/scenarios/leo_orekit_third_body.yaml --backend orekit --output /tmp/astro-orekit-third-body.json` | Writes suite `Trajectory` with `backend = "orekit"` for two-body, J2, high-fidelity numerical-path, atmospheric-drag, SRP, and Sun/Moon third-body scenarios. | Exits 2 with `UnsupportedBackendError` diagnostics for runtime/data failures. |
| RocketPy | `astro rocketpy-smoke` | Exits 0 when RocketPy imports and required classes exist. | Exits 1 with structured JSON explaining missing package/import failure. |
| RocketPy launch adapter | `astro launch examples/launch/rocketpy_configured_single_stage.yaml --backend rocketpy --output /tmp/astro-rocketpy-launch.json` | Runs a single-stage configured solid rocket through RocketPy and returns suite `LaunchTrajectory` products through the adapter boundary. | Exits 2 with structured `UnsupportedBackendError` diagnostics when dependencies/configuration are unavailable or an unsupported multi-stage RocketPy scenario is requested. |
| Dymos/OpenMDAO | `astro dymos-smoke` | Exits 0 when Dymos and OpenMDAO APIs import. | Exits 1 with structured JSON explaining missing package/import failure. |
| Dymos launch optimization | `astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend dymos --output /tmp/astro-dymos-optimized-launch.json` | Runs a small Dymos/OpenMDAO vertical-ascent phase transcription and returns a suite `LaunchPitchTuningResult` with Dymos phase diagnostics. | Exits 2 with structured `UnsupportedBackendError` diagnostics when Dymos/OpenMDAO cannot be imported. |
| TudatPy | `astro tudat-smoke` | Exits 0 when TudatPy imports. | Exits 1 with structured JSON; install path is platform-specific and not assumed to be pip-only. |
| JAX/JAXLIB | `astro jax-smoke` | Exits 0 when JAX and `jax.numpy` import; `astro research-propagate --backend jax` can run two-body and J2 seeded ensembles. | Exits 1 with structured JSON explaining missing `astro-suite[research]` runtime. |

## Reference Tolerances

| Comparison | Current Gate |
| --- | --- |
| Local two-body determinism | Unit tests compare sample count and deterministic products. |
| MEO/GEO examples | Tests validate and propagate the medium-Earth and geosynchronous-radius scenarios. |
| Orekit two-body vs local | `orekit_live` test compares final LEO state with `abs(position) <= 1 km` and `abs(velocity) <= 1e-3 km/s` when `ASTRO_RUN_OREKIT_LIVE=1`. |
| Orekit J2 vs local | `orekit_live` test compares final LEO J2 state with `norm(position delta) < 0.05 km` and `norm(velocity delta) < 1e-4 km/s` when `ASTRO_RUN_OREKIT_LIVE=1`. |
| Orekit native OD bridge | Unit tests map suite WGS-84 geodetic range/range-rate records into Orekit `Range`/`RangeRate` objects and construct a native `BatchLSEstimator` boundary with a numerical propagator builder. |
| Local finite burns | Tests compare finite-burn propagation against the no-maneuver local baseline and verify maneuver events, thrust-vector mass depletion, and provenance. |
| Local covariance propagation | Tests verify covariance history length, epoch alignment, symmetry, finite-difference provenance metadata, and optional white-acceleration process-noise growth. |
| Launch handoff | Tests confirm `LaunchTrajectory.insertion_state` converts into a normal orbital `Scenario` and propagates locally. |
| OD explicit measurements | Tests validate JSON, CSV, TDM range/range-rate/angle ingest/export, inertial and local-horizon angle generation, WGS-84 geodetic station measurement generation, angle wrapping, and local least-squares convergence. |
| Research Monte Carlo | Tests confirm seeded repeatability, local Monte Carlo provenance, and built-in JAX two-body/J2 runner parity against the local reference for zero dispersion. |

## Backend Boundary Rule

No optional backend may return native engine objects through public products. All backends must map
into suite-owned `Trajectory`, `EstimateResult`, `LaunchTrajectory`, `LaunchPitchTuningResult`, or
`MonteCarloResult` models and include backend provenance in product metadata when available.
