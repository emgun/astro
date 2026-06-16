# Astro Suite

Astro Suite is a Python flight dynamics project for scenario validation, local reference
propagation, launch/ascent sanity cases, synthetic orbit-determination measurements, batch OD,
and backend adapters.

## Current Scope

The current implementation slice covers:

- Pydantic scenario validation from YAML.
- Local two-body and J2 reference propagation with deterministic provenance metadata.
- Flight-dynamics trajectory product fields for events, impulsive and finite-burn maneuvers, and
  covariance history, plus local finite-difference covariance propagation, CSV ephemeris export, and
  seeded initial-state Monte Carlo propagation.
- Local launch/ascent reference propagation with vertical and pitch-program guidance, staged mass
  depletion, drag, events, and launch-to-orbit insertion handoff.
- Launch pitch-program sweep, two-knot tuning, and tuned launch-to-orbit reporting over repeated
  local ascent/orbit propagations, batch ranking, and report-to-report comparison.
- Synthetic range, range-rate, inertial right ascension, and declination measurement generation.
- Local SciPy batch least-squares orbit determination with rank and convergence checks.
- CLI workflows for validation, propagation, launch, launch-to-orbit handoff, synthetic
  measurements, synthetic OD, measurement-file OD ingest/export, and research propagation.
- Optional backend smoke gates and product boundaries for Orekit, RocketPy, Dymos/OpenMDAO, TudatPy,
  and JAX.

Launch/ascent currently uses deliberately simple local vertical and pitch-program baselines.
RocketPy and Dymos/OpenMDAO are recognized adapter boundaries, but live external-engine workflows
still require backend-specific configuration beyond the aggregate local launch schema.

## Setup

```bash
python -m pip install -e '.[dev]'
```

Optional Orekit wrapper smoke and propagation support:

```bash
python -m pip install -e '.[orekit]'
export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
export ASTRO_OREKIT_DATA_PATH="$HOME/.orekit/orekit-data.zip"
astro orekit-smoke
astro propagate examples/scenarios/leo_two_body.yaml --backend orekit --output orekit_trajectory.json
```

If `orekit-jpype`, Java, or Orekit data are not configured, `astro orekit-smoke` exits nonzero with
structured JSON explaining the missing piece. The runtime checks `ASTRO_OREKIT_DATA_PATH`,
`OREKIT_DATA_PATH`, then `~/.orekit/orekit-data.zip` for Orekit data. `astro propagate --backend
orekit` currently supports two-body propagation through Orekit's Keplerian propagator, J2 through
Orekit's numerical propagator with `J2OnlyPerturbation`, and `orekit_high_fidelity` as the numerical
propagation expansion path. Atmospheric drag is available through Orekit `DragForce` with
`SimpleExponentialAtmosphere` and `IsotropicDrag`; solar radiation pressure is available through
Orekit `SolarRadiationPressure` and `IsotropicRadiationSingleCoefficient`. The scenario
force-model flag for third-body gravity is accepted by the schema but currently produces explicit
unsupported-feature diagnostics until that Orekit force model is implemented.

Optional launch backend smoke checks:

```bash
python -m pip install -e '.[launch,optimization]'
astro rocketpy-smoke
astro dymos-smoke
```

RocketPy and Dymos/OpenMDAO are behind explicit adapter gates. The current `rocketpy` launch path
loads the optional runtime and preserves the `LaunchTrajectory` product boundary, but live RocketPy
simulation still requires backend-specific rocket/motor configuration beyond the aggregate local
launch schema.

Optional research backend smoke checks:

```bash
python -m pip install -e '.[research]'
astro tudat-smoke
astro jax-smoke
```

Tudat and JAX are optional research/cross-check boundaries. TudatPy is not currently assumed to be
available from PyPI on every platform, so its smoke command reports installation state without
promising a pip-only install path. JAX research propagation returns suite `MonteCarloResult` products
and remains separate from operational Orekit semantics.

## Commands

```bash
astro validate examples/scenarios/leo_two_body.yaml
astro propagate examples/scenarios/leo_two_body.yaml --backend local --output trajectory.json
astro propagate examples/scenarios/meo_two_body.yaml --backend local --output meo_trajectory.json
astro propagate examples/scenarios/geo_two_body.yaml --backend local --output geo_trajectory.json
astro propagate examples/scenarios/leo_j2.yaml --backend local --output j2_trajectory.json
astro propagate examples/scenarios/leo_finite_burn.yaml --backend local --output finite_burn_trajectory.json
astro propagate examples/scenarios/leo_covariance.yaml --backend local --output covariance_trajectory.json
astro propagate examples/scenarios/leo_two_body.yaml --backend orekit --output orekit_trajectory.json
astro propagate examples/scenarios/leo_j2.yaml --backend orekit --output orekit_j2_trajectory.json
astro propagate examples/scenarios/leo_orekit_high_fidelity.yaml --backend orekit --output orekit_high_fidelity_trajectory.json
astro propagate examples/scenarios/leo_orekit_drag.yaml --backend orekit --output orekit_drag_trajectory.json
astro propagate examples/scenarios/leo_orekit_srp.yaml --backend orekit --output orekit_srp_trajectory.json
astro export-trajectory trajectory.json --format csv --output trajectory.csv
astro monte-carlo examples/scenarios/leo_two_body.yaml --cases 4 --position-sigma-km 0.01 --velocity-sigma-km-s 0.000001 --seed 7 --backend local --output monte_carlo.json
astro rocketpy-smoke
astro dymos-smoke
astro tudat-smoke
astro jax-smoke
astro launch examples/launch/vertical_two_stage.yaml --backend local --output launch.json
astro launch examples/launch/pitch_program_two_stage.yaml --backend local --output pitch_launch.json
astro sweep-launch-pitch examples/launch/pitch_program_two_stage.yaml --point-index 3 --pitch-deg-values 10,20,30 --output pitch_sweep.json
astro tune-launch-pitch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --initial-span-deg 10 --iterations 2 --output pitch_tuning.json --tuned-scenario-output tuned_pitch_program.yaml
astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend local --point-indices 2,3 --iterations 2 --output launch_optimization.json
astro report-tuned-launch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --initial-span-deg 10 --iterations 2 --orbit-duration-s 600 --orbit-step-s 60 --output tuned_launch_report.json
astro batch-report-tuned-launch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --iterations-values 1,2,3 --initial-span-deg 10 --orbit-duration-s 600 --orbit-step-s 60 --output tuned_launch_batch.json
astro compare-tuned-launch-reports tuned_launch_report_baseline.json tuned_launch_report_candidate.json --output tuned_launch_comparison.json
astro handoff-launch launch.json --output insertion.yaml --duration-s 600 --step-s 60
astro propagate insertion.yaml --backend local --output insertion_trajectory.json
astro synth-measurements examples/scenarios/leo_two_station_od.yaml --backend local --output measurements.json
astro synth-measurements examples/scenarios/leo_two_station_od.yaml --backend orekit --output orekit_measurements.json
astro synth-measurements examples/scenarios/leo_two_station_angles.yaml --backend local --output angle_measurements.json
astro synth-measurements examples/scenarios/leo_two_station_topocentric.yaml --backend local --output topocentric_measurements.json
astro export-measurements measurements.json --format csv --output measurements.csv
astro export-measurements measurements.json --format tdm --output measurements.tdm
astro estimate examples/scenarios/leo_two_body.yaml --backend local --output estimate.json
astro estimate examples/scenarios/leo_two_body.yaml --backend orekit --output orekit_estimate.json
astro estimate-measurements examples/scenarios/leo_two_station_od.yaml measurements.json --backend local --output estimate.json
astro estimate-measurements examples/scenarios/leo_two_station_od.yaml measurements.json --backend orekit --output orekit_estimate.json
astro research-propagate examples/scenarios/leo_two_body.yaml --backend local --cases 4 --position-sigma-km 0.01 --velocity-sigma-km-s 0.000001 --seed 7 --output research_propagation.json
astro orekit-smoke
```

`astro estimate` is an MVP synthetic demonstration workflow. It keeps the source scenario unchanged,
adds in-memory demo geometry for observability, generates synthetic measurements, perturbs the
initial state as an estimate seed, and records that provenance in the output metadata. The `--backend`
option selects the propagation backend used for synthetic truth generation and residual propagation.
With `--backend orekit`, the suite estimator uses Orekit-backed propagation when the optional Orekit
runtime is installed; it is not yet Orekit's native `BatchLSEstimator`.

`astro export-trajectory` converts suite trajectory JSON into a CSV ephemeris table containing
epoch, position, and velocity samples. `astro monte-carlo` runs a seeded initial-state ensemble by
perturbing the scenario's Cartesian state and propagating each case through the selected backend.
This is a repeatable product workflow for uncertainty screening; it is not yet production covariance
propagation or conjunction analysis.

Local orbital propagation accepts an optional `maneuvers` schedule on `Scenario`. Impulsive
maneuvers apply their full `delta_v_km_s` at the maneuver epoch; finite burns apply the configured
total delta-v as constant inertial acceleration over `duration_s` and record maneuver start/end
events in the trajectory. This is a deterministic maneuver baseline, not a thrust-vector, mass-flow,
or attitude-control model.

Local propagation also accepts an optional `initial_covariance` 6x6 matrix. When present, the local
backend emits a `covariance_history` sample at each trajectory epoch using a finite-difference state
transition with no process noise. This is useful for product wiring and first-order sensitivity
screening; high-fidelity covariance dynamics remain a backend validation task.

`astro launch` is the launch/ascent MVP workflow. It loads a launch scenario, runs the local
vertical or pitch-program baseline, and writes a launch trajectory product with samples, stage
events, dynamic pressure, acceleration, downrange, target miss metrics, and an `insertion_state`
compatible with the shared `OrbitState` product. This local backend is a deterministic data-flow
baseline, not a production launch simulator.

`astro sweep-launch-pitch` is the first launch targeting workflow. It varies one pitch-program knot,
runs the local launch propagator for each candidate pitch angle, and writes a JSON product with
altitude miss, velocity miss, weighted score, final downrange, and the best case. It is a transparent
grid sweep rather than an optimizer; that keeps the target-miss contract clear before adding Dymos,
OpenMDAO, or RocketPy-backed targeting.

`astro tune-launch-pitch` is the first multi-knot targeting workflow. It varies two pitch-program
knots on a deterministic 3x3 grid, shrinks the search span each iteration, writes a JSON trace of
every evaluated candidate, and can write the best tuned `LaunchScenario` back to YAML. This is still
a coarse-to-fine targeting analysis tool, not a production optimizer.

`astro optimize-launch` is the neutral launch optimization entry point. With `--backend local`, it
uses the current pitch-program tuner and writes the same `LaunchPitchTuningResult` product. With
`--backend dymos`, it loads the optional Dymos/OpenMDAO runtime and requires a validated Dymos phase
runner; the current suite does not fake an optimal-control solve from the local aggregate schema.

`astro report-tuned-launch` runs the current local end-to-end launch analysis: tune two pitch knots,
propagate the tuned ascent, hand off insertion to an orbit scenario, propagate a short orbital arc,
and write one JSON product with the component products plus insertion and short-arc target metrics.
The report also includes pass/fail assessment gates using the target orbit's configured altitude
and velocity tolerances, with named checks for insertion and short-arc misses.
It is a deterministic report over local baselines, not a substitute for high-fidelity ascent design.

`astro batch-report-tuned-launch` runs the tuned launch report workflow for multiple iteration
counts and ranks the resulting reports by normalized assessment error: the sum of absolute check
misses divided by their configured tolerances. It is a deterministic tuning-depth comparison table,
not a new optimizer.

`astro compare-tuned-launch-reports` compares two saved tuned launch report JSON products without
rerunning analysis. It writes signed deltas and absolute-error improvement for insertion and
short-arc target misses, plus pass/fail changes between the baseline and candidate reports.

`astro handoff-launch` converts a launch trajectory product into a normal orbital propagation
scenario initialized from `LaunchTrajectory.insertion_state`. The generated YAML is intentionally
plain `Scenario` input, so the next step is the existing `astro propagate` command rather than a
special launch-aware propagation path.

`astro synth-measurements` and `astro estimate-measurements` both accept `--backend local` or
`--backend orekit`. The explicit ingest workflow loads a scenario plus a JSON, CSV, or CCSDS
Tracking Data Message (TDM) measurement file, then estimates from the caller-provided station
geometry and measurement records without adding demo geometry. JSON and CSV can carry the suite's
range, range-rate, inertial right-ascension/declination, and local-horizon azimuth/elevation
records. Angle records use degrees. The current azimuth/elevation model derives a local
east/north/up basis from each station's ECI position vector; it is not yet an Earth-fixed geodetic
station or EOP-aware pointing model. JSON inputs match the output of `astro synth-measurements`;
CSV and TDM inputs are auto-detected by `.csv` and `.tdm` extensions or can be forced with
`--format csv` / `--format tdm`.

`astro export-measurements` converts suite JSON measurement files into JSON, CSV, or TDM products.
JSON and CSV preserve all supported suite measurement types. TDM export is intentionally restricted
to range and range-rate records until an operational CCSDS angle mapping is designed. The example
files under `examples/measurements/` are generated from `leo_two_station_od.yaml` and cover all
three range/range-rate ingest/export formats.

`astro research-propagate` is the research backend entry point for seeded propagation ensembles.
With `--backend local`, it runs the deterministic Monte Carlo workflow. With `--backend jax`, it
loads the optional JAX runtime and runs a vectorized two-body RK4 ensemble for current two-body
scenarios. JAX remains a research backend, not a replacement for operational Orekit semantics or
validated higher-fidelity force models.

CSV inputs use one row per measurement with these required columns:

```csv
scenario_id,measurement_type,epoch,observer,observed_object,value,sigma,units
```

The optional `metadata_json` column can carry a JSON object for row-level metadata. Valid units are
`km`, `km/s`, and `deg`, depending on measurement type. The
`leo_two_station_od.yaml` example includes two stations because the one-station propagation example
is intentionally under-observed for six-state OD.

TDM ingest currently supports KVN-formatted sequential segments with `TIME_SYSTEM = UTC`,
`PARTICIPANT_n`, `PATH`, `RANGE` in `km`, and `DOPPLER_INSTANTANEOUS` or `DOPPLER_INTEGRATED`
mapped to range-rate measurements in `km/s`. TDM does not provide the suite's scenario identifier
or estimator sigmas directly, so an optional segment-level `SCENARIO_ID` extension is checked when
present, and the parser uses default sigmas of `0.01 km` for range and `1e-5 km/s` for range-rate.
TDM ingest/export does not yet support suite angle records.

## Verification

```bash
python -m pytest -v
python -m ruff check .
python -m mypy
```

Additional release and backend documentation:

- [Validation matrix](docs/validation-matrix.md)
- [Backend installation guide](docs/backend-installation.md)
- [Release checklist](docs/release-checklist.md)
