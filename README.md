# Astro Suite

Astro Suite is a Python flight dynamics project for scenario validation, local reference
propagation, launch/ascent sanity cases, synthetic orbit-determination measurements, batch OD,
and backend adapters.

## Current Scope

The current implementation slice covers:

- Pydantic scenario validation from YAML.
- Local two-body and J2 reference propagation with deterministic provenance metadata.
- Local launch/ascent reference propagation with vertical and pitch-program guidance, staged mass
  depletion, drag, events, and launch-to-orbit insertion handoff.
- Launch pitch-program sweep, two-knot tuning, and tuned launch-to-orbit reporting over repeated
  local ascent/orbit propagations.
- Synthetic range and range-rate measurement generation.
- Local SciPy batch least-squares orbit determination with rank and convergence checks.
- CLI workflows for validation, propagation, launch, launch-to-orbit handoff, synthetic
  measurements, synthetic OD, and measurement-file OD ingest/export.
- Optional Orekit Python-wrapper smoke checks through `orekit_jpype`.

Launch/ascent currently uses deliberately simple local vertical and pitch-program baselines.
RocketPy and Dymos/OpenMDAO remain future backend adapters once the launch product contracts are
stable.

## Setup

```bash
python -m pip install -e '.[dev]'
```

Optional Orekit wrapper smoke support:

```bash
python -m pip install -e '.[orekit]'
astro orekit-smoke
```

If `orekit-jpype` is not installed, `astro orekit-smoke` exits nonzero with structured JSON explaining that the optional wrapper is unavailable.

## Commands

```bash
astro validate examples/scenarios/leo_two_body.yaml
astro propagate examples/scenarios/leo_two_body.yaml --backend local --output trajectory.json
astro launch examples/launch/vertical_two_stage.yaml --backend local --output launch.json
astro launch examples/launch/pitch_program_two_stage.yaml --backend local --output pitch_launch.json
astro sweep-launch-pitch examples/launch/pitch_program_two_stage.yaml --point-index 3 --pitch-deg-values 10,20,30 --output pitch_sweep.json
astro tune-launch-pitch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --initial-span-deg 10 --iterations 2 --output pitch_tuning.json --tuned-scenario-output tuned_pitch_program.yaml
astro report-tuned-launch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --initial-span-deg 10 --iterations 2 --orbit-duration-s 600 --orbit-step-s 60 --output tuned_launch_report.json
astro handoff-launch launch.json --output insertion.yaml --duration-s 600 --step-s 60
astro propagate insertion.yaml --backend local --output insertion_trajectory.json
astro synth-measurements examples/scenarios/leo_two_station_od.yaml --output measurements.json
astro export-measurements measurements.json --format csv --output measurements.csv
astro export-measurements measurements.json --format tdm --output measurements.tdm
astro estimate examples/scenarios/leo_two_body.yaml --output estimate.json
astro estimate-measurements examples/scenarios/leo_two_station_od.yaml measurements.json --output estimate.json
astro orekit-smoke
```

`astro estimate` is an MVP synthetic demonstration workflow. It keeps the source scenario unchanged,
adds in-memory demo geometry for observability, generates synthetic measurements, perturbs the
initial state as an estimate seed, and records that provenance in the output metadata.

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

`astro report-tuned-launch` runs the current local end-to-end launch analysis: tune two pitch knots,
propagate the tuned ascent, hand off insertion to an orbit scenario, propagate a short orbital arc,
and write one JSON product with the component products plus insertion and short-arc target metrics.
It is a deterministic report over local baselines, not a substitute for high-fidelity ascent design.

`astro handoff-launch` converts a launch trajectory product into a normal orbital propagation
scenario initialized from `LaunchTrajectory.insertion_state`. The generated YAML is intentionally
plain `Scenario` input, so the next step is the existing `astro propagate` command rather than a
special launch-aware propagation path.

`astro estimate-measurements` is the explicit ingest workflow. It loads a scenario plus a JSON,
CSV, or CCSDS Tracking Data Message (TDM) measurement file, then estimates from the
caller-provided station geometry and measurement records without adding demo geometry. JSON inputs
match the output of `astro synth-measurements`; CSV and TDM inputs are auto-detected by `.csv` and
`.tdm` extensions or can be forced with `--format csv` / `--format tdm`.

`astro export-measurements` converts suite JSON measurement files into JSON, CSV, or TDM products.
The example files under `examples/measurements/` are generated from `leo_two_station_od.yaml` and
cover all three ingest/export formats.

CSV inputs use one row per measurement with these required columns:

```csv
scenario_id,measurement_type,epoch,observer,observed_object,value,sigma,units
```

The optional `metadata_json` column can carry a JSON object for row-level metadata. The
`leo_two_station_od.yaml` example includes two stations because the one-station propagation example
is intentionally under-observed for six-state OD.

TDM ingest currently supports KVN-formatted sequential segments with `TIME_SYSTEM = UTC`,
`PARTICIPANT_n`, `PATH`, `RANGE` in `km`, and `DOPPLER_INSTANTANEOUS` or `DOPPLER_INTEGRATED`
mapped to range-rate measurements in `km/s`. TDM does not provide the suite's scenario identifier
or estimator sigmas directly, so an optional segment-level `SCENARIO_ID` extension is checked when
present, and the parser uses default sigmas of `0.01 km` for range and `1e-5 km/s` for range-rate.

## Verification

```bash
python -m pytest -v
python -m ruff check .
python -m mypy
```
