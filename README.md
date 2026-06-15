# Astro Suite

Astro Suite is a Python flight dynamics project for scenario validation, local reference propagation, synthetic orbit-determination measurements, batch OD, and backend adapters.

## Current Scope

The current implementation slice covers:

- Pydantic scenario validation from YAML.
- Local two-body and J2 reference propagation with deterministic provenance metadata.
- Synthetic range and range-rate measurement generation.
- Local SciPy batch least-squares orbit determination with rank and convergence checks.
- CLI workflows for validation, propagation, synthetic measurements, synthetic OD, and
  measurement-file OD ingest.
- Optional Orekit Python-wrapper smoke checks through `orekit_jpype`.

Launch/ascent is included in the design specs and will be implemented after the shared scenario, trajectory, and backend adapter spine is stable.

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
astro synth-measurements examples/scenarios/leo_two_station_od.yaml --output measurements.json
astro estimate examples/scenarios/leo_two_body.yaml --output estimate.json
astro estimate-measurements examples/scenarios/leo_two_station_od.yaml measurements.json --output estimate.json
astro orekit-smoke
```

`astro estimate` is an MVP synthetic demonstration workflow. It keeps the source scenario unchanged,
adds in-memory demo geometry for observability, generates synthetic measurements, perturbs the
initial state as an estimate seed, and records that provenance in the output metadata.

`astro estimate-measurements` is the explicit ingest workflow. It loads a scenario plus a JSON,
CSV, or CCSDS Tracking Data Message (TDM) measurement file, then estimates from the
caller-provided station geometry and measurement records without adding demo geometry. JSON inputs
match the output of `astro synth-measurements`; CSV and TDM inputs are auto-detected by `.csv` and
`.tdm` extensions or can be forced with `--format csv` / `--format tdm`.

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
