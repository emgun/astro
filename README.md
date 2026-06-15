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

`astro estimate-measurements` is the explicit ingest workflow. It loads a scenario plus a JSON
measurement file, requires matching `scenario_id` values, and estimates from the caller-provided
station geometry and measurement records without adding demo geometry. Its measurement-file schema
matches the output of `astro synth-measurements`. The `leo_two_station_od.yaml` example includes
two stations because the one-station propagation example is intentionally under-observed for
six-state OD.

## Verification

```bash
python -m pytest -v
python -m ruff check .
python -m mypy
```
