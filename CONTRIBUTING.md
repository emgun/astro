# Contributing

Astro Suite is a Python flight-dynamics project focused on deterministic, auditable mission-analysis
workflows. Contributions should preserve the current boundary: AI-facing interfaces may plan and
explain, but suite-owned models, command registries, validators, and backend adapters compute and
verify artifacts.

## Setup

```bash
python -m pip install -e '.[dev]'
```

Optional backend extras are not required for the default test suite:

```bash
python -m pip install -e '.[orekit]'
python -m pip install -e '.[launch,optimization]'
python -m pip install -e '.[research]'
```

## Local Checks

Run these before opening a pull request:

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
```

Live Orekit, TudatPy, RocketPy, Dymos/OpenMDAO, and JAX checks are optional runtime gates. Do not
promote optional-backend behavior to a required local gate unless the default development
environment can run it without extra services or large external data.

## Contribution Guidelines

- Keep public claims precise. This project provides deterministic analysis and adapter boundaries;
  it does not claim flight qualification.
- Add tests for behavior changes, especially CLI behavior, artifact schemas, validators, and
  backend-dispatch boundaries.
- Fail closed for unsupported backends, scenario features, and assistant prompts.
- Preserve reproducibility: commands should write explicit artifacts and diagnostics instead of
  relying on hidden state.
- Keep optional runtime failures actionable and structured.
