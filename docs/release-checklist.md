# Release Checklist

Date: 2026-06-15

Use this checklist before tagging or publishing a release candidate.

## Required Local Gates

- [ ] `python -m pytest -q`
- [ ] `python -m ruff check .`
- [ ] `python -m mypy`
- [ ] `astro validate examples/scenarios/leo_two_body.yaml`
- [ ] `astro propagate examples/scenarios/leo_two_body.yaml --backend local --output /tmp/astro-local-trajectory.json`
- [ ] `astro export-trajectory /tmp/astro-local-trajectory.json --format csv --output /tmp/astro-local-trajectory.csv`
- [ ] `astro synth-measurements examples/scenarios/leo_two_station_od.yaml --backend local --output /tmp/astro-measurements.json`
- [ ] `astro estimate-measurements examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --backend local --output /tmp/astro-local-estimate.json`
- [ ] `astro launch examples/launch/pitch_program_two_stage.yaml --backend local --output /tmp/astro-launch.json`
- [ ] `astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend local --point-indices 2,3 --iterations 1 --output /tmp/astro-optimized-launch.json`
- [ ] `astro research-propagate examples/scenarios/leo_two_body.yaml --backend local --cases 2 --position-sigma-km 0.01 --velocity-sigma-km-s 0.000001 --seed 7 --output /tmp/astro-research.json`

## Optional Backend Gates

Run when the matching runtime is expected to be present:

- [ ] `astro orekit-smoke`
- [ ] `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_two_body_matches_local_reference -v`
- [ ] `astro rocketpy-smoke`
- [ ] `astro dymos-smoke`
- [ ] `astro tudat-smoke`
- [ ] `astro jax-smoke`

If an optional runtime is intentionally absent, capture the structured unavailable JSON and confirm
the message is actionable.

## Documentation Gates

- [ ] README current-scope and command list match implemented behavior.
- [ ] `docs/validation-matrix.md` reflects current command names and tolerances.
- [ ] `docs/backend-installation.md` documents every optional extra and non-pip install caveat.
- [ ] Roadmap plan statuses distinguish implemented product boundaries from live backend work that
  still requires external configuration.

## Packaging Gate

- [ ] Build succeeds with `python -m build` when build tooling is installed.
- [ ] Wheel metadata includes optional extras: `dev`, `orekit`, `launch`, `optimization`, and
  `research`.
