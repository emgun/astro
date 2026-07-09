# Constellation Coverage Maps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic target-grid sensor coverage summaries to the constellation digital twin.

**Architecture:** Extend the existing constellation twin models and runner instead of adding a separate command. Reuse member `DigitalTwinResult.geometry` samples, then aggregate per-target visibility intervals into suite-owned coverage-map summaries and design margins.

**Tech Stack:** Python, Pydantic models, Typer CLI, pytest, ruff, mypy.

---

### Task 1: Scenario And Result Models

**Files:**
- Modify: `src/astro_twin/constellation_models.py`
- Test: `tests/astro_twin/test_constellation_models.py`

- [x] Add coverage sensor, target, map config models, and coverage-map summary result models.
- [x] Add `coverage_maps` to `ConstellationTwinScenario`.
- [x] Add `coverage_map_summaries` to `ConstellationTwinResult`.
- [x] Reject duplicate coverage map names and duplicate target names.
- [x] Verify with `python -m pytest tests/astro_twin/test_constellation_models.py -q`.

### Task 2: Coverage Aggregation And Margins

**Files:**
- Modify: `src/astro_twin/constellation.py`
- Test: `tests/astro_twin/test_constellation_aggregation.py`

- [x] Add `aggregate_coverage_maps` over member geometry samples.
- [x] Evaluate target visibility using target elevation, optional range, and nadir off-boresight angle.
- [x] Aggregate per-target coverage duration, gaps, simultaneous spacecraft, and map-level statistics.
- [x] Add coverage-map minimum-fraction and maximum-gap margins to the fleet report.
- [x] Verify with `python -m pytest tests/astro_twin/test_constellation_aggregation.py -q`.

### Task 3: Runner, IO, CLI, And Example

**Files:**
- Modify: `src/astro_twin/constellation.py`
- Modify: `src/astro_twin/constellation_io.py`
- Modify: `examples/twin/constellation_leo_observers.yaml`
- Test: `tests/astro_twin/test_constellation_runner.py`
- Test: `tests/astro_twin/test_constellation_io.py`
- Test: `tests/astro_cli/test_cli.py`

- [x] Run coverage maps inside `run_constellation_twin`.
- [x] Include coverage map count and minimum target coverage in text summaries.
- [x] Add a two-target equatorial coverage-map reference to the checked constellation example.
- [x] Verify with the focused constellation/CLI test slice.

### Task 4: Documentation And Release Evidence

**Files:**
- Modify: `docs/digital-twin.md`
- Modify: `docs/validation-matrix.md`
- Modify: `docs/current-state.md`

- [x] Document scenario shape, result evidence, and claim boundaries.
- [x] Record checked CLI output and target-grid metrics.
- [x] Run final focused and broad gates, then update final gate counts.
