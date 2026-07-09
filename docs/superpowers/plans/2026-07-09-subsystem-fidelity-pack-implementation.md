# Subsystem Fidelity Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic power, thermal, ADCS, and mass-budget fidelity evidence to the Astro Suite digital twin.

**Architecture:** Extend the existing `astro_twin` model/config objects and subsystem calculators. Keep public outputs in suite-owned Pydantic models, reuse the existing runner, and add optional fields with defaults so current scenarios remain valid.

**Tech Stack:** Python, Pydantic models, Typer CLI, pytest, ruff, mypy.

---

### Task 1: Power Schedule And Battery Efficiency

**Files:**
- Modify: `src/astro_twin/models.py`
- Modify: `src/astro_twin/power.py`
- Modify: `src/astro_twin/runner.py`
- Test: `tests/astro_twin/test_models.py`
- Test: `tests/astro_twin/test_power.py`
- Test: `tests/astro_twin/test_runner.py`

- [x] **Step 1: Write failing tests**
  - Add a model test that `DigitalTwinScenario` accepts `power_loads` with `name`, `start_s`, `end_s`, and `additional_load_w`.
  - Add a model test that inverted load windows fail validation.
  - Add a power test that a scheduled load increases `PowerSample.load_w`, records `scheduled_load_w`, and applies charge/discharge efficiency to `battery_energy_wh`.
- [x] **Step 2: Verify red**
  - Run `python -m pytest tests/astro_twin/test_models.py tests/astro_twin/test_power.py -q`.
  - Expected: imports or validation fail because `PowerLoadSchedule` and sample evidence fields do not exist yet.
- [x] **Step 3: Implement**
  - Add `PowerLoadSchedule` to `models.py`.
  - Add `battery_charge_efficiency` and `battery_discharge_efficiency` defaults to `PowerConfig`.
  - Add `scheduled_load_w` and `battery_energy_wh` to `PowerSample`.
  - Add `power_loads` to `DigitalTwinScenario`.
  - Update `compute_power_timeline` and the runner to apply active scheduled loads.
- [x] **Step 4: Verify green**
  - Run `python -m pytest tests/astro_twin/test_models.py tests/astro_twin/test_power.py tests/astro_twin/test_runner.py -q`.

### Task 2: Thermal Flux And Mode Heat Scaling

**Files:**
- Modify: `src/astro_twin/models.py`
- Modify: `src/astro_twin/thermal.py`
- Test: `tests/astro_twin/test_models.py`
- Test: `tests/astro_twin/test_thermal.py`

- [x] **Step 1: Write failing tests**
  - Add a thermal-node model test for `albedo_flux_w_m2`, `planet_ir_flux_w_m2`, and `mode_internal_heat_scale`.
  - Add a thermal test that sunlit albedo, always-on planetary IR, and payload-mode heat scale increase the heat balance.
- [x] **Step 2: Verify red**
  - Run `python -m pytest tests/astro_twin/test_models.py tests/astro_twin/test_thermal.py -q`.
- [x] **Step 3: Implement**
  - Add optional flux and mode-scaling fields to `ThermalNodeConfig`.
  - Add `node_heat_balance_w` to `ThermalSample`.
  - Update `compute_thermal_timeline` to include direct solar, albedo, planetary IR, internal heat scaling, and radiated heat in a per-node heat balance.
- [x] **Step 4: Verify green**
  - Run `python -m pytest tests/astro_twin/test_models.py tests/astro_twin/test_thermal.py -q`.

### Task 3: ADCS Slew And Actuator Utilization

**Files:**
- Modify: `src/astro_twin/models.py`
- Modify: `src/astro_twin/adcs.py`
- Modify: `src/astro_twin/margins.py`
- Test: `tests/astro_twin/test_adcs.py`
- Test: `tests/astro_twin/test_margins.py`

- [x] **Step 1: Write failing tests**
  - Add an ADCS test that configured slew-rate limits produce `slew_rate_margin_deg_s`.
  - Add an ADCS test that torque demand produces `actuator_utilization_fraction` and utilization margin evidence.
  - Add a margin test that `slew_rate_margin_deg_s` and `actuator_utilization_margin_fraction` appear in the margin report.
- [x] **Step 2: Verify red**
  - Run `python -m pytest tests/astro_twin/test_adcs.py tests/astro_twin/test_margins.py -q`.
- [x] **Step 3: Implement**
  - Add defaulted slew and utilization fields to `ADCSConfig`.
  - Add slew/utilization fields to `ADCSSample`.
  - Update `compute_adcs_timeline` and `build_margin_report`.
- [x] **Step 4: Verify green**
  - Run `python -m pytest tests/astro_twin/test_adcs.py tests/astro_twin/test_margins.py -q`.

### Task 4: Mass Budget Rollup

**Files:**
- Create: `src/astro_twin/mass.py`
- Modify: `src/astro_twin/models.py`
- Modify: `src/astro_twin/runner.py`
- Modify: `src/astro_twin/margins.py`
- Test: `tests/astro_twin/test_models.py`
- Create: `tests/astro_twin/test_mass.py`
- Test: `tests/astro_twin/test_margins.py`

- [x] **Step 1: Write failing tests**
  - Add model tests for mass-budget item validation.
  - Add `test_mass.py` covering itemized base mass, contingency, rollup total, and dry-plus-payload consistency margin.
  - Add margin tests for `mass_budget_rollup_margin_kg`.
- [x] **Step 2: Verify red**
  - Run `python -m pytest tests/astro_twin/test_models.py tests/astro_twin/test_mass.py tests/astro_twin/test_margins.py -q`.
- [x] **Step 3: Implement**
  - Add `MassBudgetItemConfig` and `MassBudgetSummary`.
  - Add `mass_budget_items` to `SpacecraftBusConfig` and `mass_budget` to `DigitalTwinResult`.
  - Implement `build_mass_budget_summary`.
  - Pass mass budget summaries into margin reporting.
- [x] **Step 4: Verify green**
  - Run `python -m pytest tests/astro_twin/test_models.py tests/astro_twin/test_mass.py tests/astro_twin/test_margins.py tests/astro_twin/test_runner.py -q`.

### Task 5: Example, Docs, And Release Gates

**Files:**
- Modify: `examples/twin/leo_observer.yaml`
- Modify: `docs/digital-twin.md`
- Modify: `docs/validation-matrix.md`
- Modify: `docs/current-state.md`
- Modify: `README.md`
- Test: `tests/astro_cli/test_cli.py`

- [x] **Step 1: Update scenario/docs**
  - Add scheduled loads, thermal flux/scaling, ADCS slew/utilization assumptions, and itemized mass budget items to `examples/twin/leo_observer.yaml`.
  - Document the fields and claim boundaries in `docs/digital-twin.md`.
  - Update the validation matrix and durable current-state evidence after verification.
- [x] **Step 2: Verify public command**
  - Run `astro run-twin examples/twin/leo_observer.yaml --output /tmp/astro-subsystem-fidelity-twin.json --summary-output /tmp/astro-subsystem-fidelity-twin.txt`.
  - Inspect the JSON for `mass_budget`, `scheduled_load_w`, `node_heat_balance_w`, `slew_rate_margin_deg_s`, and `actuator_utilization_fraction`.
- [x] **Step 3: Run gates**
  - Run `python -m pytest tests/astro_twin -q`.
  - Run `python -m pytest tests/astro_cli/test_cli.py::test_run_twin_command_writes_json_and_summary -q`.
  - Run `python -m ruff check .`.
  - Run `python -m mypy`.
  - Run `python -m pytest -q`.
  - Run `git diff --check`.
  - Run `python -m pytest tests/test_packaging.py -q`.
  - Run `python -m build`.
- [x] **Step 4: Commit and publish**
  - Commit as `Add subsystem fidelity pack`.
  - Push `codex/subsystem-fidelity-pack`.
  - Open a PR against `main`.
