# Integrated Digital Twin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an integrated single-spacecraft digital twin workflow that produces orbit, power, thermal, ADCS, coverage, link budget, mass, and design-margin evidence from one checked-in mission scenario.

**Architecture:** Add a new `astro_twin` package that owns suite-level twin schemas, deterministic subsystem screening models, orchestration, IO, and report formatting. The first CLI command, `astro run-twin`, reuses existing local orbit propagation, then computes time-indexed subsystem products and aggregates a design margin report. Constellation support is deferred until the single-spacecraft product is stable.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, Typer, PyYAML, pytest, existing `astro_core` and `astro_dynamics` products.

---

## Context

Design spec: `docs/superpowers/specs/2026-07-08-integrated-digital-twin-design.md`

Branch target: create from `main`.

First public slice:

```bash
astro run-twin examples/twin/leo_observer.yaml \
  --output /tmp/astro-twin-result.json \
  --summary-output /tmp/astro-twin-summary.txt
```

Non-goals for this branch:

- No constellation aggregation.
- No optional backend requirement.
- No flight-qualification, thermal-certification, RF-certification, or operational-readiness claims.
- No external data provider calls.
- No optimizer or autonomous scheduler.

## File Map

- Create `src/astro_twin/__init__.py`: package exports.
- Create `src/astro_twin/models.py`: Pydantic scenario, subsystem, timeline, result, and margin models.
- Create `src/astro_twin/io.py`: load `DigitalTwinScenario` from YAML/JSON and write result text/JSON.
- Create `src/astro_twin/geometry.py`: derive timeline geometry from `Trajectory`.
- Create `src/astro_twin/power.py`: compute solar generation, loads, battery state of charge, and margins.
- Create `src/astro_twin/thermal.py`: compute lumped-node thermal screening.
- Create `src/astro_twin/adcs.py`: compute pointing, slew, and torque margin diagnostics.
- Create `src/astro_twin/coverage.py`: derive access windows from geometry samples.
- Create `src/astro_twin/link_budget.py`: compute C/N0, Eb/N0, data volume, and link margins.
- Create `src/astro_twin/margins.py`: aggregate subsystem checks into `DesignMarginReport`.
- Create `src/astro_twin/runner.py`: run the integrated workflow.
- Modify `src/astro_cli/main.py`: add `astro run-twin`.
- Modify `pyproject.toml`: include `astro_twin` in package and mypy lists.
- Create `examples/twin/leo_observer.yaml`: public reference twin scenario.
- Create `tests/astro_twin/`: focused tests for each module and end-to-end CLI behavior.
- Modify `docs/validation-matrix.md`: add the digital twin required gate.
- Modify `README.md` or `docs/assistant-workflows.md` only if the new workflow needs public navigation.

## Model Skeleton

Use these model names consistently across tasks.

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, FiniteFloat

from astro_core.models import AstroModel


class TwinMarginStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class MissionMode(StrEnum):
    IDLE = "idle"
    PAYLOAD = "payload"
    DOWNLINK = "downlink"


class MissionModeSchedule(AstroModel):
    mode: MissionMode
    start_s: FiniteFloat = Field(ge=0.0)
    end_s: FiniteFloat = Field(gt=0.0)


class SpacecraftBusConfig(AstroModel):
    name: str = Field(min_length=1)
    dry_mass_kg: FiniteFloat = Field(gt=0.0)
    payload_mass_kg: FiniteFloat = Field(ge=0.0)
    propellant_mass_kg: FiniteFloat = Field(ge=0.0, default=0.0)
    mass_margin_fraction_required: FiniteFloat = Field(ge=0.0, default=0.2)


class PowerConfig(AstroModel):
    solar_array_area_m2: FiniteFloat = Field(gt=0.0)
    solar_array_efficiency: FiniteFloat = Field(gt=0.0, le=1.0)
    battery_capacity_wh: FiniteFloat = Field(gt=0.0)
    initial_battery_soc_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    minimum_battery_soc_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    idle_load_w: FiniteFloat = Field(ge=0.0)
    payload_load_w: FiniteFloat = Field(ge=0.0)
    downlink_load_w: FiniteFloat = Field(ge=0.0)


class ThermalNodeConfig(AstroModel):
    name: str = Field(min_length=1)
    thermal_mass_j_k: FiniteFloat = Field(gt=0.0)
    radiator_area_m2: FiniteFloat = Field(gt=0.0)
    absorptivity: FiniteFloat = Field(ge=0.0, le=1.0)
    emissivity: FiniteFloat = Field(gt=0.0, le=1.0)
    initial_temperature_k: FiniteFloat = Field(gt=0.0)
    minimum_temperature_k: FiniteFloat = Field(gt=0.0)
    maximum_temperature_k: FiniteFloat = Field(gt=0.0)
    internal_heat_fraction: FiniteFloat = Field(ge=0.0, le=1.0)


class ADCSConfig(AstroModel):
    pointing_mode: Literal["nadir", "inertial", "ground_station_track"]
    max_pointing_error_deg: FiniteFloat = Field(ge=0.0)
    pointing_requirement_deg: FiniteFloat = Field(gt=0.0)
    max_torque_n_m: FiniteFloat = Field(gt=0.0)
    required_slew_torque_n_m: FiniteFloat = Field(ge=0.0)


class GroundSiteConfig(AstroModel):
    name: str = Field(min_length=1)
    latitude_deg: FiniteFloat = Field(ge=-90.0, le=90.0)
    longitude_deg: FiniteFloat = Field(ge=-180.0, le=180.0)
    altitude_m: FiniteFloat = 0.0
    minimum_elevation_deg: FiniteFloat = Field(ge=0.0, le=90.0)


class LinkBudgetConfig(AstroModel):
    name: str = Field(min_length=1)
    ground_site: str = Field(min_length=1)
    frequency_ghz: FiniteFloat = Field(gt=0.0)
    eirp_dbw: FiniteFloat
    receiver_g_over_t_db_k: FiniteFloat
    data_rate_bps: FiniteFloat = Field(gt=0.0)
    required_ebn0_db: FiniteFloat
    implementation_loss_db: FiniteFloat = Field(ge=0.0, default=2.0)


class DigitalTwinScenario(AstroModel):
    scenario_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    orbit_scenario: str = Field(min_length=1)
    spacecraft: SpacecraftBusConfig
    power: PowerConfig
    thermal_nodes: tuple[ThermalNodeConfig, ...] = Field(min_length=1)
    adcs: ADCSConfig
    ground_sites: tuple[GroundSiteConfig, ...] = Field(min_length=1)
    links: tuple[LinkBudgetConfig, ...] = Field(min_length=1)
    mode_schedule: tuple[MissionModeSchedule, ...] = Field(default_factory=tuple)
```

## Task 1: Package Scaffold

**Files:**
- Create: `src/astro_twin/__init__.py`
- Create: `tests/astro_twin/test_imports.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing import/package test**

```python
def test_astro_twin_public_imports() -> None:
    import astro_twin

    assert astro_twin.__all__ == [
        "DigitalTwinScenario",
        "DigitalTwinResult",
        "run_digital_twin",
    ]
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/astro_twin/test_imports.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'astro_twin'`.

- [ ] **Step 3: Add the package export**

Create `src/astro_twin/__init__.py`:

```python
"""Integrated spacecraft digital twin products for Astro Suite."""

from astro_twin.models import DigitalTwinResult, DigitalTwinScenario
from astro_twin.runner import run_digital_twin

__all__ = [
    "DigitalTwinScenario",
    "DigitalTwinResult",
    "run_digital_twin",
]
```

- [ ] **Step 4: Add the package to build/type config**

Modify `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = [
  "src/astro_core",
  "src/astro_dynamics",
  "src/astro_launch",
  "src/astro_od",
  "src/astro_backends",
  "src/astro_cli",
  "src/astro_assistant",
  "src/astro_twin",
]

[tool.mypy]
packages = [
  "astro_core",
  "astro_dynamics",
  "astro_launch",
  "astro_od",
  "astro_backends",
  "astro_cli",
  "astro_assistant",
  "astro_twin",
]
```

- [ ] **Step 5: Run the import test**

Run: `python -m pytest tests/astro_twin/test_imports.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/astro_twin/__init__.py tests/astro_twin/test_imports.py
git commit -m "Add digital twin package scaffold"
```

## Task 2: Scenario And Result Models

**Files:**
- Create: `src/astro_twin/models.py`
- Create: `tests/astro_twin/test_models.py`

- [ ] **Step 1: Write failing model validation tests**

```python
import pytest
from pydantic import ValidationError

from astro_twin.models import (
    ADCSConfig,
    DigitalTwinScenario,
    GroundSiteConfig,
    LinkBudgetConfig,
    MissionMode,
    MissionModeSchedule,
    PowerConfig,
    SpacecraftBusConfig,
    ThermalNodeConfig,
)


def _valid_scenario() -> DigitalTwinScenario:
    return DigitalTwinScenario(
        scenario_id="leo-observer",
        orbit_scenario="examples/scenarios/leo_two_body.yaml",
        spacecraft=SpacecraftBusConfig(
            name="ObserverSat",
            dry_mass_kg=120.0,
            payload_mass_kg=25.0,
            propellant_mass_kg=5.0,
            mass_margin_fraction_required=0.2,
        ),
        power=PowerConfig(
            solar_array_area_m2=2.4,
            solar_array_efficiency=0.29,
            battery_capacity_wh=1200.0,
            initial_battery_soc_fraction=0.85,
            minimum_battery_soc_fraction=0.35,
            idle_load_w=120.0,
            payload_load_w=260.0,
            downlink_load_w=360.0,
        ),
        thermal_nodes=(
            ThermalNodeConfig(
                name="bus",
                thermal_mass_j_k=45000.0,
                radiator_area_m2=1.0,
                absorptivity=0.55,
                emissivity=0.78,
                initial_temperature_k=293.0,
                minimum_temperature_k=273.0,
                maximum_temperature_k=313.0,
                internal_heat_fraction=0.45,
            ),
        ),
        adcs=ADCSConfig(
            pointing_mode="nadir",
            max_pointing_error_deg=0.08,
            pointing_requirement_deg=0.15,
            max_torque_n_m=0.08,
            required_slew_torque_n_m=0.03,
        ),
        ground_sites=(
            GroundSiteConfig(
                name="goldstone",
                latitude_deg=35.2472,
                longitude_deg=-116.7933,
                altitude_m=1000.0,
                minimum_elevation_deg=10.0,
            ),
        ),
        links=(
            LinkBudgetConfig(
                name="xband-downlink",
                ground_site="goldstone",
                frequency_ghz=8.4,
                eirp_dbw=18.0,
                receiver_g_over_t_db_k=22.0,
                data_rate_bps=2_000_000.0,
                required_ebn0_db=6.5,
                implementation_loss_db=2.0,
            ),
        ),
        mode_schedule=(
            MissionModeSchedule(mode=MissionMode.PAYLOAD, start_s=600.0, end_s=1800.0),
            MissionModeSchedule(mode=MissionMode.DOWNLINK, start_s=2400.0, end_s=3000.0),
        ),
    )


def test_digital_twin_scenario_accepts_valid_config() -> None:
    scenario = _valid_scenario()

    assert scenario.scenario_id == "leo-observer"
    assert scenario.links[0].ground_site == "goldstone"


def test_digital_twin_scenario_rejects_unknown_link_site() -> None:
    scenario = _valid_scenario()
    payload = scenario.model_dump()
    payload["links"][0]["ground_site"] = "missing"

    with pytest.raises(ValidationError, match="link ground_site must name a configured ground site"):
        DigitalTwinScenario.model_validate(payload)


def test_thermal_node_rejects_inverted_temperature_limits() -> None:
    with pytest.raises(ValidationError, match="maximum_temperature_k must exceed minimum_temperature_k"):
        ThermalNodeConfig(
            name="battery",
            thermal_mass_j_k=10000.0,
            radiator_area_m2=0.4,
            absorptivity=0.5,
            emissivity=0.8,
            initial_temperature_k=293.0,
            minimum_temperature_k=300.0,
            maximum_temperature_k=290.0,
            internal_heat_fraction=0.2,
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `python -m pytest tests/astro_twin/test_models.py -q`

Expected: FAIL because `astro_twin.models` does not exist.

- [ ] **Step 3: Implement `models.py`**

Create `src/astro_twin/models.py` using the model skeleton above, then add:

```python
class TimelineGeometrySample(AstroModel):
    epoch: datetime
    elapsed_s: FiniteFloat = Field(ge=0.0)
    position_km: tuple[float, float, float]
    altitude_km: FiniteFloat
    sunlit: bool


class PowerSample(AstroModel):
    elapsed_s: FiniteFloat = Field(ge=0.0)
    mode: MissionMode
    generated_w: FiniteFloat = Field(ge=0.0)
    load_w: FiniteFloat = Field(ge=0.0)
    battery_soc_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    net_power_w: float


class ThermalSample(AstroModel):
    elapsed_s: FiniteFloat = Field(ge=0.0)
    node_temperatures_k: dict[str, float]


class ADCSSample(AstroModel):
    elapsed_s: FiniteFloat = Field(ge=0.0)
    pointing_error_deg: FiniteFloat = Field(ge=0.0)
    pointing_margin_deg: float
    torque_margin_n_m: float


class AccessWindow(AstroModel):
    ground_site: str
    start_s: FiniteFloat = Field(ge=0.0)
    end_s: FiniteFloat = Field(gt=0.0)
    duration_s: FiniteFloat = Field(gt=0.0)
    max_elevation_deg: FiniteFloat
    min_range_km: FiniteFloat = Field(gt=0.0)


class LinkBudgetWindow(AstroModel):
    link_name: str
    ground_site: str
    start_s: FiniteFloat = Field(ge=0.0)
    end_s: FiniteFloat = Field(gt=0.0)
    duration_s: FiniteFloat = Field(gt=0.0)
    worst_ebn0_margin_db: float
    data_volume_mbit: FiniteFloat = Field(ge=0.0)


class DesignMargin(AstroModel):
    name: str
    value: float
    threshold: float
    margin: float
    status: TwinMarginStatus


class DesignMarginReport(AstroModel):
    margins: tuple[DesignMargin, ...]
    limiting_margin: DesignMargin


class DigitalTwinResult(AstroModel):
    scenario_id: str
    workflow: str = "integrated_digital_twin_v1"
    geometry: tuple[TimelineGeometrySample, ...]
    power: tuple[PowerSample, ...]
    thermal: tuple[ThermalSample, ...]
    adcs: tuple[ADCSSample, ...]
    access_windows: tuple[AccessWindow, ...]
    link_windows: tuple[LinkBudgetWindow, ...]
    margin_report: DesignMarginReport
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
```

Add `model_validator` methods:

- `MissionModeSchedule.end_s` must exceed `start_s`.
- `ThermalNodeConfig.maximum_temperature_k` must exceed `minimum_temperature_k`.
- `DigitalTwinScenario.links[*].ground_site` must exist in `ground_sites`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/astro_twin/test_models.py tests/astro_twin/test_imports.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_twin/models.py tests/astro_twin/test_models.py src/astro_twin/__init__.py
git commit -m "Add digital twin domain models"
```

## Task 3: IO And Reference Example

**Files:**
- Create: `src/astro_twin/io.py`
- Create: `examples/twin/leo_observer.yaml`
- Create: `tests/astro_twin/test_io.py`

- [ ] **Step 1: Write failing IO tests**

```python
from pathlib import Path

from astro_twin.io import load_twin_scenario


def test_load_twin_scenario_reads_reference_example() -> None:
    scenario = load_twin_scenario("examples/twin/leo_observer.yaml")

    assert scenario.scenario_id == "leo-observer"
    assert scenario.orbit_scenario == "examples/scenarios/leo_two_body.yaml"
    assert scenario.power.battery_capacity_wh == 1200.0


def test_load_twin_scenario_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    try:
        load_twin_scenario(path)
    except Exception as exc:
        assert "must contain a mapping" in str(exc)
    else:
        raise AssertionError("expected invalid scenario error")
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/astro_twin/test_io.py -q`

Expected: FAIL because `astro_twin.io` and the example do not exist.

- [ ] **Step 3: Add the reference example**

Create `examples/twin/leo_observer.yaml` with the same values used in Task 2 tests plus:

```yaml
scenario_id: leo-observer
orbit_scenario: examples/scenarios/leo_two_body.yaml
spacecraft:
  name: ObserverSat
  dry_mass_kg: 120.0
  payload_mass_kg: 25.0
  propellant_mass_kg: 5.0
  mass_margin_fraction_required: 0.2
power:
  solar_array_area_m2: 2.4
  solar_array_efficiency: 0.29
  battery_capacity_wh: 1200.0
  initial_battery_soc_fraction: 0.85
  minimum_battery_soc_fraction: 0.35
  idle_load_w: 120.0
  payload_load_w: 260.0
  downlink_load_w: 360.0
thermal_nodes:
  - name: bus
    thermal_mass_j_k: 45000.0
    radiator_area_m2: 1.0
    absorptivity: 0.55
    emissivity: 0.78
    initial_temperature_k: 293.0
    minimum_temperature_k: 273.0
    maximum_temperature_k: 313.0
    internal_heat_fraction: 0.45
adcs:
  pointing_mode: nadir
  max_pointing_error_deg: 0.08
  pointing_requirement_deg: 0.15
  max_torque_n_m: 0.08
  required_slew_torque_n_m: 0.03
ground_sites:
  - name: goldstone
    latitude_deg: 35.2472
    longitude_deg: -116.7933
    altitude_m: 1000.0
    minimum_elevation_deg: 10.0
links:
  - name: xband-downlink
    ground_site: goldstone
    frequency_ghz: 8.4
    eirp_dbw: 18.0
    receiver_g_over_t_db_k: 22.0
    data_rate_bps: 2000000.0
    required_ebn0_db: 6.5
    implementation_loss_db: 2.0
mode_schedule:
  - mode: payload
    start_s: 600.0
    end_s: 1800.0
  - mode: downlink
    start_s: 2400.0
    end_s: 3000.0
```

- [ ] **Step 4: Add IO implementation**

Create `src/astro_twin/io.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_twin.models import DigitalTwinResult, DigitalTwinScenario


def load_twin_scenario(path: Path | str) -> DigitalTwinScenario:
    scenario_path = Path(path)
    try:
        raw: Any = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(f"Could not read twin scenario {scenario_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(f"Could not parse twin scenario {scenario_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(f"Twin scenario file {scenario_path} must contain a mapping")
    try:
        return DigitalTwinScenario.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(f"Twin scenario file {scenario_path} is invalid: {exc}") from exc


def write_twin_result(path: Path | str, result: DigitalTwinResult) -> None:
    Path(path).write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_twin_result(path: Path | str) -> DigitalTwinResult:
    result_path = Path(path)
    try:
        raw: Any = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(f"Could not read twin result {result_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InvalidScenarioError(f"Could not parse twin result {result_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(f"Twin result file {result_path} must contain a JSON object")
    return DigitalTwinResult.model_validate(raw)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/astro_twin/test_io.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/astro_twin/io.py examples/twin/leo_observer.yaml tests/astro_twin/test_io.py
git commit -m "Add digital twin scenario IO"
```

## Task 4: Geometry Timeline

**Files:**
- Create: `src/astro_twin/geometry.py`
- Create: `tests/astro_twin/test_geometry.py`

- [ ] **Step 1: Write failing geometry tests**

```python
from datetime import UTC, datetime

from astro_core.models import (
    Body,
    CartesianState,
    Frame,
    OrbitRepresentation,
    OrbitState,
    TimeScale,
    Trajectory,
    TrajectorySample,
)
from astro_twin.geometry import build_geometry_timeline


def test_build_geometry_timeline_marks_sunlit_and_eclipse_samples() -> None:
    trajectory = Trajectory(
        scenario_id="test",
        backend="local",
        samples=(
            TrajectorySample(
                epoch=datetime(2026, 1, 1, tzinfo=UTC),
                elapsed_s=0.0,
                state=OrbitState(
                    epoch=datetime(2026, 1, 1, tzinfo=UTC),
                    time_scale=TimeScale.UTC,
                    frame=Frame.EME2000,
                    central_body=Body.EARTH,
                    representation=OrbitRepresentation.CARTESIAN,
                    cartesian=CartesianState(
                        position_km=(7000.0, 0.0, 0.0),
                        velocity_km_s=(0.0, 7.5, 0.0),
                    ),
                ),
            ),
            TrajectorySample(
                epoch=datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
                elapsed_s=600.0,
                state=OrbitState(
                    epoch=datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
                    time_scale=TimeScale.UTC,
                    frame=Frame.EME2000,
                    central_body=Body.EARTH,
                    representation=OrbitRepresentation.CARTESIAN,
                    cartesian=CartesianState(
                        position_km=(-7000.0, 0.0, 0.0),
                        velocity_km_s=(0.0, -7.5, 0.0),
                    ),
                ),
            ),
        ),
        metadata={},
    )

    timeline = build_geometry_timeline(trajectory)

    assert len(timeline) == 2
    assert timeline[0].sunlit is True
    assert timeline[1].sunlit is False
    assert timeline[0].altitude_km > 600.0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/astro_twin/test_geometry.py -q`

Expected: FAIL because `astro_twin.geometry` does not exist.

- [ ] **Step 3: Implement deterministic geometry**

Create `src/astro_twin/geometry.py` with:

```python
from __future__ import annotations

from math import sqrt

from astro_core.constants import R_EARTH_KM
from astro_core.models import Trajectory
from astro_twin.models import TimelineGeometrySample

_SUN_DIRECTION = (1.0, 0.0, 0.0)


def build_geometry_timeline(trajectory: Trajectory) -> tuple[TimelineGeometrySample, ...]:
    samples: list[TimelineGeometrySample] = []
    for sample in trajectory.samples:
        position = sample.state.cartesian.position_km
        radius_km = sqrt(sum(component * component for component in position))
        samples.append(
            TimelineGeometrySample(
                epoch=sample.epoch,
                elapsed_s=sample.elapsed_s,
                position_km=position,
                altitude_km=radius_km - R_EARTH_KM,
                sunlit=_is_sunlit(position),
            )
        )
    return tuple(samples)


def _is_sunlit(position_km: tuple[float, float, float]) -> bool:
    x, y, z = position_km
    if x >= 0.0:
        return True
    perpendicular_distance_km = sqrt(y * y + z * z)
    return perpendicular_distance_km > R_EARTH_KM
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/astro_twin/test_geometry.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_twin/geometry.py tests/astro_twin/test_geometry.py
git commit -m "Add digital twin geometry timeline"
```

## Task 5: Power, Thermal, And ADCS Screening

**Files:**
- Create: `src/astro_twin/power.py`
- Create: `src/astro_twin/thermal.py`
- Create: `src/astro_twin/adcs.py`
- Create: `tests/astro_twin/test_power.py`
- Create: `tests/astro_twin/test_thermal.py`
- Create: `tests/astro_twin/test_adcs.py`

- [ ] **Step 1: Write failing power test**

```python
from datetime import UTC, datetime

from astro_twin.models import MissionMode, PowerConfig, TimelineGeometrySample
from astro_twin.power import compute_power_timeline


def test_compute_power_timeline_depletes_battery_in_eclipse() -> None:
    geometry = (
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, tzinfo=UTC),
            elapsed_s=0.0,
            position_km=(7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=True,
        ),
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
            elapsed_s=600.0,
            position_km=(-7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=False,
        ),
    )
    config = PowerConfig(
        solar_array_area_m2=2.0,
        solar_array_efficiency=0.25,
        battery_capacity_wh=1000.0,
        initial_battery_soc_fraction=0.8,
        minimum_battery_soc_fraction=0.35,
        idle_load_w=100.0,
        payload_load_w=250.0,
        downlink_load_w=350.0,
    )

    samples = compute_power_timeline(config, geometry, {})

    assert samples[0].mode == MissionMode.IDLE
    assert samples[0].generated_w > samples[0].load_w
    assert samples[1].battery_soc_fraction < samples[0].battery_soc_fraction
```

- [ ] **Step 2: Write failing thermal and ADCS tests**

```python
from datetime import UTC, datetime

from astro_twin.adcs import compute_adcs_timeline
from astro_twin.models import ADCSConfig, PowerSample, ThermalNodeConfig, TimelineGeometrySample
from astro_twin.thermal import compute_thermal_timeline


def test_compute_thermal_timeline_returns_node_temperatures() -> None:
    geometry = (
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, tzinfo=UTC),
            elapsed_s=0.0,
            position_km=(7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=True,
        ),
    )
    power = (
        PowerSample(
            elapsed_s=0.0,
            mode="idle",
            generated_w=500.0,
            load_w=100.0,
            battery_soc_fraction=0.8,
            net_power_w=400.0,
        ),
    )
    node = ThermalNodeConfig(
        name="bus",
        thermal_mass_j_k=45000.0,
        radiator_area_m2=1.0,
        absorptivity=0.55,
        emissivity=0.78,
        initial_temperature_k=293.0,
        minimum_temperature_k=273.0,
        maximum_temperature_k=313.0,
        internal_heat_fraction=0.45,
    )

    samples = compute_thermal_timeline((node,), geometry, power)

    assert samples[0].node_temperatures_k["bus"] == 293.0


def test_compute_adcs_timeline_reports_positive_margins() -> None:
    geometry = (
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, tzinfo=UTC),
            elapsed_s=0.0,
            position_km=(7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=True,
        ),
    )
    config = ADCSConfig(
        pointing_mode="nadir",
        max_pointing_error_deg=0.08,
        pointing_requirement_deg=0.15,
        max_torque_n_m=0.08,
        required_slew_torque_n_m=0.03,
    )

    samples = compute_adcs_timeline(config, geometry)

    assert samples[0].pointing_margin_deg == 0.06999999999999999
    assert samples[0].torque_margin_n_m == 0.05
```

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/astro_twin/test_power.py tests/astro_twin/test_thermal.py tests/astro_twin/test_adcs.py -q`

Expected: FAIL because subsystem modules do not exist.

- [ ] **Step 4: Implement subsystem modules**

Create:

```python
# src/astro_twin/power.py
from astro_twin.models import MissionMode, PowerConfig, PowerSample, TimelineGeometrySample

_SOLAR_CONSTANT_W_M2 = 1361.0


def compute_power_timeline(
    config: PowerConfig,
    geometry: tuple[TimelineGeometrySample, ...],
    mode_by_elapsed_s: dict[float, MissionMode],
) -> tuple[PowerSample, ...]:
    soc_wh = config.initial_battery_soc_fraction * config.battery_capacity_wh
    samples: list[PowerSample] = []
    previous_elapsed_s = geometry[0].elapsed_s if geometry else 0.0
    for sample in geometry:
        mode = mode_by_elapsed_s.get(sample.elapsed_s, MissionMode.IDLE)
        generated_w = (
            _SOLAR_CONSTANT_W_M2 * config.solar_array_area_m2 * config.solar_array_efficiency
            if sample.sunlit
            else 0.0
        )
        load_w = _mode_load_w(config, mode)
        dt_h = max(0.0, sample.elapsed_s - previous_elapsed_s) / 3600.0
        soc_wh = min(config.battery_capacity_wh, max(0.0, soc_wh + (generated_w - load_w) * dt_h))
        previous_elapsed_s = sample.elapsed_s
        samples.append(
            PowerSample(
                elapsed_s=sample.elapsed_s,
                mode=mode,
                generated_w=generated_w,
                load_w=load_w,
                battery_soc_fraction=soc_wh / config.battery_capacity_wh,
                net_power_w=generated_w - load_w,
            )
        )
    return tuple(samples)


def _mode_load_w(config: PowerConfig, mode: MissionMode) -> float:
    if mode is MissionMode.PAYLOAD:
        return config.payload_load_w
    if mode is MissionMode.DOWNLINK:
        return config.downlink_load_w
    return config.idle_load_w
```

```python
# src/astro_twin/thermal.py
from astro_twin.models import PowerSample, ThermalNodeConfig, ThermalSample, TimelineGeometrySample

_SOLAR_CONSTANT_W_M2 = 1361.0
_SIGMA_W_M2_K4 = 5.670374419e-8
_SPACE_TEMPERATURE_K = 3.0


def compute_thermal_timeline(
    nodes: tuple[ThermalNodeConfig, ...],
    geometry: tuple[TimelineGeometrySample, ...],
    power: tuple[PowerSample, ...],
) -> tuple[ThermalSample, ...]:
    temperatures = {node.name: node.initial_temperature_k for node in nodes}
    samples: list[ThermalSample] = []
    previous_elapsed_s = geometry[0].elapsed_s if geometry else 0.0
    for geometry_sample, power_sample in zip(geometry, power, strict=True):
        dt_s = max(0.0, geometry_sample.elapsed_s - previous_elapsed_s)
        previous_elapsed_s = geometry_sample.elapsed_s
        next_temperatures: dict[str, float] = {}
        for node in nodes:
            current_k = temperatures[node.name]
            absorbed_w = (
                _SOLAR_CONSTANT_W_M2 * node.radiator_area_m2 * node.absorptivity
                if geometry_sample.sunlit
                else 0.0
            )
            internal_w = power_sample.load_w * node.internal_heat_fraction
            radiated_w = (
                node.emissivity
                * _SIGMA_W_M2_K4
                * node.radiator_area_m2
                * (current_k**4 - _SPACE_TEMPERATURE_K**4)
            )
            next_temperatures[node.name] = current_k + (
                (absorbed_w + internal_w - radiated_w) * dt_s / node.thermal_mass_j_k
            )
        temperatures = next_temperatures
        samples.append(
            ThermalSample(
                elapsed_s=geometry_sample.elapsed_s,
                node_temperatures_k=dict(temperatures),
            )
        )
    return tuple(samples)
```

```python
# src/astro_twin/adcs.py
from astro_twin.models import ADCSConfig, ADCSSample, TimelineGeometrySample


def compute_adcs_timeline(
    config: ADCSConfig,
    geometry: tuple[TimelineGeometrySample, ...],
) -> tuple[ADCSSample, ...]:
    return tuple(
        ADCSSample(
            elapsed_s=sample.elapsed_s,
            pointing_error_deg=config.max_pointing_error_deg,
            pointing_margin_deg=config.pointing_requirement_deg - config.max_pointing_error_deg,
            torque_margin_n_m=config.max_torque_n_m - config.required_slew_torque_n_m,
        )
        for sample in geometry
    )
```

- [ ] **Step 5: Run subsystem tests**

Run: `python -m pytest tests/astro_twin/test_power.py tests/astro_twin/test_thermal.py tests/astro_twin/test_adcs.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/astro_twin/power.py src/astro_twin/thermal.py src/astro_twin/adcs.py tests/astro_twin/test_power.py tests/astro_twin/test_thermal.py tests/astro_twin/test_adcs.py
git commit -m "Add digital twin subsystem screening models"
```

## Task 6: Coverage And Link Budget

**Files:**
- Modify: `src/astro_twin/geometry.py`
- Create: `src/astro_twin/coverage.py`
- Create: `src/astro_twin/link_budget.py`
- Create: `tests/astro_twin/test_access_geometry.py`
- Create: `tests/astro_twin/test_coverage.py`
- Create: `tests/astro_twin/test_link_budget.py`

- [ ] **Step 1: Write failing coverage, access-geometry, and link tests**

```python
from astro_twin.coverage import access_windows_from_samples
from astro_twin.geometry import elevation_and_range_km
from astro_twin.link_budget import compute_link_budget_windows
from astro_twin.models import AccessWindow, GroundSiteConfig, LinkBudgetConfig


def test_access_windows_from_samples_groups_contiguous_access() -> None:
    windows = access_windows_from_samples(
        ground_site=GroundSiteConfig(
            name="goldstone",
            latitude_deg=35.0,
            longitude_deg=-116.0,
            altitude_m=1000.0,
            minimum_elevation_deg=10.0,
        ),
        samples=[
            (0.0, 8.0, 2000.0),
            (60.0, 12.0, 1800.0),
            (120.0, 15.0, 1700.0),
            (180.0, 5.0, 2100.0),
        ],
    )

    assert windows == (
        AccessWindow(
            ground_site="goldstone",
            start_s=60.0,
            end_s=120.0,
            duration_s=60.0,
            max_elevation_deg=15.0,
            min_range_km=1700.0,
        ),
    )


def test_compute_link_budget_windows_reports_positive_margin() -> None:
    windows = (
        AccessWindow(
            ground_site="goldstone",
            start_s=60.0,
            end_s=120.0,
            duration_s=60.0,
            max_elevation_deg=15.0,
            min_range_km=1700.0,
        ),
    )
    link = LinkBudgetConfig(
        name="xband-downlink",
        ground_site="goldstone",
        frequency_ghz=8.4,
        eirp_dbw=18.0,
        receiver_g_over_t_db_k=22.0,
        data_rate_bps=2_000_000.0,
        required_ebn0_db=6.5,
        implementation_loss_db=2.0,
    )

    result = compute_link_budget_windows((link,), windows)

    assert result[0].link_name == "xband-downlink"
    assert result[0].data_volume_mbit == 120.0
    assert result[0].worst_ebn0_margin_db > 0.0


def test_elevation_and_range_km_places_overhead_spacecraft_above_mask() -> None:
    site = GroundSiteConfig(
        name="equator",
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_m=0.0,
        minimum_elevation_deg=10.0,
    )

    elevation_deg, range_km = elevation_and_range_km(
        position_km=(7000.0, 0.0, 0.0),
        site=site,
        elapsed_s=0.0,
    )

    assert elevation_deg > 80.0
    assert range_km > 600.0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/astro_twin/test_access_geometry.py tests/astro_twin/test_coverage.py tests/astro_twin/test_link_budget.py -q`

Expected: FAIL because access geometry, coverage, and link budget implementations do not exist.

- [ ] **Step 3: Implement coverage**

Create `src/astro_twin/coverage.py`:

```python
from __future__ import annotations

from collections.abc import Iterable

from astro_twin.models import AccessWindow, GroundSiteConfig


def access_windows_from_samples(
    ground_site: GroundSiteConfig,
    samples: Iterable[tuple[float, float, float]],
) -> tuple[AccessWindow, ...]:
    windows: list[AccessWindow] = []
    active: list[tuple[float, float, float]] = []
    for elapsed_s, elevation_deg, range_km in samples:
        if elevation_deg >= ground_site.minimum_elevation_deg:
            active.append((elapsed_s, elevation_deg, range_km))
            continue
        if active:
            windows.append(_window_from_active(ground_site.name, active))
            active = []
    if active:
        windows.append(_window_from_active(ground_site.name, active))
    return tuple(windows)


def _window_from_active(site_name: str, active: list[tuple[float, float, float]]) -> AccessWindow:
    start_s = active[0][0]
    end_s = active[-1][0]
    return AccessWindow(
        ground_site=site_name,
        start_s=start_s,
        end_s=end_s,
        duration_s=max(1.0, end_s - start_s),
        max_elevation_deg=max(item[1] for item in active),
        min_range_km=min(item[2] for item in active),
    )
```

- [ ] **Step 4: Implement ground access geometry**

Add to `src/astro_twin/geometry.py`:

```python
from math import asin, cos, degrees, radians, sin, sqrt

from astro_twin.models import GroundSiteConfig

_EARTH_ROTATION_RAD_S = 7.2921159e-5


def elevation_and_range_km(
    position_km: tuple[float, float, float],
    site: GroundSiteConfig,
    elapsed_s: float,
) -> tuple[float, float]:
    site_position = _site_position_eci_km(site, elapsed_s)
    relative = tuple(position_km[index] - site_position[index] for index in range(3))
    range_km = sqrt(sum(component * component for component in relative))
    zenith = _unit(site_position)
    elevation_rad = asin(sum(relative[index] * zenith[index] for index in range(3)) / range_km)
    return degrees(elevation_rad), range_km


def _site_position_eci_km(site: GroundSiteConfig, elapsed_s: float) -> tuple[float, float, float]:
    latitude = radians(site.latitude_deg)
    longitude = radians(site.longitude_deg) + _EARTH_ROTATION_RAD_S * elapsed_s
    radius_km = R_EARTH_KM + site.altitude_m / 1000.0
    return (
        radius_km * cos(latitude) * cos(longitude),
        radius_km * cos(latitude) * sin(longitude),
        radius_km * sin(latitude),
    )


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(sum(component * component for component in vector))
    return tuple(component / norm for component in vector)
```

- [ ] **Step 5: Implement link budget**

Create `src/astro_twin/link_budget.py`:

```python
from __future__ import annotations

from math import log10

from astro_twin.models import AccessWindow, LinkBudgetConfig, LinkBudgetWindow

_BOLTZMANN_DB = 228.6


def compute_link_budget_windows(
    links: tuple[LinkBudgetConfig, ...],
    access_windows: tuple[AccessWindow, ...],
) -> tuple[LinkBudgetWindow, ...]:
    results: list[LinkBudgetWindow] = []
    for link in links:
        for window in access_windows:
            if window.ground_site != link.ground_site:
                continue
            fspl_db = 92.45 + 20.0 * log10(window.min_range_km) + 20.0 * log10(link.frequency_ghz)
            cn0_db_hz = (
                link.eirp_dbw
                + link.receiver_g_over_t_db_k
                - fspl_db
                - link.implementation_loss_db
                + _BOLTZMANN_DB
            )
            ebn0_db = cn0_db_hz - 10.0 * log10(link.data_rate_bps)
            results.append(
                LinkBudgetWindow(
                    link_name=link.name,
                    ground_site=window.ground_site,
                    start_s=window.start_s,
                    end_s=window.end_s,
                    duration_s=window.duration_s,
                    worst_ebn0_margin_db=ebn0_db - link.required_ebn0_db,
                    data_volume_mbit=link.data_rate_bps * window.duration_s / 1_000_000.0,
                )
            )
    return tuple(results)
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/astro_twin/test_access_geometry.py tests/astro_twin/test_coverage.py tests/astro_twin/test_link_budget.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/astro_twin/geometry.py src/astro_twin/coverage.py src/astro_twin/link_budget.py tests/astro_twin/test_access_geometry.py tests/astro_twin/test_coverage.py tests/astro_twin/test_link_budget.py
git commit -m "Add digital twin coverage and link budget models"
```

## Task 7: Margin Aggregation

**Files:**
- Create: `src/astro_twin/margins.py`
- Create: `tests/astro_twin/test_margins.py`

- [ ] **Step 1: Write failing margin test**

```python
from astro_twin.margins import build_margin_report
from astro_twin.models import (
    ADCSConfig,
    ADCSSample,
    LinkBudgetWindow,
    MissionMode,
    PowerConfig,
    PowerSample,
    SpacecraftBusConfig,
    ThermalNodeConfig,
    ThermalSample,
)


def test_build_margin_report_identifies_limiting_margin() -> None:
    report = build_margin_report(
        spacecraft=SpacecraftBusConfig(
            name="ObserverSat",
            dry_mass_kg=120.0,
            payload_mass_kg=25.0,
            propellant_mass_kg=5.0,
            mass_margin_fraction_required=0.2,
        ),
        power_config=PowerConfig(
            solar_array_area_m2=2.4,
            solar_array_efficiency=0.29,
            battery_capacity_wh=1200.0,
            initial_battery_soc_fraction=0.85,
            minimum_battery_soc_fraction=0.35,
            idle_load_w=120.0,
            payload_load_w=260.0,
            downlink_load_w=360.0,
        ),
        thermal_nodes=(
            ThermalNodeConfig(
                name="bus",
                thermal_mass_j_k=45000.0,
                radiator_area_m2=1.0,
                absorptivity=0.55,
                emissivity=0.78,
                initial_temperature_k=293.0,
                minimum_temperature_k=273.0,
                maximum_temperature_k=313.0,
                internal_heat_fraction=0.45,
            ),
        ),
        power=(
            PowerSample(
                elapsed_s=0.0,
                mode=MissionMode.IDLE,
                generated_w=600.0,
                load_w=120.0,
                battery_soc_fraction=0.5,
                net_power_w=480.0,
            ),
        ),
        thermal=(ThermalSample(elapsed_s=0.0, node_temperatures_k={"bus": 312.0}),),
        adcs=(
            ADCSSample(
                elapsed_s=0.0,
                pointing_error_deg=0.08,
                pointing_margin_deg=0.07,
                torque_margin_n_m=0.05,
            ),
        ),
        adcs_config=ADCSConfig(
            pointing_mode="nadir",
            max_pointing_error_deg=0.08,
            pointing_requirement_deg=0.15,
            max_torque_n_m=0.08,
            required_slew_torque_n_m=0.03,
        ),
        link_windows=(
            LinkBudgetWindow(
                link_name="xband-downlink",
                ground_site="goldstone",
                start_s=0.0,
                end_s=60.0,
                duration_s=60.0,
                worst_ebn0_margin_db=3.0,
                data_volume_mbit=120.0,
            ),
        ),
    )

    assert report.limiting_margin.name == "thermal_bus_hot_margin_k"
    assert report.limiting_margin.margin == 1.0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/astro_twin/test_margins.py -q`

Expected: FAIL because `astro_twin.margins` does not exist.

- [ ] **Step 3: Implement margin aggregation**

Create `src/astro_twin/margins.py` with explicit margins for:

- `mass_margin_fraction = propellant_mass_kg / (dry_mass_kg + payload_mass_kg + propellant_mass_kg)`
- `battery_soc_margin_fraction = min_soc - minimum_battery_soc_fraction`
- `thermal_<node>_cold_margin_k = min_temp - node.minimum_temperature_k`
- `thermal_<node>_hot_margin_k = node.maximum_temperature_k - max_temp`
- `pointing_margin_deg = min(adcs.pointing_margin_deg)`
- `torque_margin_n_m = min(adcs.torque_margin_n_m)`
- `link_margin_db = min(link.worst_ebn0_margin_db)` or a fail margin when there are no links.

Use:

```python
def _status(margin: float, warn_threshold: float) -> TwinMarginStatus:
    if margin < 0.0:
        return TwinMarginStatus.FAIL
    if margin <= warn_threshold:
        return TwinMarginStatus.WARN
    return TwinMarginStatus.PASS
```

Pick `limiting_margin` as the smallest numeric `margin`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/astro_twin/test_margins.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_twin/margins.py tests/astro_twin/test_margins.py
git commit -m "Add digital twin margin aggregation"
```

## Task 8: Runner And CLI

**Files:**
- Create: `src/astro_twin/runner.py`
- Modify: `src/astro_cli/main.py`
- Create: `tests/astro_twin/test_runner.py`
- Modify: `tests/astro_cli/test_cli.py`

- [ ] **Step 1: Write failing runner test**

```python
from astro_twin.io import load_twin_scenario
from astro_twin.runner import run_digital_twin


def test_run_digital_twin_returns_integrated_result() -> None:
    scenario = load_twin_scenario("examples/twin/leo_observer.yaml")

    result = run_digital_twin(scenario)

    assert result.workflow == "integrated_digital_twin_v1"
    assert len(result.geometry) > 1
    assert len(result.power) == len(result.geometry)
    assert len(result.thermal) == len(result.geometry)
    assert len(result.adcs) == len(result.geometry)
    assert result.margin_report.limiting_margin.name
```

- [ ] **Step 2: Write failing CLI test**

```python
import json

from astro_cli.main import app
from tests.astro_cli.helpers import make_cli_runner


def test_run_twin_command_writes_json_and_summary(tmp_path) -> None:
    runner = make_cli_runner()
    output = tmp_path / "twin.json"
    summary = tmp_path / "twin.txt"

    result = runner.invoke(
        app,
        [
            "run-twin",
            "examples/twin/leo_observer.yaml",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["workflow"] == "integrated_digital_twin_v1"
    assert "Limiting margin:" in summary.read_text(encoding="utf-8")
```

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/astro_twin/test_runner.py tests/astro_cli/test_cli.py::test_run_twin_command_writes_json_and_summary -q`

Expected: FAIL because runner and CLI command do not exist.

- [ ] **Step 4: Implement runner**

Create `src/astro_twin/runner.py`:

```python
from __future__ import annotations

from astro_core.io import load_scenario
from astro_dynamics.local import propagate_local
from astro_twin.adcs import compute_adcs_timeline
from astro_twin.coverage import access_windows_from_samples
from astro_twin.geometry import build_geometry_timeline, elevation_and_range_km
from astro_twin.link_budget import compute_link_budget_windows
from astro_twin.margins import build_margin_report
from astro_twin.models import DigitalTwinResult, DigitalTwinScenario, GroundSiteConfig, MissionMode
from astro_twin.power import compute_power_timeline
from astro_twin.thermal import compute_thermal_timeline


def run_digital_twin(scenario: DigitalTwinScenario) -> DigitalTwinResult:
    orbit_scenario = load_scenario(scenario.orbit_scenario)
    trajectory = propagate_local(orbit_scenario)
    geometry = build_geometry_timeline(trajectory)
    mode_by_elapsed_s = _mode_by_elapsed_s(scenario, geometry)
    power = compute_power_timeline(scenario.power, geometry, mode_by_elapsed_s)
    thermal = compute_thermal_timeline(scenario.thermal_nodes, geometry, power)
    adcs = compute_adcs_timeline(scenario.adcs, geometry)
    access_windows = tuple(
        window
        for site in scenario.ground_sites
        for window in access_windows_from_samples(site, _access_samples_for_site(site, geometry))
    )
    link_windows = compute_link_budget_windows(scenario.links, access_windows)
    margin_report = build_margin_report(
        spacecraft=scenario.spacecraft,
        power_config=scenario.power,
        thermal_nodes=scenario.thermal_nodes,
        power=power,
        thermal=thermal,
        adcs=adcs,
        adcs_config=scenario.adcs,
        link_windows=link_windows,
    )
    return DigitalTwinResult(
        scenario_id=scenario.scenario_id,
        geometry=geometry,
        power=power,
        thermal=thermal,
        adcs=adcs,
        access_windows=access_windows,
        link_windows=link_windows,
        margin_report=margin_report,
        metadata={
            "orbit_scenario": scenario.orbit_scenario,
            "orbit_backend": trajectory.backend,
        },
        warnings=[
            "Digital twin v1 is deterministic design-screening evidence, not flight qualification.",
            "Coverage geometry uses spherical Earth and uniform Earth rotation screening assumptions.",
        ],
    )


def _mode_by_elapsed_s(
    scenario: DigitalTwinScenario,
    geometry: tuple[object, ...],
) -> dict[float, MissionMode]:
    result: dict[float, MissionMode] = {}
    for sample in geometry:
        elapsed_s = getattr(sample, "elapsed_s")
        result[elapsed_s] = MissionMode.IDLE
        for scheduled in scenario.mode_schedule:
            if scheduled.start_s <= elapsed_s <= scheduled.end_s:
                result[elapsed_s] = scheduled.mode
    return result


def _access_samples_for_site(
    site: GroundSiteConfig,
    geometry: tuple[object, ...],
) -> list[tuple[float, float, float]]:
    samples: list[tuple[float, float, float]] = []
    for sample in geometry:
        elapsed_s = getattr(sample, "elapsed_s")
        elevation_deg, range_km = elevation_and_range_km(
            position_km=getattr(sample, "position_km"),
            site=site,
            elapsed_s=elapsed_s,
        )
        samples.append((elapsed_s, elevation_deg, range_km))
    return samples
```

- [ ] **Step 5: Add summary formatter and CLI command**

Add to `src/astro_twin/io.py`:

```python
def format_twin_summary(result: DigitalTwinResult) -> str:
    min_soc = min(sample.battery_soc_fraction for sample in result.power)
    link_margin = (
        min(window.worst_ebn0_margin_db for window in result.link_windows)
        if result.link_windows
        else None
    )
    lines = [
        f"Digital twin: {result.scenario_id}",
        f"Workflow: {result.workflow}",
        f"Samples: {len(result.geometry)}",
        f"Minimum battery SOC: {min_soc}",
        f"Access windows: {len(result.access_windows)}",
        f"Worst link margin dB: {link_margin if link_margin is not None else 'unavailable'}",
        f"Limiting margin: {result.margin_report.limiting_margin.name} = {result.margin_report.limiting_margin.margin}",
    ]
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)
```

Add `run-twin` to `src/astro_cli/main.py` using existing `_write_text_or_exit`:

```python
@app.command("run-twin")
def run_twin(
    scenario_path: Annotated[Path, typer.Argument(help="Digital twin scenario YAML path.")],
    output: Annotated[Path, typer.Option("--output", help="Write digital twin JSON result.")],
    summary_output: Annotated[
        Path | None,
        typer.Option("--summary-output", help="Write a concise text summary."),
    ] = None,
) -> None:
    try:
        scenario = load_twin_scenario(scenario_path)
        result = run_digital_twin(scenario)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_text_or_exit(output, result.model_dump_json(indent=2), "digital twin result")
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        _write_text_or_exit(summary_output, format_twin_summary(result), "digital twin summary")
```

Also import `load_twin_scenario`, `format_twin_summary`, and `run_digital_twin`.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/astro_twin/test_runner.py tests/astro_cli/test_cli.py::test_run_twin_command_writes_json_and_summary -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/astro_twin/runner.py src/astro_twin/io.py src/astro_cli/main.py tests/astro_twin/test_runner.py tests/astro_cli/test_cli.py
git commit -m "Add integrated digital twin runner and CLI"
```

## Task 9: Docs, Validation Matrix, And Final Gates

**Files:**
- Modify: `README.md`
- Modify: `docs/validation-matrix.md`
- Modify: `docs/current-state.md`
- Create: `docs/digital-twin.md`

- [ ] **Step 1: Add digital twin docs**

Create `docs/digital-twin.md`:

````markdown
# Digital Twin

Astro Suite's digital twin workflow runs a deterministic single-spacecraft mission screening model.
It combines local orbit propagation with power, thermal, ADCS, coverage, link budget, mass, and
design-margin products.

```bash
astro run-twin examples/twin/leo_observer.yaml \
  --output /tmp/astro-twin-result.json \
  --summary-output /tmp/astro-twin-summary.txt
```

The v1 twin is design-screening evidence. It is not flight qualification, thermal certification, RF
certification, or operational readiness evidence.
````

- [ ] **Step 2: Update validation matrix**

Add a required local gate:

```markdown
| Integrated digital twin | `astro run-twin examples/twin/leo_observer.yaml --output /tmp/astro-twin-result.json --summary-output /tmp/astro-twin-summary.txt` and `python -m pytest tests/astro_twin -q` | Writes a suite-owned single-spacecraft digital twin product with orbit geometry, power, thermal, ADCS, coverage, link-budget, mass, and design-margin evidence. This is deterministic design-screening evidence, not flight qualification. |
```

- [ ] **Step 3: Update README navigation**

Add one sentence near the assistant or examples section:

```markdown
The integrated digital twin workflow is documented in [Digital Twin](docs/digital-twin.md).
```

- [ ] **Step 4: Update current state**

Set `integrated-digital-twin` to `review-ready`, record the implemented `astro_twin` file surface,
and record the exact verification commands that passed on the branch.

- [ ] **Step 5: Run full verification**

Run:

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
git diff --check
python -m pytest tests/test_packaging.py -q
python -m build
```

Expected:

- Ruff passes.
- Mypy passes.
- Full pytest passes with only existing optional-backend skips.
- Diff check has no output.
- Packaging tests pass.
- Build produces sdist and wheel.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/digital-twin.md docs/validation-matrix.md docs/current-state.md
git commit -m "Document integrated digital twin workflow"
```

## Execution Notes

- Use TDD for every task.
- Keep every public result suite-owned; do not expose backend-native objects.
- Preserve explicit claim boundaries in docs and result warnings.
- Do not add optional backend requirements for v1.
- Do not broaden into constellation support until the single-spacecraft result and validation gates
  are green.

## Final Verification

Before merge or PR:

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
git diff --check
python -m pytest tests/test_packaging.py -q
python -m build
```

Record the exact result counts and any skipped optional backend gates in the closeout.
