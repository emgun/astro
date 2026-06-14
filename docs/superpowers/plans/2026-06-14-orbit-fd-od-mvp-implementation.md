# Orbit Flight Dynamics OD MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working Python vertical slice for scenario validation, local orbit propagation, synthetic measurements, batch orbit determination, CLI workflows, and an Orekit Python-wrapper smoke gate.

**Architecture:** Keep the core domain model dependency-light and explicit about units, frames, epochs, and provenance. Use a deterministic local backend for two-body/J2 reference cases, then isolate Orekit behind `astro_backends` so wrapper-specific details never leak into user-facing models. Launch remains represented in the umbrella architecture but is not implemented until `Scenario`, `OrbitState`, and `Trajectory` are stable.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, SciPy, PyYAML, Typer, pytest, ruff, optional `orekit-jpype`.

---

## Spec Inputs

- `docs/superpowers/specs/2026-06-14-flight-dynamics-suite-architecture-design.md`
- `docs/superpowers/specs/2026-06-14-orbit-fd-od-mvp-design.md`
- `docs/superpowers/specs/2026-06-14-launch-ascent-module-design.md`

## Scope

This plan implements the orbital simulation, flight dynamics, and synthetic-measurement OD MVP. It does not implement launch/ascent, real tracking-data ingestion, CCSDS products, GUI workflows, GPU propagation, or high-fidelity Orekit OD. It does add an Orekit wrapper smoke gate so the riskiest backend dependency is validated early.

## File Structure

Create this structure:

```text
pyproject.toml
README.md
examples/scenarios/leo_two_body.yaml
src/astro_core/__init__.py
src/astro_core/constants.py
src/astro_core/errors.py
src/astro_core/io.py
src/astro_core/models.py
src/astro_dynamics/__init__.py
src/astro_dynamics/local.py
src/astro_od/__init__.py
src/astro_od/estimation.py
src/astro_od/measurements.py
src/astro_backends/__init__.py
src/astro_backends/orekit/__init__.py
src/astro_backends/orekit/smoke.py
src/astro_cli/__init__.py
src/astro_cli/main.py
tests/astro_core/test_models.py
tests/astro_core/test_io.py
tests/astro_dynamics/test_local.py
tests/astro_od/test_measurements.py
tests/astro_od/test_estimation.py
tests/astro_backends/test_orekit_smoke.py
tests/astro_cli/test_cli.py
tests/reference/test_reference_cases.py
tests/test_imports.py
```

Responsibilities:

- `astro_core`: constants, errors, Pydantic models, scenario file loading.
- `astro_dynamics`: local deterministic propagation backend.
- `astro_od`: synthetic measurement generation and batch least-squares estimation.
- `astro_backends.orekit`: official Orekit Python-wrapper smoke checks.
- `astro_cli`: user-facing commands.
- `tests/reference`: deterministic physical sanity checks that anchor future backend comparisons.

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/astro_core/__init__.py`
- Create: `src/astro_dynamics/__init__.py`
- Create: `src/astro_od/__init__.py`
- Create: `src/astro_backends/__init__.py`
- Create: `src/astro_backends/orekit/__init__.py`
- Create: `src/astro_cli/__init__.py`
- Create: `tests/test_imports.py`

- [ ] **Step 1: Write the failing import test**

Create `tests/test_imports.py`:

```python
def test_packages_import() -> None:
    import astro_backends
    import astro_cli
    import astro_core
    import astro_dynamics
    import astro_od

    assert astro_core.__all__ == []
    assert astro_dynamics.__all__ == []
    assert astro_od.__all__ == []
    assert astro_backends.__all__ == []
    assert astro_cli.__all__ == []
```

- [ ] **Step 2: Run the import test to verify it fails**

Run:

```bash
python -m pytest tests/test_imports.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `astro_core`.

- [ ] **Step 3: Create packaging and empty packages**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "astro-suite"
version = "0.1.0"
description = "Python flight dynamics suite for orbit propagation, orbit determination, and launch architecture."
readme = "README.md"
requires-python = ">=3.12"
license = { text = "Apache-2.0" }
authors = [{ name = "Emery Gunselman" }]
dependencies = [
  "numpy>=1.26",
  "pydantic>=2.7",
  "PyYAML>=6.0",
  "scipy>=1.12",
  "typer>=0.12",
]

[project.optional-dependencies]
dev = [
  "mypy>=1.10",
  "pytest>=8.2",
  "ruff>=0.5",
]
orekit = [
  "orekit-jpype>=13.1",
]

[project.scripts]
astro = "astro_cli.main:app"

[tool.hatch.build.targets.wheel]
packages = [
  "src/astro_core",
  "src/astro_dynamics",
  "src/astro_od",
  "src/astro_backends",
  "src/astro_cli",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.12"
strict = true
packages = ["astro_core", "astro_dynamics", "astro_od", "astro_backends", "astro_cli"]
```

Create `README.md`:

```markdown
# Astro Suite

Astro Suite is a Python flight dynamics project for scenario validation, local reference propagation, synthetic orbit-determination measurements, batch OD, and backend adapters.

The first implementation slice focuses on orbital simulation, flight dynamics, and synthetic-measurement orbit determination. Launch/ascent is designed as a first-class module in the specs and follows after the common scenario and trajectory products are stable.
```

Create each package `__init__.py` with this content:

```python
__all__: list[str] = []
```

- [ ] **Step 4: Install and run the import test**

Run:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_imports.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md src tests/test_imports.py
git commit -m "chore: scaffold python flight dynamics package"
```

## Task 2: Core Domain Models

**Files:**
- Create: `src/astro_core/constants.py`
- Create: `src/astro_core/errors.py`
- Create: `src/astro_core/models.py`
- Modify: `src/astro_core/__init__.py`
- Test: `tests/astro_core/test_models.py`

- [ ] **Step 1: Write the failing model tests**

Create `tests/astro_core/test_models.py`:

```python
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from astro_core.models import (
    Body,
    CartesianState,
    ForceModelConfig,
    ForceModelName,
    Frame,
    GroundStation,
    OrbitRepresentation,
    OrbitState,
    PropagationConfig,
    Scenario,
    Spacecraft,
    TimeScale,
)


def make_state() -> OrbitState:
    return OrbitState(
        epoch=datetime(2026, 1, 1, tzinfo=UTC),
        time_scale=TimeScale.UTC,
        frame=Frame.EME2000,
        central_body=Body.EARTH,
        representation=OrbitRepresentation.CARTESIAN,
        cartesian=CartesianState(
            position_km=(7000.0, 0.0, 0.0),
            velocity_km_s=(0.0, 7.5, 1.0),
        ),
    )


def test_orbit_state_requires_finite_cartesian_values() -> None:
    with pytest.raises(ValidationError, match="finite"):
        CartesianState(position_km=(7000.0, float("nan"), 0.0), velocity_km_s=(0.0, 7.5, 0.0))


def test_spacecraft_requires_positive_mass_and_area() -> None:
    with pytest.raises(ValidationError):
        Spacecraft(name="bad", mass_kg=0.0, area_m2=3.0, drag_coefficient=2.2, reflectivity_coefficient=1.2)

    with pytest.raises(ValidationError):
        Spacecraft(name="bad", mass_kg=100.0, area_m2=-1.0, drag_coefficient=2.2, reflectivity_coefficient=1.2)


def test_scenario_accepts_minimal_valid_orbital_case() -> None:
    scenario = Scenario(
        scenario_id="leo-demo",
        description="LEO propagation demo",
        spacecraft=Spacecraft(
            name="demo",
            mass_kg=120.0,
            area_m2=2.5,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.3,
        ),
        initial_state=make_state(),
        force_model=ForceModelConfig(gravity=ForceModelName.TWO_BODY),
        propagation=PropagationConfig(duration_s=600.0, step_s=60.0),
        ground_stations=[
            GroundStation(
                name="station-a",
                position_eci_km=(6378.1363, 0.0, 0.0),
                frame=Frame.EME2000,
                elevation_mask_deg=0.0,
            )
        ],
    )

    assert scenario.scenario_id == "leo-demo"
    assert scenario.propagation.sample_count == 11
```

- [ ] **Step 2: Run the model tests to verify they fail**

Run:

```bash
python -m pytest tests/astro_core/test_models.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing model classes.

- [ ] **Step 3: Add constants and errors**

Create `src/astro_core/constants.py`:

```python
MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.1363
J2_EARTH = 1.08262668e-3
SECONDS_PER_DAY = 86400.0
```

Create `src/astro_core/errors.py`:

```python
class AstroError(Exception):
    """Base exception for Astro Suite."""


class InvalidScenarioError(AstroError):
    """Raised when a scenario file or object is invalid."""


class UnsupportedBackendError(AstroError):
    """Raised when a requested backend is unavailable or unsupported."""


class NumericalConvergenceError(AstroError):
    """Raised when an estimator or propagator fails to converge."""
```

- [ ] **Step 4: Add domain models**

Create `src/astro_core/models.py`:

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Vector3 = tuple[float, float, float]


class Body(StrEnum):
    EARTH = "earth"


class Frame(StrEnum):
    EME2000 = "EME2000"


class TimeScale(StrEnum):
    UTC = "UTC"


class OrbitRepresentation(StrEnum):
    CARTESIAN = "cartesian"


class ForceModelName(StrEnum):
    TWO_BODY = "two_body"
    J2 = "j2"
    OREKIT_HIGH_FIDELITY = "orekit_high_fidelity"


class MeasurementType(StrEnum):
    RANGE = "range"
    RANGE_RATE = "range_rate"


class CartesianState(BaseModel):
    position_km: Vector3
    velocity_km_s: Vector3

    @field_validator("position_km", "velocity_km_s")
    @classmethod
    def values_must_be_finite(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) for component in value):
            raise ValueError("Cartesian state values must be finite")
        return value

    def position_array(self) -> np.ndarray:
        return np.array(self.position_km, dtype=float)

    def velocity_array(self) -> np.ndarray:
        return np.array(self.velocity_km_s, dtype=float)


class OrbitState(BaseModel):
    epoch: datetime
    time_scale: TimeScale
    frame: Frame
    central_body: Body
    representation: OrbitRepresentation
    cartesian: CartesianState

    @model_validator(mode="after")
    def validate_epoch(self) -> OrbitState:
        if self.epoch.tzinfo is None:
            raise ValueError("OrbitState epoch must include timezone information")
        return self


class Spacecraft(BaseModel):
    name: str = Field(min_length=1)
    mass_kg: float = Field(gt=0.0)
    area_m2: float = Field(gt=0.0)
    drag_coefficient: float = Field(ge=0.0, le=10.0)
    reflectivity_coefficient: float = Field(ge=0.0, le=5.0)


class ForceModelConfig(BaseModel):
    gravity: ForceModelName


class PropagationConfig(BaseModel):
    duration_s: float = Field(gt=0.0)
    step_s: float = Field(gt=0.0)

    @property
    def sample_count(self) -> int:
        return int(round(self.duration_s / self.step_s)) + 1

    @model_validator(mode="after")
    def validate_steps(self) -> PropagationConfig:
        steps = self.duration_s / self.step_s
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError("Propagation duration_s must be an integer multiple of step_s")
        return self


class GroundStation(BaseModel):
    name: str = Field(min_length=1)
    position_eci_km: Vector3
    frame: Frame
    elevation_mask_deg: float = Field(ge=-90.0, le=90.0)

    @field_validator("position_eci_km")
    @classmethod
    def position_must_be_finite(cls, value: Vector3) -> Vector3:
        if not all(isfinite(component) for component in value):
            raise ValueError("Ground station position values must be finite")
        return value

    def position_array(self) -> np.ndarray:
        return np.array(self.position_eci_km, dtype=float)


class MeasurementNoise(BaseModel):
    range_sigma_km: float = Field(gt=0.0, default=0.01)
    range_rate_sigma_km_s: float = Field(gt=0.0, default=1.0e-5)
    seed: int = 42


class MeasurementConfig(BaseModel):
    types: tuple[MeasurementType, ...] = (MeasurementType.RANGE, MeasurementType.RANGE_RATE)
    cadence_s: float = Field(gt=0.0, default=60.0)
    noise: MeasurementNoise = Field(default_factory=MeasurementNoise)


class MeasurementRecord(BaseModel):
    measurement_type: MeasurementType
    epoch: datetime
    observer: str
    observed_object: str
    value: float
    sigma: float = Field(gt=0.0)
    units: Literal["km", "km/s"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectorySample(BaseModel):
    epoch: datetime
    state: CartesianState


class Trajectory(BaseModel):
    scenario_id: str
    samples: list[TrajectorySample]
    force_model: ForceModelConfig
    backend: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def epochs_must_be_monotonic(self) -> Trajectory:
        epochs = [sample.epoch for sample in self.samples]
        if epochs != sorted(epochs):
            raise ValueError("Trajectory sample epochs must be monotonic")
        return self


class EstimateResult(BaseModel):
    estimated_state: OrbitState
    residuals: list[float]
    covariance: list[list[float]]
    rms: float
    iterations: int
    converged: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class Scenario(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    scenario_id: str = Field(min_length=1)
    description: str = ""
    spacecraft: Spacecraft
    initial_state: OrbitState
    force_model: ForceModelConfig
    propagation: PropagationConfig
    ground_stations: list[GroundStation] = Field(default_factory=list)
    measurements: MeasurementConfig = Field(default_factory=MeasurementConfig)
```

Modify `src/astro_core/__init__.py`:

```python
from astro_core.models import (
    Body,
    CartesianState,
    EstimateResult,
    ForceModelConfig,
    ForceModelName,
    Frame,
    GroundStation,
    MeasurementConfig,
    MeasurementNoise,
    MeasurementRecord,
    MeasurementType,
    OrbitRepresentation,
    OrbitState,
    PropagationConfig,
    Scenario,
    Spacecraft,
    TimeScale,
    Trajectory,
    TrajectorySample,
)

__all__ = [
    "Body",
    "CartesianState",
    "EstimateResult",
    "ForceModelConfig",
    "ForceModelName",
    "Frame",
    "GroundStation",
    "MeasurementConfig",
    "MeasurementNoise",
    "MeasurementRecord",
    "MeasurementType",
    "OrbitRepresentation",
    "OrbitState",
    "PropagationConfig",
    "Scenario",
    "Spacecraft",
    "TimeScale",
    "Trajectory",
    "TrajectorySample",
]
```

- [ ] **Step 5: Update the import test**

Modify `tests/test_imports.py`:

```python
def test_packages_import() -> None:
    import astro_backends
    import astro_cli
    import astro_core
    import astro_dynamics
    import astro_od

    assert "Scenario" in astro_core.__all__
    assert astro_dynamics.__all__ == []
    assert astro_od.__all__ == []
    assert astro_backends.__all__ == []
    assert astro_cli.__all__ == []
```

- [ ] **Step 6: Run model tests**

Run:

```bash
python -m pytest tests/test_imports.py tests/astro_core/test_models.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/astro_core tests/test_imports.py tests/astro_core/test_models.py
git commit -m "feat: add core flight dynamics domain models"
```

## Task 3: Scenario File Loading

**Files:**
- Create: `src/astro_core/io.py`
- Create: `examples/scenarios/leo_two_body.yaml`
- Modify: `src/astro_core/__init__.py`
- Test: `tests/astro_core/test_io.py`

- [ ] **Step 1: Write the failing scenario IO tests**

Create `tests/astro_core/test_io.py`:

```python
from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario
from astro_core.models import ForceModelName


def test_load_example_scenario() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    assert scenario.scenario_id == "leo-two-body"
    assert scenario.force_model.gravity is ForceModelName.TWO_BODY
    assert scenario.propagation.sample_count == 11


def test_load_scenario_reports_yaml_parse_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("scenario_id: [broken", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="Could not parse scenario file"):
        load_scenario(path)
```

- [ ] **Step 2: Run the IO tests to verify they fail**

Run:

```bash
python -m pytest tests/astro_core/test_io.py -v
```

Expected: FAIL because `astro_core.io` does not exist.

- [ ] **Step 3: Add the example scenario**

Create `examples/scenarios/leo_two_body.yaml`:

```yaml
scenario_id: leo-two-body
description: Deterministic LEO two-body propagation example.
spacecraft:
  name: demo-sat
  mass_kg: 120.0
  area_m2: 2.5
  drag_coefficient: 2.2
  reflectivity_coefficient: 1.3
initial_state:
  epoch: "2026-01-01T00:00:00+00:00"
  time_scale: UTC
  frame: EME2000
  central_body: earth
  representation: cartesian
  cartesian:
    position_km: [7000.0, 0.0, 0.0]
    velocity_km_s: [0.0, 7.5, 1.0]
force_model:
  gravity: two_body
propagation:
  duration_s: 600.0
  step_s: 60.0
ground_stations:
  - name: equator-eci
    position_eci_km: [6378.1363, 0.0, 0.0]
    frame: EME2000
    elevation_mask_deg: 0.0
measurements:
  types: [range, range_rate]
  cadence_s: 60.0
  noise:
    range_sigma_km: 0.01
    range_rate_sigma_km_s: 0.00001
    seed: 42
```

- [ ] **Step 4: Add scenario loader**

Create `src/astro_core/io.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_core.models import Scenario


def load_scenario(path: Path | str) -> Scenario:
    scenario_path = Path(path)
    try:
        raw: Any = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InvalidScenarioError(f"Could not read scenario file {scenario_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(f"Could not parse scenario file {scenario_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise InvalidScenarioError(f"Scenario file {scenario_path} must contain a mapping")

    try:
        return Scenario.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(f"Scenario file {scenario_path} is invalid: {exc}") from exc
```

Modify `src/astro_core/__init__.py` by adding the import and `__all__` entry:

```python
from astro_core.io import load_scenario
```

```python
"load_scenario",
```

- [ ] **Step 5: Run IO tests**

Run:

```bash
python -m pytest tests/astro_core/test_io.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/astro_core examples/scenarios/leo_two_body.yaml tests/astro_core/test_io.py
git commit -m "feat: add scenario file loading"
```

## Task 4: Local Two-Body and J2 Propagation

**Files:**
- Create: `src/astro_dynamics/local.py`
- Modify: `src/astro_dynamics/__init__.py`
- Test: `tests/astro_dynamics/test_local.py`

- [ ] **Step 1: Write failing propagation tests**

Create `tests/astro_dynamics/test_local.py`:

```python
from pathlib import Path

import numpy as np

from astro_core.io import load_scenario
from astro_core.models import ForceModelConfig, ForceModelName
from astro_dynamics.local import j2_acceleration_km_s2, propagate_local, two_body_acceleration_km_s2


def test_two_body_acceleration_points_toward_origin() -> None:
    acceleration = two_body_acceleration_km_s2(np.array([7000.0, 0.0, 0.0]))

    assert acceleration[0] < 0.0
    assert acceleration[1] == 0.0
    assert acceleration[2] == 0.0


def test_j2_acceleration_is_nonzero_for_inclined_state() -> None:
    acceleration = j2_acceleration_km_s2(np.array([5000.0, 3000.0, 4000.0]))

    assert np.linalg.norm(acceleration) > 0.0


def test_propagate_local_returns_expected_sample_count() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)

    assert trajectory.backend == "local"
    assert len(trajectory.samples) == scenario.propagation.sample_count
    assert trajectory.samples[0].state.position_km == scenario.initial_state.cartesian.position_km


def test_j2_and_two_body_propagations_diverge() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    j2_scenario = scenario.model_copy(
        update={"force_model": ForceModelConfig(gravity=ForceModelName.J2)}
    )

    two_body = propagate_local(scenario)
    j2 = propagate_local(j2_scenario)

    two_body_final = np.array(two_body.samples[-1].state.position_km)
    j2_final = np.array(j2.samples[-1].state.position_km)
    assert np.linalg.norm(two_body_final - j2_final) > 0.0
```

- [ ] **Step 2: Run propagation tests to verify they fail**

Run:

```bash
python -m pytest tests/astro_dynamics/test_local.py -v
```

Expected: FAIL because `astro_dynamics.local` does not exist.

- [ ] **Step 3: Implement local propagation**

Create `src/astro_dynamics/local.py`:

```python
from __future__ import annotations

from datetime import timedelta

import numpy as np
from numpy.typing import NDArray

from astro_core.constants import J2_EARTH, MU_EARTH_KM3_S2, R_EARTH_KM
from astro_core.models import (
    CartesianState,
    ForceModelName,
    Scenario,
    Trajectory,
    TrajectorySample,
)

FloatArray = NDArray[np.float64]


def two_body_acceleration_km_s2(position_km: FloatArray) -> FloatArray:
    radius = float(np.linalg.norm(position_km))
    return -MU_EARTH_KM3_S2 * position_km / radius**3


def j2_acceleration_km_s2(position_km: FloatArray) -> FloatArray:
    x, y, z = position_km
    radius2 = float(np.dot(position_km, position_km))
    radius = radius2**0.5
    z2_over_r2 = (z * z) / radius2
    factor = 1.5 * J2_EARTH * MU_EARTH_KM3_S2 * R_EARTH_KM**2 / radius**5
    return factor * np.array(
        [
            x * (5.0 * z2_over_r2 - 1.0),
            y * (5.0 * z2_over_r2 - 1.0),
            z * (5.0 * z2_over_r2 - 3.0),
        ],
        dtype=float,
    )


def acceleration_km_s2(position_km: FloatArray, force_model: ForceModelName) -> FloatArray:
    acceleration = two_body_acceleration_km_s2(position_km)
    if force_model is ForceModelName.J2:
        acceleration = acceleration + j2_acceleration_km_s2(position_km)
    return acceleration


def derivative(state: FloatArray, force_model: ForceModelName) -> FloatArray:
    position = state[:3]
    velocity = state[3:]
    return np.concatenate([velocity, acceleration_km_s2(position, force_model)])


def rk4_step(state: FloatArray, step_s: float, force_model: ForceModelName) -> FloatArray:
    k1 = derivative(state, force_model)
    k2 = derivative(state + 0.5 * step_s * k1, force_model)
    k3 = derivative(state + 0.5 * step_s * k2, force_model)
    k4 = derivative(state + step_s * k3, force_model)
    return state + (step_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate_local(scenario: Scenario) -> Trajectory:
    force_model = scenario.force_model.gravity
    if force_model is ForceModelName.OREKIT_HIGH_FIDELITY:
        raise ValueError("Local backend supports only two_body and j2 force models")

    initial = scenario.initial_state.cartesian
    state = np.concatenate([initial.position_array(), initial.velocity_array()])
    samples: list[TrajectorySample] = []

    for step_index in range(scenario.propagation.sample_count):
        epoch = scenario.initial_state.epoch + timedelta(
            seconds=step_index * scenario.propagation.step_s
        )
        samples.append(
            TrajectorySample(
                epoch=epoch,
                state=CartesianState(
                    position_km=tuple(float(value) for value in state[:3]),
                    velocity_km_s=tuple(float(value) for value in state[3:]),
                ),
            )
        )
        if step_index < scenario.propagation.sample_count - 1:
            state = rk4_step(state, scenario.propagation.step_s, force_model)

    return Trajectory(
        scenario_id=scenario.scenario_id,
        samples=samples,
        force_model=scenario.force_model,
        backend="local",
        metadata={"integrator": "rk4", "step_s": scenario.propagation.step_s},
    )
```

Modify `src/astro_dynamics/__init__.py`:

```python
from astro_dynamics.local import (
    acceleration_km_s2,
    j2_acceleration_km_s2,
    propagate_local,
    rk4_step,
    two_body_acceleration_km_s2,
)

__all__ = [
    "acceleration_km_s2",
    "j2_acceleration_km_s2",
    "propagate_local",
    "rk4_step",
    "two_body_acceleration_km_s2",
]
```

- [ ] **Step 4: Run propagation tests**

Run:

```bash
python -m pytest tests/astro_dynamics/test_local.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_dynamics tests/astro_dynamics/test_local.py
git commit -m "feat: add local two-body and j2 propagation"
```

## Task 5: CLI Validation and Propagation

**Files:**
- Create: `src/astro_cli/main.py`
- Modify: `src/astro_cli/__init__.py`
- Test: `tests/astro_cli/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/astro_cli/test_cli.py`:

```python
import json
from pathlib import Path

from typer.testing import CliRunner

from astro_cli.main import app


runner = CliRunner()


def test_validate_command_accepts_example_scenario() -> None:
    result = runner.invoke(app, ["validate", "examples/scenarios/leo_two_body.yaml"])

    assert result.exit_code == 0
    assert "leo-two-body" in result.stdout


def test_propagate_command_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.json"

    result = runner.invoke(
        app,
        [
            "propagate",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "local",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "leo-two-body"
    assert payload["backend"] == "local"
    assert len(payload["samples"]) == 11
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py -v
```

Expected: FAIL because `astro_cli.main` does not exist.

- [ ] **Step 3: Implement CLI commands**

Create `src/astro_cli/main.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario
from astro_dynamics.local import propagate_local

app = typer.Typer(help="Astro Suite flight dynamics workflows.")


@app.command()
def validate(scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)]) -> None:
    """Validate a scenario file."""
    try:
        scenario = load_scenario(scenario_path)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(f"valid scenario: {scenario.scenario_id}")


@app.command()
def propagate(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    backend: Annotated[str, typer.Option()] = "local",
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Propagate a scenario and write a trajectory product."""
    scenario = load_scenario(scenario_path)
    if backend != "local":
        typer.echo(f"unsupported propagation backend: {backend}", err=True)
        raise typer.Exit(code=2)

    trajectory = propagate_local(scenario)
    payload = trajectory.model_dump_json(indent=2)
    if output is None:
        typer.echo(payload)
    else:
        output.write_text(payload + "\n", encoding="utf-8")
        typer.echo(f"wrote trajectory: {output}")
```

Modify `src/astro_cli/__init__.py`:

```python
from astro_cli.main import app

__all__ = ["app"]
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Run installed CLI manually**

Run:

```bash
astro validate examples/scenarios/leo_two_body.yaml
```

Expected output includes:

```text
valid scenario: leo-two-body
```

- [ ] **Step 6: Commit**

```bash
git add src/astro_cli tests/astro_cli/test_cli.py
git commit -m "feat: add scenario validation and propagation cli"
```

## Task 6: Synthetic Measurement Generation

**Files:**
- Create: `src/astro_od/measurements.py`
- Modify: `src/astro_od/__init__.py`
- Test: `tests/astro_od/test_measurements.py`

- [ ] **Step 1: Write failing measurement tests**

Create `tests/astro_od/test_measurements.py`:

```python
from pathlib import Path

import numpy as np

from astro_core.io import load_scenario
from astro_core.models import MeasurementType
from astro_dynamics.local import propagate_local
from astro_od.measurements import generate_synthetic_measurements, range_km, range_rate_km_s


def test_range_and_range_rate_geometry() -> None:
    spacecraft_position = np.array([7000.0, 0.0, 0.0])
    spacecraft_velocity = np.array([0.0, 7.5, 0.0])
    station_position = np.array([6378.0, 0.0, 0.0])

    assert range_km(spacecraft_position, station_position) == 622.0
    assert range_rate_km_s(spacecraft_position, spacecraft_velocity, station_position) == 0.0


def test_generate_synthetic_measurements_is_deterministic() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)

    first = generate_synthetic_measurements(scenario, trajectory)
    second = generate_synthetic_measurements(scenario, trajectory)

    assert len(first) == 22
    assert first == second
    assert {record.measurement_type for record in first} == {
        MeasurementType.RANGE,
        MeasurementType.RANGE_RATE,
    }
```

- [ ] **Step 2: Run measurement tests to verify they fail**

Run:

```bash
python -m pytest tests/astro_od/test_measurements.py -v
```

Expected: FAIL because `astro_od.measurements` does not exist.

- [ ] **Step 3: Implement synthetic measurements**

Create `src/astro_od/measurements.py`:

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from astro_core.models import MeasurementRecord, MeasurementType, Scenario, Trajectory

FloatArray = NDArray[np.float64]


def range_km(spacecraft_position_km: FloatArray, station_position_km: FloatArray) -> float:
    return float(np.linalg.norm(spacecraft_position_km - station_position_km))


def range_rate_km_s(
    spacecraft_position_km: FloatArray,
    spacecraft_velocity_km_s: FloatArray,
    station_position_km: FloatArray,
) -> float:
    relative_position = spacecraft_position_km - station_position_km
    distance = np.linalg.norm(relative_position)
    line_of_sight = relative_position / distance
    return float(np.dot(spacecraft_velocity_km_s, line_of_sight))


def generate_synthetic_measurements(
    scenario: Scenario,
    trajectory: Trajectory,
) -> list[MeasurementRecord]:
    if not scenario.ground_stations:
        return []

    rng = np.random.default_rng(scenario.measurements.noise.seed)
    records: list[MeasurementRecord] = []

    for sample in trajectory.samples:
        elapsed_s = (sample.epoch - scenario.initial_state.epoch).total_seconds()
        if abs((elapsed_s / scenario.measurements.cadence_s) - round(elapsed_s / scenario.measurements.cadence_s)) > 1e-9:
            continue

        spacecraft_position = sample.state.position_array()
        spacecraft_velocity = sample.state.velocity_array()

        for station in scenario.ground_stations:
            station_position = station.position_array()
            if MeasurementType.RANGE in scenario.measurements.types:
                truth = range_km(spacecraft_position, station_position)
                sigma = scenario.measurements.noise.range_sigma_km
                records.append(
                    MeasurementRecord(
                        measurement_type=MeasurementType.RANGE,
                        epoch=sample.epoch,
                        observer=station.name,
                        observed_object=scenario.spacecraft.name,
                        value=float(truth + rng.normal(0.0, sigma)),
                        sigma=sigma,
                        units="km",
                        metadata={"truth": truth},
                    )
                )
            if MeasurementType.RANGE_RATE in scenario.measurements.types:
                truth = range_rate_km_s(spacecraft_position, spacecraft_velocity, station_position)
                sigma = scenario.measurements.noise.range_rate_sigma_km_s
                records.append(
                    MeasurementRecord(
                        measurement_type=MeasurementType.RANGE_RATE,
                        epoch=sample.epoch,
                        observer=station.name,
                        observed_object=scenario.spacecraft.name,
                        value=float(truth + rng.normal(0.0, sigma)),
                        sigma=sigma,
                        units="km/s",
                        metadata={"truth": truth},
                    )
                )

    return records
```

Modify `src/astro_od/__init__.py`:

```python
from astro_od.measurements import (
    generate_synthetic_measurements,
    range_km,
    range_rate_km_s,
)

__all__ = [
    "generate_synthetic_measurements",
    "range_km",
    "range_rate_km_s",
]
```

- [ ] **Step 4: Run measurement tests**

Run:

```bash
python -m pytest tests/astro_od/test_measurements.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_od tests/astro_od/test_measurements.py
git commit -m "feat: generate synthetic range measurements"
```

## Task 7: Batch Least-Squares Orbit Determination

**Files:**
- Create: `src/astro_od/estimation.py`
- Modify: `src/astro_od/__init__.py`
- Test: `tests/astro_od/test_estimation.py`

- [ ] **Step 1: Write failing OD recovery test**

Create `tests/astro_od/test_estimation.py`:

```python
from pathlib import Path

import numpy as np

from astro_core.io import load_scenario
from astro_core.models import CartesianState
from astro_dynamics.local import propagate_local
from astro_od.estimation import estimate_initial_state
from astro_od.measurements import generate_synthetic_measurements


def test_batch_od_recovers_synthetic_initial_state() -> None:
    truth_scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    truth_trajectory = propagate_local(truth_scenario)
    measurements = generate_synthetic_measurements(truth_scenario, truth_trajectory)

    perturbed_state = truth_scenario.initial_state.model_copy(
        update={
            "cartesian": CartesianState(
                position_km=(7001.0, -0.8, 0.6),
                velocity_km_s=(0.0005, 7.499, 1.0008),
            )
        }
    )
    estimate_scenario = truth_scenario.model_copy(update={"initial_state": perturbed_state})

    result = estimate_initial_state(estimate_scenario, measurements)

    truth_position = truth_scenario.initial_state.cartesian.position_array()
    estimated_position = result.estimated_state.cartesian.position_array()
    truth_velocity = truth_scenario.initial_state.cartesian.velocity_array()
    estimated_velocity = result.estimated_state.cartesian.velocity_array()

    assert result.converged is True
    assert np.linalg.norm(estimated_position - truth_position) < 0.2
    assert np.linalg.norm(estimated_velocity - truth_velocity) < 2.0e-4
    assert result.rms < 3.0
    assert len(result.covariance) == 6
```

- [ ] **Step 2: Run OD test to verify it fails**

Run:

```bash
python -m pytest tests/astro_od/test_estimation.py -v
```

Expected: FAIL because `astro_od.estimation` does not exist.

- [ ] **Step 3: Implement batch OD**

Create `src/astro_od/estimation.py`:

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from astro_core.errors import NumericalConvergenceError
from astro_core.models import (
    CartesianState,
    EstimateResult,
    MeasurementRecord,
    MeasurementType,
    OrbitState,
    Scenario,
)
from astro_dynamics.local import propagate_local
from astro_od.measurements import range_km, range_rate_km_s

FloatArray = NDArray[np.float64]


def scenario_with_state_vector(scenario: Scenario, vector: FloatArray) -> Scenario:
    new_state = scenario.initial_state.model_copy(
        update={
            "cartesian": CartesianState(
                position_km=tuple(float(value) for value in vector[:3]),
                velocity_km_s=tuple(float(value) for value in vector[3:]),
            )
        }
    )
    return scenario.model_copy(update={"initial_state": new_state})


def predicted_measurement(
    scenario: Scenario,
    measurement: MeasurementRecord,
    trajectory_index: dict[object, CartesianState],
) -> float:
    sample_state = trajectory_index[measurement.epoch]
    station = next(station for station in scenario.ground_stations if station.name == measurement.observer)
    spacecraft_position = sample_state.position_array()
    spacecraft_velocity = sample_state.velocity_array()
    station_position = station.position_array()

    if measurement.measurement_type is MeasurementType.RANGE:
        return range_km(spacecraft_position, station_position)
    if measurement.measurement_type is MeasurementType.RANGE_RATE:
        return range_rate_km_s(spacecraft_position, spacecraft_velocity, station_position)
    raise ValueError(f"Unsupported measurement type: {measurement.measurement_type}")


def residual_vector(state_vector: FloatArray, scenario: Scenario, measurements: list[MeasurementRecord]) -> FloatArray:
    trial_scenario = scenario_with_state_vector(scenario, state_vector)
    trajectory = propagate_local(trial_scenario)
    trajectory_index = {sample.epoch: sample.state for sample in trajectory.samples}

    residuals = []
    for measurement in measurements:
        predicted = predicted_measurement(trial_scenario, measurement, trajectory_index)
        residuals.append((predicted - measurement.value) / measurement.sigma)
    return np.array(residuals, dtype=float)


def covariance_from_jacobian(jacobian: FloatArray, rms: float) -> list[list[float]]:
    normal_matrix = jacobian.T @ jacobian
    covariance = np.linalg.pinv(normal_matrix) * rms**2
    return covariance.tolist()


def estimate_initial_state(scenario: Scenario, measurements: list[MeasurementRecord]) -> EstimateResult:
    if not measurements:
        raise NumericalConvergenceError("At least one measurement is required for estimation")

    initial = scenario.initial_state.cartesian
    x0 = np.concatenate([initial.position_array(), initial.velocity_array()])
    result = least_squares(
        residual_vector,
        x0,
        args=(scenario, measurements),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=80,
    )

    residuals = residual_vector(result.x, scenario, measurements)
    rms = float(np.sqrt(np.mean(residuals**2)))
    estimated_state: OrbitState = scenario.initial_state.model_copy(
        update={
            "cartesian": CartesianState(
                position_km=tuple(float(value) for value in result.x[:3]),
                velocity_km_s=tuple(float(value) for value in result.x[3:]),
            )
        }
    )

    return EstimateResult(
        estimated_state=estimated_state,
        residuals=[float(value) for value in residuals],
        covariance=covariance_from_jacobian(result.jac, rms),
        rms=rms,
        iterations=int(result.nfev),
        converged=bool(result.success),
        metadata={"message": str(result.message), "backend": "local_scipy_least_squares"},
    )
```

Modify `src/astro_od/__init__.py`:

```python
from astro_od.estimation import estimate_initial_state
from astro_od.measurements import (
    generate_synthetic_measurements,
    range_km,
    range_rate_km_s,
)

__all__ = [
    "estimate_initial_state",
    "generate_synthetic_measurements",
    "range_km",
    "range_rate_km_s",
]
```

- [ ] **Step 4: Run OD tests**

Run:

```bash
python -m pytest tests/astro_od/test_estimation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_od tests/astro_od/test_estimation.py
git commit -m "feat: add local batch orbit determination"
```

## Task 8: Measurement and Estimation CLI Commands

**Files:**
- Modify: `src/astro_cli/main.py`
- Test: `tests/astro_cli/test_cli.py`

- [ ] **Step 1: Extend CLI tests for measurements and estimation**

Append to `tests/astro_cli/test_cli.py`:

```python

def test_synth_measurements_command_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "measurements.json"

    result = runner.invoke(
        app,
        [
            "synth-measurements",
            "examples/scenarios/leo_two_body.yaml",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["measurements"]) == 22
    assert payload["scenario_id"] == "leo-two-body"


def test_estimate_command_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "estimate.json"

    result = runner.invoke(
        app,
        [
            "estimate",
            "examples/scenarios/leo_two_body.yaml",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["converged"] is True
    assert payload["rms"] < 3.0
```

- [ ] **Step 2: Run CLI tests to verify the new commands fail**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py -v
```

Expected: FAIL with missing `synth-measurements` and `estimate` commands.

- [ ] **Step 3: Add measurement and estimation commands**

Modify `src/astro_cli/main.py` by adding imports:

```python
import json

from astro_od.estimation import estimate_initial_state
from astro_od.measurements import generate_synthetic_measurements
```

Add these commands below `propagate`:

```python

@app.command("synth-measurements")
def synth_measurements(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Generate synthetic measurements from local truth propagation."""
    scenario = load_scenario(scenario_path)
    trajectory = propagate_local(scenario)
    measurements = generate_synthetic_measurements(scenario, trajectory)
    payload = {
        "scenario_id": scenario.scenario_id,
        "measurements": [record.model_dump(mode="json") for record in measurements],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"wrote measurements: {output}")


@app.command()
def estimate(
    scenario_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Run local synthetic-measurement batch OD."""
    scenario = load_scenario(scenario_path)
    truth = propagate_local(scenario)
    measurements = generate_synthetic_measurements(scenario, truth)
    result = estimate_initial_state(scenario, measurements)
    output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(f"wrote estimate: {output}")
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_cli tests/astro_cli/test_cli.py
git commit -m "feat: add measurement and estimation cli workflows"
```

## Task 9: Deterministic Reference Cases

**Files:**
- Create: `tests/reference/test_reference_cases.py`

- [ ] **Step 1: Write reference-case tests**

Create `tests/reference/test_reference_cases.py`:

```python
from pathlib import Path

import numpy as np

from astro_core.io import load_scenario
from astro_core.models import ForceModelConfig, ForceModelName
from astro_dynamics.local import propagate_local
from astro_od.estimation import estimate_initial_state
from astro_od.measurements import generate_synthetic_measurements


def specific_energy_km2_s2(position: np.ndarray, velocity: np.ndarray) -> float:
    mu = 398600.4418
    return float(0.5 * np.dot(velocity, velocity) - mu / np.linalg.norm(position))


def test_two_body_specific_energy_is_stable_over_short_arc() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)
    first = trajectory.samples[0].state
    last = trajectory.samples[-1].state

    first_energy = specific_energy_km2_s2(first.position_array(), first.velocity_array())
    last_energy = specific_energy_km2_s2(last.position_array(), last.velocity_array())

    assert abs(last_energy - first_energy) < 1.0e-7


def test_j2_reference_case_diverges_from_two_body_without_breaking_energy_scale() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    j2_scenario = scenario.model_copy(
        update={"force_model": ForceModelConfig(gravity=ForceModelName.J2)}
    )
    two_body = propagate_local(scenario)
    j2 = propagate_local(j2_scenario)

    two_body_position = np.array(two_body.samples[-1].state.position_km)
    j2_position = np.array(j2.samples[-1].state.position_km)

    assert 0.0 < np.linalg.norm(j2_position - two_body_position) < 10.0


def test_synthetic_od_reference_case_converges() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    measurements = generate_synthetic_measurements(scenario, propagate_local(scenario))
    result = estimate_initial_state(scenario, measurements)

    assert result.converged is True
    assert result.rms < 3.0
    assert len(result.residuals) == len(measurements)
```

- [ ] **Step 2: Run reference tests**

Run:

```bash
python -m pytest tests/reference/test_reference_cases.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/reference/test_reference_cases.py
git commit -m "test: add deterministic flight dynamics reference cases"
```

## Task 10: Orekit Python Wrapper Smoke Gate

**Files:**
- Create: `src/astro_backends/orekit/smoke.py`
- Modify: `src/astro_backends/orekit/__init__.py`
- Modify: `src/astro_cli/main.py`
- Test: `tests/astro_backends/test_orekit_smoke.py`

- [ ] **Step 1: Write failing Orekit smoke tests**

Create `tests/astro_backends/test_orekit_smoke.py`:

```python
from astro_backends.orekit.smoke import OrekitSmokeResult, run_orekit_smoke


def test_orekit_smoke_returns_structured_result_when_wrapper_missing() -> None:
    result = run_orekit_smoke(strict=False, force_unavailable=True)

    assert isinstance(result, OrekitSmokeResult)
    assert result.available is False
    assert result.wrapper == "orekit_jpype"
    assert "not installed" in result.message
```

- [ ] **Step 2: Run Orekit smoke tests to verify they fail**

Run:

```bash
python -m pytest tests/astro_backends/test_orekit_smoke.py -v
```

Expected: FAIL because `astro_backends.orekit.smoke` does not exist.

- [ ] **Step 3: Implement structured Orekit smoke check**

Create `src/astro_backends/orekit/smoke.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True)
class OrekitSmokeResult:
    available: bool
    wrapper: str
    version: str | None
    message: str

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def run_orekit_smoke(*, strict: bool = False, force_unavailable: bool = False) -> OrekitSmokeResult:
    wrapper = "orekit_jpype"
    if force_unavailable:
        return OrekitSmokeResult(
            available=False,
            wrapper=wrapper,
            version=None,
            message="orekit_jpype is not installed",
        )

    try:
        wrapper_version = version("orekit-jpype")
        import orekit_jpype as orekit
    except (ImportError, PackageNotFoundError) as exc:
        if strict:
            raise
        return OrekitSmokeResult(
            available=False,
            wrapper=wrapper,
            version=None,
            message=f"orekit_jpype is not installed: {exc}",
        )

    try:
        orekit.initVM()
        from org.orekit.frames import FramesFactory
        from org.orekit.time import TimeScalesFactory

        FramesFactory.getEME2000()
        TimeScalesFactory.getUTC()
    except Exception as exc:
        if strict:
            raise
        return OrekitSmokeResult(
            available=False,
            wrapper=wrapper,
            version=wrapper_version,
            message=f"orekit_jpype import succeeded but VM/frame/time smoke failed: {exc}",
        )

    return OrekitSmokeResult(
        available=True,
        wrapper=wrapper,
        version=wrapper_version,
        message="orekit_jpype VM, EME2000 frame, and UTC time scale are available",
    )
```

Modify `src/astro_backends/orekit/__init__.py`:

```python
from astro_backends.orekit.smoke import OrekitSmokeResult, run_orekit_smoke

__all__ = ["OrekitSmokeResult", "run_orekit_smoke"]
```

- [ ] **Step 4: Add CLI command for Orekit smoke**

Modify `src/astro_cli/main.py` by adding this import:

```python
from astro_backends.orekit.smoke import run_orekit_smoke
```

Add this command below `estimate`:

```python

@app.command("orekit-smoke")
def orekit_smoke() -> None:
    """Check whether the preferred Orekit Python wrapper can initialize."""
    result = run_orekit_smoke(strict=False)
    typer.echo(json.dumps(result.to_dict(), indent=2))
    if not result.available:
        raise typer.Exit(code=1)
```

- [ ] **Step 5: Run smoke tests**

Run:

```bash
python -m pytest tests/astro_backends/test_orekit_smoke.py -v
```

Expected: PASS.

- [ ] **Step 6: Run optional live Orekit smoke when extra is installed**

Run:

```bash
python -m pip install -e '.[orekit]'
astro orekit-smoke
```

Expected if Java and wrapper are healthy: JSON with `"available": true`.

Expected if Java or Orekit data setup is not healthy: JSON with `"available": false` and a concrete message naming the failed smoke stage.

- [ ] **Step 7: Commit**

```bash
git add src/astro_backends src/astro_cli tests/astro_backends/test_orekit_smoke.py
git commit -m "feat: add orekit python wrapper smoke gate"
```

## Task 11: Full Verification and Documentation Polish

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README with runnable workflows**

Replace `README.md` with:

```markdown
# Astro Suite

Astro Suite is a Python flight dynamics project for scenario validation, local reference propagation, synthetic orbit-determination measurements, batch OD, and backend adapters.

## Current Scope

The current implementation slice covers:

- Pydantic scenario validation.
- Local two-body and J2 reference propagation.
- Synthetic range and range-rate measurements.
- Local SciPy batch least-squares orbit determination.
- CLI workflows.
- Orekit Python-wrapper smoke checks through `orekit_jpype`.

Launch/ascent is included in the design specs and will be implemented after the common scenario, trajectory, and backend adapter spine is stable.

## Setup

```bash
python -m pip install -e '.[dev]'
```

Optional Orekit wrapper smoke support:

```bash
python -m pip install -e '.[orekit]'
astro orekit-smoke
```

## Commands

```bash
astro validate examples/scenarios/leo_two_body.yaml
astro propagate examples/scenarios/leo_two_body.yaml --backend local --output trajectory.json
astro synth-measurements examples/scenarios/leo_two_body.yaml --output measurements.json
astro estimate examples/scenarios/leo_two_body.yaml --output estimate.json
```

## Verification

```bash
python -m pytest -v
python -m ruff check .
python -m mypy src
```
```

- [ ] **Step 2: Run the full test suite**

Run:

```bash
python -m pytest -v
```

Expected: PASS.

- [ ] **Step 3: Run lint**

Run:

```bash
python -m ruff check .
```

Expected: PASS.

- [ ] **Step 4: Run type checking**

Run:

```bash
python -m mypy src
```

Expected: PASS.

- [ ] **Step 5: Run CLI smoke commands**

Run:

```bash
astro validate examples/scenarios/leo_two_body.yaml
astro propagate examples/scenarios/leo_two_body.yaml --backend local --output /tmp/astro-trajectory.json
astro synth-measurements examples/scenarios/leo_two_body.yaml --output /tmp/astro-measurements.json
astro estimate examples/scenarios/leo_two_body.yaml --output /tmp/astro-estimate.json
```

Expected: all commands exit 0 and write the requested files.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document orbit fd od mvp workflows"
```

## Plan Self-Review

Spec coverage:

- Scenario validation: Task 2 and Task 3.
- Python and scenario-file workflows: Task 3 and Task 5.
- LEO/MEO/GEO-capable propagation surface: Task 4 implements the initial local surface; examples start with LEO and the model accepts any valid Earth-centered Cartesian state.
- Ephemeris-like trajectory product: Task 4 and Task 5.
- Synthetic range/range-rate measurements: Task 6 and Task 8.
- Batch OD recovery with residuals, covariance, and diagnostics: Task 7 and Task 8.
- Deterministic tests: Task 9 and Task 11.
- Orekit official Python wrapper strategy: Task 10.
- Launch compatibility: this plan preserves the shared `Scenario`, `OrbitState`, and `Trajectory` spine but does not implement launch, matching the approved spec split.

Completeness scan:

- The plan contains no incomplete sections, deferred decisions, or intentionally vague implementation steps.

Type consistency:

- `Scenario`, `OrbitState`, `Trajectory`, `MeasurementRecord`, and `EstimateResult` are defined in Task 2 before they are used.
- `propagate_local` is defined in Task 4 before measurement and OD workflows use it.
- CLI imports are introduced in the tasks that add the corresponding commands.

## Execution Recommendation

Use subagent-driven execution for Tasks 1-4 only if the workspace supports isolated subagents well. Otherwise, use inline execution because the repo is small and early tasks change shared files together. The first checkpoint should be after Task 4, when scenario validation and local propagation are both passing.
