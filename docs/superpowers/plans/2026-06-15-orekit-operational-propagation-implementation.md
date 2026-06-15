# Orekit Operational Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote Orekit from an availability smoke gate to the first operational propagation backend for `astro propagate --backend orekit`.

**Architecture:** Keep Orekit/JVM imports isolated in `astro_backends.orekit`, expose a small Python adapter API, and return the existing `Trajectory` product. Start with Orekit two-body/Keplerian propagation so the first adapter proves wrapper initialization, time/frame conversion, unit conversion, CLI dispatch, provenance, and local-vs-Orekit validation before adding high-fidelity force models.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, pytest, Typer, optional `orekit-jpype`, Orekit Java API through JPype.

---

## Scope

In scope:

- `propagate_orekit(scenario: Scenario) -> Trajectory`.
- `astro propagate --backend orekit`.
- Runtime wrapper/data gate with actionable unavailable errors.
- Two-body Orekit propagation using Orekit Cartesian/Keplerian orbit objects.
- Unit conversion between suite km/km-s and Orekit m/m-s.
- Backend provenance metadata.
- Live Orekit test marker that skips cleanly without `orekit_jpype`.
- Local-vs-Orekit reference comparison for the LEO example.

Out of scope for this plan:

- Orekit numerical propagator with J2/drag/SRP.
- Orekit batch OD.
- Orekit measurement builders.
- Legacy JCC wrapper fallback.

## File Structure

Create:

- `src/astro_backends/orekit/runtime.py`
  - Owns optional dependency import, JVM init, frame/time object lookup, and clear unavailable diagnostics.
- `src/astro_backends/orekit/conversion.py`
  - Converts suite datetimes, vectors, frames, states, and sample values to/from Orekit-friendly units.
- `src/astro_backends/orekit/propagation.py`
  - Builds the Orekit initial orbit, runs propagation over suite sample epochs, and returns `Trajectory`.
- `tests/astro_backends/test_orekit_runtime.py`
  - Tests missing wrapper/import/runtime failures without needing Orekit installed.
- `tests/astro_backends/test_orekit_propagation.py`
  - Tests adapter behavior with fakes and optional live Orekit checks.

Modify:

- `src/astro_backends/orekit/smoke.py`
  - Reuse runtime loading and include data/propagation gate results when available.
- `src/astro_backends/orekit/__init__.py`
  - Export `propagate_orekit` and runtime result types.
- `src/astro_backends/__init__.py`
  - Export no backend-specific Java objects; only safe Python adapter functions if needed.
- `src/astro_cli/main.py`
  - Dispatch `--backend orekit` to `propagate_orekit`.
- `tests/astro_cli/test_cli.py`
  - Replace the current "orekit unsupported" propagation expectation with success/unavailable cases.
- `tests/test_imports.py`
  - Update public exports.
- `pyproject.toml`
  - Add a pytest marker for live Orekit tests.
- `README.md`
  - Document Orekit propagation workflow and failure modes.

## Task 1: Runtime Gate

**Files:**

- Create: `src/astro_backends/orekit/runtime.py`
- Modify: `src/astro_backends/orekit/smoke.py`
- Test: `tests/astro_backends/test_orekit_runtime.py`
- Test: `tests/astro_backends/test_orekit_smoke.py`

- [ ] **Step 1: Write failing runtime tests**

Create `tests/astro_backends/test_orekit_runtime.py`:

```python
from importlib.metadata import PackageNotFoundError

import pytest

from astro_backends.orekit.runtime import load_orekit_runtime
from astro_core.errors import UnsupportedBackendError


def test_load_orekit_runtime_reports_missing_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "astro_backends.orekit.runtime.version",
        lambda distribution: (_ for _ in ()).throw(PackageNotFoundError(distribution)),
    )

    with pytest.raises(UnsupportedBackendError, match="install astro-suite\\[orekit\\]"):
        load_orekit_runtime()


def test_load_orekit_runtime_reports_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("astro_backends.orekit.runtime.version", lambda distribution: "13.1.0")

    def fail_import(module_name: str) -> object:
        raise ImportError(f"cannot import {module_name}")

    monkeypatch.setattr("astro_backends.orekit.runtime.import_module", fail_import)

    with pytest.raises(UnsupportedBackendError, match="Orekit JPype wrapper import failed"):
        load_orekit_runtime()
```

- [ ] **Step 2: Run runtime tests to verify they fail**

Run:

```bash
python -m pytest tests/astro_backends/test_orekit_runtime.py -v
```

Expected: FAIL because `astro_backends.orekit.runtime` does not exist.

- [ ] **Step 3: Implement runtime gate**

Create `src/astro_backends/orekit/runtime.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from astro_core.errors import UnsupportedBackendError

WRAPPER = "orekit_jpype"
DISTRIBUTION = "orekit-jpype"


@dataclass(frozen=True)
class OrekitRuntime:
    wrapper: str
    wrapper_version: str
    frames_factory: Any
    time_scales_factory: Any
    absolute_date: Any
    vector3d: Any
    pv_coordinates: Any
    cartesian_orbit: Any
    keplerian_propagator: Any


def _unsupported(message: str) -> UnsupportedBackendError:
    return UnsupportedBackendError(f"Orekit backend unavailable: {message}")


def load_orekit_runtime() -> OrekitRuntime:
    try:
        wrapper_version = version(DISTRIBUTION)
    except PackageNotFoundError as exc:
        raise _unsupported("install astro-suite[orekit] to enable orekit-jpype") from exc

    try:
        orekit = import_module(WRAPPER)
    except ImportError as exc:
        raise _unsupported(f"Orekit JPype wrapper import failed: {exc}") from exc

    try:
        orekit.initVM()
        frames_module = import_module("org.orekit.frames")
        time_module = import_module("org.orekit.time")
        geometry_module = import_module("org.hipparchus.geometry.euclidean.threed")
        utils_module = import_module("org.orekit.utils")
        orbits_module = import_module("org.orekit.orbits")
        propagation_module = import_module("org.orekit.propagation.analytical")
    except Exception as exc:
        raise _unsupported(f"JVM, Orekit imports, or data context failed: {exc}") from exc

    return OrekitRuntime(
        wrapper=WRAPPER,
        wrapper_version=wrapper_version,
        frames_factory=frames_module.FramesFactory,
        time_scales_factory=time_module.TimeScalesFactory,
        absolute_date=time_module.AbsoluteDate,
        vector3d=geometry_module.Vector3D,
        pv_coordinates=utils_module.PVCoordinates,
        cartesian_orbit=orbits_module.CartesianOrbit,
        keplerian_propagator=propagation_module.KeplerianPropagator,
    )
```

- [ ] **Step 4: Route smoke through runtime**

Modify `src/astro_backends/orekit/smoke.py` so the success path calls `load_orekit_runtime()` and checks `runtime.frames_factory.getEME2000()` plus `runtime.time_scales_factory.getUTC()`.

Keep `OrekitSmokeResult.to_dict()` unchanged so existing CLI output remains stable.

- [ ] **Step 5: Run runtime and smoke tests**

Run:

```bash
python -m pytest tests/astro_backends/test_orekit_runtime.py tests/astro_backends/test_orekit_smoke.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/astro_backends/orekit/runtime.py src/astro_backends/orekit/smoke.py tests/astro_backends/test_orekit_runtime.py tests/astro_backends/test_orekit_smoke.py
git commit -m "feat: add orekit runtime gate"
```

## Task 2: Conversion Helpers

**Files:**

- Create: `src/astro_backends/orekit/conversion.py`
- Test: `tests/astro_backends/test_orekit_propagation.py`

- [ ] **Step 1: Write failing unit-conversion tests**

Add to `tests/astro_backends/test_orekit_propagation.py`:

```python
from datetime import datetime, timezone

from astro_backends.orekit.conversion import km_to_m, km_s_to_m_s, m_to_km, m_s_to_km_s


def test_orekit_unit_conversions_are_reversible() -> None:
    assert km_to_m(7.5) == 7500.0
    assert km_s_to_m_s(7.5) == 7500.0
    assert m_to_km(7500.0) == 7.5
    assert m_s_to_km_s(7500.0) == 7.5


def test_orekit_epoch_requires_utc() -> None:
    epoch = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert epoch.isoformat() == "2026-01-01T00:00:00+00:00"
```

- [ ] **Step 2: Run conversion tests to verify they fail**

Run:

```bash
python -m pytest tests/astro_backends/test_orekit_propagation.py::test_orekit_unit_conversions_are_reversible -v
```

Expected: FAIL because `astro_backends.orekit.conversion` does not exist.

- [ ] **Step 3: Implement conversion helpers**

Create `src/astro_backends/orekit/conversion.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from astro_core.models import Body, Frame, OrbitState, TimeScale

M_PER_KM = 1000.0


def km_to_m(value_km: float) -> float:
    return value_km * M_PER_KM


def m_to_km(value_m: float) -> float:
    return value_m / M_PER_KM


def km_s_to_m_s(value_km_s: float) -> float:
    return value_km_s * M_PER_KM


def m_s_to_km_s(value_m_s: float) -> float:
    return value_m_s / M_PER_KM


def validate_orekit_state_support(state: OrbitState) -> None:
    if state.time_scale is not TimeScale.UTC:
        raise ValueError("Orekit phase 1 supports only UTC scenario epochs")
    if state.frame is not Frame.EME2000:
        raise ValueError("Orekit phase 1 supports only EME2000 states")
    if state.central_body is not Body.EARTH:
        raise ValueError("Orekit phase 1 supports only Earth-centered states")


def absolute_date_from_datetime(runtime: Any, epoch: datetime) -> Any:
    utc_epoch = epoch.astimezone(timezone.utc)
    whole_second = int(utc_epoch.second)
    seconds_with_fraction = whole_second + utc_epoch.microsecond / 1_000_000.0
    utc = runtime.time_scales_factory.getUTC()
    return runtime.absolute_date(
        utc_epoch.year,
        utc_epoch.month,
        utc_epoch.day,
        utc_epoch.hour,
        utc_epoch.minute,
        seconds_with_fraction,
        utc,
    )
```

- [ ] **Step 4: Run conversion tests**

Run:

```bash
python -m pytest tests/astro_backends/test_orekit_propagation.py -v
```

Expected: PASS for conversion tests; any propagation tests added later are not present yet.

- [ ] **Step 5: Commit**

```bash
git add src/astro_backends/orekit/conversion.py tests/astro_backends/test_orekit_propagation.py
git commit -m "feat: add orekit conversion helpers"
```

## Task 3: Propagation Adapter

**Files:**

- Create: `src/astro_backends/orekit/propagation.py`
- Modify: `src/astro_backends/orekit/__init__.py`
- Test: `tests/astro_backends/test_orekit_propagation.py`
- Test: `tests/test_imports.py`

- [ ] **Step 1: Write failing unavailable-backend test**

Add to `tests/astro_backends/test_orekit_propagation.py`:

```python
from pathlib import Path

import pytest

from astro_backends.orekit.propagation import propagate_orekit
from astro_core.errors import UnsupportedBackendError
from astro_core.io import load_scenario


def test_propagate_orekit_reports_runtime_unavailable() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    def fail_loader() -> object:
        raise UnsupportedBackendError("Orekit backend unavailable: install astro-suite[orekit]")

    with pytest.raises(UnsupportedBackendError, match="install astro-suite\\[orekit\\]"):
        propagate_orekit(scenario, runtime_loader=fail_loader)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/astro_backends/test_orekit_propagation.py::test_propagate_orekit_reports_runtime_unavailable -v
```

Expected: FAIL because `astro_backends.orekit.propagation` does not exist.

- [ ] **Step 3: Implement adapter skeleton and validation**

Create `src/astro_backends/orekit/propagation.py` with:

```python
from __future__ import annotations

from datetime import timedelta
from typing import Callable, cast

from astro_core.constants import MU_EARTH_KM3_S2
from astro_core.errors import UnsupportedBackendError
from astro_core.models import (
    CartesianState,
    ForceModelName,
    Scenario,
    Trajectory,
    TrajectorySample,
)
from astro_backends.orekit.conversion import (
    absolute_date_from_datetime,
    km_s_to_m_s,
    km_to_m,
    m_s_to_km_s,
    m_to_km,
    validate_orekit_state_support,
)
from astro_backends.orekit.runtime import OrekitRuntime, load_orekit_runtime

RuntimeLoader = Callable[[], OrekitRuntime]


def _validate_orekit_phase1_scenario(scenario: Scenario) -> None:
    validate_orekit_state_support(scenario.initial_state)
    if scenario.force_model.gravity is not ForceModelName.TWO_BODY:
        raise UnsupportedBackendError(
            "Orekit propagation phase 1 supports only two_body gravity; "
            "orekit_high_fidelity and j2 require the numerical force-model phase"
        )


def propagate_orekit(
    scenario: Scenario,
    *,
    runtime_loader: RuntimeLoader = load_orekit_runtime,
) -> Trajectory:
    _validate_orekit_phase1_scenario(scenario)
    runtime = runtime_loader()
    frame = runtime.frames_factory.getEME2000()
    initial_date = absolute_date_from_datetime(runtime, scenario.initial_state.epoch)
    initial = scenario.initial_state.cartesian
    position = runtime.vector3d(
        km_to_m(initial.position_km[0]),
        km_to_m(initial.position_km[1]),
        km_to_m(initial.position_km[2]),
    )
    velocity = runtime.vector3d(
        km_s_to_m_s(initial.velocity_km_s[0]),
        km_s_to_m_s(initial.velocity_km_s[1]),
        km_s_to_m_s(initial.velocity_km_s[2]),
    )
    pv = runtime.pv_coordinates(position, velocity)
    orbit = runtime.cartesian_orbit(pv, frame, initial_date, MU_EARTH_KM3_S2 * 1.0e9)
    propagator = runtime.keplerian_propagator(orbit)

    samples: list[TrajectorySample] = []
    for step_index in range(scenario.propagation.sample_count):
        epoch = scenario.initial_state.epoch + timedelta(
            seconds=step_index * scenario.propagation.step_s
        )
        target_date = absolute_date_from_datetime(runtime, epoch)
        state = propagator.propagate(target_date)
        pv_coordinates = state.getPVCoordinates(frame)
        propagated_position = pv_coordinates.getPosition()
        propagated_velocity = pv_coordinates.getVelocity()
        samples.append(
            TrajectorySample(
                epoch=epoch,
                state=CartesianState(
                    position_km=(
                        m_to_km(float(propagated_position.getX())),
                        m_to_km(float(propagated_position.getY())),
                        m_to_km(float(propagated_position.getZ())),
                    ),
                    velocity_km_s=(
                        m_s_to_km_s(float(propagated_velocity.getX())),
                        m_s_to_km_s(float(propagated_velocity.getY())),
                        m_s_to_km_s(float(propagated_velocity.getZ())),
                    ),
                ),
            )
        )

    return Trajectory(
        scenario_id=scenario.scenario_id,
        samples=samples,
        force_model=scenario.force_model,
        backend="orekit",
        metadata={
            "wrapper": runtime.wrapper,
            "wrapper_version": runtime.wrapper_version,
            "propagator": "KeplerianPropagator",
            "frame": "EME2000",
            "units": "suite km/km_s converted to Orekit m/m_s",
        },
    )
```

- [ ] **Step 4: Export the adapter**

Modify `src/astro_backends/orekit/__init__.py`:

```python
from astro_backends.orekit.propagation import propagate_orekit
from astro_backends.orekit.runtime import OrekitRuntime, load_orekit_runtime
from astro_backends.orekit.smoke import OrekitSmokeResult, run_orekit_smoke

__all__ = [
    "OrekitRuntime",
    "OrekitSmokeResult",
    "load_orekit_runtime",
    "propagate_orekit",
    "run_orekit_smoke",
]
```

Update `tests/test_imports.py` so `astro_backends.__all__` remains empty and `astro_backends.orekit` exposes the new adapter through its own package.

- [ ] **Step 5: Run focused adapter tests**

Run:

```bash
python -m pytest tests/astro_backends/test_orekit_propagation.py tests/test_imports.py -v
```

Expected: PASS for unavailable-path and export tests.

- [ ] **Step 6: Commit**

```bash
git add src/astro_backends/orekit/propagation.py src/astro_backends/orekit/__init__.py tests/astro_backends/test_orekit_propagation.py tests/test_imports.py
git commit -m "feat: add orekit propagation adapter"
```

## Task 4: CLI Dispatch

**Files:**

- Modify: `src/astro_cli/main.py`
- Modify: `tests/astro_cli/test_cli.py`

- [ ] **Step 1: Replace unsupported-backend CLI test**

Replace `test_propagate_command_reports_unsupported_backend` with:

```python
def test_propagate_command_reports_unavailable_orekit_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_orekit(scenario: Scenario) -> object:
        raise UnsupportedBackendError("Orekit backend unavailable: install astro-suite[orekit]")

    monkeypatch.setattr("astro_cli.main.propagate_orekit", fail_orekit)

    result = runner.invoke(
        app,
        [
            "propagate",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "orekit",
        ],
    )

    assert result.exit_code == 2
    assert "Orekit backend unavailable" in result.stderr
```

Add:

```python
def test_propagate_command_writes_orekit_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "orekit.json"

    def fake_orekit(scenario: Scenario) -> object:
        trajectory = propagate_local(scenario)
        return trajectory.model_copy(update={"backend": "orekit"})

    monkeypatch.setattr("astro_cli.main.propagate_orekit", fake_orekit)

    result = runner.invoke(
        app,
        [
            "propagate",
            "examples/scenarios/leo_two_body.yaml",
            "--backend",
            "orekit",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backend"] == "orekit"
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py::test_propagate_command_reports_unavailable_orekit_backend tests/astro_cli/test_cli.py::test_propagate_command_writes_orekit_json -v
```

Expected: FAIL because `astro_cli.main` does not import or dispatch `propagate_orekit`.

- [ ] **Step 3: Implement CLI dispatch**

Modify `src/astro_cli/main.py` imports:

```python
from astro_backends.orekit import propagate_orekit, run_orekit_smoke
from astro_core.errors import UnsupportedBackendError
```

Modify `propagate`:

```python
    if backend == "local":
        trajectory = propagate_local(scenario)
    elif backend == "orekit":
        try:
            trajectory = propagate_orekit(scenario)
        except UnsupportedBackendError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc
    else:
        typer.echo(f"unsupported propagation backend: {backend}", err=True)
        raise typer.Exit(code=2)
```

Keep the output-writing logic unchanged.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/astro_cli/main.py tests/astro_cli/test_cli.py
git commit -m "feat: route propagation cli to orekit backend"
```

## Task 5: Live Orekit Validation and Documentation

**Files:**

- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `tests/astro_backends/test_orekit_propagation.py`

- [ ] **Step 1: Add live test marker**

Modify `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
markers = [
  "orekit_live: requires orekit-jpype, Java, and Orekit runtime support",
]
```

- [ ] **Step 2: Add optional live propagation test**

Add to `tests/astro_backends/test_orekit_propagation.py`:

```python
import pytest

from astro_backends.orekit.propagation import propagate_orekit
from astro_core.io import load_scenario
from astro_dynamics.local import propagate_local


@pytest.mark.orekit_live
def test_live_orekit_two_body_matches_local_reference() -> None:
    pytest.importorskip("orekit_jpype")
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    orekit_trajectory = propagate_orekit(scenario)
    local_trajectory = propagate_local(scenario)

    assert orekit_trajectory.backend == "orekit"
    assert len(orekit_trajectory.samples) == len(local_trajectory.samples)
    final_orekit = orekit_trajectory.samples[-1].state
    final_local = local_trajectory.samples[-1].state
    assert final_orekit.position_km == pytest.approx(final_local.position_km, abs=1.0)
    assert final_orekit.velocity_km_s == pytest.approx(final_local.velocity_km_s, abs=1.0e-3)
```

- [ ] **Step 3: Update README**

Add an Orekit propagation example:

```bash
python -m pip install -e '.[orekit]'
astro orekit-smoke
astro propagate examples/scenarios/leo_two_body.yaml --backend orekit --output orekit_trajectory.json
```

Document that phase 1 supports only `two_body` and that `j2`/`orekit_high_fidelity` are part of the next force-model phase.

- [ ] **Step 4: Run standard verification**

Run:

```bash
python -m pytest -v
python -m ruff check .
python -m mypy
```

Expected: PASS. The live test should be collected but skipped automatically when `orekit_jpype` is not installed.

- [ ] **Step 5: Run optional live verification when installed**

Run only when Java, `orekit_jpype`, and required Orekit runtime support are present:

```bash
python -m pip install -e '.[orekit]'
astro orekit-smoke
python -m pytest tests/astro_backends/test_orekit_propagation.py -m orekit_live -v
astro propagate examples/scenarios/leo_two_body.yaml --backend orekit --output /tmp/astro-orekit-trajectory.json
```

Expected: PASS and a trajectory product with `"backend": "orekit"`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md tests/astro_backends/test_orekit_propagation.py
git commit -m "test: add live orekit propagation validation"
```

## Self-Review

Spec coverage:

- Orekit adapter boundary: implemented through `runtime.py`, `conversion.py`, and `propagation.py`.
- Official Python wrapper path: uses `orekit_jpype`.
- Wrapper import, JVM init, frame/time access: preserved in smoke/runtime gate.
- Simple two-body propagation through wrapper: implemented with `propagate_orekit`.
- CLI backend selection: implemented through `astro propagate --backend orekit`.
- Deterministic validation: local-vs-Orekit comparison added with explicit tolerances.

Intentional gaps:

- Orekit J2, drag, SRP, measurements, and OD are deferred to the next roadmap goal because they require the propagation adapter and runtime/data setup to be stable first.

Placeholder scan:

- No steps use unspecified files.
- Every implementation step has target files, commands, and expected results.

Type consistency:

- `propagate_orekit` returns the existing `Trajectory`.
- Units remain km/km-s at the suite boundary and convert to m/m-s only inside the Orekit adapter.

