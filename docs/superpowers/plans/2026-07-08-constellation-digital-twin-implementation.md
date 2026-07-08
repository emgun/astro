# Constellation Digital Twin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a constellation digital twin workflow that runs multiple checked-in single-spacecraft twins and aggregates fleet access, revisit, link, data-volume, and margin evidence.

**Architecture:** Add constellation-specific models, IO, aggregation helpers, and a runner under `astro_twin` without duplicating single-spacecraft subsystem physics. Each member runs through `run_digital_twin`; the constellation layer clips member products to a common analysis window and returns a suite-owned `ConstellationTwinResult`.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, Typer, pytest, existing `astro_twin`, `astro_core`, and `astro_cli`.

---

## Context

Design spec: `docs/superpowers/specs/2026-07-08-constellation-digital-twin-design.md`

Branch target: stack from `codex/digital-twin-plan` until PR #3 lands, then rebase onto `main`.

First public slice:

```bash
astro run-constellation-twin examples/twin/constellation_leo_observers.yaml \
  --output /tmp/astro-constellation-twin.json \
  --summary-output /tmp/astro-constellation-twin.txt
```

Non-goals for this branch:

- No inter-satellite links.
- No contact scheduling or conflict resolution.
- No target-grid, sensor-FOV, or swath coverage maps.
- No constellation phasing optimizer.
- No optional backend or provider dependency.
- No operational coverage-authority claims.

## File Map

- Create `src/astro_twin/constellation_models.py`: constellation scenario, summary, and result schemas.
- Create `src/astro_twin/constellation.py`: runner and deterministic aggregation helpers.
- Create `src/astro_twin/constellation_io.py`: YAML/JSON loaders, result loader/writer, and summary formatter.
- Modify `src/astro_cli/main.py`: add `astro run-constellation-twin`.
- Create `examples/scenarios/leo_two_body_phase_minus_4deg.yaml`: phased member orbit example.
- Create `examples/twin/leo_observer_plane_a.yaml`: member A twin scenario.
- Create `examples/twin/leo_observer_plane_b.yaml`: member B twin scenario.
- Create `examples/twin/constellation_leo_observers.yaml`: constellation scenario.
- Create `tests/astro_twin/test_constellation_models.py`: model validation.
- Create `tests/astro_twin/test_constellation_io.py`: scenario/result IO and summary formatting.
- Create `tests/astro_twin/test_constellation_aggregation.py`: synthetic aggregation math tests.
- Create `tests/astro_twin/test_constellation_runner.py`: end-to-end runner test.
- Modify `tests/astro_cli/test_cli.py`: CLI artifact test.
- Modify `docs/digital-twin.md`: constellation usage and claim boundary.
- Modify `docs/validation-matrix.md`: required local constellation gate.
- Modify `docs/current-state.md`: record review-ready scope and verification evidence after gates pass.

## Task 1: Constellation Models

**Files:**
- Create: `src/astro_twin/constellation_models.py`
- Create: `tests/astro_twin/test_constellation_models.py`

- [ ] **Step 1: Write failing model validation tests**

```python
import pytest
from pydantic import ValidationError

from astro_twin.constellation_models import (
    ConstellationCoverageRequirement,
    ConstellationMemberConfig,
    ConstellationTwinScenario,
)


def test_constellation_twin_scenario_accepts_members_and_requirements() -> None:
    scenario = ConstellationTwinScenario(
        scenario_id="leo-observers",
        members=(
            ConstellationMemberConfig(
                name="plane-a",
                twin_scenario="examples/twin/leo_observer_plane_a.yaml",
            ),
            ConstellationMemberConfig(
                name="plane-b",
                twin_scenario="examples/twin/leo_observer_plane_b.yaml",
            ),
        ),
        coverage_requirements=(
            ConstellationCoverageRequirement(
                ground_site="equator-eci",
                minimum_coverage_fraction=0.25,
                maximum_revisit_gap_s=300.0,
            ),
        ),
    )

    assert scenario.scenario_id == "leo-observers"
    assert len(scenario.members) == 2
    assert scenario.coverage_requirements[0].maximum_revisit_gap_s == 300.0


def test_constellation_twin_scenario_rejects_duplicate_member_names() -> None:
    with pytest.raises(ValidationError, match="member names must be unique"):
        ConstellationTwinScenario(
            scenario_id="leo-observers",
            members=(
                ConstellationMemberConfig(name="plane-a", twin_scenario="a.yaml"),
                ConstellationMemberConfig(name="plane-a", twin_scenario="b.yaml"),
            ),
        )


def test_constellation_twin_scenario_rejects_duplicate_requirements() -> None:
    with pytest.raises(
        ValidationError,
        match="coverage requirement ground_site values must be unique",
    ):
        ConstellationTwinScenario(
            scenario_id="leo-observers",
            members=(
                ConstellationMemberConfig(name="plane-a", twin_scenario="a.yaml"),
            ),
            coverage_requirements=(
                ConstellationCoverageRequirement(ground_site="equator-eci"),
                ConstellationCoverageRequirement(ground_site="equator-eci"),
            ),
        )
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/astro_twin/test_constellation_models.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'astro_twin.constellation_models'`.

- [ ] **Step 3: Implement constellation models**

Create `src/astro_twin/constellation_models.py`:

```python
from __future__ import annotations

from typing import Any

from pydantic import Field, FiniteFloat, model_validator

from astro_core.models import AstroModel
from astro_twin.models import DesignMarginReport, DigitalTwinResult


class ConstellationMemberConfig(AstroModel):
    name: str = Field(min_length=1)
    twin_scenario: str = Field(min_length=1)


class ConstellationCoverageRequirement(AstroModel):
    ground_site: str = Field(min_length=1)
    minimum_coverage_fraction: FiniteFloat = Field(ge=0.0, le=1.0, default=0.0)
    maximum_revisit_gap_s: FiniteFloat | None = Field(default=None, gt=0.0)


class ConstellationTwinScenario(AstroModel):
    scenario_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    members: tuple[ConstellationMemberConfig, ...] = Field(min_length=1)
    coverage_requirements: tuple[ConstellationCoverageRequirement, ...] = Field(
        default_factory=tuple
    )

    @model_validator(mode="after")
    def names_must_be_unique(self) -> ConstellationTwinScenario:
        member_names = [member.name for member in self.members]
        if len(set(member_names)) != len(member_names):
            raise ValueError("member names must be unique")
        requirement_sites = [requirement.ground_site for requirement in self.coverage_requirements]
        if len(set(requirement_sites)) != len(requirement_sites):
            raise ValueError("coverage requirement ground_site values must be unique")
        return self


class FleetAccessSummary(AstroModel):
    ground_site: str
    total_access_duration_s: FiniteFloat = Field(ge=0.0)
    longest_gap_s: FiniteFloat = Field(ge=0.0)
    mean_gap_s: FiniteFloat = Field(ge=0.0)
    max_simultaneous_spacecraft: int = Field(ge=0)
    coverage_fraction: FiniteFloat = Field(ge=0.0, le=1.0)


class FleetLinkSummary(AstroModel):
    ground_site: str
    total_data_volume_mbit: FiniteFloat = Field(ge=0.0)
    worst_ebn0_margin_db: float | None = None


class MemberLinkSummary(AstroModel):
    member_name: str
    total_data_volume_mbit: FiniteFloat = Field(ge=0.0)
    worst_ebn0_margin_db: float | None = None


class MemberTwinResult(AstroModel):
    member_name: str
    result: DigitalTwinResult


class ConstellationTwinResult(AstroModel):
    scenario_id: str
    workflow: str = "constellation_digital_twin_v1"
    members: tuple[MemberTwinResult, ...]
    access_summaries: tuple[FleetAccessSummary, ...]
    link_summaries: tuple[FleetLinkSummary, ...]
    member_link_summaries: tuple[MemberLinkSummary, ...]
    fleet_margin_report: DesignMarginReport
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run model tests**

Run:

```bash
python -m pytest tests/astro_twin/test_constellation_models.py -q
```

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/astro_twin/constellation_models.py tests/astro_twin/test_constellation_models.py
git commit -m "Add constellation twin models"
```

## Task 2: Constellation IO And Summary Formatting

**Files:**
- Create: `src/astro_twin/constellation_io.py`
- Create: `tests/astro_twin/test_constellation_io.py`

- [ ] **Step 1: Write failing IO tests**

```python
import json
from pathlib import Path

import yaml

from astro_twin.constellation_io import (
    format_constellation_summary,
    load_constellation_twin_result,
    load_constellation_twin_scenario,
    write_constellation_twin_result,
)
from astro_twin.constellation_models import ConstellationTwinResult
from astro_twin.models import DesignMargin, DesignMarginReport, TwinMarginStatus


def test_load_constellation_twin_scenario_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "constellation.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "scenario_id": "leo-observers",
                "members": [
                    {"name": "plane-a", "twin_scenario": "a.yaml"},
                    {"name": "plane-b", "twin_scenario": "b.yaml"},
                ],
                "coverage_requirements": [
                    {
                        "ground_site": "equator-eci",
                        "minimum_coverage_fraction": 0.25,
                        "maximum_revisit_gap_s": 300.0,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    scenario = load_constellation_twin_scenario(path)

    assert scenario.scenario_id == "leo-observers"
    assert scenario.members[1].name == "plane-b"
    assert scenario.coverage_requirements[0].minimum_coverage_fraction == 0.25


def test_load_constellation_twin_scenario_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    try:
        load_constellation_twin_scenario(path)
    except Exception as exc:
        assert "must contain a mapping" in str(exc)
    else:
        raise AssertionError("expected invalid scenario error")


def test_write_load_and_format_constellation_result(tmp_path: Path) -> None:
    result = ConstellationTwinResult(
        scenario_id="leo-observers",
        members=(),
        access_summaries=(),
        link_summaries=(),
        member_link_summaries=(),
        fleet_margin_report=DesignMarginReport(
            margins=(
                DesignMargin(
                    name="fleet_link_margin_db_equator-eci",
                    value=3.0,
                    threshold=0.0,
                    margin=3.0,
                    status=TwinMarginStatus.WARN,
                ),
            ),
            limiting_margin=DesignMargin(
                name="fleet_link_margin_db_equator-eci",
                value=3.0,
                threshold=0.0,
                margin=3.0,
                status=TwinMarginStatus.WARN,
            ),
        ),
        metadata={"analysis_window_s": {"start_s": 0.0, "end_s": 600.0}},
        warnings=["screening only"],
    )
    output = tmp_path / "result.json"

    write_constellation_twin_result(output, result)
    loaded = load_constellation_twin_result(output)
    summary = format_constellation_summary(loaded)

    assert json.loads(output.read_text(encoding="utf-8"))["workflow"] == (
        "constellation_digital_twin_v1"
    )
    assert loaded.scenario_id == "leo-observers"
    assert "Constellation twin: leo-observers" in summary
    assert "Limiting fleet margin:" in summary
    assert "screening only" in summary
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/astro_twin/test_constellation_io.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'astro_twin.constellation_io'`.

- [ ] **Step 3: Implement IO functions**

Create `src/astro_twin/constellation_io.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_twin.constellation_models import ConstellationTwinResult, ConstellationTwinScenario


def load_constellation_twin_scenario(path: Path | str) -> ConstellationTwinScenario:
    scenario_path = Path(path)
    try:
        raw: Any = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read constellation twin scenario {scenario_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(
            f"Could not parse constellation twin scenario {scenario_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Constellation twin scenario file {scenario_path} must contain a mapping"
        )
    try:
        return ConstellationTwinScenario.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Constellation twin scenario file {scenario_path} is invalid: {exc}"
        ) from exc


def write_constellation_twin_result(path: Path | str, result: ConstellationTwinResult) -> None:
    Path(path).write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_constellation_twin_result(path: Path | str) -> ConstellationTwinResult:
    result_path = Path(path)
    try:
        raw: Any = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read constellation twin result {result_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise InvalidScenarioError(
            f"Could not parse constellation twin result {result_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Constellation twin result file {result_path} must contain a JSON object"
        )
    try:
        return ConstellationTwinResult.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Constellation twin result file {result_path} is invalid: {exc}"
        ) from exc


def format_constellation_summary(result: ConstellationTwinResult) -> str:
    analysis_window = result.metadata.get("analysis_window_s", {})
    if isinstance(analysis_window, dict):
        start_s = analysis_window.get("start_s", "unknown")
        end_s = analysis_window.get("end_s", "unknown")
    else:
        start_s = "unknown"
        end_s = "unknown"
    total_data_mbit = sum(summary.total_data_volume_mbit for summary in result.link_summaries)
    max_simultaneous = max(
        (summary.max_simultaneous_spacecraft for summary in result.access_summaries),
        default=0,
    )
    lines = [
        f"Constellation twin: {result.scenario_id}",
        f"Workflow: {result.workflow}",
        f"Members: {len(result.members)}",
        f"Analysis window s: {start_s} to {end_s}",
        f"Fleet access summaries: {len(result.access_summaries)}",
        f"Total data volume Mbit: {total_data_mbit:.3f}",
        f"Max simultaneous spacecraft: {max_simultaneous}",
        (
            "Limiting fleet margin: "
            f"{result.fleet_margin_report.limiting_margin.name} = "
            f"{result.fleet_margin_report.limiting_margin.margin:.3f}"
        ),
    ]
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)
```

- [ ] **Step 4: Run IO tests**

Run:

```bash
python -m pytest tests/astro_twin/test_constellation_io.py -q
```

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/astro_twin/constellation_io.py tests/astro_twin/test_constellation_io.py
git commit -m "Add constellation twin IO"
```

## Task 3: Aggregation Helpers And Fleet Margins

**Files:**
- Create: `src/astro_twin/constellation.py`
- Create: `tests/astro_twin/test_constellation_aggregation.py`

- [ ] **Step 1: Write failing aggregation tests**

```python
from astro_twin.constellation import (
    aggregate_access_summaries,
    aggregate_link_summaries,
    build_fleet_margin_report,
)
from astro_twin.constellation_models import ConstellationCoverageRequirement, MemberLinkSummary
from astro_twin.models import AccessWindow, DesignMargin, LinkBudgetWindow, TwinMarginStatus


def test_aggregate_access_summaries_computes_union_gaps_and_simultaneous_count() -> None:
    summaries = aggregate_access_summaries(
        member_access_windows={
            "plane-a": (
                AccessWindow(
                    ground_site="equator-eci",
                    start_s=0.0,
                    end_s=120.0,
                    duration_s=120.0,
                    max_elevation_deg=80.0,
                    min_range_km=700.0,
                ),
                AccessWindow(
                    ground_site="equator-eci",
                    start_s=300.0,
                    end_s=420.0,
                    duration_s=120.0,
                    max_elevation_deg=70.0,
                    min_range_km=900.0,
                ),
            ),
            "plane-b": (
                AccessWindow(
                    ground_site="equator-eci",
                    start_s=60.0,
                    end_s=180.0,
                    duration_s=120.0,
                    max_elevation_deg=75.0,
                    min_range_km=800.0,
                ),
            ),
        },
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.ground_site == "equator-eci"
    assert summary.total_access_duration_s == 300.0
    assert summary.longest_gap_s == 180.0
    assert summary.mean_gap_s == 150.0
    assert summary.max_simultaneous_spacecraft == 2
    assert summary.coverage_fraction == 0.5


def test_aggregate_link_summaries_groups_by_site_and_member() -> None:
    fleet, members = aggregate_link_summaries(
        member_link_windows={
            "plane-a": (
                LinkBudgetWindow(
                    link_name="xband-a",
                    ground_site="equator-eci",
                    start_s=0.0,
                    end_s=60.0,
                    duration_s=60.0,
                    worst_ebn0_margin_db=4.0,
                    data_volume_mbit=120.0,
                ),
            ),
            "plane-b": (
                LinkBudgetWindow(
                    link_name="xband-b",
                    ground_site="equator-eci",
                    start_s=120.0,
                    end_s=180.0,
                    duration_s=60.0,
                    worst_ebn0_margin_db=2.0,
                    data_volume_mbit=120.0,
                ),
            ),
        },
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )

    assert fleet[0].ground_site == "equator-eci"
    assert fleet[0].total_data_volume_mbit == 240.0
    assert fleet[0].worst_ebn0_margin_db == 2.0
    assert members == (
        MemberLinkSummary(
            member_name="plane-a",
            total_data_volume_mbit=120.0,
            worst_ebn0_margin_db=4.0,
        ),
        MemberLinkSummary(
            member_name="plane-b",
            total_data_volume_mbit=120.0,
            worst_ebn0_margin_db=2.0,
        ),
    )


def test_build_fleet_margin_report_uses_coverage_link_and_member_margins() -> None:
    access_summaries = aggregate_access_summaries(
        member_access_windows={"plane-a": ()},
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )
    link_summaries, _ = aggregate_link_summaries(
        member_link_windows={"plane-a": ()},
        analysis_start_s=0.0,
        analysis_end_s=600.0,
    )

    report = build_fleet_margin_report(
        access_summaries=access_summaries,
        link_summaries=link_summaries,
        coverage_requirements=(
            ConstellationCoverageRequirement(
                ground_site="equator-eci",
                minimum_coverage_fraction=0.25,
                maximum_revisit_gap_s=300.0,
            ),
        ),
        member_limiting_margins={
            "plane-a": DesignMargin(
                name="mass_margin_fraction",
                value=0.24,
                threshold=0.2,
                margin=0.04,
                status=TwinMarginStatus.WARN,
            )
        },
    )

    assert report.limiting_margin.name == "fleet_coverage_fraction_equator-eci"
    assert report.limiting_margin.status is TwinMarginStatus.FAIL
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/astro_twin/test_constellation_aggregation.py -q
```

Expected: FAIL because `astro_twin.constellation` does not exist.

- [ ] **Step 3: Implement aggregation helpers**

Create `src/astro_twin/constellation.py` with these helpers first. The runner is added in Task 4.

```python
from __future__ import annotations

from collections import defaultdict
from math import fsum

from astro_twin.constellation_models import (
    ConstellationCoverageRequirement,
    FleetAccessSummary,
    FleetLinkSummary,
    MemberLinkSummary,
)
from astro_twin.models import (
    AccessWindow,
    DesignMargin,
    DesignMarginReport,
    LinkBudgetWindow,
    TwinMarginStatus,
)


_STATUS_SEVERITY = {
    TwinMarginStatus.FAIL: 0,
    TwinMarginStatus.WARN: 1,
    TwinMarginStatus.PASS: 2,
}


def aggregate_access_summaries(
    *,
    member_access_windows: dict[str, tuple[AccessWindow, ...]],
    analysis_start_s: float,
    analysis_end_s: float,
) -> tuple[FleetAccessSummary, ...]:
    ground_sites = sorted(
        {
            window.ground_site
            for windows in member_access_windows.values()
            for window in windows
        }
    )
    if not ground_sites:
        ground_sites = ["equator-eci"]
    return tuple(
        _access_summary_for_site(
            site,
            tuple(
                window
                for windows in member_access_windows.values()
                for window in windows
                if window.ground_site == site
            ),
            analysis_start_s=analysis_start_s,
            analysis_end_s=analysis_end_s,
        )
        for site in ground_sites
    )


def aggregate_link_summaries(
    *,
    member_link_windows: dict[str, tuple[LinkBudgetWindow, ...]],
    analysis_start_s: float,
    analysis_end_s: float,
) -> tuple[tuple[FleetLinkSummary, ...], tuple[MemberLinkSummary, ...]]:
    by_site: dict[str, list[LinkBudgetWindow]] = defaultdict(list)
    member_summaries: list[MemberLinkSummary] = []
    for member_name, windows in sorted(member_link_windows.items()):
        clipped = tuple(
            window
            for window in windows
            if _overlaps(window.start_s, window.end_s, analysis_start_s, analysis_end_s)
        )
        for window in clipped:
            by_site[window.ground_site].append(window)
        member_summaries.append(
            MemberLinkSummary(
                member_name=member_name,
                total_data_volume_mbit=fsum(window.data_volume_mbit for window in clipped),
                worst_ebn0_margin_db=(
                    min(window.worst_ebn0_margin_db for window in clipped) if clipped else None
                ),
            )
        )
    fleet_summaries = tuple(
        FleetLinkSummary(
            ground_site=site,
            total_data_volume_mbit=fsum(window.data_volume_mbit for window in windows),
            worst_ebn0_margin_db=min(window.worst_ebn0_margin_db for window in windows),
        )
        for site, windows in sorted(by_site.items())
    )
    return fleet_summaries, tuple(member_summaries)


def build_fleet_margin_report(
    *,
    access_summaries: tuple[FleetAccessSummary, ...],
    link_summaries: tuple[FleetLinkSummary, ...],
    coverage_requirements: tuple[ConstellationCoverageRequirement, ...],
    member_limiting_margins: dict[str, DesignMargin],
) -> DesignMarginReport:
    requirement_index = {
        requirement.ground_site: requirement for requirement in coverage_requirements
    }
    margins: list[DesignMargin] = []
    for summary in access_summaries:
        requirement = requirement_index.get(summary.ground_site)
        coverage_threshold = (
            requirement.minimum_coverage_fraction if requirement is not None else 0.0
        )
        coverage_margin = summary.coverage_fraction - coverage_threshold
        margins.append(
            DesignMargin(
                name=f"fleet_coverage_fraction_{summary.ground_site}",
                value=summary.coverage_fraction,
                threshold=coverage_threshold,
                margin=coverage_margin,
                status=_status(coverage_margin, warn_threshold=0.05),
            )
        )
        if requirement is not None and requirement.maximum_revisit_gap_s is not None:
            gap_margin = requirement.maximum_revisit_gap_s - summary.longest_gap_s
            margins.append(
                DesignMargin(
                    name=f"fleet_longest_gap_s_{summary.ground_site}",
                    value=summary.longest_gap_s,
                    threshold=requirement.maximum_revisit_gap_s,
                    margin=gap_margin,
                    status=_status(gap_margin, warn_threshold=60.0),
                )
            )
    for summary in link_summaries:
        link_margin = (
            summary.worst_ebn0_margin_db
            if summary.worst_ebn0_margin_db is not None
            else -1.0
        )
        margins.append(
            DesignMargin(
                name=f"fleet_link_margin_db_{summary.ground_site}",
                value=link_margin,
                threshold=0.0,
                margin=link_margin,
                status=_status(link_margin, warn_threshold=3.0),
            )
        )
    for member_name, margin in sorted(member_limiting_margins.items()):
        margins.append(
            DesignMargin(
                name=f"member_{member_name}_{margin.name}",
                value=margin.value,
                threshold=margin.threshold,
                margin=margin.margin,
                status=margin.status,
            )
        )
    limiting_margin = min(margins, key=_limiting_key)
    return DesignMarginReport(margins=tuple(margins), limiting_margin=limiting_margin)


def _access_summary_for_site(
    site: str,
    windows: tuple[AccessWindow, ...],
    *,
    analysis_start_s: float,
    analysis_end_s: float,
) -> FleetAccessSummary:
    events: list[tuple[float, int]] = []
    for window in windows:
        start_s = max(analysis_start_s, window.start_s)
        end_s = min(analysis_end_s, window.end_s)
        if end_s <= start_s:
            continue
        events.append((start_s, 1))
        events.append((end_s, -1))
    events.sort(key=lambda item: (item[0], -item[1]))
    total_access_s = 0.0
    max_count = 0
    count = 0
    previous_s = analysis_start_s
    gaps: list[float] = []
    active_gap_start = analysis_start_s
    for event_s, delta in events:
        if count > 0:
            total_access_s += event_s - previous_s
        elif event_s > active_gap_start:
            gaps.append(event_s - active_gap_start)
        count += delta
        max_count = max(max_count, count)
        previous_s = event_s
        if count == 0:
            active_gap_start = event_s
    if count > 0:
        total_access_s += analysis_end_s - previous_s
    elif analysis_end_s > active_gap_start:
        gaps.append(analysis_end_s - active_gap_start)
    duration_s = analysis_end_s - analysis_start_s
    return FleetAccessSummary(
        ground_site=site,
        total_access_duration_s=total_access_s,
        longest_gap_s=max(gaps, default=0.0),
        mean_gap_s=fsum(gaps) / len(gaps) if gaps else 0.0,
        max_simultaneous_spacecraft=max_count,
        coverage_fraction=total_access_s / duration_s if duration_s > 0.0 else 0.0,
    )


def _overlaps(start_s: float, end_s: float, analysis_start_s: float, analysis_end_s: float) -> bool:
    return min(end_s, analysis_end_s) > max(start_s, analysis_start_s)


def _status(margin: float, warn_threshold: float) -> TwinMarginStatus:
    if margin < 0.0:
        return TwinMarginStatus.FAIL
    if margin <= warn_threshold:
        return TwinMarginStatus.WARN
    return TwinMarginStatus.PASS


def _limiting_key(margin: DesignMargin) -> tuple[int, float]:
    normalizer = abs(margin.threshold) if margin.threshold != 0.0 else 1.0
    return _STATUS_SEVERITY[margin.status], margin.margin / normalizer
```

- [ ] **Step 4: Run aggregation tests**

Run:

```bash
python -m pytest tests/astro_twin/test_constellation_aggregation.py -q
```

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/astro_twin/constellation.py tests/astro_twin/test_constellation_aggregation.py
git commit -m "Add constellation twin aggregation"
```

## Task 4: Runner And Reference Examples

**Files:**
- Modify: `src/astro_twin/constellation.py`
- Create: `examples/scenarios/leo_two_body_phase_minus_4deg.yaml`
- Create: `examples/twin/leo_observer_plane_a.yaml`
- Create: `examples/twin/leo_observer_plane_b.yaml`
- Create: `examples/twin/constellation_leo_observers.yaml`
- Create: `tests/astro_twin/test_constellation_runner.py`

- [ ] **Step 1: Write failing runner test**

```python
from astro_twin.constellation import run_constellation_twin
from astro_twin.constellation_io import load_constellation_twin_scenario


def test_run_constellation_twin_returns_fleet_result() -> None:
    scenario = load_constellation_twin_scenario(
        "examples/twin/constellation_leo_observers.yaml"
    )

    result = run_constellation_twin(scenario)

    assert result.workflow == "constellation_digital_twin_v1"
    assert len(result.members) == 2
    assert result.metadata["analysis_window_s"] == {"start_s": 0.0, "end_s": 600.0}
    assert result.access_summaries
    assert result.link_summaries
    assert result.member_link_summaries
    assert result.fleet_margin_report.limiting_margin.name
    assert any("design-screening" in warning for warning in result.warnings)
```

- [ ] **Step 2: Add reference example files**

Create `examples/scenarios/leo_two_body_phase_minus_4deg.yaml`:

```yaml
scenario_id: leo-two-body-phase-minus-4deg
description: Deterministic slightly phased LEO two-body propagation example for constellation twin tests.
spacecraft:
  name: demo-sat-phase-minus-4
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
    position_km: [6982.948351818769, -488.2953162088771, 0.0]
    velocity_km_s: [0.5231735530809398, 7.481730376948682, 1.0]
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
    seed: 43
```

Create `examples/twin/leo_observer_plane_a.yaml`:

```yaml
scenario_id: leo-observer-plane-a
orbit_scenario: examples/scenarios/leo_two_body.yaml
spacecraft:
  name: ObserverSat-A
  dry_mass_kg: 120.0
  payload_mass_kg: 25.0
  propellant_mass_kg: 45.0
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
  - name: equator-eci
    latitude_deg: 0.0
    longitude_deg: 0.0
    altitude_m: 0.0
    minimum_elevation_deg: 10.0
links:
  - name: xband-downlink-a
    ground_site: equator-eci
    frequency_ghz: 8.4
    eirp_dbw: 18.0
    receiver_g_over_t_db_k: 22.0
    data_rate_bps: 2000000.0
    required_ebn0_db: 6.5
    implementation_loss_db: 2.0
mode_schedule:
  - mode: payload
    start_s: 60.0
    end_s: 180.0
  - mode: downlink
    start_s: 0.0
    end_s: 240.0
```

Create `examples/twin/leo_observer_plane_b.yaml`:

```yaml
scenario_id: leo-observer-plane-b
orbit_scenario: examples/scenarios/leo_two_body_phase_minus_4deg.yaml
spacecraft:
  name: ObserverSat-B
  dry_mass_kg: 120.0
  payload_mass_kg: 25.0
  propellant_mass_kg: 45.0
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
  - name: equator-eci
    latitude_deg: 0.0
    longitude_deg: 0.0
    altitude_m: 0.0
    minimum_elevation_deg: 10.0
links:
  - name: xband-downlink-b
    ground_site: equator-eci
    frequency_ghz: 8.4
    eirp_dbw: 18.0
    receiver_g_over_t_db_k: 22.0
    data_rate_bps: 2000000.0
    required_ebn0_db: 6.5
    implementation_loss_db: 2.0
mode_schedule:
  - mode: payload
    start_s: 60.0
    end_s: 180.0
  - mode: downlink
    start_s: 0.0
    end_s: 240.0
```

Create `examples/twin/constellation_leo_observers.yaml`:

```yaml
scenario_id: leo-observers
members:
  - name: plane-a
    twin_scenario: examples/twin/leo_observer_plane_a.yaml
  - name: plane-b
    twin_scenario: examples/twin/leo_observer_plane_b.yaml
coverage_requirements:
  - ground_site: equator-eci
    minimum_coverage_fraction: 0.25
    maximum_revisit_gap_s: 300.0
```

- [ ] **Step 3: Run test and verify failure**

Run:

```bash
python -m pytest tests/astro_twin/test_constellation_runner.py -q
```

Expected: FAIL because `run_constellation_twin` is missing.

- [ ] **Step 4: Implement runner**

Update the top import block in `src/astro_twin/constellation.py` to include all imports needed by
the aggregation helpers and runner:

```python
from __future__ import annotations

from collections import defaultdict
from math import fsum

from astro_core.errors import InvalidScenarioError
from astro_twin.constellation_models import (
    ConstellationCoverageRequirement,
    ConstellationTwinResult,
    ConstellationTwinScenario,
    FleetAccessSummary,
    FleetLinkSummary,
    MemberLinkSummary,
    MemberTwinResult,
)
from astro_twin.io import load_twin_scenario
from astro_twin.models import (
    AccessWindow,
    DesignMargin,
    DesignMarginReport,
    LinkBudgetWindow,
    TwinMarginStatus,
)
from astro_twin.runner import run_digital_twin
```

Add the runner functions below the aggregation helpers in `src/astro_twin/constellation.py`:

```python
_CONSTELLATION_WARNING = (
    "Constellation digital twin v1 is deterministic design-screening evidence, "
    "not operational constellation coverage authority."
)


def run_constellation_twin(scenario: ConstellationTwinScenario) -> ConstellationTwinResult:
    member_results: list[MemberTwinResult] = []
    for member in scenario.members:
        twin_scenario = load_twin_scenario(member.twin_scenario)
        twin_result = run_digital_twin(twin_scenario)
        if not twin_result.geometry:
            raise InvalidScenarioError(
                f"Constellation member {member.name} has no geometry samples"
            )
        member_results.append(MemberTwinResult(member_name=member.name, result=twin_result))

    analysis_start_s, analysis_end_s = _common_analysis_window(tuple(member_results))
    available_sites = {
        window.ground_site
        for member_result in member_results
        for window in member_result.result.access_windows
    }
    for requirement in scenario.coverage_requirements:
        if requirement.ground_site not in available_sites:
            raise InvalidScenarioError(
                "coverage requirement ground_site must appear in at least one member access window"
            )

    member_access_windows = {
        member.member_name: member.result.access_windows for member in member_results
    }
    member_link_windows = {
        member.member_name: member.result.link_windows for member in member_results
    }
    access_summaries = aggregate_access_summaries(
        member_access_windows=member_access_windows,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
    )
    link_summaries, member_link_summaries = aggregate_link_summaries(
        member_link_windows=member_link_windows,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
    )
    fleet_margin_report = build_fleet_margin_report(
        access_summaries=access_summaries,
        link_summaries=link_summaries,
        coverage_requirements=scenario.coverage_requirements,
        member_limiting_margins={
            member.member_name: member.result.margin_report.limiting_margin
            for member in member_results
        },
    )
    return ConstellationTwinResult(
        scenario_id=scenario.scenario_id,
        members=tuple(member_results),
        access_summaries=access_summaries,
        link_summaries=link_summaries,
        member_link_summaries=member_link_summaries,
        fleet_margin_report=fleet_margin_report,
        metadata={
            "analysis_window_s": {
                "start_s": analysis_start_s,
                "end_s": analysis_end_s,
            }
        },
        warnings=[_CONSTELLATION_WARNING],
    )


def _common_analysis_window(members: tuple[MemberTwinResult, ...]) -> tuple[float, float]:
    start_s = max(member.result.geometry[0].elapsed_s for member in members)
    end_s = min(member.result.geometry[-1].elapsed_s for member in members)
    if end_s <= start_s:
        raise InvalidScenarioError("Constellation members must share an overlapping analysis window")
    return start_s, end_s
```

- [ ] **Step 5: Run focused runner test**

Run:

```bash
python -m pytest tests/astro_twin/test_constellation_runner.py -q
```

Expected: PASS with `1 passed`.

- [ ] **Step 6: Run constellation tests**

Run:

```bash
python -m pytest tests/astro_twin/test_constellation_models.py tests/astro_twin/test_constellation_io.py tests/astro_twin/test_constellation_aggregation.py tests/astro_twin/test_constellation_runner.py -q
```

Expected: all constellation tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/astro_twin/constellation.py examples/scenarios/leo_two_body_phase_minus_4deg.yaml examples/twin/leo_observer_plane_a.yaml examples/twin/leo_observer_plane_b.yaml examples/twin/constellation_leo_observers.yaml tests/astro_twin/test_constellation_runner.py
git commit -m "Add constellation twin runner and examples"
```

## Task 5: CLI Command

**Files:**
- Modify: `src/astro_cli/main.py`
- Modify: `tests/astro_cli/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

Add this test after `test_run_twin_command_writes_json_and_summary` in `tests/astro_cli/test_cli.py`:

```python
def test_run_constellation_twin_command_writes_json_and_summary(tmp_path: Path) -> None:
    output = tmp_path / "constellation-twin.json"
    summary = tmp_path / "constellation-twin.txt"

    result = runner.invoke(
        app,
        [
            "run-constellation-twin",
            "examples/twin/constellation_leo_observers.yaml",
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["workflow"] == "constellation_digital_twin_v1"
    assert len(payload["members"]) == 2
    summary_text = summary.read_text(encoding="utf-8")
    assert "Constellation twin: leo-observers" in summary_text
    assert "Limiting fleet margin:" in summary_text
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py::test_run_constellation_twin_command_writes_json_and_summary -q
```

Expected: FAIL because the command is not registered.

- [ ] **Step 3: Add CLI imports**

Modify the import block in `src/astro_cli/main.py`:

```python
from astro_twin.constellation import run_constellation_twin
from astro_twin.constellation_io import (
    format_constellation_summary,
    load_constellation_twin_scenario,
)
```

- [ ] **Step 4: Add CLI command**

Add this command after `run_twin`:

```python
@app.command("run-constellation-twin")
def run_constellation_twin_command(
    scenario_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Constellation twin scenario YAML path."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Write constellation digital twin JSON result."),
    ],
    summary_output: Annotated[
        Path | None,
        typer.Option("--summary-output", help="Write a concise constellation text summary."),
    ] = None,
) -> None:
    """Run the deterministic constellation digital twin workflow."""
    try:
        scenario = load_constellation_twin_scenario(scenario_path)
        result = run_constellation_twin(scenario)
    except InvalidScenarioError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_text_or_exit(output, result.model_dump_json(indent=2), "constellation twin result")
    typer.echo(f"wrote constellation twin result: {output}")
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        _write_text_or_exit(
            summary_output,
            format_constellation_summary(result),
            "constellation twin summary",
        )
        typer.echo(f"wrote constellation twin summary: {summary_output}")
```

- [ ] **Step 5: Run CLI test**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py::test_run_constellation_twin_command_writes_json_and_summary -q
```

Expected: PASS.

- [ ] **Step 6: Run public command**

Run:

```bash
astro run-constellation-twin examples/twin/constellation_leo_observers.yaml \
  --output /tmp/astro-constellation-twin.json \
  --summary-output /tmp/astro-constellation-twin.txt
sed -n '1,120p' /tmp/astro-constellation-twin.txt
```

Expected: command exits 0 and summary includes `Members: 2`, `Fleet access summaries:`, and
`Limiting fleet margin:`.

- [ ] **Step 7: Commit**

```bash
git add src/astro_cli/main.py tests/astro_cli/test_cli.py
git commit -m "Add constellation twin CLI"
```

## Task 6: Documentation, State, And Final Gates

**Files:**
- Modify: `docs/digital-twin.md`
- Modify: `docs/validation-matrix.md`
- Modify: `docs/current-state.md`

- [ ] **Step 1: Update digital twin docs**

Add this section to `docs/digital-twin.md` after the single-spacecraft validation section:

````markdown
## Constellation Twin

The constellation twin workflow runs multiple single-spacecraft twin scenarios and aggregates fleet
access, revisit, link-budget, data-volume, and margin evidence.

```bash
astro run-constellation-twin examples/twin/constellation_leo_observers.yaml \
  --output /tmp/astro-constellation-twin.json \
  --summary-output /tmp/astro-constellation-twin.txt
```

The v1 constellation product is deterministic design-screening evidence. It is not operational
coverage authority, contact scheduling, spectrum coordination, collision avoidance, or constellation
optimization.
````

- [ ] **Step 2: Update validation matrix**

Add this row to `docs/validation-matrix.md` under `Required Local Gates`, near the integrated
digital twin row:

```markdown
| Constellation digital twin | `astro run-constellation-twin examples/twin/constellation_leo_observers.yaml --output /tmp/astro-constellation-twin.json --summary-output /tmp/astro-constellation-twin.txt` and `python -m pytest tests/astro_twin/test_constellation_models.py tests/astro_twin/test_constellation_io.py tests/astro_twin/test_constellation_aggregation.py tests/astro_twin/test_constellation_runner.py -q` | Writes a suite-owned multi-spacecraft constellation screening product with embedded member twin results plus fleet access, revisit, link, data-volume, and margin evidence. This is deterministic design-screening evidence, not operational constellation coverage authority. |
```

- [ ] **Step 3: Update current state**

Update `docs/current-state.md`:

- Add the constellation implementation surface under implemented digital-twin state.
- Set or add `constellation-digital-twin` in the active work registry as `review-ready`.
- Record exact gate outputs after Step 4 below, including the public CLI summary facts, focused
  constellation tests, full pytest count, skipped count, mypy source count, and build artifacts.
- Keep claim boundaries explicit.

- [ ] **Step 4: Run final verification**

Run:

```bash
python -m pytest tests/astro_twin/test_constellation_models.py tests/astro_twin/test_constellation_io.py tests/astro_twin/test_constellation_aggregation.py tests/astro_twin/test_constellation_runner.py tests/astro_cli/test_cli.py::test_run_constellation_twin_command_writes_json_and_summary -q
astro run-constellation-twin examples/twin/constellation_leo_observers.yaml --output /tmp/astro-constellation-twin.json --summary-output /tmp/astro-constellation-twin.txt
python -m ruff check .
python -m mypy
python -m pytest -q
git diff --check
python -m pytest tests/test_packaging.py -q
python -m build
```

Expected:

- Focused constellation tests pass.
- Public CLI writes JSON and text artifacts.
- Ruff passes.
- Mypy passes.
- Full pytest passes with only existing optional-backend skips.
- Diff check has no output.
- Packaging tests pass.
- Build produces sdist and wheel.

- [ ] **Step 5: Commit docs and state**

```bash
git add docs/digital-twin.md docs/validation-matrix.md docs/current-state.md
git commit -m "Document constellation digital twin workflow"
```

## Self-Review Notes

- Spec coverage: tasks cover constellation scenario/result models, IO, aggregation rules, runner,
  reference examples, CLI, docs, validation matrix, current-state evidence, and full gates.
- Scope control: plan keeps v1 as aggregation over member `DigitalTwinResult`s and defers crosslinks,
  scheduling, target grids, optimizer work, and optional backends.
- Type consistency: model names match the design spec: `ConstellationTwinScenario`,
  `ConstellationTwinResult`, `FleetAccessSummary`, `FleetLinkSummary`, `MemberLinkSummary`,
  `MemberTwinResult`, and `ConstellationCoverageRequirement`.

## Final Verification

Before PR or merge:

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
git diff --check
python -m pytest tests/test_packaging.py -q
python -m build
```

Record exact result counts and any skipped optional backend gates in the closeout.
