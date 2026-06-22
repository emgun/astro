# Scenario Verifier Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `astro ask` flexible over checked-in local OD scenarios while preventing silent scenario substitution through deterministic plan verification.

**Architecture:** Keep the assistant compiled and bounded: natural language resolves into a typed local OD workflow with scenario and artifact-directory slots, then a deterministic verifier checks paths, capabilities, policy, and workflow consistency before command specs execute. Add an agentic-verifier-ready model as structured diagnostics, but keep blocking authority in deterministic code.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, pytest, existing Astro CLI commands and scenario products.

---

## File Structure

- Create `src/astro_assistant/scenarios.py`: scenario alias/path resolution, local example boundary checks, artifact directory derivation.
- Create `src/astro_assistant/verification.py`: deterministic plan verification and structured diagnostic models.
- Modify `src/astro_assistant/planner.py`: bind prompt slots into local OD workflow instead of hardcoding the scenario.
- Modify `src/astro_assistant/executor.py`: include verification diagnostics in traces and block execution on verification failures.
- Modify `src/astro_assistant/models.py`: add verification diagnostic/result fields to `WorkflowTrace`.
- Modify `src/astro_cli/main.py`: create assistant trace parent directories and surface verification warnings through the existing `ask` command.
- Modify `docs/assistant-workflows.md`, `docs/assistant-mcp-contract.md`, `docs/validation-matrix.md`, and `README.md`: document scenario-general local OD usage and verifier boundaries.
- Add/modify tests under `tests/astro_assistant/` and `tests/astro_cli/`.

## Task 1: Scenario Resolution Tests

**Files:**
- Create: `tests/astro_assistant/test_scenarios.py`
- Create: `src/astro_assistant/scenarios.py`

- [x] **Step 1: Write failing tests**

```python
from pathlib import Path

import pytest

from astro_assistant.scenarios import resolve_local_od_scenario


def test_resolves_explicit_checked_in_scenario_path() -> None:
    resolved = resolve_local_od_scenario(
        "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml"
    )

    assert resolved.path == "examples/scenarios/leo_two_station_angles.yaml"
    assert resolved.scenario_id == "leo-two-station-angles"
    assert resolved.artifact_dir == "/tmp/astro-assistant/leo_two_station_angles"


def test_resolves_known_scenario_alias() -> None:
    resolved = resolve_local_od_scenario("Run the local OD workflow for radiometric media")

    assert resolved.path == "examples/scenarios/leo_radiometric_media.yaml"
    assert resolved.scenario_id == "leo-radiometric-media"


def test_defaults_to_original_demo_when_no_scenario_is_named() -> None:
    resolved = resolve_local_od_scenario("Run the local OD demo")

    assert resolved.path == "examples/scenarios/leo_two_station_od.yaml"


def test_rejects_paths_outside_examples_scenarios() -> None:
    with pytest.raises(ValueError, match="scenario path must stay under examples/scenarios"):
        resolve_local_od_scenario("Run local OD on /tmp/custom.yaml")


def test_rejects_unknown_scenario_alias() -> None:
    with pytest.raises(ValueError, match="could not resolve a supported local OD scenario"):
        resolve_local_od_scenario("Run local OD on the secret mission scenario")
```

- [x] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/astro_assistant/test_scenarios.py -q`

Expected: fail because `astro_assistant.scenarios` does not exist.

- [x] **Step 3: Implement minimal scenario resolver**

Add a supported catalog for the local OD scenarios proven by live probe:

- `leo_doppler.yaml`
- `leo_geodetic_eop_table_topocentric.yaml`
- `leo_geodetic_eop_topocentric.yaml`
- `leo_geodetic_precession_nutation_topocentric.yaml`
- `leo_geodetic_topocentric.yaml`
- `leo_radiometric_links.yaml`
- `leo_radiometric_media.yaml`
- `leo_radiometric_weather_frequency.yaml`
- `leo_two_station_angles.yaml`
- `leo_two_station_od.yaml`
- `leo_two_station_topocentric.yaml`

Resolver rules:

- Explicit `examples/scenarios/<name>.yaml` wins.
- Known aliases map to supported catalog entries.
- No named scenario defaults to `leo_two_station_od.yaml`.
- Absolute paths and paths outside `examples/scenarios` fail.
- Artifact directory is `/tmp/astro-assistant/<scenario-stem>`.

- [x] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/astro_assistant/test_scenarios.py -q`

Expected: pass.

## Task 2: Deterministic Verification Tests

**Files:**
- Create: `tests/astro_assistant/test_verification.py`
- Create: `src/astro_assistant/verification.py`
- Modify: `src/astro_assistant/models.py`

- [x] **Step 1: Write failing tests**

```python
from astro_assistant.models import AstroWorkflowPlan
from astro_assistant.planner import DeterministicPlanner
from astro_assistant.verification import verify_plan


def test_verifier_accepts_resolved_supported_scenario() -> None:
    plan = DeterministicPlanner().plan(
        "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml"
    )

    result = verify_plan(plan)

    assert result.passed is True
    assert result.diagnostics == []


def test_verifier_rejects_silent_scenario_substitution() -> None:
    plan = DeterministicPlanner().plan("Run the local OD demo")
    tampered = plan.model_copy(update={"user_intent": "Run local OD on examples/scenarios/leo_two_station_angles.yaml"})

    result = verify_plan(tampered)

    assert result.passed is False
    assert any("requested scenario" in diagnostic.message for diagnostic in result.diagnostics)


def test_verifier_rejects_output_paths_outside_artifact_directory() -> None:
    plan = DeterministicPlanner().plan("Run the local OD demo")
    bad_steps = list(plan.steps)
    bad_steps[1] = bad_steps[1].model_copy(
        update={"inputs": {**bad_steps[1].inputs, "output": "/tmp/not-astro/measurements.json"}}
    )
    tampered = plan.model_copy(update={"steps": bad_steps})

    result = verify_plan(tampered)

    assert result.passed is False
    assert any("artifact directory" in diagnostic.message for diagnostic in result.diagnostics)
```

- [x] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/astro_assistant/test_verification.py -q`

Expected: fail because verification module/model fields do not exist.

- [x] **Step 3: Implement verification models and verifier**

Add Pydantic models:

- `VerificationDiagnostic` with `code`, `message`, `severity`.
- `VerificationResult` with `passed`, `diagnostics`.

Verifier checks:

- Prompt scenario resolution matches every scenario input in the plan.
- Every writable output stays under the resolved artifact directory.
- Workflow step order is exactly validate, synth, export, estimate.
- Backends are `local`.
- Export format is `tdm`.
- Required output artifact kinds and paths match the bound scenario.

- [x] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/astro_assistant/test_verification.py -q`

Expected: pass.

## Task 3: Planner Binding And Executor Enforcement

**Files:**
- Modify: `src/astro_assistant/planner.py`
- Modify: `src/astro_assistant/executor.py`
- Modify: `src/astro_assistant/models.py`
- Modify: `tests/astro_assistant/test_planner.py`
- Modify: `tests/astro_assistant/test_executor.py`

- [x] **Step 1: Add failing planner tests**

Required behaviors:

- Explicit scenario path appears in every relevant plan step.
- Artifact outputs go under `/tmp/astro-assistant/<scenario-stem>/`.
- Unsupported prompt still raises the existing unsupported message.
- The planner never silently substitutes a different scenario when one is requested.

- [x] **Step 2: Add failing executor test**

Executor should return no command results when deterministic verification fails, and the trace should include verification diagnostics.

- [x] **Step 3: Implement planner binding**

Call `resolve_local_od_scenario(prompt)` inside `local_od_demo_plan`. Build output paths from the resolved artifact dir:

- `measurements.json`
- `measurements.tdm`
- `estimate.json`

Use resolved scenario path and scenario id in descriptions where useful.

- [x] **Step 4: Implement executor enforcement**

Call `verify_plan(plan)` before policy execution. Include verification result in `WorkflowTrace`. If verification fails, return trace without command execution.

- [x] **Step 5: Run assistant tests**

Run: `python -m pytest tests/astro_assistant -q`

Expected: pass.

## Task 4: CLI UX And Documentation

**Files:**
- Modify: `src/astro_cli/main.py`
- Modify: `tests/astro_cli/test_assistant_cli.py`
- Modify: `docs/assistant-workflows.md`
- Modify: `docs/assistant-mcp-contract.md`
- Modify: `docs/validation-matrix.md`
- Modify: `README.md`

- [x] **Step 1: Add failing CLI tests**

Required behaviors:

- `astro ask` dry-run for `leo_two_station_angles.yaml` includes that path and scenario-specific output directory.
- `astro ask` with unsupported scenario exits 2 with a clear scenario-resolution message.
- Trace output parent directories are created automatically.

- [x] **Step 2: Implement assistant write-parent creation**

Keep the shared CLI writer semantics intact for existing commands. Create trace-output parents in
`astro ask`, and let the assistant executor prepare parent directories for allow-listed command
writes before execution.

- [x] **Step 3: Update docs**

Document:

- Default demo prompt.
- Explicit scenario prompt.
- Supported local OD scenarios.
- Verification model: deterministic verifier is authoritative; future agentic verifier may generate extra checks but cannot approve execution without deterministic evidence.

- [x] **Step 4: Run CLI tests**

Run: `python -m pytest tests/astro_cli/test_assistant_cli.py -q`

Expected: pass.

## Task 5: End-To-End Verification

**Files:**
- No required edits.

- [x] **Step 1: Run targeted tests**

Run: `python -m pytest tests/astro_assistant tests/astro_cli/test_assistant_cli.py -q`

Expected: all pass.

- [x] **Step 2: Run live scenario variation**

Run:

```bash
astro ask "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml and export TDM." --execute --approved --trace-output /tmp/astro-assistant/leo_two_station_angles/trace.json
```

Expected:

- All step results return code 0.
- `measurements.json`, `measurements.tdm`, `estimate.json`, and `trace.json` exist under `/tmp/astro-assistant/leo_two_station_angles/`.
- Trace verification result has `passed: true`.

- [x] **Step 3: Run unsupported scenario check**

Run:

```bash
astro ask "Run local orbit determination on examples/scenarios/leo_orekit_high_fidelity.yaml and export TDM." --dry-run
```

Expected: exit code 2 with a clear unsupported local OD scenario message.

- [x] **Step 4: Check repo status and diff**

Run: `git status --short --branch` and `git diff --stat`

Expected: only planned files changed.

- [x] **Step 5: Commit and push branch**

Commit message: `feat: generalize verified assistant scenarios`

Push: `git push -u origin codex/scenario-verifier-assistant`
