from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_core.errors import InvalidScenarioError
from astro_core.models import AstroModel
from astro_operator.errors import OperatorPolicyError
from astro_operator.models import (
    AuthorityGrant,
    EpistemicKind,
    EvidenceReference,
    MissionObjective,
    OperatorRun,
)
from astro_operator.policy import validate_operator_run_policy
from astro_operator.reasoner import ConditionalReplayDecision


class MissionOperatorSpec(AstroModel):
    base_scenario_path: str
    objective: MissionObjective
    authority: AuthorityGrant


def load_mission_operator_spec(path: Path | str) -> MissionOperatorSpec:
    return _load_yaml_model(Path(path), MissionOperatorSpec, "mission operator spec")


def load_operator_replay(path: Path | str) -> tuple[ConditionalReplayDecision, ...]:
    action_path = Path(path)
    try:
        raw: Any = yaml.safe_load(action_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise InvalidScenarioError(f"Could not load operator replay {action_path}: {exc}") from exc
    if not isinstance(raw, list):
        raise InvalidScenarioError(f"Operator replay {action_path} must contain a list")
    try:
        return tuple(ConditionalReplayDecision.model_validate(item) for item in raw)
    except ValidationError as exc:
        raise InvalidScenarioError(f"Operator replay {action_path} is invalid: {exc}") from exc


def capture_base_scenario_evidence(path: Path, run_root: Path) -> EvidenceReference:
    captured_path = run_root / "inputs" / "base-scenario.yaml"
    captured_path.parent.mkdir(parents=True)
    captured_path.write_bytes(path.read_bytes())
    return EvidenceReference(
        evidence_id="base-scenario",
        kind="mission_lifecycle_scenario_source",
        epistemic_kind=EpistemicKind.DECLARED,
        claim_scope="declared input to the mission lifecycle design-screening run",
        path=captured_path.relative_to(run_root).as_posix(),
        sha256=sha256(captured_path.read_bytes()).hexdigest(),
    )


def write_operator_run(path: Path, run: OperatorRun) -> None:
    path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")


def verify_operator_run(root: Path | str) -> OperatorRun:
    run_root = Path(root)
    trace_path = run_root / "operator-run.json"
    if run_root.is_symlink():
        raise InvalidScenarioError("Operator run root must not be a symbolic link")
    if trace_path.is_symlink():
        raise InvalidScenarioError("Operator run journal must not be a symbolic link")
    try:
        raw: Any = json.loads(trace_path.read_text(encoding="utf-8"))
        run = OperatorRun.model_validate(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise InvalidScenarioError(f"Could not verify operator run {run_root}: {exc}") from exc
    try:
        validate_operator_run_policy(run)
    except OperatorPolicyError as exc:
        raise InvalidScenarioError(f"Operator run policy verification failed: {exc}") from exc
    resolved_root = run_root.resolve()
    for evidence in run.known_evidence:
        artifact_path = Path(evidence.path)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise InvalidScenarioError(f"Operator evidence path escapes run root: {evidence.path}")
        candidate = run_root / artifact_path
        if any(part.is_symlink() for part in _path_prefixes(run_root, artifact_path)):
            raise InvalidScenarioError(
                f"Operator evidence path contains a symbolic link: {evidence.path}"
            )
        resolved = candidate.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise InvalidScenarioError(f"Operator evidence path escapes run root: {evidence.path}")
        if not candidate.is_file():
            raise InvalidScenarioError(f"Operator evidence artifact is missing: {evidence.path}")
        if sha256(candidate.read_bytes()).hexdigest() != evidence.sha256:
            raise InvalidScenarioError(f"Operator evidence digest mismatch: {evidence.path}")
    return run


def _load_yaml_model(
    path: Path, model: type[MissionOperatorSpec], label: str
) -> MissionOperatorSpec:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise InvalidScenarioError(f"Could not load {label} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(f"{label.capitalize()} {path} must contain a mapping")
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(f"{label.capitalize()} {path} is invalid: {exc}") from exc


def _path_prefixes(root: Path, relative: Path) -> tuple[Path, ...]:
    prefixes: list[Path] = []
    current = root
    for part in relative.parts:
        current = current / part
        prefixes.append(current)
    return tuple(prefixes)
