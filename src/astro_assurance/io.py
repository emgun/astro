from __future__ import annotations

import json
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_assurance.models import (
    AssuranceManifest,
    MissionAssuranceCase,
    PostLaunchAssuranceScenario,
)
from astro_core.errors import InvalidScenarioError


def load_post_launch_assurance_scenario(path: Path | str) -> PostLaunchAssuranceScenario:
    scenario_path = Path(path)
    try:
        source_bytes = scenario_path.read_bytes()
        raw: Any = yaml.safe_load(source_bytes.decode("utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read mission assurance scenario {scenario_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(
            f"Could not parse mission assurance scenario {scenario_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Mission assurance scenario file {scenario_path} must contain a mapping"
        )
    try:
        scenario = PostLaunchAssuranceScenario.model_validate(raw)
        return scenario.model_copy(
            update={
                "source_path": str(scenario_path.resolve()),
                "source_digest": sha256(source_bytes).hexdigest(),
            }
        )
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Mission assurance scenario file {scenario_path} is invalid: {exc}"
        ) from exc


def write_mission_assurance_result(path: Path | str, result: MissionAssuranceCase) -> None:
    _write_text(Path(path), result.model_dump_json(indent=2) + "\n")


def load_mission_assurance_result(path: Path | str) -> MissionAssuranceCase:
    result_path = Path(path)
    try:
        raw: Any = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read mission assurance result {result_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise InvalidScenarioError(
            f"Could not parse mission assurance result {result_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Mission assurance result file {result_path} must contain a JSON object"
        )
    try:
        return MissionAssuranceCase.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Mission assurance result file {result_path} is invalid: {exc}"
        ) from exc


def write_mission_assurance_artifact_bundle(
    directory: Path | str,
    result: MissionAssuranceCase,
) -> None:
    artifact_directory = Path(directory)
    try:
        artifact_directory.parent.mkdir(parents=True, exist_ok=True)
        lock_path = artifact_directory.parent / f".{artifact_directory.name}.lock"
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise InvalidScenarioError(
            f"Mission assurance artifact publication is already active: {artifact_directory}"
        ) from exc
    except OSError as exc:
        raise InvalidScenarioError(
            f"Could not prepare mission assurance artifact directory {artifact_directory}: {exc}"
        ) from exc
    staging_directory: Path | None = None
    try:
        if artifact_directory.exists():
            raise InvalidScenarioError(
                f"Mission assurance artifact directory already exists: {artifact_directory}"
            )
        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{artifact_directory.name}.",
                dir=artifact_directory.parent,
            )
        )
        _write_mission_assurance_staging_bundle(staging_directory, result)
        if artifact_directory.exists():
            raise InvalidScenarioError(
                "Mission assurance artifact directory appeared during publication: "
                f"{artifact_directory}"
            )
        os.replace(staging_directory, artifact_directory)
        staging_directory = None
    except (InvalidScenarioError, OSError) as exc:
        if staging_directory is not None:
            shutil.rmtree(staging_directory, ignore_errors=True)
        if isinstance(exc, InvalidScenarioError):
            raise
        raise InvalidScenarioError(
            f"Could not publish mission assurance artifact directory {artifact_directory}: {exc}"
        ) from exc
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def _write_mission_assurance_staging_bundle(
    staging_directory: Path,
    result: MissionAssuranceCase,
) -> None:
    json_artifacts, yaml_artifacts = _mission_assurance_artifact_payloads(result)
    for name, payload in json_artifacts.items():
        _write_text(staging_directory / name, json.dumps(payload, indent=2) + "\n")
    for name, payload in yaml_artifacts.items():
        _write_text(staging_directory / name, yaml.safe_dump(payload, sort_keys=False))
    _write_text(
        staging_directory / "manifest.json",
        json.dumps(result.manifest.model_dump(mode="json"), indent=2) + "\n",
    )


def _mission_assurance_artifact_payloads(
    result: MissionAssuranceCase,
) -> tuple[dict[str, Any], dict[str, Any]]:
    json_artifacts: dict[str, Any] = {
        "launch.json": result.launch_trajectory.model_dump(mode="json"),
        "measurements.json": {
            "scenario_id": result.truth_scenario.scenario_id,
            "measurements": [item.model_dump(mode="json") for item in result.measurements],
        },
        "estimate.json": result.estimate.model_dump(mode="json"),
        "candidate-maneuver.json": result.correction_maneuver.model_dump(mode="json"),
        "nominal-trajectory.json": result.nominal_trajectory.model_dump(mode="json"),
        "truth-trajectory.json": result.truth_trajectory.model_dump(mode="json"),
        "estimated-corrected-trajectory.json": result.estimated_corrected_trajectory.model_dump(
            mode="json"
        ),
        "truth-corrected-trajectory.json": result.truth_corrected_trajectory.model_dump(
            mode="json"
        ),
        "corrected-digital-twin.json": result.corrected_digital_twin.model_dump(mode="json"),
        "continuity-report.json": result.continuity_report.model_dump(mode="json"),
        "margin-report.json": result.margin_report.model_dump(mode="json"),
        "decision.json": {
            "scenario_id": result.scenario_id,
            "workflow": result.workflow,
            "passed": result.passed,
            "metadata": result.metadata,
            "warnings": result.warnings,
        },
    }
    yaml_artifacts: dict[str, Any] = {
        "nominal-scenario.yaml": result.nominal_scenario.model_dump(mode="json"),
        "truth-scenario.yaml": result.truth_scenario.model_dump(mode="json"),
        "estimated-corrected-scenario.yaml": result.estimated_corrected_scenario.model_dump(
            mode="json"
        ),
        "truth-corrected-scenario.yaml": result.truth_corrected_scenario.model_dump(mode="json"),
    }
    return json_artifacts, yaml_artifacts


def verify_mission_assurance_artifact_bundle(directory: Path | str) -> AssuranceManifest:
    artifact_directory = Path(directory)
    manifest_path = artifact_directory / "manifest.json"
    try:
        manifest = AssuranceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load mission assurance manifest {manifest_path}: {exc}"
        ) from exc
    expected_names = {entry.artifact_name for entry in manifest.entries} | {"manifest.json"}
    try:
        actual_names = {path.name for path in artifact_directory.iterdir() if path.is_file()}
    except OSError as exc:
        raise InvalidScenarioError(
            f"Could not inspect mission assurance artifact directory {artifact_directory}: {exc}"
        ) from exc
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise InvalidScenarioError(
            f"Mission assurance artifact set mismatch: missing={missing}, extra={extra}"
        )
    for entry in manifest.entries:
        artifact_path = artifact_directory / entry.artifact_name
        payload = _load_artifact_payload(artifact_path)
        if _canonical_payload_digest(payload) != entry.source_digest:
            raise InvalidScenarioError(
                f"Mission assurance artifact digest mismatch: {entry.artifact_name}"
            )
    for reference in manifest.inputs:
        input_path = Path(reference.path)
        try:
            digest = _file_digest(input_path)
        except OSError as exc:
            raise InvalidScenarioError(
                f"Could not verify mission assurance input {input_path}: {exc}"
            ) from exc
        if digest != reference.file_digest:
            raise InvalidScenarioError(f"Mission assurance input digest mismatch: {reference.role}")
    return manifest


def verify_mission_assurance_case_integrity(result: MissionAssuranceCase) -> None:
    """Verify an embedded case against its input files and manifest product digests."""
    json_artifacts, yaml_artifacts = _mission_assurance_artifact_payloads(result)
    payloads = {**json_artifacts, **yaml_artifacts}
    for entry in result.manifest.entries:
        payload = payloads.get(entry.artifact_name)
        if payload is None or _canonical_payload_digest(payload) != entry.source_digest:
            raise InvalidScenarioError(
                f"Mission assurance embedded artifact digest mismatch: {entry.artifact_name}"
            )
    for reference in result.manifest.inputs:
        input_path = Path(reference.path)
        try:
            digest = _file_digest(input_path)
        except OSError as exc:
            raise InvalidScenarioError(
                f"Could not verify mission assurance input {input_path}: {exc}"
            ) from exc
        if digest != reference.file_digest:
            raise InvalidScenarioError(f"Mission assurance input digest mismatch: {reference.role}")


def _load_artifact_payload(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        payload: Any = (
            yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
        )
    except (OSError, UnicodeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise InvalidScenarioError(
            f"Could not parse mission assurance artifact {path}: {exc}"
        ) from exc
    return payload


def _canonical_payload_digest(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _digest_bytes(canonical.encode("utf-8"))


def _file_digest(path: Path) -> str:
    return _digest_bytes(path.read_bytes())


def _digest_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def format_mission_assurance_summary(result: MissionAssuranceCase) -> str:
    limiting = result.margin_report.limiting_margin
    metadata = result.metadata
    lines = [
        f"Mission assurance case: {result.scenario_id}",
        f"Workflow: {result.workflow}",
        f"Status: {'pass' if result.passed else 'fail'}",
        f"Measurements: {len(result.measurements)}",
        f"OD converged: {result.estimate.converged}",
        f"Candidate delta-v km/s: {metadata['candidate_delta_v_km_s']:.9f}",
        "Truth position error km: "
        f"{metadata['uncorrected_truth_position_error_km']:.6f} -> "
        f"{metadata['truth_recovery_position_error_km']:.6f}",
        "Truth position error reduction: "
        f"{metadata['truth_position_error_reduction_fraction']:.6f}",
        f"Recovery disposition: {metadata['recovery_disposition']}",
        (
            f"Limiting margin: {limiting.name} = {limiting.margin:.6f} "
            f"{limiting.unit} ({limiting.status})"
        ),
    ]
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not write mission assurance artifact {path}: {exc}"
        ) from exc
