from __future__ import annotations

import os
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_assurance.model_form_models import (
    ModelFormFactorialProtocol,
    ModelFormFactorialResult,
)
from astro_core.errors import InvalidScenarioError


def load_model_form_factorial_protocol(path: Path | str) -> ModelFormFactorialProtocol:
    protocol_path = Path(path)
    try:
        source_bytes = protocol_path.read_bytes()
        raw: Any = yaml.safe_load(source_bytes.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise InvalidScenarioError(
            f"Could not read model-form factorial protocol {protocol_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError("model-form factorial protocol must contain a mapping")
    try:
        protocol = ModelFormFactorialProtocol.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(f"model-form factorial protocol is invalid: {exc}") from exc
    return protocol.model_copy(
        update={
            "assurance_scenario": str(
                _resolve_reference(protocol_path, protocol.assurance_scenario).resolve()
            ),
            "calibration_evidence": str(
                _resolve_reference(protocol_path, protocol.calibration_evidence).resolve()
            ),
            "source_path": str(protocol_path.resolve()),
            "source_digest": sha256(source_bytes).hexdigest(),
        }
    )


def load_model_form_factorial_result(path: Path | str) -> ModelFormFactorialResult:
    try:
        return ModelFormFactorialResult.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load model-form factorial result {path}: {exc}"
        ) from exc


def write_model_form_factorial_result(
    path: Path | str, result: ModelFormFactorialResult
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (result.model_dump_json(indent=2) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def verify_model_form_factorial_result(path: Path | str) -> ModelFormFactorialResult:
    from astro_assurance.model_form_runner import run_model_form_factorial

    result = load_model_form_factorial_result(path)
    for role, source_path, expected_digest in (
        ("protocol", result.protocol_source_path, result.protocol_source_digest),
        ("assurance", result.assurance_source_path, result.assurance_source_digest),
        ("calibration", result.calibration_source_path, result.calibration_source_digest),
    ):
        try:
            actual_digest = sha256(Path(source_path).read_bytes()).hexdigest()
        except OSError as exc:
            raise InvalidScenarioError(f"Could not verify {role} source: {exc}") from exc
        if actual_digest != expected_digest:
            raise InvalidScenarioError(f"model-form factorial {role} source digest mismatch")
    protocol = load_model_form_factorial_protocol(result.protocol_source_path)
    expected = run_model_form_factorial(protocol)
    if result.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise InvalidScenarioError(
            "model-form factorial result does not match exact local reexecution"
        )
    return result


def format_model_form_factorial_summary(result: ModelFormFactorialResult) -> str:
    lines = [
        f"Protocol: {result.protocol_id}",
        f"Realizations: {result.summary.requested_realizations}",
    ]
    for profile, counts in result.summary.profile_counts.items():
        lines.append(
            f"{profile.value}: completed {counts.completed}/{counts.requested}, "
            f"passed {counts.passed}/{counts.requested}"
        )
    lines.extend(
        [
            f"Calibration status: {result.calibration_promotion_status.value}",
            f"Claim boundary: {result.claim_boundary}",
        ]
    )
    return "\n".join(lines) + "\n"


def _resolve_reference(owner_path: Path, configured_path: str) -> Path:
    configured = Path(configured_path)
    if configured.is_absolute():
        return configured
    for parent in owner_path.resolve().parents:
        candidate = parent / configured
        if candidate.exists():
            return candidate
    return owner_path.parent / configured
