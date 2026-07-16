from __future__ import annotations

import os
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

import astro_assurance.covariance_validation_runner as covariance_validation_runner
from astro_assurance.covariance_validation_models import (
    CovarianceIndependenceReview,
    CovarianceSourceBinding,
    CovarianceValidationProtocol,
    CovarianceValidationResult,
    EmpiricalCovarianceArtifact,
)
from astro_core.errors import InvalidScenarioError
from astro_core.models import Trajectory


def load_covariance_validation_protocol(path: Path | str) -> CovarianceValidationProtocol:
    source = Path(path)
    try:
        source_bytes = source.read_bytes()
        raw: Any = yaml.safe_load(source_bytes.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise InvalidScenarioError(f"Could not read covariance protocol {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError("covariance validation protocol must contain a mapping")
    try:
        protocol = CovarianceValidationProtocol.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(f"covariance validation protocol is invalid: {exc}") from exc
    updates: dict[str, str] = {
        "candidate_trajectory_path": str(
            _resolve_reference(source, protocol.candidate_trajectory_path).resolve()
        ),
        "reference_trajectory_path": str(
            _resolve_reference(source, protocol.reference_trajectory_path).resolve()
        ),
        "source_path": str(source.resolve()),
        "source_digest": sha256(source_bytes).hexdigest(),
    }
    if protocol.empirical_evidence_path is not None:
        updates["empirical_evidence_path"] = str(
            _resolve_reference(source, protocol.empirical_evidence_path).resolve()
        )
    if protocol.empirical_scenario_path is not None:
        updates["empirical_scenario_path"] = str(
            _resolve_reference(source, protocol.empirical_scenario_path).resolve()
        )
    if protocol.independence_review_path is not None:
        updates["independence_review_path"] = str(
            _resolve_reference(source, protocol.independence_review_path).resolve()
        )
    return protocol.model_copy(update=updates)


def load_empirical_covariance_artifact(path: Path | str) -> EmpiricalCovarianceArtifact:
    try:
        return EmpiricalCovarianceArtifact.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load empirical covariance evidence {path}: {exc}"
        ) from exc


def load_covariance_validation_result(path: Path | str) -> CovarianceValidationResult:
    try:
        return CovarianceValidationResult.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load covariance validation result {path}: {exc}"
        ) from exc


def run_covariance_validation(protocol: CovarianceValidationProtocol) -> CovarianceValidationResult:
    if protocol.source_path is None or protocol.source_digest is None:
        raise InvalidScenarioError("covariance validation protocol lacks source provenance")
    protocol_binding = CovarianceSourceBinding(
        role="protocol",
        path=protocol.source_path,
        sha256=protocol.source_digest,
    )
    candidate_bytes, candidate_binding = _read_binding(
        "candidate_trajectory", protocol.candidate_trajectory_path
    )
    reference_bytes, reference_binding = _read_binding(
        "reference_trajectory", protocol.reference_trajectory_path
    )
    try:
        candidate = Trajectory.model_validate_json(candidate_bytes)
        reference = Trajectory.model_validate_json(reference_bytes)
    except ValidationError as exc:
        raise InvalidScenarioError(f"covariance trajectory evidence is invalid: {exc}") from exc
    empirical = None
    independence_review = None
    bindings = [protocol_binding, candidate_binding, reference_binding]
    if protocol.empirical_evidence_path is not None:
        empirical_bytes, empirical_binding = _read_binding(
            "empirical_evidence", protocol.empirical_evidence_path
        )
        try:
            empirical = EmpiricalCovarianceArtifact.model_validate_json(empirical_bytes)
        except ValidationError as exc:
            raise InvalidScenarioError(f"empirical covariance evidence is invalid: {exc}") from exc
        bindings.append(empirical_binding)
        if protocol.empirical_scenario_path is None:
            raise InvalidScenarioError("empirical covariance scenario path is missing")
        _, empirical_scenario_binding = _read_binding(
            "empirical_scenario", protocol.empirical_scenario_path
        )
        bindings.append(empirical_scenario_binding)
    if protocol.independence_review_path is not None:
        review_bytes, review_binding = _read_binding(
            "independence_review", protocol.independence_review_path
        )
        try:
            independence_review = CovarianceIndependenceReview.model_validate_json(review_bytes)
        except ValidationError as exc:
            raise InvalidScenarioError(f"covariance independence review is invalid: {exc}") from exc
        bindings.append(review_binding)
    _assert_bindings_unchanged(tuple(bindings))
    result = covariance_validation_runner.assess_covariance_validation(
        protocol,
        candidate,
        reference,
        empirical,
        independence_review,
        tuple(bindings),
    )
    _assert_bindings_unchanged(tuple(bindings))
    return result


def write_covariance_validation_result(
    path: Path | str, result: CovarianceValidationResult
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


def verify_covariance_validation_result(path: Path | str) -> CovarianceValidationResult:
    result = load_covariance_validation_result(path)
    for binding in result.source_bindings:
        try:
            digest = sha256(Path(binding.path).read_bytes()).hexdigest()
        except OSError as exc:
            raise InvalidScenarioError(f"Could not verify {binding.role} source: {exc}") from exc
        if digest != binding.sha256:
            raise InvalidScenarioError(f"covariance validation {binding.role} digest mismatch")
    protocol_binding = next(
        binding for binding in result.source_bindings if binding.role == "protocol"
    )
    expected = run_covariance_validation(
        load_covariance_validation_protocol(protocol_binding.path)
    )
    if result.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise InvalidScenarioError(
            "covariance validation result does not match exact local reassessment"
        )
    return result


def format_covariance_validation_summary(result: CovarianceValidationResult) -> str:
    lines = [
        f"Protocol: {result.protocol_id}",
        f"Disposition: {result.disposition.value}",
        (
            "Comparison epochs: "
            f"{result.comparison_summary.passed_epochs}/"
            f"{result.comparison_summary.compared_epochs} passed"
        ),
        f"Blockers: {len(result.blockers)}",
    ]
    if result.empirical_nees_summary is not None:
        lines.append(
            "Empirical NEES: "
            f"mean {result.empirical_nees_summary.mean_nees:.6g}, "
            f"coverage {result.empirical_nees_summary.coverage:.3f}"
        )
    lines.append(f"Claim boundary: {result.claim_boundary}")
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


def _read_binding(role: str, path: str) -> tuple[bytes, CovarianceSourceBinding]:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise InvalidScenarioError(f"Could not read covariance {role} source: {exc}") from exc
    return payload, CovarianceSourceBinding(
        role=role,  # type: ignore[arg-type]
        path=path,
        sha256=sha256(payload).hexdigest(),
    )


def _assert_bindings_unchanged(bindings: tuple[CovarianceSourceBinding, ...]) -> None:
    for binding in bindings:
        try:
            digest = sha256(Path(binding.path).read_bytes()).hexdigest()
        except OSError as exc:
            raise InvalidScenarioError(f"Could not re-read {binding.role} source: {exc}") from exc
        if digest != binding.sha256:
            raise InvalidScenarioError(f"{binding.role} source changed during assessment")
