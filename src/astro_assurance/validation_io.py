from __future__ import annotations

import os
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_assurance.io import (
    load_post_launch_assurance_scenario,
    verify_mission_assurance_case_integrity,
)
from astro_assurance.models import MissionAssuranceCase, PostLaunchAssuranceScenario
from astro_assurance.validation_models import (
    AssuranceValidationProfileResult,
    AssuranceValidationStatus,
    PairedAssuranceValidationProtocol,
    PairedAssuranceValidationResult,
)
from astro_assurance.validation_runner import (
    build_assurance_validation_scenario,
    derive_assurance_validation_profile_evidence,
    validation_profile_force_models,
)
from astro_core.errors import InvalidScenarioError


def load_paired_assurance_validation_protocol(
    path: Path | str,
) -> PairedAssuranceValidationProtocol:
    protocol_path = Path(path)
    try:
        source_bytes = protocol_path.read_bytes()
        raw: Any = yaml.safe_load(source_bytes.decode("utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read assurance validation protocol {protocol_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(
            f"Could not parse assurance validation protocol {protocol_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Assurance validation protocol {protocol_path} must contain a mapping"
        )
    try:
        protocol = PairedAssuranceValidationProtocol.model_validate(raw)
        assurance_path = _resolve_reference(protocol_path, protocol.assurance_scenario)
        return protocol.model_copy(
            update={
                "assurance_scenario": str(assurance_path.resolve()),
                "source_path": str(protocol_path.resolve()),
                "source_digest": sha256(source_bytes).hexdigest(),
            }
        )
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Assurance validation protocol {protocol_path} is invalid: {exc}"
        ) from exc


def load_paired_assurance_validation_result(
    path: Path | str,
) -> PairedAssuranceValidationResult:
    result_path = Path(path)
    try:
        return PairedAssuranceValidationResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load assurance validation result {result_path}: {exc}"
        ) from exc


def verify_paired_assurance_validation_result(
    path: Path | str,
) -> PairedAssuranceValidationResult:
    result = load_paired_assurance_validation_result(path)
    sources = (
        ("protocol", result.protocol_source_path, result.protocol_source_digest),
        ("assurance", result.assurance_source_path, result.assurance_source_digest),
    )
    for role, source_path, expected_digest in sources:
        try:
            actual_digest = sha256(Path(source_path).read_bytes()).hexdigest()
        except OSError as exc:
            raise InvalidScenarioError(
                f"Could not verify {role} source {source_path}: {exc}"
            ) from exc
        if actual_digest != expected_digest:
            raise InvalidScenarioError(f"{role} source digest mismatch: {source_path}")
    protocol = load_paired_assurance_validation_protocol(result.protocol_source_path)
    base = load_post_launch_assurance_scenario(result.assurance_source_path)
    if protocol.protocol_id != result.protocol_id:
        raise InvalidScenarioError("validation result protocol id does not match its source")
    if protocol.claim_boundary != result.claim_boundary:
        raise InvalidScenarioError("validation result claim boundary does not match its protocol")
    if Path(protocol.assurance_scenario).resolve() != Path(result.assurance_source_path).resolve():
        raise InvalidScenarioError("validation result assurance source does not match its protocol")
    expected_realizations = tuple(realization for realization in protocol.realizations)
    actual_realizations = tuple(pair.realization for pair in result.pairs)
    if actual_realizations != expected_realizations:
        raise InvalidScenarioError("validation result coordinates do not match its protocol")
    for pair, realization in zip(result.pairs, protocol.realizations, strict=True):
        matched_expected = build_assurance_validation_scenario(
            protocol, base, realization, pair.matched.profile
        )
        mismatched_expected = build_assurance_validation_scenario(
            protocol, base, realization, pair.mismatched.profile
        )
        matched_metrics, matched_passed = _verify_profile_evidence(
            pair.matched, base, matched_expected
        )
        mismatched_metrics, mismatched_passed = _verify_profile_evidence(
            pair.mismatched, base, mismatched_expected
        )
        expected_complete = matched_metrics is not None and mismatched_metrics is not None
        expected_deltas = (
            {
                metric: mismatched_metrics[metric] - matched_metrics[metric]
                for metric in sorted(set(matched_metrics) & set(mismatched_metrics))
            }
            if expected_complete and matched_metrics is not None and mismatched_metrics is not None
            else {}
        )
        if pair.delta_mismatched_minus_matched != expected_deltas:
            raise InvalidScenarioError(f"validation pair deltas do not match: {pair.case_id}")
        if not expected_complete:
            expected_reversal = "not_comparable"
        elif matched_passed is True and mismatched_passed is False:
            expected_reversal = "regression"
        elif matched_passed is False and mismatched_passed is True:
            expected_reversal = "improvement"
        else:
            expected_reversal = "unchanged"
        if pair.pass_reversal != expected_reversal:
            raise InvalidScenarioError(f"validation pair reversal does not match: {pair.case_id}")
    return result


def _verify_profile_evidence(
    profile_result: AssuranceValidationProfileResult,
    base: PostLaunchAssuranceScenario,
    expected: PostLaunchAssuranceScenario,
) -> tuple[dict[str, float] | None, bool | None]:
    expected_truth, expected_estimation = validation_profile_force_models(profile_result.profile)
    if (
        profile_result.truth_force_model is not expected_truth
        or profile_result.estimation_force_model is not expected_estimation
    ):
        raise InvalidScenarioError(
            f"validation profile force-role labels do not match: {profile_result.profile}"
        )
    if profile_result.status is AssuranceValidationStatus.EXECUTION_FAILURE:
        return None, None
    case = profile_result.assurance_case
    if case is None:
        raise InvalidScenarioError("successful validation profile is missing its assurance case")
    verify_mission_assurance_case_integrity(case)
    _verify_case_protocol_binding(case, expected, profile_result)
    if (
        case.truth_scenario.force_model.gravity is not expected_truth
        or case.nominal_scenario.force_model.gravity is not expected_estimation
        or case.metadata.get("truth_force_model") != expected_truth.value
        or case.metadata.get("estimation_force_model") != expected_estimation.value
    ):
        raise InvalidScenarioError(
            f"embedded assurance force roles do not match: {profile_result.profile}"
        )
    metrics, passed = derive_assurance_validation_profile_evidence(case, base)
    if profile_result.metrics != metrics:
        raise InvalidScenarioError(
            f"validation profile metrics do not match: {profile_result.profile}"
        )
    if (
        profile_result.passed is not passed
        or profile_result.assurance_case_passed is not case.passed
    ):
        raise InvalidScenarioError(
            f"validation profile pass disposition does not match: {profile_result.profile}"
        )
    return metrics, passed


def _verify_case_protocol_binding(
    case: MissionAssuranceCase,
    expected: PostLaunchAssuranceScenario,
    profile_result: AssuranceValidationProfileResult,
) -> None:
    expected_overrides = expected.input_overrides
    if expected_overrides is None:
        raise InvalidScenarioError("validation profile is missing expected input overrides")
    if (
        expected_overrides.tracking_duration_s is None
        or expected_overrides.tracking_noise_seed is None
    ):
        raise InvalidScenarioError("validation profile is missing expected tracking overrides")
    expected_metadata = {
        "validation_protocol": expected.metadata["validation_protocol"],
        "validation_case_id": expected.metadata["validation_case_id"],
        "validation_profile": profile_result.profile.value,
        "resolved_input_overrides": expected_overrides.model_dump(mode="json"),
        "correction_elapsed_s": float(expected.correction.correction_elapsed_s),
        "verification_elapsed_s": float(expected.correction.verification_elapsed_s),
        "maximum_component_delta_v_km_s": float(
            expected.correction.maximum_component_delta_v_km_s
        ),
        "maximum_total_delta_v_km_s": float(
            expected.correction.maximum_total_delta_v_km_s
        ),
    }
    if case.scenario_id != expected.scenario_id or any(
        case.metadata.get(name) != value for name, value in expected_metadata.items()
    ):
        raise InvalidScenarioError(
            f"embedded assurance case does not match protocol coordinate: {expected.scenario_id}"
        )
    if (
        float(case.truth_scenario.propagation.duration_s)
        != float(expected_overrides.tracking_duration_s)
        or int(case.truth_scenario.measurements.noise.seed)
        != expected_overrides.tracking_noise_seed
    ):
        raise InvalidScenarioError(
            f"embedded assurance tracking configuration does not match: {expected.scenario_id}"
        )
    position_delta = tuple(
        truth - nominal
        for truth, nominal in zip(
            case.truth_scenario.initial_state.cartesian.position_km,
            case.nominal_scenario.initial_state.cartesian.position_km,
            strict=True,
        )
    )
    velocity_delta = tuple(
        truth - nominal
        for truth, nominal in zip(
            case.truth_scenario.initial_state.cartesian.velocity_km_s,
            case.nominal_scenario.initial_state.cartesian.velocity_km_s,
            strict=True,
        )
    )
    if any(
        abs(actual - configured) > 1.0e-12
        for actual, configured in zip(
            position_delta, expected.dispersion.position_delta_km, strict=True
        )
    ) or any(
        abs(actual - configured) > 1.0e-12
        for actual, configured in zip(
            velocity_delta, expected.dispersion.velocity_delta_km_s, strict=True
        )
    ):
        raise InvalidScenarioError(
            f"embedded assurance dispersion does not match: {expected.scenario_id}"
        )


def write_paired_assurance_validation_result(
    path: Path | str,
    result: PairedAssuranceValidationResult,
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


def format_paired_assurance_validation_summary(
    result: PairedAssuranceValidationResult,
) -> str:
    summary = result.summary
    lines = [
        f"Protocol: {result.protocol_id}",
        f"Pairs complete: {summary.paired_complete}/{summary.requested_pairs}",
        f"Matched passed: {summary.matched_passed}/{summary.requested_pairs}",
        f"Mismatched passed: {summary.mismatched_passed}/{summary.requested_pairs}",
        f"Pass regressions: {summary.pass_regressions}",
        f"Pass improvements: {summary.pass_improvements}",
        f"Claim boundary: {result.claim_boundary}",
    ]
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
