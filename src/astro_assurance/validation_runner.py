from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np

from astro_assurance.errors import MissionAssuranceError
from astro_assurance.io import load_post_launch_assurance_scenario
from astro_assurance.models import (
    MissionAssuranceCase,
    MissionAssuranceInputOverrides,
    PostLaunchAssuranceScenario,
)
from astro_assurance.runner import run_post_launch_assurance
from astro_assurance.validation_calibration_io import (
    load_assurance_validation_calibration,
    validate_calibration_against_protocol,
)
from astro_assurance.validation_models import (
    AssuranceCalibrationPromotionStatus,
    AssuranceValidationCalibrationManifest,
    AssuranceValidationPairResult,
    AssuranceValidationProfile,
    AssuranceValidationProfileResult,
    AssuranceValidationRealization,
    AssuranceValidationStatus,
    PairedAssuranceValidationProtocol,
    PairedAssuranceValidationResult,
    summarize_validation_pairs,
)
from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario
from astro_core.models import ForceModelName
from astro_launch.io import load_launch_scenario
from astro_twin.io import load_twin_scenario

_PROFILES = (
    AssuranceValidationProfile.MATCHED_TWO_BODY,
    AssuranceValidationProfile.TRUTH_J2_ESTIMATOR_TWO_BODY,
)


def validate_paired_assurance_protocol(
    protocol: PairedAssuranceValidationProtocol,
) -> AssuranceValidationCalibrationManifest:
    _assert_protocol_source_unchanged(protocol)
    assurance_path = Path(protocol.assurance_scenario)
    scenario = _resolve_assurance_references(
        assurance_path,
        load_post_launch_assurance_scenario(assurance_path),
    )
    load_launch_scenario(scenario.launch_scenario)
    load_scenario(scenario.tracking_scenario)
    load_twin_scenario(scenario.twin_scenario)
    calibration = load_assurance_validation_calibration(protocol.calibration_evidence)
    validate_calibration_against_protocol(calibration, protocol)
    return calibration


def run_paired_assurance_validation(
    protocol: PairedAssuranceValidationProtocol,
) -> PairedAssuranceValidationResult:
    calibration = validate_paired_assurance_protocol(protocol)
    assurance_path = Path(protocol.assurance_scenario)
    base = _resolve_assurance_references(
        assurance_path,
        load_post_launch_assurance_scenario(assurance_path),
    )
    if base.source_path is None or base.source_digest is None:
        raise InvalidScenarioError("assurance scenario is missing source provenance")
    if protocol.source_path is None or protocol.source_digest is None:
        raise InvalidScenarioError("validation protocol is missing source provenance")
    if calibration.source_path is None or calibration.source_digest is None:
        raise InvalidScenarioError("validation calibration is missing source provenance")

    pairs = tuple(_run_pair(protocol, base, realization) for realization in protocol.realizations)
    _assert_protocol_source_unchanged(protocol)
    _assert_calibration_source_unchanged(calibration)
    return PairedAssuranceValidationResult(
        protocol_id=protocol.protocol_id,
        protocol_source_path=protocol.source_path,
        protocol_source_digest=protocol.source_digest,
        assurance_source_path=base.source_path,
        assurance_source_digest=base.source_digest,
        calibration_id=calibration.calibration_id,
        calibration_source_path=calibration.source_path,
        calibration_source_digest=calibration.source_digest,
        calibration_promotion_status=calibration.promotion_status,
        calibration_claim_boundary=calibration.claim_boundary,
        pairs=pairs,
        summary=summarize_validation_pairs(pairs),
        claim_boundary=protocol.claim_boundary,
        warnings=(
            "Matched and mismatched profile counts are separate; they are not pooled into a "
            "mission-success probability.",
            "Realization bounds are protocol inputs and require external evidence before any "
            "operational interpretation.",
            "Corrections remain simulation candidates for manual review, not flight commands.",
            *(
                (
                    "Calibration evidence is not mission calibrated; configured envelopes remain "
                    "screening assumptions.",
                )
                if calibration.promotion_status
                is not AssuranceCalibrationPromotionStatus.MISSION_CALIBRATED
                else ()
            ),
        ),
        metadata={
            **protocol.metadata,
            "paired_coordinate_policy": "same_realization_and_noise_seed_across_profiles",
            "profile_order": [profile.value for profile in _PROFILES],
            "delta_convention": "truth_j2_estimator_two_body_minus_matched_two_body",
            "calibration_parameter_count": len(calibration.parameter_bounds),
            "calibration_coverage_policy": calibration.coverage_policy,
        },
    )


def _run_pair(
    protocol: PairedAssuranceValidationProtocol,
    base: PostLaunchAssuranceScenario,
    realization: AssuranceValidationRealization,
) -> AssuranceValidationPairResult:
    profile_results = {
        profile: run_assurance_validation_profile(protocol, base, realization, profile)
        for profile in _PROFILES
    }
    matched = profile_results[AssuranceValidationProfile.MATCHED_TWO_BODY]
    mismatched = profile_results[AssuranceValidationProfile.TRUTH_J2_ESTIMATOR_TWO_BODY]
    paired_complete = (
        matched.status is AssuranceValidationStatus.SUCCESS
        and mismatched.status is AssuranceValidationStatus.SUCCESS
    )
    deltas = (
        {
            metric: float(mismatched.metrics[metric] - matched.metrics[metric])
            for metric in sorted(set(matched.metrics) & set(mismatched.metrics))
        }
        if paired_complete
        else {}
    )
    reversal: Literal["unchanged", "regression", "improvement", "not_comparable"]
    if not paired_complete:
        reversal = "not_comparable"
    elif matched.passed is True and mismatched.passed is False:
        reversal = "regression"
    elif matched.passed is False and mismatched.passed is True:
        reversal = "improvement"
    else:
        reversal = "unchanged"
    return AssuranceValidationPairResult(
        case_id=realization.case_id,
        realization=realization,
        matched=matched,
        mismatched=mismatched,
        paired_complete=paired_complete,
        delta_mismatched_minus_matched=deltas,
        pass_reversal=reversal,
    )


def run_assurance_validation_profile(
    protocol: PairedAssuranceValidationProtocol,
    base: PostLaunchAssuranceScenario,
    realization: AssuranceValidationRealization,
    profile: AssuranceValidationProfile,
) -> AssuranceValidationProfileResult:
    truth_force_model, estimation_force_model = validation_profile_force_models(profile)
    scenario = build_assurance_validation_scenario(protocol, base, realization, profile)
    try:
        assurance_case = run_post_launch_assurance(scenario)
    except MissionAssuranceError as exc:
        if exc.phase in {"input_integrity", "manifest"}:
            raise
        return AssuranceValidationProfileResult(
            profile=profile,
            truth_force_model=truth_force_model,
            estimation_force_model=estimation_force_model,
            status=AssuranceValidationStatus.EXECUTION_FAILURE,
            error_type=type(exc).__name__,
            error_message=str(exc) or repr(exc),
            failure_phase=getattr(exc, "phase", None),
        )
    digest = sha256(assurance_case.model_dump_json().encode("utf-8")).hexdigest()
    metrics, protocol_passed = derive_assurance_validation_profile_evidence(
        assurance_case, base
    )
    return AssuranceValidationProfileResult(
        profile=profile,
        truth_force_model=truth_force_model,
        estimation_force_model=estimation_force_model,
        status=AssuranceValidationStatus.SUCCESS,
        passed=protocol_passed,
        assurance_case_passed=assurance_case.passed,
        metrics=metrics,
        assurance_result_digest=digest,
        assurance_case=assurance_case,
    )


def build_assurance_validation_scenario(
    protocol: PairedAssuranceValidationProtocol,
    base: PostLaunchAssuranceScenario,
    realization: AssuranceValidationRealization,
    profile: AssuranceValidationProfile,
) -> PostLaunchAssuranceScenario:
    truth_force_model, estimation_force_model = validation_profile_force_models(profile)
    overrides_payload = realization.input_overrides.model_dump(mode="python")
    overrides_payload.update(
        {
            "tracking_duration_s": protocol.tracking_duration_s,
            "truth_force_model": truth_force_model,
            "estimation_force_model": estimation_force_model,
        }
    )
    overrides = MissionAssuranceInputOverrides.model_validate(overrides_payload)
    correction = base.correction.model_copy(
        update={
            "correction_elapsed_s": protocol.correction_elapsed_s,
            "verification_elapsed_s": protocol.verification_elapsed_s,
            "maximum_component_delta_v_km_s": (protocol.diagnostic_maximum_component_delta_v_km_s),
            "maximum_total_delta_v_km_s": protocol.diagnostic_maximum_total_delta_v_km_s,
        }
    )
    return base.model_copy(
        update={
            "scenario_id": f"{protocol.protocol_id}-{realization.case_id}-{profile.value}",
            "dispersion": realization.dispersion,
            "correction": correction,
            "input_overrides": overrides,
            "metadata": {
                **base.metadata,
                "validation_protocol": protocol.protocol_id,
                "validation_case_id": realization.case_id,
                "validation_profile": profile.value,
            },
        }
    )


def validation_profile_force_models(
    profile: AssuranceValidationProfile,
) -> tuple[ForceModelName, ForceModelName]:
    return {
        AssuranceValidationProfile.MATCHED_TWO_BODY: (
            ForceModelName.TWO_BODY,
            ForceModelName.TWO_BODY,
        ),
        AssuranceValidationProfile.TRUTH_TWO_BODY_ESTIMATOR_J2: (
            ForceModelName.TWO_BODY,
            ForceModelName.J2,
        ),
        AssuranceValidationProfile.TRUTH_J2_ESTIMATOR_TWO_BODY: (
            ForceModelName.J2,
            ForceModelName.TWO_BODY,
        ),
        AssuranceValidationProfile.MATCHED_J2: (
            ForceModelName.J2,
            ForceModelName.J2,
        ),
    }[profile]


def derive_assurance_validation_profile_evidence(
    case: MissionAssuranceCase,
    base: PostLaunchAssuranceScenario,
) -> tuple[dict[str, float], bool]:
    metrics = _metrics(case, base)
    passed = case.passed and all(
        metrics[name] >= 0.0
        for name in (
            "original_candidate_component_delta_v_margin_km_s",
            "original_candidate_total_delta_v_margin_km_s",
            "original_executed_component_delta_v_margin_km_s",
            "original_executed_total_delta_v_margin_km_s",
        )
    )
    return metrics, passed


def _metrics(
    case: MissionAssuranceCase,
    base: PostLaunchAssuranceScenario,
) -> dict[str, float]:
    covariance = np.asarray(case.estimate.covariance, dtype=np.float64)
    predicted_position = float(case.metadata["predicted_recovery_position_error_km"])
    truth_position = float(case.metadata["truth_recovery_position_error_km"])
    mass_budget = case.corrected_digital_twin.mass_budget
    if mass_budget is None:
        raise ValueError("paired assurance validation requires a twin mass budget")
    metrics = {
        "od_position_error_km": _margin_value(case, "od_position_error"),
        "od_velocity_error_km_s": _margin_value(case, "od_velocity_error"),
        "normalized_residual_rms": float(case.estimate.rms),
        "jacobian_condition_number": float(case.estimate.metadata["condition_number"]),
        "covariance_trace": float(np.trace(covariance)),
        "candidate_delta_v_km_s": float(case.metadata["candidate_delta_v_km_s"]),
        "executed_delta_v_km_s": float(case.metadata["executed_delta_v_km_s"]),
        "original_candidate_component_delta_v_margin_km_s": float(
            base.correction.maximum_component_delta_v_km_s
            - max(abs(component) for component in case.correction_maneuver.delta_v_km_s)
        ),
        "original_candidate_total_delta_v_margin_km_s": float(
            base.correction.maximum_total_delta_v_km_s - case.metadata["candidate_delta_v_km_s"]
        ),
        "original_executed_component_delta_v_margin_km_s": float(
            base.correction.maximum_component_delta_v_km_s
            - max(
                abs(component)
                for component in case.truth_corrected_scenario.maneuvers[0].delta_v_km_s
            )
        ),
        "original_executed_total_delta_v_margin_km_s": float(
            base.correction.maximum_total_delta_v_km_s - case.metadata["executed_delta_v_km_s"]
        ),
        "truth_recovery_position_error_km": truth_position,
        "truth_recovery_velocity_error_km_s": float(
            case.metadata["truth_recovery_velocity_error_km_s"]
        ),
        "position_error_reduction_fraction": float(
            case.metadata["truth_position_error_reduction_fraction"]
        ),
        "predicted_vs_achieved_position_gap_km": abs(truth_position - predicted_position),
        "propellant_reserve_kg": float(mass_budget.propellant_mass_kg),
        "minimum_battery_soc_fraction": min(
            float(sample.battery_soc_fraction) for sample in case.corrected_digital_twin.power
        ),
        "minimum_thermal_margin_k": _minimum_thermal_margin(case),
        "failed_twin_margin_count": float(
            sum(
                margin.status.value == "fail"
                for margin in case.corrected_digital_twin.margin_report.margins
            )
        ),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("paired assurance validation metrics must be finite")
    return metrics


def _margin_value(case: MissionAssuranceCase, name: str) -> float:
    matches = [margin for margin in case.margin_report.margins if margin.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one assurance margin named {name}")
    return float(matches[0].value)


def _minimum_thermal_margin(case: MissionAssuranceCase) -> float:
    margins = [
        margin
        for margin in case.corrected_digital_twin.margin_report.margins
        if margin.name.startswith("thermal_")
        and (margin.name.endswith("_cold_margin_k") or margin.name.endswith("_hot_margin_k"))
    ]
    if not margins:
        raise ValueError("paired assurance validation requires thermal margins")
    return min(float(margin.margin) for margin in margins)


def _resolve_assurance_references(
    assurance_path: Path,
    scenario: PostLaunchAssuranceScenario,
) -> PostLaunchAssuranceScenario:
    return scenario.model_copy(
        update={
            "launch_scenario": str(
                _resolve_reference(assurance_path, scenario.launch_scenario).resolve()
            ),
            "tracking_scenario": str(
                _resolve_reference(assurance_path, scenario.tracking_scenario).resolve()
            ),
            "twin_scenario": str(
                _resolve_reference(assurance_path, scenario.twin_scenario).resolve()
            ),
        }
    )


def _resolve_reference(owner_path: Path, configured_path: str) -> Path:
    configured = Path(configured_path)
    if configured.is_absolute():
        return configured
    for parent in owner_path.resolve().parents:
        candidate = parent / configured
        if candidate.exists():
            return candidate
    return owner_path.parent / configured


def _assert_protocol_source_unchanged(
    protocol: PairedAssuranceValidationProtocol,
) -> None:
    if protocol.source_path is None or protocol.source_digest is None:
        raise InvalidScenarioError("validation protocol is missing source provenance")
    try:
        current_digest = sha256(Path(protocol.source_path).read_bytes()).hexdigest()
    except OSError as exc:
        raise InvalidScenarioError(
            f"Could not re-read assurance validation protocol {protocol.source_path}: {exc}"
        ) from exc
    if current_digest != protocol.source_digest:
        raise MissionAssuranceError(
            "assurance validation protocol changed during execution", phase="input_integrity"
        )


def _assert_calibration_source_unchanged(
    calibration: AssuranceValidationCalibrationManifest,
) -> None:
    if calibration.source_path is None or calibration.source_digest is None:
        raise InvalidScenarioError("validation calibration is missing source provenance")
    try:
        current_digest = sha256(Path(calibration.source_path).read_bytes()).hexdigest()
    except OSError as exc:
        raise InvalidScenarioError(
            f"Could not re-read assurance calibration evidence {calibration.source_path}: {exc}"
        ) from exc
    if current_digest != calibration.source_digest:
        raise MissionAssuranceError(
            "assurance calibration evidence changed during execution",
            phase="input_integrity",
        )
