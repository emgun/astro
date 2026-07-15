from __future__ import annotations

from pathlib import Path

from astro_assurance.io import load_post_launch_assurance_scenario
from astro_assurance.model_form_models import (
    MODEL_FORM_FACTORIAL_CONTRAST_IDS,
    ModelFormContrastId,
    ModelFormFactorialCellResult,
    ModelFormFactorialContrastResult,
    ModelFormFactorialProtocol,
    ModelFormFactorialRealizationResult,
    ModelFormFactorialResult,
    summarize_model_form_factorial,
)
from astro_assurance.models import PostLaunchAssuranceScenario
from astro_assurance.validation_models import (
    AssuranceValidationProfile,
    AssuranceValidationProfileResult,
    AssuranceValidationRealization,
    AssuranceValidationStatus,
    PairedAssuranceValidationProtocol,
)
from astro_assurance.validation_runner import (
    _assert_calibration_source_unchanged,
    _assert_protocol_source_unchanged,
    _resolve_assurance_references,
    run_assurance_validation_profile,
    validate_paired_assurance_protocol,
)
from astro_core.errors import InvalidScenarioError


def validate_model_form_factorial_protocol(protocol: ModelFormFactorialProtocol) -> None:
    validate_paired_assurance_protocol(_paired_protocol_view(protocol))


def run_model_form_factorial(
    protocol: ModelFormFactorialProtocol,
) -> ModelFormFactorialResult:
    paired_protocol = _paired_protocol_view(protocol)
    calibration = validate_paired_assurance_protocol(paired_protocol)
    assurance_path = Path(protocol.assurance_scenario)
    base = _resolve_assurance_references(
        assurance_path, load_post_launch_assurance_scenario(assurance_path)
    )
    if protocol.source_path is None or protocol.source_digest is None:
        raise InvalidScenarioError("model-form factorial protocol lacks source provenance")
    if base.source_path is None or base.source_digest is None:
        raise InvalidScenarioError("model-form factorial assurance scenario lacks provenance")
    if calibration.source_path is None or calibration.source_digest is None:
        raise InvalidScenarioError("model-form factorial calibration lacks provenance")
    realizations = tuple(
        _run_realization(paired_protocol, base, realization, protocol.profiles)
        for realization in protocol.realizations
    )
    _assert_protocol_source_unchanged(paired_protocol)
    _assert_calibration_source_unchanged(calibration)
    _assert_source_unchanged(base.source_path, base.source_digest, "assurance scenario")
    return ModelFormFactorialResult(
        protocol_id=protocol.protocol_id,
        calibration_protocol_id=protocol.calibration_protocol_id,
        protocol_source_path=protocol.source_path,
        protocol_source_digest=protocol.source_digest,
        assurance_source_path=base.source_path,
        assurance_source_digest=base.source_digest,
        calibration_id=calibration.calibration_id,
        calibration_source_path=calibration.source_path,
        calibration_source_digest=calibration.source_digest,
        calibration_promotion_status=calibration.promotion_status,
        calibration_claim_boundary=calibration.claim_boundary,
        realizations=realizations,
        summary=summarize_model_form_factorial(realizations),
        claim_boundary=protocol.claim_boundary,
        warnings=(
            "Profile counts and contrasts are unpooled design-space evidence, not probabilities.",
            "Matched J2 behavior does not establish physical-truth or flight authority.",
        ),
        metadata={
            "profile_order": [profile.value for profile in protocol.profiles],
            "contrast_order": list(MODEL_FORM_FACTORIAL_CONTRAST_IDS),
            "common_random_numbers": "same realization seed across all four cells",
        },
    )


def _assert_source_unchanged(path: str, expected_digest: str, role: str) -> None:
    from hashlib import sha256

    try:
        current_digest = sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise InvalidScenarioError(f"Could not re-read {role} {path}: {exc}") from exc
    if current_digest != expected_digest:
        from astro_assurance.errors import MissionAssuranceError

        raise MissionAssuranceError(f"{role} changed during execution", phase="input_integrity")


def _run_realization(
    protocol: PairedAssuranceValidationProtocol,
    base: PostLaunchAssuranceScenario,
    realization: AssuranceValidationRealization,
    profiles: tuple[AssuranceValidationProfile, ...],
) -> ModelFormFactorialRealizationResult:
    profile_results = {
        profile: run_assurance_validation_profile(protocol, base, realization, profile)
        for profile in profiles
    }
    cells = tuple(
        ModelFormFactorialCellResult(
            case_id=realization.case_id, profile_result=profile_results[profile]
        )
        for profile in profiles
    )
    contrasts = _contrasts(profile_results)
    return ModelFormFactorialRealizationResult(
        case_id=realization.case_id,
        realization=realization,
        cells=cells,
        contrasts=contrasts,
    )


def _contrasts(
    results: dict[AssuranceValidationProfile, AssuranceValidationProfileResult],
) -> tuple[ModelFormFactorialContrastResult, ...]:
    low = _difference(
        "estimator_j2_minus_two_body_under_truth_two_body",
        results[AssuranceValidationProfile.TRUTH_TWO_BODY_ESTIMATOR_J2],
        results[AssuranceValidationProfile.MATCHED_TWO_BODY],
    )
    high = _difference(
        "estimator_j2_minus_two_body_under_truth_j2",
        results[AssuranceValidationProfile.MATCHED_J2],
        results[AssuranceValidationProfile.TRUTH_J2_ESTIMATOR_TWO_BODY],
    )
    interaction_complete = low.complete and high.complete
    interaction = ModelFormFactorialContrastResult(
        contrast_id="difference_in_differences_interaction",
        complete=interaction_complete,
        metric_deltas=(
            {
                metric: float(high.metric_deltas[metric] - low.metric_deltas[metric])
                for metric in sorted(set(low.metric_deltas) & set(high.metric_deltas))
            }
            if interaction_complete
            else {}
        ),
    )
    return low, high, interaction


def _difference(
    contrast_id: ModelFormContrastId,
    minuend: AssuranceValidationProfileResult,
    subtrahend: AssuranceValidationProfileResult,
) -> ModelFormFactorialContrastResult:
    complete = (
        minuend.status is AssuranceValidationStatus.SUCCESS
        and subtrahend.status is AssuranceValidationStatus.SUCCESS
    )
    return ModelFormFactorialContrastResult(
        contrast_id=contrast_id,
        complete=complete,
        metric_deltas=(
            {
                metric: float(minuend.metrics[metric] - subtrahend.metrics[metric])
                for metric in sorted(set(minuend.metrics) & set(subtrahend.metrics))
            }
            if complete
            else {}
        ),
    )


def _paired_protocol_view(
    protocol: ModelFormFactorialProtocol,
) -> PairedAssuranceValidationProtocol:
    payload = protocol.model_dump(mode="python", exclude={"profiles"})
    payload["protocol_id"] = protocol.calibration_protocol_id
    payload.pop("calibration_protocol_id")
    payload["source_path"] = protocol.source_path
    payload["source_digest"] = protocol.source_digest
    payload["claim_boundary"] = (
        "paired_simulation_design_space_validation_not_operational_probability_or_flight_authority"
    )
    return PairedAssuranceValidationProtocol.model_validate(payload)
