from __future__ import annotations

from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigvalsh  # type: ignore[import-untyped]
from scipy.stats import chi2  # type: ignore[import-untyped]

from astro_assurance.covariance_validation_models import (
    CovarianceComparisonSummary,
    CovarianceEpochComparisonDiagnostic,
    CovarianceIndependenceReview,
    CovarianceSourceBinding,
    CovarianceValidationBlocker,
    CovarianceValidationDisposition,
    CovarianceValidationProtocol,
    CovarianceValidationResult,
    EmpiricalCovarianceArtifact,
    EmpiricalNEESSummary,
    ForceFeature,
)
from astro_core.errors import InvalidScenarioError
from astro_core.models import CovarianceSample, ForceModelConfig, Trajectory, TrajectorySample

FloatArray = NDArray[np.float64]


def assess_covariance_validation(
    protocol: CovarianceValidationProtocol,
    candidate: Trajectory,
    reference: Trajectory,
    empirical: EmpiricalCovarianceArtifact | None,
    independence_review: CovarianceIndependenceReview | None,
    bindings: tuple[CovarianceSourceBinding, ...],
) -> CovarianceValidationResult:
    if protocol.source_path is None or protocol.source_digest is None:
        raise InvalidScenarioError("covariance validation protocol lacks source provenance")
    diagnostics, requested_epochs = _compare_trajectories(protocol, candidate, reference)
    summary = _comparison_summary(diagnostics, requested_epochs)
    blockers = _evidence_blockers(
        protocol, candidate, reference, empirical, independence_review
    )
    empirical_summary = None if empirical is None else _assess_empirical(protocol, empirical)
    numerical_failure = any(not item.criteria_satisfied for item in diagnostics) or (
        empirical_summary is not None and not empirical_summary.criteria_satisfied
    )
    disposition = (
        CovarianceValidationDisposition.CRITERIA_FAILED
        if numerical_failure
        else CovarianceValidationDisposition.ADDITIONAL_EVIDENCE_REQUIRED
        if blockers
        else CovarianceValidationDisposition.CRITERIA_SATISFIED
    )
    return CovarianceValidationResult(
        protocol_id=protocol.protocol_id,
        source_bindings=bindings,
        diagnostics=diagnostics,
        comparison_summary=summary,
        empirical_nees_summary=empirical_summary,
        blockers=blockers,
        disposition=disposition,
        claim_boundary=protocol.claim_boundary,
    )

def _compare_trajectories(
    protocol: CovarianceValidationProtocol,
    candidate: Trajectory,
    reference: Trajectory,
) -> tuple[tuple[CovarianceEpochComparisonDiagnostic, ...], int]:
    if candidate.scenario_id != reference.scenario_id:
        raise InvalidScenarioError("covariance trajectories must use the same scenario id")
    candidate_covariances = _covariances_by_epoch(candidate)
    reference_covariances = _covariances_by_epoch(reference)
    candidate_states = _states_by_epoch(candidate)
    reference_states = _states_by_epoch(reference)
    if set(candidate_covariances) != set(reference_covariances):
        raise InvalidScenarioError("covariance trajectories must use exact matching epochs")
    epochs = sorted(candidate_covariances)
    if any(epoch not in candidate_states or epoch not in reference_states for epoch in epochs):
        raise InvalidScenarioError("every covariance epoch must have a matching trajectory state")
    return (
        tuple(
            _compare_epoch(
                protocol,
                candidate_covariances[epoch],
                reference_covariances[epoch],
                candidate_states[epoch],
                reference_states[epoch],
            )
            for epoch in epochs
        ),
        len(epochs),
    )


def _covariances_by_epoch(trajectory: Trajectory) -> dict[datetime, CovarianceSample]:
    result: dict[datetime, CovarianceSample] = {}
    for sample in trajectory.covariance_history:
        if sample.epoch in result:
            raise InvalidScenarioError("covariance trajectory contains duplicate covariance epochs")
        result[sample.epoch] = sample
    if not result:
        raise InvalidScenarioError("covariance trajectory contains no covariance history")
    return result


def _states_by_epoch(trajectory: Trajectory) -> dict[datetime, TrajectorySample]:
    return {sample.epoch: sample for sample in trajectory.samples}


def _matrix(value: list[list[float]] | tuple[tuple[float, ...], ...]) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


def _validate_positive_definite(matrix: FloatArray, label: str) -> tuple[float, float]:
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    symmetric = (matrix + matrix.T) / 2.0
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric)))
    if minimum_eigenvalue <= 0.0:
        raise InvalidScenarioError(f"{label} covariance must be positive definite")
    condition = float(np.linalg.cond(symmetric))
    if not np.isfinite(condition):
        raise InvalidScenarioError(f"{label} covariance condition number must be finite")
    return symmetry_error, condition


def _compare_epoch(
    protocol: CovarianceValidationProtocol,
    candidate: CovarianceSample,
    reference: CovarianceSample,
    candidate_state: TrajectorySample,
    reference_state: TrajectorySample,
) -> CovarianceEpochComparisonDiagnostic:
    candidate_matrix = _matrix(candidate.covariance)
    reference_matrix = _matrix(reference.covariance)
    candidate_symmetry, candidate_condition = _validate_positive_definite(
        candidate_matrix, "candidate"
    )
    reference_symmetry, reference_condition = _validate_positive_definite(
        reference_matrix, "reference"
    )
    candidate_symmetric = (candidate_matrix + candidate_matrix.T) / 2.0
    reference_symmetric = (reference_matrix + reference_matrix.T) / 2.0
    candidate_minimum = float(np.min(np.linalg.eigvalsh(candidate_symmetric)))
    reference_minimum = float(np.min(np.linalg.eigvalsh(reference_symmetric)))
    relative_error = float(
        np.linalg.norm(candidate_matrix - reference_matrix, ord="fro")
        / np.linalg.norm(reference_matrix, ord="fro")
    )
    generalized = eigvalsh(candidate_symmetric, reference_symmetric)
    trace_ratio = float(np.trace(candidate_symmetric) / np.trace(reference_symmetric))
    candidate_transition = candidate.accumulated_state_transition_matrix
    reference_transition = reference.accumulated_state_transition_matrix
    if candidate_transition is None or reference_transition is None:
        raise InvalidScenarioError(
            "every covariance comparison epoch requires accumulated state-transition matrices"
        )
    reference_transition_matrix = _matrix(reference_transition)
    transition_error = float(
        np.linalg.norm(_matrix(candidate_transition) - reference_transition_matrix, ord="fro")
        / max(np.linalg.norm(reference_transition_matrix, ord="fro"), np.finfo(float).eps)
    )
    position_delta = float(
        np.linalg.norm(
            candidate_state.state.position_array() - reference_state.state.position_array()
        )
    )
    velocity_delta = float(
        np.linalg.norm(
            candidate_state.state.velocity_array() - reference_state.state.velocity_array()
        )
    )
    threshold = protocol.thresholds
    values = {
        "symmetry": max(candidate_symmetry, reference_symmetry)
        <= threshold.symmetry_tolerance,
        "candidate_minimum_eigenvalue": candidate_minimum >= threshold.minimum_eigenvalue,
        "reference_minimum_eigenvalue": reference_minimum >= threshold.minimum_eigenvalue,
        "candidate_condition_number": candidate_condition <= threshold.maximum_condition_number,
        "reference_condition_number": reference_condition <= threshold.maximum_condition_number,
        "relative_covariance_frobenius_error": relative_error
        <= threshold.maximum_relative_covariance_frobenius_error,
        "covariance_trace_ratio_minimum": trace_ratio
        >= threshold.covariance_trace_ratio_minimum,
        "covariance_trace_ratio_maximum": trace_ratio
        <= threshold.covariance_trace_ratio_maximum,
        "accumulated_state_transition_frobenius_error": transition_error
        <= threshold.maximum_accumulated_state_transition_frobenius_error,
        "generalized_eigenvalue_minimum": float(generalized[0])
        >= threshold.generalized_eigenvalue_minimum,
        "generalized_eigenvalue_maximum": float(generalized[-1])
        <= threshold.generalized_eigenvalue_maximum,
        "state_position_delta": position_delta <= threshold.maximum_state_position_delta_km,
        "state_velocity_delta": velocity_delta <= threshold.maximum_state_velocity_delta_km_s,
    }
    failed = tuple(name for name, passed in values.items() if not passed)
    return CovarianceEpochComparisonDiagnostic(
        epoch=candidate.epoch,
        symmetry_error=max(candidate_symmetry, reference_symmetry),
        candidate_minimum_eigenvalue=candidate_minimum,
        reference_minimum_eigenvalue=reference_minimum,
        candidate_condition_number=candidate_condition,
        reference_condition_number=reference_condition,
        relative_covariance_frobenius_error=relative_error,
        covariance_trace_ratio=trace_ratio,
        accumulated_state_transition_frobenius_error=transition_error,
        generalized_eigenvalue_minimum=float(generalized[0]),
        generalized_eigenvalue_maximum=float(generalized[-1]),
        state_position_delta_km=position_delta,
        state_velocity_delta_km_s=velocity_delta,
        criteria_satisfied=not failed,
        failed_criteria=failed,
    )


def _comparison_summary(
    diagnostics: tuple[CovarianceEpochComparisonDiagnostic, ...], requested: int
) -> CovarianceComparisonSummary:
    return CovarianceComparisonSummary(
        requested_epochs=max(requested, 0),
        compared_epochs=len(diagnostics),
        passed_epochs=sum(item.criteria_satisfied for item in diagnostics),
        maximum_symmetry_error=max(item.symmetry_error for item in diagnostics),
        minimum_candidate_eigenvalue=min(
            item.candidate_minimum_eigenvalue for item in diagnostics
        ),
        maximum_candidate_condition_number=max(
            item.candidate_condition_number for item in diagnostics
        ),
        maximum_relative_covariance_frobenius_error=max(
            item.relative_covariance_frobenius_error for item in diagnostics
        ),
        covariance_trace_ratio_minimum=min(
            item.covariance_trace_ratio for item in diagnostics
        ),
        covariance_trace_ratio_maximum=max(
            item.covariance_trace_ratio for item in diagnostics
        ),
        maximum_accumulated_state_transition_frobenius_error=max(
            item.accumulated_state_transition_frobenius_error for item in diagnostics
        ),
        generalized_eigenvalue_minimum=min(
            item.generalized_eigenvalue_minimum for item in diagnostics
        ),
        generalized_eigenvalue_maximum=max(
            item.generalized_eigenvalue_maximum for item in diagnostics
        ),
        maximum_state_position_delta_km=max(
            item.state_position_delta_km for item in diagnostics
        ),
        maximum_state_velocity_delta_km_s=max(
            item.state_velocity_delta_km_s for item in diagnostics
        ),
    )


def _force_features(force_model: ForceModelConfig) -> set[ForceFeature]:
    features: set[ForceFeature] = {force_model.gravity.value}  # type: ignore[arg-type]
    if force_model.gravity_degree is not None and force_model.gravity_degree > 2:
        features.add("high_order_gravity")
    if force_model.atmospheric_drag:
        features.add("atmospheric_drag")
    if force_model.solar_radiation_pressure:
        features.add("solar_radiation_pressure")
    if force_model.third_body_gravity:
        features.add("third_body_gravity")
    return features


def _producer_provenance_matches(trajectory: Trajectory, implementation: str) -> bool:
    if trajectory.metadata.get("covariance_implementation") != implementation:
        return False
    native_contracts = {
        "orekit_native_variational": (
            "orekit",
            "orekit_native_variational_equations",
            "orekit_native_variational",
            "covariance_native_variational_api",
        ),
        "tudat_native_variational": (
            "tudat",
            "tudat_native_variational_equations",
            "tudat_native_variational",
            "native_variational_solver",
        ),
    }
    contract = native_contracts.get(implementation)
    if contract is None:
        return True
    backend, model, sample_model, api_field = contract
    return (
        trajectory.backend == backend
        and trajectory.metadata.get("covariance_model") == model
        and bool(trajectory.metadata.get(api_field))
        and bool(trajectory.covariance_history)
        and all(
            sample.metadata.get("state_transition_model")
            == ("identity" if index == 0 else sample_model)
            and sample.metadata.get("covariance_model") == model
            for index, sample in enumerate(trajectory.covariance_history)
        )
    )


def _evidence_blockers(
    protocol: CovarianceValidationProtocol,
    candidate: Trajectory,
    reference: Trajectory,
    empirical: EmpiricalCovarianceArtifact | None,
    independence_review: CovarianceIndependenceReview | None,
) -> tuple[CovarianceValidationBlocker, ...]:
    blockers: list[CovarianceValidationBlocker] = []
    expected_semantics = protocol.units_policy.model_dump(mode="json")
    if candidate.metadata.get("covariance_units_policy") != expected_semantics or (
        reference.metadata.get("covariance_units_policy") != expected_semantics
    ):
        blockers.append(
            CovarianceValidationBlocker(
                blocker_id="trajectory_semantics_unbound",
                category="input",
                statement="Trajectory evidence does not bind the protocol covariance semantics.",
                required_evidence=(
                    "Candidate and reference trajectory metadata with exact "
                    "covariance_units_policy."
                ),
            )
        )
    if not protocol.independence.independent_implementations:
        blockers.append(
            CovarianceValidationBlocker(
                blocker_id="independent_implementation_missing",
                category="independence",
                statement="Candidate and reference are not declared independent implementations.",
                required_evidence="A reviewed independent implementation comparison.",
            )
        )
    elif independence_review is None:
        blockers.append(
            CovarianceValidationBlocker(
                blocker_id="independence_review_missing",
                category="independence",
                statement="No reviewed independence artifact is bound.",
                required_evidence="A typed, digest-bound implementation-independence review.",
            )
        )
    elif (
        not _producer_provenance_matches(
            candidate, protocol.independence.candidate_implementation
        )
        or not _producer_provenance_matches(
            reference, protocol.independence.reference_implementation
        )
        or (
            {
                protocol.independence.candidate_implementation,
                protocol.independence.reference_implementation,
            }
            == {"orekit_native_variational", "tudat_native_variational"}
            and candidate.backend == reference.backend
        )
    ):
        blockers.append(
            CovarianceValidationBlocker(
                blocker_id="implementation_provenance_mismatch",
                category="independence",
                statement=(
                    "Trajectory producer provenance does not match the preregistered "
                    "implementations."
                ),
                required_evidence=(
                    "Producer-emitted covariance_implementation identities matching "
                    "the protocol."
                ),
            )
        )
    elif (
        independence_review.candidate_implementation
        != protocol.independence.candidate_implementation
        or independence_review.reference_implementation
        != protocol.independence.reference_implementation
    ):
        raise InvalidScenarioError(
            "independence review implementation identities do not match protocol"
        )
    covered = _force_features(candidate.force_model) & _force_features(reference.force_model)
    missing = sorted(set(protocol.required_force_features) - covered)
    if missing:
        blockers.append(
            CovarianceValidationBlocker(
                blocker_id="required_force_features_missing",
                category="force_feature",
                statement=(
                    "Passing comparison does not cover required features: "
                    f"{', '.join(missing)}."
                ),
                required_evidence=(
                    "Aligned covariance comparison covering every required force feature."
                ),
            )
        )
    if empirical is None:
        blockers.append(
            CovarianceValidationBlocker(
                blocker_id="empirical_consistency_missing",
                category="empirical",
                statement="No raw empirical covariance-consistency evidence is bound.",
                required_evidence="Independent raw state-error and predicted-covariance samples.",
            )
        )
    elif len(empirical.samples) < protocol.thresholds.minimum_empirical_samples:
        blockers.append(
            CovarianceValidationBlocker(
                blocker_id="empirical_sample_count_insufficient",
                category="empirical",
                statement="Empirical campaign does not meet the preregistered sample count.",
                required_evidence=(
                    f"At least {protocol.thresholds.minimum_empirical_samples} independent samples."
                ),
            )
        )
    elif not empirical.independent_realizations:
        blockers.append(
            CovarianceValidationBlocker(
                blocker_id="empirical_realization_independence_missing",
                category="empirical",
                statement="Empirical campaign does not declare independent realizations.",
                required_evidence="A reviewed independence basis for the empirical realizations.",
            )
        )
    elif any(not sample.independent_truth for sample in empirical.samples):
        blockers.append(
            CovarianceValidationBlocker(
                blocker_id="empirical_truth_independence_missing",
                category="empirical",
                statement="At least one empirical sample lacks independent truth.",
                required_evidence="Independent truth for every empirical realization.",
            )
        )
    if len(candidate.covariance_history) < protocol.thresholds.minimum_epochs:
        blockers.append(
            CovarianceValidationBlocker(
                blocker_id="comparison_epoch_count_insufficient",
                category="comparison",
                statement="Comparison does not meet the preregistered epoch count.",
                required_evidence=f"At least {protocol.thresholds.minimum_epochs} aligned epochs.",
            )
        )
    return tuple(blockers)


def _assess_empirical(
    protocol: CovarianceValidationProtocol,
    empirical: EmpiricalCovarianceArtifact,
) -> EmpiricalNEESSummary:
    if empirical.units_policy != protocol.units_policy:
        raise InvalidScenarioError("empirical covariance units policy does not match protocol")
    values: list[float] = []
    for sample in empirical.samples:
        covariance = _matrix(sample.predicted_covariance)
        _validate_positive_definite(covariance, f"empirical sample {sample.sample_id}")
        error = np.asarray(sample.state_error, dtype=np.float64)
        values.append(float(error @ np.linalg.solve(covariance, error)))
    count = len(values)
    alpha = 1.0 - float(protocol.thresholds.confidence_level)
    individual_lower = float(chi2.ppf(alpha / 2.0, df=6))
    individual_upper = float(chi2.ppf(1.0 - alpha / 2.0, df=6))
    mean_lower = float(chi2.ppf(alpha / 2.0, df=6 * count) / count)
    mean_upper = float(chi2.ppf(1.0 - alpha / 2.0, df=6 * count) / count)
    within = sum(individual_lower <= value <= individual_upper for value in values)
    coverage = within / count
    mean_nees = float(np.mean(values))
    return EmpiricalNEESSummary(
        sample_count=count,
        confidence_level=protocol.thresholds.confidence_level,
        individual_lower_bound=individual_lower,
        individual_upper_bound=individual_upper,
        mean_lower_bound=mean_lower,
        mean_upper_bound=mean_upper,
        mean_nees=mean_nees,
        samples_within_bounds=within,
        coverage=coverage,
        criteria_satisfied=(
            mean_lower <= mean_nees <= mean_upper
            and coverage >= protocol.thresholds.minimum_coverage
        ),
    )
