from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata, spearmanr  # type: ignore[import-untyped]

from astro_uq.models import (
    CampaignDefinition,
    CampaignSensitivityReport,
    CaseObservation,
    MetricValueKind,
    OutcomeStatus,
    ParameterRealization,
    RequirementOperator,
    SensitivityEstimate,
    SensitivityParameter,
    SensitivityTargetKind,
    SensitivityTargetSummary,
)

_MINIMUM_SAMPLES = 30
_MINIMUM_RESIDUAL_DEGREES_OF_FREEDOM = 20
_MINIMUM_UNIQUE_VALUES = 5
_TIE_WARNING_FRACTION = 0.2
_CONDITION_WARNING = 1.0e4
_MAXIMUM_CONDITION_NUMBER = 1.0e8


class SensitivityAnalysisError(ValueError):
    """Campaign evidence cannot support the requested sensitivity analysis."""


def analyze_campaign_sensitivity(
    definition: CampaignDefinition,
    samples: tuple[ParameterRealization, ...],
    observations: tuple[CaseObservation, ...],
    *,
    metric_ids: Sequence[str] = (),
    requirement_margin_ids: Sequence[str] = (),
    definition_digest: str,
    samples_digest: str,
    cases_digest: str,
) -> CampaignSensitivityReport:
    """Compute rank associations over one completed campaign design space."""
    parameters = definition.uncertainty.parameters
    parameter_count = len(parameters)
    minimum_samples = max(_MINIMUM_SAMPLES, 5 * (parameter_count + 1))
    if parameter_count == 0:
        raise SensitivityAnalysisError("sensitivity analysis requires continuous parameters")
    if definition.uncertainty.model_variants:
        raise SensitivityAnalysisError("sensitivity v1 does not mix discrete model variants")
    if len(observations) < minimum_samples:
        raise SensitivityAnalysisError(
            f"sensitivity analysis requires at least {minimum_samples} cases for "
            f"{parameter_count} parameters"
        )
    residual_degrees_of_freedom = len(observations) - parameter_count - 1
    if residual_degrees_of_freedom < _MINIMUM_RESIDUAL_DEGREES_OF_FREEDOM:
        raise SensitivityAnalysisError(
            "sensitivity analysis requires at least "
            f"{_MINIMUM_RESIDUAL_DEGREES_OF_FREEDOM} residual degrees of freedom"
        )
    if any(observation.outcome_status is not OutcomeStatus.SUCCESS for observation in observations):
        raise SensitivityAnalysisError(
            "sensitivity analysis requires every campaign case to succeed"
        )
    if not metric_ids and not requirement_margin_ids:
        raise SensitivityAnalysisError("select at least one metric or requirement margin")
    if len(set(metric_ids)) != len(metric_ids) or len(set(requirement_margin_ids)) != len(
        requirement_margin_ids
    ):
        raise SensitivityAnalysisError("sensitivity target ids must be unique")

    sample_by_id = {sample.sample_id: sample for sample in samples}
    observation_by_id = {observation.sample_id: observation for observation in observations}
    if len(sample_by_id) != len(samples) or len(observation_by_id) != len(observations):
        raise SensitivityAnalysisError("sample and case ids must be unique")
    if not set(observation_by_id) <= set(sample_by_id):
        raise SensitivityAnalysisError("every case id must match one planned sample")
    ordered_ids = tuple(
        sample.sample_id for sample in samples if sample.sample_id in observation_by_id
    )
    active_samples = tuple(sample_by_id[item] for item in ordered_ids)
    weights = np.asarray([float(sample_by_id[item].weight) for item in ordered_ids])
    if not np.allclose(weights, weights[0], rtol=0.0, atol=1.0e-15):
        raise SensitivityAnalysisError("sensitivity v1 requires equal sample weights")

    parameter_ids = tuple(parameter.parameter_id for parameter in parameters)
    inputs = _input_matrix(active_samples, parameter_ids)
    ranked_inputs = _rank_columns(inputs, labels=parameter_ids, kind="parameter")
    standardized_inputs = _standardize_columns(ranked_inputs)
    design_condition = float(np.linalg.cond(standardized_inputs))
    design_rank = int(np.linalg.matrix_rank(standardized_inputs))
    if design_rank != parameter_count:
        raise SensitivityAnalysisError(
            "ranked parameter design is singular: "
            f"rank {design_rank}, expected {parameter_count}"
        )
    if not np.isfinite(design_condition) or design_condition > _MAXIMUM_CONDITION_NUMBER:
        raise SensitivityAnalysisError(
            "ranked parameter design is singular or ill-conditioned; "
            f"condition number {design_condition:.6g} exceeds {_MAXIMUM_CONDITION_NUMBER:.6g}"
        )

    target_specs = _resolve_targets(definition, metric_ids, requirement_margin_ids)
    target_summaries = tuple(
        _analyze_target(
            target_id=target_id,
            kind=kind,
            unit=unit,
            source_metric_id=source_metric_id,
            requirement_operator=operator,
            orientation=orientation,
            values=_target_values(
                ordered_ids,
                observation_by_id,
                target_id=target_id,
                kind=kind,
            ),
            parameter_ids=parameter_ids,
            ranked_inputs=ranked_inputs,
        )
        for target_id, kind, unit, source_metric_id, operator, orientation in target_specs
    )
    warnings = _diagnostic_warnings(
        ranked_inputs=ranked_inputs,
        parameter_ids=parameter_ids,
        target_summaries=target_summaries,
        design_condition=design_condition,
    )
    return CampaignSensitivityReport(
        campaign_id=definition.campaign_id,
        definition_digest=definition_digest,
        samples_digest=samples_digest,
        cases_digest=cases_digest,
        claim_boundary=definition.evaluator.claim_boundary,
        sampler_kind=definition.sampler.kind,
        sample_count=len(active_samples),
        effective_sample_size=float(len(active_samples)),
        parameter_count=parameter_count,
        residual_degrees_of_freedom=residual_degrees_of_freedom,
        ranked_design_rank=design_rank,
        ranked_design_condition_number=max(1.0, design_condition),
        parameters=tuple(
            SensitivityParameter(
                parameter_id=parameter.parameter_id,
                target=parameter.target,
                unit=parameter.unit,
                uncertainty_kind=parameter.uncertainty_kind,
            )
            for parameter in parameters
        ),
        targets=target_summaries,
        warnings=(
            "Rank associations describe the configured campaign design space; they are not causal "
            "effects, Sobol variance indices, operational probabilities, or certification "
            "evidence.",
            "PRCC measures monotonic association after linear residualization in rank space and "
            "may miss interactions or non-monotonic effects.",
            *warnings,
        ),
    )


def _input_matrix(
    samples: tuple[ParameterRealization, ...], parameter_ids: tuple[str, ...]
) -> NDArray[np.float64]:
    rows: list[list[float]] = []
    for sample in samples:
        row: list[float] = []
        for parameter_id in parameter_ids:
            value = sample.physical_values.get(parameter_id)
            if value is None or isinstance(value, str | bool):
                raise SensitivityAnalysisError(
                    f"parameter {parameter_id!r} must have a numeric physical value in every case"
                )
            row.append(float(value))
        rows.append(row)
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise SensitivityAnalysisError("parameter values must be finite")
    return matrix


def _resolve_targets(
    definition: CampaignDefinition,
    metric_ids: Sequence[str],
    requirement_margin_ids: Sequence[str],
) -> tuple[
    tuple[
        str,
        SensitivityTargetKind,
        str,
        str,
        RequirementOperator | None,
        Literal["association_only", "higher_is_safer"],
    ],
    ...,
]:
    metric_by_id = {metric.metric_id: metric for metric in definition.metrics}
    requirement_by_id = {
        requirement.requirement_id: requirement for requirement in definition.requirements
    }
    targets: list[
        tuple[
            str,
            SensitivityTargetKind,
            str,
            str,
            RequirementOperator | None,
            Literal["association_only", "higher_is_safer"],
        ]
    ] = []
    for metric_id in metric_ids:
        metric = metric_by_id.get(metric_id)
        if metric is None:
            raise SensitivityAnalysisError(f"unknown campaign metric {metric_id!r}")
        if metric.value_kind not in {
            MetricValueKind.NUMERIC,
            MetricValueKind.EVENT_TIME,
            MetricValueKind.TIME_SERIES_SUMMARY,
        }:
            raise SensitivityAnalysisError(f"metric {metric_id!r} is not numeric")
        if metric.unit is None:
            raise SensitivityAnalysisError(f"metric {metric_id!r} has no unit")
        targets.append(
            (
                metric_id,
                SensitivityTargetKind.METRIC,
                metric.unit,
                metric_id,
                None,
                "association_only",
            )
        )
    for requirement_id in requirement_margin_ids:
        requirement = requirement_by_id.get(requirement_id)
        if requirement is None:
            raise SensitivityAnalysisError(f"unknown campaign requirement {requirement_id!r}")
        if requirement.operator in {RequirementOperator.IS_TRUE, RequirementOperator.IS_FALSE}:
            raise SensitivityAnalysisError(
                f"requirement {requirement_id!r} has a boolean sentinel margin, not a distance"
            )
        metric = metric_by_id[requirement.metric_id]
        if metric.unit is None:
            raise SensitivityAnalysisError(
                f"requirement {requirement_id!r} does not reference a numeric metric"
            )
        targets.append(
            (
                requirement_id,
                SensitivityTargetKind.REQUIREMENT_MARGIN,
                metric.unit,
                requirement.metric_id,
                requirement.operator,
                "higher_is_safer",
            )
        )
    if len({(item[0], item[1]) for item in targets}) != len(targets):
        raise SensitivityAnalysisError("sensitivity targets must be unique")
    return tuple(targets)


def _target_values(
    ordered_ids: tuple[str, ...],
    observations: dict[str, CaseObservation],
    *,
    target_id: str,
    kind: SensitivityTargetKind,
) -> NDArray[np.float64]:
    values: list[float] = []
    for sample_id in ordered_ids:
        observation = observations[sample_id]
        if kind is SensitivityTargetKind.METRIC:
            value = observation.metric_values.get(target_id)
            if value is None or isinstance(value, str | bool):
                raise SensitivityAnalysisError(
                    f"metric {target_id!r} must be numeric in every successful case"
                )
            values.append(float(value))
            continue
        matches = [
            requirement.margin
            for requirement in observation.requirements
            if requirement.requirement_id == target_id
        ]
        if len(matches) != 1 or matches[0] is None:
            raise SensitivityAnalysisError(
                f"requirement {target_id!r} must have one numeric margin in every successful case"
            )
        values.append(float(matches[0]))
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise SensitivityAnalysisError(f"target {target_id!r} values must be finite")
    if np.unique(result).size == 1:
        raise SensitivityAnalysisError(f"target {target_id!r} is constant")
    return result


def _analyze_target(
    *,
    target_id: str,
    kind: SensitivityTargetKind,
    unit: str,
    source_metric_id: str,
    requirement_operator: RequirementOperator | None,
    orientation: Literal["association_only", "higher_is_safer"],
    values: NDArray[np.float64],
    parameter_ids: tuple[str, ...],
    ranked_inputs: NDArray[np.float64],
) -> SensitivityTargetSummary:
    ranked_target = np.asarray(rankdata(values, method="average"), dtype=np.float64)
    target_unique_count = int(np.unique(ranked_target).size)
    if target_unique_count < _MINIMUM_UNIQUE_VALUES:
        raise SensitivityAnalysisError(
            f"target {target_id!r} requires at least {_MINIMUM_UNIQUE_VALUES} unique values"
        )
    coefficients: list[tuple[str, float, float]] = []
    for index, parameter_id in enumerate(parameter_ids):
        rho = float(spearmanr(ranked_inputs[:, index], ranked_target).statistic)
        prcc = _partial_rank_correlation(ranked_inputs, ranked_target, index)
        if not np.isfinite(rho) or not np.isfinite(prcc):
            raise SensitivityAnalysisError(
                f"target {target_id!r} produced a non-finite rank coefficient"
            )
        coefficients.append((parameter_id, rho, prcc))
    ranked = sorted(coefficients, key=lambda item: (-abs(item[2]), item[0]))
    rank_by_parameter = {
        parameter_id: rank
        for rank, (parameter_id, _rho, _prcc) in enumerate(ranked, 1)
    }
    estimates = tuple(
        SensitivityEstimate(
            parameter_id=parameter_id,
            spearman_rho=rho,
            partial_rank_correlation=prcc,
            absolute_prcc_rank=rank_by_parameter[parameter_id],
            input_unique_count=int(np.unique(ranked_inputs[:, index]).size),
            input_tie_fraction=_tie_fraction(ranked_inputs[:, index]),
        )
        for index, (parameter_id, rho, prcc) in enumerate(coefficients)
    )
    return SensitivityTargetSummary(
        target_id=target_id,
        kind=kind,
        unit=unit,
        source_metric_id=source_metric_id,
        orientation=orientation,
        requirement_operator=requirement_operator,
        sample_count=len(values),
        target_unique_count=target_unique_count,
        target_tie_fraction=_tie_fraction(ranked_target),
        estimates=estimates,
        largest_absolute_prcc_parameter_id=ranked[0][0],
    )


def _rank_columns(
    values: NDArray[np.float64], *, labels: tuple[str, ...], kind: str
) -> NDArray[np.float64]:
    ranked = np.empty_like(values)
    for index, label in enumerate(labels):
        column = values[:, index]
        unique_count = int(np.unique(column).size)
        if unique_count < _MINIMUM_UNIQUE_VALUES:
            raise SensitivityAnalysisError(
                f"{kind} {label!r} requires at least {_MINIMUM_UNIQUE_VALUES} unique values"
            )
        ranked[:, index] = rankdata(column, method="average")
    return ranked


def _standardize_columns(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = values - np.mean(values, axis=0)
    scale = np.std(centered, axis=0, ddof=1)
    if np.any(scale <= 0.0):
        raise SensitivityAnalysisError("ranked parameter design contains a constant column")
    return np.asarray(centered / scale, dtype=np.float64)


def _partial_rank_correlation(
    ranked_inputs: NDArray[np.float64], ranked_target: NDArray[np.float64], index: int
) -> float:
    controls = np.delete(ranked_inputs, index, axis=1)
    parameter = ranked_inputs[:, index]
    if controls.shape[1] == 0:
        return float(np.corrcoef(parameter, ranked_target)[0, 1])
    design = np.column_stack((np.ones(len(controls)), controls))
    parameter_residual = parameter - design @ np.linalg.lstsq(design, parameter, rcond=None)[0]
    target_residual = ranked_target - design @ np.linalg.lstsq(design, ranked_target, rcond=None)[0]
    if np.std(parameter_residual) <= 0.0 or np.std(target_residual) <= 0.0:
        raise SensitivityAnalysisError("partial-rank residuals are constant")
    return float(np.corrcoef(parameter_residual, target_residual)[0, 1])


def _tie_fraction(values: NDArray[np.float64]) -> float:
    return 1.0 - float(np.unique(values).size) / len(values)


def _diagnostic_warnings(
    *,
    ranked_inputs: NDArray[np.float64],
    parameter_ids: tuple[str, ...],
    target_summaries: tuple[SensitivityTargetSummary, ...],
    design_condition: float,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if design_condition > _CONDITION_WARNING:
        warnings.append(
            f"Ranked design condition number {design_condition:.6g} exceeds warning threshold "
            f"{_CONDITION_WARNING:.6g}."
        )
    for index, parameter_id in enumerate(parameter_ids):
        tie_fraction = _tie_fraction(ranked_inputs[:, index])
        if tie_fraction > _TIE_WARNING_FRACTION:
            warnings.append(
                f"Parameter {parameter_id!r} tie fraction {tie_fraction:.6f} exceeds "
                f"{_TIE_WARNING_FRACTION:.6f}."
            )
    for target in target_summaries:
        if target.target_tie_fraction > _TIE_WARNING_FRACTION:
            warnings.append(
                f"Target {target.target_id!r} tie fraction {target.target_tie_fraction:.6f} "
                f"exceeds {_TIE_WARNING_FRACTION:.6f}."
            )
    return tuple(warnings)
