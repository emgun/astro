from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np

from astro_core.models import AstroModel
from astro_uq.correlations import gaussian_copula_transform
from astro_uq.distributions import build_realization
from astro_uq.evaluators import AuthoritativeCallableEvaluator, evaluate_authoritatively
from astro_uq.io import (
    CASES_FILE,
    SAMPLES_FILE,
    CampaignArtifactStore,
    CampaignIOError,
    canonical_hash,
)
from astro_uq.metrics import MetricRegistry, evaluate_requirements
from astro_uq.models import (
    CampaignDefinition,
    CampaignResult,
    CampaignState,
    CampaignStatistics,
    CaseObservation,
    ConfidenceIntervalStopping,
    EvaluatorKind,
    FixedCountStopping,
    OutcomeStatus,
    ParameterRealization,
    RequirementOutcome,
    RetentionPolicy,
    SamplePlan,
    ScenarioRealization,
)
from astro_uq.parameters import ParameterBindingError, ParameterRegistry, model_digest
from astro_uq.samplers import generate_samples
from astro_uq.statistics import effective_sample_size, summarize_campaign
from astro_uq.stopping import (
    ConfidenceIntervalRule,
    MetricStabilityRule,
    StoppingDecision,
    confidence_interval_decision,
    fixed_count_decision,
    metric_stability_decision,
)


@dataclass(frozen=True)
class CampaignRuntime:
    scenario: AstroModel
    parameters: ParameterRegistry
    metrics: MetricRegistry
    evaluate: Callable[[AstroModel], AstroModel]
    serialize: Callable[[AstroModel], tuple[str, ...]] = lambda _result: ()


_WORKER_DEFINITION: CampaignDefinition | None = None
_WORKER_RUNTIME: CampaignRuntime | None = None


def run_campaign(
    definition: CampaignDefinition,
    runtime: CampaignRuntime,
    *,
    output_dir: str | Path,
    software_compatibility: Mapping[str, str],
    resume: bool = False,
    workers: int = 1,
    runtime_factory: Callable[[], CampaignRuntime] | None = None,
) -> CampaignResult:
    if workers < 1:
        raise CampaignIOError("workers must be at least one")
    if workers > 1 and runtime_factory is None:
        raise CampaignIOError("parallel execution requires a per-worker runtime factory")
    if definition.evaluator.kind is not EvaluatorKind.AUTHORITATIVE:
        raise CampaignIOError(
            f"evaluator kind {definition.evaluator.kind.value!r} is not implemented; "
            "campaign execution fails closed"
        )
    store = CampaignArtifactStore(output_dir)
    definition_digest = canonical_hash(definition)
    sample_plan = SamplePlan(
        sampler=definition.sampler,
        campaign_digest=definition_digest,
    )
    parameter_ids = tuple(parameter.parameter_id for parameter in definition.uncertainty.parameters)
    batch = generate_samples(
        sample_plan,
        parameter_ids=parameter_ids,
        model_variants=definition.uncertainty.model_variants,
    )
    realized_samples = tuple(_physical_realization(raw, definition) for raw in batch.samples)
    if isinstance(definition.stopping, ConfidenceIntervalStopping):
        sample_weights = np.asarray(
            [float(sample.weight) for sample in realized_samples], dtype=np.float64
        )
        if sample_weights.size and not np.allclose(
            sample_weights,
            sample_weights[0],
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise CampaignIOError(
                "CI half-width stopping requires equal sample weights; "
                "weighted intervals are not implemented"
            )

    with store:
        if resume:
            resume_state = store.resume(
                definition,
                software_compatibility=software_compatibility,
            )
            completed = set(resume_state.completed_sample_ids)
            observations = [CaseObservation.model_validate(case) for case in resume_state.cases]
            stored_samples = tuple(
                ParameterRealization.model_validate(sample) for sample in resume_state.samples
            )
            if stored_samples != realized_samples:
                prefix = realized_samples[: len(stored_samples)]
                if resume_state.cases or stored_samples != prefix:
                    raise CampaignIOError(
                        "stored sample evidence does not match the deterministic sample plan"
                    )
                for missing_sample in realized_samples[len(stored_samples) :]:
                    store.append_sample(missing_sample)
                stored_samples = realized_samples
            expected_ids = {sample.sample_id for sample in realized_samples}
            if not completed <= expected_ids:
                raise CampaignIOError("case index contains sample ids outside the sample plan")
            completed_statistics = (
                CampaignStatistics.model_validate(resume_state.statistics)
                if resume_state.state is CampaignState.COMPLETED
                else None
            )
            prior_statistics = (
                None
                if resume_state.statistics is None
                else CampaignStatistics.model_validate(resume_state.statistics)
            )
        else:
            store.initialize(definition, software_compatibility=software_compatibility)
            completed = set()
            observations = []
            completed_statistics = None
            prior_statistics = None
            for sample in realized_samples:
                store.append_sample(sample)

        if completed_statistics is not None:
            statistics = completed_statistics
        else:
            store.set_state(CampaignState.RUNNING)
        try:
            pending_samples: tuple[ParameterRealization, ...]
            if completed_statistics is not None:
                pending_samples = ()
            else:
                pending_samples = tuple(
                    sample for sample in realized_samples if sample.sample_id not in completed
                )
            sample_limit = _sample_limit(definition)
            pending_samples = pending_samples[: max(0, sample_limit - len(observations))]
            convergence_history: list[dict[str, object]] = (
                []
                if prior_statistics is None
                else [dict(record) for record in prior_statistics.convergence_history]
            )
            weights = {sample.sample_id: float(sample.weight) for sample in realized_samples}
            batch_size = _stopping_batch_size(definition)
            for batch_start in range(0, len(pending_samples), batch_size):
                sample_batch = pending_samples[batch_start : batch_start + batch_size]
                if workers == 1:
                    new_observations = tuple(
                        _evaluate_sample(definition, runtime, sample) for sample in sample_batch
                    )
                else:
                    assert runtime_factory is not None
                    new_observations = _parallel_observations(
                        definition,
                        sample_batch,
                        workers=workers,
                        runtime_factory=runtime_factory,
                    )
                for observation in new_observations:
                    observations.append(observation)
                    store.append_case(observation)
                    completed.add(observation.sample_id)
                decision, record = _stopping_decision(
                    definition,
                    tuple(observations),
                    realized_samples,
                    convergence_history,
                )
                convergence_history.append(record)
                checkpoint_weights = {
                    observation.sample_id: weights[observation.sample_id]
                    for observation in observations
                }
                store.write_statistics(
                    summarize_campaign(
                        requested_samples=definition.sampler.samples,
                        observations=tuple(observations),
                        weights=checkpoint_weights,
                        requirement_ids=tuple(
                            requirement.requirement_id for requirement in definition.requirements
                        ),
                        convergence_history=tuple(convergence_history),
                    )
                )
                if decision.stop:
                    break

            observed_weights = {
                observation.sample_id: weights[observation.sample_id]
                for observation in observations
            }
            if completed_statistics is None:
                statistics = summarize_campaign(
                    requested_samples=definition.sampler.samples,
                    observations=tuple(observations),
                    weights=observed_weights,
                    requirement_ids=tuple(
                        requirement.requirement_id for requirement in definition.requirements
                    ),
                    convergence_history=tuple(convergence_history),
                )
                store.write_statistics(statistics)
                store.write_summary(_format_summary(definition, statistics, batch.warnings))
                store.set_state(CampaignState.COMPLETED)
        except BaseException:
            store.set_state(CampaignState.INTERRUPTED)
            raise

    return CampaignResult(
        campaign_id=definition.campaign_id,
        definition_digest=definition_digest,
        state=CampaignState.COMPLETED,
        statistics=statistics,
        case_index_path=CASES_FILE,
        sample_index_path=SAMPLES_FILE,
        warnings=batch.warnings,
        claim_boundary=definition.evaluator.claim_boundary,
        metadata={"sampler": batch.metadata, "workers": workers},
    )


def _evaluate_sample(
    definition: CampaignDefinition,
    runtime: CampaignRuntime,
    sample: ParameterRealization,
) -> CaseObservation:
    evaluator = AuthoritativeCallableEvaluator[AstroModel, AstroModel](
        evaluator_id=definition.evaluator.evaluator_id,
        evaluate_callable=runtime.evaluate,
        serialize_callable=lambda _result: (),
    )
    scenario, scenario_evidence = _resolve_scenario(definition, runtime, sample)
    outcome, result = evaluate_authoritatively(
        evaluator,
        scenario,
        scenario_evidence,
        serialize_result=False,
    )
    started = perf_counter()
    try:
        observation = _observe(definition, runtime.metrics, outcome, result)
    except Exception as exc:
        observation = _case_failure_from_exception(
            definition, outcome, exc, phase="metric_extraction"
        )
        return _add_timing_phase(
            observation,
            field="metric_extraction_s",
            elapsed_s=max(0.0, perf_counter() - started),
        )
    observation = _add_timing_phase(
        observation,
        field="metric_extraction_s",
        elapsed_s=max(0.0, perf_counter() - started),
    )
    if (
        observation.outcome_status is not OutcomeStatus.SUCCESS
        or result is None
        or not _should_retain_result(definition, sample, observation)
    ):
        return observation
    started = perf_counter()
    try:
        artifact_refs = runtime.serialize(result)
    except Exception as exc:
        observation = _add_timing_phase(
            observation,
            field="serialization_s",
            elapsed_s=max(0.0, perf_counter() - started),
        )
        return _case_failure_from_observation(
            definition, observation, exc, phase="serialization"
        )
    elapsed = max(0.0, perf_counter() - started)
    observation = _add_timing_phase(observation, field="serialization_s", elapsed_s=elapsed)
    return observation.model_copy(update={"artifact_refs": artifact_refs})


def _add_timing_phase(
    observation: CaseObservation,
    *,
    field: str,
    elapsed_s: float,
) -> CaseObservation:
    timing = observation.evaluation_timing
    if timing is None:
        return observation
    current_value = getattr(timing, field)
    current = 0.0 if current_value is None else float(current_value)
    timing = timing.model_copy(
        update={field: current + elapsed_s, "total_s": float(timing.total_s) + elapsed_s}
    )
    return observation.model_copy(update={"evaluation_timing": timing})


def _case_failure_from_exception(
    definition: CampaignDefinition,
    outcome: object,
    exc: Exception,
    *,
    phase: str,
) -> CaseObservation:
    from astro_uq.models import EvaluationOutcome

    evaluated = EvaluationOutcome.model_validate(outcome)
    metadata = {
        **evaluated.metadata,
        "phase": phase,
        "error_type": type(exc).__name__,
        "error_message": str(exc) or repr(exc),
        "traceback_summary": list(traceback.format_exception(exc)),
    }
    return CaseObservation(
        sample_id=evaluated.sample_id,
        outcome_status=OutcomeStatus.EXECUTION_FAILURE,
        evaluator_id=evaluated.evaluator_id,
        evaluation_timing=evaluated.timing,
        claim_boundary=definition.evaluator.claim_boundary,
        metadata=metadata,
    )


def _case_failure_from_observation(
    definition: CampaignDefinition,
    observation: CaseObservation,
    exc: Exception,
    *,
    phase: str,
) -> CaseObservation:
    metadata = {
        **observation.metadata,
        "phase": phase,
        "error_type": type(exc).__name__,
        "error_message": str(exc) or repr(exc),
        "traceback_summary": list(traceback.format_exception(exc)),
    }
    return CaseObservation(
        sample_id=observation.sample_id,
        outcome_status=OutcomeStatus.EXECUTION_FAILURE,
        evaluator_id=observation.evaluator_id,
        evaluation_timing=observation.evaluation_timing,
        claim_boundary=definition.evaluator.claim_boundary,
        metadata=metadata,
    )


def _initialize_worker(
    definition: CampaignDefinition,
    runtime_factory: Callable[[], CampaignRuntime],
) -> None:
    global _WORKER_DEFINITION, _WORKER_RUNTIME
    _WORKER_DEFINITION = definition
    _WORKER_RUNTIME = runtime_factory()


def _evaluate_in_worker(sample: ParameterRealization) -> CaseObservation:
    if _WORKER_DEFINITION is None or _WORKER_RUNTIME is None:
        raise RuntimeError("campaign worker was not initialized")
    return _evaluate_sample(_WORKER_DEFINITION, _WORKER_RUNTIME, sample)


def _parallel_observations(
    definition: CampaignDefinition,
    samples: tuple[ParameterRealization, ...],
    *,
    workers: int,
    runtime_factory: Callable[[], CampaignRuntime],
) -> tuple[CaseObservation, ...]:
    """Evaluate with bounded in-flight work and deterministic sample ordering."""
    if not samples:
        return ()
    observations: list[CaseObservation] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(definition, runtime_factory),
    ) as executor:
        iterator = iter(samples)
        pending = []
        for _ in range(min(workers, len(samples))):
            pending.append(executor.submit(_evaluate_in_worker, next(iterator)))
        while pending:
            future = pending.pop(0)
            observations.append(future.result())
            try:
                sample = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(_evaluate_in_worker, sample))
    return tuple(observations)


def _sample_limit(definition: CampaignDefinition) -> int:
    stopping = definition.stopping
    if isinstance(stopping, FixedCountStopping):
        return definition.sampler.samples
    return min(definition.sampler.samples, stopping.maximum_samples)


def _stopping_batch_size(definition: CampaignDefinition) -> int:
    stopping = definition.stopping
    if isinstance(stopping, FixedCountStopping):
        return definition.sampler.samples
    return stopping.batch_size


def _stopping_decision(
    definition: CampaignDefinition,
    observations: tuple[CaseObservation, ...],
    samples: tuple[ParameterRealization, ...],
    history: list[dict[str, object]],
) -> tuple[StoppingDecision, dict[str, object]]:
    completed = len(observations)
    stopping = definition.stopping
    if isinstance(stopping, FixedCountStopping):
        decision = fixed_count_decision(completed, definition.sampler.samples)
        return decision, {"completed_samples": completed, "reason": decision.reason}

    weights_by_id = {sample.sample_id: float(sample.weight) for sample in samples}
    weights = np.asarray(
        [weights_by_id[observation.sample_id] for observation in observations],
        dtype=np.float64,
    )
    ess = effective_sample_size(weights)
    if isinstance(stopping, ConfidenceIntervalStopping):
        successes = sum(
            requirement.passed is True
            for observation in observations
            for requirement in observation.requirements
            if requirement.requirement_id == stopping.requirement_id
        )
        decision = confidence_interval_decision(
            successes=successes,
            completed=completed,
            effective_sample_size=ess,
            rule=ConfidenceIntervalRule(
                target_half_width=float(stopping.target_half_width),
                minimum_samples=stopping.minimum_samples,
                maximum_samples=stopping.maximum_samples,
                confidence=float(stopping.confidence),
                minimum_effective_sample_size=(
                    None
                    if stopping.minimum_effective_sample_size is None
                    else float(stopping.minimum_effective_sample_size)
                ),
                batch_size=stopping.batch_size,
            ),
        )
        return decision, {
            "completed_samples": completed,
            "effective_sample_size": ess,
            "reason": decision.reason,
            "requirement_id": stopping.requirement_id,
            "successes": successes,
        }

    statistics = summarize_campaign(
        requested_samples=definition.sampler.samples,
        observations=observations,
        weights={
            observation.sample_id: weights_by_id[observation.sample_id]
            for observation in observations
        },
    )
    metric = next(
        (item for item in statistics.metrics if item.metric_id == stopping.metric_id),
        None,
    )
    means = tuple(
        float(cast(float, record["metric_mean"]))
        for record in history
        if record.get("metric_id") == stopping.metric_id and record.get("metric_mean") is not None
    )
    if metric is not None and metric.mean is not None:
        means = (*means, float(metric.mean))
    decision = metric_stability_decision(
        means,
        completed=completed,
        effective_sample_size=ess,
        rule=MetricStabilityRule(
            absolute_tolerance=float(stopping.absolute_tolerance),
            minimum_samples=stopping.minimum_samples,
            maximum_samples=stopping.maximum_samples,
            window=stopping.window,
            minimum_effective_sample_size=(
                None
                if stopping.minimum_effective_sample_size is None
                else float(stopping.minimum_effective_sample_size)
            ),
            batch_size=stopping.batch_size,
        ),
    )
    return decision, {
        "completed_samples": completed,
        "effective_sample_size": ess,
        "metric_id": stopping.metric_id,
        "metric_mean": None if metric is None else metric.mean,
        "reason": decision.reason,
    }


def _should_retain_result(
    definition: CampaignDefinition,
    sample: ParameterRealization,
    observation: CaseObservation,
) -> bool:
    retention = definition.retention
    if retention.policy is RetentionPolicy.ALL:
        return True
    if retention.policy in {RetentionPolicy.NONE, RetentionPolicy.FAILURES}:
        return False
    if retention.policy is RetentionPolicy.AUDIT_SAMPLE:
        if retention.audit_fraction <= 0.0:
            return False
        value = int(canonical_hash(sample.sample_id)[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        return value < retention.audit_fraction

    return any(
        requirement.margin is not None
        and abs(float(requirement.margin)) <= retention.boundary_tolerance
        for requirement in observation.requirements
    )


def _physical_realization(
    raw: ParameterRealization,
    definition: CampaignDefinition,
) -> ParameterRealization:
    normalized = dict(raw.normalized_values)
    for correlation in definition.uncertainty.correlations:
        point = np.asarray(
            [[normalized[parameter_id] for parameter_id in correlation.parameter_ids]],
            dtype=np.float64,
        )
        transformed = gaussian_copula_transform(
            point,
            correlation,
            correlation.parameter_ids,
        )[0]
        for parameter_id, value in zip(correlation.parameter_ids, transformed, strict=True):
            normalized[parameter_id] = float(value)
    return build_realization(
        definition.uncertainty.parameters,
        normalized,
        sample_id=raw.sample_id,
        sample_index=raw.sample_index,
        weight=float(raw.weight),
    ).model_copy(update={"model_variants": raw.model_variants})


def _resolve_scenario(
    definition: CampaignDefinition,
    runtime: CampaignRuntime,
    sample: ParameterRealization,
) -> tuple[AstroModel, ScenarioRealization]:
    try:
        return runtime.parameters.apply(
            workflow=definition.workflow.kind,
            scenario=runtime.scenario,
            uncertainty=definition.uncertainty,
            realization=sample,
        )
    except ParameterBindingError as exc:
        digest = model_digest(runtime.scenario)
        return runtime.scenario, ScenarioRealization(
            sample_id=sample.sample_id,
            base_scenario_digest=digest,
            resolved_scenario_digest=digest,
            valid=False,
            validation_errors=(str(exc),),
        )


def _observe(
    definition: CampaignDefinition,
    metrics: MetricRegistry,
    outcome: object,
    result: AstroModel | None,
) -> CaseObservation:
    from astro_uq.models import EvaluationOutcome

    evaluated = EvaluationOutcome.model_validate(outcome)
    values = {}
    requirements: tuple[RequirementOutcome, ...] = ()
    metadata = dict(evaluated.metadata)
    if evaluated.status is OutcomeStatus.SUCCESS and result is not None:
        values = metrics.extract(
            workflow=definition.workflow.kind,
            result=result,
            specifications=definition.metrics,
        )
        requirements = evaluate_requirements(values, definition.requirements)
        workflow = getattr(result, "workflow", None)
        if isinstance(workflow, str) and workflow:
            metadata["source_workflow"] = workflow
        warnings = getattr(result, "warnings", None)
        if isinstance(warnings, list | tuple) and all(
            isinstance(warning, str) for warning in warnings
        ):
            metadata["source_warnings"] = list(warnings)
    return CaseObservation(
        sample_id=evaluated.sample_id,
        outcome_status=evaluated.status,
        metric_values=values,
        requirements=requirements,
        evaluator_id=evaluated.evaluator_id,
        evaluation_timing=evaluated.timing,
        artifact_refs=evaluated.artifact_refs,
        claim_boundary=definition.evaluator.claim_boundary,
        metadata=metadata,
    )


def _format_summary(
    definition: CampaignDefinition,
    statistics: object,
    warnings: tuple[str, ...],
) -> str:
    from astro_uq.models import CampaignStatistics

    summary = CampaignStatistics.model_validate(statistics)
    lines = [
        f"Campaign: {definition.campaign_id}",
        f"Workflow: {definition.workflow.kind}",
        f"Completed: {summary.completed_samples}/{summary.requested_samples}",
        f"Claim boundary: {definition.evaluator.claim_boundary}",
    ]
    lines.extend(
        f"Requirement {requirement_id}: {probability:.6f}"
        for requirement_id, probability in sorted(summary.requirement_probabilities.items())
    )
    lines.extend(f"Warning: {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"
