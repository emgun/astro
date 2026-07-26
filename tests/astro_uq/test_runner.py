from __future__ import annotations

from pathlib import Path

import pytest

import astro_uq.runner as runner_module
from astro_core.models import AstroModel
from astro_uq.io import CampaignArtifactStore, CampaignIOError
from astro_uq.metrics import MetricExtractor, MetricRegistry
from astro_uq.models import (
    CampaignDefinition,
    CaseObservation,
    ConfidenceIntervalStopping,
    DistributionKind,
    DistributionSpec,
    EvaluatorKind,
    EvaluatorSpec,
    MetricSpec,
    MetricStabilityStopping,
    MetricValueKind,
    ModelVariant,
    ParameterRealization,
    RequirementOperator,
    RequirementSpec,
    RetentionPolicy,
    RetentionSpec,
    SamplerKind,
    SamplerSpec,
    UncertainParameter,
    UncertaintyKind,
    UncertaintyModel,
    WorkflowSpec,
)
from astro_uq.parameters import ParameterBinding, ParameterRegistry, ParameterValue
from astro_uq.runner import CampaignRuntime, run_campaign


def _case_scientific_evidence(path: Path) -> list[dict[str, object]]:
    return [
        CaseObservation.model_validate_json(line).model_dump(
            mode="json", exclude={"evaluation_timing"}
        )
        for line in path.read_text().splitlines()
    ]


class FixtureScenario(AstroModel):
    value: float


class FixtureResult(AstroModel):
    doubled: float


def _definition(samples: int = 4) -> CampaignDefinition:
    return CampaignDefinition(
        campaign_id="runner-fixture",
        workflow=WorkflowSpec(kind="fixture", scenario="fixture.yaml"),
        uncertainty=UncertaintyModel(
            parameters=(
                UncertainParameter(
                    parameter_id="value",
                    target="fixture.value",
                    unit="1",
                    uncertainty_kind=UncertaintyKind.ALEATORY,
                    distribution=DistributionSpec(
                        kind=DistributionKind.UNIFORM,
                        low=0.0,
                        high=1.0,
                    ),
                ),
            )
        ),
        sampler=SamplerSpec(kind=SamplerKind.PSEUDORANDOM, samples=samples, seed=4),
        evaluator=EvaluatorSpec(
            evaluator_id="fixture",
            kind=EvaluatorKind.AUTHORITATIVE,
            workflow="fixture",
            implementation_version="1",
            claim_boundary="test_fixture",
        ),
        metrics=(
            MetricSpec(
                metric_id="doubled",
                extractor="fixture.doubled",
                value_kind=MetricValueKind.NUMERIC,
                unit="1",
            ),
        ),
        requirements=(
            RequirementSpec(
                requirement_id="positive",
                metric_id="doubled",
                operator=RequirementOperator.GE,
                value=0.0,
            ),
        ),
    )


def _runtime() -> CampaignRuntime:
    parameters = ParameterRegistry()

    def update(model: AstroModel, value: ParameterValue) -> AstroModel:
        return FixtureScenario(value=float(value))

    parameters.register(
        ParameterBinding(
            target="fixture.value",
            workflow="fixture",
            unit="1",
            value_type=float,
            getter=lambda model: FixtureScenario.model_validate(model).value,
            updater=update,
        )
    )
    metrics = MetricRegistry()
    metrics.register(
        MetricExtractor(
            extractor_id="fixture.doubled",
            workflow="fixture",
            value_kind=MetricValueKind.NUMERIC,
            unit="1",
            extract=lambda result: FixtureResult.model_validate(result).doubled,
        )
    )
    return CampaignRuntime(
        scenario=FixtureScenario(value=0.0),
        parameters=parameters,
        metrics=metrics,
        evaluate=lambda scenario: FixtureResult(
            doubled=2.0 * FixtureScenario.model_validate(scenario).value
        ),
    )


def _runtime_with_serializer(calls: list[float]) -> CampaignRuntime:
    runtime = _runtime()

    def serialize(result: AstroModel) -> tuple[str, ...]:
        doubled = FixtureResult.model_validate(result).doubled
        calls.append(doubled)
        return (f"results/{doubled:.6f}.json",)

    return CampaignRuntime(
        scenario=runtime.scenario,
        parameters=runtime.parameters,
        metrics=runtime.metrics,
        evaluate=runtime.evaluate,
        serialize=serialize,
    )


def _runtime_with_failing_serializer() -> CampaignRuntime:
    runtime = _runtime()

    def serialize(_result: AstroModel) -> tuple[str, ...]:
        raise OSError("artifact write failed")

    return CampaignRuntime(
        scenario=runtime.scenario,
        parameters=runtime.parameters,
        metrics=runtime.metrics,
        evaluate=runtime.evaluate,
        serialize=serialize,
    )


def _interrupting_runtime(*, after: int) -> CampaignRuntime:
    runtime = _runtime()
    calls = 0

    def evaluate(scenario: AstroModel) -> AstroModel:
        nonlocal calls
        calls += 1
        if calls > after:
            raise KeyboardInterrupt
        return runtime.evaluate(scenario)

    return CampaignRuntime(
        scenario=runtime.scenario,
        parameters=runtime.parameters,
        metrics=runtime.metrics,
        evaluate=evaluate,
    )


def _runtime_with_bad_metric() -> CampaignRuntime:
    runtime = _runtime()
    metrics = MetricRegistry()
    metrics.register(
        MetricExtractor(
            extractor_id="fixture.doubled",
            workflow="fixture",
            value_kind=MetricValueKind.NUMERIC,
            unit="1",
            extract=lambda _result: (_ for _ in ()).throw(ValueError("bad metric")),
        )
    )
    return CampaignRuntime(
        scenario=runtime.scenario,
        parameters=runtime.parameters,
        metrics=metrics,
        evaluate=runtime.evaluate,
    )


def test_runner_writes_complete_reproducible_campaign(tmp_path: Path) -> None:
    first = run_campaign(
        _definition(),
        _runtime(),
        output_dir=tmp_path / "first",
        software_compatibility={"astro": "test"},
    )
    second = run_campaign(
        _definition(),
        _runtime(),
        output_dir=tmp_path / "second",
        software_compatibility={"astro": "test"},
    )

    assert first.statistics == second.statistics
    assert first.statistics.completed_samples == 4
    assert first.statistics.requirement_probabilities == {"positive": 1.0}
    assert (tmp_path / "first" / "summary.txt").exists()


def test_parallel_runner_matches_serial_evidence(tmp_path: Path) -> None:
    serial = run_campaign(
        _definition(samples=7),
        _runtime(),
        output_dir=tmp_path / "serial",
        software_compatibility={"astro": "test"},
    )
    parallel = run_campaign(
        _definition(samples=7),
        _runtime(),
        output_dir=tmp_path / "parallel",
        software_compatibility={"astro": "test"},
        workers=2,
        runtime_factory=_runtime,
    )

    assert parallel.statistics == serial.statistics
    assert parallel.metadata["workers"] == 2
    assert (tmp_path / "parallel" / "samples.jsonl").read_bytes() == (
        tmp_path / "serial" / "samples.jsonl"
    ).read_bytes()
    assert _case_scientific_evidence(
        tmp_path / "parallel" / "cases.jsonl"
    ) == _case_scientific_evidence(tmp_path / "serial" / "cases.jsonl")


def test_fixed_campaign_uses_one_parallel_pool_and_checkpoints_each_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def evaluate_without_processes(
        definition: CampaignDefinition,
        samples: tuple[object, ...],
        *,
        workers: int,
        runtime_factory: object,
        on_observation: object = None,
    ) -> tuple[CaseObservation, ...]:
        del runtime_factory
        typed_samples = tuple(ParameterRealization.model_validate(sample) for sample in samples)
        observations = tuple(
            runner_module._evaluate_sample(definition, _runtime(), sample)
            for sample in typed_samples
        )
        calls.append((len(samples), workers))
        assert callable(on_observation)
        for observation in observations:
            on_observation(observation)
        return observations

    monkeypatch.setattr(runner_module, "_parallel_observations", evaluate_without_processes)

    result = run_campaign(
        _definition(),
        _runtime(),
        output_dir=tmp_path / "parallel",
        software_compatibility={"astro": "test"},
        workers=4,
        runtime_factory=_runtime,
    )

    assert calls == [(4, 4)]
    assert result.statistics.completed_samples == 4
    assert len(result.statistics.convergence_history) == 4
    assert [record["completed_samples"] for record in result.statistics.convergence_history] == [
        1,
        2,
        3,
        4,
    ]


def test_parallel_runner_requires_runtime_factory_before_writing_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "campaign"
    with pytest.raises(CampaignIOError, match="per-worker runtime factory"):
        run_campaign(
            _definition(),
            _runtime(),
            output_dir=output,
            software_compatibility={"astro": "test"},
            workers=2,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("policy", "expected_serialized"),
    [(RetentionPolicy.ALL, 4), (RetentionPolicy.NONE, 0), (RetentionPolicy.FAILURES, 0)],
)
def test_runner_applies_success_artifact_retention_policy(
    tmp_path: Path,
    policy: RetentionPolicy,
    expected_serialized: int,
) -> None:
    calls: list[float] = []
    definition = _definition().model_copy(update={"retention": RetentionSpec(policy=policy)})
    output = tmp_path / policy.value

    run_campaign(
        definition,
        _runtime_with_serializer(calls),
        output_dir=output,
        software_compatibility={"astro": "test"},
    )

    cases = [
        CaseObservation.model_validate_json(line)
        for line in (output / "cases.jsonl").read_text().splitlines()
    ]
    assert len(calls) == expected_serialized
    assert sum(bool(case.artifact_refs) for case in cases) == expected_serialized
    assert all(case.evaluation_timing is not None for case in cases)
    assert all(
        case.evaluation_timing.metric_extraction_s is not None
        for case in cases
        if case.evaluation_timing is not None
    )


def test_serialization_failure_preserves_prior_phase_timing(tmp_path: Path) -> None:
    definition = _definition(samples=1).model_copy(
        update={"retention": RetentionSpec(policy=RetentionPolicy.ALL)}
    )
    output = tmp_path / "serialization-failure"

    result = run_campaign(
        definition,
        _runtime_with_failing_serializer(),
        output_dir=output,
        software_compatibility={"astro": "test"},
    )

    case = CaseObservation.model_validate_json((output / "cases.jsonl").read_text())
    assert result.statistics.outcome_counts == {"execution_failure": 1}
    assert case.metadata["phase"] == "serialization"
    assert case.evaluation_timing is not None
    assert case.evaluation_timing.metric_extraction_s is not None
    assert case.evaluation_timing.serialization_s >= 0.0


@pytest.mark.parametrize(
    "retention",
    [
        RetentionSpec(
            policy=RetentionPolicy.FAILURES_AND_BOUNDARIES,
            boundary_tolerance=2.0,
        ),
        RetentionSpec(policy=RetentionPolicy.AUDIT_SAMPLE, audit_fraction=1.0),
    ],
)
def test_runner_retains_boundary_and_deterministic_audit_results(
    tmp_path: Path, retention: RetentionSpec
) -> None:
    calls: list[float] = []
    definition = _definition().model_copy(update={"retention": retention})

    run_campaign(
        definition,
        _runtime_with_serializer(calls),
        output_dir=tmp_path / retention.policy.value,
        software_compatibility={"astro": "test"},
    )

    assert len(calls) == 4


@pytest.mark.parametrize(
    "policy",
    [
        RetentionPolicy.ALL,
        RetentionPolicy.NONE,
        RetentionPolicy.FAILURES,
        RetentionPolicy.FAILURES_AND_BOUNDARIES,
        RetentionPolicy.AUDIT_SAMPLE,
    ],
)
def test_metric_failure_is_a_case_outcome_for_every_retention_policy(
    tmp_path: Path, policy: RetentionPolicy
) -> None:
    definition = _definition().model_copy(update={"retention": RetentionSpec(policy=policy)})

    result = run_campaign(
        definition,
        _runtime_with_bad_metric(),
        output_dir=tmp_path / policy.value,
        software_compatibility={"astro": "test"},
    )

    assert result.statistics.outcome_counts == {"execution_failure": 4}
    cases = [
        CaseObservation.model_validate_json(line)
        for line in (tmp_path / policy.value / "cases.jsonl").read_text().splitlines()
    ]
    assert len(cases) == 4
    assert {case.metadata["phase"] for case in cases} == {"metric_extraction"}
    assert all(case.evaluation_timing is not None for case in cases)
    assert all(
        case.evaluation_timing.metric_extraction_s is not None
        for case in cases
        if case.evaluation_timing is not None
    )


def test_runner_stops_at_ci_batch_and_records_convergence(tmp_path: Path) -> None:
    definition = _definition(samples=8).model_copy(
        update={
            "stopping": ConfidenceIntervalStopping(
                requirement_id="positive",
                target_half_width=0.5,
                minimum_samples=2,
                maximum_samples=8,
                batch_size=2,
            )
        }
    )

    result = run_campaign(
        definition,
        _runtime(),
        output_dir=tmp_path / "ci",
        software_compatibility={"astro": "test"},
        workers=2,
        runtime_factory=_runtime,
    )

    assert result.statistics.completed_samples == 2
    assert result.statistics.convergence_history[-1]["reason"] == ("confidence_interval_converged")
    assert len((tmp_path / "ci" / "cases.jsonl").read_text().splitlines()) == 2


def test_ci_stopping_rejects_unequal_weights_before_writing_evidence(
    tmp_path: Path,
) -> None:
    baseline = _definition(samples=2)
    definition = baseline.model_copy(
        update={
            "uncertainty": UncertaintyModel(
                model_variants=(
                    ModelVariant(
                        variant_id="one",
                        target="fixture.variant",
                        value="one",
                        weight=1.0,
                    ),
                    ModelVariant(
                        variant_id="two",
                        target="fixture.variant",
                        value="two",
                        weight=3.0,
                    ),
                )
            ),
            "sampler": SamplerSpec(kind=SamplerKind.ENSEMBLE, samples=2),
            "stopping": ConfidenceIntervalStopping(
                requirement_id="positive",
                target_half_width=0.5,
                minimum_samples=2,
                maximum_samples=2,
            ),
        }
    )
    output = tmp_path / "weighted-ci"

    with pytest.raises(CampaignIOError, match="equal sample weights"):
        run_campaign(
            definition,
            _runtime(),
            output_dir=output,
            software_compatibility={"astro": "test"},
        )
    assert not output.exists()


def test_runner_stops_when_metric_batch_means_stabilize(tmp_path: Path) -> None:
    definition = _definition(samples=8).model_copy(
        update={
            "stopping": MetricStabilityStopping(
                metric_id="doubled",
                absolute_tolerance=2.0,
                minimum_samples=4,
                maximum_samples=8,
                window=2,
                batch_size=2,
            )
        }
    )

    result = run_campaign(
        definition,
        _runtime(),
        output_dir=tmp_path / "stability",
        software_compatibility={"astro": "test"},
    )

    assert result.statistics.completed_samples == 4
    assert result.statistics.convergence_history[-1]["reason"] == ("metric_stability_converged")


def test_interrupted_adaptive_campaign_resumes_to_uninterrupted_evidence(
    tmp_path: Path,
) -> None:
    definition = _definition(samples=4).model_copy(
        update={
            "stopping": ConfidenceIntervalStopping(
                requirement_id="positive",
                target_half_width=0.01,
                minimum_samples=2,
                maximum_samples=4,
                batch_size=2,
            )
        }
    )
    interrupted_output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_campaign(
            definition,
            _interrupting_runtime(after=2),
            output_dir=interrupted_output,
            software_compatibility={"astro": "test"},
        )
    assert len((interrupted_output / "cases.jsonl").read_text().splitlines()) == 2

    resumed = run_campaign(
        definition,
        _runtime(),
        output_dir=interrupted_output,
        software_compatibility={"astro": "test"},
        resume=True,
    )
    uninterrupted = run_campaign(
        definition,
        _runtime(),
        output_dir=tmp_path / "uninterrupted",
        software_compatibility={"astro": "test"},
    )

    assert resumed.statistics == uninterrupted.statistics
    assert _case_scientific_evidence(
        interrupted_output / "cases.jsonl"
    ) == _case_scientific_evidence(tmp_path / "uninterrupted" / "cases.jsonl")


def test_runner_resume_does_not_duplicate_cases(tmp_path: Path) -> None:
    output = tmp_path / "campaign"
    first = run_campaign(
        _definition(),
        _runtime(),
        output_dir=output,
        software_compatibility={"astro": "test"},
    )
    resumed = run_campaign(
        _definition(),
        _runtime(),
        output_dir=output,
        software_compatibility={"astro": "test"},
        resume=True,
    )

    assert resumed.statistics == first.statistics
    assert len((output / "cases.jsonl").read_text().splitlines()) == 4


def test_resume_completes_an_interrupted_initial_sample_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "partial-initialization"
    original = CampaignArtifactStore.append_sample
    calls = 0

    def interrupt_after_prefix(store: CampaignArtifactStore, sample: object) -> None:
        nonlocal calls
        calls += 1
        if calls > 2:
            raise KeyboardInterrupt
        original(store, sample)

    monkeypatch.setattr(CampaignArtifactStore, "append_sample", interrupt_after_prefix)
    with pytest.raises(KeyboardInterrupt):
        run_campaign(
            _definition(),
            _runtime(),
            output_dir=output,
            software_compatibility={"astro": "test"},
        )
    monkeypatch.setattr(CampaignArtifactStore, "append_sample", original)

    resumed = run_campaign(
        _definition(),
        _runtime(),
        output_dir=output,
        software_compatibility={"astro": "test"},
        resume=True,
    )

    assert resumed.statistics.completed_samples == 4
    assert len((output / "samples.jsonl").read_text().splitlines()) == 4


def test_runner_fails_closed_for_unimplemented_surrogate_evaluator(tmp_path: Path) -> None:
    baseline = _definition()
    definition = baseline.model_copy(
        update={
            "evaluator": baseline.evaluator.model_copy(
                update={"kind": EvaluatorKind.SURROGATE, "model_artifact": "model.json"}
            )
        }
    )

    with pytest.raises(CampaignIOError, match="fails closed"):
        run_campaign(
            definition,
            _runtime(),
            output_dir=tmp_path / "campaign",
            software_compatibility={"astro": "test"},
        )
