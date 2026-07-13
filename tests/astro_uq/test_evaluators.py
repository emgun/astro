from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from astro_core.errors import NumericalConvergenceError
from astro_core.models import AstroModel
from astro_uq.evaluators import (
    AuthoritativeCallableEvaluator,
    Evaluator,
    InvalidRealizationError,
    PolicyRejectionError,
    evaluate_authoritatively,
)
from astro_uq.models import OutcomeStatus, ScenarioRealization

DIGEST = "12" * 32


class FixtureScenario(AstroModel):
    value: float


class FixtureResult(AstroModel):
    doubled: float


def _realization(*, valid: bool = True) -> ScenarioRealization:
    return ScenarioRealization(
        sample_id="sample-7",
        base_scenario_digest=DIGEST,
        resolved_scenario_digest=DIGEST,
        valid=valid,
        validation_errors=() if valid else ("mass must be positive",),
    )


def _clock(values: tuple[float, ...]) -> Callable[[], float]:
    iterator: Iterator[float] = iter(values)
    return lambda: next(iterator)


def test_callable_evaluator_records_each_timing_and_returns_result() -> None:
    calls: list[str] = []
    evaluator: Evaluator[FixtureScenario, FixtureResult] = AuthoritativeCallableEvaluator[
        FixtureScenario, FixtureResult
    ](
        evaluator_id="fixture",
        setup_callable=lambda: calls.append("setup"),
        evaluate_callable=lambda scenario: FixtureResult(doubled=scenario.value * 2.0),
        serialize_callable=lambda result: (f"cases/{result.doubled:.0f}.json",),
    )

    outcome, result = evaluate_authoritatively(
        evaluator,
        FixtureScenario(value=3.0),
        _realization(),
        clock=_clock((0.0, 1.0, 3.0, 4.0, 9.0, 10.0, 12.0, 12.0)),
    )

    assert calls == ["setup"]
    assert result == FixtureResult(doubled=6.0)
    assert outcome.status is OutcomeStatus.SUCCESS
    assert outcome.artifact_refs == ("cases/6.json",)
    assert outcome.timing.model_dump() == {
        "setup_s": 2.0,
        "evaluation_s": 5.0,
        "metric_extraction_s": None,
        "serialization_s": 2.0,
        "total_s": 12.0,
    }


def test_invalid_realization_produces_one_outcome_without_running_evaluator() -> None:
    calls: list[str] = []
    evaluator: Evaluator[FixtureScenario, FixtureResult] = AuthoritativeCallableEvaluator[
        FixtureScenario, FixtureResult
    ](
        evaluator_id="fixture",
        setup_callable=lambda: calls.append("setup"),
        evaluate_callable=lambda scenario: FixtureResult(doubled=scenario.value),
        serialize_callable=lambda result: (),
    )

    outcome, result = evaluate_authoritatively(
        evaluator,
        FixtureScenario(value=-1.0),
        _realization(valid=False),
        clock=_clock((0.0, 0.25)),
    )

    assert calls == []
    assert result is None
    assert outcome.status is OutcomeStatus.INVALID_REALIZATION
    assert outcome.error_message == "mass must be positive"
    assert outcome.metadata["phase"] == "realization"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (InvalidRealizationError("bad binding"), OutcomeStatus.INVALID_REALIZATION),
        (PolicyRejectionError("backend denied"), OutcomeStatus.POLICY_REJECTION),
        (NumericalConvergenceError("did not converge"), OutcomeStatus.NUMERICAL_FAILURE),
        (RuntimeError("worker exited"), OutcomeStatus.EXECUTION_FAILURE),
    ],
)
def test_evaluation_failures_map_to_exactly_one_typed_outcome(
    error: Exception, expected: OutcomeStatus
) -> None:
    def fail(_: FixtureScenario) -> FixtureResult:
        raise error

    evaluator: Evaluator[FixtureScenario, FixtureResult] = AuthoritativeCallableEvaluator[
        FixtureScenario, FixtureResult
    ](
        evaluator_id="fixture",
        evaluate_callable=fail,
        serialize_callable=lambda result: (),
    )

    outcome, result = evaluate_authoritatively(
        evaluator,
        FixtureScenario(value=1.0),
        _realization(),
    )

    assert result is None
    assert outcome.status is expected
    assert outcome.sample_id == "sample-7"
    assert outcome.error_type == type(error).__name__
    assert outcome.metadata["sample_id"] == "sample-7"
    assert outcome.metadata["phase"] == "evaluation"
    assert type(error).__name__ in "".join(outcome.metadata["traceback_summary"])


def test_serialization_failure_is_an_execution_outcome_without_partial_result() -> None:
    def fail_serialization(_: FixtureResult) -> tuple[str, ...]:
        raise OSError("artifact directory unavailable")

    evaluator: Evaluator[FixtureScenario, FixtureResult] = AuthoritativeCallableEvaluator[
        FixtureScenario, FixtureResult
    ](
        evaluator_id="fixture",
        evaluate_callable=lambda scenario: FixtureResult(doubled=scenario.value * 2.0),
        serialize_callable=fail_serialization,
    )

    outcome, result = evaluate_authoritatively(
        evaluator,
        FixtureScenario(value=2.0),
        _realization(),
    )

    assert result is None
    assert outcome.status is OutcomeStatus.EXECUTION_FAILURE
    assert outcome.metadata["phase"] == "serialization"
    assert outcome.error_message == "artifact directory unavailable"


def test_workflow_phase_context_is_preserved_in_failure_metadata() -> None:
    class PhaseFailure(RuntimeError):
        lifecycle_phase = "launch"

    evaluator: Evaluator[FixtureScenario, FixtureResult] = AuthoritativeCallableEvaluator[
        FixtureScenario, FixtureResult
    ](
        evaluator_id="fixture",
        evaluate_callable=lambda _scenario: (_ for _ in ()).throw(PhaseFailure("insertion failed")),
        serialize_callable=lambda _result: (),
    )

    outcome, result = evaluate_authoritatively(
        evaluator,
        FixtureScenario(value=1.0),
        _realization(),
    )

    assert result is None
    assert outcome.metadata["workflow_phase"] == "launch"
