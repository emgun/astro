from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from astro_surrogates.models import (
    EvaluatorIdentity,
    FieldSchema,
    PromotionStatus,
    SimulationDatasetSpec,
    SurrogatePromotionDecision,
)

SHA = "a" * 64


def evaluator(name: str) -> EvaluatorIdentity:
    return EvaluatorIdentity(
        evaluator_id=name,
        implementation=f"astro.{name}",
        version="1",
        configuration_digest=SHA,
        claim_boundary="research evidence only",
    )


def test_dataset_spec_is_strict_immutable_and_records_provenance() -> None:
    spec = SimulationDatasetSpec(
        dataset_id="orbit-residual-v1",
        scenario_family="leo",
        teacher=evaluator("teacher"),
        baseline=evaluator("baseline"),
        parameter_domain={"altitude_km": (400.0, 800.0)},
        features=(FieldSchema(name="x", unit="km", frame="GCRF", description="position"),),
        targets=(FieldSchema(name="dx", unit="km", frame="GCRF", description="residual"),),
        time_coordinate=FieldSchema(
            name="elapsed_s", unit="s", frame="mission_elapsed", description="elapsed time"
        ),
        split_policy="complete episodes by scenario and orbit",
        seed=7,
        retention_policy="all",
        software_versions={"astro": "0.1.0"},
        environment={"python": "3.12", "platform": "test"},
    )
    assert spec.teacher.evaluator_id == "teacher"
    with pytest.raises(ValidationError):
        SimulationDatasetSpec.model_validate({**spec.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError):
        spec.seed = 8  # type: ignore[misc]


def test_dataset_bounds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="low < high"):
        SimulationDatasetSpec(
            dataset_id="bad",
            scenario_family="leo",
            teacher=evaluator("teacher"),
            baseline=evaluator("baseline"),
            parameter_domain={"altitude_km": (800.0, 400.0)},
            features=(FieldSchema(name="x", unit="km", frame="GCRF", description="x"),),
            targets=(FieldSchema(name="dx", unit="km", frame="GCRF", description="dx"),),
            time_coordinate=FieldSchema(name="t", unit="s", frame="elapsed", description="t"),
            split_policy="episode",
            seed=0,
            retention_policy="all",
            software_versions={"astro": "1"},
            environment={"python": "3.12"},
        )


def test_promotion_states_exclude_operational_and_closed_gate_is_rejected() -> None:
    decision = SurrogatePromotionDecision(
        decision_id="benchmark-gate-closed",
        model_artifact_digest=SHA,
        validation_report_digest=SHA,
        status=PromotionStatus.REJECTED,
        decided_at=datetime.now(UTC),
        decided_by="independent-review",
        reasons=("uncertainty campaign benchmark gate is closed",),
        claim_boundary="unpromoted research artifact",
    )
    assert decision.status is PromotionStatus.REJECTED
    with pytest.raises(ValidationError):
        SurrogatePromotionDecision.model_validate(
            {**decision.model_dump(), "status": "operational"}
        )
