from __future__ import annotations

from pathlib import Path

from astro_mission.io import load_mission_lifecycle_scenario
from astro_operator.lifecycle import LifecycleCandidateEvaluator
from astro_operator.models import CandidateProposal, DesignVariable


def _evaluator(tmp_path: Path) -> LifecycleCandidateEvaluator:
    return LifecycleCandidateEvaluator(
        base_scenario=load_mission_lifecycle_scenario(
            "examples/lifecycle/leo_round_trip.yaml"
        ),
        design_variables=(
            DesignVariable(
                variable_id="wet_mass",
                target="spacecraft_wet_mass_kg",
                lower_bound=470.0,
                upper_bound=530.0,
                unit="kg",
            ),
        ),
        output_root=tmp_path,
    )


def test_lifecycle_evaluator_emits_simulated_metrics_and_digest_bound_artifacts(
    tmp_path: Path,
) -> None:
    observation = _evaluator(tmp_path).evaluate(
        CandidateProposal(candidate_id="baseline", assignments={})
    )

    assert observation.evaluation_status == "evaluated"
    assert observation.passed
    assert any(
        metric.metric_id == "margin:deorbit:propellant_reserve"
        for metric in observation.metrics
    )
    assert [item.epistemic_kind.value for item in observation.evidence] == [
        "simulated",
        "simulated",
    ]
    assert all((tmp_path / item.path).is_file() for item in observation.evidence)


def test_lifecycle_failure_becomes_an_observation_the_reasoner_can_adapt_to(
    tmp_path: Path,
) -> None:
    observation = _evaluator(tmp_path).evaluate(
        CandidateProposal(candidate_id="lighter", assignments={"wet_mass": 480.0})
    )

    assert observation.evaluation_status == "evaluation_failed"
    assert not observation.passed
    assert observation.metrics == ()
    assert observation.evidence[-1].kind == "mission_lifecycle_evaluation_error"
    assert "propellant reserve" in observation.warnings[0]
