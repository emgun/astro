from __future__ import annotations

import pytest

from astro_assistant.models import AstroToolName
from astro_assistant.planner import mission_lifecycle_review_plan
from astro_assistant.policy import evaluate_plan
from astro_assistant.registry import build_command_spec
from astro_assistant.verification import verify_plan


def test_lifecycle_review_plan_is_fixed_and_approval_gated() -> None:
    plan = mission_lifecycle_review_plan(
        "Review lifecycle evidence",
        result_path="/tmp/astro/lifecycle.json",
        scenario_path="examples/lifecycle/leo_round_trip.yaml",
        output="/tmp/astro/lifecycle-review.json",
    )

    assert verify_plan(plan).passed
    assert not evaluate_plan(plan, dry_run=False, approved=False).allowed
    assert evaluate_plan(plan, dry_run=False, approved=True).allowed
    assert [step.tool for step in plan.steps] == [
        AstroToolName.VERIFY_MISSION_LIFECYCLE_RESULT,
        AstroToolName.REVIEW_MISSION_LIFECYCLE,
    ]
    assert build_command_spec(plan.steps[0]).argv == [
        "astro",
        "verify-mission-lifecycle-result",
        "/tmp/astro/lifecycle.json",
        "examples/lifecycle/leo_round_trip.yaml",
    ]


@pytest.mark.parametrize(
    ("output", "summary_output"),
    [
        ("/tmp/astro/lifecycle.json", None),
        ("/tmp/astro/review.json", "examples/lifecycle/leo_round_trip.yaml"),
        ("/tmp/astro/review.json", "/tmp/astro/./review.json"),
    ],
)
def test_lifecycle_review_plan_rejects_path_collisions(
    output: str, summary_output: str | None
) -> None:
    plan = mission_lifecycle_review_plan(
        "Review lifecycle evidence",
        result_path="/tmp/astro/lifecycle.json",
        scenario_path="examples/lifecycle/leo_round_trip.yaml",
        output=output,
        summary_output=summary_output,
    )

    verification = verify_plan(plan)
    assert not verification.passed
    assert verification.diagnostics[0].code == "path_collision"


def test_lifecycle_review_plan_rejects_source_discontinuity() -> None:
    plan = mission_lifecycle_review_plan(
        "Review lifecycle evidence",
        result_path="/tmp/astro/lifecycle.json",
        scenario_path="examples/lifecycle/leo_round_trip.yaml",
        output="/tmp/astro/review.json",
    )
    plan.steps[1].inputs["scenario_path"] = "examples/lifecycle/other.yaml"

    verification = verify_plan(plan)
    assert not verification.passed
    assert verification.diagnostics[0].code == "source_discontinuity"


def test_lifecycle_tools_reject_plan_id_substitution() -> None:
    plan = mission_lifecycle_review_plan(
        "Review lifecycle evidence",
        result_path="/tmp/astro/lifecycle.json",
        scenario_path="examples/lifecycle/leo_round_trip.yaml",
        output="/tmp/astro/review.json",
    )
    plan.plan_id = "custom-lifecycle"
    plan.steps = plan.steps[1:]

    verification = verify_plan(plan)
    assert not verification.passed
    assert verification.diagnostics[0].code == "protected_plan_id"
