from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_assistant.executor import WorkflowExecutor
from astro_assistant.models import AstroToolName
from astro_assistant.planner import (
    assurance_review_comparison_plan,
    assurance_review_plan,
)
from astro_assistant.policy import evaluate_plan
from astro_assistant.registry import build_command_spec
from astro_assistant.verification import verify_plan

RESULT = "/tmp/astro/paired.json"
OUTPUT = "/tmp/astro/review.json"
SUMMARY = "/tmp/astro/review.txt"


def test_assurance_review_plan_is_fixed_verified_and_approval_gated() -> None:
    plan = assurance_review_plan(
        "Review paired assurance evidence",
        result_path=RESULT,
        output=OUTPUT,
        summary_output=SUMMARY,
    )

    assert [step.tool for step in plan.steps] == [
        AstroToolName.VERIFY_ASSURANCE_VALIDATION,
        AstroToolName.REVIEW_ASSURANCE_VALIDATION,
    ]
    assert verify_plan(plan).passed
    assert not evaluate_plan(plan, dry_run=False, approved=False).allowed
    assert evaluate_plan(plan, dry_run=False, approved=True).allowed
    assert build_command_spec(plan.steps[0]).argv == [
        "astro",
        "verify-assurance-validation",
        RESULT,
    ]
    assert build_command_spec(plan.steps[1]).argv == [
        "astro",
        "review-assurance-validation",
        RESULT,
        "--output",
        OUTPUT,
        "--summary-output",
        SUMMARY,
    ]


def test_assurance_review_plan_rejects_source_discontinuity() -> None:
    plan = assurance_review_plan(
        "Review paired assurance evidence", result_path=RESULT, output=OUTPUT
    )
    plan.steps[1].inputs["result_path"] = "/tmp/astro/substituted.json"

    verification = verify_plan(plan)
    assert not verification.passed
    assert verification.diagnostics[0].code == "source_discontinuity"


@pytest.mark.parametrize("path", ["", "--help", "bad\x00path"])
def test_assurance_review_commands_reject_unsafe_paths(path: str) -> None:
    plan = assurance_review_plan(
        "Review paired assurance evidence", result_path=RESULT, output=OUTPUT
    )
    plan.steps[1].inputs["output"] = path

    with pytest.raises(ValidationError):
        build_command_spec(plan.steps[1])


def test_assurance_review_dry_run_never_executes() -> None:
    plan = assurance_review_plan(
        "Review paired assurance evidence", result_path=RESULT, output=OUTPUT
    )
    trace = WorkflowExecutor(
        command_runner=lambda _argv, _cwd: (_ for _ in ()).throw(AssertionError("executed"))
    ).run(plan, dry_run=True, approved=False, cwd=None)

    assert trace.verification.passed
    assert trace.results == []


def test_assurance_comparison_plan_is_fixed_and_approval_gated() -> None:
    plan = assurance_review_comparison_plan(
        "Compare assurance reviews",
        baseline_path="/tmp/astro/baseline.json",
        candidate_path="/tmp/astro/candidate.json",
        output="/tmp/astro/comparison.json",
    )

    assert verify_plan(plan).passed
    assert not evaluate_plan(plan, dry_run=False, approved=False).allowed
    assert evaluate_plan(plan, dry_run=False, approved=True).allowed
    command = build_command_spec(plan.steps[0])
    assert command.argv == [
        "astro",
        "compare-assurance-reviews",
        "/tmp/astro/baseline.json",
        "/tmp/astro/candidate.json",
        "--output",
        "/tmp/astro/comparison.json",
    ]


def test_assurance_comparison_plan_rejects_same_source() -> None:
    plan = assurance_review_comparison_plan(
        "Compare assurance reviews",
        baseline_path="/tmp/astro/same.json",
        candidate_path="/tmp/astro/same.json",
        output="/tmp/astro/comparison.json",
    )

    verification = verify_plan(plan)
    assert not verification.passed
    assert verification.diagnostics[0].code == "path_collision"


@pytest.mark.parametrize(
    ("output", "summary_output"),
    [
        ("/tmp/astro/baseline.json", None),
        ("/tmp/astro/comparison.json", "/tmp/astro/candidate.json"),
        ("/tmp/astro/comparison.json", "/tmp/astro/./comparison.json"),
    ],
)
def test_assurance_comparison_plan_rejects_all_path_collisions(
    output: str, summary_output: str | None
) -> None:
    plan = assurance_review_comparison_plan(
        "Compare assurance reviews",
        baseline_path="/tmp/astro/baseline.json",
        candidate_path="/tmp/astro/candidate.json",
        output=output,
        summary_output=summary_output,
    )

    verification = verify_plan(plan)
    assert not verification.passed
    assert verification.diagnostics[0].code == "path_collision"
