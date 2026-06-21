from pydantic import BaseModel, Field

from astro_assistant.models import AstroWorkflowPlan, RiskLevel


class PolicyDecision(BaseModel):
    allowed: bool
    warnings: list[str] = Field(default_factory=list)


def evaluate_plan(plan: AstroWorkflowPlan, *, dry_run: bool, approved: bool) -> PolicyDecision:
    warnings: list[str] = []
    risks = {step.risk for step in plan.steps}

    if dry_run:
        return PolicyDecision(allowed=True)

    if RiskLevel.OPTIONAL_BACKEND in risks:
        warnings.append("optional backend execution is not enabled in the first assistant slice")

    if plan.requires_approval and RiskLevel.WRITES_ARTIFACTS in risks and not approved:
        warnings.append("execution requires approval because the plan writes artifacts")

    return PolicyDecision(allowed=len(warnings) == 0, warnings=warnings)
