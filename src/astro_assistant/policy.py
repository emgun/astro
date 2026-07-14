from pydantic import BaseModel, Field

from astro_assistant.models import AstroToolName, AstroWorkflowPlan, RiskLevel


class PolicyDecision(BaseModel):
    allowed: bool
    warnings: list[str] = Field(default_factory=list)


def evaluate_plan(plan: AstroWorkflowPlan, *, dry_run: bool, approved: bool) -> PolicyDecision:
    warnings: list[str] = []
    risks = {step.risk for step in plan.steps}

    if dry_run:
        return PolicyDecision(allowed=True)

    campaign_risks = {
        AstroToolName.VALIDATE_CAMPAIGN: RiskLevel.READ_ONLY,
        AstroToolName.RUN_CAMPAIGN: RiskLevel.WRITES_ARTIFACTS,
        AstroToolName.SUMMARIZE_CAMPAIGN: RiskLevel.READ_ONLY,
        AstroToolName.VERIFY_ASSURANCE_VALIDATION: RiskLevel.READ_ONLY,
        AstroToolName.REVIEW_ASSURANCE_VALIDATION: RiskLevel.WRITES_ARTIFACTS,
    }
    for step in plan.steps:
        expected_risk = campaign_risks.get(step.tool)
        if expected_risk is not None and step.risk is not expected_risk:
            warnings.append(
                f"{step.tool.value} must use {expected_risk.value} risk classification"
            )

    if RiskLevel.OPTIONAL_BACKEND in risks:
        warnings.append("optional backend execution is not enabled in the first assistant slice")

    always_approved_writes = any(
        step.tool in {AstroToolName.RUN_CAMPAIGN, AstroToolName.REVIEW_ASSURANCE_VALIDATION}
        for step in plan.steps
    )
    writes_requiring_approval = always_approved_writes or (
        plan.requires_approval and RiskLevel.WRITES_ARTIFACTS in risks
    )
    if writes_requiring_approval and not approved:
        warnings.append("execution requires approval because the plan writes artifacts")

    return PolicyDecision(allowed=len(warnings) == 0, warnings=warnings)
