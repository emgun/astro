from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import typer

from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario
from astro_core.models import Scenario
from astro_dynamics.backends import propagate_with_backend
from astro_mission.io import load_mission_lifecycle_scenario
from astro_mission.models import MissionLifecycleScenario
from astro_mission.runner import run_mission_lifecycle
from astro_reentry.backends import simulate_reentry_with_backend
from astro_reentry.io import load_reentry_scenario
from astro_reentry.models import ReentryScenario
from astro_twin.io import load_twin_scenario
from astro_twin.models import DigitalTwinScenario
from astro_twin.runner import run_digital_twin
from astro_uq.adapters.lifecycle import lifecycle_metric_registry, lifecycle_parameter_registry
from astro_uq.adapters.orbit import orbit_metric_registry, orbit_parameter_registry
from astro_uq.adapters.reentry import reentry_metric_registry, reentry_parameter_registry
from astro_uq.adapters.twin import twin_metric_registry, twin_parameter_registry
from astro_uq.io import (
    CAMPAIGN_FILE,
    CASES_FILE,
    CampaignArtifactStore,
    CampaignIOError,
    load_campaign_definition,
    read_jsonl,
)
from astro_uq.metrics import MetricError, MetricRegistry
from astro_uq.models import CampaignDefinition, CampaignStatistics, CaseObservation, OutcomeStatus
from astro_uq.parameters import ParameterBindingError, ParameterRegistry
from astro_uq.runner import CampaignRuntime, run_campaign
from astro_uq.statistics import summarize_campaign

SOFTWARE_COMPATIBILITY = {"astro-suite": "0.1.0", "campaign-runtime": "1.0"}


def _error(message: str, *, campaign_id: str | None = None, sample_id: str | None = None) -> None:
    payload: dict[str, Any] = {"error": message}
    if campaign_id is not None:
        payload["campaign_id"] = campaign_id
    if sample_id is not None:
        payload["sample_id"] = sample_id
    typer.echo(json.dumps(payload, sort_keys=True), err=True)


def _scenario_path(definition_path: Path, configured_path: str) -> Path:
    configured = Path(configured_path)
    if configured.is_absolute() or configured.exists():
        return configured
    return definition_path.parent / configured


def _runtime(definition: CampaignDefinition, definition_path: Path) -> CampaignRuntime:
    scenario_path = _scenario_path(definition_path, definition.workflow.scenario)
    backend = definition.evaluator.backend or "local"
    if definition.workflow.kind == "orbit":
        return CampaignRuntime(
            scenario=load_scenario(scenario_path),
            parameters=orbit_parameter_registry(),
            metrics=orbit_metric_registry(),
            evaluate=lambda resolved: propagate_with_backend(
                Scenario.model_validate(resolved), backend
            ),
        )
    if definition.workflow.kind == "digital_twin":
        scenario = load_twin_scenario(scenario_path)
        return CampaignRuntime(
            scenario=scenario,
            parameters=twin_parameter_registry(scenario),
            metrics=twin_metric_registry(scenario),
            evaluate=lambda resolved: run_digital_twin(
                DigitalTwinScenario.model_validate(resolved)
            ),
        )
    if definition.workflow.kind == "reentry":
        return CampaignRuntime(
            scenario=load_reentry_scenario(scenario_path),
            parameters=reentry_parameter_registry(),
            metrics=reentry_metric_registry(),
            evaluate=lambda resolved: simulate_reentry_with_backend(
                ReentryScenario.model_validate(resolved), backend
            ),
        )
    if definition.workflow.kind == "mission_lifecycle":
        return CampaignRuntime(
            scenario=load_mission_lifecycle_scenario(scenario_path),
            parameters=lifecycle_parameter_registry(),
            metrics=lifecycle_metric_registry(),
            evaluate=lambda resolved: run_mission_lifecycle(
                MissionLifecycleScenario.model_validate(resolved)
            ),
        )
    raise CampaignIOError(f"unsupported campaign workflow {definition.workflow.kind!r}")


def _validate_registries(
    definition: CampaignDefinition,
    parameters: ParameterRegistry,
    metrics: MetricRegistry,
) -> None:
    workflow = definition.workflow.kind
    for parameter in definition.uncertainty.parameters:
        binding = parameters.resolve(workflow, parameter.target)
        if binding.unit != parameter.unit:
            raise ParameterBindingError(
                f"unit mismatch for {parameter.parameter_id}: "
                f"expected {binding.unit}, received {parameter.unit}"
            )

    for specification in definition.metrics:
        extractor = metrics.resolve(workflow, specification.extractor)
        if extractor.value_kind is not specification.value_kind:
            raise MetricError(f"metric kind mismatch for {specification.metric_id}")
        if extractor.unit != specification.unit:
            raise MetricError(f"metric unit mismatch for {specification.metric_id}")


def validate_campaign(
    definition_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """Validate a supported campaign definition without executing it."""
    campaign_id: str | None = None
    try:
        definition = load_campaign_definition(definition_path)
        campaign_id = definition.campaign_id
        runtime = _runtime(definition, definition_path)
        _validate_registries(definition, runtime.parameters, runtime.metrics)
    except (CampaignIOError, InvalidScenarioError, MetricError, ParameterBindingError) as exc:
        _error(str(exc), campaign_id=campaign_id)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps({"campaign_id": campaign_id, "valid": True}, sort_keys=True))


def run_campaign_command(
    definition_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    resume: Annotated[bool, typer.Option("--resume")] = False,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    max_cases: Annotated[int | None, typer.Option("--max-cases", min=1)] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Execute a supported uncertainty campaign."""
    campaign_id: str | None = None
    try:
        definition = load_campaign_definition(definition_path)
        configured_samples = definition.sampler.samples
        if max_cases is not None and max_cases < configured_samples:
            limited = definition.model_copy(
                update={"sampler": definition.sampler.model_copy(update={"samples": max_cases})}
            )
            try:
                definition = CampaignDefinition.model_validate(limited.model_dump(mode="python"))
            except ValueError as exc:
                raise CampaignIOError(f"max-cases is incompatible with campaign: {exc}") from exc
        campaign_id = definition.campaign_id
        runtime = _runtime(definition, definition_path)
        _validate_registries(definition, runtime.parameters, runtime.metrics)
        if dry_run:
            typer.echo(
                json.dumps(
                    {
                        "campaign_id": campaign_id,
                        "configured_samples": configured_samples,
                        "planned_samples": definition.sampler.samples,
                        "valid": True,
                        "workers": workers,
                    },
                    sort_keys=True,
                )
            )
            return
        result = run_campaign(
            definition,
            runtime,
            output_dir=output_dir,
            software_compatibility=SOFTWARE_COMPATIBILITY,
            resume=resume,
            workers=workers,
            runtime_factory=(
                partial(_runtime, definition, definition_path) if workers > 1 else None
            ),
        )
    except (
        CampaignIOError,
        InvalidScenarioError,
        MetricError,
        ParameterBindingError,
        OSError,
    ) as exc:
        _error(str(exc), campaign_id=campaign_id)
        raise typer.Exit(code=2) from exc

    failed = [
        case
        for case in read_jsonl(output_dir / CASES_FILE)
        if case.get("outcome_status") != OutcomeStatus.SUCCESS.value
    ]
    if failed:
        for case in failed:
            _error(
                f"campaign case ended with {case.get('outcome_status', 'unknown')}",
                campaign_id=definition.campaign_id,
                sample_id=case.get("sample_id"),
            )
        raise typer.Exit(code=1)
    typer.echo(result.model_dump_json())


def _format_summary(definition: CampaignDefinition, statistics: CampaignStatistics) -> str:
    lines = [
        f"Campaign: {definition.campaign_id}",
        f"Workflow: {definition.workflow.kind}",
        f"Completed: {statistics.completed_samples}/{statistics.requested_samples}",
        f"Claim boundary: {definition.evaluator.claim_boundary}",
    ]
    lines.extend(
        f"Requirement {requirement_id}: {probability:.6f}"
        for requirement_id, probability in sorted(statistics.requirement_probabilities.items())
    )
    return "\n".join(lines) + "\n"


def summarize_campaign_command(
    output_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
) -> None:
    """Regenerate campaign statistics and the concise text summary."""
    campaign_id: str | None = None
    try:
        manifest = json.loads((output_dir / CAMPAIGN_FILE).read_text(encoding="utf-8"))
        definition = CampaignDefinition.model_validate(manifest["definition"])
        campaign_id = definition.campaign_id
        store = CampaignArtifactStore(output_dir)
        with store:
            resume_state = store.resume(
                definition,
                software_compatibility=manifest["software_compatibility"],
                require_completed_statistics=False,
            )
            samples = resume_state.samples
            observations = tuple(
                CaseObservation.model_validate(case) for case in resume_state.cases
            )
            prior_statistics = (
                None
                if resume_state.statistics is None
                else CampaignStatistics.model_validate(resume_state.statistics)
            )
            weights = {str(sample["sample_id"]): float(sample["weight"]) for sample in samples}
            observation_ids = {observation.sample_id for observation in observations}
            statistics = summarize_campaign(
                requested_samples=definition.sampler.samples,
                observations=observations,
                weights={sample_id: weights[sample_id] for sample_id in observation_ids},
                requirement_ids=tuple(
                    requirement.requirement_id for requirement in definition.requirements
                ),
                convergence_history=(
                    () if prior_statistics is None else prior_statistics.convergence_history
                ),
            )
            summary = _format_summary(definition, statistics)
            store.write_statistics(statistics)
            store.write_summary(summary)
    except (KeyError, ValueError, TypeError, OSError, CampaignIOError, json.JSONDecodeError) as exc:
        _error(f"could not summarize campaign: {exc}", campaign_id=campaign_id)
        raise typer.Exit(code=2) from exc
    typer.echo(summary, nl=False)
