from __future__ import annotations

import json
import os
import platform
import socket
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any

import typer

from astro_assurance.errors import MissionAssuranceError
from astro_assurance.io import load_post_launch_assurance_scenario
from astro_assurance.models import PostLaunchAssuranceScenario
from astro_assurance.runner import run_post_launch_assurance
from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario
from astro_core.models import AstroModel, Scenario
from astro_dynamics.backends import propagate_with_backend
from astro_launch.io import load_launch_scenario
from astro_mission.errors import MissionLifecycleError
from astro_mission.io import load_mission_lifecycle_scenario
from astro_mission.models import MissionLifecycleScenario
from astro_mission.runner import resolve_lifecycle_twin_scenario, run_mission_lifecycle
from astro_reentry.backends import simulate_reentry_with_backend
from astro_reentry.io import load_reentry_scenario
from astro_reentry.models import ReentryScenario
from astro_twin.io import load_twin_scenario
from astro_twin.models import DigitalTwinScenario
from astro_twin.runner import run_digital_twin
from astro_uq.adapters.assurance import assurance_metric_registry, assurance_parameter_registry
from astro_uq.adapters.lifecycle import lifecycle_metric_registry, lifecycle_parameter_registry
from astro_uq.adapters.orbit import orbit_metric_registry, orbit_parameter_registry
from astro_uq.adapters.reentry import reentry_metric_registry, reentry_parameter_registry
from astro_uq.adapters.twin import twin_metric_registry, twin_parameter_registry
from astro_uq.io import (
    CAMPAIGN_FILE,
    CASES_FILE,
    CampaignArtifactStore,
    CampaignIOError,
    atomic_write_json,
    load_campaign_definition,
    read_jsonl,
)
from astro_uq.metrics import MetricError, MetricRegistry
from astro_uq.models import (
    CampaignDefinition,
    CampaignSensitivityReport,
    CampaignState,
    CampaignStatistics,
    CampaignTimingProfile,
    CaseObservation,
    OutcomeStatus,
    ParameterRealization,
    TimingMachineMetadata,
    TimingRuntimeMetadata,
)
from astro_uq.parameters import ParameterBindingError, ParameterRegistry, model_digest
from astro_uq.profiling import summarize_case_timings
from astro_uq.runner import CampaignRuntime, run_campaign
from astro_uq.sensitivity import analyze_campaign_sensitivity
from astro_uq.statistics import summarize_campaign

SOFTWARE_COMPATIBILITY = {"astro-suite": "0.1.0", "campaign-runtime": "1.2"}
_RESOLVED_DEPENDENCIES_KEY = "resolved_dependencies"


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


def _referenced_scenario_path(owner_path: Path, configured_path: str) -> Path:
    configured = Path(configured_path)
    if configured.is_absolute():
        return configured
    for parent in owner_path.resolve().parents:
        candidate = parent / configured
        if candidate.exists():
            return candidate
    return configured


def _bind_resolved_dependencies(
    definition: CampaignDefinition, definition_path: Path
) -> CampaignDefinition:
    if definition.workflow.kind == "mission_assurance":
        scenario_path = _scenario_path(definition_path, definition.workflow.scenario)
        scenario = _resolve_assurance_scenario(
            scenario_path, load_post_launch_assurance_scenario(scenario_path)
        )
        metadata = dict(definition.metadata)
        metadata[_RESOLVED_DEPENDENCIES_KEY] = _assurance_dependencies(scenario)
        return CampaignDefinition.model_validate(
            definition.model_copy(update={"metadata": metadata}).model_dump(mode="python")
        )
    if definition.workflow.kind == "digital_twin":
        scenario_path = _scenario_path(definition_path, definition.workflow.scenario)
        twin_scenario = load_twin_scenario(scenario_path)
        orbit_scenario = load_scenario(
            _referenced_scenario_path(scenario_path, twin_scenario.orbit_scenario)
        )
        metadata = dict(definition.metadata)
        metadata[_RESOLVED_DEPENDENCIES_KEY] = {
            "twin_template_digest": model_digest(twin_scenario),
            "orbit_scenario": twin_scenario.orbit_scenario,
            "orbit_scenario_digest": model_digest(orbit_scenario),
        }
        return CampaignDefinition.model_validate(
            definition.model_copy(update={"metadata": metadata}).model_dump(mode="python")
        )
    if definition.workflow.kind != "mission_lifecycle":
        return definition
    scenario_path = _scenario_path(definition_path, definition.workflow.scenario)
    lifecycle_scenario = load_mission_lifecycle_scenario(scenario_path)
    twin_scenario = load_twin_scenario(lifecycle_scenario.twin_scenario)
    resolved_twin_scenario = resolve_lifecycle_twin_scenario(
        twin_scenario, lifecycle_scenario
    )
    metadata = dict(definition.metadata)
    metadata[_RESOLVED_DEPENDENCIES_KEY] = {
        "twin_scenario": lifecycle_scenario.twin_scenario,
        "twin_template_digest": model_digest(twin_scenario),
        "resolved_twin_scenario_digest": model_digest(resolved_twin_scenario),
    }
    return CampaignDefinition.model_validate(
        definition.model_copy(update={"metadata": metadata}).model_dump(mode="python")
    )


def _runtime(definition: CampaignDefinition, definition_path: Path) -> CampaignRuntime:
    scenario_path = _scenario_path(definition_path, definition.workflow.scenario)
    backend = definition.evaluator.backend or "local"
    if definition.workflow.kind == "mission_assurance":
        assurance_scenario = _resolve_assurance_scenario(
            scenario_path, load_post_launch_assurance_scenario(scenario_path)
        )
        assurance_twin_scenario = load_twin_scenario(assurance_scenario.twin_scenario)
        if definition.metadata.get(_RESOLVED_DEPENDENCIES_KEY) != _assurance_dependencies(
            assurance_scenario
        ):
            raise CampaignIOError("resolved mission assurance dependency digest mismatch")

        def evaluate(resolved: AstroModel) -> AstroModel:
            parsed = PostLaunchAssuranceScenario.model_validate(resolved).model_copy(
                update={
                    "source_path": assurance_scenario.source_path,
                    "source_digest": assurance_scenario.source_digest,
                }
            )
            return run_post_launch_assurance(parsed)

        return CampaignRuntime(
            scenario=assurance_scenario,
            parameters=assurance_parameter_registry(assurance_twin_scenario),
            metrics=assurance_metric_registry(),
            evaluate=evaluate,
        )
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
        twin_template = load_twin_scenario(scenario_path)
        orbit_scenario_path = _referenced_scenario_path(scenario_path, twin_template.orbit_scenario)
        resolved_twin = twin_template.model_copy(
            update={"orbit_scenario": str(orbit_scenario_path)}
        )
        dependency = definition.metadata.get(_RESOLVED_DEPENDENCIES_KEY)
        expected_dependency = {
            "twin_template_digest": model_digest(twin_template),
            "orbit_scenario": twin_template.orbit_scenario,
            "orbit_scenario_digest": model_digest(load_scenario(orbit_scenario_path)),
        }
        if dependency != expected_dependency:
            raise CampaignIOError("resolved digital twin dependency digest mismatch")
        return CampaignRuntime(
            scenario=resolved_twin,
            parameters=twin_parameter_registry(resolved_twin),
            metrics=twin_metric_registry(resolved_twin),
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
        lifecycle_scenario = load_mission_lifecycle_scenario(scenario_path)
        lifecycle_twin_scenario = load_twin_scenario(lifecycle_scenario.twin_scenario)
        resolved_lifecycle_twin_scenario = resolve_lifecycle_twin_scenario(
            lifecycle_twin_scenario, lifecycle_scenario
        )
        dependency = definition.metadata.get(_RESOLVED_DEPENDENCIES_KEY)
        expected_dependency = {
            "twin_scenario": lifecycle_scenario.twin_scenario,
            "twin_template_digest": model_digest(lifecycle_twin_scenario),
            "resolved_twin_scenario_digest": model_digest(
                resolved_lifecycle_twin_scenario
            ),
        }
        if dependency != expected_dependency:
            raise CampaignIOError("resolved lifecycle twin dependency digest mismatch")
        return CampaignRuntime(
            scenario=lifecycle_scenario,
            parameters=lifecycle_parameter_registry(lifecycle_twin_scenario),
            metrics=lifecycle_metric_registry(lifecycle_twin_scenario),
            evaluate=lambda resolved: run_mission_lifecycle(
                MissionLifecycleScenario.model_validate(resolved)
            ),
        )
    raise CampaignIOError(f"unsupported campaign workflow {definition.workflow.kind!r}")


def _assurance_dependencies(scenario: PostLaunchAssuranceScenario) -> dict[str, str]:
    launch = load_launch_scenario(scenario.launch_scenario)
    tracking = load_scenario(scenario.tracking_scenario)
    twin = load_twin_scenario(scenario.twin_scenario)
    if scenario.source_path is None or scenario.source_digest is None:
        raise CampaignIOError("mission assurance scenario is missing source provenance")
    return {
        "assurance_scenario": scenario.source_path,
        "assurance_scenario_digest": scenario.source_digest,
        "launch_scenario": scenario.launch_scenario,
        "launch_scenario_digest": model_digest(launch),
        "tracking_scenario": scenario.tracking_scenario,
        "tracking_scenario_digest": model_digest(tracking),
        "twin_scenario": scenario.twin_scenario,
        "twin_scenario_digest": model_digest(twin),
    }


def _resolve_assurance_scenario(
    scenario_path: Path,
    scenario: PostLaunchAssuranceScenario,
) -> PostLaunchAssuranceScenario:
    return scenario.model_copy(
        update={
            "launch_scenario": str(
                _referenced_scenario_path(scenario_path, scenario.launch_scenario).resolve()
            ),
            "tracking_scenario": str(
                _referenced_scenario_path(scenario_path, scenario.tracking_scenario).resolve()
            ),
            "twin_scenario": str(
                _referenced_scenario_path(scenario_path, scenario.twin_scenario).resolve()
            ),
        }
    )


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
        definition = _bind_resolved_dependencies(
            load_campaign_definition(definition_path), definition_path
        )
        campaign_id = definition.campaign_id
        runtime = _runtime(definition, definition_path)
        _validate_registries(definition, runtime.parameters, runtime.metrics)
    except (
        CampaignIOError,
        InvalidScenarioError,
        MetricError,
        MissionAssuranceError,
        MissionLifecycleError,
        ParameterBindingError,
    ) as exc:
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
        definition = _bind_resolved_dependencies(
            load_campaign_definition(definition_path), definition_path
        )
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
        MissionAssuranceError,
        MissionLifecycleError,
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


def profile_campaign_command(
    output_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Summarize machine-scoped evaluator timing from completed campaign cases."""
    campaign_id: str | None = None
    try:
        if output is not None and output.resolve().is_relative_to(output_dir.resolve()):
            raise CampaignIOError("profile output must be outside the source campaign directory")
        manifest = json.loads((output_dir / CAMPAIGN_FILE).read_text(encoding="utf-8"))
        definition = CampaignDefinition.model_validate(manifest["definition"])
        campaign_id = definition.campaign_id
        store = CampaignArtifactStore(output_dir)
        with store:
            resume_state = store.resume(
                definition,
                software_compatibility=manifest["software_compatibility"],
            )
            verified_manifest = json.loads(
                (output_dir / CAMPAIGN_FILE).read_text(encoding="utf-8")
            )
        if resume_state.state is not CampaignState.COMPLETED:
            raise CampaignIOError("only completed campaigns can be profiled")
        observations = tuple(
            CaseObservation.model_validate(case) for case in resume_state.cases
        )
        profile = CampaignTimingProfile(
            campaign_id=definition.campaign_id,
            definition_digest=resume_state.definition_digest,
            cases_digest=str(verified_manifest["cases_digest"]),
            claim_boundary=definition.evaluator.claim_boundary,
            software_compatibility={
                str(key): str(value)
                for key, value in manifest["software_compatibility"].items()
            },
            machine=TimingMachineMetadata(
                hostname=socket.gethostname(),
                operating_system=platform.system(),
                operating_system_release=platform.release(),
                architecture=platform.machine(),
                processor=platform.processor() or None,
                logical_cpu_count=os.cpu_count(),
            ),
            runtime=TimingRuntimeMetadata(
                python_version=platform.python_version(),
                astro_version=_astro_version(),
            ),
            timing=summarize_case_timings(observations),
        )
        if output is not None:
            atomic_write_json(output, profile)
    except (KeyError, ValueError, TypeError, OSError, CampaignIOError, json.JSONDecodeError) as exc:
        _error(f"could not profile campaign: {exc}", campaign_id=campaign_id)
        raise typer.Exit(code=2) from exc
    typer.echo(profile.model_dump_json())


def _astro_version() -> str:
    try:
        return version("astro-suite")
    except PackageNotFoundError:
        return "unknown"


def analyze_campaign_sensitivity_command(
    output_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output")],
    metric: Annotated[list[str] | None, typer.Option("--metric")] = None,
    requirement_margin: Annotated[
        list[str] | None, typer.Option("--requirement-margin")
    ] = None,
) -> None:
    """Attribute campaign metrics and requirement margins to declared parameters."""
    campaign_id: str | None = None
    try:
        if output.resolve().is_relative_to(output_dir.resolve()):
            raise CampaignIOError(
                "sensitivity output must be outside the source campaign directory"
            )
        manifest = json.loads((output_dir / CAMPAIGN_FILE).read_text(encoding="utf-8"))
        definition = CampaignDefinition.model_validate(manifest["definition"])
        campaign_id = definition.campaign_id
        store = CampaignArtifactStore(output_dir)
        with store:
            resume_state = store.resume(
                definition,
                software_compatibility=manifest["software_compatibility"],
            )
            verified_manifest = json.loads(
                (output_dir / CAMPAIGN_FILE).read_text(encoding="utf-8")
            )
        if resume_state.state is not CampaignState.COMPLETED:
            raise CampaignIOError("only completed campaigns can be analyzed for sensitivity")
        report: CampaignSensitivityReport = analyze_campaign_sensitivity(
            definition,
            tuple(ParameterRealization.model_validate(sample) for sample in resume_state.samples),
            tuple(CaseObservation.model_validate(case) for case in resume_state.cases),
            metric_ids=tuple(metric or ()),
            requirement_margin_ids=tuple(requirement_margin or ()),
            definition_digest=resume_state.definition_digest,
            samples_digest=str(verified_manifest["samples_digest"]),
            cases_digest=str(verified_manifest["cases_digest"]),
        )
        atomic_write_json(output, report)
    except (
        KeyError,
        ValueError,
        TypeError,
        OSError,
        CampaignIOError,
        json.JSONDecodeError,
    ) as exc:
        _error(f"could not analyze campaign sensitivity: {exc}", campaign_id=campaign_id)
        raise typer.Exit(code=2) from exc
    typer.echo(report.model_dump_json())
