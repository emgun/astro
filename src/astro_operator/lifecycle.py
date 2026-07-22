from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from astro_mission.errors import MissionLifecycleError
from astro_mission.io import write_mission_lifecycle_result
from astro_mission.models import MissionLifecycleInputOverrides, MissionLifecycleScenario
from astro_mission.runner import run_mission_lifecycle
from astro_operator.errors import OperatorEvaluationError
from astro_operator.models import (
    CandidateObservation,
    CandidateProposal,
    DesignVariable,
    EpistemicKind,
    EvidenceReference,
    ObservedMetric,
)

_SCALAR_OVERRIDE_FIELDS = {
    "launch_upper_stage_thrust_n",
    "spacecraft_wet_mass_kg",
    "twin_solar_array_efficiency",
    "twin_solar_array_area_m2",
    "twin_battery_capacity_wh",
    "reentry_atmosphere_density_scale_factor",
    "reentry_vehicle_drag_coefficient",
}


def resolve_lifecycle_references(
    scenario: MissionLifecycleScenario, source_path: Path
) -> MissionLifecycleScenario:
    """Resolve declared lifecycle dependencies without relying on the process CWD."""

    return scenario.model_copy(
        update={
            "launch_scenario": str(
                _resolve_declared_reference(scenario.launch_scenario, source_path)
            ),
            "twin_scenario": str(
                _resolve_declared_reference(scenario.twin_scenario, source_path)
            ),
            "reentry_scenario": str(
                _resolve_declared_reference(scenario.reentry_scenario, source_path)
            ),
        }
    )


class LifecycleCandidateEvaluator:
    """Evaluate reasoner-proposed designs through the existing lifecycle runner."""

    def __init__(
        self,
        *,
        base_scenario: MissionLifecycleScenario,
        design_variables: tuple[DesignVariable, ...],
        output_root: Path,
    ) -> None:
        self._base_scenario = base_scenario
        self._variables = {item.variable_id: item for item in design_variables}
        unsupported = {
            item.target for item in design_variables if item.target not in _SCALAR_OVERRIDE_FIELDS
        }
        if unsupported:
            raise OperatorEvaluationError(
                f"unsupported lifecycle override targets: {', '.join(sorted(unsupported))}"
            )
        self._output_root = output_root

    def evaluate(self, candidate: CandidateProposal) -> CandidateObservation:
        candidate_directory = self._output_root / "candidates" / candidate.candidate_id
        if candidate_directory.exists():
            raise OperatorEvaluationError(
                f"candidate artifact directory already exists: {candidate_directory}"
            )
        candidate_directory.mkdir(parents=True)

        override_payload = (
            self._base_scenario.input_overrides.model_dump(mode="python")
            if self._base_scenario.input_overrides is not None
            else {}
        )
        for variable_id, value in candidate.assignments.items():
            override_payload[self._variables[variable_id].target] = value
        overrides = MissionLifecycleInputOverrides.model_validate(override_payload)
        scenario = self._base_scenario.model_copy(
            update={
                "scenario_id": f"{self._base_scenario.scenario_id}--{candidate.candidate_id}",
                "input_overrides": overrides,
                "metadata": {
                    **self._base_scenario.metadata,
                    "operator_candidate_id": candidate.candidate_id,
                },
            }
        )

        scenario_path = candidate_directory / "scenario.json"
        scenario_path.write_text(scenario.model_dump_json(indent=2) + "\n", encoding="utf-8")
        try:
            result = run_mission_lifecycle(scenario)
        except MissionLifecycleError as exc:
            error_path = candidate_directory / "evaluation-error.json"
            error_path.write_text(
                json.dumps(
                    {
                        "candidate_id": candidate.candidate_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "operator_scenario_sha256": sha256(
                            scenario_path.read_bytes()
                        ).hexdigest(),
                        "resolved_input_overrides": overrides.model_dump(
                            mode="json",
                            exclude_none=True,
                            exclude_defaults=True,
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return CandidateObservation(
                candidate=candidate,
                evaluation_status="evaluation_failed",
                passed=False,
                metrics=(),
                evidence=(
                    _artifact_reference(
                        evidence_id=f"candidate:{candidate.candidate_id}:scenario",
                        kind="mission_lifecycle_scenario",
                        path=scenario_path,
                        root=self._output_root,
                    ),
                    _artifact_reference(
                        evidence_id=f"candidate:{candidate.candidate_id}:error",
                        kind="mission_lifecycle_evaluation_error",
                        path=error_path,
                        root=self._output_root,
                    ),
                ),
                warnings=(str(exc),),
                metadata={
                    "claim_boundary": self._base_scenario.metadata.get("claim_boundary")
                },
            )
        result_path = candidate_directory / "result.json"
        result = result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "operator_scenario_sha256": sha256(scenario_path.read_bytes()).hexdigest(),
                }
            }
        )
        write_mission_lifecycle_result(result_path, result)

        metrics = tuple(
            ObservedMetric(
                metric_id=f"margin:{margin.phase}:{margin.name}",
                value=margin.margin,
                unit=margin.unit,
                status=margin.status.value,
            )
            for margin in result.margin_report.margins
        )
        evidence = (
            _artifact_reference(
                evidence_id=f"candidate:{candidate.candidate_id}:scenario",
                kind="mission_lifecycle_scenario",
                path=scenario_path,
                root=self._output_root,
            ),
            _artifact_reference(
                evidence_id=f"candidate:{candidate.candidate_id}:result",
                kind="mission_lifecycle_result",
                path=result_path,
                root=self._output_root,
            ),
        )
        return CandidateObservation(
            candidate=candidate,
            evaluation_status="evaluated",
            passed=result.passed,
            metrics=metrics,
            evidence=evidence,
            warnings=tuple(result.warnings),
            metadata={
                "workflow": result.workflow,
                "overall_status": result.margin_report.overall_status.value,
                "continuity_all_passed": result.continuity_report.all_passed,
                "claim_boundary": self._base_scenario.metadata.get("claim_boundary"),
            },
        )


def _artifact_reference(
    *, evidence_id: str, kind: str, path: Path, root: Path
) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=evidence_id,
        kind=kind,
        epistemic_kind=(
            EpistemicKind.OBSERVED if kind.endswith("error") else EpistemicKind.SIMULATED
        ),
        claim_scope="deterministic mission lifecycle design-screening evidence",
        path=path.relative_to(root).as_posix(),
        sha256=sha256(path.read_bytes()).hexdigest(),
    )


def _resolve_declared_reference(declared: str, source_path: Path) -> Path:
    reference = Path(declared)
    if reference.is_absolute():
        return reference
    for ancestor in (source_path.parent, *source_path.parents):
        candidate = ancestor / reference
        if candidate.is_file():
            return candidate.resolve()
    raise OperatorEvaluationError(
        f"could not resolve lifecycle dependency {declared} from {source_path}"
    )
