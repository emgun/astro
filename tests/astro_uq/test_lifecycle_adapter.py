from __future__ import annotations

from pathlib import Path

from astro_mission.io import load_mission_lifecycle_scenario
from astro_mission.models import MissionLifecycleScenario
from astro_mission.runner import run_mission_lifecycle
from astro_uq.adapters.lifecycle import (
    lifecycle_metric_registry,
    lifecycle_parameter_registry,
)
from astro_uq.models import (
    CampaignDefinition,
    DistributionKind,
    DistributionSpec,
    EvaluatorKind,
    EvaluatorSpec,
    MetricSpec,
    MetricValueKind,
    ParameterRealization,
    RequirementOperator,
    RequirementSpec,
    SamplerKind,
    SamplerSpec,
    UncertainParameter,
    UncertaintyKind,
    UncertaintyModel,
    WorkflowSpec,
)
from astro_uq.runner import CampaignRuntime, run_campaign


def test_lifecycle_deorbit_binding_revalidates_scenario() -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    uncertainty = UncertaintyModel(
        parameters=(
            UncertainParameter(
                parameter_id="delta_v",
                target="lifecycle.deorbit.delta_v_km_s",
                unit="km/s",
                uncertainty_kind=UncertaintyKind.EPISTEMIC,
                distribution=DistributionSpec(
                    kind=DistributionKind.UNIFORM,
                    low=0.09,
                    high=0.11,
                ),
            ),
        )
    )

    resolved, evidence = lifecycle_parameter_registry().apply(
        workflow="mission_lifecycle",
        scenario=scenario,
        uncertainty=uncertainty,
        realization=ParameterRealization(
            sample_id="sample-0",
            sample_index=0,
            normalized_values={"delta_v": 0.5},
            physical_values={"delta_v": 0.1},
        ),
    )

    assert resolved.deorbit.delta_v_km_s == 0.1  # type: ignore[attr-defined]
    assert evidence.resolved_scenario_digest != evidence.base_scenario_digest


def test_lifecycle_cross_phase_bindings_use_typed_input_overrides() -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    definitions = (
        ("thrust", "lifecycle.launch.upper_stage_thrust_n", "N", 201000.0),
        ("wet_mass", "lifecycle.spacecraft.wet_mass_kg", "kg", 505.0),
        (
            "power",
            "lifecycle.digital_twin.power.solar_array_efficiency",
            "1",
            0.3,
        ),
        (
            "density",
            "lifecycle.reentry.atmosphere.density_scale_factor",
            "1",
            1.05,
        ),
        ("drag", "lifecycle.reentry.vehicle.drag_coefficient", "1", 1.6),
    )
    uncertainty = UncertaintyModel(
        parameters=tuple(
            UncertainParameter(
                parameter_id=parameter_id,
                target=target,
                unit=unit,
                uncertainty_kind=UncertaintyKind.EPISTEMIC,
                distribution=DistributionSpec(kind=DistributionKind.CONSTANT, value=value),
            )
            for parameter_id, target, unit, value in definitions
        )
    )
    realization = ParameterRealization(
        sample_id="sample-cross-phase",
        sample_index=0,
        normalized_values={item[0]: 0.5 for item in definitions},
        physical_values={item[0]: item[3] for item in definitions},
    )

    resolved, evidence = lifecycle_parameter_registry().apply(
        workflow="mission_lifecycle",
        scenario=scenario,
        uncertainty=uncertainty,
        realization=realization,
    )
    validated = MissionLifecycleScenario.model_validate(resolved)

    assert validated.input_overrides is not None
    assert validated.input_overrides.launch_upper_stage_thrust_n == 201000.0
    assert validated.input_overrides.spacecraft_wet_mass_kg == 505.0
    assert validated.input_overrides.twin_solar_array_efficiency == 0.3
    assert validated.input_overrides.reentry_atmosphere_density_scale_factor == 1.05
    assert validated.input_overrides.reentry_vehicle_drag_coefficient == 1.6
    assert evidence.resolved_scenario_digest != evidence.base_scenario_digest


def test_lifecycle_metrics_extract_checked_result() -> None:
    result = run_mission_lifecycle(
        load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    )
    values = lifecycle_metric_registry().extract(
        workflow="mission_lifecycle",
        result=result,
        specifications=(
            MetricSpec(
                metric_id="mission_passed",
                extractor="lifecycle.passed",
                value_kind=MetricValueKind.BOOLEAN,
            ),
            MetricSpec(
                metric_id="reserve",
                extractor="lifecycle.propellant_reserve_margin_kg",
                value_kind=MetricValueKind.NUMERIC,
                unit="kg",
            ),
            MetricSpec(
                metric_id="propellant_used",
                extractor="lifecycle.deorbit_propellant_used_kg",
                value_kind=MetricValueKind.NUMERIC,
                unit="kg",
            ),
            MetricSpec(
                metric_id="peak_dynamic_pressure",
                extractor="lifecycle.reentry_peak_dynamic_pressure_pa",
                value_kind=MetricValueKind.NUMERIC,
                unit="Pa",
            ),
        ),
    )

    assert values["mission_passed"] is True
    assert isinstance(values["reserve"], float)
    assert values["propellant_used"] == result.metadata["propellant_used_kg"]
    assert values["peak_dynamic_pressure"] == result.reentry_result.peaks.dynamic_pressure.value


def test_failed_launch_is_counted_and_records_stopped_phase(tmp_path: Path) -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    definition = CampaignDefinition(
        campaign_id="failed-lifecycle-launch",
        workflow=WorkflowSpec(kind="mission_lifecycle", scenario="checked.yaml"),
        uncertainty=UncertaintyModel(
            parameters=(
                UncertainParameter(
                    parameter_id="thrust",
                    target="lifecycle.launch.upper_stage_thrust_n",
                    unit="N",
                    uncertainty_kind=UncertaintyKind.EPISTEMIC,
                    distribution=DistributionSpec(
                        kind=DistributionKind.CONSTANT,
                        value=100000.0,
                    ),
                ),
            )
        ),
        sampler=SamplerSpec(kind=SamplerKind.PSEUDORANDOM, samples=1, seed=3),
        evaluator=EvaluatorSpec(
            evaluator_id="local-lifecycle",
            kind=EvaluatorKind.AUTHORITATIVE,
            workflow="mission_lifecycle",
            implementation_version="1",
            claim_boundary="test_design_screening",
        ),
        metrics=(
            MetricSpec(
                metric_id="mission_passed",
                extractor="lifecycle.passed",
                value_kind=MetricValueKind.BOOLEAN,
            ),
        ),
        requirements=(
            RequirementSpec(
                requirement_id="mission_success",
                metric_id="mission_passed",
                operator=RequirementOperator.IS_TRUE,
            ),
        ),
    )
    runtime = CampaignRuntime(
        scenario=scenario,
        parameters=lifecycle_parameter_registry(),
        metrics=lifecycle_metric_registry(),
        evaluate=lambda model: run_mission_lifecycle(
            MissionLifecycleScenario.model_validate(model)
        ),
    )

    result = run_campaign(
        definition,
        runtime,
        output_dir=tmp_path / "failed-launch",
        software_compatibility={"astro": "test"},
    )

    assert result.statistics.outcome_counts == {"execution_failure": 1}
    assert result.statistics.requirement_probabilities == {"mission_success": 0.0}
    case = (tmp_path / "failed-launch" / "cases.jsonl").read_text().strip()
    assert '"workflow_phase":"launch"' in case
    assert '"artifact_refs":[]' in case
