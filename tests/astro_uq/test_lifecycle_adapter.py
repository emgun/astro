from __future__ import annotations

from pathlib import Path

from astro_mission.io import load_mission_lifecycle_scenario
from astro_mission.models import MissionLifecycleScenario
from astro_mission.runner import run_mission_lifecycle
from astro_uq.adapters.lifecycle import (
    lifecycle_metric_registry,
    lifecycle_parameter_registry,
)
from astro_uq.metrics import evaluate_requirements
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
            MetricSpec(
                metric_id="battery_margin",
                extractor="lifecycle.twin_battery_soc_margin_fraction",
                value_kind=MetricValueKind.NUMERIC,
                unit="1",
            ),
            MetricSpec(
                metric_id="thermal_margin",
                extractor="lifecycle.twin_minimum_thermal_margin_k",
                value_kind=MetricValueKind.NUMERIC,
                unit="K",
            ),
            MetricSpec(
                metric_id="pointing_margin",
                extractor="lifecycle.twin_pointing_margin_deg",
                value_kind=MetricValueKind.NUMERIC,
                unit="deg",
            ),
            MetricSpec(
                metric_id="torque_margin",
                extractor="lifecycle.twin_torque_margin_n_m",
                value_kind=MetricValueKind.NUMERIC,
                unit="N*m",
            ),
            MetricSpec(
                metric_id="slew_margin",
                extractor="lifecycle.twin_slew_rate_margin_deg_s",
                value_kind=MetricValueKind.NUMERIC,
                unit="deg/s",
            ),
            MetricSpec(
                metric_id="actuator_margin",
                extractor="lifecycle.twin_actuator_utilization_margin_fraction",
                value_kind=MetricValueKind.NUMERIC,
                unit="1",
            ),
            MetricSpec(
                metric_id="has_contact",
                extractor="lifecycle.twin_has_contact",
                value_kind=MetricValueKind.BOOLEAN,
            ),
            MetricSpec(
                metric_id="worst_observed_link_margin",
                extractor="lifecycle.twin_worst_observed_link_margin_db",
                value_kind=MetricValueKind.NUMERIC,
                unit="dB",
            ),
            MetricSpec(
                metric_id="propellant_fraction_margin",
                extractor="lifecycle.twin_propellant_fraction_margin",
                value_kind=MetricValueKind.NUMERIC,
                unit="1",
            ),
            MetricSpec(
                metric_id="mass_budget_margin",
                extractor="lifecycle.twin_mass_budget_rollup_margin_kg",
                value_kind=MetricValueKind.NUMERIC,
                unit="kg",
            ),
            MetricSpec(
                metric_id="access_count",
                extractor="lifecycle.twin_access_window_count",
                value_kind=MetricValueKind.NUMERIC,
                unit="1",
            ),
            MetricSpec(
                metric_id="access_duration",
                extractor="lifecycle.twin_total_access_duration_s",
                value_kind=MetricValueKind.NUMERIC,
                unit="s",
            ),
        ),
    )

    assert values["mission_passed"] is True
    assert isinstance(values["reserve"], float)
    assert values["propellant_used"] == result.metadata["propellant_used_kg"]
    assert values["peak_dynamic_pressure"] == result.reentry_result.peaks.dynamic_pressure.value
    twin_margins = {
        margin.name: margin.margin for margin in result.digital_twin.margin_report.margins
    }
    assert values["battery_margin"] == twin_margins["battery_soc_margin_fraction"]
    assert values["thermal_margin"] == min(
        value for name, value in twin_margins.items() if name.startswith("thermal_")
    )
    assert values["pointing_margin"] == twin_margins["pointing_margin_deg"]
    assert values["torque_margin"] == twin_margins["torque_margin_n_m"]
    assert values["slew_margin"] == twin_margins["slew_rate_margin_deg_s"]
    assert values["actuator_margin"] == twin_margins[
        "actuator_utilization_margin_fraction"
    ]
    assert values["has_contact"] is True
    assert values["worst_observed_link_margin"] == min(
        window.worst_ebn0_margin_db for window in result.digital_twin.link_windows
    )
    assert values["propellant_fraction_margin"] == twin_margins["mass_margin_fraction"]
    assert values["mass_budget_margin"] == twin_margins[
        "mass_budget_rollup_margin_kg"
    ]
    assert values["access_count"] == float(len(result.digital_twin.access_windows))
    assert values["access_duration"] == sum(
        window.duration_s for window in result.digital_twin.access_windows
    )


def test_lifecycle_link_margin_is_missing_without_contact() -> None:
    result = run_mission_lifecycle(
        load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    )
    twin = result.digital_twin.model_copy(update={"link_windows": ()})
    without_contact = result.model_copy(update={"digital_twin": twin})

    values = lifecycle_metric_registry().extract(
        workflow="mission_lifecycle",
        result=without_contact,
        specifications=(
            MetricSpec(
                metric_id="has_contact",
                extractor="lifecycle.twin_has_contact",
                value_kind=MetricValueKind.BOOLEAN,
            ),
            MetricSpec(
                metric_id="worst_observed_link_margin",
                extractor="lifecycle.twin_worst_observed_link_margin_db",
                value_kind=MetricValueKind.NUMERIC,
                unit="dB",
            ),
        ),
    )

    assert values == {"has_contact": False, "worst_observed_link_margin": None}
    outcomes = evaluate_requirements(
        values,
        (
            RequirementSpec(
                requirement_id="contact_available",
                metric_id="has_contact",
                operator=RequirementOperator.IS_TRUE,
            ),
            RequirementSpec(
                requirement_id="observed_link_margin",
                metric_id="worst_observed_link_margin",
                operator=RequirementOperator.GE,
                value=0.0,
            ),
        ),
    )
    assert outcomes[0].passed is False
    assert outcomes[0].margin == -1.0
    assert outcomes[1].passed is None
    assert outcomes[1].reason == "metric_missing_or_not_applicable"


def test_lifecycle_link_margin_uses_worst_observed_contact() -> None:
    result = run_mission_lifecycle(
        load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    )
    first = result.digital_twin.link_windows[0]
    worse = first.model_copy(update={"worst_ebn0_margin_db": 2.5})
    twin = result.digital_twin.model_copy(
        update={"link_windows": (*result.digital_twin.link_windows, worse)}
    )
    with_extra_contact = result.model_copy(update={"digital_twin": twin})

    values = lifecycle_metric_registry().extract(
        workflow="mission_lifecycle",
        result=with_extra_contact,
        specifications=(
            MetricSpec(
                metric_id="worst_observed_link_margin",
                extractor="lifecycle.twin_worst_observed_link_margin_db",
                value_kind=MetricValueKind.NUMERIC,
                unit="dB",
            ),
        ),
    )

    assert values == {"worst_observed_link_margin": 2.5}


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
