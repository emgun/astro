from pathlib import Path

from astro_core.io import load_scenario
from astro_dynamics.ephemeris import (
    dump_trajectory_aem,
    dump_trajectory_ephemeris_csv,
    dump_trajectory_oem,
    dump_trajectory_opm,
    load_trajectory_aem,
    load_trajectory_oem,
    load_trajectory_opm,
)
from astro_dynamics.local import propagate_local


def test_dump_trajectory_ephemeris_csv_writes_samples() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)

    payload = dump_trajectory_ephemeris_csv(trajectory)

    lines = payload.splitlines()
    assert lines[0] == (
        "scenario_id,backend,epoch,"
        "x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s"
    )
    assert len(lines) == len(trajectory.samples) + 1
    assert lines[1].startswith("leo-two-body,local,2026-01-01T00:00:00+00:00,7000")


def test_dump_trajectory_oem_writes_ccsds_oem_kvn_product() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)

    payload = dump_trajectory_oem(trajectory)

    lines = payload.splitlines()
    assert lines[0] == "CCSDS_OEM_VERS = 2.0"
    assert "ORIGINATOR = ASTRO_SUITE" in lines
    assert "META_START" in lines
    assert "OBJECT_NAME = leo-two-body" in lines
    assert "CENTER_NAME = EARTH" in lines
    assert "REF_FRAME = EME2000" in lines
    assert "TIME_SYSTEM = UTC" in lines
    assert "META_STOP" in lines
    assert "COMMENT backend = local" in lines
    assert "2026-01-01T00:00:00.000000Z 7000.0 0.0 0.0 0.0 7.5 1.0" in lines
    assert len(
        [
            line
            for line in lines
            if line.startswith("2026-01-01T")
        ]
    ) == len(trajectory.samples)


def test_dump_trajectory_opm_writes_ccsds_opm_kvn_state_message() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)

    payload = dump_trajectory_opm(trajectory)

    lines = payload.splitlines()
    assert lines[0] == "CCSDS_OPM_VERS = 2.0"
    assert "ORIGINATOR = ASTRO_SUITE" in lines
    assert "META_START" in lines
    assert "OBJECT_NAME = leo-two-body" in lines
    assert "CENTER_NAME = EARTH" in lines
    assert "REF_FRAME = EME2000" in lines
    assert "TIME_SYSTEM = UTC" in lines
    assert "META_STOP" in lines
    assert "COMMENT backend = local" in lines
    assert "EPOCH = 2026-01-01T00:00:00.000000Z" in lines
    assert "X = 7000.0" in lines
    assert "Y_DOT = 7.5" in lines


def test_dump_trajectory_opm_writes_covariance_block_when_available() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_covariance.yaml"))
    trajectory = propagate_local(scenario)

    payload = dump_trajectory_opm(trajectory)

    lines = payload.splitlines()
    assert "COVARIANCE_START" in lines
    assert "COV_REF_FRAME = EME2000" in lines
    assert "CX_X = 1.0" in lines
    assert "CY_Y = 1.0" in lines
    assert "CX_DOT_X_DOT = 1e-06" in lines
    assert "CZ_DOT_Z_DOT = 1e-06" in lines
    assert "COVARIANCE_STOP" in lines


def test_dump_trajectory_aem_writes_ccsds_quaternion_product() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_velocity_aligned_burn.yaml"))
    trajectory = propagate_local(scenario)

    payload = dump_trajectory_aem(trajectory)

    lines = payload.splitlines()
    attitude_sample_count = sum(sample.attitude is not None for sample in trajectory.samples)
    assert lines[0] == "CCSDS_AEM_VERS = 2.0"
    assert "ORIGINATOR = ASTRO_SUITE" in lines
    assert "META_START" in lines
    assert "OBJECT_NAME = leo-velocity-aligned-burn" in lines
    assert "CENTER_NAME = EARTH" in lines
    assert "REF_FRAME_A = EME2000" in lines
    assert "REF_FRAME_B = SC_BODY_1" in lines
    assert "TIME_SYSTEM = UTC" in lines
    assert "ATTITUDE_TYPE = QUATERNION" in lines
    assert "COMMENT quaternion_order = QC Q1 Q2 Q3" in lines
    assert "DATA_START" in lines
    assert "DATA_STOP" in lines
    assert len(
        [
            line
            for line in lines
            if line.startswith("2026-01-01T")
        ]
    ) == attitude_sample_count


def test_dump_trajectory_aem_rejects_trajectory_without_attitude_samples() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)

    try:
        dump_trajectory_aem(trajectory)
    except ValueError as exc:
        assert "attitude samples" in str(exc)
    else:
        raise AssertionError("expected AEM export to require attitude samples")


def test_load_trajectory_aem_attaches_attitude_to_base_trajectory() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_velocity_aligned_burn.yaml"))
    trajectory = propagate_local(scenario)
    payload = dump_trajectory_aem(trajectory)
    base_trajectory = trajectory.model_copy(
        update={
            "samples": [
                sample.model_copy(update={"attitude": None})
                for sample in trajectory.samples
            ],
            "metadata": {**trajectory.metadata, "attitude_model": "stripped-for-test"},
        }
    )

    loaded = load_trajectory_aem(payload, base_trajectory=base_trajectory)

    attitude_samples = [sample for sample in trajectory.samples if sample.attitude is not None]
    loaded_attitude_samples = [sample for sample in loaded.samples if sample.attitude is not None]
    assert [sample.epoch for sample in loaded.samples] == [
        sample.epoch for sample in trajectory.samples
    ]
    assert [sample.state for sample in loaded.samples] == [
        sample.state for sample in trajectory.samples
    ]
    assert len(loaded_attitude_samples) == len(attitude_samples)
    assert loaded_attitude_samples[0].attitude is not None
    assert attitude_samples[0].attitude is not None
    assert loaded_attitude_samples[0].attitude.body_to_inertial_quaternion == (
        attitude_samples[0].attitude.body_to_inertial_quaternion
    )
    assert loaded_attitude_samples[0].attitude.frame == attitude_samples[0].attitude.frame
    assert loaded_attitude_samples[0].attitude.metadata["source_format"] == "ccsds_aem_kvn"
    assert loaded.metadata["attitude_source_format"] == "ccsds_aem_kvn"
    assert loaded.metadata["aem_time_system"] == "UTC"
    assert loaded.metadata["aem_ref_frame_a"] == "EME2000"
    assert loaded.metadata["aem_attitude_sample_count"] == len(attitude_samples)


def test_load_trajectory_aem_rejects_unmatched_attitude_epoch() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_velocity_aligned_burn.yaml"))
    trajectory = propagate_local(scenario)
    lines = dump_trajectory_aem(trajectory).splitlines()
    data_start = lines.index("DATA_START")
    lines[data_start + 1] = lines[data_start + 1].replace(
        "2026-01-01T00:00:00.000000Z",
        "2026-01-02T00:00:00.000000Z",
    )
    payload = "\n".join(lines)

    try:
        load_trajectory_aem(payload, base_trajectory=trajectory)
    except ValueError as exc:
        assert "must match base trajectory sample epochs" in str(exc)
    else:
        raise AssertionError("expected unmatched AEM epoch to fail")


def test_load_trajectory_oem_round_trips_export_with_scenario_force_model() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)
    payload = dump_trajectory_oem(trajectory)

    loaded = load_trajectory_oem(payload, force_model=scenario.force_model)

    assert loaded.scenario_id == trajectory.scenario_id
    assert loaded.backend == trajectory.backend
    assert loaded.force_model == scenario.force_model
    assert [sample.epoch for sample in loaded.samples] == [
        sample.epoch for sample in trajectory.samples
    ]
    assert [sample.state for sample in loaded.samples] == [
        sample.state for sample in trajectory.samples
    ]
    assert {sample.mass_kg for sample in loaded.samples} == {None}
    assert loaded.metadata["source_format"] == "ccsds_oem_kvn"
    assert loaded.metadata["oem_time_system"] == "UTC"
    assert loaded.metadata["oem_ref_frame"] == "EME2000"


def test_load_trajectory_opm_round_trips_first_sample_with_scenario_force_model() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)
    payload = dump_trajectory_opm(trajectory)

    loaded = load_trajectory_opm(payload, force_model=scenario.force_model)

    assert loaded.scenario_id == trajectory.scenario_id
    assert loaded.backend == trajectory.backend
    assert loaded.force_model == scenario.force_model
    assert len(loaded.samples) == 1
    assert loaded.samples[0].epoch == trajectory.samples[0].epoch
    assert loaded.samples[0].state == trajectory.samples[0].state
    assert loaded.metadata["source_format"] == "ccsds_opm_kvn"
    assert loaded.metadata["opm_time_system"] == "UTC"
    assert loaded.metadata["opm_ref_frame"] == "EME2000"
    assert loaded.metadata["opm_state_units"] == "km_and_km_per_s"


def test_load_trajectory_opm_imports_covariance_block() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    payload = Path("examples/trajectories/leo_initial_state.opm").read_text(encoding="utf-8")

    loaded = load_trajectory_opm(payload, force_model=scenario.force_model)

    assert len(loaded.covariance_history) == 1
    covariance_sample = loaded.covariance_history[0]
    assert covariance_sample.epoch == loaded.samples[0].epoch
    assert covariance_sample.covariance[0][0] == 1.0
    assert covariance_sample.covariance[3][3] == 0.000001
    assert covariance_sample.state_transition_matrix is not None
    assert covariance_sample.state_transition_matrix[0][0] == 1.0
    assert covariance_sample.metadata["source_format"] == "ccsds_opm_kvn"
    assert covariance_sample.metadata["covariance_model"] == "imported_opm_single_epoch"
    assert loaded.metadata["opm_covariance_sample_count"] == 1


def test_load_trajectory_opm_rejects_incomplete_covariance_block() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    payload = Path("examples/trajectories/leo_initial_state.opm").read_text(encoding="utf-8")
    payload = payload.replace("CZ_DOT_Z_DOT = 0.000001", "")

    try:
        load_trajectory_opm(payload, force_model=scenario.force_model)
    except ValueError as exc:
        assert "CZ_DOT_Z_DOT" in str(exc)
    else:
        raise AssertionError("expected incomplete OPM covariance block to fail")


def test_load_trajectory_opm_rejects_missing_state_key() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)
    payload = dump_trajectory_opm(trajectory).replace("Z_DOT = 1.0", "")

    try:
        load_trajectory_opm(payload, force_model=scenario.force_model)
    except ValueError as exc:
        assert "Z_DOT" in str(exc)
    else:
        raise AssertionError("expected missing OPM state key to fail")


def test_load_trajectory_oem_rejects_unsupported_frame() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)
    payload = dump_trajectory_oem(trajectory).replace("REF_FRAME = EME2000", "REF_FRAME = ITRF")

    try:
        load_trajectory_oem(payload, force_model=scenario.force_model)
    except ValueError as exc:
        assert "REF_FRAME" in str(exc)
    else:
        raise AssertionError("expected unsupported OEM frame to fail")
