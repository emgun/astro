from pathlib import Path

from astro_core.io import load_scenario
from astro_dynamics.ephemeris import (
    dump_trajectory_aem,
    dump_trajectory_ephemeris_csv,
    dump_trajectory_oem,
    load_trajectory_oem,
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
