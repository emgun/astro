from pathlib import Path

from astro_core.io import load_scenario
from astro_dynamics.ephemeris import dump_trajectory_ephemeris_csv, dump_trajectory_oem
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
