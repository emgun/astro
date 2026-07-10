def test_astro_mission_public_imports() -> None:
    import astro_mission

    assert astro_mission.__all__ == [
        "DeorbitPhaseConfig",
        "MissionLifecycleResult",
        "MissionLifecycleScenario",
        "OrbitPhaseConfig",
        "run_mission_lifecycle",
    ]
