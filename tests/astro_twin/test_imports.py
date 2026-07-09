def test_astro_twin_public_imports() -> None:
    import astro_twin

    assert astro_twin.__all__ == [
        "DigitalTwinScenario",
        "DigitalTwinResult",
        "run_digital_twin",
    ]
