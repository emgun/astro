def test_packages_import() -> None:
    import astro_backends
    import astro_cli
    import astro_core
    import astro_dynamics
    import astro_od

    assert astro_core.__all__ == []
    assert astro_dynamics.__all__ == []
    assert astro_od.__all__ == []
    assert astro_backends.__all__ == []
    assert astro_cli.__all__ == []
