from astro_backends.orekit.smoke import OrekitSmokeResult, run_orekit_smoke


def test_run_orekit_smoke_reports_forced_unavailable_without_importing_wrapper() -> None:
    result = run_orekit_smoke(strict=False, force_unavailable=True)

    assert isinstance(result, OrekitSmokeResult)
    assert result.available is False
    assert result.wrapper == "orekit_jpype"
    assert result.version is None
    assert "not installed" in result.message
    assert result.to_dict() == {
        "available": False,
        "wrapper": "orekit_jpype",
        "version": None,
        "message": result.message,
    }
