def test_astro_assistant_imports() -> None:
    import astro_assistant

    assert astro_assistant.__all__ == [
        "AstroWorkflowPlan",
        "WorkflowStep",
        "WorkflowTrace",
    ]
