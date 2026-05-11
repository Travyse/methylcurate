def test_find_common_cpgs_returns_set():
    """Verify find_common_cpgs is annotated to return set, not None."""
    import inspect

    from methylcurate.tools.qc.feature_selection import find_common_cpgs

    sig = inspect.signature(find_common_cpgs)
    annotation = str(sig.return_annotation)
    assert "set" in annotation
    assert "None" not in annotation
