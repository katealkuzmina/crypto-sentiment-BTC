from scripts.build_notebook_skeleton import build_skeleton_notebook


def test_skeleton_has_expected_section_headers():
    nb = build_skeleton_notebook()
    markdown_sources = [c["source"] for c in nb["cells"] if c["cell_type"] == "markdown"]

    assert any(s.startswith("# Crypto News Sentiment") for s in markdown_sources)
    assert any("1. EDA" in s for s in markdown_sources)
    assert any("2. Sentiment index and returns" in s for s in markdown_sources)
    assert any("3. Hypothesis tests" in s for s in markdown_sources)
    assert any("4. Held-out validation" in s for s in markdown_sources)
    assert any("5. Write-up" in s for s in markdown_sources)


def test_skeleton_setup_cell_imports_project_modules():
    nb = build_skeleton_notebook()
    code_sources = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    setup_cell = code_sources[0]

    assert "from src.returns import" in setup_cell
    assert "from src.aggregate import" in setup_cell


def test_skeleton_is_valid_notebook_format(tmp_path):
    import nbformat as nbf

    nb = build_skeleton_notebook()
    out_path = tmp_path / "test.ipynb"
    with out_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

    reloaded = nbf.read(out_path, as_version=4)
    nbf.validate(reloaded)
