# Releasing scINTILLA

The PyPI distribution is **`scintilla-py`**; the import name and the command-line tool stay
`scintilla`. The existing PyPI project called `scintilla` is unrelated, and changing the
capitalisation does not make that name available.

Releases are published by the `release` GitHub workflow with **PyPI trusted publishing**, so no
API token is stored anywhere.

## One-off set-up

1. On PyPI, open the `scintilla-py` project, then *Publishing*, and add a trusted publisher:
   owner `soumickmj`, repository `scINTILLA`, workflow `release.yaml`, environment `pypi`.
2. In the GitHub repository, create an environment called `pypi` (optionally with required reviewers).
3. Enable Read the Docs for the repository (it reads `.readthedocs.yaml`), and add the repository to
   Codecov (put its token in the `CODECOV_TOKEN` secret).

## Making a release

1. Make sure `CHANGELOG.md` has a section for the new version and that CI is green on `master`.
2. Set the version in `src/scintilla/__init__.py` (the only place it is defined) and run
   `uv lock` to refresh `uv.lock`.
3. Check the artefacts locally:

   ```bash
   uv build --no-sources
   uvx twine check --strict dist/*
   uv run --with dist/*.whl --no-project python -c "import scintilla; print(scintilla.__version__)"
   ```

4. Tag and publish a GitHub release (`v0.2.0`). Publishing the release triggers the workflow,
   which builds the sdist and wheel, runs `twine check` and uploads them to PyPI.
5. Archive the release on Zenodo (enable the GitHub integration once) and add the software DOI to
   `CITATION.cff`.

A published file name can never be reused on PyPI, so correct mistakes with a new version, not by
re-uploading.

## Dependency notes

* `uv.lock` reproduces the repository environment; the wheel metadata declares the ranges that users
  resolve in their own environments. Commit `uv.lock` together with every dependency change.
* The `full` extra includes Louvain's backend together with `setuptools<82`, which supplies its legacy
  `pkg_resources` import. This runtime constraint is separate from the build backend (hatchling).
* The lower bounds in `pyproject.toml` are checked in CI by installing the oldest allowed versions
  (`uv pip install --resolution lowest-direct -e ".[test]"`).
