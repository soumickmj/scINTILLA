# Contributing to scINTILLA

Thank you for considering a contribution. This project follows the
[scverse contributing guide](https://scverse.org/contributing/) in spirit; the
essentials are below.

## Set-up

```bash
git clone https://github.com/soumickmj/scINTILLA.git
cd scINTILLA
uv sync --extra test --extra full      # development environment
uv run pytest                          # fast tests (slow sweeps are deselected)
uv run pytest -m slow                  # long benchmark sweeps
uv run ruff check .                    # lint
```

Python 3.11 or newer is required. Optional backends (Leiden, Harmony, XGBoost, ...) live
in the `full` extra; tests that need one use `pytest.importorskip`, so the minimal install
must stay green.

## The API contract

Public functions follow the scanpy conventions, and new ones must as well:

* the first argument is `adata: AnnData`; everything else is keyword-only where practical;
* results are written to `.obs`, `.var`, `.obsm`, `.layers` or `.uns["scintilla"]`, under a
  name controlled by `key_added`;
* `copy=False` modifies `adata` and returns `None` (or a results object, where documented);
  `copy=True` returns a modified copy and leaves the input untouched;
* stochastic functions take `random_state` and give identical output for identical seeds;
* sparse input stays sparse unless the algorithm genuinely needs a dense array, and any
  densification is local and documented;
* no `print` in library code: use the `scintilla` logger (see `scintilla.settings.verbosity`);
* a method that fails inside a benchmark must appear in the results as failed, never
  silently disappear.

Numerical behaviour of the label-quality scorers is pinned by
`tests/test_label_quality_baseline.py`. A change that moves those numbers needs an explicit
note in the changelog.

## Pull requests

1. Branch from `master`.
2. Add tests, including a dense and a sparse case where the function touches `.X`.
3. Run `ruff check .` and `pytest`.
4. Add a line to `CHANGELOG.md` under "Unreleased".

Docstrings use the numpydoc format.

## Releasing

Only maintainers need this section. The PyPI distribution is `scintilla-py`; the import name and the
command-line tool stay `scintilla`. (The PyPI project called `scintilla` is unrelated, and changing the
capitalisation does not make that name available.) Releases are published by the `release` GitHub
workflow using PyPI trusted publishing, so no API token is stored anywhere.

**One-off set-up**

1. On PyPI, open the `scintilla-py` project, then *Manage*, then *Publishing*, and add a trusted publisher
   (GitHub): owner `soumickmj`, repository `scINTILLA`, workflow `release.yaml`, environment `pypi`. Until
   this exists the upload step fails with `invalid-publisher`.
2. In the GitHub repository, create an environment called `pypi` (optionally with required reviewers).
3. Enable Read the Docs for the repository (it reads `.readthedocs.yaml`) and add the repository to
   Codecov, putting its token in the `CODECOV_TOKEN` secret.

**Making a release**

1. Check that `CHANGELOG.md` has a section for the new version and that CI is green on `master`.
2. Set the version in `src/scintilla/__init__.py`, the only place it is defined, and run `uv lock`.
3. Check the artefacts locally:

   ```bash
   uv build --no-sources
   uvx twine check --strict dist/*
   uv run --with dist/*.whl --no-project python -c "import scintilla; print(scintilla.__version__)"
   ```

4. Tag and publish a GitHub release named after the version with a leading `v` (`v0.2.1` for 0.2.1; the
   workflow refuses any other tag). Publishing it starts the workflow, which
   builds the sdist and wheel, runs `twine check` and uploads them to PyPI.
5. Archive the release on Zenodo (enable the GitHub integration once) and add the software DOI to
   `CITATION.cff`.

A file name published to PyPI can never be reused, so correct a mistake with a new version.

The PyPI project page shows the README that was included in the uploaded release, so edits to the
README on GitHub appear on PyPI only with the next release.

**Publishing by hand.** The release workflow is the normal route. If you need to upload from your own
machine, check the artefacts as above, then use an API token scoped to the existing `scintilla-py`
project and enter it privately rather than writing it into a file:

```bash
read -rsp "PyPI API token: " UV_PUBLISH_TOKEN
export UV_PUBLISH_TOKEN
uv publish --trusted-publishing never dist/*
unset UV_PUBLISH_TOKEN
```

To rehearse on TestPyPI first, use a TestPyPI token and its upload endpoint:

```bash
uv publish --trusted-publishing never --publish-url https://test.pypi.org/legacy/ dist/*
```

**Dependencies.** `uv.lock` reproduces the repository environment, while the wheel metadata declares the
ranges that users resolve for themselves; commit the lock file with every dependency change. CI also
installs the oldest versions that `pyproject.toml` allows
(`uv pip install --resolution lowest-direct -e ".[test]"`), so a new lower bound must be one you have
actually tested.
