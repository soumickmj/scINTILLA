# Publishing scINTILLA to PyPI

The PyPI distribution name is **`scintilla-py`**, configured in
`pyproject.toml`. Python imports and the CLI remain `scintilla`.
The existing PyPI project named `scintilla` belongs to an unrelated project;
changing capitalisation does not make that name available.

## Release preparation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
work from the root of a clean checkout:

```bash
uv sync --locked --python 3.11
uv run --locked pytest
uv run --locked ruff check --select F821,F822,F823,E9 scintilla tests
uv build --no-sources --out-dir dist/0.1.0
uv run --locked python -m twine check --strict dist/0.1.0/*
uv publish --dry-run --trusted-publishing never dist/0.1.0/*
```

The first release retains version `0.1.0`. For later releases, update both
`[project].version` in `pyproject.toml` and `scintilla.__version__` in
`scintilla/__init__.py`, run `uv lock`, and replace `0.1.0` in the commands
with the new version. Version-specific output directories keep old release
artifacts out of the upload. Commit `uv.lock` alongside dependency changes.

The lockfile reproduces repository environments; wheel metadata declares
the requirements that PyPI users resolve in their own environments.
The `dev` dependency group supplies release checks and tests and is not a
runtime dependency of the published package. `full` remains an optional
extra.

The `full` extra includes Louvain's backend and `setuptools<82`, which
supplies its legacy `pkg_resources` import. This runtime constraint is
separate from the isolated setuptools build backend.

## Local publication with an API token

Create a [PyPI account](https://pypi.org/account/register/) and an
[API token](https://pypi.org/help/#apitoken). The token used for a first
upload must permit creating the new project; after the project exists,
use a project-scoped token for subsequent releases.

Enter the token privately in your Bash terminal, then upload the checked
artifacts:

```bash
read -rsp "PyPI API token: " UV_PUBLISH_TOKEN
export UV_PUBLISH_TOKEN
uv publish --trusted-publishing never dist/0.1.0/*
unset UV_PUBLISH_TOKEN
```

Store the token outside the repository. Uploading creates the PyPI project
under the account that owns the token. Confirm the distribution name and
release version before this step: PyPI release filenames cannot be reused
for a different build.

## TestPyPI

TestPyPI has separate accounts, tokens and project names. Upload the same
checked artifacts using a TestPyPI token and its upload endpoint:

```bash
read -rsp "TestPyPI API token: " UV_PUBLISH_TOKEN
export UV_PUBLISH_TOKEN
uv publish --trusted-publishing never \
    --publish-url https://test.pypi.org/legacy/ dist/0.1.0/*
unset UV_PUBLISH_TOKEN
```

For installation testing, download your project's wheel from its
TestPyPI project page and install that local wheel with `uv pip install`
inside a fresh `uv venv`. Dependencies then resolve from normal PyPI.

## Release contents

The source archive includes README, licence, user guide, label-quality
guide, `uv.lock` and these publishing instructions. `scverse_plan.md` is deliberately
excluded from release archives.

References: [uv package publishing](https://docs.astral.sh/uv/guides/package/),
[PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/).
