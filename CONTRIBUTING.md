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

## Code of conduct

Participation is governed by the [Code of Conduct](https://github.com/soumickmj/scINTILLA/blob/master/CODE_OF_CONDUCT.md).
