"""Sphinx configuration for the scintilla-py documentation."""

import os
from datetime import datetime

import scintilla

# -- Project information -----------------------------------------------------

project = "scintilla-py"
author = "Sina Kanannejad, Noemi Bongiorni, Elisa Nordera, Sara Redaelli, Irene Rusconi, Rachele Zanin, Alice Giustacchini, Soumick Chatterjee"
copyright = f"{datetime.now():%Y}, {author}"  # noqa: A001
release = scintilla.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {"members": False}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
typehints_defaults = "braces-after"
always_document_param_types = False

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "jupyter_execute"]

# The long-form guides were written as plain Markdown with skipped heading levels.
suppress_warnings = ["myst.header"]
myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath", "deflist"]

nb_execution_mode = "cache"
nb_execution_timeout = 600
nb_execution_raise_on_error = True
nb_execution_show_tb = True

# Cross-links to the scverse libraries. Set SCINTILLA_DOCS_OFFLINE=1 to build without network access.
if os.environ.get("SCINTILLA_DOCS_OFFLINE"):
    intersphinx_mapping = {}
else:
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
        "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
        "mudata": ("https://mudata.readthedocs.io/en/stable/", None),
        "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
        "sklearn": ("https://scikit-learn.org/stable/", None),
        "matplotlib": ("https://matplotlib.org/stable/", None),
    }

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = f"scintilla-py {version}"
html_theme_options = {
    "repository_url": "https://github.com/soumickmj/scINTILLA",
    "repository_branch": "master",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
}
