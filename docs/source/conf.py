"""Sphinx configuration for the cvzx documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "cvzx"
copyright = "2026, the cvzx contributors"  # noqa: A001
author = "cvzx contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

# `CVZXGraph` is deliberately defined once per graph backend
# (`cvzx.backends.nx.graph.CVZXGraph`, `cvzx.backends.rx.graph.CVZXGraph`) -- mirrored
# implementations of the same schema, not a naming collision to fix. Every bare `CVZXGraph`
# cross-reference elsewhere in the docs is genuinely ambiguous between the two on purpose, so
# this warning class is suppressed rather than chasing down and fully qualifying every mention.
suppress_warnings = ["ref.python"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "cvzx"
