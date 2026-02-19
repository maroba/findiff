"""Sphinx configuration for findiff documentation."""

import datetime
import os
import sys

from findiff import __version__

sys.path.insert(0, os.path.abspath("../.."))

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "nbsphinx",
]

nbsphinx_execute = "always"
nbsphinx_kernel_name = "python3"

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = "findiff"
copyright = f"2018-{datetime.datetime.now().year}, Matthias Baer"
author = "Matthias Baer"

version = __version__
release = __version__

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

pygments_style = "sphinx"
highlight_language = "python"

autoclass_content = "both"

# Suppress warnings from informal type annotations in docstrings
nitpicky = False

# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for LaTeX output ---------------------------------------------

latex_documents = [
    (master_doc, "findiff.tex", "findiff Documentation", "Matthias Baer", "manual"),
]

# -- Options for manual page output ---------------------------------------

man_pages = [(master_doc, "findiff", "findiff Documentation", [author], 1)]

# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "findiff",
        "findiff Documentation",
        author,
        "findiff",
        "A Python package for finite difference numerical derivatives and partial differential equations.",
        "Scientific/Engineering",
    ),
]
