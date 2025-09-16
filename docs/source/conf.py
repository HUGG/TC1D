# Configuration file for the Sphinx documentation builder.

# -- Path setup
from pathlib import Path
import sys

sys.path.insert(0, str(Path("..", "..", "src").resolve()))

# -- Project information

project = "TC1D"
copyright = "2022-2025, David Whipp, University of Helsinki"
author = "David Whipp"

release = ""
version = ""

# -- General configuration

extensions = [
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
]

# intersphinx_mapping = {
#    'python': ('https://docs.python.org/3/', None),
#    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
# }
# intersphinx_disabled_domains = ['std']

templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output

html_theme = "sphinx_book_theme"

# HTML theme options
html_theme_options = {
    # "external_links": [],
    "repository_url": "https://github.com/HUGG/tc1d/",
    "repository_branch": "main",
    "path_to_docs": "docs/source/",
    # "twitter_url": "https://twitter.com/pythongis",
    # "google_analytics_id": "UA-159257488-1",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "collapse_navigation": False,
    },
}

# The master toctree document.
master_doc = "index"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Do not execute cells
# jupyter_execute_notebooks = "off"

# Allow errors
# nb_execution_allow_errors = True

# Allow myst admonition style
myst_admonition_enable = True

# Define level for myst heading implicit anchors
myst_heading_anchors = 3

# -- Options for EPUB output
epub_show_urls = "footnote"

# Enable math config options
myst_enable_extensions = ["dollarmath"]

# MathJax config
mathjax3_config = {
    "loader": {"load": ["[tex]/upgreek"]},
    "tex": {"packages": {"[+]": ["upgreek"]}},
}

# Use bibtex for citations
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# Autodoc mock imports
autodoc_mock_imports = [
    "corner",
    "emcee",
    "matplotlib",
    "neighpy",
    "numpy",
    "schwimmbad",
    "scipy",
    "sklearn",
]
