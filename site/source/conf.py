# Copyright (C) 2021-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import re
import sys

sys.path.insert(0, os.path.abspath("../.."))

from datumaro.version import VERSION

# -- Project information -----------------------------------------------------

project = "Datumaro"
copyright = "2023, Datumaro Contributors"
author = "Datumaro Contributors"

# The full version, including alpha/beta/rc tags
release = VERSION


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.viewcode",  # Find the source files
    "sphinx_copybutton",  # Copy buttons for code blocks
    "sphinx.ext.autosectionlabel",  # Refer sections its title
    "sphinx.ext.intersphinx",  # Generate links to the documentation
    # of objects in external projects
    "sphinxcontrib.mermaid",  # allows Mermaid graphs
]

# Add any paths that contain templates here, relative to this directory.
templates_path = [
    "_templates",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme = "pydata_sphinx_theme"
html_static_path = [
    "_static",
]

html_theme_options = {
   "navbar_center": [],
   "logo": {
      "image_light": "datumaro-logo.png",
      "image_dark": "datumaro-logo.png",
   }
}
html_css_files = [
    "css/custom.css",
]

# -- Extension configuration -------------------------------------------------
autodoc_docstring_signature = True
autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

nitpick_ignore_regex = [
    ("py:class", r"^(.*[\s\"(\._)]+.*)+$"),  # Hiding warnings contain " ", """ or "._"
    ("py:class", ""),
]

# Members to be included.
include_members_list = [
    "__init__",
    "__iter__",
    "__eq__",
    "__len__",
    "__contains__",
    "__getitem__",
]


def skip_member(app, what, name, obj, skip, options):
    if all(name != a for a in include_members_list):
        return name.startswith("_")


def replace(app, what, name, obj, options, lines):
    exclude_plugins_name = [
        "transform",
        "extractor",
        "converter",
        "launcher",
        "importer",
        "validator",
    ]
    names = re.sub(r"([A-Z])", r" \1", name.replace("_", "").split(".")[-1]).split()
    for n, a in enumerate(names):
        if len(a) != 1:
            for b in exclude_plugins_name:
                if a.lower() == b:
                    names.pop(n)
    if all(1 == len(a) for a in names):
        prog_name = "".join(names).lower()
    else:
        prog_name = "_".join(names).lower()
    for i, line in enumerate(lines):
        if line:
            prog = str("%(prog)s")
            lines[i] = lines[i].replace(prog, prog_name)
            lines[i] = lines[i].replace("'frame_'", r"'frame\_'")  # fix unwanted link
            if not "'|n'" in lines[i]:
                if not "'|s'" in lines[i]:
                    lines[i] = lines[i].replace("|n", "\n").replace("|s", " ")


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
    app.connect("autodoc-process-docstring", replace)