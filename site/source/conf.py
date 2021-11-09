# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

from logging import fatal
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'Datumaro API documentation'
copyright = '2021, Intel'
author = 'Intel'

# The full version, including alpha/beta/rc tags
release = '0.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.intersphinx', # Link to other projects documentation
    'sphinx.ext.viewcode', # Find the source files
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates',
    'docs',
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_themes', ]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = ['_themes', ]
html_theme_options = {
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#30638E',
    # Toc options
    'titles_only': False,
    'display_version': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css' ]

# -- Extension configuration -------------------------------------------------
autodoc_docstring_signature = True
autodoc_member_order = 'bysource'

def skip_member(app, what, name, obj, skip, options):
    if 'undoc-members' in options:
        return name.startswith('_')

def replace(app, what, name, obj, options, lines):
    for i in range(len(lines)):
        if not "'|n'" in lines[i]:
            if not "'|s'" in lines[i]:
                lines[i] = lines[i].replace("|n", "\n").replace("|s", "  ")

def setup(app):
    app.connect('autodoc-skip-member', skip_member)
    app.connect('autodoc-process-docstring', replace)
