# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import subprocess
import inspect
import sys
from pygit2 import Repository
sys.path.insert(0, '..')
import auto_LiRPA

subprocess.run(['python', 'process.py'])

# -- Project information -----------------------------------------------------

project = 'auto_LiRPA'
author = '<a href="https://github.com/KaidiXu/auto_LiRPA#developers-and-copyright">auto-LiRPA authors</a>'
copyright = f'2021, {author}'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.linkcode',
    'm2r2',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'src', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

repo = Repository('../')
branch = repo.head.shorthand

# Resolve function for the linkcode extension.
def linkcode_resolve(domain, info):
    def find_source():
        obj = auto_LiRPA
        parts = info['fullname'].split('.')
        if info['module'].endswith(f'.{parts[0]}'):
            module = info['module'][:-len(parts[0])-1]
        else:
            module = info['module']
        obj = sys.modules[module]
        for part in parts:
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    fn, lineno_start, lineno_end = find_source()
    filename = f'{fn}#L{lineno_start}-L{lineno_end}'

    return f"https://github.com/KaidiXu/auto_LiRPA/blob/{branch}/doc/{filename}"
