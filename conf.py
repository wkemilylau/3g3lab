import os
import sys
sys.path.insert(0, os.path.abspath('.'))

project = '3g3lab'
copyright = '2021, Ta-Chu Kao'
author = 'Ta-Chu Kao'

extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode", "numpydoc"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

exclude_patterns = ["_build", ".DS_Store"]

html_theme = 'sphinx_rtd_theme'

#html_static_path = ['_static']