# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Flowcam'
copyright = '2023, Ather Abbas'
author = 'Ather Abbas'

import os
from sphinx_gallery.sorting import ExampleTitleSortKey, FileNameSortKey

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
'sphinx.ext.autodoc',
'sphinx_toggleprompt',
'sphinx_copybutton',
"sphinx_gallery.gen_gallery",
]

templates_path = ['_templates']
exclude_patterns = []

sphinx_gallery_conf = {
    'backreferences_dir': 'gen_modules/backreferences',
    #'doc_module': ('sphinx_gallery', 'numpy'),
    'reference_url': {
        'sphinx_gallery': None,
    },
    'examples_dirs': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'scripts'),
    'gallery_dirs': 'auto_examples',
    'compress_images': ('images', 'thumbnails'),
    'filename_pattern': '',
    #'ignore_pattern': 'utils.py',

    #'show_memory': True,
    #'junit': os.path.join('sphinx-gallery', 'junit-results.xml'),
    # capture raw HTML or, if not present, __repr__ of last expression in
    # each code block
    'capture_repr': ('_repr_html_', '__repr__'),
    'matplotlib_animations': True,
    'image_srcset': ["2x"],
    'within_subsection_order': ExampleTitleSortKey,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
