import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../python"))

project = "AstroFlow"
author = "lintian233"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.12.dev3"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

html_theme = "alabaster"
html_static_path = ["_static"]
html_logo = None
html_title = "AstroFlow documentation"
html_theme_options = {
    "description": "GPU-accelerated single-pulse and FRB search pipeline",
    "github_user": "lintian233",
    "github_repo": "astroflow",
    "github_button": True,
    "github_type": "star",
    "show_powered_by": True,
    "show_related": True,
    "fixed_sidebar": True,
    "page_width": "1320px",
    "sidebar_width": "300px",
    "font_family": "'Times New Roman', Times, serif",
    "head_font_family": "'Times New Roman', Times, serif",
    "code_font_family": "'SFMono-Regular', Consolas, 'Liberation Mono', monospace",
}
html_sidebars = {
    "**": [
        "about.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}
html_css_files = ["astroflow.css"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

autodoc_mock_imports = [
    "astropy",
    "cv2",
    "loguru",
    "matplotlib",
    "numba",
    "numpy",
    "pandas",
    "scipy",
    "seaborn",
    "sympy",
    "torch",
    "torchvision",
    "tqdm",
    "ultralytics",
    "your",
    "astroflow._astroflow_core",
]
