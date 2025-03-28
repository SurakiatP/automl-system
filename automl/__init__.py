# __init__.py

"""
This file marks the `automl` directory as a Python package.
You can optionally expose key modules or helpers here.
"""

__version__ = "0.1.0"

# Optional: expose core functions/classes
from .pipeline import run_pipeline
from .config import load_config
