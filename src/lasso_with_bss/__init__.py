"""
lasso_with_bss package

Bootstrap stability selection (BSS) for Lasso regression.
"""

__version__ = "0.1.0"

from .lasso import bootstrap_lasso
from .io import load_folder_as_df

__all__ = ["bootstrap_lasso", "load_folder_as_df", "__version__"]
