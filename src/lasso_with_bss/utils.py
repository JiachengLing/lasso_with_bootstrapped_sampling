# src/lasso_lab/utils.py
# General small functions

import os
import random
import numpy as np

def set_seed(seed: int = 42):
    """unified random seeds"""
    import sklearn
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def ensure_dir(path: str):
    """ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def pretty_dict(d: dict) -> str:
    return "\n".join(f"{k:>12}: {v}" for k, v in d.items())