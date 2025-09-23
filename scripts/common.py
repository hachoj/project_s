import os
import sys


def bootstrap_project_root():
    """Ensure project root is on sys.path and return it.

    Assumes this file lives in the `scripts/` directory directly under the
    project root. Safe to call multiple times.
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def get_device(log: bool = True):
    """Pick CUDA if available, otherwise CPU. Optionally logs selection."""
    import torch  # local import to avoid hard dependency at import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log:
        print(f"Using device: {device}")
    return device

