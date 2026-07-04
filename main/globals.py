"""
Package-wide path constants and configuration

Override the data root at runtime via the EMIS3D_ROOT environment variable:
    export EMIS3D_ROOT=/path/to/your/emis3d_data

"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Root of the user data tree. Override with EMIS3D_ROOT to decouple data
# from the source checkout (required when installed via pip)
KPRAD_PARENT_DIRECTORY: Path = Path(os.environ.get("KPRAD_DEVLOPMENT_ROOT", _REPO_ROOT))

# User-created input data (equilibria, radDists, run configs, etc.)
KPRAD_INPUTS_DIRECTORY: Path = KPRAD_PARENT_DIRECTORY / "configs"


# --- Bad user-specific directorys here
CRETIN_PATH = "/Users/plh/research/cretin/{el}_rates_CRETIN/{el}_rate_ne{ne}_Ta{ta}.dat"
OUTPUT_DIR = "/Users/plh/research/kprad/"
DEFAULT_CONFIG_PATH = (
    "/Users/plh/Documents/git/kprad_devlopment/configs/206990_PELLET.yaml"
)


# Explicit export list
__all__ = [
    "KPRAD_PARENT_DIRECTORY",
    "KPRAD_INPUTS_DIRECTORY",
    "CRETIN_PATH",
    "OUTPUT_DIR",
    "DEFAULT_CONFIG_PATH",
]
