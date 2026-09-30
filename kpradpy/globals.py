"""Package-wide paths, resolved from the environment with sensible defaults.

Nothing in here is machine-specific. Every location can be overridden with
an environment variable, and every default is relative to the repository so
a fresh clone runs without editing any file.

    KPRAD_DATA      root of external data (CRETIN tables, gfiles, .mat refs)
                    default: <repo>/data
    KPRAD_OUTPUT    where run results and figures are written
                    default: <repo>/output
    KPRAD_CRETIN    CRETIN rate-file template; overrides the KPRAD_DATA layout
                    default: $KPRAD_DATA/cretin/{el}_rates_CRETIN/{el}_rate_ne{ne}_Ta{ta}.dat
    KPRAD_CONFIG    default YAML used when main() is called with no argument
                    default: <repo>/configs/180016_SPI.yaml

Expected layout under KPRAD_DATA (nothing is required until a run asks for it):

    data/
    ├── cretin/<El>_rates_CRETIN/<El>_rate_ne<i>_Ta<j>.dat
    ├── gfiles/g<shot>.<time>
    └── matlab/Jeffs_kprad.mat
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# ---- User-created input configurations
KPRAD_INPUTS_DIRECTORY: Path = REPO_ROOT / "configs"

# ---- External data
KPRAD_DATA: Path = Path(os.environ.get("KPRAD_DATA", REPO_ROOT / "data")).expanduser()
GFILE_DIR: Path = KPRAD_DATA / "gfiles"
MATLAB_DIR: Path = KPRAD_DATA / "matlab"

CRETIN_PATH: str = os.environ.get(
    "KPRAD_CRETIN",
    str(KPRAD_DATA / "cretin" / "{el}_rates_CRETIN" / "{el}_rate_ne{ne}_Ta{ta}.dat"),
)

# ---- Outputs
OUTPUT_DIR: str = str(
    Path(os.environ.get("KPRAD_OUTPUT", REPO_ROOT / "output")).expanduser()
)

# ---- Default run
DEFAULT_CONFIG_PATH: str = os.environ.get(
    "KPRAD_CONFIG", str(KPRAD_INPUTS_DIRECTORY / "180016_SPI.yaml")
)

# Kept for backward compatibility with code that imported the old name.
KPRAD_PARENT_DIRECTORY: Path = REPO_ROOT


def resolve_data_path(path: str | os.PathLike | None) -> str | None:
    """Turn a config-file path into an absolute one.

    Absolute paths and ``~`` are honoured as given. A relative path is taken
    relative to ``KPRAD_DATA``, so configs can say ``gfiles/g206990.01480``
    and work on any machine that sets that one variable. ``$VAR`` and
    ``${VAR}`` references are expanded.
    """
    if path is None:
        return None
    p = Path(os.path.expandvars(str(path))).expanduser()
    if not p.is_absolute():
        p = KPRAD_DATA / p
    return str(p)


__all__ = [
    "REPO_ROOT",
    "KPRAD_INPUTS_DIRECTORY",
    "KPRAD_DATA",
    "GFILE_DIR",
    "MATLAB_DIR",
    "CRETIN_PATH",
    "OUTPUT_DIR",
    "DEFAULT_CONFIG_PATH",
    "KPRAD_PARENT_DIRECTORY",
    "resolve_data_path",
]
