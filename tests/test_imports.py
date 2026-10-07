"""Every module must import cleanly on a bare clone (no external data)."""

import importlib
import pytest

MODULES = [
    "kpradpy",
    "kpradpy.globals",
    "kpradpy.util.constants",
    "kpradpy.util.physics",
    "kpradpy.util.layout",
    "kpradpy.util.solver",
    "kpradpy.util.injectors",
    "kpradpy.util.util_injectors",
    "kpradpy.util.postprocess",
    "kpradpy.util.config",
    "kpradpy.util.profile",
    "kpradpy.util.equilibrium",
    "kpradpy.util.read_rate_nLTE",
    "kpradpy.util.atomic_cretin",
    "kpradpy.util.plotting",
    "kpradpy.main_script",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_no_stale_package_references():
    """Guard against the kprad -> main -> kpradpy rename leaving imports behind."""
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parent.parent / "kpradpy"
    bad = []
    for py in root.rglob("*.py"):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            if re.search(r"^\s*(from|import)\s+(main|kprad)\.", line):
                bad.append(f"{py.relative_to(root)}:{i}: {line.strip()}")
    assert not bad, "stale imports:\n" + "\n".join(bad)


def test_default_paths_are_not_machine_specific():
    """No hardcoded home-directory paths, and with no KPRAD_* variables set,
    every default path lies inside the repository.

    (An earlier version asserted "/Users/" not in each path, which fails on
    any Mac because the repo itself lives under /Users/.)
    """
    import os
    import re
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent

    # 1. Source code: no literal home-directory paths
    bad = [
        f"{py.relative_to(root)}:{i}"
        for py in (root / "kpradpy").rglob("*.py")
        for i, line in enumerate(py.read_text().splitlines(), 1)
        if re.search(r"['\"](/Users/|/home/)", line)
    ]
    assert not bad, "hardcoded home paths:\n" + "\n".join(bad)

    # 2. Defaults: fresh interpreter, KPRAD_* variables removed
    env = {k: v for k, v in os.environ.items() if not k.startswith("KPRAD_")}
    names = ("CRETIN_PATH", "OUTPUT_DIR", "DEFAULT_CONFIG_PATH")
    code = "from kpradpy import globals as g; " + "; ".join(
        f"print(g.{n})" for n in ("REPO_ROOT",) + names
    )
    out = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    repo, *paths = out.splitlines()
    for name, path in zip(names, paths):
        assert path.startswith(repo), f"{name} = {path} is outside the repo"
