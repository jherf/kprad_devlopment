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
    from kpradpy import globals as g
    for name in ("CRETIN_PATH", "OUTPUT_DIR", "DEFAULT_CONFIG_PATH"):
        assert "/Users/" not in str(getattr(g, name)), name
