from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_driver(name):
    """Import an e2e driver script by path (they define constants and helpers at module level)."""
    path = ROOT / "tests/gpu/naive_n0p5_e2e" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
