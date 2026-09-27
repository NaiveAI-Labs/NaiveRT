from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_package_import_does_not_eagerly_load_backends() -> None:
    command = (
        "import json, sys; import naivert.runtime; "
        "print(json.dumps(sorted(name for name in sys.modules "
        "if name.startswith('naivert.runtime.'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []
