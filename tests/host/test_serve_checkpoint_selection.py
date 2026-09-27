"""Verify serving checkpoint selection before any engine or GPU is initialized."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("target_env,draft_env,cli,expected", [
    (None, None, [], (None, None)),
    ("", "", [], (None, None)),
    ("/mounted/target with spaces", "/mounted/draft", [],
     ("/mounted/target with spaces", "/mounted/draft")),
    ("/mounted/target", None, [], ("/mounted/target", None)),
    (None, "/mounted/draft", [], (None, "/mounted/draft")),
    ("/mounted/target", "/mounted/draft", ["--target", "/cli/target", "--draft", "/cli/draft"],
     ("/cli/target", "/cli/draft")),
    ("/mounted/target", "/mounted/draft", ["--checkpoint", "/cli/draft"],
     ("/mounted/target", "/cli/draft")),
])
def test_checkpoint_selection(monkeypatch, tmp_path, target_env, draft_env, cli, expected):
    for name, value in [("NAIVERT_TARGET_CHECKPOINT", target_env), ("NAIVERT_DRAFT_CHECKPOINT", draft_env)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("serve_checkpoint_test", ROOT / "examples/serve_sessions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = []
    def resolve(name, roots, marker):
        selected.append((name, marker))
        return tmp_path / str(len(selected))
    monkeypatch.setattr(module, "resolve", resolve)
    monkeypatch.setattr(module, "build_app", lambda *a, **kw: object())
    monkeypatch.setattr(module, "run_app", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", ["serve_sessions.py", *cli])
    module.main()
    entry = module.resolve_model(module.DEFAULT_MODEL)
    assert selected == [(expected[0] or entry.target, module.TARGET_MARK),
                        (expected[1] or entry.draft, module.DRAFT_MARK)]
