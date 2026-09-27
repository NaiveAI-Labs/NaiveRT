"""Public Naive N names resolve to checkpoint pairs; nothing user-facing is a lineage name."""
import pytest

from naivert.models.registry import DEFAULT_MODEL, MODELS, SERIES, resolve_model


def test_default_is_registered_and_public():
    m = resolve_model(None)
    assert m.key == DEFAULT_MODEL and m.key in MODELS
    assert m.key.startswith("naive-n") and m.display.startswith("Naive N") and m.series == SERIES


def test_every_entry_names_both_checkpoints_and_its_lineage():
    for key, m in MODELS.items():
        assert m.key == key and m.target and m.draft and m.lineage
        assert key.startswith("naive-n") and m.display.startswith("Naive N")


def test_unknown_name_lists_the_known_ones():
    with pytest.raises(KeyError, match="naive-n0.5"):
        resolve_model("naive-n9")
