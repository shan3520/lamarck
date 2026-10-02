import pytest

from lamarck.harness import Edit, EditConflict, Harness


def seed():
    return Harness({"system.md": "Be careful.\n", "rules.md": "- Rule one.\n"})


def test_additive_edits_and_render():
    h = seed().apply([Edit("add_skill", "skills/verify.md", "Check the output."), Edit("append_rule", "rules.md", "Rule two.")])
    out = h.render()
    assert "Be careful." in out
    assert "- Rule one.\n- Rule two." in out
    assert "### verify\nCheck the output." in out


def test_append_rule_is_idempotent():
    e = Edit("append_rule", "rules.md", "Rule two.")
    assert seed().apply([e, e]).components["rules.md"].count("Rule two.") == 1


def test_fingerprint_changes_with_content_only():
    assert seed().fingerprint() == seed().fingerprint()
    assert seed().fingerprint() != seed().apply([Edit("append_rule", "rules.md", "x")]).fingerprint()


def test_invalid_edits_rejected():
    with pytest.raises(ValueError):
        Edit("delete_everything", "rules.md")
    with pytest.raises(ValueError):
        Edit("add_skill", "rules.md", "not a skill path")


def test_merge_unions_additive_edits_and_detects_conflicts():
    a = [Edit("add_skill", "skills/a.md", "A")]
    b = [Edit("add_skill", "skills/b.md", "B"), Edit("add_skill", "skills/a.md", "A")]
    merged = Harness.check_mergeable([a, b])
    assert [e.target for e in merged] == ["skills/a.md", "skills/b.md"]
    with pytest.raises(EditConflict):
        Harness.check_mergeable([[Edit("set_component", "system.md", "X")], [Edit("set_component", "system.md", "Y")]])
    with pytest.raises(EditConflict):
        Harness.check_mergeable([[Edit("add_skill", "skills/a.md", "A")], [Edit("add_skill", "skills/a.md", "different")]])


def test_save_load_roundtrip(tmp_path):
    h = seed().apply([Edit("add_skill", "skills/verify.md", "Check.")])
    h.save(tmp_path / "h")
    assert Harness.load(tmp_path / "h").components == h.components


def test_diff_shows_added_skill():
    d = seed().diff(seed().apply([Edit("add_skill", "skills/verify.md", "Check.")]))
    assert "+### verify" in d
