import sys

import pytest

from lamarck.tasks import HELDOUT, TRAIN, Task, TaskSuite, task_specific_literals, verify


def t(verifier, **kw):
    return Task(id=kw.pop("id", "t1"), prompt=kw.pop("prompt", "p"), verifier=verifier, **kw)


@pytest.mark.parametrize(
    "verifier,output,ok",
    [
        ({"type": "exact", "value": "42"}, " 42\n", True),
        ({"type": "exact", "value": "42"}, "43", False),
        ({"type": "contains", "values": ["a", "b"]}, "a b", True),
        ({"type": "contains", "values": ["a", "z"]}, "a b", False),
        ({"type": "not_contains", "values": ["guarantee"]}, "we guarantee", False),
        ({"type": "regex", "pattern": r"Ticket #\d{5}\b"}, "Re: Ticket #12345", True),
        ({"type": "regex", "pattern": r"Ticket #\d{5}\b"}, "ticket 12345", False),
        ({"type": "json", "required_keys": ["a"]}, '```json\n{"a": 1}\n```', True),
        ({"type": "json", "strict": True, "required_keys": ["a"]}, '```json\n{"a": 1}\n```', False),
        ({"type": "json", "matches": {"p": "P[1-4]"}}, '{"p": "P2"}', True),
        ({"type": "json", "matches": {"p": "P[1-4]"}}, '{"p": "high"}', False),
        ({"type": "all", "checks": [{"type": "contains", "values": ["x"]}, {"type": "regex", "pattern": "y$"}]}, "x y", True),
    ],
)
def test_verifiers(verifier, output, ok):
    assert verify(t(verifier), output).passed is ok


def test_failure_detail_uses_label():
    v = verify(t({"type": "contains", "values": ["Sig"], "label": "replies must be signed"}), "hello")
    assert v.detail == "FAIL: replies must be signed"


def test_command_verifier(tmp_path):
    ok = verify(t({"type": "command", "cmd": f'"{sys.executable}" -c "print(1)"'}), "", workdir=tmp_path)
    bad = verify(t({"type": "command", "cmd": f'"{sys.executable}" -c "raise SystemExit(3)"'}), "", workdir=tmp_path)
    assert ok.passed and not bad.passed


def test_splits_are_deterministic_and_mixed():
    suite = TaskSuite([t({"type": "exact", "value": "x"}, id=f"task-{i}") for i in range(50)])
    a, b = suite.assign_splits(0.3), suite.assign_splits(0.3)
    assert [x.split for x in a.tasks] == [x.split for x in b.tasks]
    assert {x.split for x in a.tasks} == {TRAIN, HELDOUT}


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        TaskSuite([t({"type": "exact", "value": "x"}), t({"type": "exact", "value": "y"})])


def test_task_specific_literals():
    suite = TaskSuite([
        t({"type": "contains", "values": ["Shared Sig", "only-in-one"]}, id="task-a", prompt="ticket 48213 broke"),
        t({"type": "contains", "values": ["Shared Sig"]}, id="task-b", prompt="ticket 51007 broke"),
    ])
    lits = task_specific_literals(suite)
    assert {"only-in-one", "task-a", "task-b", "48213", "51007"} <= lits
    assert "Shared Sig" not in lits


def test_all_verifier_gives_partial_credit_and_lists_every_failure():
    v = {"type": "all", "checks": [
        {"type": "contains", "values": ["Sig"], "label": "must be signed"},
        {"type": "regex", "pattern": r"#\d{5}", "label": "must cite ticket"},
        {"type": "not_contains", "values": ["guarantee"]},
    ]}
    r = verify(t(v), "we guarantee it")
    assert not r.passed and abs(r.score - 0.0) < 1e-9
    r = verify(t(v), "Ticket #12345")
    assert not r.passed and abs(r.score - 2 / 3) < 1e-9
    assert r.detail == "FAIL: must be signed"
    r = verify(t(v), "nothing here")
    assert r.detail.splitlines() == ["FAIL: must be signed", "also FAIL: must cite ticket"]
    assert verify(t(v), "Sig #12345").score == 1.0


def test_python_placeholder_and_relative_repo_paths(tmp_path):
    import json
    from pathlib import Path

    from lamarck.tasks import expand_python

    assert expand_python("{python} -m pytest") == f'"{Path(sys.executable).as_posix()}" -m pytest'
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "tasks.json").write_text(json.dumps({"tasks": [
        {"id": "git-1", "prompt": "p", "verifier": {"type": "command", "cmd": "{python} -c \"print(1)\""}, "meta": {"repo": "repos/x"}},
        {"id": "git-2", "prompt": "p", "verifier": {"type": "exact", "value": "x"}, "meta": {"repo": str(tmp_path / "abs")}},
    ]}))
    suite = TaskSuite.load(tmp_path / "s" / "tasks.json")
    assert Path(suite.tasks[0].meta["repo"]) == (tmp_path / "s" / "repos" / "x").resolve()
    assert suite.tasks[1].meta["repo"] == str(tmp_path / "abs")
    assert verify(suite.tasks[0], "", workdir=tmp_path).passed
