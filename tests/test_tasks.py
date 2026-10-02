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
