"""Git-history task mining and the CommandTarget, on a throwaway repository."""

import subprocess
import sys
from pathlib import Path

import pytest

from lamarck.gitmine import MineConfig, is_test_file, mine_git_tasks
from lamarck.harness import Harness
from lamarck.targets.command import CommandTarget


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    (r / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    git(r, "add", ".")
    git(r, "commit", "-qm", "initial")
    # a commit that adds a feature together with its test -> a task
    (r / "calc.py").write_text("def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n")
    (r / "test_mul.py").write_text("from calc import mul\nassert mul(3, 4) == 12\nprint('ok')\n")
    git(r, "add", ".")
    git(r, "commit", "-qm", "Add mul()", "-m", "Users need multiplication.")
    # docs-only commit -> not a task
    (r / "README.md").write_text("calc\n")
    git(r, "add", ".")
    git(r, "commit", "-qm", "docs")
    return r


def test_is_test_file():
    assert is_test_file("tests/unit/foo.py") and is_test_file("pkg/test_x.py") and is_test_file("web/a.spec.ts")
    assert not is_test_file("src/contest.py")


def test_mine_git_tasks(repo):
    suite = mine_git_tasks(repo, MineConfig(test_cmd=f'"{sys.executable}" {{tests}}', heldout_fraction=0.0))
    assert len(suite.tasks) == 1
    t = suite.tasks[0]
    assert t.prompt.startswith("Add mul()") and "test_mul.py" in t.prompt
    assert t.meta["test_files"] == ["test_mul.py"] and t.meta["src_files"] == ["calc.py"]
    assert t.verifier["cmd"].endswith("test_mul.py")


def test_command_target_runs_agent_in_snapshot(repo, tmp_path):
    task = mine_git_tasks(repo, MineConfig(test_cmd=f'"{sys.executable}" {{tests}}', heldout_fraction=0.0)).tasks[0]
    h = Harness({"system.md": "Always run the tests."})
    fix = tmp_path / "fix.py"
    fixed_source = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
    fix.write_text(
        "import pathlib\n"
        f"pathlib.Path('calc.py').write_text({fixed_source!r})\n"
        "print(pathlib.Path('CLAUDE.md').read_text())\n"
        "print('LAMARCK_TOKENS=1234')\n"
    )
    good = CommandTarget(agent_cmd=f'"{sys.executable}" "{fix.as_posix()}"').run(h, task, 0)
    assert good.passed, good.trace
    assert "Always run the tests." in good.output and good.tokens == 1234
    lazy = CommandTarget(agent_cmd="echo done").run(h, task, 0)
    assert not lazy.passed and "verification command failed" in lazy.detail
    assert not (Path(repo) / "CLAUDE.md").exists()


def test_snapshot_has_no_git_history_to_peek_at(repo):
    task = mine_git_tasks(repo, MineConfig(test_cmd=f'"{sys.executable}" {{tests}}', heldout_fraction=0.0)).tasks[0]
    peek = CommandTarget(agent_cmd=f'git show {task.meta["commit"]}:calc.py > calc.py').run(Harness({}), task, 0)
    assert not peek.passed  # the fix commit is unreachable from the snapshot


def test_agent_quota_exit_code_is_fatal(repo):
    from lamarck.evaluate import FatalRunError

    task = mine_git_tasks(repo, MineConfig(test_cmd=f'"{sys.executable}" {{tests}}', heldout_fraction=0.0)).tasks[0]
    with pytest.raises(FatalRunError):
        CommandTarget(agent_cmd=f'"{sys.executable}" -c "raise SystemExit(75)"').run(Harness({}), task, 0)


def test_validate_keeps_fail_to_pass_and_drops_already_passing(repo):
    from dataclasses import replace

    from lamarck.gitmine import validate_suite
    from lamarck.tasks import TaskSuite

    task = mine_git_tasks(repo, MineConfig(test_cmd=f'"{sys.executable}" {{tests}}', heldout_fraction=0.0)).tasks[0]
    assert "Run them with:" in task.prompt and "test_mul.py" in task.prompt
    already = replace(task, id="git-already", meta={**task.meta, "base": task.meta["commit"]})
    broken = replace(task, id="git-broken", verifier={"type": "command", "cmd": f'"{sys.executable}" -c "raise SystemExit(1)"'})
    kept, results = validate_suite(TaskSuite([task, already, broken]), log=None)
    assert [t.id for t in kept.tasks] == [task.id]
    notes = {v.task_id: v.note for v in results}
    assert "already pass" in notes["git-already"] and "do not pass" in notes["git-broken"]
