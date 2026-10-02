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


def test_command_target_runs_agent_in_worktree(repo):
    task = mine_git_tasks(repo, MineConfig(test_cmd=f'"{sys.executable}" {{tests}}', heldout_fraction=0.0)).tasks[0]
    h = Harness({"system.md": "Always run the tests."})
    # an 'agent' that solves the task by restoring the fixed source file, and checks it can read its harness
    solver = f'git checkout {task.meta["commit"]} -- calc.py && "{sys.executable}" -c "import pathlib; print(pathlib.Path(\'CLAUDE.md\').read_text())"'
    good = CommandTarget(agent_cmd=solver).run(h, task, 0)
    assert good.passed, good.trace
    assert "Always run the tests." in good.output
    lazy = CommandTarget(agent_cmd="echo done").run(h, task, 0)
    assert not lazy.passed and "verification command failed" in lazy.detail
    # worktrees are cleaned up
    wt = subprocess.run(["git", "-C", str(repo), "worktree", "list"], capture_output=True, text=True).stdout
    assert len(wt.strip().splitlines()) == 1
    assert not (Path(repo) / "CLAUDE.md").exists()
