import sys
import time
from pathlib import Path

import pytest

from lamarck import proc
from lamarck.agents.mini import find_bash, outside_repo, run_command

PY = Path(sys.executable).as_posix()


@pytest.mark.parametrize("cmd", [
    'find / -name "*.py"', "ls /", "cat ~/.ssh/id_rsa", "ls /c/Users", r"type C:\Windows\x", "cat C:/Users/x",
    "cat ../../secret", "cd / && ls", "grep -r foo /etc",
])
def test_guard_blocks_commands_outside_the_repo(cmd):
    assert outside_repo(cmd)


@pytest.mark.parametrize("cmd", [
    "ls", "grep -rn timed src/", "cat src/itsdangerous/timed.py", f'"{PY}" -m pytest -q tests/x.py',
    "python - <<'EOF'\nopen('a.py','w').write('x=1/2')\nEOF", "ls tests/ 2>/dev/null", "cat ../README.md",
    "sed -i 's/a/b/' src/x.py", "echo 'a/b' > f.txt", "pytest tests/test_a.py::test_b",
])
def test_guard_allows_normal_repo_work(cmd):
    assert not outside_repo(cmd)


def test_blocked_command_is_not_executed(tmp_path):
    code, out = run_command("ls / > leaked.txt", tmp_path, 10, find_bash())
    assert code == 126 and "Blocked" in out and not (tmp_path / "leaked.txt").exists()


def test_timeout_kills_grandchildren():
    # a child that spawns a grandchild which would hold the pipes open for 60s
    code = f'import subprocess, sys; subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); time_ = __import__("time"); time_.sleep(60)'
    t0 = time.monotonic()
    res = proc.run([sys.executable, "-c", code], timeout=2)
    assert res.timed_out and time.monotonic() - t0 < 20
