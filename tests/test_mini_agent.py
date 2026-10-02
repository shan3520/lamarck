from types import SimpleNamespace

from lamarck.agents import mini
from lamarck.chat_compat import GatewayHTTPError


def reply(text):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                           usage=SimpleNamespace(input_tokens=50, output_tokens=10), model="m")


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.prompts.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_agent_runs_commands_reads_harness_and_submits(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENT.md").write_text("Always run the tests before you submit.")
    (tmp_path / "task.md").write_text("Create hello.txt containing hi.")
    client = Scripted([
        reply("I'll create it.\n```bash\necho hi > hello.txt\n```"),
        reply("Check.\n```bash\ncat hello.txt\n```"),
        reply("Done. SUBMIT"),
    ])
    code = mini.main(["--model", "m", "--prompt-file", str(tmp_path / "task.md")], client=client)
    out = capsys.readouterr().out
    assert code == 0
    assert (tmp_path / "hello.txt").read_text().strip() == "hi"
    assert "END: submitted" in out and "LAMARCK_TOKENS=180" in out
    assert client.prompts[0]["system"].startswith("Always run the tests before you submit.")
    assert "exit code 0\nhi" in client.prompts[2]["messages"][0]["content"]


def test_agent_exits_75_on_daily_quota(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.md").write_text("x")
    client = Scripted([GatewayHTTPError(429, "daily quota (per-day limit) reached")])
    assert mini.main(["--model", "m", "--prompt-file", str(tmp_path / "task.md")], client=client) == 75


def test_agent_respects_step_and_token_caps(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.md").write_text("loop forever")
    client = Scripted([reply("```bash\necho again\n```")] * 10)
    mini.main(["--model", "m", "--prompt-file", str(tmp_path / "task.md"), "--max-steps", "10", "--max-tokens-total", "150"], client=client)
    out = capsys.readouterr().out
    assert "END: token cap" in out and len(client.prompts) == 3


def test_history_is_compacted():
    hist = [(f"cmd{i}", f"exit code 0\nout{i}") for i in range(8)]
    text = mini.build_user("task", hist, keep=3, step=9, max_steps=12)
    assert "Earlier steps (summarized)" in text and "$ cmd0" in text
    assert text.count("# You ran") == 3 and "out7" in text and "# Output\nexit code 0\nout0" not in text


def test_truncate_keeps_head_and_tail():
    t = mini.truncate("A" * 100 + "B" * 100, 50)
    assert t.startswith("A" * 25) and t.endswith("B" * 25) and "omitted" in t


def test_echo_submit_counts_as_submission(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.md").write_text("x")
    client = Scripted([reply("```bash\necho done\n```"), reply('All good.\n```bash\necho "SUBMIT"\n```')])
    mini.main(["--model", "m", "--prompt-file", str(tmp_path / "task.md")], client=client)
    assert "END: submitted after 1 commands" in capsys.readouterr().out and len(client.prompts) == 2
