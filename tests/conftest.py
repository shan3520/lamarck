import pytest

CREDENTIAL_VARS = ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Tests must never see real credentials or a real .env (they could spend money)."""
    for k in CREDENTIAL_VARS:
        monkeypatch.setenv(k, "placeholder")  # registers the original value for restore
        monkeypatch.delenv(k)
    monkeypatch.chdir(tmp_path)
