"""Credentials and gateway setup.

Keys live in environment variables or a git-ignored `.env` file, never in config
files, the command line, or logs.

OpenRouter: put OPENROUTER_API_KEY in `.env`. Lamarck then points the Anthropic SDK
at OpenRouter's Anthropic-compatible endpoint (ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN,
with ANTHROPIC_API_KEY explicitly empty, as OpenRouter's docs require) and maps
model ids (claude-opus-5-5 -> anthropic/claude-opus-5.5).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

OPENROUTER_BASE = "https://openrouter.ai/api"


def load_dotenv(*paths: str | Path) -> list[Path]:
    """Load KEY=VALUE lines from the given files. Existing environment variables win."""
    loaded = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip().removeprefix("export ").strip(), value.strip().strip("'\"")
            if key and value and key not in os.environ:
                os.environ[key] = value
        loaded.append(path)
    return loaded


def configure_gateway() -> str:
    """Return the provider in use: 'openrouter', 'custom-gateway' or 'anthropic'."""
    base = os.environ.get("ANTHROPIC_BASE_URL", "")
    # An explicit OpenRouter key wins over inherited default Anthropic settings (e.g. a parent
    # tool's own ANTHROPIC_AUTH_TOKEN), so runs bill the account the user chose.
    if os.environ.get("OPENROUTER_API_KEY") and (not base or "anthropic.com" in base):
        os.environ["ANTHROPIC_BASE_URL"] = OPENROUTER_BASE
        os.environ["ANTHROPIC_AUTH_TOKEN"] = os.environ["OPENROUTER_API_KEY"]
        os.environ["ANTHROPIC_API_KEY"] = ""
        return "openrouter"
    return provider()


def provider() -> str:
    base = os.environ.get("ANTHROPIC_BASE_URL", "")
    if not base or "anthropic.com" in base:
        return "anthropic"
    return "openrouter" if "openrouter.ai" in base else "custom-gateway"


def model_id(model: str) -> str:
    """Translate Anthropic model ids to the active provider's naming."""
    if provider() == "openrouter" and "/" not in model:
        return "anthropic/" + re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", model)
    return model
