"""The harness: everything around a frozen model that we are allowed to evolve.

A harness is a set of named text components (a system prompt, a rules file,
skill files). Edits are small and, by default, *additive* — the property
DarwinX relies on to merge complementary variants.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SYSTEM = "system.md"
RULES = "rules.md"
SKILL_PREFIX = "skills/"

ADDITIVE_OPS = ("add_skill", "append_rule")
ALL_OPS = ADDITIVE_OPS + ("set_component", "remove_component")


@dataclass(frozen=True)
class Edit:
    """One harness change. `target` is a component name, e.g. ``skills/verify.md``."""

    op: str
    target: str
    content: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.op not in ALL_OPS:
            raise ValueError(f"unknown edit op {self.op!r}; expected one of {ALL_OPS}")
        if self.op == "add_skill" and not self.target.startswith(SKILL_PREFIX):
            raise ValueError(f"add_skill target must start with {SKILL_PREFIX!r}: {self.target!r}")

    @property
    def additive(self) -> bool:
        return self.op in ADDITIVE_OPS

    def key(self) -> tuple[str, str, str]:
        return (self.op, self.target, self.content)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Edit":
        return cls(op=d["op"], target=d["target"], content=d.get("content", ""), rationale=d.get("rationale", ""))


class EditConflict(Exception):
    """Raised when a set of edits cannot be applied together (e.g. two rewrites of one component)."""


@dataclass
class Harness:
    components: dict[str, str] = field(default_factory=dict)

    # ---- identity -------------------------------------------------------
    def fingerprint(self) -> str:
        blob = json.dumps(sorted(self.components.items()), ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def size_chars(self) -> int:
        return sum(len(v) for v in self.components.values())

    # ---- editing --------------------------------------------------------
    def apply(self, edits: list[Edit]) -> "Harness":
        comps = dict(self.components)
        for e in edits:
            if e.op == "add_skill":
                comps[e.target] = e.content.strip() + "\n"
            elif e.op == "append_rule":
                existing = comps.get(e.target, "")
                line = e.content.strip()
                if not line.startswith("- "):
                    line = "- " + line
                if line not in existing.splitlines():
                    comps[e.target] = (existing.rstrip("\n") + "\n" + line + "\n").lstrip("\n")
            elif e.op == "set_component":
                comps[e.target] = e.content
            elif e.op == "remove_component":
                comps.pop(e.target, None)
        return Harness(comps)

    @staticmethod
    def check_mergeable(edit_lists: list[list[Edit]]) -> list[Edit]:
        """Union several lineages' edits (dedup, stable order). Non-additive edits that
        disagree on the same target are a conflict."""
        merged: list[Edit] = []
        seen: set[tuple[str, str, str]] = set()
        rewrites: dict[str, str] = {}
        for edits in edit_lists:
            for e in edits:
                if e.key() in seen:
                    continue
                if not e.additive:
                    prev = rewrites.get(e.target)
                    if prev is not None and prev != e.content:
                        raise EditConflict(f"conflicting {e.op} on {e.target}")
                    rewrites[e.target] = e.content
                if e.op == "add_skill":
                    other = next((m for m in merged if m.op == "add_skill" and m.target == e.target), None)
                    if other is not None and other.content != e.content:
                        raise EditConflict(f"two different skills named {e.target}")
                seen.add(e.key())
                merged.append(e)
        return merged

    # ---- rendering ------------------------------------------------------
    def skills(self) -> dict[str, str]:
        return {k: v for k, v in sorted(self.components.items()) if k.startswith(SKILL_PREFIX)}

    def render(self) -> str:
        """Render as one instruction document (the form a CLAUDE.md / AGENTS.md / system prompt takes)."""
        parts: list[str] = []
        if self.components.get(SYSTEM, "").strip():
            parts.append(self.components[SYSTEM].strip())
        if self.components.get(RULES, "").strip():
            parts.append("## Rules\n" + self.components[RULES].strip())
        skills = self.skills()
        if skills:
            body = "\n\n".join(f"### {name[len(SKILL_PREFIX):].removesuffix('.md')}\n{text.strip()}" for name, text in skills.items())
            parts.append("## Skills\n" + body)
        others = {k: v for k, v in sorted(self.components.items()) if k not in (SYSTEM, RULES) and not k.startswith(SKILL_PREFIX)}
        for name, text in others.items():
            parts.append(f"## {name}\n{text.strip()}")
        return "\n\n".join(parts) + "\n"

    def diff(self, other: "Harness") -> str:
        a = self.render().splitlines(keepends=True)
        b = other.render().splitlines(keepends=True)
        return "".join(difflib.unified_diff(a, b, fromfile=f"harness@{self.fingerprint()}", tofile=f"harness@{other.fingerprint()}"))

    # ---- persistence ----------------------------------------------------
    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name, text in self.components.items():
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "Harness":
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"harness directory not found: {root}")
        comps = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.md"))}
        return cls(comps)
