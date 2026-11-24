from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SkillBundle:
    """Declarative Skill loaded from a SKILL.md file (inspired by DSH skill bundles)."""

    name: str
    description: str
    instructions: str
    triggers: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    source_path: str = ""


class SkillRouter:
    """
    Loads SKILL.md bundles from disk and dynamically selects relevant skills per turn
    (inspired by DSH skill-turn-selection.mjs) to avoid bloating every system prompt.
    """

    def __init__(self, skills_dir: str | Path) -> None:
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, SkillBundle] = {}
        if self.skills_dir.exists():
            self.reload()

    def reload(self) -> None:
        self._skills.clear()
        for md_file in sorted(self.skills_dir.rglob("SKILL.md")):
            bundle = self._parse_skill_file(md_file)
            if bundle:
                self._skills[bundle.name] = bundle

    @staticmethod
    def _parse_skill_file(path: Path) -> SkillBundle | None:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw.startswith("---"):
            return None
        parts = raw.split("---", 2)
        if len(parts) < 3:
            return None
        meta = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()
        name = str(meta.get("name") or path.parent.name)
        return SkillBundle(
            name=name,
            description=str(meta.get("description", "")),
            instructions=body,
            triggers=[str(t).lower() for t in (meta.get("triggers") or [])],
            allowed_tools=[str(t) for t in (meta.get("allowed_tools") or [])],
            source_path=str(path),
        )

    def list_skills(self) -> list[SkillBundle]:
        return list(self._skills.values())

    def select_for_query(self, query: str, max_skills: int = 2) -> list[SkillBundle]:
        """Score skills by trigger overlap and return the top matching bundles."""
        q_lower = query.lower()
        scored: list[tuple[int, SkillBundle]] = []
        for skill in self._skills.values():
            hits = sum(1 for kw in skill.triggers if kw in q_lower)
            if skill.name.lower() in q_lower:
                hits += 2
            if hits > 0:
                scored.append((hits, skill))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [s for _, s in scored[:max_skills]]

    def build_augmented_prompt(self, base_prompt: str, query: str) -> tuple[str, list[SkillBundle]]:
        selected = self.select_for_query(query)
        if not selected:
            return base_prompt, []
        blocks = [
            f"## Active Skill: {s.name}\n{s.instructions}"
            for s in selected
        ]
        augmented = (
            f"{base_prompt}\n\n"
            "=== Dynamically Activated Skills for This Turn ===\n"
            + "\n\n".join(blocks)
        )
        return augmented, selected
