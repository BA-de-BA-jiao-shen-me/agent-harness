from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_SKILL_STATUSES: frozenset[str] = frozenset({"draft", "published", "deprecated"})


@dataclass
class SkillBundle:
    """Declarative Skill loaded from a SKILL.md file (inspired by DSH skill bundles)."""

    name: str
    description: str
    instructions: str
    triggers: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    source_path: str = ""
    status: str = "published"
    version: str = "1.0.0"
    owner: str = ""


@dataclass
class SkillReadinessReport:
    """Readiness diagnostic report for a loaded skill against the live ToolRegistry."""

    skill_name: str
    status: str
    ready: bool
    declared_tools: list[str] = field(default_factory=list)
    missing_tools: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SkillInvocationStats:
    """Runtime invocation telemetry for an activated skill."""

    skill_name: str
    invocations: int = 0
    last_invoked_at_ms: int | None = None


class SkillRouter:
    """
    Loads SKILL.md bundles from disk and dynamically selects relevant skills per turn
    (inspired by DSH skill-turn-selection.mjs), resolving both transient instructions,
    lifecycle gating (`draft` / `published` / `deprecated`), readiness diagnostics,
    invocation telemetry, and least-privilege `allowed_tools` scopes.
    """

    def __init__(self, skills_dir: str | Path | None = None) -> None:
        self.skills_dir: Path | None = Path(skills_dir) if skills_dir is not None else None
        self._skills: dict[str, SkillBundle] = {}
        self._stats: dict[str, SkillInvocationStats] = {}
        if self.skills_dir is not None and self.skills_dir.exists():
            self.reload()

    def reload(self) -> None:
        self._skills.clear()
        if self.skills_dir is None or not self.skills_dir.exists():
            return
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
        raw_status = str(meta.get("status") or "published").strip().lower()
        status = raw_status if raw_status in VALID_SKILL_STATUSES else "published"
        version = str(meta.get("version") or "1.0.0").strip() or "1.0.0"
        owner = str(meta.get("owner") or "").strip()
        return SkillBundle(
            name=name,
            description=str(meta.get("description", "")),
            instructions=body,
            triggers=[str(t).lower() for t in (meta.get("triggers") or [])],
            allowed_tools=[str(t) for t in (meta.get("allowed_tools") or [])],
            source_path=str(path),
            status=status,
            version=version,
            owner=owner,
        )

    def register_skill(self, bundle: SkillBundle) -> None:
        """Register or replace a skill bundle at runtime (useful for tests & dynamic loading)."""
        normalized = str(bundle.status).strip().lower()
        if normalized in VALID_SKILL_STATUSES:
            bundle.status = normalized
        self._skills[bundle.name] = bundle

    def set_status(self, skill_name: str, status: str) -> SkillBundle:
        """Transition a registered skill's lifecycle status (`draft` / `published` / `deprecated`)."""
        if not isinstance(status, str):
            raise ValueError(
                f"Invalid skill status {status!r}; expected one of {sorted(VALID_SKILL_STATUSES)}."
            )
        normalized = status.strip().lower()
        if normalized not in VALID_SKILL_STATUSES:
            raise ValueError(
                f"Invalid skill status {status!r}; expected one of {sorted(VALID_SKILL_STATUSES)}."
            )
        if skill_name not in self._skills:
            raise KeyError(f"Unknown skill: {skill_name!r}")
        bundle = self._skills[skill_name]
        bundle.status = normalized
        return bundle

    def list_skills(self) -> list[SkillBundle]:
        return list(self._skills.values())

    def select_for_query(
        self,
        query: str,
        max_skills: int = 2,
        *,
        include_drafts: bool = False,
    ) -> list[SkillBundle]:
        """Score eligible skills by trigger overlap and return the top matching bundles."""
        q_lower = query.lower()
        scored: list[tuple[int, SkillBundle]] = []
        for skill in self._skills.values():
            status = str(skill.status).strip().lower()
            if status == "deprecated":
                continue
            if status == "draft" and not include_drafts:
                continue
            hits = sum(1 for kw in skill.triggers if kw and kw.lower() in q_lower)
            if skill.name.lower() in q_lower:
                hits += 2
            if hits > 0:
                scored.append((hits, skill))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [s for _, s in scored[:max_skills]]

    def resolve_turn_context(
        self,
        query: str,
        max_skills: int = 2,
        *,
        include_drafts: bool = False,
    ) -> tuple[str | None, set[str] | None, list[SkillBundle]]:
        """
        Resolve per-turn transient instructions and least-privilege `allowed_tools` union
        without mutating persistent session system prompts, recording invocation telemetry.
        """
        selected = self.select_for_query(
            query, max_skills=max_skills, include_drafts=include_drafts
        )
        if not selected:
            return None, None, []

        now_ms = int(time.time() * 1000)
        for s in selected:
            stat = self._stats.get(s.name)
            if stat is None:
                stat = SkillInvocationStats(skill_name=s.name)
                self._stats[s.name] = stat
            stat.invocations += 1
            stat.last_invoked_at_ms = now_ms

        blocks = [f"## Active Skill: {s.name}\n{s.instructions}" for s in selected]
        instructions_block = (
            "=== Dynamically Activated Skills for This Turn ===\n"
            + "\n\n".join(blocks)
        )
        permitted: set[str] = set()
        for s in selected:
            permitted.update(s.allowed_tools)
        allowed_set = permitted if permitted else None
        return instructions_block, allowed_set, selected

    def check_readiness(self, tool_registry: Any) -> list[SkillReadinessReport]:
        """Inspect every loaded skill's tool bindings, triggers, and lifecycle status."""
        get_fn = getattr(tool_registry, "get", None) if tool_registry is not None else None
        registered_names: set[str] | None = None
        if not callable(get_fn) and tool_registry is not None:
            list_specs_fn = getattr(tool_registry, "list_specs", None)
            if callable(list_specs_fn):
                registered_names = {spec.name for spec in list_specs_fn()}

        def _has_tool(name: str) -> bool:
            if callable(get_fn):
                return get_fn(name) is not None
            if registered_names is not None:
                return name in registered_names
            return False

        reports: list[SkillReadinessReport] = []
        for skill in self._skills.values():
            status = str(skill.status).strip().lower()
            declared_tools = list(skill.allowed_tools)
            missing_tools = [t for t in declared_tools if not _has_tool(t)]
            warnings: list[str] = []
            if missing_tools:
                warnings.append(
                    f"Missing registered tools: {', '.join(missing_tools)}"
                )
            if not skill.triggers:
                warnings.append(
                    "Skill defines no triggers and will only match by explicit skill name."
                )
            if status == "draft":
                warnings.append(
                    "Skill is in 'draft' status and is excluded from default auto-selection."
                )
            elif status == "deprecated":
                warnings.append(
                    "Skill is 'deprecated' and will never be auto-selected."
                )
            ready = len(missing_tools) == 0 and status != "deprecated"
            reports.append(
                SkillReadinessReport(
                    skill_name=skill.name,
                    status=status,
                    ready=ready,
                    declared_tools=declared_tools,
                    missing_tools=missing_tools,
                    warnings=warnings,
                )
            )
        return reports

    def get_invocation_stats(self) -> dict[str, dict[str, Any]]:
        """Return JSON-serializable invocation statistics for activated skills."""
        return {name: asdict(stat) for name, stat in self._stats.items()}

    def build_augmented_prompt(
        self,
        base_prompt: str,
        query: str,
        *,
        include_drafts: bool = False,
    ) -> tuple[str, list[SkillBundle]]:
        instructions_block, _allowed, selected = self.resolve_turn_context(
            query, include_drafts=include_drafts
        )
        if not instructions_block:
            return base_prompt, []
        return f"{base_prompt}\n\n{instructions_block}", selected
