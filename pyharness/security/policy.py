from __future__ import annotations

import fnmatch
import json
import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from pyharness.security.auth import Principal


class PolicyAction(StrEnum):
    """Decision outcome produced by the dynamic Access Control Policy (ACP) engine."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PolicyMode(StrEnum):
    """Execution mode for a policy rule: enforced or shadow-audit only."""

    ENFORCE = "enforce"
    AUDIT = "audit"


class PolicyRule(BaseModel):
    """Declarative rule governing tool execution based on role and argument patterns."""

    id: str
    description: str = ""
    mode: PolicyMode = PolicyMode.ENFORCE
    action: PolicyAction
    tools: list[str]
    roles: list[str] = Field(default_factory=list)
    exclude_roles: list[str] = Field(default_factory=list)
    arg_patterns: dict[str, str] = Field(default_factory=dict)
    require_absent_patterns: dict[str, str] = Field(default_factory=dict)
    required_approver_roles: list[str] = Field(
        default_factory=lambda: ["approver", "admin"]
    )
    reason: str = ""


class PolicyAuditMatch(BaseModel):
    """Shadow-mode audit match recorded without altering the enforced decision."""

    rule_id: str
    would_action: PolicyAction
    reason: str


class PolicyEvaluation(BaseModel):
    """Result of evaluating all rules against a principal and tool invocation."""

    action: PolicyAction
    matched_rule_id: str | None = None
    mode: PolicyMode = PolicyMode.ENFORCE
    reason: str = ""
    required_approver_roles: list[str] = Field(
        default_factory=lambda: ["approver", "admin"]
    )
    audit_matches: list[PolicyAuditMatch] = Field(default_factory=list)


_ACTION_PRECEDENCE: dict[PolicyAction, int] = {
    PolicyAction.ALLOW: 1,
    PolicyAction.ASK: 2,
    PolicyAction.DENY: 3,
}


@lru_cache(maxsize=512)
def _compile_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)


_SQL_COMMENT_OR_STRING_RE = re.compile(
    r"'(?:''|[^'])*'"      # Single-quoted SQL string literal (preserve)
    r"|--[^\r\n]*"         # Single-line SQL comment (strip)
    r"|/\*.*?(?:\*/|$)",   # Block SQL comment, closed or unclosed (strip)
    re.DOTALL,
)


def _strip_sql_comments(sql: str) -> str:
    """Strip ANSI SQL `--` and `/* ... */` comments outside string literals."""
    return _SQL_COMMENT_OR_STRING_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith("'") else " ",
        sql,
    )


def _extract_arg_target(arguments: dict[str, Any], key: str) -> str:
    if key == "*":
        return json.dumps(arguments, ensure_ascii=False, default=str)
    value = arguments.get(key)
    if value is None:
        return ""
    text = str(value)
    if key == "sql":
        return _strip_sql_comments(text)
    return text


class PolicyEngine:
    """
    Dynamic Access Control Policy (ACP) evaluator.

    - Evaluates ALL configured rules on every tool call.
    - Matching `AUDIT` rules are recorded in `audit_matches` without affecting the action.
    - Matching `ENFORCE` rules resolve using strict safety precedence: DENY > ASK > ALLOW.
    """

    def __init__(
        self,
        rules: list[PolicyRule] | None = None,
        default_action: PolicyAction = PolicyAction.ALLOW,
    ) -> None:
        self.rules: list[PolicyRule] = list(rules or [])
        self.default_action = default_action

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicyEngine:
        policy_path = Path(path)
        raw_text = policy_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Policy file {policy_path} must contain a YAML mapping.")

        raw_default = data.get("default_action", PolicyAction.ALLOW)
        default_action = PolicyAction(raw_default)

        raw_rules = data.get("rules", [])
        if not isinstance(raw_rules, list):
            raise ValueError(f"Policy file {policy_path} 'rules' field must be a list.")

        rules = [PolicyRule.model_validate(item) for item in raw_rules]
        return cls(rules=rules, default_action=default_action)

    @staticmethod
    def _rule_matches(
        rule: PolicyRule,
        principal: Principal | None,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        if not rule.tools or not any(
            fnmatch.fnmatchcase(tool_name, pattern) for pattern in rule.tools
        ):
            return False

        principal_roles: frozenset[str] = (
            principal.roles if principal is not None else frozenset()
        )

        # Exclude roles take precedence: if principal has any excluded role, skip rule.
        if rule.exclude_roles and principal_roles.intersection(rule.exclude_roles):
            return False

        # If roles are specified, principal must possess at least one listed role.
        if rule.roles and not principal_roles.intersection(rule.roles):
            return False

        # All required argument patterns must match.
        for arg_key, pattern in rule.arg_patterns.items():
            target = _extract_arg_target(arguments, arg_key)
            if _compile_regex(pattern).search(target) is None:
                return False

        # All require_absent_patterns must NOT match.
        for arg_key, pattern in rule.require_absent_patterns.items():
            target = _extract_arg_target(arguments, arg_key)
            if _compile_regex(pattern).search(target) is not None:
                return False

        return True

    def evaluate(
        self,
        principal: Principal | None,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> PolicyEvaluation:
        safe_args = arguments if isinstance(arguments, dict) else {}
        audit_matches: list[PolicyAuditMatch] = []
        winner: PolicyRule | None = None
        winner_rank = 0

        for rule in self.rules:
            if not self._rule_matches(rule, principal, tool_name, safe_args):
                continue

            if rule.mode == PolicyMode.AUDIT:
                audit_matches.append(
                    PolicyAuditMatch(
                        rule_id=rule.id,
                        would_action=rule.action,
                        reason=rule.reason or rule.description,
                    )
                )
                continue

            rank = _ACTION_PRECEDENCE[rule.action]
            if rank > winner_rank:
                winner = rule
                winner_rank = rank

        if winner is not None:
            return PolicyEvaluation(
                action=winner.action,
                matched_rule_id=winner.id,
                mode=PolicyMode.ENFORCE,
                reason=winner.reason or winner.description,
                required_approver_roles=list(winner.required_approver_roles),
                audit_matches=audit_matches,
            )

        return PolicyEvaluation(
            action=self.default_action,
            matched_rule_id=None,
            mode=PolicyMode.ENFORCE,
            reason="No enforced policy rule matched; applying default action.",
            required_approver_roles=["approver", "admin"],
            audit_matches=audit_matches,
        )
