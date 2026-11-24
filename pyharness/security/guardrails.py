from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pyharness.runtime.schemas import HarnessHook, ToolCall, ToolResult


@dataclass
class GuardrailScanResult:
    """Outcome of DLP redaction and indirect prompt injection scanning."""

    sanitized_text: str
    redactions: list[str] = field(default_factory=list)
    injection_detected: bool = False
    matched_injection_patterns: list[str] = field(default_factory=list)


class DLPScrubber:
    """Regex-based Data Loss Prevention (DLP) scrubber for secrets and PII."""

    DEFAULT_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
        (
            "api_key",
            re.compile(r"\b(sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
            "[REDACTED_API_KEY]",
        ),
        (
            "bearer_token",
            re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*"),
            "Bearer [REDACTED_TOKEN]",
        ),
        (
            "db_password",
            re.compile(r"(?i)(postgres(?:ql)?|mysql|redis|mongodb)://([^:/\s]+):([^@\s]+)@"),
            r"\1://\2:[REDACTED_PWD]@",
        ),
        (
            "cn_phone",
            re.compile(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)"),
            r"\1****\2",
        ),
        (
            "cn_id_card",
            re.compile(
                r"(?<!\d)([1-9]\d{5})(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}([\dXx])(?!\d)"
            ),
            r"\1***********\2",
        ),
    )

    def __init__(
        self,
        extra_patterns: list[tuple[str, re.Pattern[str], str]] | None = None,
    ) -> None:
        self._patterns: list[tuple[str, re.Pattern[str], str]] = list(
            self.DEFAULT_PATTERNS
        )
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def scrub(self, text: str) -> tuple[str, list[str]]:
        if not text:
            return text, []

        sanitized = text
        redactions: list[str] = []
        for rule_name, pattern, replacement in self._patterns:
            sanitized, count = pattern.subn(replacement, sanitized)
            if count > 0:
                redactions.append(rule_name)
        return sanitized, redactions


class PromptInjectionDetector:
    """Heuristic detector for EN & CN indirect prompt injection payloads."""

    DEFAULT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "en_ignore_previous_instructions",
            re.compile(r"(?i)\bignore\s+(?:all\s+)?previous\s+instructions\b"),
        ),
        (
            "en_disregard_directives",
            re.compile(r"(?i)\bdisregard\s+(?:the\s+)?(?:above|system|previous)\b"),
        ),
        (
            "en_system_prompt_override",
            re.compile(r"(?i)\bsystem\s+prompt\s+override\b"),
        ),
        (
            "en_forced_execution",
            re.compile(r"(?i)\byou\s+must\s+now\s+execute\b"),
        ),
        (
            "cn_ignore_previous_instructions",
            re.compile(r"忽略(?:此前|之前|以上)(?:所有|全部)?指令"),
        ),
        (
            "cn_disregard_safety_rules",
            re.compile(r"无视(?:系统|安全)(?:规则|限制|指令)"),
        ),
        (
            "cn_forced_tool_invocation",
            re.compile(r"立刻调用.*工具"),
        ),
    )

    def __init__(
        self,
        extra_patterns: list[tuple[str, re.Pattern[str]]] | None = None,
    ) -> None:
        self._patterns: list[tuple[str, re.Pattern[str]]] = list(self.DEFAULT_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    def detect(self, text: str) -> list[str]:
        if not text:
            return []
        matched: list[str] = []
        for rule_name, pattern in self._patterns:
            if pattern.search(text) is not None:
                matched.append(rule_name)
        return matched


class GuardrailHook(HarnessHook):
    """
    Composable post-tool guardrail hook implementing:
    1. DLP redaction of credentials, tokens, DB passwords, and Chinese PII.
    2. Indirect prompt injection detection and XML isolation tagging.
    3. Optional async security alerting callback (`on_alert`).
    """

    DEFAULT_UNTRUSTED_PATTERNS: frozenset[str] = frozenset({"rag_*", "mcp_*", "web_*"})

    def __init__(
        self,
        scrubber: DLPScrubber | None = None,
        detector: PromptInjectionDetector | None = None,
        untrusted_tools: set[str] | None = None,
        on_alert: Callable[[str, str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.scrubber = scrubber or DLPScrubber()
        self.detector = detector or PromptInjectionDetector()
        self.untrusted_tools = (
            set(untrusted_tools) if untrusted_tools is not None else None
        )
        self.on_alert = on_alert

    def _should_check_injection(self, tool_name: str) -> bool:
        if self.untrusted_tools is None:
            return True
        for pattern in self.untrusted_tools:
            if tool_name == pattern or fnmatch.fnmatchcase(tool_name, pattern):
                return True
            if pattern.endswith("_") and tool_name.startswith(pattern):
                return True
        return False

    def scan_text(self, text: str, check_injection: bool = True) -> GuardrailScanResult:
        sanitized, redactions = self.scrubber.scrub(text)
        matched_injection = self.detector.detect(text) if check_injection else []
        return GuardrailScanResult(
            sanitized_text=sanitized,
            redactions=redactions,
            injection_detected=bool(matched_injection),
            matched_injection_patterns=matched_injection,
        )

    async def after_tool(
        self,
        session_id: str,
        turn_id: str,
        call: ToolCall,
        result: ToolResult,
    ) -> ToolResult:
        check_injection = self._should_check_injection(call.name)
        scan = self.scan_text(result.content, check_injection=check_injection)

        content = scan.sanitized_text
        if scan.injection_detected:
            content = (
                f'<untrusted_tool_output tool="{call.name}" '
                f'warning="indirect_prompt_injection_detected">\n'
                f"{scan.sanitized_text}\n"
                f"</untrusted_tool_output>"
            )

        updated_metadata = dict(result.metadata)
        updated_metadata["guardrail_redactions"] = list(scan.redactions)
        updated_metadata["injection_detected"] = scan.injection_detected
        if scan.matched_injection_patterns:
            updated_metadata["matched_injection_patterns"] = list(
                scan.matched_injection_patterns
            )

        updated_result = result.model_copy(
            update={
                "content": content,
                "metadata": updated_metadata,
            }
        )

        if self.on_alert is not None and (scan.redactions or scan.injection_detected):
            await self.on_alert(
                session_id,
                turn_id,
                {
                    "tool_name": call.name,
                    "tool_call_id": call.id,
                    "redactions": list(scan.redactions),
                    "injection_detected": scan.injection_detected,
                    "matched_injection_patterns": list(scan.matched_injection_patterns),
                },
            )

        return updated_result
