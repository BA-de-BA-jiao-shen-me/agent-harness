from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pyharness.security.auth import Principal


class QuotaExceededError(RuntimeError):
    """Raised when a principal exceeds RPM rate limits or daily token/USD quotas."""

    def __init__(
        self,
        reason: str,
        retry_after_seconds: int = 60,
        quota_type: str = "rate_limit",
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after_seconds = retry_after_seconds
        self.quota_type = quota_type


class QuotaManager:
    """
    Distributed FinOps quota & rate-limit manager backed by `CoordinationBackend`.

    Enforces:
    - Per-tenant/user sliding-window RPM rate limits.
    - Per-principal daily token quotas and USD cost ceilings.
    - Daily usage recording and FinOps chargeback summaries.
    """

    def __init__(
        self,
        coordination: Any,
        rpm_limit: int = 60,
        window_seconds: int = 60,
        enabled: bool = True,
    ) -> None:
        self.coordination = coordination
        self.rpm_limit = rpm_limit
        self.window_seconds = window_seconds
        self.enabled = enabled

    @staticmethod
    def _today_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _seconds_until_utc_midnight() -> int:
        now = datetime.now(timezone.utc)
        next_day = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max(60, int((next_day - now).total_seconds()))

    async def check_before_turn(self, principal: Principal) -> None:
        if not self.enabled:
            return

        if self.rpm_limit > 0:
            rate_key = f"rpm:{principal.tenant_id}:{principal.user_id}"
            allowed, retry_after = await self.coordination.check_rate_limit(
                rate_key,
                self.rpm_limit,
                self.window_seconds,
            )
            if not allowed:
                raise QuotaExceededError(
                    reason=(
                        f"Rate limit exceeded for user '{principal.user_id}' in tenant "
                        f"'{principal.tenant_id}' ({self.rpm_limit} requests per "
                        f"{self.window_seconds}s)."
                    ),
                    retry_after_seconds=max(1, int(retry_after)),
                    quota_type="rpm",
                )

        date_str = self._today_utc()
        usage = await self.coordination.get_daily_usage(
            date_str,
            principal.tenant_id,
            principal.user_id,
        )
        used_tokens = int(usage.get("tokens", 0))
        used_cost_usd = float(usage.get("cost_usd", 0.0))

        if principal.daily_token_quota > 0 and used_tokens >= principal.daily_token_quota:
            raise QuotaExceededError(
                reason=(
                    f"Daily token quota exceeded for user '{principal.user_id}' "
                    f"({used_tokens}/{principal.daily_token_quota} tokens)."
                ),
                retry_after_seconds=self._seconds_until_utc_midnight(),
                quota_type="daily_budget",
            )

        if (
            principal.daily_cost_limit_usd > 0
            and used_cost_usd >= principal.daily_cost_limit_usd
        ):
            raise QuotaExceededError(
                reason=(
                    f"Daily cost limit exceeded for user '{principal.user_id}' "
                    f"(${used_cost_usd:.4f}/${principal.daily_cost_limit_usd:.2f})."
                ),
                retry_after_seconds=self._seconds_until_utc_midnight(),
                quota_type="daily_budget",
            )

    async def record_usage(
        self,
        principal: Principal,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        await self.coordination.record_daily_usage(
            date_str=self._today_utc(),
            tenant_id=principal.tenant_id,
            department=principal.department,
            user_id=principal.user_id,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            cost_usd=float(cost_usd),
        )

    async def get_usage_report(
        self,
        tenant_id: str | None = None,
        date_str: str | None = None,
    ) -> dict[str, Any]:
        target_date = date_str or self._today_utc()
        entries: list[dict[str, Any]] = await self.coordination.list_usage_summary(
            target_date,
            tenant_id=tenant_id,
        )

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost_usd = 0.0
        by_department: dict[str, dict[str, Any]] = {}

        for entry in entries:
            prompt = int(entry.get("prompt_tokens", 0))
            completion = int(entry.get("completion_tokens", 0))
            tokens = int(entry.get("tokens", prompt + completion))
            cost = float(entry.get("cost_usd", 0.0))
            dept = str(entry.get("department", "default"))

            total_prompt_tokens += prompt
            total_completion_tokens += completion
            total_tokens += tokens
            total_cost_usd += cost

            dept_bucket = by_department.setdefault(
                dept,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            dept_bucket["prompt_tokens"] += prompt
            dept_bucket["completion_tokens"] += completion
            dept_bucket["tokens"] += tokens
            dept_bucket["cost_usd"] = round(dept_bucket["cost_usd"] + cost, 6)

        return {
            "date": target_date,
            "tenant_id": tenant_id,
            "enabled": self.enabled,
            "rpm_limit": self.rpm_limit,
            "window_seconds": self.window_seconds,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost_usd, 6),
            "by_department": by_department,
            "entries": entries,
        }
