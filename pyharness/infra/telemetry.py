from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Status as _OtelStatus
    from opentelemetry.trace import StatusCode as _OtelStatusCode

    _HAS_OTEL = True
except ImportError:  # pragma: no cover - graceful no-op fallback when OTel is absent
    _otel_trace = None
    _OtelStatus = None
    _OtelStatusCode = None
    _HAS_OTEL = False

# OpenTelemetry GenAI semantic convention key aliases
_GENAI_SEMCONV_MAP: dict[str, str] = {
    "system": "gen_ai.system",
    "operation": "gen_ai.operation.name",
    "model": "gen_ai.request.model",
    "response_model": "gen_ai.response.model",
    "prompt_tokens": "gen_ai.usage.input_tokens",
    "completion_tokens": "gen_ai.usage.output_tokens",
    "tool_name": "gen_ai.tool.name",
    "tool_call_id": "gen_ai.tool.call.id",
}


class _NoOpSpan:
    """Zero-dependency fallback span preserving the OpenTelemetry Span call surface."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.exceptions: list[BaseException] = []
        self.status: Any = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, exception: BaseException) -> None:
        self.exceptions.append(exception)

    def set_status(self, status: Any, description: str | None = None) -> None:
        self.status = (status, description) if description is not None else status

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        return None

    def is_recording(self) -> bool:
        return False


def _sanitize_otel_value(value: Any) -> Any:
    if isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (bool, str, int, float)) for v in value):
            return list(value)
        return [str(v) for v in value]
    return str(value)


def _escape_prom_label(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_labels(labels: dict[str, Any]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{_escape_prom_label(v)}"' for k, v in sorted(labels.items())]
    return "{" + ",".join(parts) + "}"


class TelemetryRecorder:
    """
    Enterprise Observability & Telemetry Recorder combining:
    1. OpenTelemetry distributed tracing (`opentelemetry.trace`) with GenAI semantic
       attributes and automatic graceful no-op fallback.
    2. Thread-safe in-process metric counters, latency sums, gauges, structured JSON
       snapshots (`snapshot`), and Prometheus text exposition (`render_prometheus`).
    """

    def __init__(self, service_name: str = "pyharness") -> None:
        self.service_name = service_name
        self._lock = threading.Lock()
        self._tracer = (
            _otel_trace.get_tracer(service_name) if (_HAS_OTEL and _otel_trace is not None) else None
        )

        # Thread-safe metric series
        self._turns: dict[tuple[str, str], int] = defaultdict(int)  # (status, tenant_id)
        self._turn_duration_ms_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._tokens: dict[tuple[str, str, str], int] = defaultdict(int)  # (tenant_id, model, token_type)
        self._cost_usd: dict[tuple[str, str], float] = defaultdict(float)  # (tenant_id, model)
        self._policy_decisions: dict[tuple[str, str, str], int] = defaultdict(int)  # (action, mode, rule_id)
        self._tool_executions: dict[tuple[str, bool], int] = defaultdict(int)  # (tool_name, is_error)
        self._tool_duration_ms_sum: dict[str, float] = defaultdict(float)  # tool_name
        self._security_alerts: dict[tuple[str, str], int] = defaultdict(int)  # (alert_type, tenant_id)
        self._active_approvals: int = 0

    @property
    def otel_enabled(self) -> bool:
        return self._tracer is not None

    @staticmethod
    def _apply_span_attributes(span: Any, attributes: dict[str, Any] | None) -> None:
        if not attributes:
            return
        for raw_key, raw_val in attributes.items():
            if raw_val is None:
                continue
            sanitized = _sanitize_otel_value(raw_val)
            semconv_key = _GENAI_SEMCONV_MAP.get(raw_key, raw_key)
            span.set_attribute(semconv_key, sanitized)
            if semconv_key != raw_key:
                span.set_attribute(raw_key, sanitized)

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """
        Start an OpenTelemetry span (or no-op span fallback), attach all non-None
        attributes using OTel GenAI semantic keys, and record any raised exception.
        """
        if self._tracer is None:
            span = _NoOpSpan(name)
            span.set_attribute("service.name", self.service_name)
            self._apply_span_attributes(span, attributes)
            try:
                yield span
            except BaseException as exc:
                span.record_exception(exc)
                span.set_status("ERROR", str(exc))
                raise
            return

        with self._tracer.start_as_current_span(name) as span:
            span.set_attribute("service.name", self.service_name)
            self._apply_span_attributes(span, attributes)
            try:
                yield span
            except BaseException as exc:
                span.record_exception(exc)
                if _OtelStatus is not None and _OtelStatusCode is not None:
                    span.set_status(_OtelStatus(_OtelStatusCode.ERROR, str(exc)))
                raise

    def record_turn(self, status: str, tenant_id: str, duration_ms: float) -> None:
        with self._lock:
            key = (str(status), str(tenant_id))
            self._turns[key] += 1
            self._turn_duration_ms_sum[key] += max(0.0, float(duration_ms))

    def record_tokens(
        self,
        tenant_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        t_id = str(tenant_id)
        m_name = str(model or "unknown")
        with self._lock:
            self._tokens[(t_id, m_name, "prompt")] += max(0, int(prompt_tokens))
            self._tokens[(t_id, m_name, "completion")] += max(0, int(completion_tokens))
            if cached_tokens > 0 or (t_id, m_name, "cached") in self._tokens:
                self._tokens[(t_id, m_name, "cached")] += max(0, int(cached_tokens))
            self._cost_usd[(t_id, m_name)] += max(0.0, float(cost_usd))

    def record_policy_decision(self, action: str, mode: str, rule_id: str) -> None:
        with self._lock:
            self._policy_decisions[(str(action), str(mode), str(rule_id))] += 1

    def record_tool_execution(
        self,
        tool_name: str,
        is_error: bool,
        duration_ms: float,
    ) -> None:
        t_name = str(tool_name)
        with self._lock:
            self._tool_executions[(t_name, bool(is_error))] += 1
            self._tool_duration_ms_sum[t_name] += max(0.0, float(duration_ms))

    def record_security_alert(self, alert_type: str, tenant_id: str = "default") -> None:
        with self._lock:
            self._security_alerts[(str(alert_type), str(tenant_id))] += 1

    def set_active_approvals(self, count: int) -> None:
        with self._lock:
            self._active_approvals = max(0, int(count))

    def snapshot(self) -> dict[str, Any]:
        """Return a thread-safe JSON-serializable snapshot of all recorded telemetry."""
        with self._lock:
            turns = [
                {
                    "status": status,
                    "tenant_id": tenant_id,
                    "count": count,
                    "duration_ms_sum": round(self._turn_duration_ms_sum[(status, tenant_id)], 3),
                }
                for (status, tenant_id), count in sorted(self._turns.items())
            ]
            tokens = [
                {
                    "tenant_id": tenant_id,
                    "model": model,
                    "type": token_type,
                    "count": count,
                }
                for (tenant_id, model, token_type), count in sorted(self._tokens.items())
            ]
            cost_usd = [
                {
                    "tenant_id": tenant_id,
                    "model": model,
                    "cost_usd": round(cost, 6),
                }
                for (tenant_id, model), cost in sorted(self._cost_usd.items())
            ]
            policy_decisions = [
                {
                    "action": action,
                    "mode": mode,
                    "rule_id": rule_id,
                    "count": count,
                }
                for (action, mode, rule_id), count in sorted(self._policy_decisions.items())
            ]
            tool_executions = [
                {
                    "tool_name": tool_name,
                    "is_error": is_error,
                    "count": count,
                }
                for (tool_name, is_error), count in sorted(self._tool_executions.items())
            ]
            tool_duration_ms = {
                tool_name: round(dur, 3)
                for tool_name, dur in sorted(self._tool_duration_ms_sum.items())
            }
            security_alerts = [
                {
                    "alert_type": alert_type,
                    "tenant_id": tenant_id,
                    "count": count,
                }
                for (alert_type, tenant_id), count in sorted(self._security_alerts.items())
            ]
            active_approvals = self._active_approvals

        return {
            "service_name": self.service_name,
            "otel_enabled": self.otel_enabled,
            "turns": turns,
            "tokens": tokens,
            "cost_usd": cost_usd,
            "policy_decisions": policy_decisions,
            "tool_executions": tool_executions,
            "tool_duration_ms_sum": tool_duration_ms,
            "security_alerts": security_alerts,
            "active_approvals": active_approvals,
        }

    def render_prometheus(self) -> str:
        """Render all telemetry metrics in valid Prometheus text exposition format (v0.0.4)."""
        lines: list[str] = []
        with self._lock:
            lines.append("# HELP pyharness_turns_total Total agent turns completed by status and tenant.")
            lines.append("# TYPE pyharness_turns_total counter")
            for (status, tenant_id), count in sorted(self._turns.items()):
                lbl = _format_labels({"status": status, "tenant_id": tenant_id})
                lines.append(f"pyharness_turns_total{lbl} {count}")

            lines.append("# HELP pyharness_turn_duration_ms_sum Total agent turn wall-clock duration in milliseconds.")
            lines.append("# TYPE pyharness_turn_duration_ms_sum counter")
            for (status, tenant_id), total_ms in sorted(self._turn_duration_ms_sum.items()):
                lbl = _format_labels({"status": status, "tenant_id": tenant_id})
                lines.append(f"pyharness_turn_duration_ms_sum{lbl} {total_ms:.3f}")

            lines.append("# HELP pyharness_tokens_total Total LLM tokens processed by tenant, model, and type.")
            lines.append("# TYPE pyharness_tokens_total counter")
            for (tenant_id, model, token_type), count in sorted(self._tokens.items()):
                lbl = _format_labels({"tenant_id": tenant_id, "model": model, "type": token_type})
                lines.append(f"pyharness_tokens_total{lbl} {count}")

            lines.append("# HELP pyharness_cost_usd_total Cumulative estimated LLM spend in USD by tenant and model.")
            lines.append("# TYPE pyharness_cost_usd_total counter")
            for (tenant_id, model), cost in sorted(self._cost_usd.items()):
                lbl = _format_labels({"tenant_id": tenant_id, "model": model})
                lines.append(f"pyharness_cost_usd_total{lbl} {cost:.6f}")

            lines.append("# HELP pyharness_policy_decisions_total Total governance policy decisions by action, mode, and rule.")
            lines.append("# TYPE pyharness_policy_decisions_total counter")
            for (action, mode, rule_id), count in sorted(self._policy_decisions.items()):
                lbl = _format_labels({"action": action, "mode": mode, "rule_id": rule_id})
                lines.append(f"pyharness_policy_decisions_total{lbl} {count}")

            lines.append("# HELP pyharness_tool_executions_total Total tool invocations by tool name and error status.")
            lines.append("# TYPE pyharness_tool_executions_total counter")
            for (tool_name, is_error), count in sorted(self._tool_executions.items()):
                lbl = _format_labels({"tool_name": tool_name, "is_error": "true" if is_error else "false"})
                lines.append(f"pyharness_tool_executions_total{lbl} {count}")

            lines.append("# HELP pyharness_tool_duration_ms_sum Cumulative tool execution latency in milliseconds.")
            lines.append("# TYPE pyharness_tool_duration_ms_sum counter")
            for tool_name, total_ms in sorted(self._tool_duration_ms_sum.items()):
                lbl = _format_labels({"tool_name": tool_name})
                lines.append(f"pyharness_tool_duration_ms_sum{lbl} {total_ms:.3f}")

            lines.append("# HELP pyharness_security_alerts_total Total security guardrail alerts by alert type and tenant.")
            lines.append("# TYPE pyharness_security_alerts_total counter")
            for (alert_type, tenant_id), count in sorted(self._security_alerts.items()):
                lbl = _format_labels({"alert_type": alert_type, "tenant_id": tenant_id})
                lines.append(f"pyharness_security_alerts_total{lbl} {count}")

            lines.append("# HELP pyharness_active_approvals Current count of pending human-in-the-loop approvals.")
            lines.append("# TYPE pyharness_active_approvals gauge")
            lines.append(f"pyharness_active_approvals {self._active_approvals}")

        return "\n".join(lines) + "\n"
