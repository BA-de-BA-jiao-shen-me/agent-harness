from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from pyharness.runtime.engine import ToolRegistry
from pyharness.runtime.schemas import RiskLevel, ToolSpec

_SAFE_BASE_ENV_KEYS = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
)
_WINDOWS_REQUIRED_ENV_KEYS = (
    "SystemRoot",
    "COMSPEC",
    "PATHEXT",
    "WINDIR",
)


@dataclass
class MCPToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk_level: RiskLevel = RiskLevel.MEDIUM


class StdioMCPTransport:
    """
    Production stdio JSON-RPC 2.0 transport for local/sidecar MCP servers.

    Security & reliability guarantees:
    - Scrubs child environment to a minimal OS allowlist so parent secrets (e.g. LLM_API_KEY)
      never leak into untrusted third-party MCP servers.
    - On Windows (`os.name == "nt"`), preserves `SystemRoot` and essential OS vars so
      Python/Node child processes initialize cleanly.
    - Multiplexes concurrent RPCs over a single persistent subprocess by matching JSON-RPC `id`s
      to pending `asyncio.Future`s in a background stdout reader loop.
    - Drains stderr continuously to prevent pipe deadlocks and fails pending futures fast on EOF.
    - Guarantees zero zombie processes on `aclose()` via stdin close -> terminate -> kill fallback.
    """

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not command:
            raise ValueError("StdioMCPTransport requires a non-empty command list.")
        self.command = list(command)
        self.env = dict(env) if env else None
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds

        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[Any, asyncio.Future[dict[str, Any]]] = {}
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=50)

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def _build_scrubbed_env(self) -> dict[str, str]:
        child_env: dict[str, str] = {}
        keys = list(_SAFE_BASE_ENV_KEYS)
        if os.name == "nt":
            keys.extend(_WINDOWS_REQUIRED_ENV_KEYS)

        # Case-insensitive lookup helper so Windows `Path` / `SYSTEMROOT` work seamlessly
        env_lookup = (
            {k.upper(): v for k, v in os.environ.items()}
            if os.name == "nt"
            else dict(os.environ)
        )
        for key in keys:
            val = env_lookup.get(key.upper()) if os.name == "nt" else env_lookup.get(key)
            if val is not None:
                child_env[key] = val

        child_env.setdefault("PYTHONUNBUFFERED", "1")
        child_env.setdefault("PYTHONIOENCODING", "utf-8")

        if self.env:
            for k, v in self.env.items():
                child_env[str(k)] = str(v)
        return child_env

    async def start(self) -> None:
        async with self._start_lock:
            if self.is_running:
                return
            self._stderr_tail.clear()
            self._proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=self._build_scrubbed_env(),
                limit=10 * 1024 * 1024,
            )
            self._reader_task = asyncio.create_task(self._read_loop())
            self._stderr_task = asyncio.create_task(self._drain_stderr())

    def _fail_all_pending(self, reason: str) -> None:
        stderr_hint = ""
        if self._stderr_tail:
            stderr_hint = f" | stderr: {' ; '.join(list(self._stderr_tail)[-5:])}"
        err = RuntimeError(f"{reason}{stderr_hint}")
        pending_items = list(self._pending.items())
        self._pending.clear()
        for _, fut in pending_items:
            if not fut.done():
                fut.set_exception(err)

    async def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self._stderr_tail.append(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                raw_line = await self._proc.stdout.readline()
                if not raw_line:
                    self._fail_all_pending("Stdio MCP server closed stdout unexpectedly")
                    break
                text = raw_line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    # Ignore stray non-JSON log lines written to stdout by child servers
                    continue
                if not isinstance(msg, dict):
                    continue
                if "id" in msg and msg["id"] is not None:
                    fut = self._pending.pop(msg["id"], None)
                    if fut is not None and not fut.done():
                        fut.set_result(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_all_pending(f"Stdio MCP reader loop error: {exc}")

    async def _write_json_line(self, payload: dict[str, Any]) -> None:
        if not self.is_running:
            await self.start()
        assert self._proc is not None and self._proc.stdin is not None
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            if self._proc is None or self._proc.stdin is None or self._proc.returncode is not None:
                raise RuntimeError("Stdio MCP process is not running.")
            self._proc.stdin.write(encoded)
            await self._proc.stdin.drain()

    async def send_rpc(
        self,
        request: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if "id" not in request or request["id"] is None:
            raise ValueError("JSON-RPC request must include a non-null 'id'.")
        if not self.is_running:
            await self.start()

        req_id = request["id"]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut
        eff_timeout = self.timeout_seconds if timeout is None else timeout
        try:
            await self._write_json_line(request)
            return await asyncio.wait_for(fut, timeout=eff_timeout)
        finally:
            self._pending.pop(req_id, None)

    async def send_notification(self, notification: dict[str, Any]) -> None:
        await self._write_json_line(notification)

    async def aclose(self) -> None:
        proc = self._proc
        reader_task = self._reader_task
        stderr_task = self._stderr_task
        self._proc = None
        self._reader_task = None
        self._stderr_task = None

        self._fail_all_pending("StdioMCPTransport closed")

        for task in (reader_task, stderr_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (reader_task, stderr_task):
            if task is not None:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        if proc is None:
            return

        if proc.stdin is not None:
            try:
                proc.stdin.close()
                wait_closed = getattr(proc.stdin, "wait_closed", None)
                if callable(wait_closed):
                    await asyncio.wait_for(wait_closed(), timeout=0.5)
            except Exception:
                pass

        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    await proc.wait()

    async def __aenter__(self) -> StdioMCPTransport:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()


class InMemoryMCPServer:
    """
    Standards-compliant Model Context Protocol (MCP) JSON-RPC 2.0 server implementation.
    Supports `initialize`, `notifications/initialized`, `tools/list`, and `tools/call`.
    """

    def __init__(self, server_name: str = "pyharness-enterprise-mcp") -> None:
        self.server_name = server_name
        self._defs: dict[str, MCPToolDefinition] = {}
        self._handlers: dict[str, Callable[..., Awaitable[Any] | Any]] = {}
        self.initialized = False

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., Awaitable[Any] | Any],
        risk_level: RiskLevel = RiskLevel.MEDIUM,
    ) -> None:
        self._defs[name] = MCPToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            risk_level=risk_level,
        )
        self._handlers[name] = handler

    async def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any] | None:
        req_id = request.get("id")
        method = request.get("method") or ""
        params = request.get("params") or {}

        # JSON-RPC 2.0 notifications (e.g., `notifications/initialized`) carry no `id`
        if method.startswith("notifications/"):
            if method == "notifications/initialized":
                self.initialized = True
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": self.server_name, "version": "0.2.0"},
                    "capabilities": {"tools": {}},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": d.name,
                            "description": d.description,
                            "inputSchema": d.input_schema,
                            "annotations": {"riskLevel": d.risk_level.value},
                        }
                        for d in self._defs.values()
                    ]
                },
            }

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}
            handler = self._handlers.get(tool_name)
            if handler is None:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"MCP tool not found: {tool_name}"},
                }
            try:
                res = handler(**arguments)
                if hasattr(res, "__await__"):
                    res = await res
                text = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": text}],
                        "isError": False,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"MCP tool error: {exc}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unsupported MCP method: {method}"},
        }


class MCPClientBridge:
    """
    Enterprise MCP Client Bridge (inspired by DSH askai-tool-bridge-plugin.mjs).

    Supports three transports:
    1. `InMemoryMCPServer` (local in-process server / tests)
    2. Streamable HTTP (`http_endpoint`) with custom headers, dynamic OAuth/Bearer
       `token_provider`, and stateful `Mcp-Session-Id` propagation across requests.
    3. Local/sidecar `StdioMCPTransport` over scrubbed subprocess pipes.

    Completes standard MCP handshake (`initialize` -> `notifications/initialized` -> `tools/list`),
    maps risk annotations to PyHarness `RiskLevel`, and mounts governed proxies into `ToolRegistry`.
    """

    def __init__(
        self,
        server: InMemoryMCPServer | None = None,
        http_endpoint: str | None = None,
        stdio_transport: StdioMCPTransport | None = None,
        namespace_prefix: str = "mcp_",
        headers: dict[str, str] | None = None,
        token_provider: Callable[[], str | None] | None = None,
        timeout_seconds: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        transports = sum(t is not None for t in (server, http_endpoint, stdio_transport))
        if transports == 0:
            raise ValueError(
                "Provide one of server (InMemoryMCPServer), http_endpoint, or stdio_transport."
            )
        if transports > 1:
            raise ValueError(
                "Provide only one MCP transport: server, http_endpoint, or stdio_transport."
            )

        self._server = server
        self._http_endpoint = http_endpoint
        self._stdio = stdio_transport
        self._prefix = namespace_prefix
        self._headers = dict(headers or {})
        self._token_provider = token_provider
        self._timeout_seconds = timeout_seconds
        self._session_id: str | None = None
        self._rpc_id = 0
        self._owns_client = http_client is None and http_endpoint is not None
        self._client = (
            http_client
            or (httpx.AsyncClient(timeout=timeout_seconds) if http_endpoint else None)
        )

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def _build_http_headers(self) -> dict[str, str]:
        hdrs: dict[str, str] = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        hdrs.update(self._headers)
        if self._session_id:
            hdrs["Mcp-Session-Id"] = self._session_id
        if self._token_provider is not None:
            token = self._token_provider()
            if token:
                hdrs["Authorization"] = (
                    token if token.startswith("Bearer ") else f"Bearer {token}"
                )
        return hdrs

    def _capture_http_session(self, resp: httpx.Response) -> None:
        mcp_sid = resp.headers.get("Mcp-Session-Id")
        if mcp_sid:
            self._session_id = mcp_sid

    @staticmethod
    def _parse_http_jsonrpc(resp: httpx.Response) -> dict[str, Any]:
        content_type = (resp.headers.get("content-type") or "").lower()
        if "text/event-stream" in content_type:
            last_obj: dict[str, Any] = {}
            for raw_line in resp.text.splitlines():
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                parsed = json.loads(data_str)
                if isinstance(parsed, dict):
                    last_obj = parsed
            return last_obj
        if not resp.content:
            return {}
        return resp.json()

    async def _send_rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._rpc_id += 1
        req = {"jsonrpc": "2.0", "id": self._rpc_id, "method": method, "params": params or {}}
        if self._server is not None:
            res = await self._server.handle_jsonrpc(req)
            return res or {}
        if self._stdio is not None:
            return await self._stdio.send_rpc(req, timeout=self._timeout_seconds)

        assert self._http_endpoint is not None and self._client is not None
        resp = await self._client.post(
            self._http_endpoint,
            json=req,
            headers=self._build_http_headers(),
        )
        self._capture_http_session(resp)
        resp.raise_for_status()
        return self._parse_http_jsonrpc(resp)

    async def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        notif: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            notif["params"] = params
        if self._server is not None:
            await self._server.handle_jsonrpc(notif)
            return
        if self._stdio is not None:
            await self._stdio.send_notification(notif)
            return

        assert self._http_endpoint is not None and self._client is not None
        resp = await self._client.post(
            self._http_endpoint,
            json=notif,
            headers=self._build_http_headers(),
        )
        self._capture_http_session(resp)
        resp.raise_for_status()

    async def mount_into_registry(
        self,
        registry: ToolRegistry,
        risk_overrides: dict[str, RiskLevel] | None = None,
    ) -> list[str]:
        """Discover remote/local/stdio MCP tools and register governed proxy handlers."""
        if self._stdio is not None and not self._stdio.is_running:
            await self._stdio.start()

        await self._send_rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pyharness-mcp-client", "version": "0.3.0"},
            },
        )
        await self._send_notification("notifications/initialized")
        listing = await self._send_rpc("tools/list")
        tools_meta = listing.get("result", {}).get("tools", [])
        mounted_names: list[str] = []
        overrides = risk_overrides or {}

        for item in tools_meta:
            raw_name = item["name"]
            exposed_name = (
                f"{self._prefix}{raw_name}"
                if not raw_name.startswith(self._prefix)
                else raw_name
            )
            ann_risk = (item.get("annotations") or {}).get("riskLevel", RiskLevel.MEDIUM.value)
            try:
                default_risk = RiskLevel(ann_risk)
            except ValueError:
                default_risk = RiskLevel.MEDIUM
            risk = overrides.get(raw_name, default_risk)

            async def _proxy_call(_raw_name: str = raw_name, **kwargs: Any) -> str:
                rpc_res = await self._send_rpc(
                    "tools/call", {"name": _raw_name, "arguments": kwargs}
                )
                if "error" in rpc_res:
                    raise RuntimeError(rpc_res["error"].get("message", "Unknown MCP RPC error"))
                result = rpc_res.get("result") or {}
                texts = [
                    c.get("text", "")
                    for c in result.get("content", [])
                    if c.get("type") == "text"
                ]
                joined = "\n".join(texts)
                if result.get("isError"):
                    raise RuntimeError(joined)
                return joined

            registry.register(
                ToolSpec(
                    name=exposed_name,
                    description=f"[MCP] {item.get('description', '')}",
                    parameters=item.get("inputSchema") or {"type": "object", "properties": {}},
                    risk_level=risk,
                    timeout_seconds=self._timeout_seconds,
                ),
                _proxy_call,
            )
            mounted_names.append(exposed_name)

        return mounted_names

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        if self._stdio is not None:
            await self._stdio.aclose()

    async def __aenter__(self) -> MCPClientBridge:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()


def create_builtin_enterprise_mcp_server() -> InMemoryMCPServer:
    """Sample enterprise MCP server exposing incident lookup and sensitive alert dispatch."""
    server = InMemoryMCPServer("enterprise-ops-mcp")

    def list_incidents(severity: str = "high") -> list[dict[str, Any]]:
        data = [
            {"id": "INC-2026-01", "site": "金山石化A区", "severity": "high", "status": "open", "issue": "压力阀温升异常"},
            {"id": "INC-2026-02", "site": "张江配电站", "severity": "medium", "status": "resolved", "issue": "断路器辅助触点氧化"},
            {"id": "INC-2026-03", "site": "金山石化B区", "severity": "high", "status": "open", "issue": "气体监测探头零漂"},
        ]
        if severity == "all":
            return data
        return [d for d in data if d["severity"] == severity.lower()]

    def dispatch_emergency_alert(incident_id: str, channel: str = "ops-oncall") -> dict[str, Any]:
        return {
            "dispatched": True,
            "incident_id": incident_id,
            "channel": channel,
            "message": f"Emergency alert sent for {incident_id} to {channel}",
        }

    server.register_tool(
        name="list_incidents",
        description="查询企业运维工单与安全隐患列表（支持按 severity=high/medium/all 过滤）",
        input_schema={
            "type": "object",
            "properties": {"severity": {"type": "string", "enum": ["high", "medium", "all"]}},
            "required": [],
        },
        handler=list_incidents,
        risk_level=RiskLevel.LOW,
    )
    server.register_tool(
        name="dispatch_emergency_alert",
        description="向值班告警群发送高危紧急停机/整改告警（高危外部副作用，需人工审批）",
        input_schema={
            "type": "object",
            "properties": {
                "incident_id": {"type": "string"},
                "channel": {"type": "string"},
            },
            "required": ["incident_id"],
        },
        handler=dispatch_emergency_alert,
        risk_level=RiskLevel.HIGH,
    )
    return server
