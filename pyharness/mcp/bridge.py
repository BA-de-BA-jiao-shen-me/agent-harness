from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from pyharness.runtime.engine import ToolRegistry
from pyharness.runtime.schemas import RiskLevel, ToolSpec


@dataclass
class MCPToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk_level: RiskLevel = RiskLevel.MEDIUM


class InMemoryMCPServer:
    """
    Standards-compliant Model Context Protocol (MCP) JSON-RPC 2.0 server implementation.
    Supports `initialize`, `tools/list`, and `tools/call` — usable both in-process for
    zero-latency local execution and mounted over HTTP Streamable MCP endpoints.
    """

    def __init__(self, server_name: str = "pyharness-enterprise-mcp") -> None:
        self.server_name = server_name
        self._defs: dict[str, MCPToolDefinition] = {}
        self._handlers: dict[str, Callable[..., Awaitable[Any] | Any]] = {}

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

    async def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": self.server_name, "version": "0.1.0"},
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
    MCP Client Bridge (inspired by DSH askai-tool-bridge-plugin.mjs).
    Discovers tools from either an in-process MCP server or a remote HTTP JSON-RPC MCP endpoint,
    maps risk annotations to PyHarness `RiskLevel`, and mounts them into `ToolRegistry`.
    """

    def __init__(
        self,
        server: InMemoryMCPServer | None = None,
        http_endpoint: str | None = None,
        namespace_prefix: str = "mcp_",
    ) -> None:
        if server is None and http_endpoint is None:
            raise ValueError("Provide either an InMemoryMCPServer or an http_endpoint.")
        self._server = server
        self._http_endpoint = http_endpoint
        self._prefix = namespace_prefix
        self._rpc_id = 0

    async def _send_rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._rpc_id += 1
        req = {"jsonrpc": "2.0", "id": self._rpc_id, "method": method, "params": params or {}}
        if self._server is not None:
            return await self._server.handle_jsonrpc(req)
        assert self._http_endpoint is not None
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self._http_endpoint, json=req)
            resp.raise_for_status()
            return resp.json()

    async def mount_into_registry(
        self,
        registry: ToolRegistry,
        risk_overrides: dict[str, RiskLevel] | None = None,
    ) -> list[str]:
        """Discover remote/local MCP tools and register governed proxy handlers."""
        await self._send_rpc("initialize")
        listing = await self._send_rpc("tools/list")
        tools_meta = listing.get("result", {}).get("tools", [])
        mounted_names: list[str] = []
        overrides = risk_overrides or {}

        for item in tools_meta:
            raw_name = item["name"]
            exposed_name = f"{self._prefix}{raw_name}" if not raw_name.startswith(self._prefix) else raw_name
            ann_risk = (item.get("annotations") or {}).get("riskLevel", RiskLevel.MEDIUM.value)
            risk = overrides.get(raw_name, RiskLevel(ann_risk))

            async def _proxy_call(_raw_name: str = raw_name, **kwargs: Any) -> str:
                rpc_res = await self._send_rpc(
                    "tools/call", {"name": _raw_name, "arguments": kwargs}
                )
                if "error" in rpc_res:
                    raise RuntimeError(rpc_res["error"].get("message", "Unknown MCP RPC error"))
                result = rpc_res["result"]
                texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
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
                ),
                _proxy_call,
            )
            mounted_names.append(exposed_name)

        return mounted_names


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
