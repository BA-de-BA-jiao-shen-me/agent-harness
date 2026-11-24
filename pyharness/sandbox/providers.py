from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


class SandboxConfigurationError(RuntimeError):
    """Raised when an explicitly requested SandboxProvider is misconfigured or unreachable (fail-closed)."""


class SandboxTimeoutError(TimeoutError):
    """Raised when isolated sandbox execution exceeds its wall-clock budget."""


@dataclass
class SandboxExecutionResult:
    stdout: str
    result: Any
    provider: str


# Self-contained guest runner executed inside Docker Container, E2B MicroVM, or Isolated Subprocess.
# Performs second-layer AST verification inside the guest + captures stdout cleanly per process.
_GUEST_RUNNER_SCRIPT = r"""
import ast, base64, io, json, math, statistics, sys
from contextlib import redirect_stdout

_ALLOWED_IMPORTS = {"math", "statistics", "json", "datetime", "collections", "re"}
_FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "breakpoint", "input", "dir", "type",
}
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "int": int,
    "len": len, "list": list, "map": map, "max": max, "min": min, "pow": pow,
    "print": print, "range": range, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}

def _audit(code: str):
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    raise PermissionError(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in _ALLOWED_IMPORTS:
                raise PermissionError(f"Forbidden import from: {node.module}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in {"format", "format_map"}:
                raise PermissionError(f"Forbidden attribute access: {node.attr}")
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in _FORBIDDEN_CALLS:
                raise PermissionError(f"Forbidden identifier: {node.id}")
    return tree

def _safe_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root not in _ALLOWED_IMPORTS:
        raise PermissionError(f"Forbidden import: {name}")
    fromlist = args[2] if len(args) > 2 else kwargs.get("fromlist", ())
    return __import__(name, None, None, fromlist or ())

def main():
    raw_b64 = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
    code = base64.b64decode(raw_b64.encode("ascii")).decode("utf-8")
    try:
        tree = _audit(code)
        safe_globals = {
            "__builtins__": {**_SAFE_BUILTINS, "__import__": _safe_import},
            "math": math,
            "statistics": statistics,
            "json": json,
        }
        local_ns = {}
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(tree, "<isolated_sandbox>", "exec"), safe_globals, local_ns)
        res = local_ns.get("result")
        if not isinstance(res, (int, float, str, list, dict, bool, type(None))):
            res = repr(res)
        sys.stdout.write(json.dumps({"ok": True, "stdout": buf.getvalue().strip(), "result": res}, ensure_ascii=False))
    except Exception as exc:
        sys.stdout.write(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
"""


def _decode_docker_multiplexed_logs(raw: bytes) -> str:
    """Decode Docker Engine multiplexed stdout/stderr stream frames (8-byte header per frame)."""
    if not raw:
        return ""
    # If raw response is already plain JSON (e.g. TTY or proxy), return directly
    stripped = raw.strip()
    if stripped.startswith(b"{") and stripped.endswith(b"}"):
        return stripped.decode("utf-8", errors="replace")

    chunks: list[bytes] = []
    offset = 0
    n = len(raw)
    while offset + 8 <= n:
        stream_type = raw[offset]
        if stream_type not in (0, 1, 2):
            return raw.decode("utf-8", errors="replace").strip()
        frame_len = struct.unpack(">I", raw[offset + 4 : offset + 8])[0]
        offset += 8
        if offset + frame_len > n:
            chunks.append(raw[offset:])
            break
        if stream_type == 1:  # stdout
            chunks.append(raw[offset : offset + frame_len])
        offset += frame_len
    return b"".join(chunks).decode("utf-8", errors="replace").strip()


class SandboxProvider(ABC):
    """
    Cordis-style Pluggable Sandbox Seam (`@deepseek-ai/dsh-sandbox-*` equivalent).
    Decouples the model-facing `run_python_sandbox` tool from the underlying isolation boundary.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, code: str, timeout_seconds: float = 10.0) -> SandboxExecutionResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class SubprocessSandboxProvider(SandboxProvider):
    """
    Hardened Out-of-Process Isolated Python Sandbox (for zero-Docker local dev & CI).

    Protections over legacy in-process `exec()`:
    - Runs in a separate OS process via `python -I -B` (isolated mode).
    - Scrubs host environment variables (`LLM_API_KEY` etc. do NOT exist in child env).
    - Isolates `sys.stdout` per process (zero cross-session thread bleed).
    - Terminates runaway CPU/infinite loops immediately via kernel `proc.kill()`.
    """

    @property
    def provider_name(self) -> str:
        return "subprocess"

    @staticmethod
    def _minimal_clean_env() -> dict[str, str]:
        if os.name == "nt":
            # Windows CPython requires SystemRoot to initialize BCrypt RNG on startup; zero secrets passed.
            return {"SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"), "PATH": ""}
        return {"LC_ALL": "C.UTF-8", "PATH": ""}

    async def execute(self, code: str, timeout_seconds: float = 10.0) -> SandboxExecutionResult:
        b64_payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-B",
            "-c",
            _GUEST_RUNNER_SCRIPT,
            b64_payload,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._minimal_clean_env(),
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            if isinstance(exc, asyncio.TimeoutError):
                raise SandboxTimeoutError(
                    f"Subprocess sandbox timed out and was killed after {timeout_seconds}s"
                ) from exc
            raise

        raw_out = stdout_bytes.decode("utf-8", errors="replace").strip()
        if not raw_out:
            err_txt = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Sandbox child process failed (code={proc.returncode}): {err_txt}")

        parsed = json.loads(raw_out)
        if not parsed.get("ok"):
            raise RuntimeError(f"{parsed.get('error_type', 'SandboxError')}: {parsed.get('error')}")
        return SandboxExecutionResult(
            stdout=parsed.get("stdout", ""),
            result=parsed.get("result"),
            provider=self.provider_name,
        )


class DockerSandboxProvider(SandboxProvider):
    """
    Enterprise Ephemeral Container Sandbox via Docker Engine REST API (supports gVisor `runsc`).

    Security posture:
    - `NetworkDisabled=True` & `NetworkMode="none"`: blocks outbound internet AND loopback
      control-plane attacks (mitigating CVE-2026-82533 class vulnerabilities).
    - `ReadonlyRootfs=True` + `noexec,nosuid` `/tmp` tmpfs.
    - `CapDrop=["ALL"]`, `no-new-privileges:true`, unprivileged `65534:65534` (`nobody`).
    - Hard cgroups limits: `Memory=256MB`, `MemorySwap=256MB`, `NanoCpus=0.5`, `PidsLimit=32`.
    - Guaranteed `DELETE /containers/{id}?force=true` cleanup on completion, timeout, or cancel.
    """

    def __init__(
        self,
        docker_host: str | None = None,
        image: str = "python:3.11-slim",
        runtime: str | None = None,
        memory_bytes: int = 256 * 1024 * 1024,
        nano_cpus: int = 500_000_000,
        pids_limit: int = 32,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.image = image
        self.runtime = (runtime or "").strip() or None
        self.memory_bytes = memory_bytes
        self.nano_cpus = nano_cpus
        self.pids_limit = pids_limit
        self._owns_client = http_client is None

        host = (docker_host or os.getenv("DOCKER_HOST") or "").strip()
        if http_client is not None:
            self._client = http_client
            self._base_url = "http://docker-engine"
        elif host.startswith(("tcp://", "http://", "https://")):
            normalized = "http://" + host.removeprefix("tcp://") if host.startswith("tcp://") else host
            self._base_url = normalized.rstrip("/")
            self._client = httpx.AsyncClient(timeout=30.0)
        else:
            uds_path = host.removeprefix("unix://") if host.startswith("unix://") else "/var/run/docker.sock"
            transport = httpx.AsyncHTTPTransport(uds=uds_path)
            self._base_url = "http://docker-engine"
            self._client = httpx.AsyncClient(transport=transport, timeout=30.0)

    @property
    def provider_name(self) -> str:
        return f"docker+{self.runtime}" if self.runtime else "docker"

    async def ping(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/_ping", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def execute(self, code: str, timeout_seconds: float = 10.0) -> SandboxExecutionResult:
        b64_payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
        host_config: dict[str, Any] = {
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=64m"},
            "Memory": self.memory_bytes,
            "MemorySwap": self.memory_bytes,
            "NanoCpus": self.nano_cpus,
            "PidsLimit": self.pids_limit,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges:true"],
        }
        if self.runtime:
            host_config["Runtime"] = self.runtime

        create_body = {
            "Image": self.image,
            "Cmd": ["python3", "-I", "-B", "-c", _GUEST_RUNNER_SCRIPT, b64_payload],
            "User": "65534:65534",
            "NetworkDisabled": True,
            "Env": [],
            "AttachStdout": True,
            "AttachStderr": True,
            "Tty": False,
            "HostConfig": host_config,
        }

        create_resp = await self._client.post(
            f"{self._base_url}/containers/create", json=create_body
        )
        create_resp.raise_for_status()
        container_id = create_resp.json()["Id"]

        try:
            start_resp = await self._client.post(f"{self._base_url}/containers/{container_id}/start")
            start_resp.raise_for_status()

            try:
                wait_resp = await asyncio.wait_for(
                    self._client.post(
                        f"{self._base_url}/containers/{container_id}/wait",
                        timeout=timeout_seconds + 2.0,
                    ),
                    timeout=timeout_seconds,
                )
                wait_resp.raise_for_status()
            except asyncio.TimeoutError as exc:
                raise SandboxTimeoutError(
                    f"Docker sandbox container exceeded {timeout_seconds}s and was force-destroyed"
                ) from exc

            logs_resp = await self._client.get(
                f"{self._base_url}/containers/{container_id}/logs",
                params={"stdout": "true", "stderr": "true"},
            )
            logs_resp.raise_for_status()
            decoded = _decode_docker_multiplexed_logs(logs_resp.content)
            if not decoded:
                raise RuntimeError("Docker sandbox produced empty output (possible OOM kill).")

            last_line = decoded.splitlines()[-1].strip()
            parsed = json.loads(last_line)
            if not parsed.get("ok"):
                raise RuntimeError(f"{parsed.get('error_type', 'SandboxError')}: {parsed.get('error')}")
            return SandboxExecutionResult(
                stdout=parsed.get("stdout", ""),
                result=parsed.get("result"),
                provider=self.provider_name,
            )
        finally:
            try:
                await self._client.delete(
                    f"{self._base_url}/containers/{container_id}",
                    params={"force": "true", "v": "true"},
                )
            except Exception:
                pass

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class E2BSandboxProvider(SandboxProvider):
    """
    Remote Firecracker MicroVM Sandbox Provider via E2B REST & Execution API.

    Ideal for untrusted multi-tenant SaaS workloads requiring hardware-virtualized guest kernels.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.e2b.dev",
        template_id: str = "code-interpreter-v1",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise SandboxConfigurationError("E2BSandboxProvider requires a non-empty E2B_API_KEY.")
        self.api_key = api_key.strip()
        self.api_url = api_url.rstrip("/")
        self.template_id = template_id
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=30.0)

    @property
    def provider_name(self) -> str:
        return "e2b-microvm"

    async def execute(self, code: str, timeout_seconds: float = 10.0) -> SandboxExecutionResult:
        b64_payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
        headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}

        create_resp = await self._client.post(
            f"{self.api_url}/sandboxes",
            headers=headers,
            json={"templateID": self.template_id, "timeout": max(30, int(timeout_seconds) + 15)},
        )
        create_resp.raise_for_status()
        sb_info = create_resp.json()
        sandbox_id = sb_info["sandboxID"]
        exec_url = sb_info.get("execUrl") or f"{self.api_url}/sandboxes/{sandbox_id}/code/execution"
        envd_token = sb_info.get("envdAccessToken", "")

        try:
            exec_headers = dict(headers)
            if envd_token:
                exec_headers["X-Access-Token"] = envd_token

            exec_resp = await asyncio.wait_for(
                self._client.post(
                    exec_url,
                    headers=exec_headers,
                    json={
                        "language": "python",
                        "cmd": ["python3", "-I", "-B", "-c", _GUEST_RUNNER_SCRIPT, b64_payload],
                        "code": code,
                    },
                    timeout=timeout_seconds + 2.0,
                ),
                timeout=timeout_seconds,
            )
            exec_resp.raise_for_status()
            body = exec_resp.json()

            if "stdout" in body and isinstance(body["stdout"], str) and body["stdout"].strip().startswith("{"):
                parsed = json.loads(body["stdout"].strip().splitlines()[-1])
                if not parsed.get("ok"):
                    raise RuntimeError(f"{parsed.get('error_type', 'SandboxError')}: {parsed.get('error')}")
                return SandboxExecutionResult(
                    stdout=parsed.get("stdout", ""),
                    result=parsed.get("result"),
                    provider=self.provider_name,
                )

            if body.get("error"):
                raise RuntimeError(f"E2B Execution Error: {body['error']}")
            return SandboxExecutionResult(
                stdout=str(body.get("stdout", "")).strip(),
                result=body.get("result"),
                provider=self.provider_name,
            )
        except asyncio.TimeoutError as exc:
            raise SandboxTimeoutError(
                f"E2B MicroVM execution timed out after {timeout_seconds}s"
            ) from exc
        finally:
            try:
                await self._client.delete(
                    f"{self.api_url}/sandboxes/{sandbox_id}", headers=headers
                )
            except Exception:
                pass

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def create_sandbox_provider(
    mode: str = "auto",
    *,
    docker_host: str | None = None,
    docker_image: str | None = None,
    docker_runtime: str | None = None,
    e2b_api_key: str | None = None,
    e2b_api_url: str | None = None,
) -> SandboxProvider:
    """
    Factory resolving the active SandboxProvider seam.
    - Explicit modes (`docker`, `e2b`) fail closed if required credentials/endpoints are missing.
    - `auto` selects `e2b` -> `docker` -> `subprocess` gracefully for zero-config local/CI use.
    """
    normalized = (mode or "auto").strip().lower()
    resolved_e2b_key = (e2b_api_key if e2b_api_key is not None else os.getenv("E2B_API_KEY", "")).strip()
    resolved_e2b_url = (e2b_api_url or os.getenv("E2B_API_URL") or "https://api.e2b.dev").strip()
    resolved_docker_host = (docker_host if docker_host is not None else os.getenv("DOCKER_HOST", "")).strip()
    resolved_image = (docker_image or os.getenv("DOCKER_SANDBOX_IMAGE") or "python:3.11-slim").strip()
    resolved_runtime = (docker_runtime if docker_runtime is not None else os.getenv("DOCKER_SANDBOX_RUNTIME", "")).strip() or None

    if normalized == "e2b":
        return E2BSandboxProvider(api_key=resolved_e2b_key, api_url=resolved_e2b_url)

    if normalized == "docker":
        if not resolved_docker_host and not Path("/var/run/docker.sock").exists():
            raise SandboxConfigurationError(
                "HARNESS_SANDBOX_PROVIDER='docker' requested, but neither DOCKER_HOST nor "
                "/var/run/docker.sock is available (fail-closed)."
            )
        return DockerSandboxProvider(
            docker_host=resolved_docker_host or None,
            image=resolved_image,
            runtime=resolved_runtime,
        )

    if normalized == "subprocess":
        return SubprocessSandboxProvider()

    if normalized == "auto":
        if resolved_e2b_key:
            return E2BSandboxProvider(api_key=resolved_e2b_key, api_url=resolved_e2b_url)
        if resolved_docker_host or Path("/var/run/docker.sock").exists():
            return DockerSandboxProvider(
                docker_host=resolved_docker_host or None,
                image=resolved_image,
                runtime=resolved_runtime,
            )
        return SubprocessSandboxProvider()

    raise SandboxConfigurationError(
        f"Unknown sandbox provider mode '{mode}'. Choose from: auto, docker, e2b, subprocess."
    )
