from __future__ import annotations

import hmac
import ipaddress
import secrets
import time
import urllib.parse
import uuid
from typing import Any

import httpx
from pydantic import BaseModel, Field

_REDACTED_RELAY_SECRET = "[REDACTED_RELAY_SECRET]"

_STRIPPED_REQUEST_HEADERS: frozenset[str] = frozenset(
    {"authorization", "x-api-key", "cookie", "proxy-authorization"}
)

_STRIPPED_RESPONSE_HEADERS: frozenset[str] = frozenset(
    {"set-cookie", "authorization"}
)

_BLOCKED_SSRF_HOSTS: frozenset[str] = frozenset(
    {"localhost", "metadata.google.internal", "169.254.169.254"}
)


class CredentialRelayError(PermissionError):
    """Raised when a credential relay lease, target, or destination check fails."""


class CredentialTarget(BaseModel):
    """Host-side upstream credential target; secret material is never exposed to sandboxes."""

    name: str
    secret_value: str = Field(..., repr=False)
    allowed_hosts: list[str] = Field(default_factory=list)
    inject_header: str = "Authorization"
    inject_prefix: str = "Bearer "
    tenant_id: str | None = None
    allow_private_ips: bool = False
    allow_http: bool = False

    def safe_summary(self) -> dict[str, Any]:
        """Return a sanitized metadata view that never includes `secret_value`."""
        return {
            "name": self.name,
            "allowed_hosts": list(self.allowed_hosts),
            "inject_header": self.inject_header,
            "inject_prefix": self.inject_prefix,
            "tenant_id": self.tenant_id,
            "allow_private_ips": self.allow_private_ips,
            "allow_http": self.allow_http,
        }


class CredentialLease(BaseModel):
    """Short-lived, call-budgeted handle allowing a sandbox to proxy via `CredentialRelay`."""

    lease_id: str
    lease_token: str = Field(..., repr=False)
    target_name: str
    tenant_id: str = "default"
    session_id: str = ""
    allowed_hosts: list[str] = Field(default_factory=list)
    expires_at: float
    max_calls: int = 50
    calls_used: int = 0
    revoked: bool = False


def _normalize_host(host: str) -> str:
    cleaned = host.strip().lower().rstrip(".")
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    return cleaned


def _host_matches_allowlist(hostname: str, allowed_hosts: list[str]) -> bool:
    if not allowed_hosts or not hostname:
        return False
    if hostname.startswith(".") or ".." in hostname or any(c.isspace() for c in hostname):
        return False

    for raw_pattern in allowed_hosts:
        pattern = _normalize_host(raw_pattern)
        if not pattern:
            continue
        if pattern.startswith("*.") and len(pattern) > 2:
            suffix = pattern[1:]  # e.g. ".github.com"
            if hostname.endswith(suffix) and len(hostname) > len(suffix):
                return True
        elif hostname == pattern:
            return True
    return False


def _is_restricted_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _is_restricted_ip(ip.ipv4_mapped)
    return False


class CredentialRelay:
    """
    Zero-trust host-side credential relay gateway (inspired by MOVO plugin-credential-relay.mjs).

    Untrusted MCP plugins and sandboxes receive only a short-lived `(lease_id, lease_token)`
    pair and invoke `proxy_request()`. Upstream secrets remain strictly on the host side,
    are injected only for allowlisted HTTPS destinations with SSRF protection, and are
    scrubbed if echoed back in upstream responses.
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._owns_client: bool = http_client is None
        self._client: httpx.AsyncClient = (
            http_client
            if http_client is not None
            else httpx.AsyncClient(follow_redirects=False)
        )
        self._targets: dict[str, CredentialTarget] = {}
        self._leases: dict[str, CredentialLease] = {}

    def register_secret(self, target: CredentialTarget) -> None:
        """Register or update a host-side credential target."""
        if not target.name or not target.name.strip():
            raise ValueError("CredentialTarget.name must be a non-empty string.")
        if not target.inject_header or not target.inject_header.strip():
            raise ValueError("CredentialTarget.inject_header must be a non-empty string.")
        self._targets[target.name] = target.model_copy(deep=True)

    def list_targets(self) -> list[dict[str, Any]]:
        """List registered credential targets without exposing `secret_value`."""
        return [target.safe_summary() for target in self._targets.values()]

    def issue_lease(
        self,
        target_name: str,
        *,
        ttl_seconds: float = 300.0,
        max_calls: int = 50,
        session_id: str = "",
        tenant_id: str = "default",
    ) -> CredentialLease:
        """Issue a scoped, time- and call-bounded lease for an existing credential target."""
        target = self._targets.get(target_name)
        if target is None:
            raise CredentialRelayError(f"Unknown credential target: {target_name!r}")
        if target.tenant_id is not None and target.tenant_id != tenant_id:
            raise CredentialRelayError(
                f"Tenant {tenant_id!r} is not authorized to access target {target_name!r}"
            )

        lease = CredentialLease(
            lease_id=f"lease_{uuid.uuid4().hex[:12]}",
            lease_token=secrets.token_urlsafe(24),
            target_name=target.name,
            tenant_id=tenant_id,
            session_id=session_id,
            allowed_hosts=list(target.allowed_hosts),
            expires_at=time.time() + float(ttl_seconds),
            max_calls=int(max_calls),
            calls_used=0,
            revoked=False,
        )
        self._leases[lease.lease_id] = lease
        return lease

    def revoke_lease(self, lease_id: str) -> bool:
        """Revoke an active lease by ID; returns True if an active lease was revoked."""
        lease = self._leases.get(lease_id)
        if lease is None or lease.revoked:
            return False
        lease.revoked = True
        return True

    def reap_expired(self, now: float | None = None) -> int:
        """Remove expired or revoked leases from memory and return the count removed."""
        current_time = time.time() if now is None else float(now)
        to_remove = [
            lease_id
            for lease_id, lease in self._leases.items()
            if lease.revoked or current_time >= lease.expires_at
        ]
        for lease_id in to_remove:
            self._leases.pop(lease_id, None)
        return len(to_remove)

    def _validate_destination(self, url: str, target: CredentialTarget) -> None:
        """Enforce scheme, host allowlist, and SSRF IP restrictions on an outbound URL."""
        if not isinstance(url, str) or not url.strip():
            raise CredentialRelayError("Destination URL must be a non-empty string")

        parsed = urllib.parse.urlparse(url.strip())
        scheme = (parsed.scheme or "").lower()
        allowed_schemes = {"https", "http"} if target.allow_http else {"https"}
        if scheme not in allowed_schemes:
            raise CredentialRelayError(
                f"Disallowed URL scheme {scheme!r}; expected one of {sorted(allowed_schemes)}"
            )

        if parsed.username is not None or parsed.password is not None:
            raise CredentialRelayError("Embedded URL credentials are not allowed")

        try:
            _ = parsed.port
        except ValueError as exc:
            raise CredentialRelayError("Invalid destination URL port") from exc

        raw_hostname = parsed.hostname
        if not raw_hostname:
            raise CredentialRelayError("Destination URL is missing a hostname")

        hostname = _normalize_host(raw_hostname)
        if not hostname:
            raise CredentialRelayError("Destination URL is missing a hostname")

        if not _host_matches_allowlist(hostname, target.allowed_hosts):
            raise CredentialRelayError(f"Host not allowlisted: {hostname!r}")

        if not target.allow_private_ips:
            if (
                hostname in _BLOCKED_SSRF_HOSTS
                or hostname.endswith(".localhost")
                or hostname.endswith(".metadata.google.internal")
            ):
                raise CredentialRelayError(
                    f"SSRF blocked: private or restricted IP ({hostname})"
                )

            ip_candidate = hostname.split("%", 1)[0]
            try:
                parsed_ip = ipaddress.ip_address(ip_candidate)
            except ValueError:
                parsed_ip = None

            if parsed_ip is not None and _is_restricted_ip(parsed_ip):
                raise CredentialRelayError(
                    f"SSRF blocked: private or restricted IP ({hostname})"
                )

    async def proxy_request(
        self,
        *,
        lease_id: str,
        lease_token: str,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json_body: Any | None = None,
        content: str | bytes | None = None,
        timeout_seconds: float = 15.0,
    ) -> dict[str, Any]:
        """Validate lease + destination, inject host secret, proxy request, and scrub response."""
        lease = self._leases.get(lease_id)
        if lease is None:
            raise CredentialRelayError(f"Unknown credential lease: {lease_id!r}")
        if lease.revoked:
            raise CredentialRelayError(f"Credential lease has been revoked: {lease_id!r}")
        if not isinstance(lease_token, str) or not hmac.compare_digest(
            lease.lease_token, lease_token
        ):
            raise CredentialRelayError("Invalid credential lease token")
        if time.time() >= lease.expires_at:
            raise CredentialRelayError(f"Credential lease has expired: {lease_id!r}")
        if lease.calls_used >= lease.max_calls:
            raise CredentialRelayError(
                f"Credential lease call budget exhausted: {lease_id!r}"
            )

        target = self._targets.get(lease.target_name)
        if target is None:
            raise CredentialRelayError(
                f"Credential target is no longer registered: {lease.target_name!r}"
            )

        normalized_method = (method or "").strip().upper()
        if not normalized_method:
            raise CredentialRelayError("HTTP method must be a non-empty string")

        self._validate_destination(url, target)

        lease.calls_used += 1

        inject_header_lower = target.inject_header.lower()
        out_headers: dict[str, str] = {}
        if headers:
            for key, value in headers.items():
                key_lower = str(key).lower()
                if (
                    key_lower in _STRIPPED_REQUEST_HEADERS
                    or key_lower == inject_header_lower
                ):
                    continue
                out_headers[str(key)] = str(value)
        out_headers[target.inject_header] = f"{target.inject_prefix}{target.secret_value}"

        resp = await self._client.request(
            normalized_method,
            url,
            headers=out_headers,
            json=json_body,
            content=content,
            timeout=timeout_seconds,
            follow_redirects=False,
        )

        scrubbed_text = resp.text
        if target.secret_value:
            scrubbed_text = scrubbed_text.replace(
                target.secret_value, _REDACTED_RELAY_SECRET
            )

        safe_headers: dict[str, str] = {}
        for key, value in resp.headers.items():
            if key.lower() in _STRIPPED_RESPONSE_HEADERS:
                continue
            if target.secret_value and target.secret_value in value:
                value = value.replace(target.secret_value, _REDACTED_RELAY_SECRET)
            safe_headers[key] = value

        return {
            "status_code": resp.status_code,
            "headers": safe_headers,
            "text": scrubbed_text,
            "lease_calls_remaining": lease.max_calls - lease.calls_used,
        }

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned by this `CredentialRelay`."""
        if self._owns_client:
            await self._client.aclose()


__all__ = [
    "CredentialRelayError",
    "CredentialTarget",
    "CredentialLease",
    "CredentialRelay",
]
