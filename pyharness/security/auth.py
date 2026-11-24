from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Iterator, Mapping


class AuthenticationError(Exception):
    """Raised when incoming credentials are missing (in enforce mode), invalid, or expired."""


class AuthorizationError(Exception):
    """Raised when an authenticated principal lacks permissions for an operation."""


@dataclass(frozen=True)
class Principal:
    """Immutable authenticated caller identity propagated via ContextVar."""

    user_id: str
    tenant_id: str
    department: str = "default"
    roles: frozenset[str] = frozenset({"member"})
    daily_token_quota: int = 500_000
    daily_cost_limit_usd: float = 20.0
    auth_method: str = "anonymous"

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles

    def has_any_role(self, required: set[str] | frozenset[str] | list[str]) -> bool:
        return self.is_admin or bool(self.roles.intersection(required))

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "department": self.department,
            "roles": sorted(self.roles),
            "daily_token_quota": self.daily_token_quota,
            "daily_cost_limit_usd": self.daily_cost_limit_usd,
            "auth_method": self.auth_method,
            "is_admin": self.is_admin,
        }


current_principal: ContextVar[Principal | None] = ContextVar(
    "pyharness_principal", default=None
)


@contextmanager
def use_principal(principal: Principal | None) -> Iterator[Principal | None]:
    """Bind `principal` to the current async/thread execution context and reset on exit."""
    token = current_principal.set(principal)
    try:
        yield principal
    finally:
        current_principal.reset(token)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    if not segment or not isinstance(segment, str):
        raise AuthenticationError("Malformed JWT segment.")
    padding = "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode(segment.encode("ascii") + padding.encode("ascii"))
    except (ValueError, UnicodeEncodeError, binascii.Error) as exc:
        raise AuthenticationError("Invalid base64url encoding in JWT.") from exc


def create_hs256_jwt(
    payload: dict[str, Any],
    secret: str,
    expires_in_seconds: int = 3600,
) -> str:
    """Create a compact HS256 JWT using Python stdlib only."""
    if not secret:
        raise ValueError("JWT secret must not be empty.")

    now = int(time.time())
    claims = dict(payload)
    claims.setdefault("iat", now)
    claims.setdefault("exp", now + int(expires_in_seconds))

    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    payload_b64 = _b64url_encode(
        json.dumps(
            claims,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    )
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(signature)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def verify_hs256_jwt(
    token: str,
    secret: str,
    issuer: str | None = None,
    audience: str | None = None,
) -> dict[str, Any]:
    """Verify an HS256 JWT signature, algorithm, expiration, and optional iss/aud claims."""
    if not token or not isinstance(token, str):
        raise AuthenticationError("Missing or invalid JWT token.")
    if not secret:
        raise AuthenticationError("JWT verification secret is not configured.")

    parts = token.strip().split(".")
    if len(parts) != 3:
        raise AuthenticationError("Malformed JWT: expected 3 dot-separated segments.")
    header_b64, payload_b64, sig_b64 = parts

    try:
        header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthenticationError("Malformed JWT header JSON.") from exc

    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise AuthenticationError("Unsupported or invalid JWT algorithm; expected HS256.")

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(
        secret.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    actual_sig = _b64url_decode(sig_b64)
    if not hmac.compare_digest(actual_sig, expected_sig):
        raise AuthenticationError("Invalid JWT signature.")

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthenticationError("Malformed JWT payload JSON.") from exc

    if not isinstance(payload, dict):
        raise AuthenticationError("JWT payload must be a JSON object.")

    exp = payload.get("exp")
    if isinstance(exp, bool) or not isinstance(exp, (int, float)):
        raise AuthenticationError("JWT missing valid numeric 'exp' claim.")
    now = time.time()
    if now >= float(exp):
        raise AuthenticationError("JWT token has expired.")

    nbf = payload.get("nbf")
    if nbf is not None:
        if isinstance(nbf, bool) or not isinstance(nbf, (int, float)):
            raise AuthenticationError("JWT has invalid 'nbf' claim.")
        if now < float(nbf):
            raise AuthenticationError("JWT token is not yet valid.")

    if issuer is not None and payload.get("iss") != issuer:
        raise AuthenticationError("JWT issuer mismatch.")

    if audience is not None:
        aud = payload.get("aud")
        if isinstance(aud, str):
            aud_ok = aud == audience
        elif isinstance(aud, list):
            aud_ok = audience in aud
        else:
            aud_ok = False
        if not aud_ok:
            raise AuthenticationError("JWT audience mismatch.")

    return payload


def _parse_roles(raw: Any) -> frozenset[str]:
    if raw is None:
        return frozenset({"member"})
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
        return frozenset(items) if items else frozenset({"member"})
    if isinstance(raw, (list, set, frozenset, tuple)):
        items = [str(item).strip() for item in raw if str(item).strip()]
        return frozenset(items) if items else frozenset({"member"})
    return frozenset({"member"})


def _principal_from_claims(data: Mapping[str, Any], auth_method: str) -> Principal:
    user_id = str(data.get("user_id") or data.get("sub") or "").strip()
    if not user_id:
        raise AuthenticationError("Authenticated identity is missing 'sub'/'user_id'.")
    tenant_id = str(data.get("tenant_id") or "default").strip() or "default"
    department = str(data.get("department") or "default").strip() or "default"
    roles = _parse_roles(data.get("roles"))
    try:
        daily_token_quota = int(data.get("daily_token_quota", 500_000))
        daily_cost_limit_usd = float(data.get("daily_cost_limit_usd", 20.0))
    except (TypeError, ValueError) as exc:
        raise AuthenticationError("Invalid quota fields in principal claims.") from exc
    return Principal(
        user_id=user_id,
        tenant_id=tenant_id,
        department=department,
        roles=roles,
        daily_token_quota=daily_token_quota,
        daily_cost_limit_usd=daily_cost_limit_usd,
        auth_method=auth_method,
    )


class AuthManager:
    """
    Zero-external-dependency enterprise authentication gateway.

    Resolution priority:
    1. `Authorization: Bearer <HS256 JWT>`
    2. `X-API-Key: <key>`
    3. Trusted reverse-proxy headers (`X-Forwarded-User` + `X-Tenant-Id`) if explicitly enabled
    4. `dev_anonymous` fallback (or `AuthenticationError` when `mode == "enforce"`)
    """

    def __init__(
        self,
        mode: str = "dev_anonymous",
        jwt_secret: str = "pyharness-dev-secret",
        api_keys: dict[str, Principal] | None = None,
        trusted_proxy_headers: bool = False,
        jwt_issuer: str | None = None,
        jwt_audience: str | None = None,
        default_dev_principal: Principal | None = None,
    ) -> None:
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"dev_anonymous", "enforce"}:
            raise ValueError(
                f"Unsupported auth mode {mode!r}; expected 'dev_anonymous' or 'enforce'."
            )
        self.mode = normalized_mode
        self.jwt_secret = jwt_secret
        self.api_keys: dict[str, Principal] = dict(api_keys or {})
        self.trusted_proxy_headers = bool(trusted_proxy_headers)
        self.jwt_issuer = jwt_issuer or None
        self.jwt_audience = jwt_audience or None
        self.default_dev_principal = default_dev_principal or Principal(
            user_id="dev_operator",
            tenant_id="default",
            department="engineering",
            roles=frozenset({"member", "approver", "admin"}),
            auth_method="dev_anonymous",
        )

    @classmethod
    def from_env(cls) -> AuthManager:
        mode = os.getenv("HARNESS_AUTH_MODE", "dev_anonymous").strip() or "dev_anonymous"
        jwt_secret = os.getenv("HARNESS_JWT_SECRET", "pyharness-dev-secret")
        jwt_issuer = os.getenv("HARNESS_JWT_ISSUER", "").strip() or None
        jwt_audience = os.getenv("HARNESS_JWT_AUDIENCE", "").strip() or None
        trusted_proxy = os.getenv("HARNESS_TRUST_PROXY_HEADERS", "").strip().lower() in {
            "true",
            "1",
            "yes",
        }

        api_keys: dict[str, Principal] = {}
        raw_api_keys = os.getenv("HARNESS_API_KEYS_JSON", "").strip()
        if raw_api_keys:
            parsed = json.loads(raw_api_keys)
            if not isinstance(parsed, dict):
                raise ValueError("HARNESS_API_KEYS_JSON must be a JSON object mapping key -> principal.")
            for api_key, entry in parsed.items():
                if not isinstance(entry, Mapping):
                    raise ValueError(f"Invalid principal configuration for API key {api_key!r}.")
                api_keys[str(api_key)] = _principal_from_claims(entry, auth_method="api_key")

        return cls(
            mode=mode,
            jwt_secret=jwt_secret,
            api_keys=api_keys,
            trusted_proxy_headers=trusted_proxy,
            jwt_issuer=jwt_issuer,
            jwt_audience=jwt_audience,
        )

    def _lookup_api_key(self, candidate: str) -> Principal | None:
        candidate_bytes = candidate.encode("utf-8")
        matched: Principal | None = None
        for stored_key, principal in self.api_keys.items():
            if hmac.compare_digest(candidate_bytes, stored_key.encode("utf-8")):
                matched = principal
        return matched

    def authenticate(self, headers: Mapping[str, str]) -> Principal:
        lowered = {str(k).lower(): str(v) for k, v in headers.items()}

        # Priority 1: Authorization: Bearer <HS256 JWT>
        auth_header = lowered.get("authorization", "").strip()
        if auth_header:
            parts = auth_header.split(None, 1)
            if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
                raise AuthenticationError(
                    "Invalid Authorization header; expected 'Bearer <token>'."
                )
            claims = verify_hs256_jwt(
                parts[1].strip(),
                self.jwt_secret,
                issuer=self.jwt_issuer,
                audience=self.jwt_audience,
            )
            return _principal_from_claims(claims, auth_method="jwt")

        # Priority 2: X-API-Key
        if "x-api-key" in lowered:
            api_key = lowered["x-api-key"].strip()
            principal = self._lookup_api_key(api_key) if api_key else None
            if principal is None:
                raise AuthenticationError("Invalid API key.")
            if principal.auth_method == "anonymous":
                return replace(principal, auth_method="api_key")
            return principal

        # Priority 3: Trusted reverse-proxy headers (opt-in only)
        if self.trusted_proxy_headers:
            forwarded_user = lowered.get("x-forwarded-user", "").strip()
            forwarded_tenant = lowered.get("x-tenant-id", "").strip()
            if forwarded_user:
                roles = _parse_roles(lowered.get("x-user-roles"))
                department = lowered.get("x-department", "").strip() or "default"
                return Principal(
                    user_id=forwarded_user,
                    tenant_id=forwarded_tenant or "default",
                    department=department,
                    roles=roles,
                    auth_method="trusted_proxy",
                )

        # Fallback
        if self.mode == "enforce":
            raise AuthenticationError(
                "Authentication required: provide Authorization Bearer JWT or X-API-Key."
            )
        return self.default_dev_principal

    def issue_token(
        self,
        principal: Principal,
        expires_in_seconds: int = 3600,
    ) -> str:
        """Issue an HS256 JWT representing `principal`."""
        claims: dict[str, Any] = {
            "sub": principal.user_id,
            "user_id": principal.user_id,
            "tenant_id": principal.tenant_id,
            "department": principal.department,
            "roles": sorted(principal.roles),
            "daily_token_quota": principal.daily_token_quota,
            "daily_cost_limit_usd": principal.daily_cost_limit_usd,
        }
        if self.jwt_issuer is not None:
            claims["iss"] = self.jwt_issuer
        if self.jwt_audience is not None:
            claims["aud"] = self.jwt_audience
        return create_hs256_jwt(
            claims,
            self.jwt_secret,
            expires_in_seconds=expires_in_seconds,
        )
