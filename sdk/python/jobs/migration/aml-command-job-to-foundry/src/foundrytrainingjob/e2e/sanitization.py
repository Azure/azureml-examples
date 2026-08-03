from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_REDACTED = "<redacted>"
_SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "code",
    "refresh_token",
    "sig",
    "signature",
    "token",
}
_SENSITIVE_KEY_SUBSTRINGS = (
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "clientsecret",
    "connection_string",
    "connectionstring",
    "password",
    "refresh_token",
    "sas",
    "secret",
    "token",
)
_SENSITIVE_QUERY_PATTERN = re.compile(
    r"(?P<prefix>[?&])(?P<key>access_token|api_key|code|refresh_token|sas|sig|signature|token)=(?P<value>[^&#\s]+)",
    re.IGNORECASE,
)
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_JWT_TOKEN_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9._-]+\.[A-Za-z0-9._-]+\b"
)
_FREEFORM_EQUALS_PATTERN = re.compile(
    r"(?P<prefix>(?:^|[\s,;({\[]))(?P<key>[A-Za-z0-9_.-]+)\s*=\s*(?P<value>[^\s,;&)\]}]+)"
)
_FREEFORM_COLON_PATTERN = re.compile(
    r'(?P<prefix>(?:^|[\s,{(\[]))(?P<key>"?[A-Za-z0-9_.-]+"?)\s*:\s*(?P<quote>["\']?)(?P<value>[^"\',\s}\]]+)(?P=quote)'
)


def _normalize_key(key: str) -> str:
    with_word_boundaries = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return with_word_boundaries.replace("-", "_").lower()


def _is_sensitive_key(key: str) -> bool:
    normalized = _normalize_key(key)
    return any(fragment in normalized for fragment in _SENSITIVE_KEY_SUBSTRINGS)


def _sanitize_string(value: str) -> str:
    sanitized = _BEARER_TOKEN_PATTERN.sub(f"Bearer {_REDACTED}", value)
    sanitized = _JWT_TOKEN_PATTERN.sub(_REDACTED, sanitized)
    sanitized = _SENSITIVE_QUERY_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{match.group('key')}={_REDACTED}",
        sanitized,
    )
    sanitized = _FREEFORM_EQUALS_PATTERN.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('key')}={_REDACTED}"
            if _is_sensitive_key(match.group("key"))
            else match.group(0)
        ),
        sanitized,
    )
    sanitized = _FREEFORM_COLON_PATTERN.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('key')}: "
            f"{match.group('quote')}{_REDACTED}{match.group('quote')}"
            if _is_sensitive_key(match.group("key").strip("\"'"))
            else match.group(0)
        ),
        sanitized,
    )

    try:
        parsed = urlsplit(sanitized)
    except ValueError:
        return sanitized

    if not parsed.scheme or not parsed.netloc or not parsed.query:
        return sanitized

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if not any(key.lower() in _SENSITIVE_QUERY_KEYS for key, _ in query_pairs):
        return sanitized

    sanitized_query = urlencode(
        [
            (key, _REDACTED if key.lower() in _SENSITIVE_QUERY_KEYS else item_value)
            for key, item_value in query_pairs
        ]
    )
    return urlunsplit(parsed._replace(query=sanitized_query))


def sanitize_for_report(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized_mapping: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            sanitized_key = _sanitize_string(key)
            if _is_sensitive_key(key):
                sanitized_mapping[sanitized_key] = _REDACTED
            else:
                sanitized_mapping[sanitized_key] = sanitize_for_report(raw_value)
        return sanitized_mapping

    if isinstance(value, list):
        return [sanitize_for_report(item) for item in value]

    if isinstance(value, tuple):
        return tuple(sanitize_for_report(item) for item in value)

    if isinstance(value, set):
        return {sanitize_for_report(item) for item in value}

    if isinstance(value, str):
        return _sanitize_string(value)

    return value
