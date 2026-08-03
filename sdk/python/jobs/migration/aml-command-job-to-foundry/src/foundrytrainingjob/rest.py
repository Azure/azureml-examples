from __future__ import annotations

import contextvars
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Final, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from azure.core.credentials import TokenCredential

from .auth import DEFAULT_AZURE_AI_FOUNDRY_RESOURCE, get_foundry_access_token
from .e2e.sanitization import sanitize_for_report

_LOG = logging.getLogger(__name__)

# Header used to route Foundry requests to the training-jobs surface. Enabled
# per-run via ``--foundry-jobs`` on the E2E CLI (see ``foundry_header_scope``).
FOUNDRY_JOB_ROUTE_HEADER: Final[str] = "x-ms-foundry-job-route"
FOUNDRY_TRAINING_JOBS_ROUTE: Final[str] = "foundryTrainingJob"

# Context variable carrying headers that should be injected into every Foundry
# request issued within an active :class:`foundry_header_scope`. Default is
# ``None`` (no injected headers). Using a ContextVar keeps concurrent scenario
# runs isolated and mirrors the ``current_retry_scope`` pattern already used
# for canary retries.
_CURRENT_EXTRA_HEADERS: contextvars.ContextVar[
    "Mapping[str, str] | None"
] = contextvars.ContextVar("foundrytrainingjob_extra_headers", default=None)


class foundry_header_scope:
    """Inject default headers into every Foundry request within the block.

    Headers provided here are merged into the base request headers before any
    per-call ``extra_headers`` (so a per-call header still wins on conflict).
    Usage::

        with foundry_header_scope({FOUNDRY_JOB_ROUTE_HEADER: FOUNDRY_TRAINING_JOBS_ROUTE}):
            scenario.run(...)
    """

    def __init__(self, headers: Mapping[str, str] | None) -> None:
        self._headers = dict(headers) if headers else None
        self._token: contextvars.Token["Mapping[str, str] | None"] | None = None

    def __enter__(self) -> "foundry_header_scope":
        self._token = _CURRENT_EXTRA_HEADERS.set(self._headers)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _CURRENT_EXTRA_HEADERS.reset(self._token)
            self._token = None


def current_extra_headers() -> "Mapping[str, str] | None":
    """Return the headers injected by the active ``foundry_header_scope``, if any."""
    return _CURRENT_EXTRA_HEADERS.get()


DEFAULT_ACCEPT_LANGUAGE: Final[str] = "en-US,en;q=0.9,en-IN;q=0.8"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 120
DEFAULT_MAX_RETRIES: Final[int] = 10
DEFAULT_RETRY_BACKOFF_SECONDS: Final[float] = 1.0
DEFAULT_MAX_RETRY_BACKOFF_SECONDS: Final[float] = 8.0
_RETRYABLE_METHODS: Final[frozenset[str]] = frozenset({"GET", "DELETE", "PUT"})
_TRANSIENT_STATUS_CODES: Final[frozenset[int]] = frozenset(
    {408, 429, 500, 502, 503, 504}
)

# Network-level retry (internet outage, DNS failure, connection refused).
# Separate from HTTP transient retry because network errors prove the request
# never reached the server — always safe to retry regardless of HTTP method
# and regardless of ``disable_retry`` (which exists to prevent duplicate
# server-side side-effects, and network failures cannot produce any).
NETWORK_RETRY_MAX_SECONDS: Final[float] = 300.0  # 5 minutes
NETWORK_RETRY_INITIAL_BACKOFF: Final[float] = 2.0
NETWORK_RETRY_MAX_BACKOFF: Final[float] = 30.0


@dataclass(frozen=True)
class FoundryRestResponse:
    status_code: int
    headers: dict[str, str]
    text: str

    def json(self) -> Any | None:
        if not self.text:
            return None

        return json.loads(self.text)

    def header(self, name: str) -> str | None:
        target = name.lower()
        for key, value in self.headers.items():
            if key.lower() == target:
                return value

        return None

    @property
    def apim_request_id(self) -> str | None:
        return self.header("apim-request-id")


def _parse_response_body(response_text: str) -> Any:
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except ValueError:
        return response_text


_RESPONSE_BODY_PREVIEW_MAX_LEN = 800


def _format_response_body_preview(body: Any) -> str:
    """Format a response body for inclusion in an error message.

    Returns a sanitized, size-capped string suitable for a log line. Structured
    bodies are compacted via ``json.dumps`` so error details like
    ``{"error":{"code":"InvalidRequest","message":"..."}}`` are visible on a
    single-line exception message without overwhelming the console.
    """
    if body is None or body == "" or body == [] or body == {}:
        return ""
    if isinstance(body, (dict, list)):
        try:
            rendered = json.dumps(body, separators=(",", ":"), sort_keys=False)
        except (TypeError, ValueError):
            rendered = str(body)
    else:
        rendered = str(body)
    rendered = sanitize_for_report(rendered)
    if len(rendered) > _RESPONSE_BODY_PREVIEW_MAX_LEN:
        rendered = rendered[:_RESPONSE_BODY_PREVIEW_MAX_LEN] + "...<truncated>"
    return rendered


class FoundryRequestError(RuntimeError):
    def as_evidence(self) -> dict[str, Any]:
        return {}


def _sanitized_url(url: str) -> str:
    return sanitize_for_report(url)


class FoundryHttpError(FoundryRequestError):
    def __init__(self, *, method: str, url: str, response: FoundryRestResponse) -> None:
        self.method = method
        self.url = url
        self.response = response
        self.response_body = _parse_response_body(response.text)

        message = (
            f"Foundry REST {method} failed with status {response.status_code} "
            f"for {sanitize_for_report(url)}."
        )
        if response.apim_request_id:
            message = f"{message} apim-request-id: {response.apim_request_id}."
        body_preview = _format_response_body_preview(self.response_body)
        if body_preview:
            message = f"{message} body: {body_preview}"
        super().__init__(message)

    def as_evidence(self) -> dict[str, Any]:
        evidence = {
            "method": self.method,
            "url": _sanitized_url(self.url),
            "statusCode": self.response.status_code,
            "apimRequestId": self.response.apim_request_id,
        }
        if self.response_body not in (None, "", {}):
            evidence["responseBody"] = self.response_body
        return evidence


class FoundryNetworkError(FoundryRequestError):
    def __init__(self, *, method: str, url: str, reason: Any) -> None:
        self.method = method
        self.url = url
        self.reason = reason
        super().__init__(
            f"Foundry REST {method} failed for {sanitize_for_report(url)}: "
            f"{sanitize_for_report(str(reason))}"
        )

    def as_evidence(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "url": _sanitized_url(self.url),
            "reason": str(self.reason),
        }


def create_request_id() -> str:
    return f"{uuid4().hex}.{uuid4().hex[:16]}"


def _should_retry_request(
    method: str,
    *,
    status_code: int | None = None,
    network_error: bool = False,
    retry_transient: bool = False,
    disable_retry: bool = False,
) -> bool:
    if disable_retry:
        return False
    if network_error:
        # Network errors now have their own time-budgeted retry path inside
        # ``_request_foundry`` and are NOT decided by this predicate.
        return False
    normalized_method = method.upper()
    if normalized_method not in _RETRYABLE_METHODS and not retry_transient:
        return False
    return status_code in _TRANSIENT_STATUS_CODES


def _retry_delay_seconds(attempt: int) -> float:
    backoff = min(
        DEFAULT_MAX_RETRY_BACKOFF_SECONDS,
        DEFAULT_RETRY_BACKOFF_SECONDS * (2**attempt),
    )
    return backoff + random.uniform(0, 0.25)


def _maybe_record_canary_retry(
    *,
    method: str,
    url: str,
    canary_attempt: int,
    error_response: "FoundryRestResponse",
) -> bool:
    """Check the active canary-retry scope; record and return True if we should retry.

    Canary retries intentionally do NOT consult ``disable_retry``:
      * ``disable_retry=True`` exists for PUT /jobs create to prevent duplicate
        compute allocation if a generic 504 retry sends a second create request.
      * A signature-matched 400 proves the server rejected the request BEFORE
        any allocation / DB write happened (broken pod returned a client-error
        shaped response). Retrying is always safe.
      * Keeping canary retry orthogonal means ``submit_job`` (which uses
        ``disable_retry=True`` on purpose) still gets the benefit of the
        retry layer.

    ``canary_attempt`` is a counter separate from the generic-transient
    counter so a burst of 5xx retries cannot starve the canary retry budget.

    Deferred import avoids a circular dependency between rest.py and
    e2e.retry_policy (which may itself need rest primitives in future).
    """
    try:
        from .e2e.retry_policy import current_retry_scope, match_signature, RetryEvent
    except Exception:  # pragma: no cover - defensive
        return False
    scope = current_retry_scope()
    if scope is None:
        return False
    if canary_attempt >= scope.max_retries:
        return False
    signature = match_signature(
        scope.signatures,
        status_code=error_response.status_code,
        body_text=error_response.text,
    )
    if signature is None:
        return False
    sanitized_url = _sanitized_url(url)
    scope.account.record(
        RetryEvent(
            signature_name=signature.name,
            status_code=error_response.status_code,
            method=method,
            url=sanitized_url,
            attempt=canary_attempt,
            apim_request_id=error_response.apim_request_id,
        )
    )
    # Warn-level log so operators correlating a mid-run hang with pod
    # instability can see retries happening without waiting for the summary.
    _LOG.warning(
        "canary-retry: signature=%s method=%s status=%d attempt=%d url=%s apim_request_id=%s",
        signature.name,
        method,
        error_response.status_code,
        canary_attempt,
        sanitized_url,
        error_response.apim_request_id,
    )
    return True


def _request_foundry(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    access_token: str | None = None,
    resource_or_scope: str = DEFAULT_AZURE_AI_FOUNDRY_RESOURCE,
    credential: TokenCredential | None = None,
    request_id: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    raise_on_http_error: bool = True,
    retry_transient: bool = False,
    disable_retry: bool = False,
) -> FoundryRestResponse:
    resolved_access_token = (
        access_token
        or get_foundry_access_token(
            resource_or_scope,
            credential=credential,
        ).token
    )

    headers = {
        "Authorization": f"Bearer {resolved_access_token}",
        "accept": "*/*",
        "accept-language": DEFAULT_ACCEPT_LANGUAGE,
        "request-id": request_id or create_request_id(),
    }

    body = None
    if payload is not None:
        headers["content-type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")

    # Headers injected by an active ``foundry_header_scope`` apply to every
    # request in the block. Per-call ``extra_headers`` are merged afterwards so
    # an explicit per-call header always wins on conflict.
    scope_headers = _CURRENT_EXTRA_HEADERS.get()
    if scope_headers:
        headers.update(scope_headers)

    if extra_headers:
        headers.update(extra_headers)

    request = Request(url=url, data=body, headers=headers, method=method)
    # Three counters so each retry class has its own budget and cannot
    # starve the others:
    #   * generic_attempt — HTTP 408/429/5xx on retryable methods (count-based).
    #   * canary_attempt  — narrow canary-signature retries (count-based).
    #   * network_attempt — URLError / internet outage (time-budgeted, see
    #                       NETWORK_RETRY_MAX_SECONDS); retries regardless of
    #                       method or disable_retry because a URLError proves
    #                       the request never reached the server.
    generic_attempt = 0
    canary_attempt = 0
    network_attempt = 0
    start_time = time.monotonic()
    while True:
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                response_text = response.read().decode("utf-8", errors="replace")
                return FoundryRestResponse(
                    status_code=response.status,
                    headers=dict(response.headers.items()),
                    text=response_text,
                )
        except HTTPError as error:
            response_text = error.read().decode("utf-8", errors="replace")
            error_response = FoundryRestResponse(
                status_code=error.code,
                headers=dict(error.headers.items()),
                text=response_text,
            )
            # Generic transient retry path (408/429/5xx + retryable methods).
            if generic_attempt < DEFAULT_MAX_RETRIES and _should_retry_request(
                method,
                status_code=error.code,
                retry_transient=retry_transient,
                disable_retry=disable_retry,
            ):
                time.sleep(_retry_delay_seconds(generic_attempt))
                generic_attempt += 1
                continue
            # Narrow canary-signature retry path. Active only inside a
            # ``canary_retry_scope`` and only for signatures on the allow-list.
            # Decoupled from ``disable_retry`` on purpose: see
            # ``_maybe_record_canary_retry`` docstring for the rationale.
            if _maybe_record_canary_retry(
                method=method,
                url=url,
                canary_attempt=canary_attempt,
                error_response=error_response,
            ):
                time.sleep(_retry_delay_seconds(canary_attempt))
                canary_attempt += 1
                continue
            if not raise_on_http_error:
                return error_response
            raise FoundryHttpError(
                method=method,
                url=url,
                response=error_response,
            ) from error
        except (URLError, TimeoutError) as error:
            # Network errors (DNS failure, connection refused, internet
            # outage, read timeout) prove the request never reached the
            # server, so retrying is always safe — independent of HTTP
            # method and of ``disable_retry``. Use a time-based budget
            # (NETWORK_RETRY_MAX_SECONDS) so a long outage gets a fair
            # chance to recover without unbounded retries.
            #
            # Note: ``urlopen()`` raises ``socket.timeout`` (== ``TimeoutError``,
            # a sibling of ``URLError`` not a subclass) on read timeouts. Catch
            # both so a sustained-slow MFE PUT is retried with backoff instead
            # of fatally failing on the first 120s timeout.
            elapsed = time.monotonic() - start_time
            if elapsed < NETWORK_RETRY_MAX_SECONDS:
                backoff = min(
                    NETWORK_RETRY_MAX_BACKOFF,
                    NETWORK_RETRY_INITIAL_BACKOFF * (2**network_attempt),
                ) + random.uniform(0, 1.0)
                error_reason = (
                    type(error.reason).__name__
                    if isinstance(error, URLError)
                    and hasattr(error, "reason")
                    and error.reason is not None
                    else type(error).__name__
                )
                _LOG.warning(
                    "network-retry: method=%s attempt=%d elapsed=%.0fs "
                    "backoff=%.1fs error=%s url=%s",
                    method,
                    network_attempt,
                    elapsed,
                    backoff,
                    error_reason,
                    _sanitized_url(url),
                )
                time.sleep(backoff)
                network_attempt += 1
                continue
            raise FoundryNetworkError(
                method=method,
                url=url,
                reason=getattr(error, "reason", str(error)),
            ) from error


def get_foundry_json(
    url: str,
    *,
    access_token: str | None = None,
    resource_or_scope: str = DEFAULT_AZURE_AI_FOUNDRY_RESOURCE,
    credential: TokenCredential | None = None,
    request_id: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    raise_on_http_error: bool = True,
    disable_retry: bool = False,
) -> FoundryRestResponse:
    return _request_foundry(
        "GET",
        url,
        access_token=access_token,
        resource_or_scope=resource_or_scope,
        credential=credential,
        request_id=request_id,
        extra_headers=extra_headers,
        timeout_seconds=timeout_seconds,
        raise_on_http_error=raise_on_http_error,
        disable_retry=disable_retry,
    )


def put_foundry_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    access_token: str | None = None,
    resource_or_scope: str = DEFAULT_AZURE_AI_FOUNDRY_RESOURCE,
    credential: TokenCredential | None = None,
    request_id: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    raise_on_http_error: bool = True,
    disable_retry: bool = False,
) -> FoundryRestResponse:
    return _request_foundry(
        "PUT",
        url,
        payload=payload,
        access_token=access_token,
        resource_or_scope=resource_or_scope,
        credential=credential,
        request_id=request_id,
        extra_headers=extra_headers,
        timeout_seconds=timeout_seconds,
        raise_on_http_error=raise_on_http_error,
        disable_retry=disable_retry,
    )


def post_foundry_json(
    url: str,
    payload: Mapping[str, Any] | None = None,
    *,
    access_token: str | None = None,
    resource_or_scope: str = DEFAULT_AZURE_AI_FOUNDRY_RESOURCE,
    credential: TokenCredential | None = None,
    request_id: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    raise_on_http_error: bool = True,
    retry_transient: bool = False,
    disable_retry: bool = False,
) -> FoundryRestResponse:
    return _request_foundry(
        "POST",
        url,
        payload=payload,
        access_token=access_token,
        resource_or_scope=resource_or_scope,
        credential=credential,
        request_id=request_id,
        extra_headers=extra_headers,
        timeout_seconds=timeout_seconds,
        raise_on_http_error=raise_on_http_error,
        retry_transient=retry_transient,
        disable_retry=disable_retry,
    )


def delete_foundry(
    url: str,
    *,
    access_token: str | None = None,
    resource_or_scope: str = DEFAULT_AZURE_AI_FOUNDRY_RESOURCE,
    credential: TokenCredential | None = None,
    request_id: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    raise_on_http_error: bool = True,
    disable_retry: bool = False,
) -> FoundryRestResponse:
    return _request_foundry(
        "DELETE",
        url,
        access_token=access_token,
        resource_or_scope=resource_or_scope,
        credential=credential,
        request_id=request_id,
        extra_headers=extra_headers,
        timeout_seconds=timeout_seconds,
        raise_on_http_error=raise_on_http_error,
        disable_retry=disable_retry,
    )
