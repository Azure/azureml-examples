from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Final

from azure.core.credentials import AccessToken, TokenCredential
from azure.core.exceptions import AzureError, ClientAuthenticationError
from azure.identity import (
    AzureCliCredential,
    AzureDeveloperCliCredential,
    AzurePowerShellCredential,
    CredentialUnavailableError,
    EnvironmentCredential,
    ManagedIdentityCredential,
    SharedTokenCacheCredential,
    VisualStudioCodeCredential,
    WorkloadIdentityCredential,
)

DEFAULT_AZURE_AI_FOUNDRY_RESOURCE: Final[str] = "https://ai.azure.com"
DEFAULT_AZURE_AI_FOUNDRY_SCOPE: Final[
    str
] = f"{DEFAULT_AZURE_AI_FOUNDRY_RESOURCE}/.default"

AUTH_MAX_ATTEMPTS_ENV_VAR: Final[str] = "FOUNDRY_TRAININGJOB_AUTH_MAX_ATTEMPTS"
AUTH_INITIAL_BACKOFF_ENV_VAR: Final[
    str
] = "FOUNDRY_TRAININGJOB_AUTH_INITIAL_BACKOFF_SECONDS"
AUTH_MAX_BACKOFF_ENV_VAR: Final[str] = "FOUNDRY_TRAININGJOB_AUTH_MAX_BACKOFF_SECONDS"
AUTH_RETRY_BUDGET_ENV_VAR: Final[str] = "FOUNDRY_TRAININGJOB_AUTH_RETRY_BUDGET_SECONDS"
AUTH_TOKEN_REFRESH_SKEW_ENV_VAR: Final[
    str
] = "FOUNDRY_TRAININGJOB_AUTH_TOKEN_REFRESH_SKEW_SECONDS"
AUTH_CLI_PROCESS_TIMEOUT_ENV_VAR: Final[
    str
] = "FOUNDRY_TRAININGJOB_AUTH_CLI_PROCESS_TIMEOUT_SECONDS"

# 19 attempts plus a 300-second retry budget keep transient CLI/profile/cache
# contention from collapsing unattended daily E2E runs, without retrying forever.
DEFAULT_AUTH_MAX_ATTEMPTS: Final[int] = 19
DEFAULT_AUTH_INITIAL_BACKOFF_SECONDS: Final[float] = 1.0
DEFAULT_AUTH_MAX_BACKOFF_SECONDS: Final[float] = 30.0
DEFAULT_AUTH_RETRY_BUDGET_SECONDS: Final[float] = 300.0
DEFAULT_AUTH_TOKEN_REFRESH_SKEW_SECONDS: Final[int] = 300
DEFAULT_AUTH_CLI_PROCESS_TIMEOUT_SECONDS: Final[int] = 30

_LOG = logging.getLogger(__name__)


def to_scope(resource_or_scope: str = DEFAULT_AZURE_AI_FOUNDRY_RESOURCE) -> str:
    """Normalize a resource URI into the scope format expected by Azure SDK credentials."""
    value = resource_or_scope.strip()
    if not value:
        raise ValueError("resource_or_scope must be a non-empty string.")

    if value.endswith("/.default"):
        return value

    return f"{value.rstrip('/')}/.default"


def _read_int_env(name: str, default: int, *, minimum: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return parsed


def _read_float_env(name: str, default: float, *, minimum: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return parsed


def _safe_exception_type(exc: BaseException | None) -> str:
    return type(exc).__name__ if exc is not None else "None"


def _safe_scope_summary(scopes: tuple[str, ...]) -> str:
    if not scopes:
        return "0 scopes"
    if len(scopes) == 1 and scopes[0] == DEFAULT_AZURE_AI_FOUNDRY_SCOPE:
        return DEFAULT_AZURE_AI_FOUNDRY_SCOPE
    return f"{len(scopes)} scopes"


def _workload_identity_configured() -> bool:
    return bool(
        os.getenv("AZURE_TENANT_ID")
        and os.getenv("AZURE_CLIENT_ID")
        and os.getenv("AZURE_FEDERATED_TOKEN_FILE")
    )


def _environment_credential_configured() -> bool:
    return bool(
        os.getenv("AZURE_TENANT_ID")
        and os.getenv("AZURE_CLIENT_ID")
        and (
            os.getenv("AZURE_CLIENT_SECRET")
            or os.getenv("AZURE_CLIENT_CERTIFICATE_PATH")
            or (os.getenv("AZURE_USERNAME") and os.getenv("AZURE_PASSWORD"))
        )
    )


@dataclass
class _TokenAcquisitionState:
    event: threading.Event
    failure_message: str | None = None


class FallbackTokenCredential(TokenCredential):
    """TokenCredential chain that keeps trying providers after sanitized auth failures."""

    def __init__(
        self, *credentials: TokenCredential | tuple[TokenCredential, bool]
    ) -> None:
        if not credentials:
            raise ValueError("at least one credential is required.")
        providers: list[tuple[TokenCredential, bool]] = []
        for credential in credentials:
            if isinstance(credential, tuple):
                providers.append(credential)
            else:
                providers.append((credential, False))
        self._providers = tuple(providers)
        self._credentials = tuple(credential for credential, _ in self._providers)

    def get_token(
        self,
        *scopes: str,
        claims: str | None = None,
        tenant_id: str | None = None,
        enable_cae: bool = False,
        **kwargs: Any,
    ) -> AccessToken:
        failures: list[str] = []
        for credential, fail_fast_on_auth_error in self._providers:
            provider_name = type(credential).__name__
            try:
                return credential.get_token(
                    *scopes,
                    claims=claims,
                    tenant_id=tenant_id,
                    enable_cae=enable_cae,
                    **kwargs,
                )
            except CredentialUnavailableError as exc:
                error_type = _safe_exception_type(exc)
                failures.append(f"{provider_name}:{error_type}")
                _LOG.debug(
                    "Foundry credential provider %s failed with %s; trying the next provider.",
                    provider_name,
                    error_type,
                )
            except (ClientAuthenticationError, AzureError, OSError) as exc:
                error_type = _safe_exception_type(exc)
                if fail_fast_on_auth_error:
                    message = (
                        "Explicit Foundry credential provider failed with "
                        f"{error_type} for {_safe_scope_summary(tuple(scopes))}; "
                        "not falling back to a different identity."
                    )
                    raise CredentialUnavailableError(message=message) from None
                failures.append(f"{provider_name}:{error_type}")
                _LOG.debug(
                    "Foundry credential provider %s failed with %s; trying the next provider.",
                    provider_name,
                    error_type,
                )

        failure_summary = ", ".join(failures) if failures else "none"
        message = (
            "No Foundry credential provider acquired a token for "
            f"{_safe_scope_summary(tuple(scopes))}. "
            f"Provider failure types: {failure_summary}."
        )
        raise CredentialUnavailableError(message=message) from None


class RetryingTokenCredential(TokenCredential):
    """TokenCredential wrapper with bounded retries and in-process token caching."""

    def __init__(
        self,
        credential: TokenCredential,
        *,
        max_attempts: int = DEFAULT_AUTH_MAX_ATTEMPTS,
        initial_backoff_seconds: float = DEFAULT_AUTH_INITIAL_BACKOFF_SECONDS,
        max_backoff_seconds: float = DEFAULT_AUTH_MAX_BACKOFF_SECONDS,
        retry_budget_seconds: float = DEFAULT_AUTH_RETRY_BUDGET_SECONDS,
        token_refresh_skew_seconds: int = DEFAULT_AUTH_TOKEN_REFRESH_SKEW_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        jitter: Callable[[float, float], float] = random.uniform,
        provider_description: str | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1.")
        if not math.isfinite(initial_backoff_seconds) or initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be finite and >= 0.")
        if not math.isfinite(max_backoff_seconds) or max_backoff_seconds < 0:
            raise ValueError("max_backoff_seconds must be finite and >= 0.")
        if not math.isfinite(retry_budget_seconds) or retry_budget_seconds < 0:
            raise ValueError("retry_budget_seconds must be finite and >= 0.")
        if token_refresh_skew_seconds < 0:
            raise ValueError("token_refresh_skew_seconds must be >= 0.")

        self._credential = credential
        self._max_attempts = max_attempts
        self._initial_backoff_seconds = initial_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._retry_budget_seconds = retry_budget_seconds
        self._token_refresh_skew_seconds = token_refresh_skew_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._jitter = jitter
        self._provider_description = provider_description or type(credential).__name__
        self._cache: dict[tuple[Any, ...], AccessToken] = {}
        self._inflight: dict[tuple[Any, ...], _TokenAcquisitionState] = {}
        self._lock = threading.Lock()

    def get_token(
        self,
        *scopes: str,
        claims: str | None = None,
        tenant_id: str | None = None,
        enable_cae: bool = False,
        **kwargs: Any,
    ) -> AccessToken:
        cache_key = (
            tuple(scopes),
            claims,
            tenant_id,
            enable_cae,
            tuple(sorted((key, repr(value)) for key, value in kwargs.items())),
        )
        acquisition_state: _TokenAcquisitionState | None = None
        failure_message: str | None = None
        try:
            while True:
                cached = self._get_cached_token(cache_key)
                if cached is not None:
                    return cached
                with self._lock:
                    waiting_state = self._inflight.get(cache_key)
                    if waiting_state is None:
                        acquisition_state = _TokenAcquisitionState(threading.Event())
                        self._inflight[cache_key] = acquisition_state
                        break
                waiting_state.event.wait()
                if waiting_state.failure_message is not None:
                    raise CredentialUnavailableError(
                        message=waiting_state.failure_message
                    ) from None

            token = self._get_token_with_retries(
                cache_key,
                scopes,
                claims=claims,
                tenant_id=tenant_id,
                enable_cae=enable_cae,
                **kwargs,
            )
            with self._lock:
                self._cache[cache_key] = token
            return token
        except CredentialUnavailableError as exc:
            failure_message = str(exc)
            raise
        finally:
            if acquisition_state is not None:
                with self._lock:
                    acquisition_state.failure_message = failure_message
                    if self._inflight.get(cache_key) is acquisition_state:
                        del self._inflight[cache_key]
                    acquisition_state.event.set()

    def _get_token_with_retries(
        self,
        cache_key: tuple[Any, ...],
        scopes: tuple[str, ...],
        *,
        claims: str | None,
        tenant_id: str | None,
        enable_cae: bool,
        **kwargs: Any,
    ) -> AccessToken:
        last_error: BaseException | None = None
        attempts_made = 0
        started_at = self._monotonic()
        last_attempt_seconds = 0.0
        for attempt in range(1, self._max_attempts + 1):
            if attempt > 1:
                elapsed_before_attempt = self._monotonic() - started_at
                remaining_budget = self._retry_budget_seconds - elapsed_before_attempt
                if elapsed_before_attempt > self._retry_budget_seconds:
                    break
                if last_attempt_seconds > 0 and remaining_budget < last_attempt_seconds:
                    break
            attempts_made = attempt
            cached = self._get_cached_token(cache_key)
            if cached is not None:
                return cached

            attempt_started_at = self._monotonic()
            try:
                token = self._credential.get_token(
                    *scopes,
                    claims=claims,
                    tenant_id=tenant_id,
                    enable_cae=enable_cae,
                    **kwargs,
                )
            except (
                CredentialUnavailableError,
                ClientAuthenticationError,
                AzureError,
                OSError,
            ) as exc:
                last_attempt_seconds = max(self._monotonic() - attempt_started_at, 0.0)
                last_error = exc
                if attempt >= self._max_attempts:
                    break
                delay = self._retry_delay(attempt)
                elapsed = self._monotonic() - started_at
                remaining_budget = self._retry_budget_seconds - elapsed
                if remaining_budget < 0:
                    break
                delay = min(delay, remaining_budget)
                if (
                    last_attempt_seconds > 0
                    and remaining_budget - delay < last_attempt_seconds
                ):
                    break
                _LOG.warning(
                    "Foundry credential token acquisition failed on attempt %s/%s with %s; "
                    "retrying in %.1f seconds.",
                    attempt,
                    self._max_attempts,
                    _safe_exception_type(exc),
                    delay,
                )
                self._sleep(delay)
                continue

            return token

        elapsed = self._monotonic() - started_at
        message = (
            "Failed to acquire a Foundry access token after "
            f"{attempts_made} attempts over {elapsed:.1f} seconds "
            f"for {_safe_scope_summary(tuple(scopes))}. "
            f"Last error type: {_safe_exception_type(last_error)}. "
            f"Credential providers tried: {self._provider_description}. "
            f"Retry budget: {self._retry_budget_seconds:.1f} seconds."
        )
        raise CredentialUnavailableError(message=message) from None

    def _get_cached_token(self, cache_key: tuple[Any, ...]) -> AccessToken | None:
        with self._lock:
            token = self._cache.get(cache_key)
        if token is None:
            return None
        if token.expires_on - time.time() <= self._token_refresh_skew_seconds:
            return None
        return token

    def _retry_delay(self, failed_attempt: int) -> float:
        base = self._initial_backoff_seconds * (2 ** (failed_attempt - 1))
        capped = min(base, self._max_backoff_seconds)
        if capped <= 0:
            return 0.0
        return min(self._jitter(capped * 0.75, capped), self._max_backoff_seconds)


def _build_default_credential_chain() -> FallbackTokenCredential:
    cli_process_timeout = _read_int_env(
        AUTH_CLI_PROCESS_TIMEOUT_ENV_VAR,
        DEFAULT_AUTH_CLI_PROCESS_TIMEOUT_SECONDS,
        minimum=1,
    )
    credentials: list[TokenCredential | tuple[TokenCredential, bool]] = [
        (EnvironmentCredential(), _environment_credential_configured())
    ]
    if _workload_identity_configured():
        credentials.append((WorkloadIdentityCredential(), True))
    credentials.extend(
        [
            ManagedIdentityCredential(),
            SharedTokenCacheCredential(),
            VisualStudioCodeCredential(),
            AzurePowerShellCredential(process_timeout=cli_process_timeout),
            AzureDeveloperCliCredential(process_timeout=cli_process_timeout),
            AzureCliCredential(process_timeout=cli_process_timeout),
        ]
    )
    return FallbackTokenCredential(*credentials)


def _default_provider_description() -> str:
    provider_names = ["EnvironmentCredential"]
    if _workload_identity_configured():
        provider_names.append("WorkloadIdentityCredential")
    provider_names.extend(
        [
            "ManagedIdentityCredential",
            "SharedTokenCacheCredential",
            "VisualStudioCodeCredential",
            "AzurePowerShellCredential",
            "AzureDeveloperCliCredential",
            "AzureCliCredential",
        ]
    )
    return ", ".join(provider_names)


@lru_cache(maxsize=1)
def _get_cached_foundry_credential() -> RetryingTokenCredential:
    return RetryingTokenCredential(
        _build_default_credential_chain(),
        max_attempts=_read_int_env(
            AUTH_MAX_ATTEMPTS_ENV_VAR, DEFAULT_AUTH_MAX_ATTEMPTS, minimum=1
        ),
        initial_backoff_seconds=_read_float_env(
            AUTH_INITIAL_BACKOFF_ENV_VAR,
            DEFAULT_AUTH_INITIAL_BACKOFF_SECONDS,
            minimum=0.0,
        ),
        max_backoff_seconds=_read_float_env(
            AUTH_MAX_BACKOFF_ENV_VAR,
            DEFAULT_AUTH_MAX_BACKOFF_SECONDS,
            minimum=0.0,
        ),
        retry_budget_seconds=_read_float_env(
            AUTH_RETRY_BUDGET_ENV_VAR,
            DEFAULT_AUTH_RETRY_BUDGET_SECONDS,
            minimum=0.0,
        ),
        token_refresh_skew_seconds=_read_int_env(
            AUTH_TOKEN_REFRESH_SKEW_ENV_VAR,
            DEFAULT_AUTH_TOKEN_REFRESH_SKEW_SECONDS,
            minimum=0,
        ),
        provider_description=_default_provider_description(),
    )


def get_foundry_credential() -> TokenCredential:
    """
    Return a cached, resilient credential for notebook, CLI, and E2E usage.

    The default chain favors unattended Azure Identity providers (environment,
    workload identity, managed identity, shared token cache) and keeps Azure CLI
    as the final fallback rather than the only token source.
    """
    return _get_cached_foundry_credential()


def get_foundry_access_token(
    resource_or_scope: str = DEFAULT_AZURE_AI_FOUNDRY_RESOURCE,
    *,
    credential: TokenCredential | None = None,
) -> AccessToken:
    credential = credential or get_foundry_credential()
    return credential.get_token(to_scope(resource_or_scope))


def build_bearer_auth_header(
    resource_or_scope: str = DEFAULT_AZURE_AI_FOUNDRY_RESOURCE,
    *,
    credential: TokenCredential | None = None,
) -> dict[str, str]:
    access_token = get_foundry_access_token(resource_or_scope, credential=credential)
    return {"Authorization": f"Bearer {access_token.token}"}
