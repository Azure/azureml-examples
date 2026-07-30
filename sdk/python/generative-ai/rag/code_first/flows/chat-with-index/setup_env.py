import os
from typing import Union
from urllib.parse import urlparse

from promptflow import tool
from promptflow.connections import AzureOpenAIConnection, OpenAIConnection

# Allow-listed hostname suffixes for OpenAI-compatible endpoints. Only URLs
# whose host matches one of these suffixes are permitted to flow into the
# OPENAI_API_BASE environment variable, which is later used by the OpenAI
# HTTP client. This mitigates Server-Side Request Forgery (CWE-918) by
# preventing an attacker-controlled connection object from redirecting
# outbound requests to arbitrary internal endpoints (e.g. IMDS).
_ALLOWED_API_BASE_HOST_SUFFIXES = (
    ".openai.azure.com",
    ".cognitiveservices.azure.com",
    ".api.cognitive.microsoft.com",
    "api.openai.com",
)


def _validate_api_base(api_base: str) -> str:
    """Return ``api_base`` after validating it is an https URL pointing to
    an allow-listed OpenAI-compatible host. Raises ``ValueError`` otherwise.
    """
    if not isinstance(api_base, str) or not api_base:
        raise ValueError("connection.api_base must be a non-empty string")
    # Reject any whitespace or control characters. RFC 3986 forbids them in
    # URLs, and ``urlparse`` silently strips leading whitespace which would
    # otherwise let a value like " https://..." bypass the scheme check.
    if any(c.isspace() or ord(c) < 0x20 or c == "\x7f" for c in api_base):
        raise ValueError(
            "connection.api_base must not contain whitespace or control characters"
        )
    parsed = urlparse(api_base)
    if parsed.scheme != "https":
        raise ValueError(
            f"connection.api_base must use https scheme, got: {parsed.scheme!r}"
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("connection.api_base is missing a hostname")
    if not any(
        host == suffix.lstrip(".") or host.endswith(suffix)
        for suffix in _ALLOWED_API_BASE_HOST_SUFFIXES
    ):
        raise ValueError(
            f"connection.api_base host {host!r} is not in the allow-list "
            f"of OpenAI-compatible endpoints"
        )
    return api_base


@tool
def setup_env(connection: Union[AzureOpenAIConnection, OpenAIConnection], config: dict):
    if not connection or not config:
        return

    if isinstance(connection, AzureOpenAIConnection):
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = _validate_api_base(connection.api_base)
        os.environ["OPENAI_API_KEY"] = connection.api_key
        os.environ["OPENAI_API_VERSION"] = connection.api_version

    if isinstance(connection, OpenAIConnection):
        os.environ["OPENAI_API_KEY"] = connection.api_key
        if connection.organization is not None:
            os.environ["OPENAI_ORG_ID"] = connection.organization

    for key in config:
        os.environ[key] = str(config[key])

    return "Ready"
