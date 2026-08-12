from __future__ import annotations

from io import BytesIO
import socket
from urllib.error import HTTPError, URLError

import pytest

from foundrytrainingjob import rest


class _Response:
    status = 200
    headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def read(self) -> bytes:
        return b'{"ok":true}'


def _set_urlopen_outcomes(monkeypatch, *outcomes):
    remaining = iter(outcomes)
    calls = []

    def fake_urlopen(request, *, timeout):
        calls.append((request, timeout))
        outcome = next(remaining)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(rest, "urlopen", fake_urlopen)
    monkeypatch.setattr(rest.time, "sleep", lambda delay: None)
    monkeypatch.setattr(rest.random, "uniform", lambda lower, upper: 0.0)
    return calls


def _http_error(status_code: int) -> HTTPError:
    return HTTPError(
        "https://project.example/jobs",
        status_code,
        "transient error",
        {},
        BytesIO(b'{"error":{"code":"Transient"}}'),
    )


def test_post_is_not_replayed_after_ambiguous_timeout(monkeypatch):
    calls = _set_urlopen_outcomes(monkeypatch, TimeoutError("read timed out"))

    with pytest.raises(rest.FoundryNetworkError, match="read timed out"):
        rest.post_foundry_json(
            "https://project.example/jobs/job/cancel",
            access_token="token",
            retry_transient=True,
        )

    assert len(calls) == 1


def test_disable_retry_prevents_retry_for_pre_send_failure(monkeypatch):
    calls = _set_urlopen_outcomes(
        monkeypatch,
        URLError(socket.gaierror("name resolution failed")),
    )

    with pytest.raises(rest.FoundryNetworkError):
        rest.get_foundry_json(
            "https://project.example/jobs",
            access_token="token",
            disable_retry=True,
        )

    assert len(calls) == 1


def test_disable_retry_prevents_retry_for_transient_http_response(monkeypatch):
    calls = _set_urlopen_outcomes(monkeypatch, _http_error(503))

    with pytest.raises(rest.FoundryHttpError):
        rest.get_foundry_json(
            "https://project.example/jobs",
            access_token="token",
            disable_retry=True,
        )

    assert len(calls) == 1


@pytest.mark.parametrize("status_code", [408, 503, 504])
def test_post_is_not_replayed_after_ambiguous_server_error(
    monkeypatch,
    status_code,
):
    calls = _set_urlopen_outcomes(monkeypatch, _http_error(status_code))

    with pytest.raises(rest.FoundryHttpError):
        rest.post_foundry_json(
            "https://project.example/jobs/job/cancel",
            access_token="token",
            retry_transient=True,
        )

    assert len(calls) == 1


def test_pre_send_failure_can_retry_mutation_within_budget(monkeypatch):
    calls = _set_urlopen_outcomes(
        monkeypatch,
        URLError(ConnectionRefusedError("connection refused")),
        _Response(),
    )

    response = rest.post_foundry_json(
        "https://project.example/jobs/job/cancel",
        access_token="token",
    )

    assert response.status_code == 200
    assert len(calls) == 2


def test_safe_network_retries_are_bounded(monkeypatch):
    monkeypatch.setattr(rest, "NETWORK_RETRY_MAX_ATTEMPTS", 2)
    calls = _set_urlopen_outcomes(
        monkeypatch,
        TimeoutError("read timed out"),
        TimeoutError("read timed out"),
        TimeoutError("read timed out"),
    )

    with pytest.raises(rest.FoundryNetworkError):
        rest.get_foundry_json(
            "https://project.example/jobs",
            access_token="token",
        )

    assert len(calls) == 3


def test_disable_retry_prevents_canary_retry():
    assert (
        rest._maybe_record_canary_retry(
            method="PUT",
            url="https://project.example/jobs/job-name",
            canary_attempt=0,
            error_response=rest.FoundryRestResponse(
                status_code=400,
                headers={},
                text='{"error":{"code":"CanarySignature"}}',
            ),
            disable_retry=True,
        )
        is False
    )


def test_safe_get_can_retry_ambiguous_timeout(monkeypatch):
    calls = _set_urlopen_outcomes(
        monkeypatch,
        TimeoutError("read timed out"),
        _Response(),
    )

    response = rest.get_foundry_json(
        "https://project.example/jobs",
        access_token="token",
    )

    assert response.status_code == 200
    assert len(calls) == 2
