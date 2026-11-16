from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from foundrytrainingjob import artifacts
from foundrytrainingjob.rest import FoundryRestResponse


def _response(body: dict) -> FoundryRestResponse:
    return FoundryRestResponse(status_code=200, headers={}, text=json.dumps(body))


def test_list_job_artifacts_rejects_cross_origin_next_link_before_request(
    monkeypatch,
):
    history_context = SimpleNamespace(
        experiment_id="experiment-id",
        experiment_route_source=None,
        experiment_project_response=None,
    )
    monkeypatch.setattr(
        artifacts,
        "resolve_run_history_context",
        lambda *args, **kwargs: history_context,
    )
    requests: list[tuple[str, str, str | None]] = []

    def fake_request(method: str, url: str, *, access_token: str | None = None):
        requests.append((method, url, access_token))
        return _response(
            {
                "value": [{"path": "outputs/result.json"}],
                "nextLink": "https://attacker.example/artifacts?page=2",
            }
        )

    monkeypatch.setattr(artifacts, "_foundry_api_request", fake_request)

    with pytest.raises(ValueError, match="original Foundry origin"):
        artifacts.list_job_artifacts(
            "job-name",
            project_endpoint="https://project.example",
            project_name="project-name",
            api_version="2025-05-15-preview",
            access_token="secret-token",
        )

    assert len(requests) == 1
    assert requests[0][2] == "secret-token"


def test_validated_pagination_url_resolves_same_origin_relative_link():
    assert (
        artifacts._validated_pagination_url(
            "/api/projects/project/history/artifacts?page=2",
            request_url="https://project.example/api/projects/project/history/artifacts",
        )
        == "https://project.example/api/projects/project/history/artifacts?page=2"
    )
