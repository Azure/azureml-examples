from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from . import constants as foundry_constants
from .auth import get_foundry_access_token
from .command_jobs import get_job_status_by_name
from .rest import FoundryRestResponse, get_foundry_json
from .run_context import (
    JobWorkspaceContext,
    PROJECT_ROUTE_SOURCE,
    RouteTelemetry,
    _foundry_api_request,
    _project_scoped_api_url,
    resolve_job_workspace,
)

DEFAULT_API_VERSION = foundry_constants.DEFAULT_API_VERSION
DEFAULT_PROJECT_ENDPOINT = foundry_constants.DEFAULT_PROJECT_ENDPOINT
DEFAULT_PROJECT_NAME = foundry_constants.DEFAULT_PROJECT_NAME
DEFAULT_SUBSCRIPTION_ID = foundry_constants.DEFAULT_SUBSCRIPTION_ID


@dataclass(frozen=True)
class HistoryRunResult:
    response: FoundryRestResponse
    run: dict[str, Any]
    route_source: str = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None

    def summary(self) -> dict[str, Any]:
        summary = {
            "statusCode": self.response.status_code,
            "apimRequestId": self.response.apim_request_id,
            "runId": self.run.get("runId"),
            "runUuid": self.run.get("runUuid"),
            "status": self.run.get("status"),
            "displayName": self.run.get("displayName"),
            "experimentId": self.run.get("experimentId"),
            "startTimeUtc": self.run.get("startTimeUtc"),
            "endTimeUtc": self.run.get("endTimeUtc"),
            "duration": self.run.get("duration"),
            "target": self.run.get("target"),
            "runTypeV2": self.run.get("runTypeV2"),
        }
        summary.update(
            RouteTelemetry(
                route_source=self.route_source,
                project_response=self.project_response,
            ).evidence()
        )
        return summary


@dataclass(frozen=True)
class ExperimentNameResult:
    name: str
    route_source: str = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None


def _parse_response_json(response: FoundryRestResponse) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass(frozen=True)
class RunHistoryContext:
    response: FoundryRestResponse
    run: dict[str, Any]
    workspace: JobWorkspaceContext
    experiment_id: str
    experiment_name: str
    data_container_id: str | None
    route_source: str = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None
    experiment_route_source: str = PROJECT_ROUTE_SOURCE
    experiment_project_response: FoundryRestResponse | None = None


def _job_properties(job: dict[str, Any]) -> dict[str, Any]:
    properties = job.get("properties")
    return properties if isinstance(properties, dict) else {}


def _experiment_name_from_job(job: dict[str, Any]) -> str:
    properties = _job_properties(job)
    experiment_name = properties.get("experimentName") or job.get("experimentName")
    return str(experiment_name or "").strip()


def _resolve_job_experiment_name(
    job_name: str,
    *,
    access_token: str | None,
    project_endpoint: str,
    project_name: str,
    api_version: str,
) -> tuple[FoundryRestResponse, str]:
    job_status = get_job_status_by_name(
        job_name,
        access_token=access_token,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
    )
    if job_status.response.status_code != 200:
        return job_status.response, ""

    experiment_name = _experiment_name_from_job(job_status.job)
    if not experiment_name:
        raise RuntimeError(f"Job {job_name} did not include an experimentName.")
    return job_status.response, experiment_name


def _resolve_experiment_name(
    *,
    project_endpoint: str,
    project_name: str,
    api_version: str,
    experiment_id: str,
    access_token: str,
) -> ExperimentNameResult:
    project_url = _project_scoped_api_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        path=f"history/experimentids/{quote(experiment_id, safe='')}",
        api_version=api_version,
    )
    project_response = _foundry_api_request(
        "GET", project_url, access_token=access_token
    )
    project_body = _parse_response_json(project_response)
    project_name_value = str(project_body.get("name") or "").strip()
    if project_response.status_code == 200 and project_name_value:
        return ExperimentNameResult(
            name=project_name_value,
            route_source=PROJECT_ROUTE_SOURCE,
            project_response=project_response,
        )

    raise RuntimeError(
        f"Cannot resolve experiment '{experiment_id}' from project route: "
        f"{project_response.status_code}"
    )


def get_run_history(
    job_name: str,
    *,
    access_token: str | None = None,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
) -> HistoryRunResult:
    job_response, experiment_name = _resolve_job_experiment_name(
        job_name,
        access_token=access_token,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
    )
    if not experiment_name:
        return HistoryRunResult(
            response=job_response,
            run=_parse_response_json(job_response),
            route_source=PROJECT_ROUTE_SOURCE,
            project_response=job_response,
        )

    url = _project_scoped_api_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        path=(
            f"history/experiments/{quote(experiment_name, safe='')}"
            f"/runs/{quote(job_name, safe='')}"
        ),
        api_version=api_version,
    )
    response = get_foundry_json(
        url,
        access_token=access_token,
        raise_on_http_error=False,
    )

    return HistoryRunResult(
        response=response,
        run=_parse_response_json(response),
        route_source=PROJECT_ROUTE_SOURCE,
        project_response=response,
    )


def resolve_run_history_context(
    job_name: str,
    *,
    access_token: str | None = None,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
    subscription_id: str = DEFAULT_SUBSCRIPTION_ID,
) -> RunHistoryContext:
    token = access_token or get_foundry_access_token().token
    history_result = get_run_history(
        job_name,
        access_token=token,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
    )
    if history_result.response.status_code != 200:
        raise RuntimeError(
            f"Cannot get history for {job_name}: {history_result.response.status_code}"
        )

    workspace = resolve_job_workspace(
        job_name,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        subscription_id=subscription_id,
        access_token=token,
    )

    experiment_id = str(history_result.run.get("experimentId") or "")
    if not experiment_id:
        raise RuntimeError(f"History for {job_name} did not include an experimentId.")
    experiment_name_result = _resolve_experiment_name(
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        experiment_id=experiment_id,
        access_token=token,
    )

    return RunHistoryContext(
        response=history_result.response,
        run=history_result.run,
        workspace=workspace,
        experiment_id=experiment_id,
        experiment_name=experiment_name_result.name,
        data_container_id=history_result.run.get("dataContainerId"),
        route_source=getattr(history_result, "route_source", PROJECT_ROUTE_SOURCE),
        project_response=getattr(history_result, "project_response", None),
        experiment_route_source=experiment_name_result.route_source,
        experiment_project_response=experiment_name_result.project_response,
    )
