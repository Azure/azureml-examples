from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote

from . import constants as foundry_constants
from .auth import get_foundry_access_token
from .rest import FoundryRestResponse, _request_foundry, create_request_id

DEFAULT_SUBSCRIPTION_ID = foundry_constants.DEFAULT_SUBSCRIPTION_ID
DEFAULT_RESOURCE_GROUP = foundry_constants.DEFAULT_RESOURCE_GROUP
DEFAULT_WORKSPACE_NAME = foundry_constants.DEFAULT_WORKSPACE_NAME
DEFAULT_MLFLOW_TRACKING_HOST = foundry_constants.DEFAULT_MLFLOW_TRACKING_HOST
PROJECT_ROUTE_SOURCE = "project"
CONTINUATION_ROUTE_SOURCE = "continuation"
MLFLOW_REGIONAL_ROUTE_SOURCE = "mlflow-regional"


def _foundry_api_request(
    method: str,
    url: str,
    *,
    access_token: str,
    timeout: int = 120,
    raise_on_http_error: bool = False,
) -> FoundryRestResponse:
    return _request_foundry(
        method,
        url,
        access_token=access_token,
        request_id=create_request_id(),
        timeout_seconds=timeout,
        raise_on_http_error=raise_on_http_error,
    )


def _build_project_base_url(
    project_endpoint: str,
    project_name: str | None = None,
) -> tuple[str, str]:
    endpoint = project_endpoint.rstrip("/")
    match = re.search(r"/api/projects/([^/]+)$", endpoint)
    if match:
        return endpoint, match.group(1)

    project_match = re.search(r"/projects/([^/]+)/?$", endpoint)
    if project_match:
        project_name = project_match.group(1)
        base = endpoint[: project_match.start()]
        return f"{base}/api/projects/{project_name}", project_name

    resolved_project_name = project_name or endpoint.rsplit("/", 1)[-1]
    return f"{endpoint}/api/projects/{resolved_project_name}", resolved_project_name


def _project_api_url(base_url: str, path: str, api_version: str) -> str:
    separator = "&" if "?" in path else "?"
    return f"{base_url}/{path.lstrip('/')}{separator}api-version={api_version}"


def _project_scoped_api_url(
    *,
    project_endpoint: str,
    project_name: str,
    path: str,
    api_version: str,
) -> str:
    base_url, _ = _build_project_base_url(project_endpoint, project_name=project_name)
    return _project_api_url(base_url, path, api_version)


def _project_job_api_url(
    *,
    project_endpoint: str,
    project_name: str,
    job_name: str,
    path: str,
    api_version: str,
) -> str:
    job_path = f"jobs/{quote(job_name, safe='')}/{path.lstrip('/')}"
    return _project_scoped_api_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        path=job_path,
        api_version=api_version,
    )


@dataclass(frozen=True)
class RouteTelemetry:
    route_source: str = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None

    def evidence(self) -> dict[str, object]:
        evidence: dict[str, object] = {"routeSource": self.route_source}
        if self.project_response is not None:
            evidence["projectStatusCode"] = self.project_response.status_code
            evidence["projectApimRequestId"] = self.project_response.apim_request_id
        return evidence


def route_telemetry_evidence(value: object) -> dict[str, object]:
    route_source = getattr(value, "route_source", None)
    if not route_source:
        return {}
    evidence: dict[str, object] = {"routeSource": route_source}
    project_response = getattr(value, "project_response", None)
    if project_response is not None:
        evidence["projectStatusCode"] = project_response.status_code
        evidence["projectApimRequestId"] = project_response.apim_request_id
    return evidence


def build_workspace_path(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
) -> str:
    return (
        f"/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}"
        f"/providers/Microsoft.MachineLearningServices"
        f"/workspaces/{workspace_name}"
    )


@dataclass(frozen=True)
class JobWorkspaceContext:
    job_name: str
    resource_group: str
    workspace_name: str
    subscription_id: str = DEFAULT_SUBSCRIPTION_ID
    mlflow_tracking_host: str = DEFAULT_MLFLOW_TRACKING_HOST

    @property
    def workspace_path(self) -> str:
        return build_workspace_path(
            self.subscription_id,
            self.resource_group,
            self.workspace_name,
        )

    @property
    def mlflow_base_url(self) -> str:
        return (
            f"{self.mlflow_tracking_host.rstrip('/')}/mlflow/v1.0"
            f"/subscriptions/{quote(self.subscription_id, safe='')}"
            f"/resourceGroups/{quote(self.resource_group, safe='')}"
            "/providers/Microsoft.MachineLearningServices"
            f"/workspaces/{quote(self.workspace_name, safe='')}"
            "/api/2.0/mlflow"
        )

    def mlflow_run_url(self, run_id: str) -> str:
        return f"{self.mlflow_base_url}/runs/get?run_id={quote(run_id, safe='')}"


def resolve_job_workspace(
    job_name: str,
    *,
    project_endpoint: str,
    project_name: str = foundry_constants.DEFAULT_PROJECT_NAME,
    api_version: str,
    subscription_id: str = DEFAULT_SUBSCRIPTION_ID,
    mlflow_tracking_host: str = DEFAULT_MLFLOW_TRACKING_HOST,
    access_token: str | None = None,
    print_progress: bool = False,
) -> JobWorkspaceContext:
    token = access_token or get_foundry_access_token().token
    base_url, _ = _build_project_base_url(project_endpoint, project_name=project_name)
    job_url = _project_api_url(base_url, f"jobs/{job_name}", api_version)
    job_resp = _foundry_api_request("GET", job_url, access_token=token)

    resource_group = DEFAULT_RESOURCE_GROUP
    workspace_name = DEFAULT_WORKSPACE_NAME

    if job_resp.status_code == 200:
        job_body = job_resp.json() or {}
        # Parse workspace path from the job's ARM id field, which is always
        # present and has the canonical subscription/RG/workspace triple.
        # Example id: /subscriptions/.../resourceGroups/.../providers/
        #   Microsoft.MachineLearningServices/workspaces/<ws>/jobs/<name>
        job_id_field = job_body.get("id", "")
        ws_match = re.search(
            r"/subscriptions/[^/]+/resourceGroups/([^/]+)"
            r"/providers/Microsoft\.MachineLearningServices/workspaces/([^/]+)",
            job_id_field,
            re.IGNORECASE,
        )
        if ws_match:
            resource_group = ws_match.group(1)
            workspace_name = ws_match.group(2)

    if print_progress:
        print(f"  Resource Group: {resource_group}")
        print(f"  Workspace:      {workspace_name}")

    return JobWorkspaceContext(
        job_name=job_name,
        resource_group=resource_group,
        workspace_name=workspace_name,
        subscription_id=subscription_id,
        mlflow_tracking_host=mlflow_tracking_host,
    )
