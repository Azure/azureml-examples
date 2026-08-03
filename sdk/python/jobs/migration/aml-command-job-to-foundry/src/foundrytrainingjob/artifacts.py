from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import quote, urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from . import constants as foundry_constants
from .auth import get_foundry_access_token
from .history import resolve_run_history_context
from .rest import FoundryRestResponse
from .run_context import (
    PROJECT_ROUTE_SOURCE,
    _foundry_api_request,
    _project_scoped_api_url,
)

DEFAULT_SUBSCRIPTION_ID = foundry_constants.DEFAULT_SUBSCRIPTION_ID
RUN_ARTIFACT_ORIGIN = "ExperimentRun"


def _empty_response() -> FoundryRestResponse:
    return FoundryRestResponse(status_code=0, headers={}, text="")


def _append_query(url: str, query: dict[str, str | int]) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode(query)}"


def _parse_response_json(response: FoundryRestResponse) -> dict:
    if not response.text:
        return {}
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _content_info_has_uri(response: FoundryRestResponse) -> bool:
    if response.status_code != 200 or not response.text:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and bool(body.get("contentUri"))


def _project_history_artifacts_url(
    *,
    project_endpoint: str,
    project_name: str,
    api_version: str,
    job_name: str,
    experiment_id: str,
    path_prefix: str | None = None,
) -> str:
    url = _project_scoped_api_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        path=(
            f"history/experimentids/{quote(experiment_id, safe='')}"
            f"/runs/{quote(job_name, safe='')}/artifacts"
        ),
        api_version=api_version,
    )
    if path_prefix:
        url = _append_query(url, {"path": path_prefix})
    return url


def _project_history_artifact_content_info_url(
    *,
    project_endpoint: str,
    project_name: str,
    api_version: str,
    job_name: str,
    experiment_id: str,
    artifact_path: str,
) -> str:
    url = _project_scoped_api_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        path=(
            f"history/experimentids/{quote(experiment_id, safe='')}"
            f"/runs/{quote(job_name, safe='')}/artifacts/contentinfo"
        ),
        api_version=api_version,
    )
    return _append_query(url, {"path": artifact_path})


def _project_history_artifact_metadata_url(
    *,
    project_endpoint: str,
    project_name: str,
    api_version: str,
    job_name: str,
    experiment_id: str,
    artifact_path: str,
) -> str:
    url = _project_scoped_api_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        path=(
            f"history/experimentids/{quote(experiment_id, safe='')}"
            f"/runs/{quote(job_name, safe='')}/artifacts/metadata"
        ),
        api_version=api_version,
    )
    return _append_query(url, {"path": artifact_path})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactListResult:
    """Result of listing artifacts via the Run History API."""

    artifacts: list[dict]
    experiment_id: str
    experiment_name: str
    resource_group: str
    workspace_name: str
    storage_account: str | None = None
    blob_container: str | None = None
    selected_log_artifact: str | None = None
    pages: int = 1
    is_complete: bool = True
    response: FoundryRestResponse = field(default_factory=_empty_response)
    route_source: str = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None
    experiment_route_source: str | None = None
    experiment_project_response: FoundryRestResponse | None = None


@dataclass(frozen=True)
class ArtifactContentResult:
    """Result of fetching artifact content via a SAS URI."""

    artifact_path: str
    content: str
    origin: str | None = None
    container: str | None = None
    route_source: str = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None


@dataclass(frozen=True)
class ProjectArtifactListResult:
    """Result of listing artifacts via the workspace-scoped artifact service."""

    response: FoundryRestResponse
    artifacts: list[dict]
    pages: int = 1
    next_link: str | None = None
    sample_artifact_path: str | None = None
    route_source: str = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None
    experiment_route_source: str | None = None
    experiment_project_response: FoundryRestResponse | None = None


@dataclass(frozen=True)
class ProjectArtifactMetadataResult:
    """Result of fetching metadata for a workspace-scoped artifact."""

    response: FoundryRestResponse
    metadata: dict
    route_source: str = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None
    experiment_route_source: str | None = None
    experiment_project_response: FoundryRestResponse | None = None


@dataclass(frozen=True)
class ProjectArtifactContentInfoResult:
    """Result of fetching content info for a workspace-scoped artifact."""

    response: FoundryRestResponse
    content_info: dict
    route_source: str = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None
    experiment_route_source: str | None = None
    experiment_project_response: FoundryRestResponse | None = None


@dataclass(frozen=True)
class ProjectArtifactContentResult:
    """Result of fetching artifact-service content for an artifact."""

    response: FoundryRestResponse
    artifact_path: str
    content: str
    route_source: str = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None
    experiment_route_source: str | None = None
    experiment_project_response: FoundryRestResponse | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_run_history_artifacts(
    job_name: str,
    *,
    project_endpoint: str,
    project_name: str = foundry_constants.DEFAULT_PROJECT_NAME,
    api_version: str,
    subscription_id: str = DEFAULT_SUBSCRIPTION_ID,
    access_token: str | None = None,
    print_progress: bool = True,
) -> ArtifactListResult:
    """List all artifacts for a job via the Run History API.

    Returns an :class:`ArtifactListResult` with every artifact entry plus
    auto-detected workspace coordinates and (when available) storage account
    details.
    """
    token = access_token or get_foundry_access_token().token
    if print_progress:
        print(f"Step 1 — Resolving Run History context for job: {job_name}...")
    history_context = resolve_run_history_context(
        job_name,
        access_token=token,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        subscription_id=subscription_id,
    )
    experiment_id = history_context.experiment_id
    experiment_name = history_context.experiment_name
    workspace_context = history_context.workspace
    if print_progress:
        print(f"  Experiment ID:   {experiment_id}")
        print(f"  Experiment Name: {experiment_name}")
        print(f"\nStep 2 — Extracting workspace from job services...")
    resource_group = workspace_context.resource_group
    workspace_name = workspace_context.workspace_name

    # Step 3: paginate the project-scoped Run History artifacts endpoint
    project_artifacts_url = _project_history_artifacts_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        job_name=job_name,
        experiment_id=experiment_id,
    )

    if print_progress:
        print(f"\nListing artifacts (paginated)...")

    artifact_list: list[dict] = []
    page = 1
    url: str | None = project_artifacts_url
    last_response: FoundryRestResponse | None = None
    is_complete = True
    route_source = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None

    while url:
        resp = _foundry_api_request("GET", url, access_token=token)
        last_response = resp
        if resp.status_code != 200:
            project_response = resp
            is_complete = False
            if print_progress:
                print(f"  Failed with status {resp.status_code}")
            break

        body = resp.json()
        items = body.get("value", [])
        for item in items:
            artifact_list.append(item)
            if print_progress:
                path = item.get("path", "<unknown>")
                atype = item.get("type", "?")
                print(f"  [{len(artifact_list):3d}] ({atype:10s}) {path}")

        continuation = body.get("continuationToken")
        next_link = body.get("nextLink")
        if next_link:
            page += 1
            if print_progress:
                print(f"\n  ... page {page} ...")
            url = str(next_link)
        elif continuation:
            page += 1
            if print_progress:
                print(f"\n  ... page {page} ...")
            url = _append_query(
                project_artifacts_url,
                {"continuationToken": str(continuation)},
            )
        else:
            url = None

    if print_progress:
        print(f"\nTotal artifacts: {len(artifact_list)}  (across {page} page(s))")

    # Optionally probe storage account from the first artifact's SAS URI
    storage_account = None
    blob_container = None
    if artifact_list:
        first_path = artifact_list[0].get("path", "")
        ci_url = _project_history_artifact_content_info_url(
            project_endpoint=project_endpoint,
            project_name=project_name,
            api_version=api_version,
            job_name=job_name,
            experiment_id=experiment_id,
            artifact_path=first_path,
        )
        ci_resp = _foundry_api_request("GET", ci_url, access_token=token)
        if ci_resp.status_code == 200:
            sas_uri = ci_resp.json().get("contentUri", "")
            sa_match = re.search(
                r"https://([^.]+)\.blob\.core\.windows\.net/([^/?]+)", sas_uri
            )
            if sa_match:
                storage_account = sa_match.group(1)
                blob_container = sa_match.group(2)

    # Pick the first log artifact
    log_artifacts = [
        a
        for a in artifact_list
        if "std_log" in a.get("path", "") or a.get("path", "").endswith(".txt")
    ]
    selected = (
        log_artifacts[0]["path"]
        if log_artifacts
        else (artifact_list[0]["path"] if artifact_list else None)
    )

    return ArtifactListResult(
        response=last_response
        if last_response is not None
        else FoundryRestResponse(status_code=0, headers={}, text=""),
        artifacts=artifact_list,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        resource_group=resource_group,
        workspace_name=workspace_name,
        storage_account=storage_account,
        blob_container=blob_container,
        selected_log_artifact=selected,
        pages=page,
        is_complete=is_complete,
        route_source=route_source,
        project_response=project_response
        if project_response is not None
        else (last_response if route_source == PROJECT_ROUTE_SOURCE else None),
        experiment_route_source=history_context.experiment_route_source,
        experiment_project_response=history_context.experiment_project_response,
    )


def list_job_artifacts(
    job_name: str,
    *,
    project_endpoint: str,
    api_version: str,
    project_name: str = foundry_constants.DEFAULT_PROJECT_NAME,
    access_token: str | None = None,
    path_prefix: str | None = None,
    max_pages: int = 10,
    subscription_id: str = DEFAULT_SUBSCRIPTION_ID,
) -> ProjectArtifactListResult:
    token = access_token or get_foundry_access_token().token
    history_context = resolve_run_history_context(
        job_name,
        access_token=token,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        subscription_id=subscription_id,
    )
    project_url = _project_history_artifacts_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        job_name=job_name,
        experiment_id=history_context.experiment_id,
        path_prefix=path_prefix,
    )

    collected_artifacts: list[dict] = []
    pages = 0
    last_response: FoundryRestResponse | None = None
    next_link: str | None = None
    current_url: str | None = project_url
    route_source = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None

    while current_url and pages < max_pages:
        last_response = _foundry_api_request(
            "GET",
            current_url,
            access_token=token,
        )
        pages += 1
        if last_response.status_code != 200:
            project_response = last_response
            next_link = None
            break
        body = _parse_response_json(last_response)

        collected_artifacts.extend(body.get("value", []))
        next_link = body.get("nextLink")
        continuation = body.get("continuationToken")
        if next_link:
            current_url = str(next_link)
        elif continuation:
            current_url = _append_query(
                project_url,
                {"continuationToken": str(continuation)},
            )
        else:
            current_url = None

    return ProjectArtifactListResult(
        response=last_response
        if last_response is not None
        else FoundryRestResponse(status_code=0, headers={}, text=""),
        artifacts=collected_artifacts,
        pages=pages or 1,
        next_link=next_link,
        sample_artifact_path=collected_artifacts[0].get("path")
        if collected_artifacts
        else None,
        route_source=route_source,
        project_response=project_response
        if project_response is not None
        else (last_response if route_source == PROJECT_ROUTE_SOURCE else None),
        experiment_route_source=history_context.experiment_route_source,
        experiment_project_response=history_context.experiment_project_response,
    )


def get_job_artifact_metadata(
    job_name: str,
    artifact_path: str,
    *,
    project_endpoint: str,
    api_version: str,
    project_name: str = foundry_constants.DEFAULT_PROJECT_NAME,
    access_token: str | None = None,
    subscription_id: str = DEFAULT_SUBSCRIPTION_ID,
) -> ProjectArtifactMetadataResult:
    token = access_token or get_foundry_access_token().token
    history_context = resolve_run_history_context(
        job_name,
        access_token=token,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        subscription_id=subscription_id,
    )
    project_url = _project_history_artifact_metadata_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        job_name=job_name,
        experiment_id=history_context.experiment_id,
        artifact_path=artifact_path,
    )
    response = _foundry_api_request("GET", project_url, access_token=token)
    route_source = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = response
    return ProjectArtifactMetadataResult(
        response=response,
        metadata=response.json() if response.text else {},
        route_source=route_source,
        project_response=project_response,
        experiment_route_source=history_context.experiment_route_source,
        experiment_project_response=history_context.experiment_project_response,
    )


def get_job_artifact_content_info(
    job_name: str,
    artifact_path: str,
    *,
    project_endpoint: str,
    api_version: str,
    project_name: str = foundry_constants.DEFAULT_PROJECT_NAME,
    access_token: str | None = None,
    subscription_id: str = DEFAULT_SUBSCRIPTION_ID,
) -> ProjectArtifactContentInfoResult:
    token = access_token or get_foundry_access_token().token
    history_context = resolve_run_history_context(
        job_name,
        access_token=token,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        subscription_id=subscription_id,
    )
    project_url = _project_history_artifact_content_info_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        job_name=job_name,
        experiment_id=history_context.experiment_id,
        artifact_path=artifact_path,
    )
    response = _foundry_api_request("GET", project_url, access_token=token)
    route_source = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = response
    return ProjectArtifactContentInfoResult(
        response=response,
        content_info=response.json() if response.text else {},
        route_source=route_source,
        project_response=project_response,
        experiment_route_source=history_context.experiment_route_source,
        experiment_project_response=history_context.experiment_project_response,
    )


def get_job_artifact_content(
    job_name: str,
    artifact_path: str,
    *,
    project_endpoint: str,
    api_version: str,
    project_name: str = foundry_constants.DEFAULT_PROJECT_NAME,
    access_token: str | None = None,
    tail_bytes: int | None = None,
    subscription_id: str = DEFAULT_SUBSCRIPTION_ID,
) -> ProjectArtifactContentResult:
    content_info = get_job_artifact_content_info(
        job_name,
        artifact_path,
        project_endpoint=project_endpoint,
        api_version=api_version,
        project_name=project_name,
        access_token=access_token,
        subscription_id=subscription_id,
    )
    content_uri = content_info.content_info.get("contentUri", "")
    if not content_uri:
        raise RuntimeError(
            f"Artifact content-info for '{artifact_path}' did not return a contentUri."
        )

    request_headers = (
        {"Range": f"bytes=-{tail_bytes}"} if tail_bytes is not None else None
    )
    req = Request(url=content_uri, headers=request_headers or {}, method="GET")
    try:
        with urlopen(req, timeout=30) as blob_resp:
            content = blob_resp.read().decode("utf-8", errors="replace")
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"Blob download failed for '{artifact_path}': {error.code} — {error_body}"
        ) from error

    return ProjectArtifactContentResult(
        response=content_info.response,
        artifact_path=artifact_path,
        content=content,
        route_source=content_info.route_source,
        project_response=content_info.project_response,
        experiment_route_source=content_info.experiment_route_source,
        experiment_project_response=content_info.experiment_project_response,
    )


def fetch_artifact_content(
    artifact_path: str,
    *,
    job_name: str,
    project_endpoint: str,
    project_name: str = foundry_constants.DEFAULT_PROJECT_NAME,
    api_version: str,
    experiment_id: str,
    access_token: str | None = None,
    print_progress: bool = True,
    max_display_chars: int = 5000,
) -> ArtifactContentResult:
    """Fetch an artifact's content by getting a SAS URI, then downloading from blob."""
    token = access_token or get_foundry_access_token().token

    contentinfo_url = _project_history_artifact_content_info_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        job_name=job_name,
        experiment_id=experiment_id,
        artifact_path=artifact_path,
    )
    route_source = PROJECT_ROUTE_SOURCE
    project_response: FoundryRestResponse | None = None

    if print_progress:
        print(f"Fetching SAS URI for: {artifact_path}...")
    resp = _foundry_api_request("GET", contentinfo_url, access_token=token)
    project_response = resp

    if resp.status_code != 200:
        raise RuntimeError(
            f"contentinfo failed for '{artifact_path}': "
            f"{resp.status_code} — {resp.text[:500]}"
        )

    ci_body = resp.json()
    content_uri = ci_body.get("contentUri", "")
    origin = ci_body.get("origin")
    container = ci_body.get("container")

    if print_progress:
        masked = (
            content_uri.split("?")[0] + "?<SAS>" if "?" in content_uri else content_uri
        )
        print(f"  Origin:    {origin}")
        print(f"  Container: {container}")
        print(f"  URI:       {masked[:120]}")
        print(f"\nDownloading content from SAS URI...")

    req = Request(url=content_uri, method="GET")
    try:
        with urlopen(req, timeout=30) as blob_resp:
            log_content = blob_resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"Blob download failed for '{artifact_path}': {e.code} — {error_body}"
        ) from e

    if print_progress:
        print(f"  Downloaded {len(log_content)} chars")
        print(f"\n{'─' * 70}")
        print(log_content[:max_display_chars])
        if len(log_content) > max_display_chars:
            print(
                f"\n... truncated ({len(log_content) - max_display_chars} more chars)"
            )
        print(f"{'─' * 70}")

    return ArtifactContentResult(
        artifact_path=artifact_path,
        content=log_content,
        origin=origin,
        container=container,
        route_source=route_source,
        project_response=project_response
        if project_response is not None
        else (resp if route_source == PROJECT_ROUTE_SOURCE else None),
    )
