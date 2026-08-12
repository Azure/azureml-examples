from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from secrets import choice as secrets_choice
from typing import Any
from urllib.parse import quote, unquote, urlencode, urljoin, urlsplit

from . import constants as job_constants
from .rest import (
    DEFAULT_TIMEOUT_SECONDS,
    delete_foundry,
    FoundryRestResponse,
    create_request_id,
    get_foundry_json,
    post_foundry_json,
    put_foundry_json,
)

DEFAULT_API_VERSION = job_constants.DEFAULT_API_VERSION
DEFAULT_PROJECT_ENDPOINT = job_constants.DEFAULT_PROJECT_ENDPOINT
DEFAULT_PROJECT_NAME = job_constants.DEFAULT_PROJECT_NAME
DEFAULT_COMMAND = job_constants.DEFAULT_COMMAND
DEFAULT_CURATED_ENVIRONMENT_ID = job_constants.DEFAULT_CURATED_ENVIRONMENT_ID
DEFAULT_CUSTOM_ACR_ENVIRONMENT_ID = job_constants.DEFAULT_CUSTOM_ACR_ENVIRONMENT_ID
DEFAULT_COMPUTE_ID = job_constants.DEFAULT_COMPUTE_ID
DEFAULT_GPU_COMPUTE_ID = job_constants.DEFAULT_GPU_COMPUTE_ID
DEFAULT_INSTANCE_TYPE = job_constants.DEFAULT_INSTANCE_TYPE
DEFAULT_INSTANCE_COUNT = job_constants.DEFAULT_INSTANCE_COUNT
DEFAULT_GPU_INSTANCE_TYPE = job_constants.DEFAULT_GPU_INSTANCE_TYPE
DEFAULT_GPU_INSTANCE_COUNT = job_constants.DEFAULT_GPU_INSTANCE_COUNT
DEFAULT_IDENTITY_UAI = job_constants.DEFAULT_IDENTITY_UAI
DEFAULT_AISUPERCOMPUTER_PROPERTIES = job_constants.DEFAULT_AISUPERCOMPUTER_PROPERTIES
DEFAULT_ENVIRONMENT_VARIABLES = job_constants.DEFAULT_ENVIRONMENT_VARIABLES
DEFAULT_CANARY_TAGS = job_constants.DEFAULT_CANARY_TAGS
DEFAULT_CPU_RESOURCES = job_constants.DEFAULT_CPU_RESOURCES
DEFAULT_GPU_RESOURCES = job_constants.DEFAULT_GPU_RESOURCES
JOB_LIST_TARGET_TAG_PREFIX = "e2eListTarget"
JOB_NAME_SUFFIX_ALPHABET = job_constants.JOB_NAME_SUFFIX_ALPHABET


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_mapping(source: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        if key in source:
            return _as_dict(source[key])
    return {}


def _compact_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _environment_reference(job_properties: Mapping[str, Any]) -> str | None:
    # Foundry renamed the request-side field from `environmentId` to
    # `environmentImageReference`. GET responses may still expose the legacy
    # `environmentId` key for jobs that were created before the rename, so we
    # read both when summarising a job, but only WRITE `environmentImageReference`
    # in any request we construct.
    environment_image_reference = job_properties.get("environmentImageReference")
    if environment_image_reference:
        return str(environment_image_reference)
    environment_id = job_properties.get("environmentId")
    if environment_id:
        return str(environment_id)
    return None


def _parse_response_json(response: FoundryRestResponse) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}

    return payload if isinstance(payload, dict) else {}


def _build_job_status_summary(
    job_status_response: FoundryRestResponse,
    job_status: Mapping[str, Any],
    *,
    fallback_job_name: str | None = None,
) -> dict[str, Any]:
    job_properties = _as_dict(job_status.get("properties"))
    job_runtime_properties = _as_dict(job_properties.get("properties"))
    job_resources = _as_dict(job_properties.get("resources"))
    job_services = _as_dict(job_properties.get("services"))
    job_inputs = _first_mapping(job_properties, "inputs", "Inputs")
    job_outputs = _first_mapping(job_properties, "outputs", "Outputs")
    default_output = _first_mapping(job_outputs, "default")
    job_tags = _as_dict(job_properties.get("tags"))
    job_system_data = _as_dict(job_status.get("systemData"))
    job_distribution = _as_dict(job_properties.get("distribution"))
    job_limits = _as_dict(job_properties.get("limits"))
    job_queue_settings = _as_dict(job_properties.get("queueSettings"))
    job_error = _as_dict(job_status.get("error"))
    error_message = job_error.get("message")
    if error_message is None and job_status_response.status_code >= 400:
        error_message = job_status_response.text or None

    return _compact_dict(
        {
            "statusCode": job_status_response.status_code,
            "apimRequestId": job_status_response.apim_request_id,
            "jobId": job_status.get("id"),
            "jobName": job_status.get("name", fallback_job_name),
            "displayName": job_properties.get("displayName"),
            "status": job_properties.get("status"),
            "createdAt": job_system_data.get("createdAt"),
            "createdBy": job_system_data.get("createdBy"),
            "jobType": job_properties.get("jobType"),
            "isArchived": job_properties.get("isArchived"),
            "startTimeUtc": job_runtime_properties.get("StartTimeUtc"),
            "endTimeUtc": job_runtime_properties.get("EndTimeUtc"),
            "instanceCount": job_resources.get("instanceCount"),
            "instanceType": job_resources.get("instanceType"),
            "shmSize": job_resources.get("shmSize"),
            "dockerArgs": job_resources.get("dockerArgs"),
            "environmentImageReference": job_properties.get(
                "environmentImageReference"
            ),
            # Legacy field retained for jobs created before the rename. New
            # Foundry jobs should only populate environmentImageReference.
            "environmentId": job_properties.get("environmentId"),
            "environmentReference": _environment_reference(job_properties),
            "computeId": job_properties.get("computeId"),
            "codeId": job_properties.get("codeId"),
            "distributionType": job_distribution.get("distributionType"),
            "processCountPerInstance": job_distribution.get("processCountPerInstance"),
            "timeout": job_limits.get("timeout"),
            "jobTier": job_queue_settings.get("jobTier"),
            "inputNames": list(job_inputs) or None,
            "outputNames": list(job_outputs) or None,
            "submittedAtUtc": job_tags.get("submittedAtUtc"),
            "runAttemptCount": job_tags.get("run_attempt_count"),
            "errorCode": job_error.get("code"),
            "errorMessage": error_message,
            "defaultOutputUri": default_output.get("uri"),
            "jobLogsUrl": _as_dict(job_services.get("JobLogs")).get("endpoint"),
            "trackingUrl": _as_dict(job_services.get("Tracking")).get("endpoint"),
        }
    )


def _create_job_name_suffix() -> str:
    return "".join(secrets_choice(JOB_NAME_SUFFIX_ALPHABET) for _ in range(4))


def _create_submitted_at_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def job_list_target_tag(job_name: str) -> str:
    digest = sha256(str(job_name).encode("utf-8")).hexdigest()[:16]
    return f"{JOB_LIST_TARGET_TAG_PREFIX}_{digest}"


def _build_job_url(
    *,
    project_endpoint: str,
    project_name: str,
    api_version: str,
    job_name: str,
) -> str:
    normalized_endpoint = project_endpoint.rstrip("/")
    if normalized_endpoint.endswith("/api/projects"):
        return f"{normalized_endpoint}/{project_name}/jobs/{job_name}?api-version={api_version}"
    if "/api/projects/" in normalized_endpoint:
        return f"{normalized_endpoint}/jobs/{job_name}?api-version={api_version}"

    return (
        f"{normalized_endpoint}/api/projects/{project_name}"
        f"/jobs/{job_name}?api-version={api_version}"
    )


def _build_job_action_url(
    *,
    project_endpoint: str,
    project_name: str,
    api_version: str,
    job_name: str,
    action: str,
) -> str:
    normalized_endpoint = project_endpoint.rstrip("/")
    if normalized_endpoint.endswith("/api/projects"):
        return (
            f"{normalized_endpoint}/{project_name}/jobs/{job_name}/{action}"
            f"?api-version={api_version}"
        )
    if "/api/projects/" in normalized_endpoint:
        return (
            f"{normalized_endpoint}/jobs/{job_name}/{action}?api-version={api_version}"
        )

    return (
        f"{normalized_endpoint}/api/projects/{project_name}"
        f"/jobs/{job_name}/{action}?api-version={api_version}"
    )


def _sanitize_operation_location(operation: str) -> str:
    cleaned = operation.strip().strip("'\"").strip()
    for suffix in ("%27", "%22"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned


def _extract_operation_id_from_location(operation: str) -> str | None:
    cleaned = _sanitize_operation_location(operation)
    if "/" not in cleaned and "?" not in cleaned:
        return unquote(cleaned)

    parsed = urlsplit(cleaned)
    path_parts = [part for part in parsed.path.split("/") if part]
    operation_markers = {
        "mfeOperationResults",
        "foundryMfeOperationResults",
        "operations",
    }
    for index, path_part in enumerate(path_parts):
        if unquote(path_part) not in operation_markers:
            continue
        operation_index = index + 1
        if operation_index < len(path_parts):
            return unquote(path_parts[operation_index])

    return None


def extract_job_operation_id(operation: str) -> str | None:
    """Extract a Foundry job operation id from a Location header, URL, or id."""
    return _extract_operation_id_from_location(operation)


def _build_jobs_base_url(*, project_endpoint: str, project_name: str) -> str:
    normalized_endpoint = project_endpoint.rstrip("/")
    if normalized_endpoint.endswith("/api/projects"):
        return f"{normalized_endpoint}/{project_name}/jobs"
    if "/api/projects/" in normalized_endpoint:
        return f"{normalized_endpoint}/jobs"
    return f"{normalized_endpoint}/api/projects/{project_name}/jobs"


def _build_job_operation_poll_url(
    operation: str,
    *,
    project_endpoint: str,
    project_name: str,
    api_version: str,
    endpoint: str,
) -> str:
    if endpoint not in {"result", "status"}:
        raise ValueError(f"Unsupported job operation polling endpoint '{endpoint}'.")

    operation_id = extract_job_operation_id(operation)
    if not operation_id:
        raise ValueError(
            "Could not extract a job operation id from the operation Location header."
        )

    base_url = _build_jobs_base_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
    )
    encoded_operation_id = quote(operation_id, safe="")
    return (
        f"{base_url}/operations/{encoded_operation_id}/{endpoint}"
        f"?api-version={api_version}"
    )


def build_job_operation_result_url(
    operation: str,
    *,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
) -> str:
    """Build the project-scoped job operation result polling URL."""
    return _build_job_operation_poll_url(
        operation,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        endpoint="result",
    )


def build_job_operation_status_url(
    operation: str,
    *,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
) -> str:
    """Build the project-scoped job operation status polling URL."""
    return _build_job_operation_poll_url(
        operation,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        endpoint="status",
    )


def build_job_url(
    job_name: str,
    *,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
) -> str:
    """Build the Foundry job URL for a known job name."""
    return _build_job_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        job_name=job_name,
    )


def _normalize_request_body(request_body: Mapping[str, Any]) -> dict[str, Any]:
    normalized_request_body = deepcopy(dict(request_body))
    properties = normalized_request_body.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("request_body must contain a top-level 'properties' mapping.")

    normalized_request_body["properties"] = dict(properties)
    return normalized_request_body


def _format_placeholder_hits(hits: list[tuple[str, str]]) -> str:
    formatted_hits = []
    for path, value in hits[:5]:
        display_value = value if len(value) <= 120 else f"{value[:117]}..."
        formatted_hits.append(f"{path}={display_value!r}")
    if len(hits) > 5:
        formatted_hits.append(f"... and {len(hits) - 5} more")
    return ", ".join(formatted_hits)


_DIRECT_RESOURCE_PROPERTY_KEYS: tuple[str, ...] = (
    "computeId",
    "environmentImageReference",
    "environmentId",
    "userAssignedIdentityId",
    "codeId",
)
_NESTED_RESOURCE_PROPERTY_KEYS: tuple[str, ...] = (
    "uri",
    "path",
    "assetId",
    "assetName",
    "assetVersion",
    "connectionName",
    "connection_name",
)


def _append_placeholder_hit(
    hits: list[tuple[str, str]],
    *,
    path: str,
    value: Any,
) -> None:
    if isinstance(value, str) and job_constants.contains_placeholder_token(value):
        hits.append((path, value))


def _collect_nested_resource_placeholders(
    hits: list[tuple[str, str]],
    *,
    container: Any,
    path: str,
) -> None:
    if not isinstance(container, Mapping):
        return

    for entry_name, entry in container.items():
        if not isinstance(entry, Mapping):
            continue
        entry_path = f"{path}[{entry_name!r}]"
        for key in _NESTED_RESOURCE_PROPERTY_KEYS:
            if key in entry:
                _append_placeholder_hit(
                    hits,
                    path=f"{entry_path}.{key}",
                    value=entry[key],
                )


def _validate_request_body_has_no_placeholders(request_body: Mapping[str, Any]) -> None:
    properties = request_body.get("properties")
    if not isinstance(properties, Mapping):
        return

    hits: list[tuple[str, str]] = []
    for key in _DIRECT_RESOURCE_PROPERTY_KEYS:
        if key in properties:
            _append_placeholder_hit(
                hits,
                path=f"$.properties.{key}",
                value=properties[key],
            )

    for container_key in ("inputs", "outputs"):
        _collect_nested_resource_placeholders(
            hits,
            container=properties.get(container_key),
            path=f"$.properties.{container_key}",
        )

    if not hits:
        return

    details = _format_placeholder_hits(hits)
    raise ValueError(
        "request_body contains placeholder token(s) in Foundry resource/config "
        f"fields: {details}. Replace placeholders with concrete resource values "
        "or set the corresponding FOUNDRY_TRAININGJOB__* environment variables."
    )


def _add_submitted_at_tag(
    request_body: Mapping[str, Any],
    *,
    submitted_at_utc: str,
    job_name: str,
) -> dict[str, Any]:
    normalized_request_body = _normalize_request_body(request_body)
    tags = _as_dict(normalized_request_body["properties"].get("tags"))
    tags.setdefault("submittedAtUtc", submitted_at_utc)
    tags.setdefault(job_list_target_tag(job_name), "true")
    normalized_request_body["properties"]["tags"] = tags
    return normalized_request_body


@dataclass(frozen=True)
class JobRun:
    job_name: str
    job_url: str
    request_id: str
    submitted_at_utc: str
    response: FoundryRestResponse
    created_job: dict[str, Any]
    request_body: dict[str, Any]

    @property
    def job_id(self) -> str | None:
        return self.created_job.get("id")

    @property
    def display_name(self) -> str | None:
        properties = _as_dict(self.request_body.get("properties"))
        return properties.get("displayName")

    def creation_summary(self) -> dict[str, Any]:
        return _compact_dict(
            {
                "jobId": self.job_id,
                "requestId": self.request_id,
                "apimRequestId": self.response.apim_request_id,
            }
        )

    def get_status(self, *, access_token: str | None = None) -> JobStatusResult:
        return get_job_status(
            self.job_url,
            access_token=access_token,
            fallback_job_name=self.job_name,
        )

    def status_summary(self, *, access_token: str | None = None) -> dict[str, Any]:
        return self.get_status(access_token=access_token).summary()


@dataclass(frozen=True)
class JobStatusResult:
    response: FoundryRestResponse
    job: dict[str, Any]
    fallback_job_name: str | None = None

    def summary(self) -> dict[str, Any]:
        return _build_job_status_summary(
            self.response,
            self.job,
            fallback_job_name=self.fallback_job_name,
        )

    def body(self) -> dict[str, Any] | str:
        return self.job if self.job else self.response.text


@dataclass(frozen=True)
class JobListResult:
    response: FoundryRestResponse
    jobs: list[dict[str, Any]]
    next_link: str | None = None
    pages: int = 1

    def summary(self) -> dict[str, Any]:
        return _compact_dict(
            {
                "statusCode": self.response.status_code,
                "apimRequestId": self.response.apim_request_id,
                "jobCount": len(self.jobs),
                "nextLink": self.next_link,
                "pages": self.pages,
            }
        )


@dataclass(frozen=True)
class JobMutationResult:
    job_name: str
    action: str
    response: FoundryRestResponse
    project_endpoint: str | None = None
    project_name: str | None = None
    api_version: str | None = None

    @property
    def location(self) -> str | None:
        raw_location = self.response.header("Location")
        if raw_location is None:
            return None
        return _sanitize_operation_location(raw_location)

    @property
    def operation_id(self) -> str | None:
        if self.location is None:
            return None
        return extract_job_operation_id(self.location)

    @property
    def operation_result_url(self) -> str | None:
        if (
            self.location is None
            or self.project_endpoint is None
            or self.project_name is None
            or self.api_version is None
        ):
            return None
        try:
            return build_job_operation_result_url(
                self.location,
                project_endpoint=self.project_endpoint,
                project_name=self.project_name,
                api_version=self.api_version,
            )
        except ValueError:
            return None

    def summary(self) -> dict[str, Any]:
        return _compact_dict(
            {
                "jobName": self.job_name,
                "action": self.action,
                "statusCode": self.response.status_code,
                "apimRequestId": self.response.apim_request_id,
                "location": self.location,
                "operationId": self.operation_id,
                "operationResultUrl": self.operation_result_url,
                "retryAfter": self.response.header("Retry-After"),
            }
        )


@dataclass(frozen=True)
class JobOperationPollResult:
    operation_id: str
    operation_url: str
    endpoint: str
    request_id: str
    response: FoundryRestResponse

    @property
    def body(self) -> dict[str, Any]:
        return _parse_response_json(self.response)

    @property
    def operation_status(self) -> str | None:
        body = self.body
        status = body.get("status") or body.get("Status")
        return str(status) if status is not None else None

    def summary(self) -> dict[str, Any]:
        body = self.body
        error = _as_dict(body.get("error"))
        return _compact_dict(
            {
                "operationId": self.operation_id,
                "operationEndpoint": self.endpoint,
                "operationUrl": self.operation_url,
                "statusCode": self.response.status_code,
                "requestId": self.request_id,
                "apimRequestId": self.response.apim_request_id,
                "retryAfter": self.response.header("Retry-After"),
                "operationStatus": self.operation_status,
                "errorCode": error.get("code") or body.get("code"),
                "errorMessage": error.get("message") or body.get("message"),
            }
        )


def _print_job_status_result(
    job_status: JobStatusResult,
    *,
    print_full_response: bool = False,
) -> None:
    print("Status request completed.")
    print(f"APIM request id: {job_status.response.apim_request_id or 'Unavailable'}")
    if print_full_response:
        print("Full job response:")
        job_response = job_status.body()
        if isinstance(job_response, str):
            print(job_response)
        else:
            print(json.dumps(job_response, indent=2))
        return

    status_summary = job_status.summary()
    print(f"Current status: {status_summary.get('status', 'Unknown')}")
    print("Status summary:")
    print(json.dumps(status_summary, indent=2))


def print_job_run_status(
    job_run: JobRun,
    *,
    access_token: str | None = None,
    print_full_response: bool = False,
) -> JobStatusResult:
    print(f"Fetching current status for: {job_run.job_name}")
    job_status = job_run.get_status(access_token=access_token)
    _print_job_status_result(job_status, print_full_response=print_full_response)
    return job_status


def _build_jobs_collection_url(
    *,
    project_endpoint: str,
    project_name: str,
    api_version: str,
    job_type: str | None = None,
    list_view_type: str | None = None,
    top: int | None = None,
    tag: str | None = None,
    properties: str | None = None,
    skip_token: str | None = None,
) -> str:
    normalized_endpoint = project_endpoint.rstrip("/")
    if normalized_endpoint.endswith("/api/projects"):
        base_url = f"{normalized_endpoint}/{project_name}/jobs"
    elif "/api/projects/" in normalized_endpoint:
        base_url = f"{normalized_endpoint}/jobs"
    else:
        base_url = f"{normalized_endpoint}/api/projects/{project_name}/jobs"

    query_params: dict[str, str | int] = {"api-version": api_version}
    if job_type is not None:
        query_params["jobType"] = job_type
    if list_view_type is not None:
        query_params["listViewType"] = list_view_type
    if top is not None:
        query_params["$top"] = top
    if tag is not None:
        query_params["tag"] = tag
    if properties is not None:
        query_params["properties"] = properties
    if skip_token is not None:
        query_params["skipToken"] = skip_token

    return f"{base_url}?{urlencode(query_params)}"


def _url_origin(url: str) -> tuple[str, str]:
    parsed = urlsplit(url)
    return parsed.scheme.lower(), parsed.netloc.lower()


def _validate_jobs_collection_next_link(
    next_link: str, *, expected_origin: tuple[str, str]
) -> str:
    parsed = urlsplit(next_link)
    if not parsed.scheme and not parsed.netloc:
        resolved = urljoin(f"{expected_origin[0]}://{expected_origin[1]}", next_link)
        parsed = urlsplit(resolved)
        next_link = resolved

    if (parsed.scheme.lower(), parsed.netloc.lower()) != expected_origin:
        raise ValueError(
            f"Refusing to follow cross-origin jobs nextLink '{next_link}'."
        )

    return next_link


def print_job_status(
    job_name: str,
    *,
    access_token: str | None = None,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
    print_full_response: bool = False,
) -> JobStatusResult:
    """Fetch and print the current status for an existing job by name."""
    print(f"Fetching current status for: {job_name}")
    job_status = get_job_status_by_name(
        job_name,
        access_token=access_token,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
    )
    _print_job_status_result(job_status, print_full_response=print_full_response)
    return job_status


def submit_job(
    request_body: Mapping[str, Any],
    *,
    access_token: str | None = None,
    job_name: str | None = None,
    job_name_prefix: str = "job",
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
    add_submitted_at_tag: bool = True,
) -> JobRun:
    submitted_at_utc = _create_submitted_at_utc()
    effective_job_name = job_name or f"{job_name_prefix}-{_create_job_name_suffix()}"
    job_url = build_job_url(
        effective_job_name,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
    )
    request_id = create_request_id()
    final_request_body = (
        _add_submitted_at_tag(
            request_body,
            submitted_at_utc=submitted_at_utc,
            job_name=effective_job_name,
        )
        if add_submitted_at_tag
        else _normalize_request_body(request_body)
    )
    _validate_request_body_has_no_placeholders(final_request_body)

    response = put_foundry_json(
        job_url,
        final_request_body,
        access_token=access_token,
        request_id=request_id,
        disable_retry=True,
    )

    return JobRun(
        job_name=effective_job_name,
        job_url=job_url,
        request_id=request_id,
        submitted_at_utc=submitted_at_utc,
        response=response,
        created_job=_parse_response_json(response),
        request_body=final_request_body,
    )


def list_jobs(
    *,
    access_token: str | None = None,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
    job_type: str | None = None,
    list_view_type: str | None = None,
    top: int | None = None,
    tag: str | None = None,
    properties: str | None = None,
    skip_token: str | None = None,
    next_link: str | None = None,
    max_pages: int = 1,
) -> JobListResult:
    pages = 0
    collected_jobs: list[dict[str, Any]] = []
    first_page_url = _build_jobs_collection_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        job_type=job_type,
        list_view_type=list_view_type,
        top=top,
        tag=tag,
        properties=properties,
        skip_token=skip_token,
    )
    expected_origin = _url_origin(first_page_url)
    current_url = (
        _validate_jobs_collection_next_link(next_link, expected_origin=expected_origin)
        if next_link
        else first_page_url
    )
    last_response: FoundryRestResponse | None = None
    next_page_link: str | None = None

    while current_url and pages < max_pages:
        last_response = get_foundry_json(
            current_url,
            access_token=access_token,
            raise_on_http_error=False,
        )
        pages += 1

        body = _parse_response_json(last_response)
        if last_response.status_code != 200:
            next_page_link = None
            break

        collected_jobs.extend(body.get("value", []))
        next_page_link = body.get("nextLink")
        current_url = (
            _validate_jobs_collection_next_link(
                next_page_link,
                expected_origin=expected_origin,
            )
            if next_page_link
            else None
        )

    return JobListResult(
        response=last_response
        if last_response is not None
        else FoundryRestResponse(status_code=0, headers={}, text=""),
        jobs=collected_jobs,
        next_link=next_page_link,
        pages=pages or 1,
    )


def get_job_status_summary(
    job_url: str,
    *,
    access_token: str | None = None,
    fallback_job_name: str | None = None,
) -> dict[str, Any]:
    return get_job_status(
        job_url,
        access_token=access_token,
        fallback_job_name=fallback_job_name,
    ).summary()


def _get_job_operation_poll(
    operation: str,
    *,
    endpoint: str,
    access_token: str | None = None,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> JobOperationPollResult:
    operation_id = extract_job_operation_id(operation)
    if not operation_id:
        raise ValueError(
            "Could not extract a job operation id from the operation Location header."
        )
    operation_url = _build_job_operation_poll_url(
        operation_id,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        endpoint=endpoint,
    )
    request_id = create_request_id()
    response = get_foundry_json(
        operation_url,
        access_token=access_token,
        request_id=request_id,
        timeout_seconds=timeout_seconds,
        raise_on_http_error=False,
    )
    return JobOperationPollResult(
        operation_id=operation_id,
        operation_url=operation_url,
        endpoint=endpoint,
        request_id=request_id,
        response=response,
    )


def get_job_operation_result(
    operation: str,
    *,
    access_token: str | None = None,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> JobOperationPollResult:
    """Poll a job mutation operation's project-scoped result route."""
    return _get_job_operation_poll(
        operation,
        endpoint="result",
        access_token=access_token,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        timeout_seconds=timeout_seconds,
    )


def get_job_operation_status(
    operation: str,
    *,
    access_token: str | None = None,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> JobOperationPollResult:
    """Poll a job mutation operation's project-scoped status route."""
    return _get_job_operation_poll(
        operation,
        endpoint="status",
        access_token=access_token,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        timeout_seconds=timeout_seconds,
    )


def cancel_job(
    job_name: str,
    *,
    access_token: str | None = None,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
) -> JobMutationResult:
    action_url = _build_job_action_url(
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
        job_name=job_name,
        action="cancel",
    )
    response = post_foundry_json(
        action_url,
        access_token=access_token,
        raise_on_http_error=False,
        retry_transient=True,
    )
    return JobMutationResult(
        job_name=job_name,
        action="cancel",
        response=response,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
    )


def delete_job(
    job_name: str,
    *,
    access_token: str | None = None,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
) -> JobMutationResult:
    response = delete_foundry(
        build_job_url(
            job_name,
            project_endpoint=project_endpoint,
            project_name=project_name,
            api_version=api_version,
        ),
        access_token=access_token,
        raise_on_http_error=False,
    )
    return JobMutationResult(
        job_name=job_name,
        action="delete",
        response=response,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
    )


def get_job_status_summary_by_name(
    job_name: str,
    *,
    access_token: str | None = None,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
) -> dict[str, Any]:
    """Return a compact job status summary for an existing job by name."""
    return get_job_status_by_name(
        job_name,
        access_token=access_token,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
    ).summary()


def get_job_status(
    job_url: str,
    *,
    access_token: str | None = None,
    fallback_job_name: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> JobStatusResult:
    job_status_response = get_foundry_json(
        job_url,
        access_token=access_token,
        timeout_seconds=timeout_seconds,
        raise_on_http_error=False,
    )

    return JobStatusResult(
        response=job_status_response,
        job=_parse_response_json(job_status_response),
        fallback_job_name=fallback_job_name,
    )


def get_job_status_by_name(
    job_name: str,
    *,
    access_token: str | None = None,
    project_endpoint: str = DEFAULT_PROJECT_ENDPOINT,
    project_name: str = DEFAULT_PROJECT_NAME,
    api_version: str = DEFAULT_API_VERSION,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> JobStatusResult:
    """Fetch the current status for an existing job by name."""
    job_url = build_job_url(
        job_name,
        project_endpoint=project_endpoint,
        project_name=project_name,
        api_version=api_version,
    )
    return get_job_status(
        job_url,
        access_token=access_token,
        fallback_job_name=job_name,
        timeout_seconds=timeout_seconds,
    )
