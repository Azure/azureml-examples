"""Foundry model asset upload and registration helper."""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

from azure.core.credentials import TokenCredential

from . import constants as foundry_constants
from .auth import get_foundry_access_token
from .rest import _request_foundry, create_request_id


_MAX_PARALLEL_UPLOADS = 8
_OPERATION_POLL_INTERVAL_SECONDS = 2.0
_OPERATION_TIMEOUT_SECONDS = 30 * 60
_SUCCEEDED_OPERATION_STATUSES = frozenset({"succeeded", "completed"})
_FAILED_OPERATION_STATUSES = frozenset({"failed", "canceled", "cancelled"})


@dataclass(frozen=True)
class ModelAssetUploadResult:
    asset_id: str
    name: str
    version: str
    provisioning_status: str
    blob_uri: str
    project_endpoint: str
    project_name: str | None
    uploaded_files: tuple[str, ...] = field(default_factory=tuple)
    total_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["uploaded_files"] = list(result["uploaded_files"])
        return result


@dataclass(frozen=True)
class ModelAssetDownloadResult:
    asset_id: str
    name: str
    version: str
    blob_uri: str
    target_dir: str
    downloaded_files: tuple[str, ...] = field(default_factory=tuple)
    total_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["downloaded_files"] = list(self.downloaded_files)
        return result


def _container_client_from_sas(sas_uri: str):
    from azure.storage.blob import ContainerClient  # type: ignore[import-not-found]

    return ContainerClient.from_container_url(container_url=sas_uri)


def _blob_path(blob_uri: str) -> str:
    path = urlsplit(blob_uri).path.lstrip("/")
    if "/" not in path:
        return ""
    return unquote(path.split("/", 1)[1]).rstrip("/")


def _safe_download_path(target_dir: Path, relative_path: str) -> Path:
    candidate = (target_dir / relative_path).resolve()
    try:
        candidate.relative_to(target_dir)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to download model blob {relative_path!r}: path escapes "
            f"target_dir {target_dir}."
        ) from exc
    return candidate


def _normalize_non_empty_string(value: str, *, name: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string.")
    return normalized


def _build_project_endpoint(project_endpoint: str, project_name: str | None) -> str:
    endpoint = _normalize_non_empty_string(
        project_endpoint, name="project_endpoint"
    ).rstrip("/")
    if "/api/projects/" in endpoint:
        return endpoint
    name = _normalize_non_empty_string(project_name or "", name="project_name")
    if endpoint.endswith("/api/projects"):
        return f"{endpoint}/{name}"
    return f"{endpoint}/api/projects/{name}"


def _redact_credentials(body: Any) -> Any:
    if isinstance(body, dict):
        result = {}
        for key, value in body.items():
            if key in ("credential", "credentials", "sasUri", "sas_uri"):
                result[key] = "<redacted>"
            else:
                result[key] = _redact_credentials(value)
        return result
    if isinstance(body, list):
        return [_redact_credentials(item) for item in body]
    return body


def _upload_file(
    container_client: Any, source_path: Path, blob_name: str
) -> tuple[str, int]:
    size = int(source_path.stat().st_size)
    with source_path.open("rb") as file_handle:
        container_client.upload_blob(name=blob_name, data=file_handle, overwrite=True)
    return blob_name, size


def _upload_local_path(
    container_client: Any,
    source_path: Path,
    *,
    blob_prefix: str | None = None,
) -> tuple[list[str], int]:
    prefix = str(blob_prefix or "").strip("/")
    if source_path.is_file():
        blob_name = f"{prefix}/{source_path.name}" if prefix else source_path.name
        blob_name, size = _upload_file(container_client, source_path, blob_name)
        return [blob_name], size

    files = sorted(entry for entry in source_path.rglob("*") if entry.is_file())
    if not files:
        raise ValueError(f"local_path contains no files: {source_path}")

    uploaded_files: list[str] = []
    total_bytes = 0
    worker_count = min(_MAX_PARALLEL_UPLOADS, len(files))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _upload_file,
                container_client,
                entry,
                (
                    f"{prefix}/{entry.relative_to(source_path).as_posix()}"
                    if prefix
                    else entry.relative_to(source_path).as_posix()
                ),
            ): entry
            for entry in files
        }
        for future in concurrent.futures.as_completed(futures):
            blob_name, size = future.result()
            uploaded_files.append(blob_name)
            total_bytes += size

    return sorted(uploaded_files), total_bytes


def _operation_location(response: Any, request_url: str) -> str:
    for header_name in ("Operation-Location", "Location", "Azure-AsyncOperation"):
        value = response.header(header_name)
        if value:
            return urljoin(request_url, value)
    raise RuntimeError(
        "createAsync response did not include an operation Location header."
    )


def _operation_status(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    for key in ("status", "provisioningState", "state"):
        value = body.get(key)
        if value:
            return str(value)
    properties = body.get("properties")
    if isinstance(properties, dict):
        for key in ("status", "provisioningState", "state"):
            value = properties.get(key)
            if value:
                return str(value)
    return None


def _wait_for_operation(
    operation_url: str,
    *,
    access_token: str,
) -> str:
    deadline = time.monotonic() + _OPERATION_TIMEOUT_SECONDS
    last_status: str | None = None
    while time.monotonic() < deadline:
        response = _request_foundry(
            "GET",
            operation_url,
            access_token=access_token,
            request_id=create_request_id(),
            raise_on_http_error=False,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Model createAsync operation failed to poll: "
                f"{response.status_code} - {response.text[:500]}"
            )
        body = response.json()
        last_status = _operation_status(body)
        normalized_status = (last_status or "").lower()
        if normalized_status in _SUCCEEDED_OPERATION_STATUSES:
            return str(last_status)
        if normalized_status in _FAILED_OPERATION_STATUSES:
            raise RuntimeError(
                f"Model createAsync operation reached terminal state "
                f"{last_status}: {_redact_credentials(body)!r}"
            )
        time.sleep(_OPERATION_POLL_INTERVAL_SECONDS)

    detail = f" Last status: {last_status}." if last_status else ""
    raise TimeoutError(f"Timed out waiting for model createAsync operation.{detail}")


def _existing_model_result(
    body: Any,
    *,
    name: str,
    version: str,
    project_endpoint: str,
    project_name: str | None,
    provisioning_status: str | None = None,
) -> ModelAssetUploadResult | None:
    if not isinstance(body, dict) or not body.get("id"):
        return None
    return ModelAssetUploadResult(
        asset_id=str(body["id"]),
        name=str(body.get("name") or name),
        version=str(body.get("version") or version),
        provisioning_status=(
            provisioning_status or _operation_status(body) or "Succeeded"
        ),
        blob_uri=str(body.get("blobUri") or ""),
        project_endpoint=project_endpoint,
        project_name=project_name,
    )


def _wait_for_model_resource(
    model_url: str,
    *,
    access_token: str,
    name: str,
    version: str,
    project_endpoint: str,
    project_name: str | None,
) -> ModelAssetUploadResult:
    deadline = time.monotonic() + _OPERATION_TIMEOUT_SECONDS
    last_status_code: int | None = None
    while time.monotonic() < deadline:
        response = _request_foundry(
            "GET",
            model_url,
            access_token=access_token,
            request_id=create_request_id(),
            raise_on_http_error=False,
        )
        last_status_code = response.status_code
        if response.status_code == 200:
            body = response.json()
            status = _operation_status(body)
            if (status or "").lower() in _FAILED_OPERATION_STATUSES:
                raise RuntimeError(
                    f"Model {name}:{version} reached terminal state {status}: "
                    f"{_redact_credentials(body)!r}"
                )
            result = _existing_model_result(
                body,
                name=name,
                version=version,
                project_endpoint=project_endpoint,
                project_name=project_name,
                provisioning_status=status,
            )
            if result is not None:
                return result
        elif response.status_code != 404:
            raise RuntimeError(
                f"GET model {name}:{version} while polling failed: "
                f"{response.status_code} - {response.text[:500]}"
            )
        time.sleep(_OPERATION_POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Timed out waiting for model {name}:{version}; last HTTP status "
        f"{last_status_code}."
    )


def upload_and_register_model(
    local_path: str | Path,
    *,
    name: str,
    version: str = "1",
    project_endpoint: str = foundry_constants.DEFAULT_PROJECT_ENDPOINT,
    project_name: str | None = foundry_constants.DEFAULT_PROJECT_NAME,
    api_version: str = foundry_constants.DEFAULT_API_VERSION,
    description: str | None = None,
    tags: dict[str, str] | None = None,
    blob_prefix: str | None = None,
    credential: TokenCredential | None = None,
) -> ModelAssetUploadResult:
    """Upload a local file/folder and register it as a Foundry model asset version."""
    resolved_name = _normalize_non_empty_string(name, name="name")
    resolved_version = _normalize_non_empty_string(version, name="version")
    resolved_endpoint = _build_project_endpoint(project_endpoint, project_name)
    src_path = Path(local_path).expanduser().resolve()
    if not src_path.exists():
        raise ValueError(f"local_path does not exist: {src_path}")

    token = get_foundry_access_token(credential=credential).token
    model_url = (
        f"{resolved_endpoint.rstrip('/')}/models/{resolved_name}/versions/"
        f"{resolved_version}?api-version={api_version}"
    )
    existing_response = _request_foundry(
        "GET",
        model_url,
        access_token=token,
        request_id=create_request_id(),
        raise_on_http_error=False,
    )
    if existing_response.status_code == 200:
        existing_result = _existing_model_result(
            existing_response.json(),
            name=resolved_name,
            version=resolved_version,
            project_endpoint=resolved_endpoint,
            project_name=project_name,
        )
        if existing_result is not None:
            return existing_result
    elif existing_response.status_code != 404:
        raise RuntimeError(
            f"GET model {resolved_name}:{resolved_version} before upload failed: "
            f"{existing_response.status_code} - {existing_response.text[:500]}"
        )

    start_url = (
        f"{resolved_endpoint.rstrip('/')}/models/{resolved_name}/versions/"
        f"{resolved_version}/startPendingUpload?api-version={api_version}"
    )
    start_resp = _request_foundry(
        "POST",
        start_url,
        payload={},
        access_token=token,
        request_id=create_request_id(),
    )
    if start_resp.status_code >= 400:
        raise RuntimeError(
            f"startPendingUpload for model {resolved_name}:{resolved_version} "
            f"failed: {start_resp.status_code} - {start_resp.text[:500]}"
        )
    start_body = start_resp.json()
    blob_ref = start_body.get("blobReferenceForConsumption") or {}
    blob_uri = blob_ref.get("blobUri")
    cred_block = blob_ref.get("credential") or {}
    sas_uri = cred_block.get("sasUri") or cred_block.get("sas_uri")
    if not blob_uri or not sas_uri:
        raise RuntimeError(
            f"startPendingUpload response for model {resolved_name}:"
            f"{resolved_version} did not include blobUri/sasUri: "
            f"{_redact_credentials(start_body)!r}"
        )

    container_client = _container_client_from_sas(str(sas_uri))
    try:
        uploaded_files, total_bytes = _upload_local_path(
            container_client,
            src_path,
            blob_prefix=blob_prefix,
        )
    finally:
        close = getattr(container_client, "close", None)
        if callable(close):
            close()

    register_token = get_foundry_access_token(credential=credential).token
    register_url = model_url.replace(
        f"?api-version={api_version}",
        f"/createAsync?api-version={api_version}",
    )
    register_payload: dict[str, Any] = {"blobUri": blob_uri}
    if description is not None:
        register_payload["description"] = description
    if tags:
        register_payload["tags"] = dict(tags)
    register_resp = _request_foundry(
        "POST",
        register_url,
        payload=register_payload,
        access_token=register_token,
        request_id=create_request_id(),
        disable_retry=True,
    )
    if register_resp.status_code >= 400:
        raise RuntimeError(
            f"POST createAsync for model {resolved_name}:{resolved_version} failed: "
            f"{register_resp.status_code} - {register_resp.text[:500]}"
        )
    operation_url = _operation_location(register_resp, register_url)
    if urlsplit(operation_url).netloc.lower() == urlsplit(model_url).netloc.lower():
        provisioning_status = _wait_for_operation(
            operation_url,
            access_token=register_token,
        )
    else:
        resource_result = _wait_for_model_resource(
            model_url,
            access_token=register_token,
            name=resolved_name,
            version=resolved_version,
            project_endpoint=resolved_endpoint,
            project_name=project_name,
        )
        return ModelAssetUploadResult(
            **{
                **resource_result.__dict__,
                "uploaded_files": tuple(uploaded_files),
                "total_bytes": total_bytes,
            }
        )

    final_token = get_foundry_access_token(credential=credential).token
    final_resp = _request_foundry(
        "GET",
        model_url,
        access_token=final_token,
        request_id=create_request_id(),
        raise_on_http_error=False,
    )
    if final_resp.status_code != 200:
        raise RuntimeError(
            f"GET model {resolved_name}:{resolved_version} after createAsync failed: "
            f"{final_resp.status_code} - {final_resp.text[:500]}"
        )
    final_body = final_resp.json()
    asset_id = final_body.get("id") if isinstance(final_body, dict) else None
    if not asset_id:
        raise RuntimeError(
            f"GET model {resolved_name}:{resolved_version} returned no asset id: "
            f"{final_body!r}"
        )

    return ModelAssetUploadResult(
        asset_id=str(asset_id),
        name=resolved_name,
        version=resolved_version,
        provisioning_status=provisioning_status,
        blob_uri=str(blob_uri),
        project_endpoint=resolved_endpoint,
        project_name=project_name,
        uploaded_files=tuple(uploaded_files),
        total_bytes=total_bytes,
    )


def download_model_asset(
    target_dir: str | Path,
    *,
    name: str,
    version: str,
    project_endpoint: str = foundry_constants.DEFAULT_PROJECT_ENDPOINT,
    project_name: str | None = foundry_constants.DEFAULT_PROJECT_NAME,
    api_version: str = foundry_constants.DEFAULT_API_VERSION,
    credential: TokenCredential | None = None,
    overwrite: bool = False,
) -> ModelAssetDownloadResult:
    """Download all blobs backing a Foundry Model V3 asset version."""

    resolved_name = _normalize_non_empty_string(name, name="name")
    resolved_version = _normalize_non_empty_string(version, name="version")
    resolved_endpoint = _build_project_endpoint(project_endpoint, project_name)
    destination = Path(target_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(destination.iterdir()):
        raise FileExistsError(
            f"target_dir {destination} is not empty; pass overwrite=True to "
            "replace downloaded model files."
        )

    token = get_foundry_access_token(credential=credential).token
    model_url = (
        f"{resolved_endpoint.rstrip('/')}/models/{resolved_name}/versions/"
        f"{resolved_version}?api-version={api_version}"
    )
    model_response = _request_foundry(
        "GET",
        model_url,
        access_token=token,
        request_id=create_request_id(),
        raise_on_http_error=False,
    )
    if model_response.status_code != 200:
        raise RuntimeError(
            f"GET model {resolved_name}:{resolved_version} failed: "
            f"{model_response.status_code} - {model_response.text[:500]}"
        )
    model = model_response.json()
    if not isinstance(model, dict):
        raise RuntimeError("GET model response was not a JSON object.")
    asset_id = model.get("id")
    blob_uri = model.get("blobUri")
    if not asset_id or not blob_uri:
        raise RuntimeError(
            f"GET model {resolved_name}:{resolved_version} returned no id/blobUri: "
            f"{_redact_credentials(model)!r}"
        )

    credential_url = model_url.replace(
        f"?api-version={api_version}",
        f"/credentials?api-version={api_version}",
    )
    credential_response = _request_foundry(
        "POST",
        credential_url,
        payload={"blobUri": blob_uri},
        access_token=token,
        request_id=create_request_id(),
        raise_on_http_error=False,
    )
    if credential_response.status_code != 200:
        raise RuntimeError(
            f"POST model credentials for {resolved_name}:{resolved_version} failed: "
            f"{credential_response.status_code} - {credential_response.text[:500]}"
        )
    credential_body = credential_response.json()
    if not isinstance(credential_body, dict):
        raise RuntimeError("Model credentials response was not a JSON object.")
    blob_reference = (
        credential_body.get("blobReferenceForConsumption")
        or credential_body.get("blobReference")
        or {}
    )
    credential_block = blob_reference.get("credential") or {}
    sas_uri = credential_block.get("sasUri") or credential_block.get("sas_uri")
    if not sas_uri:
        raise RuntimeError(
            "Model credentials response did not contain a SAS URI: "
            f"{_redact_credentials(credential_body)!r}"
        )

    prefix = _blob_path(str(blob_uri))
    container_client = _container_client_from_sas(str(sas_uri))
    downloaded_files: list[str] = []
    total_bytes = 0
    try:
        blob_names = container_client.list_blob_names(
            name_starts_with=f"{prefix}/" if prefix else None
        )
        for raw_blob_name in blob_names:
            blob_name = str(raw_blob_name)
            relative_path = blob_name
            if prefix and blob_name.startswith(f"{prefix}/"):
                relative_path = blob_name[len(prefix) + 1 :]
            if not relative_path:
                continue
            local_path = _safe_download_path(destination, relative_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            stream = container_client.get_blob_client(blob_name).download_blob()
            with local_path.open("wb") as file_handle:
                written = stream.readinto(file_handle)
            downloaded_files.append(str(local_path.relative_to(destination)))
            total_bytes += int(written or 0)
    finally:
        close = getattr(container_client, "close", None)
        if callable(close):
            close()
    if not downloaded_files:
        raise RuntimeError(
            f"Model {resolved_name}:{resolved_version} contained no downloadable blobs."
        )

    return ModelAssetDownloadResult(
        asset_id=str(asset_id),
        name=resolved_name,
        version=resolved_version,
        blob_uri=str(blob_uri),
        target_dir=str(destination),
        downloaded_files=tuple(sorted(downloaded_files)),
        total_bytes=total_bytes,
    )
