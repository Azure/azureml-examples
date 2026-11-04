from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    DatasetVersion,
    FileDatasetVersion,
    FolderDatasetVersion,
)
from azure.core.credentials import TokenCredential

from . import constants as foundry_constants
from .auth import get_foundry_credential

DEFAULT_DATASET_VERSION = "1"


def _normalize_non_empty_string(value: str, *, name: str) -> str:
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{name} must be a non-empty string.")
    return normalized_value


def _normalize_optional_string(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    return _normalize_non_empty_string(value, name=name)


def build_project_endpoint(
    project_endpoint: str, *, project_name: str | None = None
) -> str:
    """
    Build a project-scoped Foundry endpoint accepted by ``AIProjectClient``.

    ``project_endpoint`` can be either:
    - the base account endpoint, for example ``https://<account>.services.ai.azure.com``
    - the ``.../api/projects`` prefix
    - a fully project-scoped endpoint, for example
      ``https://<account>.services.ai.azure.com/api/projects/<project-name>``
    """
    normalized_endpoint = _normalize_non_empty_string(
        project_endpoint, name="project_endpoint"
    ).rstrip("/")
    normalized_project_name = _normalize_optional_string(
        project_name, name="project_name"
    )

    if "/api/projects/" in normalized_endpoint:
        return normalized_endpoint

    if normalized_endpoint.endswith("/api/projects"):
        if normalized_project_name is None:
            raise ValueError(
                "project_name is required when project_endpoint ends with '/api/projects'."
            )
        return f"{normalized_endpoint}/{normalized_project_name}"

    if normalized_project_name is None:
        raise ValueError(
            "project_name is required when project_endpoint is not already project-scoped."
        )

    return f"{normalized_endpoint}/api/projects/{normalized_project_name}"


def _default_dataset_name(local_path: Path) -> str:
    dataset_name = local_path.stem if local_path.is_file() else local_path.name
    return _normalize_non_empty_string(dataset_name, name="dataset_name")


def _resolve_connection_name(*, connection_name: str | None) -> str:
    normalized_connection_name = _normalize_optional_string(
        connection_name, name="connection_name"
    )
    if normalized_connection_name is not None:
        return foundry_constants.require_concrete_value(
            normalized_connection_name,
            name="connection_name",
            env_var="FOUNDRY_TRAININGJOB__STORAGE_CONNECTION_NAME",
        )

    return foundry_constants.require_concrete_value(
        foundry_constants.DEFAULT_STORAGE_CONNECTION_NAME,
        name="connection_name",
        env_var="FOUNDRY_TRAININGJOB__STORAGE_CONNECTION_NAME",
    )


@dataclass(frozen=True)
class DatasetUploadResult:
    dataset_id: str
    name: str
    version: str
    dataset_type: str | None
    data_uri: str | None
    connection_name: str | None
    local_path: str

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetReferenceResult:
    dataset_id: str
    name: str
    version: str
    dataset_type: str
    data_uri: str
    connection_name: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _build_upload_result(
    dataset: DatasetVersion,
    *,
    local_path: Path,
    connection_name: str,
) -> DatasetUploadResult:
    dataset_id = getattr(dataset, "id", None)
    if not dataset_id:
        raise ValueError(
            "The created dataset did not include an id in the service response."
        )

    return DatasetUploadResult(
        dataset_id=dataset_id,
        name=getattr(dataset, "name", None) or local_path.name,
        version=getattr(dataset, "version", None) or DEFAULT_DATASET_VERSION,
        dataset_type=getattr(dataset, "type", None),
        data_uri=getattr(dataset, "data_uri", None),
        connection_name=getattr(dataset, "connection_name", None) or connection_name,
        local_path=str(local_path),
    )


def upload_dataset(
    local_path: str | Path,
    *,
    dataset_name: str | None = None,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    project_endpoint: str = foundry_constants.DEFAULT_PROJECT_ENDPOINT,
    project_name: str | None = foundry_constants.DEFAULT_PROJECT_NAME,
    connection_name: str | None = foundry_constants.DEFAULT_STORAGE_CONNECTION_NAME,
    credential: TokenCredential | None = None,
    **upload_kwargs: Any,
) -> DatasetUploadResult:
    """
    Upload a local file or folder to Foundry as a dataset and return its service identifier.

    The function accepts either a file path or a directory path and chooses the matching SDK
    upload method automatically.
    """
    path = Path(local_path).expanduser()
    if not path.exists():
        raise ValueError(f"The provided local_path does not exist: {path}")

    resolved_path = path.resolve()
    resolved_dataset_name = _normalize_optional_string(
        dataset_name, name="dataset_name"
    ) or _default_dataset_name(resolved_path)
    resolved_dataset_version = _normalize_non_empty_string(
        dataset_version, name="dataset_version"
    )
    resolved_project_endpoint = build_project_endpoint(
        project_endpoint,
        project_name=project_name,
    )
    resolved_credential = credential or get_foundry_credential()
    resolved_connection_name = _resolve_connection_name(connection_name=connection_name)

    with AIProjectClient(
        endpoint=resolved_project_endpoint, credential=resolved_credential
    ) as project_client:
        if resolved_path.is_file():
            dataset = project_client.datasets.upload_file(
                name=resolved_dataset_name,
                version=resolved_dataset_version,
                file_path=str(resolved_path),
                connection_name=resolved_connection_name,
                **upload_kwargs,
            )
        else:
            dataset = project_client.datasets.upload_folder(
                name=resolved_dataset_name,
                version=resolved_dataset_version,
                folder=str(resolved_path),
                connection_name=resolved_connection_name,
                **upload_kwargs,
            )

    return _build_upload_result(
        dataset,
        local_path=resolved_path,
        connection_name=resolved_connection_name,
    )


def upload_dataset_id(*args: Any, **kwargs: Any) -> str:
    return upload_dataset(*args, **kwargs).dataset_id


def register_reference_dataset(
    data_uri: str,
    *,
    dataset_name: str,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    dataset_type: str,
    connection_name: str,
    project_endpoint: str = foundry_constants.DEFAULT_PROJECT_ENDPOINT,
    project_name: str | None = foundry_constants.DEFAULT_PROJECT_NAME,
    description: str | None = None,
    tags: dict[str, str] | None = None,
    credential: TokenCredential | None = None,
) -> DatasetReferenceResult:
    """Register an existing storage URI as a zero-copy Foundry dataset."""

    resolved_uri = _normalize_non_empty_string(data_uri, name="data_uri")
    resolved_name = _normalize_non_empty_string(dataset_name, name="dataset_name")
    resolved_version = _normalize_non_empty_string(
        dataset_version, name="dataset_version"
    )
    resolved_type = _normalize_non_empty_string(
        dataset_type, name="dataset_type"
    ).lower()
    if resolved_type not in {"uri_file", "uri_folder"}:
        raise ValueError(
            "dataset_type must be 'uri_file' or 'uri_folder' for a reference "
            f"dataset, got {dataset_type!r}."
        )
    resolved_connection = _normalize_non_empty_string(
        connection_name, name="connection_name"
    )
    resolved_project_endpoint = build_project_endpoint(
        project_endpoint,
        project_name=project_name,
    )
    resolved_credential = credential or get_foundry_credential()
    model_type = (
        FileDatasetVersion if resolved_type == "uri_file" else FolderDatasetVersion
    )
    dataset_version_model = model_type(
        name=resolved_name,
        version=resolved_version,
        data_uri=resolved_uri,
        is_reference=True,
        connection_name=resolved_connection,
        description=description,
        tags=dict(tags or {}),
    )

    with AIProjectClient(
        endpoint=resolved_project_endpoint,
        credential=resolved_credential,
    ) as project_client:
        dataset = project_client.datasets.create_or_update(
            name=resolved_name,
            version=resolved_version,
            dataset_version=dataset_version_model,
        )

    dataset_id = getattr(dataset, "id", None)
    if not dataset_id:
        raise ValueError(
            "The registered reference dataset did not include an id in the "
            "service response."
        )
    return DatasetReferenceResult(
        dataset_id=str(dataset_id),
        name=str(getattr(dataset, "name", None) or resolved_name),
        version=str(getattr(dataset, "version", None) or resolved_version),
        dataset_type=str(getattr(dataset, "type", None) or resolved_type),
        data_uri=str(getattr(dataset, "data_uri", None) or resolved_uri),
        connection_name=str(
            getattr(dataset, "connection_name", None) or resolved_connection
        ),
    )


@dataclass(frozen=True)
class DatasetDownloadResult:
    dataset_id: str | None
    name: str
    version: str
    dataset_type: str | None
    data_uri: str | None
    target_dir: str
    downloaded_files: tuple[str, ...] = field(default_factory=tuple)
    total_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["downloaded_files"] = list(result["downloaded_files"])
        return result


def _parse_blob_uri(blob_uri: str) -> tuple[str, str]:
    """Split a fully qualified blob URL into (container, blob_path)."""
    parts = urlsplit(blob_uri)
    path = parts.path.lstrip("/")
    if not path:
        return ("", "")
    if "/" in path:
        container, blob_path = path.split("/", 1)
        return (unquote(container), unquote(blob_path))
    return (unquote(path), "")


def _container_client_from_sas(sas_uri: str):
    from azure.storage.blob import ContainerClient  # type: ignore[import-not-found]

    return ContainerClient.from_container_url(container_url=sas_uri)


def _safe_join_under(target_path: Path, rel_path: str) -> Path:
    """Resolve ``target_path / rel_path`` and reject paths escaping ``target_path``."""
    candidate = (target_path / rel_path).resolve()
    try:
        candidate.relative_to(target_path)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write blob {rel_path!r}: resolved path "
            f"{candidate} escapes target_dir {target_path}."
        ) from exc
    return candidate


def download_dataset(
    target_dir: str | Path,
    *,
    dataset_name: str,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    project_endpoint: str = foundry_constants.DEFAULT_PROJECT_ENDPOINT,
    project_name: str | None = foundry_constants.DEFAULT_PROJECT_NAME,
    credential: TokenCredential | None = None,
    overwrite: bool = False,
) -> DatasetDownloadResult:
    """Download every blob backing a Foundry dataset version into ``target_dir``."""
    resolved_dataset_name = _normalize_non_empty_string(
        dataset_name, name="dataset_name"
    )
    resolved_dataset_version = _normalize_non_empty_string(
        dataset_version, name="dataset_version"
    )
    resolved_project_endpoint = build_project_endpoint(
        project_endpoint, project_name=project_name
    )
    resolved_credential = credential or get_foundry_credential()

    target_path = Path(target_dir).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(target_path.iterdir()):
        raise FileExistsError(
            f"target_dir {target_path} is not empty; pass overwrite=True to "
            f"re-download into it."
        )

    downloaded_files: list[str] = []
    total_bytes = 0

    with AIProjectClient(
        endpoint=resolved_project_endpoint, credential=resolved_credential
    ) as project_client:
        dataset = project_client.datasets.get(
            name=resolved_dataset_name, version=resolved_dataset_version
        )
        dataset_id = getattr(dataset, "id", None)
        dataset_type = getattr(dataset, "type", None)
        data_uri = getattr(dataset, "data_uri", None)
        if not data_uri:
            raise RuntimeError(
                f"datasets.get for {resolved_dataset_name}:{resolved_dataset_version} "
                f"returned no data_uri; cannot derive blob path."
            )

        creds = project_client.datasets.get_credentials(
            name=resolved_dataset_name, version=resolved_dataset_version
        )
        blob_ref = getattr(creds, "blob_reference", None)
        cred_block = getattr(blob_ref, "credential", None) if blob_ref else None
        sas_uri = getattr(cred_block, "sas_uri", None) if cred_block else None
        if not sas_uri:
            raise RuntimeError(
                f"datasets.get_credentials for "
                f"{resolved_dataset_name}:{resolved_dataset_version} did not "
                f"return a SAS URI on blob_reference.credential."
            )

        _, blob_path_in_container = _parse_blob_uri(str(data_uri))
        container_client = _container_client_from_sas(str(sas_uri))

        try:
            if str(dataset_type) == "uri_file":
                if blob_path_in_container:
                    blob_paths = (blob_path_in_container,)
                else:
                    blob_paths = tuple(
                        str(blob_name)
                        for blob_name in container_client.list_blob_names()
                    )
                if len(blob_paths) != 1:
                    raise RuntimeError(
                        f"uri_file dataset data_uri {data_uri!r} resolved to "
                        f"{len(blob_paths)} blobs; expected exactly one."
                    )
                blob_path = blob_paths[0]
                file_name = Path(blob_path).name or "data"
                local_file = _safe_join_under(target_path, file_name)
                blob_client = container_client.get_blob_client(blob_path)
                stream = blob_client.download_blob()
                with local_file.open("wb") as fh:
                    written = stream.readinto(fh)
                downloaded_files.append(str(local_file.relative_to(target_path)))
                total_bytes += int(written or 0)
            else:
                prefix = (
                    blob_path_in_container.rstrip("/") if blob_path_in_container else ""
                )
                if prefix:
                    blob_iter = container_client.list_blob_names(
                        name_starts_with=prefix + "/"
                    )
                else:
                    blob_iter = container_client.list_blob_names()
                for blob_name in blob_iter:
                    blob_path = str(blob_name)
                    rel_path = blob_path
                    if prefix and blob_path.startswith(prefix + "/"):
                        rel_path = blob_path[len(prefix) + 1 :]
                    elif prefix and blob_path == prefix:
                        rel_path = Path(blob_path).name
                    local_file = _safe_join_under(target_path, rel_path)
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    blob_client = container_client.get_blob_client(blob_path)
                    stream = blob_client.download_blob()
                    with local_file.open("wb") as fh:
                        written = stream.readinto(fh)
                    downloaded_files.append(str(local_file.relative_to(target_path)))
                    total_bytes += int(written or 0)
        finally:
            close = getattr(container_client, "close", None)
            if callable(close):
                close()

    return DatasetDownloadResult(
        dataset_id=dataset_id,
        name=resolved_dataset_name,
        version=resolved_dataset_version,
        dataset_type=str(dataset_type) if dataset_type is not None else None,
        data_uri=str(data_uri) if data_uri else None,
        target_dir=str(target_path),
        downloaded_files=tuple(sorted(downloaded_files)),
        total_bytes=total_bytes,
    )
