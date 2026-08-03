"""End-to-end migration of AML command jobs to Foundry compute."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from . import constants as foundry_constants
from .aml_command_job_migration import (
    audit_aml_command_job_compatibility,
    create_foundry_asset_version,
    translate_aml_command_job,
)
from .aml_command_job_permissions import inspect_connection_permission_info
from .auth import get_foundry_access_token
from .command_jobs import get_job_status_by_name, submit_job
from .dataset import register_reference_dataset, upload_dataset
from .e2e.sanitization import sanitize_for_report
from .model_asset import upload_and_register_model


_FULL_ASSET_ID_PATTERN = re.compile(
    r"(?:azureml:)?/subscriptions/(?P<subscription>[^/]+)/resourcegroups/"
    r"(?P<resource_group>[^/]+)/providers/microsoft\.machinelearningservices/"
    r"workspaces/(?P<workspace>[^/]+)/(?P<collection>data|models|codes|environments)/"
    r"(?P<name>[^/]+)/(?:versions/(?P<version>[^/?#]+)|labels/(?P<label>[^/?#]+))",
    re.IGNORECASE,
)
_COMPACT_ASSET_ID_PATTERN = re.compile(
    r"(?:azureml:)?(?P<name>[^:@/]+)(?::(?P<version>[^/?#]+)|@(?P<label>[^/?#]+))$",
    re.IGNORECASE,
)
_COLLECTION_KIND_MAP = {
    "data": "data",
    "models": "model",
    "codes": "code",
    "environments": "environment",
}
_LITERAL_INPUT_TYPES = frozenset({"literal", "string", "integer", "number", "boolean"})
_MODEL_INPUT_TYPES = frozenset({"custom_model", "mlflow_model", "triton_model"})
_DATA_INPUT_TYPES = frozenset({"uri_file", "uri_folder", "mltable"})
_DATASET_TRANSFER_MODES = frozenset({"upload", "reference"})
_AML_TERMINAL_STATUSES = frozenset(
    {"Completed", "Failed", "Canceled", "Cancelled", "NotResponding"}
)
_FOUNDRY_TERMINAL_STATUSES = frozenset({"Completed", "Failed", "Canceled", "Cancelled"})
_DEFAULT_EXPORT_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
_ASSET_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_AML_DATASTORE_URI_PATTERN = re.compile(
    r"^azureml://(?:.+/)?datastores/(?P<datastore>[^/]+)/paths/(?P<path>.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AmlWorkspace:
    subscription_id: str
    resource_group: str
    workspace_name: str
    export_compute: str
    identity_datastore_name: str = "foundrymigrationidentityblob"


@dataclass(frozen=True)
class FoundryTarget:
    project_endpoint: str
    project_name: str
    storage_connection_name: str
    compute_id: str
    instance_type: str
    api_version: str
    user_assigned_identity_id: str | None = None
    job_tier: str | None = "Premium"


@dataclass(frozen=True)
class MigrationRequest:
    source: AmlWorkspace
    target: FoundryTarget
    source_job_name: str
    work_dir: str | Path
    environment_image_reference: str | None = None
    source_code_path: str | Path | None = None
    target_job_name: str | None = None
    target_job_name_prefix: str | None = None
    output_asset_name_prefix: str | None = None
    asset_version: str | None = None
    dataset_transfer_mode: str = "upload"
    source_storage_connection_name: str | None = None
    wait_for_completion: bool = True
    poll_interval_seconds: float = 15.0
    timeout_seconds: float = 3600.0
    export_environment_image: str = _DEFAULT_EXPORT_IMAGE


@dataclass(frozen=True)
class MigrationResult:
    manifest_path: str
    source_job_name: str
    target_job_name: str
    target_status: str | None
    request_body: dict[str, Any]
    asset_mappings: dict[str, str]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AmlAssetReference:
    kind: str
    name: str
    version: str | None = None
    label: str | None = None
    subscription_id: str | None = None
    resource_group: str | None = None
    workspace_name: str | None = None

    @property
    def version_or_label(self) -> str:
        value = self.version or self.label
        if value is None:
            raise ValueError(
                f"AML {self.kind} asset {self.name!r} has no version or label."
            )
        return value


def parse_aml_asset_reference(
    value: Any,
    *,
    expected_kind: str | None = None,
) -> AmlAssetReference | None:
    """Parse full ARM and compact ``azureml:name:version`` asset references."""

    if value is None:
        return None
    normalized = unquote(str(value).strip())
    if not normalized or normalized.lower().startswith("azureml://"):
        return None

    full_match = _FULL_ASSET_ID_PATTERN.search(normalized)
    if full_match:
        kind = _COLLECTION_KIND_MAP[full_match.group("collection").lower()]
        if expected_kind and kind != expected_kind:
            return None
        return AmlAssetReference(
            kind=kind,
            name=full_match.group("name"),
            version=full_match.group("version"),
            label=full_match.group("label"),
            subscription_id=full_match.group("subscription"),
            resource_group=full_match.group("resource_group"),
            workspace_name=full_match.group("workspace"),
        )

    compact_match = _COMPACT_ASSET_ID_PATTERN.fullmatch(normalized)
    if compact_match and expected_kind:
        return AmlAssetReference(
            kind=expected_kind,
            name=compact_match.group("name"),
            version=compact_match.group("version"),
            label=compact_match.group("label"),
        )
    return None


def _safe_attribute(value: Any, name: str) -> Any:
    if isinstance(value, Mapping) and name in value:
        return value[name]
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _wire_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    enum_value = _safe_attribute(value, "value")
    if enum_value is not None and not isinstance(value, (str, bytes, int, float, bool)):
        return enum_value
    return value


def _reference_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("id", "path"):
            candidate = value.get(key)
            if candidate:
                return str(candidate)
    for name in ("id", "path"):
        candidate = _safe_attribute(value, name)
        if candidate:
            return str(candidate)
    asset_name = _safe_attribute(value, "name")
    asset_version = _safe_attribute(value, "version")
    if asset_name and asset_version:
        return f"azureml:{asset_name}:{asset_version}"
    return str(value)


def _compact(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _json_safe_value(value: Any) -> Any:
    wire_value = _wire_value(value)
    if wire_value is None or isinstance(wire_value, (str, int, float, bool)):
        return wire_value
    if isinstance(wire_value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in wire_value.items()}
    if isinstance(wire_value, (list, tuple, set)):
        return [_json_safe_value(item) for item in wire_value]
    return str(wire_value)


def _sha256_json(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _identity_text(value: Any, *, casefold: bool = False) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().rstrip("/")
    if not normalized:
        return None
    return normalized.casefold() if casefold else normalized


def _migration_invocation_identity(
    request: MigrationRequest,
    *,
    dataset_transfer_mode: str,
) -> dict[str, Any]:
    source_code_path = (
        os.path.normcase(str(Path(request.source_code_path).expanduser().resolve()))
        if request.source_code_path is not None
        else None
    )
    return {
        "schemaVersion": 1,
        "source": {
            "subscriptionId": _identity_text(
                request.source.subscription_id,
                casefold=True,
            ),
            "resourceGroup": _identity_text(
                request.source.resource_group,
                casefold=True,
            ),
            "workspaceName": _identity_text(
                request.source.workspace_name,
                casefold=True,
            ),
            "jobName": _identity_text(
                request.source_job_name,
                casefold=True,
            ),
            "exportCompute": _identity_text(
                request.source.export_compute,
                casefold=True,
            ),
            "identityDatastoreName": _identity_text(
                request.source.identity_datastore_name,
                casefold=True,
            ),
        },
        "target": {
            "projectEndpoint": _identity_text(
                request.target.project_endpoint,
                casefold=True,
            ),
            "projectName": _identity_text(
                request.target.project_name,
                casefold=True,
            ),
            "storageConnectionName": _identity_text(
                request.target.storage_connection_name,
                casefold=True,
            ),
            "computeId": _identity_text(
                request.target.compute_id,
                casefold=True,
            ),
            "instanceType": _identity_text(
                request.target.instance_type,
                casefold=True,
            ),
            "apiVersion": _identity_text(
                request.target.api_version,
                casefold=True,
            ),
            "userAssignedIdentityId": _identity_text(
                request.target.user_assigned_identity_id,
                casefold=True,
            ),
            "jobTier": _identity_text(
                request.target.job_tier,
                casefold=True,
            ),
        },
        "migration": {
            "requestedAssetVersion": _identity_text(request.asset_version),
            "datasetTransferMode": dataset_transfer_mode,
            "sourceStorageConnectionName": _identity_text(
                request.source_storage_connection_name,
                casefold=True,
            ),
            "environmentImageReference": _identity_text(
                request.environment_image_reference
            ),
            "sourceCodePath": source_code_path,
            "targetJobName": _identity_text(
                request.target_job_name,
                casefold=True,
            ),
            "targetJobNamePrefix": _identity_text(
                request.target_job_name_prefix,
                casefold=True,
            ),
            "outputAssetNamePrefix": _identity_text(
                request.output_asset_name_prefix,
                casefold=True,
            ),
            "exportEnvironmentImage": _identity_text(request.export_environment_image),
        },
    }


def _serialize_binding(value: Any, *, output: bool) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if not output and isinstance(value, (str, bool, int, float)):
        literal_type = (
            "boolean"
            if isinstance(value, bool)
            else "integer"
            if isinstance(value, int)
            else "number"
            if isinstance(value, float)
            else "string"
        )
        return {"type": literal_type, "value": value}

    binding_type = _wire_value(_safe_attribute(value, "type"))
    literal_value = _safe_attribute(value, "value")
    if literal_value is None:
        raw_data = _safe_attribute(value, "_data")
        if isinstance(raw_data, (str, int, float, bool)):
            literal_value = raw_data
    path = _safe_attribute(value, "path")
    mode = _wire_value(_safe_attribute(value, "mode"))
    serialized = _compact(
        {
            "type": str(binding_type) if binding_type is not None else None,
            "value": literal_value,
            "path": str(path) if path is not None else None,
            "mode": str(mode) if mode is not None else None,
            "description": _safe_attribute(value, "description"),
            "path_on_compute": _safe_attribute(value, "path_on_compute"),
            "default": _json_safe_value(_safe_attribute(value, "default")),
            "optional": _safe_attribute(value, "optional"),
            "min": _safe_attribute(value, "min"),
            "max": _safe_attribute(value, "max"),
            "enum": _json_safe_value(_safe_attribute(value, "enum")),
            "datastore": _safe_attribute(value, "datastore"),
            "intellectual_property": (
                True if _safe_attribute(value, "intellectual_property") else None
            ),
            "early_available": (
                True if _safe_attribute(value, "early_available") else None
            ),
        }
    )
    if not output and literal_value is not None:
        serialized.setdefault("type", "literal")
        serialized.pop("path", None)
        serialized.pop("mode", None)
    return serialized


def _serialize_bindings(value: Any, *, output: bool) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): _serialize_binding(binding, output=output)
        for name, binding in value.items()
    }


def _serialize_named_attributes(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    return _compact({name: _wire_value(_safe_attribute(value, name)) for name in names})


def _serialize_services(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): _serialize_named_attributes(
            service,
            ("type", "port", "properties", "nodes"),
        )
        for name, service in value.items()
    }


def materialize_aml_command_job(job: Any) -> dict[str, Any]:
    """Convert an AML SDK ``Command`` entity into the translation contract."""

    if isinstance(job, Mapping) and not hasattr(job, "_to_rest_object"):
        return deepcopy(dict(job))

    resources = _serialize_named_attributes(
        _safe_attribute(job, "resources"),
        (
            "instance_count",
            "instance_type",
            "shm_size",
            "docker_args",
            "locations",
        ),
    )
    limits = _serialize_named_attributes(
        _safe_attribute(job, "limits"),
        ("timeout",),
    )
    distribution = _serialize_named_attributes(
        _safe_attribute(job, "distribution"),
        (
            "type",
            "process_count_per_instance",
            "worker_count",
            "parameter_server_count",
            "chief_count",
            "ray_port",
            "port",
            "address",
            "include_dashboard",
            "dashboard_port",
            "head_node_additional_args",
            "worker_node_additional_args",
        ),
    )
    queue_settings = _serialize_named_attributes(
        _safe_attribute(job, "queue_settings"),
        ("job_tier", "priority"),
    )
    direct_job_tier = _wire_value(_safe_attribute(job, "job_tier"))
    if isinstance(direct_job_tier, Mapping) and not direct_job_tier:
        direct_job_tier = None
    direct_priority = _wire_value(_safe_attribute(job, "priority"))
    if isinstance(direct_priority, Mapping) and not direct_priority:
        direct_priority = None

    result = _compact(
        {
            "name": _safe_attribute(job, "name"),
            "type": _wire_value(_safe_attribute(job, "type")) or "command",
            "display_name": _safe_attribute(job, "display_name"),
            "description": _safe_attribute(job, "description"),
            "experiment_name": _safe_attribute(job, "experiment_name"),
            "command": _safe_attribute(job, "command"),
            "code": _reference_value(_safe_attribute(job, "code")),
            "environment": _reference_value(_safe_attribute(job, "environment")),
            "compute": _reference_value(_safe_attribute(job, "compute")),
            "environment_variables": deepcopy(
                _safe_attribute(job, "environment_variables") or {}
            ),
            "properties": _json_safe_value(_safe_attribute(job, "properties") or {}),
            "identity": (
                _wire_value(_safe_attribute(_safe_attribute(job, "identity"), "type"))
                or (
                    type(_safe_attribute(job, "identity")).__name__
                    if _safe_attribute(job, "identity") is not None
                    else None
                )
            ),
            "is_deterministic": _safe_attribute(job, "is_deterministic"),
            "parent_job_name": _safe_attribute(job, "parent_job_name"),
            "notification_setting": bool(_safe_attribute(job, "notification_setting")),
            "intellectual_property": bool(
                _safe_attribute(job, "intellectual_property")
            ),
            "parameters": _json_safe_value(_safe_attribute(job, "parameters") or {}),
            "inputs": _serialize_bindings(_safe_attribute(job, "inputs"), output=False),
            "outputs": _serialize_bindings(
                _safe_attribute(job, "outputs"), output=True
            ),
            "resources": resources or None,
            "limits": limits or None,
            "distribution": distribution or None,
            "queue_settings": queue_settings or None,
            "job_tier": direct_job_tier or queue_settings.get("job_tier"),
            "priority": direct_priority or queue_settings.get("priority"),
            "services": _serialize_services(_safe_attribute(job, "services")) or None,
            "tags": deepcopy(_safe_attribute(job, "tags") or {}),
        }
    )
    return result


def _normalize_asset_name(value: str) -> str:
    normalized = _ASSET_NAME_PATTERN.sub("-", value).strip("-.")
    if not normalized:
        raise ValueError(f"Could not derive an asset name from {value!r}.")
    return normalized[:255]


def _status_text(value: Any) -> str:
    normalized = _wire_value(value)
    return str(normalized) if normalized is not None else ""


def _single_file_under(path: Path) -> Path:
    files = sorted(entry for entry in path.rglob("*") if entry.is_file())
    if len(files) != 1:
        raise RuntimeError(f"Expected one file under {path}, but found {len(files)}.")
    return files[0]


def _content_root(path: Path, *, preferred_name: str | None = None) -> Path:
    if preferred_name:
        preferred = path / preferred_name
        if preferred.exists():
            return preferred
    children = list(path.iterdir()) if path.exists() else []
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return path


def _folder_reference_root_and_suffix(uri: str) -> tuple[str, str | None]:
    parsed = urlsplit(uri)
    if parsed.scheme.lower() != "https" or ".blob." not in parsed.netloc.lower():
        raise ValueError(f"Not an Azure Blob HTTPS URI: {uri!r}")
    path = unquote(parsed.path.lstrip("/"))
    container_name, separator, blob_path = path.partition("/")
    if not container_name:
        raise ValueError(f"Azure Blob URI has no container: {uri!r}")
    container_uri = parsed._replace(
        path=f"/{container_name}",
        query="",
        fragment="",
    ).geturl()
    suffix = blob_path.strip("/") if separator else ""
    return container_uri, suffix or None


def _download_azure_blob_path_with_identity(
    uri: str,
    target_dir: Path,
    *,
    credential: Any,
) -> tuple[str, ...]:
    from azure.storage.blob import BlobServiceClient

    parsed = urlsplit(uri)
    if parsed.scheme.lower() != "https" or ".blob." not in parsed.netloc.lower():
        raise ValueError(f"Not an Azure Blob HTTPS URI: {uri!r}")
    path = unquote(parsed.path.lstrip("/"))
    if "/" not in path:
        raise ValueError(f"Azure Blob URI has no blob path: {uri!r}")
    container_name, blob_path = path.split("/", 1)
    blob_path = blob_path.rstrip("/")
    if not blob_path:
        raise ValueError(f"Azure Blob URI has an empty blob path: {uri!r}")

    target_dir.mkdir(parents=True, exist_ok=True)
    service_client = BlobServiceClient(
        f"{parsed.scheme}://{parsed.netloc}", credential=credential
    )
    container_client = service_client.get_container_client(container_name)
    downloaded: list[str] = []
    try:
        names = sorted(
            str(name)
            for name in container_client.list_blob_names(name_starts_with=blob_path)
            if str(name) == blob_path or str(name).startswith(blob_path + "/")
        )
        if not names:
            raise RuntimeError(f"No blobs found at {uri!r}.")
        for name in names:
            relative_name = (
                Path(name).name if name == blob_path else name[len(blob_path) + 1 :]
            )
            local_path = (target_dir / relative_name).resolve()
            try:
                local_path.relative_to(target_dir)
            except ValueError as error:
                raise ValueError(
                    f"Refusing to download blob outside target_dir: {name!r}."
                ) from error
            local_path.parent.mkdir(parents=True, exist_ok=True)
            stream = container_client.get_blob_client(name).download_blob()
            with local_path.open("wb") as file_handle:
                stream.readinto(file_handle)
            downloaded.append(str(local_path.relative_to(target_dir)))
    finally:
        close_container = getattr(container_client, "close", None)
        if callable(close_container):
            close_container()
        service_client.close()
    return tuple(sorted(downloaded))


def _download_workspace_blob_prefix(
    ml_client: Any,
    credential: Any,
    *,
    prefix: str,
    target_dir: Path,
) -> tuple[str, ...]:
    datastore = ml_client.datastores.get("workspaceblobstore")
    account_name = str(_safe_attribute(datastore, "account_name") or "")
    container_name = str(_safe_attribute(datastore, "container_name") or "")
    endpoint = str(_safe_attribute(datastore, "endpoint") or "core.windows.net")
    if not account_name or not container_name:
        raise RuntimeError(
            "workspaceblobstore did not expose account_name/container_name."
        )
    uri = (
        f"https://{account_name}.blob.{endpoint}/{container_name}/{prefix.lstrip('/')}"
    )
    return _download_azure_blob_path_with_identity(
        uri,
        target_dir,
        credential=credential,
    )


def _download_asset_path_with_identity(
    ml_client: Any,
    credential: Any,
    *,
    uri: str,
    target_dir: Path,
) -> tuple[str, ...]:
    uri = _resolve_aml_datastore_uri(ml_client, uri)
    return _download_azure_blob_path_with_identity(
        str(uri),
        target_dir,
        credential=credential,
    )


def _resolve_aml_datastore_uri(ml_client: Any, uri: str) -> str:
    datastore_match = _AML_DATASTORE_URI_PATTERN.fullmatch(str(uri).strip())
    if datastore_match:
        datastore = ml_client.datastores.get(datastore_match.group("datastore"))
        account_name = str(_safe_attribute(datastore, "account_name") or "")
        container_name = str(_safe_attribute(datastore, "container_name") or "")
        endpoint = str(_safe_attribute(datastore, "endpoint") or "core.windows.net")
        if not account_name or not container_name:
            raise RuntimeError(
                f"AML datastore {datastore_match.group('datastore')!r} did not "
                "expose account_name/container_name."
            )
        uri = (
            f"https://{account_name}.blob.{endpoint}/{container_name}/"
            f"{datastore_match.group('path')}"
        )
    return str(uri)


class _MigrationJournal:
    def __init__(self, path: Path, initial: Mapping[str, Any]) -> None:
        self.path = path
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"Migration manifest must be a JSON object: {path}")
            self.data = loaded
        else:
            self.data = deepcopy(dict(initial))
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(sanitize_for_report(self.data), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)


@dataclass(frozen=True)
class _TransferSpec:
    binding_name: str
    source_uri: str
    input_type: str
    export_name: str


class AmlCommandJobMigrator:
    """Move an AML command job and its dependencies to Foundry compute."""

    def __init__(
        self,
        request: MigrationRequest,
        *,
        credential: Any | None = None,
        ml_client: Any | None = None,
        emit: Any = print,
        sleeper: Any = time.sleep,
    ) -> None:
        self.request = request
        self.emit = emit
        self._sleep = sleeper
        if request.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero.")
        if request.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        self.dataset_transfer_mode = str(request.dataset_transfer_mode).lower()
        if self.dataset_transfer_mode not in _DATASET_TRANSFER_MODES:
            raise ValueError("dataset_transfer_mode must be 'upload' or 'reference'.")
        if (
            self.dataset_transfer_mode == "reference"
            and not str(request.source_storage_connection_name or "").strip()
        ):
            raise ValueError(
                "source_storage_connection_name is required when "
                "dataset_transfer_mode is 'reference'."
            )

        if credential is None:
            from azure.identity import AzureCliCredential

            credential = AzureCliCredential()
        self.credential = credential
        if ml_client is None:
            from azure.ai.ml import MLClient

            ml_client = MLClient(
                credential,
                request.source.subscription_id,
                request.source.resource_group,
                request.source.workspace_name,
            )
        self.ml_client = ml_client
        self._foundry_access_token: str | None = None
        self._foundry_access_token_expires_on = 0
        self.work_dir = Path(request.work_dir).expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.asset_version = request.asset_version or create_foundry_asset_version()
        self.invocation_identity = _migration_invocation_identity(
            request,
            dataset_transfer_mode=self.dataset_transfer_mode,
        )
        self.journal = _MigrationJournal(
            self.work_dir / "migration-manifest.json",
            {
                "schemaVersion": 1,
                "createdAtUtc": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "subscriptionId": request.source.subscription_id,
                    "resourceGroup": request.source.resource_group,
                    "workspaceName": request.source.workspace_name,
                    "jobName": request.source_job_name,
                    "exportCompute": request.source.export_compute,
                },
                "target": {
                    "projectEndpoint": request.target.project_endpoint,
                    "projectName": request.target.project_name,
                    "computeId": request.target.compute_id,
                    "instanceType": request.target.instance_type,
                    "datasetTransferMode": self.dataset_transfer_mode,
                    "sourceStorageConnectionName": (
                        request.source_storage_connection_name
                    ),
                },
                "assetVersion": self.asset_version,
                "invocation": self.invocation_identity,
                "invocationSha256": _sha256_json(self.invocation_identity),
                "code": {},
                "inputs": {},
                "warnings": [],
            },
        )
        manifest_version = str(self.journal.data.get("assetVersion") or "")
        if manifest_version:
            self.asset_version = manifest_version
        self._validate_manifest_identity()

    def _validate_manifest_identity(self) -> None:
        source = self.journal.data.get("source") or {}
        target = self.journal.data.get("target") or {}
        expected = {
            "source.subscriptionId": (
                source.get("subscriptionId"),
                self.request.source.subscription_id,
            ),
            "source.resourceGroup": (
                source.get("resourceGroup"),
                self.request.source.resource_group,
            ),
            "source.workspaceName": (
                source.get("workspaceName"),
                self.request.source.workspace_name,
            ),
            "source.jobName": (
                source.get("jobName"),
                self.request.source_job_name,
            ),
            "target.projectEndpoint": (
                target.get("projectEndpoint"),
                self.request.target.project_endpoint,
            ),
            "target.projectName": (
                target.get("projectName"),
                self.request.target.project_name,
            ),
            "target.computeId": (
                target.get("computeId"),
                self.request.target.compute_id,
            ),
            "target.datasetTransferMode": (
                target.get("datasetTransferMode") or "upload",
                self.dataset_transfer_mode,
            ),
            "target.sourceStorageConnectionName": (
                target.get("sourceStorageConnectionName"),
                self.request.source_storage_connection_name,
            ),
        }
        mismatches = {
            key: {"manifest": actual, "request": requested}
            for key, (actual, requested) in expected.items()
            if str(actual or "").rstrip("/").lower()
            != str(requested or "").rstrip("/").lower()
        }
        if mismatches:
            raise ValueError(
                "Existing migration manifest belongs to a different source or "
                f"target: {mismatches}. Use a new work_dir."
            )
        recorded_invocation = self.journal.data.get("invocation")
        recorded_fingerprint = self.journal.data.get("invocationSha256")
        expected_fingerprint = _sha256_json(self.invocation_identity)
        if (
            not isinstance(recorded_invocation, Mapping)
            or recorded_fingerprint != expected_fingerprint
        ):
            raise ValueError(
                "Existing migration manifest belongs to a different migration "
                "invocation or predates complete resume validation. Use a new "
                "work_dir."
            )

    def _record_source_job(self, source_job: Mapping[str, Any]) -> None:
        source_fingerprint = _sha256_json(source_job)
        recorded_fingerprint = self.journal.data.get("sourceJobSha256")
        if recorded_fingerprint and recorded_fingerprint != source_fingerprint:
            raise ValueError(
                "The source AML command-job definition changed since this "
                "migration work_dir was created. Use a new work_dir."
            )
        self.journal.data["sourceJob"] = deepcopy(dict(source_job))
        self.journal.data["sourceJobSha256"] = source_fingerprint
        self._save()

    def _record_foundry_request(
        self,
        foundry_state: dict[str, Any],
        request_body: Mapping[str, Any],
    ) -> Path:
        request_fingerprint = _sha256_json(request_body)
        recorded_fingerprint = foundry_state.get("requestBodySha256")
        if foundry_state.get("name") and not recorded_fingerprint:
            recorded_body = self.journal.data.get("requestBody")
            if isinstance(recorded_body, Mapping):
                recorded_fingerprint = _sha256_json(recorded_body)
        if foundry_state.get("name") and recorded_fingerprint != request_fingerprint:
            raise ValueError(
                f"Recorded Foundry job {foundry_state.get('name')!r} was submitted "
                "with a different request body. Use a new work_dir."
            )

        foundry_state["requestBodySha256"] = request_fingerprint
        self.journal.data["requestBody"] = deepcopy(dict(request_body))
        self.journal.data["requestBodySha256"] = request_fingerprint
        payload_path = self.work_dir / "foundry-job-request.json"
        payload_path.write_text(
            json.dumps(
                sanitize_for_report(request_body),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.journal.data["requestBodyPath"] = str(payload_path)
        self._save()
        return payload_path

    def _save(self) -> None:
        self.journal.save()

    def _get_foundry_token(self) -> str:
        if (
            self._foundry_access_token is None
            or self._foundry_access_token_expires_on <= int(time.time()) + 120
        ):
            access_token = get_foundry_access_token(credential=self.credential)
            self._foundry_access_token = access_token.token
            self._foundry_access_token_expires_on = int(
                getattr(access_token, "expires_on", 0) or 0
            )
            if self._foundry_access_token_expires_on == 0:
                self._foundry_access_token_expires_on = int(time.time()) + 30 * 60
        return self._foundry_access_token

    def _client_for_reference(self, reference: AmlAssetReference) -> Any:
        if not all(
            (
                reference.subscription_id,
                reference.resource_group,
                reference.workspace_name,
            )
        ):
            return self.ml_client
        source = self.request.source
        if (
            reference.subscription_id.lower() == source.subscription_id.lower()
            and reference.resource_group.lower() == source.resource_group.lower()
            and reference.workspace_name.lower() == source.workspace_name.lower()
        ):
            return self.ml_client

        from azure.ai.ml import MLClient

        return MLClient(
            self.credential,
            reference.subscription_id,
            reference.resource_group,
            reference.workspace_name,
        )

    def _resolve_environment_image(self, source_environment: Any) -> str:
        override = self.request.environment_image_reference
        if override:
            return str(override)

        reference_text = _reference_value(source_environment)
        reference = parse_aml_asset_reference(
            reference_text,
            expected_kind="environment",
        )
        if reference is None:
            image = _safe_attribute(source_environment, "image")
            if image:
                self._validate_image_only_environment(source_environment)
                return str(image)
            if reference_text and not reference_text.lower().startswith("azureml:"):
                return reference_text
            raise ValueError(
                "Could not resolve the AML environment to a container image. "
                "Pass environment_image_reference explicitly."
            )

        environment_client = self._client_for_reference(reference)
        environment = environment_client.environments.get(
            reference.name,
            version=reference.version,
            label=reference.label,
        )
        self._validate_image_only_environment(environment)
        image = _safe_attribute(environment, "image")
        if not image:
            raise ValueError(
                f"AML environment {reference.name!r} has no reusable image. Pass "
                "environment_image_reference with a prebuilt image containing the "
                "same dependencies."
            )
        return str(image)

    @staticmethod
    def _validate_image_only_environment(environment: Any) -> None:
        if _safe_attribute(environment, "build"):
            raise ValueError(
                "AML build-context environments are not portable to Foundry jobs. "
                "Publish the built image and pass environment_image_reference."
            )
        if _safe_attribute(environment, "conda_file"):
            raise ValueError(
                "AML Conda overlays are not evaluated by Foundry jobs. Publish an "
                "equivalent prebuilt image and pass environment_image_reference."
            )

    def _download_code(self, source_code: Any, source_name: str) -> Path | None:
        code_state = self.journal.data.setdefault("code", {})
        existing_path = code_state.get("localPath")
        if existing_path and Path(existing_path).exists():
            return Path(existing_path)

        explicit_path = self.request.source_code_path
        reference_text = _reference_value(source_code)
        local_reference = (
            Path(reference_text).expanduser()
            if reference_text and Path(reference_text).expanduser().exists()
            else None
        )
        if explicit_path is not None or local_reference is not None:
            source_path = Path(explicit_path or local_reference).expanduser().resolve()
            if not source_path.exists():
                raise FileNotFoundError(
                    f"source_code_path does not exist: {source_path}"
                )
            destination = self.work_dir / "source-code"
            if destination.exists():
                shutil.rmtree(destination)
            if source_path.is_dir():
                shutil.copytree(source_path, destination)
                local_path = destination
            else:
                destination.mkdir(parents=True)
                local_path = destination / source_path.name
                shutil.copy2(source_path, local_path)
        elif reference_text is None:
            return None
        else:
            reference = parse_aml_asset_reference(
                reference_text,
                expected_kind="code",
            )
            if reference is None or reference.version is None:
                raise ValueError(
                    f"AML code reference {reference_text!r} is not a versioned code "
                    "asset. Pass source_code_path to supply its snapshot."
                )
            destination = self.work_dir / "source-code"
            destination.mkdir(parents=True, exist_ok=True)
            code_client = self._client_for_reference(reference)
            code_operations = getattr(code_client, "_code", None)
            if code_operations is None or not hasattr(code_operations, "download"):
                raise RuntimeError(
                    "Installed azure-ai-ml does not expose code asset download."
                )
            try:
                code_operations.download(
                    reference.name,
                    reference.version,
                    str(destination),
                )
            except Exception as download_error:
                code_asset = code_operations.get(
                    reference.name,
                    reference.version,
                )
                code_path = _reference_value(_safe_attribute(code_asset, "path"))
                if not code_path:
                    raise RuntimeError(
                        f"AML code download failed and code asset "
                        f"{reference.name}:{reference.version} exposed no path."
                    ) from download_error
                if destination.exists():
                    shutil.rmtree(destination)
                destination.mkdir(parents=True)
                _download_azure_blob_path_with_identity(
                    code_path,
                    destination,
                    credential=self.credential,
                )
                self.journal.data.setdefault("warnings", []).append(
                    f"Code asset {reference.name!r}: AML SDK download failed "
                    f"({download_error}); downloaded its blob path with Entra ID."
                )
            local_path = _content_root(destination, preferred_name=reference.name)

        code_state.update(
            {
                "sourceUri": reference_text,
                "localPath": str(local_path),
                "sourceJob": source_name,
            }
        )
        self._save()
        return local_path

    def _download_registered_model(
        self,
        reference: AmlAssetReference,
        *,
        binding_name: str,
    ) -> Path:
        destination = self.work_dir / "inputs" / binding_name / "model"
        destination.mkdir(parents=True, exist_ok=True)
        model_client = self._client_for_reference(reference)
        version = reference.version
        if version is None:
            model = model_client.models.get(
                reference.name,
                label=reference.label,
            )
            version = str(_safe_attribute(model, "version") or "")
        else:
            model = model_client.models.get(reference.name, version=version)
        if not version:
            raise ValueError(
                f"Could not resolve a version for AML model {reference.name!r}."
            )
        model_path = _reference_value(_safe_attribute(model, "path"))
        try:
            model_client.models.download(
                reference.name,
                version,
                download_path=str(destination),
            )
        except Exception as download_error:
            if not model_path:
                raise RuntimeError(
                    f"AML model download failed and model {reference.name}:"
                    f"{version} exposed no backing path."
                ) from download_error
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)
            _download_asset_path_with_identity(
                model_client,
                self.credential,
                uri=model_path,
                target_dir=destination,
            )
            self.journal.data.setdefault("warnings", []).append(
                f"Input {binding_name!r}: AML SDK model download failed "
                f"({download_error}); downloaded its backing path with Entra ID."
            )
        content_root = _content_root(destination, preferred_name=reference.name)
        if model_path:
            backing_name = unquote(urlsplit(model_path).path.rstrip("/")).rsplit(
                "/",
                1,
            )[-1]
            backing_root = content_root / backing_name
            if backing_name and backing_root.is_dir():
                content_root = backing_root
        return content_root

    def _download_registered_data(
        self,
        reference: AmlAssetReference,
        *,
        binding_name: str,
    ) -> Path:
        destination = self.work_dir / "inputs" / binding_name / "data"
        data_client, data_path = self._registered_data_path(reference)
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        _download_asset_path_with_identity(
            data_client,
            self.credential,
            uri=data_path,
            target_dir=destination,
        )
        return _content_root(destination, preferred_name=reference.name)

    def _registered_data_path(
        self,
        reference: AmlAssetReference,
    ) -> tuple[Any, str]:
        data_client = self._client_for_reference(reference)
        data = data_client.data.get(
            reference.name,
            version=reference.version,
            label=reference.label,
        )
        data_path = _reference_value(_safe_attribute(data, "path"))
        if not data_path:
            raise RuntimeError(
                f"AML data asset {reference.name!r} exposed no backing path."
            )
        return data_client, data_path

    def _resolve_reference_data_uri(
        self,
        source_uri: str,
    ) -> str:
        reference = parse_aml_asset_reference(source_uri, expected_kind="data")
        if reference is not None:
            data_client, data_path = self._registered_data_path(reference)
        else:
            data_client = self.ml_client
            data_path = source_uri
        resolved_uri = _resolve_aml_datastore_uri(data_client, data_path)
        if urlsplit(resolved_uri).scheme.lower() != "https":
            raise ValueError(
                f"Source data URI {source_uri!r} resolved to {resolved_uri!r}; "
                "zero-copy Dataset V3 references require an HTTPS storage URI."
            )
        return resolved_uri

    def _validate_reference_connection(
        self,
        source_inputs: Mapping[str, Any],
    ) -> None:
        if self.dataset_transfer_mode != "reference":
            return

        connection_name = str(self.request.source_storage_connection_name or "")
        connection = inspect_connection_permission_info(
            project_endpoint=self.request.target.project_endpoint,
            project_name=self.request.target.project_name,
            connection_name=connection_name,
            credential=self.credential,
        )
        if not connection.available:
            raise ValueError(
                f"Could not inspect source-storage connection {connection_name!r}: "
                f"{connection.error or 'unknown error'}."
            )
        connection_host = (urlsplit(connection.target or "").hostname or "").casefold()
        if not connection_host:
            raise ValueError(
                f"Source-storage connection {connection_name!r} has no HTTPS "
                "target host."
            )

        validated_hosts: set[str] = set()
        for raw_name, raw_binding in source_inputs.items():
            if not isinstance(raw_binding, Mapping):
                continue
            input_type = str(raw_binding.get("type") or "").lower()
            if input_type not in _DATA_INPUT_TYPES:
                continue
            source_uri = raw_binding.get("path") or raw_binding.get("uri")
            if not source_uri:
                raise ValueError(f"AML input {str(raw_name)!r} has no path/uri.")
            resolved_uri = self._resolve_reference_data_uri(str(source_uri))
            source_host = (urlsplit(resolved_uri).hostname or "").casefold()
            if not source_host or source_host != connection_host:
                raise ValueError(
                    f"Source-storage connection {connection_name!r} targets "
                    f"{connection_host!r}, which does not match input "
                    f"{str(raw_name)!r} host {source_host or 'unknown'!r}."
                )
            validated_hosts.add(source_host)

        self.journal.data["referenceConnectionValidation"] = {
            "connectionName": connection_name,
            "connectionHost": connection_host,
            "validatedInputHosts": sorted(validated_hosts),
        }
        self._save()

    def _wait_for_aml_job(self, job_name: str) -> Any:
        deadline = time.monotonic() + self.request.timeout_seconds
        while True:
            job = self.ml_client.jobs.get(job_name)
            status = _status_text(_safe_attribute(job, "status"))
            self.emit(f"AML export job {job_name}: {status or 'Unknown'}")
            if status in _AML_TERMINAL_STATUSES:
                if status != "Completed":
                    raise RuntimeError(
                        f"AML export job {job_name} reached terminal status {status}."
                    )
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for AML export job {job_name}.")
            self._sleep(self.request.poll_interval_seconds)

    def _export_inputs(self, specs: list[_TransferSpec]) -> dict[str, Path]:
        if not specs:
            return {}

        from azure.ai.ml import Input, Output, command
        from azure.ai.ml.entities import Environment, UserIdentityConfiguration

        export_script = (
            "from pathlib import Path\n"
            "import argparse\n"
            "import shutil\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--pair', action='append', nargs=2, required=True)\n"
            "args = parser.parse_args()\n"
            "for source_text, target_text in args.pair:\n"
            "    source = Path(source_text)\n"
            "    target = Path(target_text)\n"
            "    target.mkdir(parents=True, exist_ok=True)\n"
            "    if source.is_dir():\n"
            "        for child in source.iterdir():\n"
            "            destination = target / child.name\n"
            "            if child.is_dir():\n"
            "                shutil.copytree(child, destination, dirs_exist_ok=True)\n"
            "            else:\n"
            "                shutil.copy2(child, destination)\n"
            "    else:\n"
            "        shutil.copy2(source, target / source.name)\n"
        )
        encoded_script = base64.b64encode(export_script.encode("utf-8")).decode("ascii")
        inputs = {
            spec.export_name: Input(
                type=spec.input_type,
                path=spec.source_uri,
                mode="download",
            )
            for spec in specs
        }
        export_prefix = f"aml-foundry-command-migration/{self.asset_version}/export"
        outputs = {
            f"exported_{spec.export_name}": Output(
                type="uri_folder",
                path=(
                    f"azureml://datastores/{self.request.source.identity_datastore_name}/paths/"
                    f"{export_prefix}/exported_{spec.export_name}"
                ),
                mode="upload",
            )
            for spec in specs
        }
        command_parts = [
            f"python -c \"import base64;exec(base64.b64decode('{encoded_script}'))\""
        ]
        for spec in specs:
            command_parts.append(
                f'--pair "${{{{inputs.{spec.export_name}}}}}" '
                f'"${{{{outputs.exported_{spec.export_name}}}}}"'
            )
        export_name = f"foundry-migration-export-{self.asset_version[-10:]}"
        export_job = command(
            name=export_name,
            display_name=f"Export dependencies for {self.request.source_job_name}",
            description="Copies AML job dependencies for Foundry migration.",
            command=" ".join(command_parts),
            environment=Environment(image=self.request.export_environment_image),
            compute=self.request.source.export_compute,
            inputs=inputs,
            outputs=outputs,
            identity=UserIdentityConfiguration(),
            experiment_name="foundry-command-job-migration",
            timeout=int(self.request.timeout_seconds),
            tags={
                "migration.sourceJob": self.request.source_job_name,
                "migration.purpose": "asset-export",
            },
        )
        created_name = self.journal.data.get("exportJobName")
        if created_name:
            existing_job = self.ml_client.jobs.get(str(created_name))
            existing_status = _status_text(_safe_attribute(existing_job, "status"))
            if existing_status == "Completed":
                self.emit(f"Reusing completed AML export job {created_name}.")
            elif existing_status in _AML_TERMINAL_STATUSES:
                raise RuntimeError(
                    f"Recorded AML export job {created_name} has status "
                    f"{existing_status}; use a new work_dir to retry it."
                )
            else:
                self._wait_for_aml_job(str(created_name))
        else:
            created = self.ml_client.jobs.create_or_update(export_job)
            created_name = str(_safe_attribute(created, "name") or export_name)
            self.journal.data["exportJobName"] = created_name
            self._save()
            self._wait_for_aml_job(created_name)

        download_root = self.work_dir / "export-download"
        result: dict[str, Path] = {}
        for spec in specs:
            output_name = f"exported_{spec.export_name}"
            candidate = download_root / output_name
            _download_workspace_blob_prefix(
                self.ml_client,
                self.credential,
                prefix=f"{export_prefix}/{output_name}",
                target_dir=candidate,
            )
            result[spec.binding_name] = candidate
        return result

    def _upload_code(self, local_path: Path | None, source_name: str) -> str | None:
        if local_path is None:
            return None
        code_state = self.journal.data.setdefault("code", {})
        existing_id = code_state.get("foundryAssetId")
        if existing_id:
            return str(existing_id)
        asset_name = _normalize_asset_name(f"migrated-{source_name}-code")
        uploaded = upload_dataset(
            local_path,
            dataset_name=asset_name,
            dataset_version=self.asset_version,
            project_endpoint=self.request.target.project_endpoint,
            project_name=self.request.target.project_name,
            connection_name=self.request.target.storage_connection_name,
            credential=self.credential,
        )
        code_state.update(
            {
                "assetName": asset_name,
                "assetVersion": self.asset_version,
                "foundryAssetId": uploaded.dataset_id,
            }
        )
        self._save()
        return uploaded.dataset_id

    def _prepare_inputs(
        self,
        source_inputs: Mapping[str, Any],
        source_name: str,
    ) -> dict[str, str]:
        input_state = self.journal.data.setdefault("inputs", {})
        transfer_specs: list[_TransferSpec] = []
        local_paths: dict[str, Path] = {}
        resolved_reference_uris: dict[str, str] = {}

        for index, (raw_name, raw_binding) in enumerate(source_inputs.items()):
            name = str(raw_name)
            binding = dict(raw_binding) if isinstance(raw_binding, Mapping) else {}
            input_type = str(
                binding.get("type")
                or ("literal" if "value" in binding else "uri_folder")
            ).lower()
            source_input_type = input_type
            if input_type in _LITERAL_INPUT_TYPES:
                continue
            source_uri = binding.get("path") or binding.get("uri")
            if not source_uri:
                raise ValueError(f"AML input {name!r} has no path/uri.")
            source_uri = str(source_uri)
            source_uri_fingerprint = _sha256_json(source_uri)
            record = input_state.setdefault(
                name,
                {
                    "sourceUri": sanitize_for_report(source_uri),
                    "sourceUriSha256": source_uri_fingerprint,
                    "inputType": input_type,
                },
            )
            recorded_source_fingerprint = record.get("sourceUriSha256")
            if not recorded_source_fingerprint:
                recorded_source_uri = str(record.get("sourceUri") or "")
                if recorded_source_uri == source_uri:
                    record["sourceUriSha256"] = source_uri_fingerprint
                    record["sourceUri"] = sanitize_for_report(source_uri)
                    recorded_source_fingerprint = source_uri_fingerprint
            if (
                recorded_source_fingerprint != source_uri_fingerprint
                or str(record.get("inputType") or "").lower() != source_input_type
            ):
                raise ValueError(
                    f"Input {name!r} changed since the migration manifest was "
                    "created. Use a new work_dir."
                )
            foundry_input_type = record.get("foundryInputType")
            if foundry_input_type and isinstance(raw_binding, dict):
                raw_binding["type"] = str(foundry_input_type)
                input_type = str(foundry_input_type).lower()
            if record.get("foundryAssetId"):
                continue
            if (
                self.dataset_transfer_mode == "reference"
                and input_type in _DATA_INPUT_TYPES
            ):
                reference_uri = self._resolve_reference_data_uri(source_uri)
                resolved_reference_uris[name] = reference_uri
                record["referenceDataUri"] = sanitize_for_report(reference_uri)
                record["referenceDataUriSha256"] = _sha256_json(reference_uri)
                self._save()
                continue
            existing_local = record.get("localPath")
            if existing_local and Path(existing_local).exists():
                local_paths[name] = Path(existing_local)
                continue

            model_reference = (
                parse_aml_asset_reference(source_uri, expected_kind="model")
                if input_type in _MODEL_INPUT_TYPES
                else None
            )
            if model_reference is not None:
                try:
                    local_path = self._download_registered_model(
                        model_reference,
                        binding_name=name,
                    )
                    local_paths[name] = local_path
                    record["localPath"] = str(local_path)
                    self._save()
                except Exception as error:
                    warning = (
                        f"Input {name!r}: direct AML model download failed "
                        f"({error}); exporting it through source compute instead."
                    )
                    self.journal.data.setdefault("warnings", []).append(warning)
                    transfer_specs.append(
                        _TransferSpec(
                            binding_name=name,
                            source_uri=source_uri,
                            input_type=input_type,
                            export_name=f"asset_{index}",
                        )
                    )
            else:
                data_reference = parse_aml_asset_reference(
                    source_uri,
                    expected_kind="data",
                )
                if data_reference is not None:
                    try:
                        local_path = self._download_registered_data(
                            data_reference,
                            binding_name=name,
                        )
                        local_paths[name] = local_path
                        record["localPath"] = str(local_path)
                        self._save()
                        continue
                    except Exception as error:
                        warning = (
                            f"Input {name!r}: direct AML data download failed "
                            f"({error}); exporting it through source compute instead."
                        )
                        self.journal.data.setdefault("warnings", []).append(warning)
                transfer_specs.append(
                    _TransferSpec(
                        binding_name=name,
                        source_uri=source_uri,
                        input_type=input_type,
                        export_name=f"asset_{index}",
                    )
                )

        exported = self._export_inputs(transfer_specs)
        for spec in transfer_specs:
            local_path = exported[spec.binding_name]
            local_paths[spec.binding_name] = local_path
            input_state[spec.binding_name]["localPath"] = str(local_path)
        if transfer_specs:
            self._save()

        mappings: dict[str, str] = {}
        for raw_name, raw_binding in source_inputs.items():
            name = str(raw_name)
            binding = dict(raw_binding) if isinstance(raw_binding, Mapping) else {}
            input_type = str(
                binding.get("type")
                or ("literal" if "value" in binding else "uri_folder")
            ).lower()
            if input_type in _LITERAL_INPUT_TYPES:
                continue
            record = input_state[name]
            source_uri = binding.get("path") or binding.get("uri")
            if not source_uri:
                raise ValueError(f"AML input {name!r} has no path/uri.")
            source_uri = str(source_uri)
            existing_id = record.get("foundryAssetId")
            if existing_id:
                mappings[source_uri] = str(existing_id)
                continue
            asset_name = _normalize_asset_name(f"migrated-{source_name}-{name}")
            if (
                self.dataset_transfer_mode == "reference"
                and input_type in _DATA_INPUT_TYPES
            ):
                foundry_input_type = (
                    "uri_folder" if input_type == "mltable" else input_type
                )
                registration_uri = resolved_reference_uris.get(name)
                if not registration_uri:
                    registration_uri = self._resolve_reference_data_uri(source_uri)
                if foundry_input_type == "uri_folder":
                    registration_uri, path_suffix = _folder_reference_root_and_suffix(
                        registration_uri
                    )
                    record["registeredDataUri"] = sanitize_for_report(registration_uri)
                    if path_suffix:
                        record["foundryInputPathSuffix"] = path_suffix
                registered_dataset = register_reference_dataset(
                    registration_uri,
                    dataset_name=asset_name,
                    dataset_version=self.asset_version,
                    dataset_type=foundry_input_type,
                    connection_name=str(self.request.source_storage_connection_name),
                    project_endpoint=self.request.target.project_endpoint,
                    project_name=self.request.target.project_name,
                    description=f"Zero-copy reference from AML job input {name}",
                    tags={"migration.sourceJob": source_name},
                    credential=self.credential,
                )
                foundry_id = registered_dataset.dataset_id
                record["foundryInputType"] = foundry_input_type
                record["transferMode"] = "reference"
                source_binding = source_inputs.get(name)
                if isinstance(source_binding, dict):
                    source_binding["type"] = foundry_input_type
                self.journal.data.setdefault("warnings", []).append(
                    f"Input {name!r}: registered a zero-copy Dataset V3 "
                    f"reference using Foundry connection "
                    f"{self.request.source_storage_connection_name!r}."
                )
            elif input_type in _MODEL_INPUT_TYPES:
                local_path = local_paths[name]
                uploaded_model = upload_and_register_model(
                    local_path,
                    name=asset_name,
                    version=self.asset_version,
                    project_endpoint=self.request.target.project_endpoint,
                    project_name=self.request.target.project_name,
                    api_version=self.request.target.api_version,
                    description=f"Migrated from AML job input {name}",
                    tags={"migration.sourceJob": source_name},
                    blob_prefix="model",
                    credential=self.credential,
                )
                foundry_id = uploaded_model.asset_id
            else:
                local_path = local_paths[name]
                upload_path = (
                    _single_file_under(local_path)
                    if input_type == "uri_file"
                    else local_path
                )
                uploaded_dataset = upload_dataset(
                    upload_path,
                    dataset_name=asset_name,
                    dataset_version=self.asset_version,
                    project_endpoint=self.request.target.project_endpoint,
                    project_name=self.request.target.project_name,
                    connection_name=self.request.target.storage_connection_name,
                    credential=self.credential,
                )
                foundry_id = uploaded_dataset.dataset_id
                if (
                    input_type == "mltable"
                    and uploaded_dataset.dataset_type
                    and str(uploaded_dataset.dataset_type).lower() != "mltable"
                ):
                    foundry_input_type = str(uploaded_dataset.dataset_type).lower()
                    record["foundryInputType"] = foundry_input_type
                    source_binding = source_inputs.get(name)
                    if isinstance(source_binding, dict):
                        source_binding["type"] = foundry_input_type
                    warning = (
                        f"Input {name!r}: Dataset V3 uploaded the MLTable folder as "
                        f"{uploaded_dataset.dataset_type!r}; using that type for the "
                        "Foundry job binding."
                    )
                    self.journal.data.setdefault("warnings", []).append(warning)
            record.update(
                {
                    "assetName": asset_name,
                    "assetVersion": self.asset_version,
                    "foundryAssetId": foundry_id,
                }
            )
            mappings[source_uri] = foundry_id
            self._save()
        return mappings

    def _wait_for_foundry_job(self, job_name: str) -> str:
        deadline = time.monotonic() + self.request.timeout_seconds
        while True:
            token = self._get_foundry_token()
            status_result = get_job_status_by_name(
                job_name,
                access_token=token,
                project_endpoint=self.request.target.project_endpoint,
                project_name=self.request.target.project_name,
                api_version=self.request.target.api_version,
            )
            summary = status_result.summary()
            status = str(summary.get("status") or "")
            self.emit(f"Foundry job {job_name}: {status or 'Unknown'}")
            self.journal.data.setdefault("foundryJob", {}).update(
                {
                    "name": job_name,
                    "status": status or None,
                    "apimRequestId": summary.get("apimRequestId"),
                }
            )
            self._save()
            if status in _FOUNDRY_TERMINAL_STATUSES:
                if status != "Completed":
                    raise RuntimeError(
                        f"Foundry job {job_name} reached terminal status {status}."
                    )
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Foundry job {job_name}.")
            self._sleep(self.request.poll_interval_seconds)

    def _prepare_foundry_attempt(self) -> tuple[int, dict[str, Any]]:
        attempts = self.journal.data.setdefault("foundryJobAttempts", [])
        foundry_state = self.journal.data.setdefault("foundryJob", {})
        status = str(foundry_state.get("status") or "").lower()
        if status in {"failed", "canceled", "cancelled"}:
            if self.request.target_job_name:
                raise RuntimeError(
                    f"Recorded Foundry job {foundry_state.get('name')!r} has "
                    f"status {foundry_state.get('status')!r}. Retry with a new "
                    "target_job_name or a new work_dir."
                )
            archived = deepcopy(foundry_state)
            archived["requestBodyPath"] = self.journal.data.get("requestBodyPath")
            attempts.append(archived)
            foundry_state = {}
            self.journal.data["foundryJob"] = foundry_state
            self._save()
        return len(attempts) + 1, foundry_state

    def migrate(self) -> MigrationResult:
        self.emit(f"Reading AML command job {self.request.source_job_name}.")
        source_entity = self.ml_client.jobs.get(self.request.source_job_name)
        source_job = materialize_aml_command_job(source_entity)
        if str(source_job.get("type") or "").lower() != "command":
            raise ValueError("Only AML command jobs can be migrated.")
        source_name = str(source_job.get("name") or self.request.source_job_name)
        self._record_source_job(source_job)
        self._validate_reference_connection(source_job.get("inputs") or {})
        compatibility_warnings = audit_aml_command_job_compatibility(
            source_job,
            user_assigned_identity_id=(self.request.target.user_assigned_identity_id),
        )
        existing_warnings = list(self.journal.data.get("warnings") or [])
        for warning in compatibility_warnings:
            if warning not in existing_warnings:
                existing_warnings.append(warning)
                self.emit(f"WARNING: {sanitize_for_report(warning)}")
        self.journal.data["warnings"] = existing_warnings
        self._save()

        environment_image = self._resolve_environment_image(
            _safe_attribute(source_entity, "environment")
        )
        self.journal.data["environmentImageReference"] = environment_image
        self._save()

        local_code = self._download_code(
            _safe_attribute(source_entity, "code"),
            source_name,
        )
        code_id = self._upload_code(local_code, source_name)
        asset_mappings = self._prepare_inputs(
            source_job.get("inputs") or {},
            source_name,
        )

        attempt_number, foundry_state = self._prepare_foundry_attempt()
        default_output_prefix = f"migrated-{source_name}-{self.asset_version[-6:]}"
        if attempt_number > 1:
            default_output_prefix += f"-retry{attempt_number}"
        output_prefix = self.request.output_asset_name_prefix or _normalize_asset_name(
            default_output_prefix
        )
        translation = translate_aml_command_job(
            source_job,
            foundry_compute_id=self.request.target.compute_id,
            foundry_instance_type=self.request.target.instance_type,
            environment_image_reference=environment_image,
            migrated_asset_ids=asset_mappings,
            code_id=code_id,
            user_assigned_identity_id=self.request.target.user_assigned_identity_id,
            model_input_path_suffixes={
                str(name): "model"
                for name, binding in dict(source_job.get("inputs") or {}).items()
                if isinstance(binding, Mapping)
                and str(binding.get("type") or "").lower() in _MODEL_INPUT_TYPES
            },
            input_path_suffixes={
                str(name): str(record["foundryInputPathSuffix"])
                for name, record in dict(self.journal.data.get("inputs") or {}).items()
                if isinstance(record, Mapping) and record.get("foundryInputPathSuffix")
            },
            output_asset_name_prefix=output_prefix,
            output_asset_version=self.asset_version,
        )
        if self.request.target.job_tier:
            resources = translation.request_body["properties"]["resources"]
            resource_properties = resources.setdefault("properties", {})
            resource_properties["AISuperComputer"] = {
                "slaTier": str(self.request.target.job_tier),
                "priority": "high",
                "imageVersion": "",
            }
        warnings = list(self.journal.data.get("warnings") or [])
        for warning in translation.warnings:
            if warning not in warnings:
                warnings.append(warning)
                self.emit(f"WARNING: {sanitize_for_report(warning)}")
        self.journal.data["warnings"] = warnings
        self._record_foundry_request(foundry_state, translation.request_body)

        target_job_name = foundry_state.get("name")
        if not target_job_name:
            token = self._get_foundry_token()
            run = submit_job(
                translation.request_body,
                access_token=token,
                job_name=self.request.target_job_name,
                job_name_prefix=(
                    self.request.target_job_name_prefix
                    or _normalize_asset_name(f"migrated-{source_name}")[:55]
                ),
                project_endpoint=self.request.target.project_endpoint,
                project_name=self.request.target.project_name,
                api_version=self.request.target.api_version,
            )
            if run.response.status_code >= 400:
                raise RuntimeError(
                    f"Foundry job submission failed with HTTP "
                    f"{run.response.status_code}: {run.response.text[:1000]}"
                )
            target_job_name = run.job_name
            foundry_state.update(
                {
                    "name": target_job_name,
                    "submissionRequestId": run.request_id,
                    "apimRequestId": run.response.apim_request_id,
                    "statusCode": run.response.status_code,
                }
            )
            self._save()

        target_status = foundry_state.get("status")
        if self.request.wait_for_completion:
            target_status = self._wait_for_foundry_job(str(target_job_name))
        return MigrationResult(
            manifest_path=str(self.journal.path),
            source_job_name=source_name,
            target_job_name=str(target_job_name),
            target_status=str(target_status) if target_status else None,
            request_body=translation.request_body,
            asset_mappings=asset_mappings,
            warnings=tuple(warnings),
        )
