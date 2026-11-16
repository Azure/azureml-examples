"""Create a capability-rich AML command job for migration validation."""

from __future__ import annotations

import hashlib
import json
import time
import concurrent.futures
import contextlib
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aml_command_job_migrator import AmlWorkspace, FoundryTarget, MigrationResult
from .dataset import download_dataset
from .model_asset import download_model_asset


_DEFAULT_FIXTURE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
_TERMINAL_STATUSES = frozenset(
    {"Completed", "Failed", "Canceled", "Cancelled", "NotResponding"}
)
_MAX_UPLOAD_WORKERS = 8


@dataclass(frozen=True)
class FixtureRequest:
    source: AmlWorkspace
    work_dir: str | Path
    environment_image: str = _DEFAULT_FIXTURE_IMAGE
    job_name: str | None = None
    asset_version: str | None = None
    poll_interval_seconds: float = 15.0
    timeout_seconds: float = 3600.0


@dataclass(frozen=True)
class FixtureResult:
    job_name: str
    status: str
    asset_version: str
    data_asset_id: str
    file_asset_id: str
    mltable_asset_id: str
    model_asset_id: str
    work_dir: str
    expected_summary: dict[str, Any]


@dataclass(frozen=True)
class FixtureValidationResult:
    summary: dict[str, Any]
    results_dataset_id: str | None
    summary_dataset_id: str | None
    metrics_table_dataset_id: str | None
    model_asset_id: str
    model_downloaded_files: tuple[str, ...]
    model_total_bytes: int
    validation_dir: str


def _version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:17]


def _status_text(value: Any) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value or "")


def _capture_aml_job_stream(
    ml_client: Any,
    job_name: str,
    target_path: Path,
) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    stream_error: Exception | None = None
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            ml_client.jobs.stream(job_name)
        except Exception as error:
            stream_error = error
    if stream_error is not None:
        buffer.write(f"\n[stream collector raised {type(stream_error).__name__}]\n")
    target_path.write_text(buffer.getvalue(), encoding="utf-8")
    return target_path


def _digest_path(path: Path) -> str:
    files = (
        [path]
        if path.is_file()
        else sorted(entry for entry in path.rglob("*") if entry.is_file())
    )
    digest = hashlib.sha256()
    for entry in files:
        relative_name = (
            entry.relative_to(path).as_posix() if path.is_dir() else entry.name
        )
        digest.update(relative_name.encode("utf-8"))
        digest.update(entry.read_bytes())
    return digest.hexdigest()


def _write_fixture_files(root: Path) -> dict[str, Path]:
    code_dir = root / "code"
    data_dir = root / "training-data"
    table_dir = root / "training-table"
    model_dir = root / "seed-model"
    code_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "part-000.jsonl").write_text(
        '{"id": 1, "text": "alpha"}\n{"id": 2, "text": "beta"}\n',
        encoding="utf-8",
    )
    nested_dir = data_dir / "nested"
    nested_dir.mkdir(exist_ok=True)
    (nested_dir / "part-001.jsonl").write_text(
        '{"id": 3, "text": "gamma"}\n',
        encoding="utf-8",
    )
    config_file = root / "config.json"
    config_file.write_text(
        '{"batch_size": 4, "optimizer": "adamw"}\n',
        encoding="utf-8",
    )
    (table_dir / "records.jsonl").write_text(
        '{"id": 10, "score": 0.75}\n{"id": 11, "score": 0.85}\n',
        encoding="utf-8",
    )
    (table_dir / "MLTable").write_text(
        "paths:\n"
        "  - file: ./records.jsonl\n"
        "transformations:\n"
        "  - read_json_lines:\n"
        "      encoding: utf8\n",
        encoding="utf-8",
    )
    (model_dir / "weights.json").write_text(
        '{"format": "fixture", "weights": [0.1, 0.2, 0.3]}\n',
        encoding="utf-8",
    )
    (model_dir / "metadata.json").write_text(
        '{"framework": "none", "purpose": "migration validation"}\n',
        encoding="utf-8",
    )
    (code_dir / "exercise_job.py").write_text(
        "from __future__ import annotations\n"
        "import argparse\n"
        "import hashlib\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "def digest(path_text: str) -> str:\n"
        "    path = Path(path_text)\n"
        "    files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())\n"
        "    value = hashlib.sha256()\n"
        "    for item in files:\n"
        "        value.update(item.relative_to(path).as_posix().encode() if path.is_dir() else item.name.encode())\n"
        "        value.update(item.read_bytes())\n"
        "    return value.hexdigest()\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--training-data', required=True)\n"
        "parser.add_argument('--config-file', required=True)\n"
        "parser.add_argument('--training-table', required=True)\n"
        "parser.add_argument('--seed-model', required=True)\n"
        "parser.add_argument('--epochs', type=int, required=True)\n"
        "parser.add_argument('--learning-rate', type=float, required=True)\n"
        "parser.add_argument('--message', required=True)\n"
        "parser.add_argument('--enabled', required=True)\n"
        "parser.add_argument('--results', required=True)\n"
        "parser.add_argument('--summary', required=True)\n"
        "parser.add_argument('--metrics-table', required=True)\n"
        "parser.add_argument('--trained-model', required=True)\n"
        "args = parser.parse_args()\n"
        "if os.environ.get('STATIC_SETTING') != 'preserved':\n"
        "    raise RuntimeError('STATIC_SETTING was not preserved')\n"
        "env_data_path = os.environ.get('DATA_FROM_ENV', '')\n"
        "if not env_data_path or '{{' in env_data_path:\n"
        "    raise RuntimeError('DATA_FROM_ENV was not expanded')\n"
        "if not Path(env_data_path).exists():\n"
        "    raise RuntimeError('DATA_FROM_ENV does not reference readable input content')\n"
        "env_data_digest = digest(env_data_path)\n"
        "argument_data_digest = digest(args.training_data)\n"
        "if env_data_digest != argument_data_digest:\n"
        "    raise RuntimeError('templated environment variable content does not match input content')\n"
        "config = json.loads(Path(args.config_file).read_text(encoding='utf-8'))\n"
        "summary = {\n"
        "    'fixture': 'aml-foundry-command-job-migration',\n"
        "    'epochs': args.epochs,\n"
        "    'learningRate': args.learning_rate,\n"
        "    'message': args.message,\n"
        "    'enabled': args.enabled.lower() == 'true',\n"
        "    'config': config,\n"
        "    'trainingDataDigest': argument_data_digest,\n"
        "    'templatedEnvironmentDigest': env_data_digest,\n"
        "    'trainingTableDigest': digest(args.training_table),\n"
        "    'seedModelDigest': digest(args.seed_model),\n"
        "}\n"
        "results = Path(args.results)\n"
        "results.mkdir(parents=True, exist_ok=True)\n"
        "(results / 'result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')\n"
        "summary_path = Path(args.summary)\n"
        "summary_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "summary_path.write_text(json.dumps(summary, sort_keys=True), encoding='utf-8')\n"
        "metrics_table = Path(args.metrics_table)\n"
        "metrics_table.mkdir(parents=True, exist_ok=True)\n"
        "(metrics_table / 'metrics.jsonl').write_text(json.dumps(summary, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "(metrics_table / 'MLTable').write_text('paths:\\n  - file: ./metrics.jsonl\\ntransformations:\\n  - read_json_lines:\\n      encoding: utf8\\n', encoding='utf-8')\n"
        "trained_model = Path(args.trained_model)\n"
        "trained_model.mkdir(parents=True, exist_ok=True)\n"
        "(trained_model / 'model.json').write_text(json.dumps({'source': summary['seedModelDigest'], 'epochs': args.epochs}), encoding='utf-8')\n"
        "print('MIGRATION_FIXTURE_COMPLETED')\n"
        "print('MIGRATION_FIXTURE_RECORD:' + json.dumps(summary, sort_keys=True))\n",
        encoding="utf-8",
    )
    return {
        "code": code_dir,
        "data": data_dir,
        "file": config_file,
        "table": table_dir,
        "model": model_dir,
    }


def _upload_fixture_paths_with_identity(
    paths: dict[str, Path],
    *,
    version: str,
    ml_client: Any,
    credential: Any,
    datastore_name: str = "workspaceblobstore",
) -> dict[str, str]:
    """Upload fixture bytes to workspaceblobstore using an Entra ID token."""

    from azure.storage.blob import BlobServiceClient

    datastore = ml_client.datastores.get(datastore_name)
    account_name = str(getattr(datastore, "account_name", None) or "")
    container_name = str(getattr(datastore, "container_name", None) or "")
    endpoint = str(getattr(datastore, "endpoint", None) or "core.windows.net")
    if not account_name or not container_name:
        raise RuntimeError(
            "workspaceblobstore did not expose account_name/container_name."
        )
    account_url = f"https://{account_name}.blob.{endpoint}"
    service_client = BlobServiceClient(account_url, credential=credential)
    container_client = service_client.get_container_client(container_name)
    root_prefix = f"aml-foundry-command-migration/{version}"
    upload_items: list[tuple[Path, str]] = []
    uris: dict[str, str] = {}
    for kind, source_path in paths.items():
        prefix = f"{root_prefix}/{kind}"
        if source_path.is_file():
            blob_name = f"{prefix}/{source_path.name}"
            upload_items.append((source_path, blob_name))
            uri_path = blob_name
        else:
            files = sorted(entry for entry in source_path.rglob("*") if entry.is_file())
            if not files:
                raise ValueError(f"Fixture path contains no files: {source_path}")
            upload_items.extend(
                (entry, f"{prefix}/{entry.relative_to(source_path).as_posix()}")
                for entry in files
            )
            uri_path = prefix
        if kind == "code":
            uris[kind] = f"{account_url}/{container_name}/{uri_path}"
        else:
            uris[kind] = f"azureml://datastores/{datastore_name}/paths/{uri_path}"

    def upload(item: tuple[Path, str]) -> None:
        source_path, blob_name = item
        with source_path.open("rb") as file_handle:
            container_client.upload_blob(
                name=blob_name,
                data=file_handle,
                overwrite=True,
            )

    try:
        worker_count = min(_MAX_UPLOAD_WORKERS, len(upload_items))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            for future in (executor.submit(upload, item) for item in upload_items):
                future.result()
    finally:
        close_container = getattr(container_client, "close", None)
        if callable(close_container):
            close_container()
        service_client.close()
    return uris


def _ensure_identity_datastore(
    ml_client: Any,
    *,
    name: str,
) -> Any:
    """Create/reuse a keyless datastore over the workspace default container."""

    from azure.ai.ml.entities import AzureBlobDatastore
    from azure.core.exceptions import ResourceNotFoundError

    source = ml_client.datastores.get("workspaceblobstore")
    try:
        existing = ml_client.datastores.get(name)
    except ResourceNotFoundError:
        existing = None
    if existing is not None:
        expected = (
            str(getattr(source, "account_name", "")).lower(),
            str(getattr(source, "container_name", "")).lower(),
        )
        actual = (
            str(getattr(existing, "account_name", "")).lower(),
            str(getattr(existing, "container_name", "")).lower(),
        )
        credential_class = type(getattr(existing, "credentials", None)).__name__
        if actual != expected or credential_class not in {
            "NoneType",
            "NoneCredentialConfiguration",
        }:
            raise ValueError(
                f"AML datastore {name!r} exists but is not a credentialless "
                "alias of workspaceblobstore. Choose another identity datastore name."
            )
        return existing

    return ml_client.datastores.create_or_update(
        AzureBlobDatastore(
            name=name,
            account_name=str(source.account_name),
            container_name=str(source.container_name),
            endpoint=str(getattr(source, "endpoint", None) or "core.windows.net"),
            protocol=str(getattr(source, "protocol", None) or "https"),
            credentials=None,
            description=(
                "Credentialless workspace storage alias for AML-to-Foundry "
                "command-job migration."
            ),
        )
    )


def _create_protected_aml_code_asset(
    local_path: Path,
    *,
    name: str,
    version: str,
    description: str,
    ml_client: Any,
    credential: Any,
) -> Any:
    """Upload and register an AML-protected code snapshot without shared keys."""

    from azure.ai.ml._restclient.arm_ml_service.models import PendingUploadRequestDto
    from azure.ai.ml._utils._asset_utils import get_upload_files_from_folder
    from azure.ai.ml.entities._assets import Code
    from azure.storage.blob import ContainerClient

    source_path = Path(local_path).expanduser().resolve()
    code = Code(
        name=name,
        version=version,
        path=str(source_path),
        description=description,
    )
    pending = ml_client._code._service_client.code_versions.create_or_get_start_pending_upload(
        resource_group_name=ml_client._code._resource_group_name,
        workspace_name=ml_client._code._workspace_name,
        name=name,
        version=version,
        body=PendingUploadRequestDto(pending_upload_type="TemporaryBlobReference"),
    )
    blob_reference = getattr(pending, "blob_reference_for_consumption", None)
    container_uri = str(getattr(blob_reference, "blob_uri", None) or "").rstrip("/")
    if not container_uri:
        raise RuntimeError("AML code pending upload returned no blob URI.")

    source_name = source_path.name
    upload_files = sorted(
        get_upload_files_from_folder(
            source_path,
            prefix=source_name,
            ignore_file=code._ignore_file,
        )
    )
    if not upload_files:
        raise ValueError(f"Code path contains no uploadable files: {source_path}")
    container_client = ContainerClient.from_container_url(
        container_url=container_uri,
        credential=credential,
    )
    try:
        for file_path, blob_name in upload_files:
            with Path(file_path).open("rb") as file_handle:
                container_client.upload_blob(
                    name=str(blob_name),
                    data=file_handle,
                    overwrite=True,
                )
        indicator_blob = str(upload_files[0][1])
        container_client.get_blob_client(indicator_blob).set_blob_metadata(
            {
                "upload_status": "completed",
                "name": name,
                "version": version,
            }
        )
    finally:
        close = getattr(container_client, "close", None)
        if callable(close):
            close()

    code._path = f"{container_uri}/{source_name}"
    return ml_client._code.create_or_update(code)


def create_aml_migration_fixture(
    request: FixtureRequest,
    *,
    credential: Any | None = None,
    ml_client: Any | None = None,
    emit: Any = print,
    sleeper: Any = time.sleep,
) -> FixtureResult:
    """Register source assets, run the AML fixture, and wait for completion."""

    if request.poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero.")
    if request.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero.")
    if credential is None:
        from azure.identity import AzureCliCredential

        credential = AzureCliCredential()
    if ml_client is None:
        from azure.ai.ml import MLClient

        ml_client = MLClient(
            credential,
            request.source.subscription_id,
            request.source.resource_group,
            request.source.workspace_name,
        )

    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import (
        Data,
        Environment,
        Model,
        UserIdentityConfiguration,
    )

    root = Path(request.work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = _write_fixture_files(root)
    expected_summary = {
        "fixture": "aml-foundry-command-job-migration",
        "epochs": 3,
        "learningRate": 0.125,
        "message": "resurrected-command-job",
        "enabled": True,
        "config": {"batch_size": 4, "optimizer": "adamw"},
        "trainingDataDigest": _digest_path(paths["data"]),
        "templatedEnvironmentDigest": _digest_path(paths["data"]),
        "trainingTableDigest": _digest_path(paths["table"]),
        "seedModelDigest": _digest_path(paths["model"]),
    }
    version = request.asset_version or _version()
    identity_datastore = _ensure_identity_datastore(
        ml_client,
        name=request.source.identity_datastore_name,
    )
    uploaded_paths = _upload_fixture_paths_with_identity(
        paths,
        version=version,
        ml_client=ml_client,
        credential=credential,
        datastore_name=str(identity_datastore.name),
    )
    suffix = version[-10:]
    data_name = f"foundry-migration-data-{suffix}"
    file_name = f"foundry-migration-config-{suffix}"
    table_name = f"foundry-migration-table-{suffix}"
    model_name = f"foundry-migration-model-{suffix}"
    code_name = f"foundry-migration-code-{suffix}"

    code_asset = _create_protected_aml_code_asset(
        paths["code"],
        name=code_name,
        version="1",
        description="Code snapshot for AML-to-Foundry migration validation.",
        ml_client=ml_client,
        credential=credential,
    )

    data_asset = ml_client.data.create_or_update(
        Data(
            name=data_name,
            version=version,
            type=AssetTypes.URI_FOLDER,
            path=uploaded_paths["data"],
            description="Folder input for AML-to-Foundry migration validation.",
        )
    )
    file_asset = ml_client.data.create_or_update(
        Data(
            name=file_name,
            version=version,
            type=AssetTypes.URI_FILE,
            path=uploaded_paths["file"],
            description="File input for AML-to-Foundry migration validation.",
        )
    )
    table_asset = ml_client.data.create_or_update(
        Data(
            name=table_name,
            version=version,
            type=AssetTypes.MLTABLE,
            path=uploaded_paths["table"],
            description="MLTable input for AML-to-Foundry migration validation.",
        )
    )
    model_asset = ml_client.models.create_or_update(
        Model(
            name=model_name,
            version="1",
            type=AssetTypes.CUSTOM_MODEL,
            path=uploaded_paths["model"],
            description="Model input for AML-to-Foundry migration validation.",
        )
    )
    job_name = request.job_name or f"aml-foundry-migration-{suffix}"
    output_root = (
        f"azureml://datastores/{request.source.identity_datastore_name}/paths/"
        f"aml-foundry-command-migration/{version}/outputs"
    )
    source_job = command(
        name=job_name,
        display_name="AML command job migration capability exercise",
        description=(
            "Exercises registered data/file/MLTable/model inputs, all primitive inputs, "
            "folder/file/MLTable/model outputs, code, environment variables, and limits."
        ),
        code=str(code_asset.id),
        command=(
            'export DATA_FROM_ENV="${{inputs.training_data}}" && '
            "python exercise_job.py "
            '--training-data "${{inputs.training_data}}" '
            '--config-file "${{inputs.config_file}}" '
            '--training-table "${{inputs.training_table}}" '
            '--seed-model "${{inputs.seed_model}}" '
            "--epochs ${{inputs.epochs}} "
            "--learning-rate ${{inputs.learning_rate}} "
            '--message "${{inputs.message}}" '
            "--enabled ${{inputs.enabled}} "
            '--results "${{outputs.results}}" '
            '--summary "${{outputs.summary}}" '
            '--metrics-table "${{outputs.metrics_table}}" '
            '--trained-model "${{outputs.trained_model}}"'
        ),
        environment=Environment(image=request.environment_image),
        compute=request.source.export_compute,
        inputs={
            "epochs": 3,
            "learning_rate": 0.125,
            "message": "resurrected-command-job",
            "enabled": True,
            "training_data": Input(
                type=AssetTypes.URI_FOLDER,
                path=str(data_asset.id),
                mode="download",
            ),
            "config_file": Input(
                type=AssetTypes.URI_FILE,
                path=str(file_asset.id),
                mode="download",
            ),
            "training_table": Input(
                type=AssetTypes.MLTABLE,
                path=str(table_asset.id),
                mode="download",
            ),
            "seed_model": Input(
                type=AssetTypes.CUSTOM_MODEL,
                path=str(model_asset.id),
                mode="download",
            ),
        },
        outputs={
            "results": Output(
                type=AssetTypes.URI_FOLDER,
                path=f"{output_root}/results",
                mode="upload",
            ),
            "summary": Output(
                type=AssetTypes.URI_FILE,
                path=f"{output_root}/summary/summary.json",
                mode="upload",
            ),
            "metrics_table": Output(
                type=AssetTypes.MLTABLE,
                path=f"{output_root}/metrics_table",
                mode="upload",
            ),
            "trained_model": Output(
                type=AssetTypes.CUSTOM_MODEL,
                path=f"{output_root}/trained_model",
                mode="upload",
            ),
        },
        identity=UserIdentityConfiguration(),
        environment_variables={
            "STATIC_SETTING": "preserved",
            "DATA_FROM_ENV": "${{inputs.training_data}}",
        },
        instance_count=1,
        shm_size="2g",
        timeout=int(request.timeout_seconds),
        experiment_name="aml-foundry-command-job-migration",
        tags={
            "scenario": "aml-foundry-command-job-migration",
            "capabilityCoverage": "data-file-mltable-model-code-outputs",
        },
    )
    created_job = ml_client.jobs.create_or_update(source_job)
    created_name = str(getattr(created_job, "name", None) or job_name)
    deadline = time.monotonic() + request.timeout_seconds
    while True:
        current_job = ml_client.jobs.get(created_name)
        status = _status_text(getattr(current_job, "status", None))
        emit(f"AML fixture job {created_name}: {status or 'Unknown'}")
        if status in _TERMINAL_STATUSES:
            if status != "Completed":
                stream_path = _capture_aml_job_stream(
                    ml_client,
                    created_name,
                    root / "failed-job-stream.log",
                )
                raise RuntimeError(
                    f"AML fixture job {created_name} reached terminal status "
                    f"{status}. Stream diagnostics: {stream_path}"
                )
            return FixtureResult(
                job_name=created_name,
                status=status,
                asset_version=version,
                data_asset_id=str(data_asset.id),
                file_asset_id=str(file_asset.id),
                mltable_asset_id=str(table_asset.id),
                model_asset_id=str(model_asset.id),
                work_dir=str(root),
                expected_summary=expected_summary,
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for AML fixture job {created_name}.")
        sleeper(request.poll_interval_seconds)


def validate_migrated_fixture(
    migration: MigrationResult,
    *,
    target: FoundryTarget,
    work_dir: str | Path,
    credential: Any | None = None,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 10.0,
    emit: Any = print,
    sleeper: Any = time.sleep,
    expected_summary: dict[str, Any] | None = None,
) -> FixtureValidationResult:
    """Validate migrated dataset outputs and the registered model output."""

    if migration.target_status != "Completed":
        raise RuntimeError(
            f"Cannot validate fixture outputs while target status is "
            f"{migration.target_status!r}."
        )
    if timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("Validation timeout and poll interval must be positive.")
    if credential is None:
        from azure.identity import AzureCliCredential

        credential = AzureCliCredential()

    outputs = migration.request_body.get("properties", {}).get("outputs", {})
    required = {"results", "summary", "metrics_table", "trained_model"}
    missing = required.difference(outputs)
    if missing:
        raise ValueError(
            f"Migrated fixture request is missing outputs: {sorted(missing)}"
        )
    validation_root = Path(work_dir).expanduser().resolve()
    validation_root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds

    downloaded = {}
    for output_name in ("results", "summary", "metrics_table"):
        output = outputs[output_name]
        asset_name = str(output.get("assetName") or "")
        asset_version = str(output.get("assetVersion") or "")
        if not asset_name or not asset_version:
            raise ValueError(f"Output {output_name!r} has no assetName/assetVersion.")
        target_dir = validation_root / output_name
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                downloaded[output_name] = download_dataset(
                    target_dir,
                    dataset_name=asset_name,
                    dataset_version=asset_version,
                    project_endpoint=target.project_endpoint,
                    project_name=target.project_name,
                    credential=credential,
                    overwrite=True,
                )
                break
            except Exception as error:
                last_error = error
                emit(
                    f"Waiting for Foundry output {asset_name}:{asset_version}: "
                    f"{error}"
                )
                sleeper(poll_interval_seconds)
        else:
            raise TimeoutError(
                f"Timed out downloading Foundry output {asset_name}:{asset_version}."
            ) from last_error

    result_files = [
        path
        for path in (validation_root / "results").rglob("result.json")
        if path.is_file()
    ]
    summary_files = [
        path for path in (validation_root / "summary").rglob("*") if path.is_file()
    ]
    metrics_files = [
        path
        for path in (validation_root / "metrics_table").rglob("metrics.jsonl")
        if path.is_file()
    ]
    mltable_files = [
        path
        for path in (validation_root / "metrics_table").rglob("MLTable")
        if path.is_file()
    ]
    metrics_dataset_type = str(downloaded["metrics_table"].dataset_type).lower()
    if metrics_dataset_type != "uri_folder":
        raise RuntimeError(
            "Adapted MLTable output must register as a Foundry uri_folder, "
            f"got {metrics_dataset_type!r}."
        )
    if (
        len(result_files) != 1
        or len(summary_files) != 1
        or len(metrics_files) != 1
        or len(mltable_files) != 1
    ):
        raise RuntimeError(
            "Expected one result.json, one summary output file, one "
            "metrics.jsonl, and one MLTable definition, found "
            f"{len(result_files)}, {len(summary_files)}, {len(metrics_files)}, "
            f"and {len(mltable_files)}."
        )
    result_summary = json.loads(result_files[0].read_text(encoding="utf-8"))
    file_summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
    metrics_summary = json.loads(metrics_files[0].read_text(encoding="utf-8"))
    if result_summary != file_summary or result_summary != metrics_summary:
        raise RuntimeError(
            "Folder, file, and MLTable outputs contain different summaries."
        )
    expected = expected_summary or {
        "fixture": "aml-foundry-command-job-migration",
        "epochs": 3,
        "learningRate": 0.125,
        "message": "resurrected-command-job",
        "enabled": True,
    }
    mismatches = {
        key: {"expected": value, "actual": result_summary.get(key)}
        for key, value in expected.items()
        if result_summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Migrated fixture output mismatch: {mismatches}")

    model_output = outputs["trained_model"]
    model_name = str(model_output.get("assetName") or "")
    model_version = str(model_output.get("assetVersion") or "")
    if not model_name or not model_version:
        raise ValueError("trained_model output has no assetName/assetVersion.")
    model_download = None
    last_model_error: Exception | None = None
    model_dir = validation_root / "trained_model"
    while time.monotonic() < deadline:
        try:
            model_download = download_model_asset(
                model_dir,
                name=model_name,
                version=model_version,
                project_endpoint=target.project_endpoint,
                project_name=target.project_name,
                api_version=target.api_version,
                credential=credential,
                overwrite=True,
            )
            break
        except Exception as error:
            last_model_error = error
            emit(
                f"Waiting for Foundry model output {model_name}:{model_version}: "
                f"{error}"
            )
            sleeper(poll_interval_seconds)
    else:
        raise TimeoutError(
            f"Timed out downloading Foundry model output {model_name}:{model_version}."
        ) from last_model_error
    if model_download is None:
        raise RuntimeError("Model output download completed without a result.")
    model_files = [path for path in model_dir.rglob("model.json") if path.is_file()]
    if len(model_files) != 1:
        raise RuntimeError(
            f"Expected one model.json in the model output, found {len(model_files)}."
        )
    output_model = json.loads(model_files[0].read_text(encoding="utf-8"))
    expected_model = {
        "source": result_summary.get("seedModelDigest"),
        "epochs": result_summary.get("epochs"),
    }
    if output_model != expected_model:
        raise RuntimeError(
            f"Migrated model output mismatch: expected {expected_model}, got "
            f"{output_model}."
        )

    return FixtureValidationResult(
        summary=result_summary,
        results_dataset_id=downloaded["results"].dataset_id,
        summary_dataset_id=downloaded["summary"].dataset_id,
        metrics_table_dataset_id=downloaded["metrics_table"].dataset_id,
        model_asset_id=model_download.asset_id,
        model_downloaded_files=model_download.downloaded_files,
        model_total_bytes=model_download.total_bytes,
        validation_dir=str(validation_root),
    )
