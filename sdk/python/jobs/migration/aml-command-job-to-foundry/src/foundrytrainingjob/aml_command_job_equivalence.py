"""Collect and compare AML/Foundry command-job logs and outputs."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlsplit

from .aml_command_job_fixture import (
    FixtureRequest,
    FixtureResult,
    FixtureValidationResult,
    create_aml_migration_fixture,
    validate_migrated_fixture,
)
from .aml_command_job_migration import create_foundry_asset_version
from .aml_command_job_migrator import (
    AmlCommandJobMigrator,
    AmlWorkspace,
    FoundryTarget,
    MigrationRequest,
    MigrationResult,
    materialize_aml_command_job,
)
from .artifacts import get_job_artifact_content, list_job_artifacts
from .auth import get_foundry_access_token
from .e2e.sanitization import sanitize_for_report


APPLICATION_LOG_PREFIX = "MIGRATION_FIXTURE_RECORD:"
_COMPLETION_MARKER = "MIGRATION_FIXTURE_COMPLETED"
_SUBSCRIPTION_PATTERN = re.compile(r"/subscriptions/([^/]+)", re.IGNORECASE)


@dataclass(frozen=True)
class JobEvidence:
    service: str
    job_name: str
    status: str
    log_root: str
    output_root: str
    log_files: tuple[str, ...]
    output_files: tuple[str, ...]
    application_records: tuple[dict[str, Any], ...]
    output_values: dict[str, Any]
    log_digests: dict[str, str] = field(default_factory=dict)
    output_digests: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["log_files"] = list(self.log_files)
        result["output_files"] = list(self.output_files)
        result["application_records"] = list(self.application_records)
        return result


@dataclass(frozen=True)
class JobEquivalenceReport:
    source_job_name: str
    target_job_name: str
    logs_equivalent: bool
    outputs_equivalent: bool
    equivalent: bool
    source_log_record_digests: tuple[str, ...]
    target_log_record_digests: tuple[str, ...]
    source_output_digests: dict[str, str]
    target_output_digests: dict[str, str]
    log_mismatches: tuple[str, ...] = field(default_factory=tuple)
    output_mismatches: tuple[str, ...] = field(default_factory=tuple)
    source_log_root: str = ""
    target_log_root: str = ""
    source_output_root: str = ""
    target_output_root: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "source_log_record_digests",
            "target_log_record_digests",
            "log_mismatches",
            "output_mismatches",
        ):
            result[key] = list(result[key])
        return result

    def assert_equivalent(self) -> None:
        if self.equivalent:
            return
        details = [*self.log_mismatches, *self.output_mismatches]
        raise AssertionError(
            "AML and Foundry job evidence differs: " + "; ".join(details)
        )


@dataclass(frozen=True)
class MigrationEquivalenceRequest:
    source: AmlWorkspace
    target: FoundryTarget
    work_dir: str | Path
    environment_image: str = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
    export_environment_image: str = (
        "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
    )
    fixture_job_name: str | None = None
    target_job_name: str | None = None
    asset_version: str | None = None
    dataset_transfer_mode: str = "upload"
    source_storage_connection_name: str | None = None
    poll_interval_seconds: float = 15.0
    timeout_seconds: float = 3600.0
    validation_timeout_seconds: float = 600.0
    existing_source_job_name: str | None = None
    existing_source_asset_version: str | None = None


@dataclass(frozen=True)
class MigrationEquivalenceResult:
    fixture: FixtureResult
    migration: MigrationResult
    validation: FixtureValidationResult
    source_evidence: JobEvidence
    target_evidence: JobEvidence
    equivalence: JobEquivalenceReport

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_report(
            {
                "fixture": asdict(self.fixture),
                "migration": {
                    "manifestPath": self.migration.manifest_path,
                    "sourceJobName": self.migration.source_job_name,
                    "targetJobName": self.migration.target_job_name,
                    "targetStatus": self.migration.target_status,
                    "assetMappings": self.migration.asset_mappings,
                    "warnings": list(self.migration.warnings),
                },
                "validation": asdict(self.validation),
                "sourceEvidence": self.source_evidence.to_dict(),
                "targetEvidence": self.target_evidence.to_dict(),
                "equivalence": self.equivalence.to_dict(),
            }
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_application_records(texts: Iterable[str]) -> tuple[dict[str, Any], ...]:
    """Extract deduplicated canonical application records from wrapped logs."""

    records: dict[str, dict[str, Any]] = {}
    for text in texts:
        for raw_line in str(text).splitlines():
            marker_index = raw_line.find(APPLICATION_LOG_PREFIX)
            if marker_index < 0:
                continue
            payload = raw_line[marker_index + len(APPLICATION_LOG_PREFIX) :].strip()
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed {APPLICATION_LOG_PREFIX} record: {payload!r}"
                ) from error
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"{APPLICATION_LOG_PREFIX} payload must be a JSON object."
                )
            records[_canonical_json(parsed)] = parsed
    return tuple(records[key] for key in sorted(records))


def _load_json_output(
    root: Path, relative_candidates: tuple[str, ...]
) -> tuple[Any, str]:
    for relative_candidate in relative_candidates:
        exact_path = root / relative_candidate
        if exact_path.is_file():
            return json.loads(exact_path.read_text(encoding="utf-8")), str(
                exact_path.relative_to(root)
            )

    file_names = {Path(candidate).name for candidate in relative_candidates}
    matches = sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file() and path.name in file_names
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one of {relative_candidates!r} under {root}, found "
            f"{[str(path.relative_to(root)) for path in matches]}."
        )
    return json.loads(matches[0].read_text(encoding="utf-8")), str(
        matches[0].relative_to(root)
    )


def snapshot_fixture_outputs(
    root: str | Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Load the logical JSON outputs independent of service path layout."""

    output_root = Path(root).expanduser().resolve()
    results, results_path = _load_json_output(
        output_root,
        ("results/result.json", "result.json"),
    )
    summary, summary_path = _load_json_output(
        output_root,
        ("summary/summary.json", "summary.json"),
    )
    metrics_table, metrics_table_path = _load_json_output(
        output_root,
        ("metrics_table/metrics.jsonl", "metrics.jsonl"),
    )
    model, model_path = _load_json_output(
        output_root,
        ("trained_model/model.json", "model.json"),
    )
    return (
        {
            "results": results,
            "summary": summary,
            "metrics_table": metrics_table,
            "trained_model": model,
        },
        tuple(sorted((results_path, summary_path, metrics_table_path, model_path))),
    )


def _download_blob_prefix_with_identity(
    *,
    account_url: str,
    container_name: str,
    prefix: str,
    target_dir: Path,
    credential: Any,
) -> tuple[str, ...]:
    from azure.storage.blob import BlobServiceClient

    target_dir.mkdir(parents=True, exist_ok=True)
    service_client = BlobServiceClient(account_url, credential=credential)
    container_client = service_client.get_container_client(container_name)
    downloaded: list[str] = []
    try:
        for raw_blob_name in container_client.list_blob_names(
            name_starts_with=prefix.rstrip("/") + "/"
        ):
            blob_name = str(raw_blob_name)
            relative_name = blob_name[len(prefix.rstrip("/")) + 1 :]
            if not relative_name:
                continue
            local_path = (target_dir / relative_name).resolve()
            try:
                local_path.relative_to(target_dir)
            except ValueError as error:
                raise ValueError(
                    f"Refusing to download blob outside evidence root: {blob_name!r}."
                ) from error
            local_path.parent.mkdir(parents=True, exist_ok=True)
            stream = container_client.get_blob_client(blob_name).download_blob()
            with local_path.open("wb") as file_handle:
                stream.readinto(file_handle)
            downloaded.append(str(local_path.relative_to(target_dir)))
    finally:
        close_container = getattr(container_client, "close", None)
        if callable(close_container):
            close_container()
        service_client.close()
    return tuple(sorted(downloaded))


def _datastore_blob_coordinates(
    ml_client: Any,
    datastore_name: str,
) -> tuple[str, str]:
    datastore = ml_client.datastores.get(datastore_name)
    account_name = str(getattr(datastore, "account_name", None) or "")
    container_name = str(getattr(datastore, "container_name", None) or "")
    endpoint = str(getattr(datastore, "endpoint", None) or "core.windows.net")
    if not account_name or not container_name:
        raise RuntimeError(
            f"AML datastore {datastore_name!r} did not expose "
            "account_name/container_name."
        )
    return f"https://{account_name}.blob.{endpoint}", container_name


def _workspace_blob_coordinates(ml_client: Any) -> tuple[str, str]:
    return _datastore_blob_coordinates(ml_client, "workspaceblobstore")


def collect_aml_fixture_evidence(
    fixture: FixtureResult,
    *,
    ml_client: Any,
    credential: Any,
    work_dir: str | Path,
) -> JobEvidence:
    """Capture AML stream logs and deterministic source outputs."""

    evidence_root = Path(work_dir).expanduser().resolve()
    log_root = evidence_root / "logs"
    output_root = evidence_root / "outputs"
    log_root.mkdir(parents=True, exist_ok=True)

    stream_buffer = io.StringIO()
    with contextlib.redirect_stdout(stream_buffer), contextlib.redirect_stderr(
        stream_buffer
    ):
        ml_client.jobs.stream(fixture.job_name)
    stream_text = stream_buffer.getvalue()
    stream_path = log_root / "aml-job-stream.log"
    stream_path.write_text(stream_text, encoding="utf-8")

    artifact_account_url, artifact_container_name = _datastore_blob_coordinates(
        ml_client,
        "workspaceartifactstore",
    )
    user_log_root = log_root / "user_logs"
    user_log_files = _download_blob_prefix_with_identity(
        account_url=artifact_account_url,
        container_name=artifact_container_name,
        prefix=f"ExperimentRun/dcid.{fixture.job_name}/user_logs",
        target_dir=user_log_root,
        credential=credential,
    )
    if not user_log_files:
        raise RuntimeError(
            f"AML fixture {fixture.job_name} produced no user_logs artifacts."
        )
    user_log_texts = [
        (user_log_root / relative_path).read_text(
            encoding="utf-8",
            errors="replace",
        )
        for relative_path in user_log_files
    ]

    account_url, container_name = _workspace_blob_coordinates(ml_client)
    output_prefix = f"aml-foundry-command-migration/{fixture.asset_version}/outputs"
    output_files = _download_blob_prefix_with_identity(
        account_url=account_url,
        container_name=container_name,
        prefix=output_prefix,
        target_dir=output_root,
        credential=credential,
    )
    if not output_files:
        raise RuntimeError(
            f"AML fixture {fixture.job_name} produced no blobs under {output_prefix}."
        )
    output_values, logical_files = snapshot_fixture_outputs(output_root)
    all_log_texts = (stream_text, *user_log_texts)
    records = extract_application_records(all_log_texts)
    if _COMPLETION_MARKER not in "\n".join(all_log_texts):
        raise RuntimeError(f"AML logs did not contain {_COMPLETION_MARKER!r}.")
    if not records:
        raise RuntimeError(
            f"AML job stream contained no {APPLICATION_LOG_PREFIX} records."
        )
    return JobEvidence(
        service="AML",
        job_name=fixture.job_name,
        status=fixture.status,
        log_root=str(log_root),
        output_root=str(output_root),
        log_files=(
            stream_path.name,
            *(f"user_logs/{relative_path}" for relative_path in user_log_files),
        ),
        output_files=logical_files,
        application_records=records,
        output_values=output_values,
        log_digests={
            stream_path.name: _file_digest(stream_path),
            **{
                f"user_logs/{relative_path}": _file_digest(
                    user_log_root / relative_path
                )
                for relative_path in user_log_files
            },
        },
        output_digests={
            name: _sha256_text(_canonical_json(value))
            for name, value in output_values.items()
        },
    )


def _target_subscription_id(target: FoundryTarget) -> str:
    match = _SUBSCRIPTION_PATTERN.search(target.compute_id)
    if match:
        return match.group(1)
    raise ValueError(
        "Could not derive a subscription ID from FoundryTarget.compute_id."
    )


def collect_foundry_fixture_evidence(
    migration: MigrationResult,
    validation: FixtureValidationResult,
    *,
    target: FoundryTarget,
    credential: Any,
    work_dir: str | Path,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 10.0,
    emit: Any = print,
    sleeper: Any = time.sleep,
) -> JobEvidence:
    """Capture Foundry user logs and already-downloaded target outputs."""

    if migration.target_status != "Completed":
        raise RuntimeError(
            f"Foundry job status is {migration.target_status!r}, not 'Completed'."
        )
    evidence_root = Path(work_dir).expanduser().resolve()
    log_root = evidence_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(validation.validation_dir).expanduser().resolve()
    deadline = time.monotonic() + timeout_seconds
    subscription_id = _target_subscription_id(target)
    last_problem = "artifact list not attempted"
    user_log_paths: list[str] = []
    while time.monotonic() < deadline:
        token = get_foundry_access_token(credential=credential).token
        artifact_list = list_job_artifacts(
            migration.target_job_name,
            project_endpoint=target.project_endpoint,
            project_name=target.project_name,
            api_version=target.api_version,
            access_token=token,
            path_prefix="user_logs/",
            subscription_id=subscription_id,
        )
        if artifact_list.response.status_code == 200:
            user_log_paths = sorted(
                str(item.get("path"))
                for item in artifact_list.artifacts
                if isinstance(item, Mapping)
                and str(item.get("path", "")).startswith("user_logs/")
            )
            if user_log_paths:
                break
            last_problem = "no user_logs artifacts were listed"
        else:
            last_problem = (
                f"artifact list returned HTTP {artifact_list.response.status_code}"
            )
        emit(f"Waiting for Foundry user logs: {last_problem}.")
        sleeper(poll_interval_seconds)
    else:
        raise TimeoutError(f"Timed out resolving Foundry user logs: {last_problem}.")

    log_texts: list[str] = []
    log_files: list[str] = []
    log_digests: dict[str, str] = {}
    token = get_foundry_access_token(credential=credential).token
    for artifact_path in user_log_paths:
        content = get_job_artifact_content(
            migration.target_job_name,
            artifact_path,
            project_endpoint=target.project_endpoint,
            project_name=target.project_name,
            api_version=target.api_version,
            access_token=token,
            subscription_id=subscription_id,
        ).content
        relative = Path(*artifact_path.split("/"))
        local_path = (log_root / relative).resolve()
        try:
            local_path.relative_to(log_root)
        except ValueError as error:
            raise ValueError(
                f"Refusing to persist artifact outside log root: {artifact_path!r}."
            ) from error
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")
        relative_text = str(local_path.relative_to(log_root))
        log_files.append(relative_text)
        log_digests[relative_text] = _file_digest(local_path)
        log_texts.append(content)

    records = extract_application_records(log_texts)
    combined_logs = "\n".join(log_texts)
    if _COMPLETION_MARKER not in combined_logs:
        raise RuntimeError(f"Foundry user logs did not contain {_COMPLETION_MARKER!r}.")
    if not records:
        raise RuntimeError(
            f"Foundry user logs contained no {APPLICATION_LOG_PREFIX} records."
        )
    output_values, output_files = snapshot_fixture_outputs(output_root)
    return JobEvidence(
        service="Foundry",
        job_name=migration.target_job_name,
        status=str(migration.target_status),
        log_root=str(log_root),
        output_root=str(output_root),
        log_files=tuple(sorted(log_files)),
        output_files=output_files,
        application_records=records,
        output_values=output_values,
        log_digests=log_digests,
        output_digests={
            name: _sha256_text(_canonical_json(value))
            for name, value in output_values.items()
        },
    )


def compare_job_evidence(
    source: JobEvidence,
    target: JobEvidence,
) -> JobEquivalenceReport:
    source_records = tuple(
        _canonical_json(value) for value in source.application_records
    )
    target_records = tuple(
        _canonical_json(value) for value in target.application_records
    )
    log_mismatches: list[str] = []
    if source_records != target_records:
        log_mismatches.append(
            "canonical application log records differ between AML and Foundry"
        )

    output_mismatches: list[str] = []
    logical_names = sorted(set(source.output_values) | set(target.output_values))
    for logical_name in logical_names:
        if logical_name not in source.output_values:
            output_mismatches.append(
                f"output {logical_name!r} is missing from AML evidence"
            )
        elif logical_name not in target.output_values:
            output_mismatches.append(
                f"output {logical_name!r} is missing from Foundry evidence"
            )
        elif _canonical_json(source.output_values[logical_name]) != _canonical_json(
            target.output_values[logical_name]
        ):
            output_mismatches.append(
                f"normalized output {logical_name!r} differs between AML and Foundry"
            )

    source_record_digests = tuple(_sha256_text(value) for value in source_records)
    target_record_digests = tuple(_sha256_text(value) for value in target_records)
    logs_equivalent = not log_mismatches
    outputs_equivalent = not output_mismatches
    return JobEquivalenceReport(
        source_job_name=source.job_name,
        target_job_name=target.job_name,
        logs_equivalent=logs_equivalent,
        outputs_equivalent=outputs_equivalent,
        equivalent=logs_equivalent and outputs_equivalent,
        source_log_record_digests=source_record_digests,
        target_log_record_digests=target_record_digests,
        source_output_digests=dict(source.output_digests),
        target_output_digests=dict(target.output_digests),
        log_mismatches=tuple(log_mismatches),
        output_mismatches=tuple(output_mismatches),
        source_log_root=source.log_root,
        target_log_root=target.log_root,
        source_output_root=source.output_root,
        target_output_root=target.output_root,
    )


def _reuse_completed_fixture(
    request: MigrationEquivalenceRequest,
    *,
    ml_client: Any,
    work_dir: Path,
) -> FixtureResult:
    job_name = str(request.existing_source_job_name or "").strip()
    asset_version = str(request.existing_source_asset_version or "").strip()
    if not job_name or not asset_version:
        raise ValueError(
            "existing_source_job_name and existing_source_asset_version must "
            "both be supplied to reuse an AML fixture."
        )
    job = ml_client.jobs.get(job_name)
    materialized = materialize_aml_command_job(job)
    status = str(getattr(job, "status", None) or "")
    if status != "Completed":
        raise RuntimeError(
            f"Existing AML fixture {job_name} has status {status!r}, not 'Completed'."
        )
    inputs = materialized.get("inputs") or {}

    def input_path(name: str) -> str:
        binding = inputs.get(name) or {}
        return str(binding.get("path") or binding.get("uri") or "")

    return FixtureResult(
        job_name=job_name,
        status=status,
        asset_version=asset_version,
        data_asset_id=input_path("training_data"),
        file_asset_id=input_path("config_file"),
        mltable_asset_id=input_path("training_table"),
        model_asset_id=input_path("seed_model"),
        work_dir=str(work_dir / "source-fixture"),
        expected_summary={},
    )


def run_aml_foundry_equivalence_exercise(
    request: MigrationEquivalenceRequest,
    *,
    credential: Any | None = None,
    ml_client: Any | None = None,
    emit: Any = print,
    sleeper: Any = time.sleep,
) -> MigrationEquivalenceResult:
    """Create, migrate, execute, and compare a command job across both services."""

    if credential is None:
        from azure.identity import AzureCliCredential

        credential = AzureCliCredential(process_timeout=60)
    if ml_client is None:
        from azure.ai.ml import MLClient

        ml_client = MLClient(
            credential,
            request.source.subscription_id,
            request.source.resource_group,
            request.source.workspace_name,
        )
    version = (
        request.asset_version
        or request.existing_source_asset_version
        or create_foundry_asset_version()
    )
    work_dir = Path(request.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if request.existing_source_job_name:
        fixture = _reuse_completed_fixture(
            request,
            ml_client=ml_client,
            work_dir=work_dir,
        )
        emit(f"Reusing completed AML fixture job {fixture.job_name}.")
    else:
        fixture = create_aml_migration_fixture(
            FixtureRequest(
                source=request.source,
                work_dir=work_dir / "source-fixture",
                environment_image=request.environment_image,
                job_name=request.fixture_job_name,
                asset_version=version,
                poll_interval_seconds=request.poll_interval_seconds,
                timeout_seconds=request.timeout_seconds,
            ),
            credential=credential,
            ml_client=ml_client,
            emit=emit,
            sleeper=sleeper,
        )
    source_evidence = collect_aml_fixture_evidence(
        fixture,
        ml_client=ml_client,
        credential=credential,
        work_dir=work_dir / "evidence" / "aml",
    )
    migration = AmlCommandJobMigrator(
        MigrationRequest(
            source=request.source,
            target=request.target,
            source_job_name=fixture.job_name,
            work_dir=work_dir / "migration",
            target_job_name=request.target_job_name,
            asset_version=version,
            dataset_transfer_mode=request.dataset_transfer_mode,
            source_storage_connection_name=(request.source_storage_connection_name),
            wait_for_completion=True,
            poll_interval_seconds=request.poll_interval_seconds,
            timeout_seconds=request.timeout_seconds,
            export_environment_image=request.export_environment_image,
        ),
        credential=credential,
        ml_client=ml_client,
        emit=emit,
        sleeper=sleeper,
    ).migrate()
    validation = validate_migrated_fixture(
        migration,
        target=request.target,
        work_dir=work_dir / "validation",
        credential=credential,
        timeout_seconds=request.validation_timeout_seconds,
        poll_interval_seconds=request.poll_interval_seconds,
        emit=emit,
        sleeper=sleeper,
        expected_summary=source_evidence.output_values["results"],
    )
    target_evidence = collect_foundry_fixture_evidence(
        migration,
        validation,
        target=request.target,
        credential=credential,
        work_dir=work_dir / "evidence" / "foundry",
        timeout_seconds=request.validation_timeout_seconds,
        poll_interval_seconds=request.poll_interval_seconds,
        emit=emit,
        sleeper=sleeper,
    )
    equivalence = compare_job_evidence(source_evidence, target_evidence)
    equivalence.assert_equivalent()
    result = MigrationEquivalenceResult(
        fixture=fixture,
        migration=migration,
        validation=validation,
        source_evidence=source_evidence,
        target_evidence=target_evidence,
        equivalence=equivalence,
    )
    report_path = work_dir / "equivalence-report.json"
    temporary_path = report_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(report_path)
    return result
