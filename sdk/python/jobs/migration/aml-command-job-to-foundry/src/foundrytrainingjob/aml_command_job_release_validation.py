"""Cost-bounded release qualification for AML command-job migration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .aml_command_job_analysis import analyze_aml_command_job
from .aml_command_job_equivalence import (
    JobEquivalenceReport,
    JobEvidence,
    MigrationEquivalenceRequest,
    MigrationEquivalenceResult,
    run_aml_foundry_equivalence_exercise,
)
from .aml_command_job_migration import create_foundry_asset_version
from .aml_command_job_migrator import AmlWorkspace, FoundryTarget
from .aml_command_job_fixture import (
    FixtureRequest,
    FixtureResult,
    FixtureValidationResult,
    create_aml_migration_fixture,
)
from .aml_command_job_migrator import MigrationResult
from .dataset import build_project_endpoint
from .e2e.sanitization import sanitize_for_report


_DATA_INPUTS = {
    "training_data": "uri_folder",
    "config_file": "uri_file",
    "training_table": "mltable",
}
_LITERAL_INPUTS = {
    "epochs": "3",
    "learning_rate": "0.125",
    "message": "resurrected-command-job",
    "enabled": "true",
}
_MODEL_INPUTS = {"seed_model": "custom_model"}
_SOURCE_OUTPUTS = {
    "results": "uri_folder",
    "summary": "uri_file",
    "metrics_table": "mltable",
    "trained_model": "custom_model",
}
_TARGET_OUTPUTS = {
    **_SOURCE_OUTPUTS,
    "metrics_table": "uri_folder",
}
_IMPLEMENTATION_FILES = (
    "aml_command_job_analysis.py",
    "aml_command_job_equivalence.py",
    "aml_command_job_fixture.py",
    "aml_command_job_migration.py",
    "aml_command_job_migration_cli.py",
    "aml_command_job_migrator.py",
    "aml_command_job_permissions.py",
    "aml_command_job_release_validation.py",
    "dataset.py",
    "e2e/sanitization.py",
    "model_asset.py",
)
_QUALIFIED_SCOPE = {
    "jobTypes": ["standalone command"],
    "topology": ["single instance", "no distribution"],
    "literalInputs": ["string", "integer", "number", "boolean"],
    "assetInputs": ["uri_folder", "uri_file", "mltable", "custom_model"],
    "inputModes": ["download"],
    "dataTransferModes": ["upload", "reference"],
    "sourceOutputs": ["uri_folder", "uri_file", "mltable", "custom_model"],
    "targetOutputs": [
        "uri_folder",
        "uri_file",
        "AML mltable adapted to uri_folder",
        "custom_model",
    ],
    "sourceOutputModes": ["upload"],
    "other": [
        "code snapshot copy",
        "public OCI image reference",
        "target compute replacement",
        "target UAI replacement",
        "static and input-templated environment variables",
        "shared memory and timeout",
        "display name, description, experiment name, and tags",
    ],
}
_EXCLUDED_SCOPE = (
    "pipeline, component, sweep, and AutoML jobs",
    "multi-node or MPI, PyTorch, TensorFlow, and Ray distributions",
    "direct, mount, and evaluation input modes",
    "direct and explicit rw_mount source output modes",
    "MLflow, SafeTensors, and Triton model formats",
    "Conda overlays, Docker build contexts, and unproven private ACR paths",
    "interactive service definitions",
    "nontrivial MLTable transformations",
    "optional, ranged, enum, and early-available component contracts",
    "lineage, notification, secret, and source-output-destination semantics",
)


@dataclass(frozen=True)
class ReleaseValidationRequest:
    source: AmlWorkspace
    target: FoundryTarget
    source_storage_connection_name: str
    work_dir: str | Path
    environment_image: str = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
    export_environment_image: str = (
        "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
    )
    fixture_job_name: str | None = None
    asset_version: str | None = None
    poll_interval_seconds: float = 15.0
    timeout_seconds: float = 3600.0
    validation_timeout_seconds: float = 600.0
    existing_source_job_name: str | None = None
    existing_source_asset_version: str | None = None
    reuse_completed_upload: bool = False


@dataclass(frozen=True)
class ReleaseValidationResult:
    report_path: str
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(sanitize_for_report(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _load_job_evidence(value: dict[str, Any]) -> JobEvidence:
    return JobEvidence(
        service=str(value["service"]),
        job_name=str(value["job_name"]),
        status=str(value["status"]),
        log_root=str(value["log_root"]),
        output_root=str(value["output_root"]),
        log_files=tuple(value.get("log_files") or ()),
        output_files=tuple(value.get("output_files") or ()),
        application_records=tuple(value.get("application_records") or ()),
        output_values=dict(value.get("output_values") or {}),
        log_digests=dict(value.get("log_digests") or {}),
        output_digests=dict(value.get("output_digests") or {}),
    )


def _load_equivalence_result(work_dir: Path) -> MigrationEquivalenceResult:
    report_path = work_dir / "equivalence-report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"No completed upload equivalence report exists at {report_path}."
        )
    value = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_path = Path(value["migration"]["manifestPath"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_value = dict(value["validation"])
    validation_value["model_downloaded_files"] = tuple(
        validation_value.get("model_downloaded_files") or ()
    )
    equivalence_value = dict(value["equivalence"])
    for key in (
        "source_log_record_digests",
        "target_log_record_digests",
        "log_mismatches",
        "output_mismatches",
    ):
        equivalence_value[key] = tuple(equivalence_value.get(key) or ())
    return MigrationEquivalenceResult(
        fixture=FixtureResult(**value["fixture"]),
        migration=MigrationResult(
            manifest_path=str(manifest_path),
            source_job_name=str(value["migration"]["sourceJobName"]),
            target_job_name=str(value["migration"]["targetJobName"]),
            target_status=value["migration"].get("targetStatus"),
            request_body=dict(manifest["requestBody"]),
            asset_mappings=dict(value["migration"].get("assetMappings") or {}),
            warnings=tuple(value["migration"].get("warnings") or ()),
        ),
        validation=FixtureValidationResult(**validation_value),
        source_evidence=_load_job_evidence(value["sourceEvidence"]),
        target_evidence=_load_job_evidence(value["targetEvidence"]),
        equivalence=JobEquivalenceReport(**equivalence_value),
    )


def _implementation_evidence() -> list[dict[str, Any]]:
    package_root = Path(__file__).resolve().parent
    return [
        {
            "path": file_name,
            "sha256": _sha256_file(package_root / file_name),
            "bytes": (package_root / file_name).stat().st_size,
        }
        for file_name in _IMPLEMENTATION_FILES
    ]


def _evidence_inventory(root: Path, report_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != report_path and not path.name.endswith(".tmp")
    ]


def _normalized(value: Any) -> str:
    return str(value or "").strip().rstrip("/").lower()


def _validate_result_shape(
    result: MigrationEquivalenceResult,
    *,
    transfer_mode: str,
) -> dict[str, Any]:
    if result.fixture.status != "Completed":
        raise AssertionError(f"AML fixture status is {result.fixture.status!r}.")
    if result.migration.target_status != "Completed":
        raise AssertionError(
            f"Foundry target status is {result.migration.target_status!r}."
        )
    if not result.equivalence.equivalent:
        raise AssertionError("AML and Foundry logs/outputs are not equivalent.")

    properties = result.migration.request_body.get("properties") or {}
    inputs = properties.get("inputs") or {}
    outputs = properties.get("outputs") or {}
    expected_input_names = set(_LITERAL_INPUTS) | set(_DATA_INPUTS) | set(_MODEL_INPUTS)
    if set(inputs) != expected_input_names:
        raise AssertionError(
            f"Input coverage differs: expected {sorted(expected_input_names)}, "
            f"got {sorted(inputs)}."
        )
    if set(outputs) != set(_SOURCE_OUTPUTS):
        raise AssertionError(
            f"Output coverage differs: expected {sorted(_SOURCE_OUTPUTS)}, "
            f"got {sorted(outputs)}."
        )

    for name, expected_value in _LITERAL_INPUTS.items():
        binding = inputs[name]
        if _normalized(binding.get("jobInputType")) != "literal":
            raise AssertionError(f"{name} is not a Foundry literal input.")
        if _normalized(binding.get("value")) != expected_value:
            raise AssertionError(f"Literal {name} differs: {binding.get('value')!r}.")

    observed_input_types: dict[str, str] = {}
    for name, expected_type in {**_DATA_INPUTS, **_MODEL_INPUTS}.items():
        binding = inputs[name]
        actual_type = _normalized(binding.get("jobInputType"))
        allowed_types = (
            {"mltable", "uri_folder"} if name == "training_table" else {expected_type}
        )
        if actual_type not in allowed_types:
            raise AssertionError(
                f"Input {name} type differs: expected {sorted(allowed_types)}, "
                f"got {actual_type!r}."
            )
        if not str(binding.get("uri") or "").startswith("azureai://"):
            raise AssertionError(f"Input {name} has no project asset URI.")
        observed_input_types[name] = actual_type

    observed_output_types = {
        name: _normalized(outputs[name].get("jobOutputType"))
        for name in _TARGET_OUTPUTS
    }
    if observed_output_types != _TARGET_OUTPUTS:
        raise AssertionError(
            f"Output types differ: expected {_TARGET_OUTPUTS}, got {observed_output_types}."
        )
    for name, binding in outputs.items():
        if not binding.get("assetName") or not binding.get("assetVersion"):
            raise AssertionError(f"Output {name} has no target asset identity.")

    if not str(properties.get("codeId") or "").startswith("azureai://"):
        raise AssertionError("Migrated request has no Dataset V3 codeId.")
    if not properties.get("environmentImageReference"):
        raise AssertionError("Migrated request has no environment image reference.")
    if not properties.get("computeId") or not properties.get("userAssignedIdentityId"):
        raise AssertionError("Migrated request has no compute or target UAI.")
    if properties.get("environmentVariables") != {"STATIC_SETTING": "preserved"}:
        raise AssertionError(
            "Static/templated environment-variable adaptation differs."
        )
    if "DATA_FROM_ENV" not in str(properties.get("command") or ""):
        raise AssertionError("Templated environment variable was not moved to command.")
    resources = properties.get("resources") or {}
    if resources.get("instanceCount") != 1 or resources.get("shmSize") != "2g":
        raise AssertionError("Single-instance/shared-memory resources differ.")
    if (properties.get("limits") or {}).get("timeout") != "PT3600S":
        raise AssertionError("Command timeout translation differs.")

    logical_outputs = set(_SOURCE_OUTPUTS)
    if set(result.source_evidence.output_values) != logical_outputs:
        raise AssertionError("AML evidence does not contain every logical output.")
    if set(result.target_evidence.output_values) != logical_outputs:
        raise AssertionError("Foundry evidence does not contain every logical output.")
    if (
        result.equivalence.source_output_digests
        != result.equivalence.target_output_digests
    ):
        raise AssertionError("AML and Foundry logical output digests differ.")
    if not result.validation.metrics_table_dataset_id:
        raise AssertionError("MLTable output was not registered in Dataset V3.")

    manifest = json.loads(
        Path(result.migration.manifest_path).read_text(encoding="utf-8")
    )
    manifest_inputs = manifest.get("inputs") or {}
    for name in _DATA_INPUTS:
        record = manifest_inputs.get(name) or {}
        if transfer_mode == "reference":
            if record.get("transferMode") != "reference":
                raise AssertionError(f"Input {name} was not recorded as reference.")
            if not record.get("referenceDataUri"):
                raise AssertionError(f"Input {name} has no source reference URI.")
            if name != "config_file" and (
                not record.get("registeredDataUri")
                or not record.get("foundryInputPathSuffix")
            ):
                raise AssertionError(
                    f"Folder input {name} has no container-root adaptation."
                )
        elif record.get("transferMode") == "reference" or not record.get("localPath"):
            raise AssertionError(f"Input {name} was not materialized for upload.")

    return {
        "transferMode": transfer_mode,
        "sourceJobName": result.fixture.job_name,
        "targetJobName": result.migration.target_job_name,
        "targetStatus": result.migration.target_status,
        "literalInputs": dict(_LITERAL_INPUTS),
        "assetInputTypes": observed_input_types,
        "outputTypes": observed_output_types,
        "sourceOutputDigests": dict(result.equivalence.source_output_digests),
        "targetOutputDigests": dict(result.equivalence.target_output_digests),
        "logsEquivalent": result.equivalence.logs_equivalent,
        "outputsEquivalent": result.equivalence.outputs_equivalent,
        "manifestPath": result.migration.manifest_path,
    }


def _inspect_dataset_bindings(
    result: MigrationEquivalenceResult,
    *,
    target: FoundryTarget,
    transfer_mode: str,
    expected_connection_name: str,
    credential: Any,
) -> dict[str, dict[str, Any]]:
    from azure.ai.projects import AIProjectClient

    manifest = json.loads(
        Path(result.migration.manifest_path).read_text(encoding="utf-8")
    )
    endpoint = build_project_endpoint(
        target.project_endpoint,
        project_name=target.project_name,
    )
    observed: dict[str, dict[str, Any]] = {}
    with AIProjectClient(endpoint=endpoint, credential=credential) as client:
        expected_connection = client.connections.get(name=expected_connection_name)
        expected_connection_target = str(
            getattr(expected_connection, "target", None) or ""
        )
        expected_host = _normalized(urlsplit(expected_connection_target).hostname)
        if not expected_host:
            raise AssertionError(
                f"Connection {expected_connection_name!r} exposes no storage host."
            )
        for name in _DATA_INPUTS:
            record = manifest["inputs"][name]
            dataset = client.datasets.get(
                name=str(record["assetName"]),
                version=str(record["assetVersion"]),
            )
            is_reference = getattr(dataset, "is_reference", None)
            connection_name = str(getattr(dataset, "connection_name", None) or "")
            data_uri = str(getattr(dataset, "data_uri", None) or "")
            if transfer_mode == "reference" and is_reference is not True:
                raise AssertionError(f"Dataset {name} is not a reference dataset.")
            if transfer_mode == "upload" and is_reference is True:
                raise AssertionError(
                    f"Dataset {name} unexpectedly remains a reference."
                )
            if transfer_mode == "reference":
                if _normalized(connection_name) != _normalized(
                    expected_connection_name
                ):
                    raise AssertionError(
                        f"Dataset {name} connection differs: {connection_name!r}."
                    )
                expected_uri = str(
                    record.get("registeredDataUri") or record["referenceDataUri"]
                )
                if _normalized(data_uri) != _normalized(expected_uri):
                    raise AssertionError(
                        f"Dataset {name} reference URI differs from AML source URI."
                    )
                path_suffix = str(record.get("foundryInputPathSuffix") or "")
                reconstructed_uri = (
                    f"{data_uri.rstrip('/')}/{path_suffix}" if path_suffix else data_uri
                )
                if _normalized(reconstructed_uri) != _normalized(
                    record["referenceDataUri"]
                ):
                    raise AssertionError(
                        f"Dataset {name} root plus suffix does not reconstruct "
                        "the AML source URI."
                    )
            elif _normalized(urlsplit(data_uri).hostname) != expected_host:
                raise AssertionError(
                    f"Dataset {name} was uploaded to {urlsplit(data_uri).hostname!r}, "
                    f"not connection target host {expected_host!r}."
                )
            observed[name] = {
                "id": str(getattr(dataset, "id", None) or ""),
                "type": str(getattr(dataset, "type", None) or ""),
                "isReference": is_reference,
                "connectionName": connection_name,
                "dataUri": data_uri,
                "dataUriHost": urlsplit(data_uri).hostname,
                "expectedConnectionTarget": expected_connection_target,
            }
    return observed


def _analysis_summary(report: Any) -> dict[str, Any]:
    summary = dict(report.summary)
    return {
        "policy": report.policy,
        "policyPassed": report.policy_passed,
        "runtimePermissionsSatisfied": summary.get("runtimePermissionsSatisfied"),
        "missingPermissionIds": summary.get("missingPermissionIds", []),
        "unknownPermissionIds": summary.get("unknownPermissionIds", []),
        "familiesWithoutLiveEvidence": summary.get("familiesWithoutLiveEvidence", []),
        "countsBySupport": summary.get("countsBySupport", {}),
        "countsBySemanticFidelity": summary.get("countsBySemanticFidelity", {}),
    }


def _assert_migratable(report: Any, *, transfer_mode: str) -> None:
    summary = report.summary
    if not report.policy_passed:
        raise AssertionError(f"{transfer_mode} migratable analysis policy failed.")
    if summary.get("runtimePermissionsSatisfied") is not True:
        raise AssertionError(
            f"{transfer_mode} runtime permission preflight was not satisfied."
        )
    if summary.get("missingPermissionIds") or summary.get("unknownPermissionIds"):
        raise AssertionError(
            f"{transfer_mode} runtime permission preflight has unresolved checks."
        )


def run_aml_foundry_release_validation(
    request: ReleaseValidationRequest,
    *,
    credential: Any | None = None,
    ml_client: Any | None = None,
    emit: Any = print,
    sleeper: Any = __import__("time").sleep,
) -> ReleaseValidationResult:
    """Qualify upload and reference migration with one AML source fixture."""

    root = Path(request.work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "release-validation-report.json"
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "startedAtUtc": _utc_now(),
        "status": "running",
        "qualifiedScope": _QUALIFIED_SCOPE,
        "excludedScope": list(_EXCLUDED_SCOPE),
        "implementationEvidence": _implementation_evidence(),
        "costBound": {
            "sourceAmlJobs": 0 if request.existing_source_job_name else 1,
            "batchedAmlExportJobs": 0 if request.reuse_completed_upload else 1,
            "foundryJobs": 1 if request.reuse_completed_upload else 2,
            "totalCloudJobs": (
                1
                if request.reuse_completed_upload
                else 3
                if request.existing_source_job_name
                else 4
            ),
            "note": "Both transfer modes reuse one completed AML source job.",
        },
        "source": {
            "subscriptionId": request.source.subscription_id,
            "resourceGroup": request.source.resource_group,
            "workspaceName": request.source.workspace_name,
            "exportCompute": request.source.export_compute,
        },
        "target": {
            "projectEndpoint": request.target.project_endpoint,
            "projectName": request.target.project_name,
            "storageConnectionName": request.target.storage_connection_name,
            "sourceStorageConnectionName": request.source_storage_connection_name,
            "computeId": request.target.compute_id,
            "instanceType": request.target.instance_type,
            "userAssignedIdentityId": request.target.user_assigned_identity_id,
        },
        "runs": {},
    }
    _write_json(report_path, report)

    try:
        if not request.source_storage_connection_name.strip():
            raise ValueError(
                "source_storage_connection_name is required for release validation."
            )
        if bool(request.existing_source_job_name) != bool(
            request.existing_source_asset_version
        ):
            raise ValueError(
                "existing_source_job_name and existing_source_asset_version must "
                "be supplied together."
            )
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

        upload_version = request.asset_version or create_foundry_asset_version()
        suffix = upload_version[-10:]
        if request.existing_source_job_name:
            source_job_name = request.existing_source_job_name
            source_asset_version = str(request.existing_source_asset_version)
            report["sourceFixture"] = {
                "created": False,
                "jobName": source_job_name,
                "assetVersion": source_asset_version,
            }
        else:
            source_fixture = create_aml_migration_fixture(
                FixtureRequest(
                    source=request.source,
                    work_dir=root / "source-fixture",
                    environment_image=request.environment_image,
                    job_name=(
                        request.fixture_job_name
                        or f"aml-foundry-release-source-{suffix}"
                    ),
                    asset_version=upload_version,
                    poll_interval_seconds=request.poll_interval_seconds,
                    timeout_seconds=request.timeout_seconds,
                ),
                credential=credential,
                ml_client=ml_client,
                emit=emit,
                sleeper=sleeper,
            )
            source_job_name = source_fixture.job_name
            source_asset_version = source_fixture.asset_version
            report["sourceFixture"] = {
                "created": True,
                "jobName": source_job_name,
                "assetVersion": source_asset_version,
                "status": source_fixture.status,
            }

        upload_analysis = analyze_aml_command_job(
            source=request.source,
            target=request.target,
            source_job_name=source_job_name,
            policy="migratable",
            dataset_transfer_mode="upload",
            credential=credential,
            ml_client=ml_client,
        )
        _write_json(
            root / "upload" / "migration-analysis.json", upload_analysis.to_dict()
        )
        _assert_migratable(upload_analysis, transfer_mode="upload")
        reference_analysis = analyze_aml_command_job(
            source=request.source,
            target=request.target,
            source_job_name=source_job_name,
            policy="migratable",
            dataset_transfer_mode="reference",
            source_storage_connection_name=request.source_storage_connection_name,
            credential=credential,
            ml_client=ml_client,
        )
        _write_json(
            root / "reference" / "migration-analysis.json",
            reference_analysis.to_dict(),
        )
        _assert_migratable(reference_analysis, transfer_mode="reference")
        report["preflight"] = {
            "upload": _analysis_summary(upload_analysis),
            "reference": _analysis_summary(reference_analysis),
        }
        _write_json(report_path, report)

        if request.reuse_completed_upload:
            upload = _load_equivalence_result(root / "upload")
            if upload.fixture.job_name != source_job_name:
                raise AssertionError(
                    "Completed upload evidence belongs to source job "
                    f"{upload.fixture.job_name!r}, not {source_job_name!r}."
                )
            emit(
                f"Reusing completed upload evidence for "
                f"{upload.migration.target_job_name}."
            )
        else:
            upload = run_aml_foundry_equivalence_exercise(
                MigrationEquivalenceRequest(
                    source=request.source,
                    target=request.target,
                    work_dir=root / "upload",
                    environment_image=request.environment_image,
                    export_environment_image=request.export_environment_image,
                    target_job_name=f"aml-foundry-release-{suffix}-upload",
                    asset_version=upload_version,
                    dataset_transfer_mode="upload",
                    poll_interval_seconds=request.poll_interval_seconds,
                    timeout_seconds=request.timeout_seconds,
                    validation_timeout_seconds=request.validation_timeout_seconds,
                    existing_source_job_name=source_job_name,
                    existing_source_asset_version=source_asset_version,
                ),
                credential=credential,
                ml_client=ml_client,
                emit=emit,
                sleeper=sleeper,
            )
        upload_shape = _validate_result_shape(upload, transfer_mode="upload")
        upload_datasets = _inspect_dataset_bindings(
            upload,
            target=request.target,
            transfer_mode="upload",
            expected_connection_name=request.target.storage_connection_name,
            credential=credential,
        )
        report["runs"]["upload"] = {
            **upload_shape,
            "analysis": _analysis_summary(upload_analysis),
            "datasets": upload_datasets,
            "equivalenceReport": "upload/equivalence-report.json",
        }
        _write_json(report_path, report)

        reference_version = create_foundry_asset_version()
        reference = run_aml_foundry_equivalence_exercise(
            MigrationEquivalenceRequest(
                source=request.source,
                target=request.target,
                work_dir=root / "reference",
                environment_image=request.environment_image,
                export_environment_image=request.export_environment_image,
                target_job_name=f"aml-foundry-release-{suffix}-reference",
                asset_version=reference_version,
                dataset_transfer_mode="reference",
                source_storage_connection_name=request.source_storage_connection_name,
                poll_interval_seconds=request.poll_interval_seconds,
                timeout_seconds=request.timeout_seconds,
                validation_timeout_seconds=request.validation_timeout_seconds,
                existing_source_job_name=source_job_name,
                existing_source_asset_version=source_asset_version,
            ),
            credential=credential,
            ml_client=ml_client,
            emit=emit,
            sleeper=sleeper,
        )
        reference_shape = _validate_result_shape(
            reference,
            transfer_mode="reference",
        )
        reference_datasets = _inspect_dataset_bindings(
            reference,
            target=request.target,
            transfer_mode="reference",
            expected_connection_name=request.source_storage_connection_name,
            credential=credential,
        )
        report["runs"]["reference"] = {
            **reference_shape,
            "analysis": _analysis_summary(reference_analysis),
            "datasets": reference_datasets,
            "equivalenceReport": "reference/equivalence-report.json",
        }
        report["releaseDecision"] = {
            "qualifiedScopeReady": True,
            "unrestrictedCustomerReleaseReady": False,
            "reason": (
                "The qualified single-instance command-job scope passed; "
                "the explicitly excluded variants still lack migration-specific "
                "live parity evidence."
            ),
        }
        report["status"] = "passed"
        report["completedAtUtc"] = _utc_now()
        report["evidenceFiles"] = _evidence_inventory(root, report_path)
        _write_json(report_path, report)
        return ReleaseValidationResult(
            report_path=str(report_path),
            report=report,
        )
    except Exception as error:
        report["status"] = "failed"
        report["completedAtUtc"] = _utc_now()
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        report["releaseDecision"] = {
            "qualifiedScopeReady": False,
            "unrestrictedCustomerReleaseReady": False,
            "reason": "Release qualification did not complete successfully.",
        }
        report["evidenceFiles"] = _evidence_inventory(root, report_path)
        _write_json(report_path, report)
        raise
