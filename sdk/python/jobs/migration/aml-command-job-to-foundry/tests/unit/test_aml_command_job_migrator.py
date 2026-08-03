from __future__ import annotations

import json
from types import SimpleNamespace
import time

import pytest

from foundrytrainingjob.dataset import DatasetUploadResult
from foundrytrainingjob.model_asset import ModelAssetUploadResult
from foundrytrainingjob.aml_command_job_migrator import (
    AmlCommandJobMigrator,
    AmlWorkspace,
    FoundryTarget,
    MigrationRequest,
    materialize_aml_command_job,
    parse_aml_asset_reference,
)
import foundrytrainingjob.aml_command_job_migrator as migrator_module


class _LiteralInput:
    type = "literal"
    value = 0.25
    description = "learning rate"

    @property
    def path(self):
        raise ValueError("Literal inputs do not have paths.")

    @property
    def mode(self):
        raise ValueError("Literal inputs do not have modes.")


class _SdkAccessorError(Exception):
    pass


class _SdkLiteralInput:
    type = "integer"
    value = 3

    @property
    def path(self):
        raise _SdkAccessorError("SDK literal input has no path")

    @property
    def mode(self):
        raise _SdkAccessorError("SDK literal input has no mode")


class _LiveSdkLiteralInput:
    type = "string"
    _data = 3

    @property
    def path(self):
        raise _SdkAccessorError("SDK literal input has no path")


class _MappingCommand(dict):
    def __init__(self):
        super().__init__({"component": "sparse-internal-value"})
        self.name = "mapping-command"
        self.type = "command"
        self.command = "python train.py"
        self.inputs = {}
        self.outputs = {}
        self.environment_variables = {}
        self.tags = {}

    def _to_rest_object(self):
        return object()


def test_parse_aml_asset_reference_supports_full_and_compact_ids():
    full = parse_aml_asset_reference(
        "azureml:/subscriptions/sub/resourceGroups/rg/providers/"
        "Microsoft.MachineLearningServices/workspaces/ws/models/seed/versions/7",
        expected_kind="model",
    )
    assert full is not None
    assert full.kind == "model"
    assert full.name == "seed"
    assert full.version == "7"
    assert full.workspace_name == "ws"

    compact = parse_aml_asset_reference(
        "azureml:training-data@latest",
        expected_kind="data",
    )
    assert compact is not None
    assert compact.name == "training-data"
    assert compact.label == "latest"

    bare = parse_aml_asset_reference(
        "training-data:7",
        expected_kind="data",
    )
    assert bare is not None
    assert bare.name == "training-data"
    assert bare.version == "7"

    assert (
        parse_aml_asset_reference(
            "azureml://datastores/workspaceblobstore/paths/input",
            expected_kind="data",
        )
        is None
    )


def test_prepare_inputs_reapplies_persisted_foundry_type_on_resume(tmp_path):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu-cluster"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/a/computes/cpu",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        asset_version="20260728123456789",
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )
    source_uri = "azureml:training-table:1"
    foundry_id = "azureai://accounts/a/projects/p/data/training-table/versions/1"
    migrator.journal.data["inputs"] = {
        "training_table": {
            "sourceUri": source_uri,
            "inputType": "mltable",
            "foundryInputType": "uri_folder",
            "foundryAssetId": foundry_id,
        }
    }
    source_inputs = {
        "training_table": {
            "type": "mltable",
            "path": source_uri,
            "mode": "download",
        }
    }

    mappings = migrator._prepare_inputs(source_inputs, "source-job")

    assert mappings == {source_uri: foundry_id}
    assert source_inputs["training_table"]["type"] == "uri_folder"


def test_reference_migration_rejects_connection_for_different_storage_host(
    tmp_path,
    monkeypatch,
):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu-cluster"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        dataset_transfer_mode="reference",
        source_storage_connection_name="source-storage",
        poll_interval_seconds=1,
        timeout_seconds=10,
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )
    monkeypatch.setattr(
        migrator_module,
        "inspect_connection_permission_info",
        lambda **kwargs: SimpleNamespace(
            available=True,
            target="https://other.blob.core.windows.net",
            error=None,
        ),
    )

    with pytest.raises(ValueError, match="does not match input 'data' host"):
        migrator._validate_reference_connection(
            {
                "data": {
                    "type": "uri_folder",
                    "path": ("https://source.blob.core.windows.net/container/path"),
                }
            }
        )

    assert "referenceConnectionValidation" not in migrator.journal.data


def test_reference_migration_records_matching_connection_validation(
    tmp_path,
    monkeypatch,
):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu-cluster"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        dataset_transfer_mode="reference",
        source_storage_connection_name="source-storage",
        poll_interval_seconds=1,
        timeout_seconds=10,
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )
    monkeypatch.setattr(
        migrator_module,
        "inspect_connection_permission_info",
        lambda **kwargs: SimpleNamespace(
            available=True,
            target="https://SOURCE.blob.core.windows.net/container",
            error=None,
        ),
    )

    migrator._validate_reference_connection(
        {
            "data": {
                "type": "uri_file",
                "path": "https://source.blob.core.windows.net/container/data.json",
            }
        }
    )

    assert migrator.journal.data["referenceConnectionValidation"] == {
        "connectionName": "source-storage",
        "connectionHost": "source.blob.core.windows.net",
        "validatedInputHosts": ["source.blob.core.windows.net"],
    }


def test_prepare_inputs_registers_zero_copy_reference_without_export(
    tmp_path,
    monkeypatch,
):
    captured = {}
    source_uri = "azureml:training-table:1"
    fake_client = SimpleNamespace(
        data=SimpleNamespace(
            get=lambda *args, **kwargs: SimpleNamespace(
                path="azureml://datastores/sourceblob/paths/training/table/"
            )
        ),
        datastores=SimpleNamespace(
            get=lambda name: SimpleNamespace(
                account_name="sourceaccount",
                container_name="source-container",
                endpoint="core.windows.net",
            )
        ),
    )
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu-cluster"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="target-storage",
            compute_id="/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/a/computes/cpu",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        asset_version="20260728123456789",
        dataset_transfer_mode="reference",
        source_storage_connection_name="aml-source-storage",
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=fake_client,
        emit=lambda message: None,
    )

    def fake_register(data_uri, **kwargs):
        captured.update({"data_uri": data_uri, **kwargs})
        return SimpleNamespace(
            dataset_id="azureai://accounts/a/projects/p/data/table/versions/1"
        )

    monkeypatch.setattr(migrator_module, "register_reference_dataset", fake_register)
    monkeypatch.setattr(
        migrator_module,
        "upload_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("zero-copy data must not be uploaded")
        ),
    )
    monkeypatch.setattr(
        migrator,
        "_export_inputs",
        lambda specs: (_ for _ in ()).throw(
            AssertionError("zero-copy data must not use an AML export job")
        )
        if specs
        else {},
    )
    source_inputs = {
        "training_table": {
            "type": "mltable",
            "path": source_uri,
            "mode": "download",
        }
    }

    mappings = migrator._prepare_inputs(source_inputs, "source-job")

    assert mappings[source_uri].endswith("/data/table/versions/1")
    assert captured["data_uri"] == (
        "https://sourceaccount.blob.core.windows.net/" "source-container"
    )
    assert captured["dataset_type"] == "uri_folder"
    assert captured["connection_name"] == "aml-source-storage"
    assert source_inputs["training_table"]["type"] == "uri_folder"
    assert migrator.journal.data["inputs"]["training_table"]["transferMode"] == (
        "reference"
    )
    assert migrator.journal.data["inputs"]["training_table"][
        "referenceDataUri"
    ].endswith("/source-container/training/table/")
    assert migrator.journal.data["inputs"]["training_table"][
        "registeredDataUri"
    ].endswith("/source-container")
    assert (
        migrator.journal.data["inputs"]["training_table"]["foundryInputPathSuffix"]
        == "training/table"
    )


def test_reference_transfer_requires_source_storage_connection(tmp_path):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu-cluster"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="target-storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        dataset_transfer_mode="reference",
    )

    with pytest.raises(
        ValueError,
        match="source_storage_connection_name is required",
    ):
        AmlCommandJobMigrator(
            request,
            credential=object(),
            ml_client=SimpleNamespace(),
        )


def test_materialize_aml_command_job_handles_sdk_like_entities():
    job = SimpleNamespace(
        name="source-job",
        type="command",
        display_name="Source job",
        description="fixture",
        experiment_name="migration",
        command="python train.py",
        code="azureml:source-code:3",
        environment="azureml:source-env:2",
        compute="azureml:cpu-cluster",
        environment_variables={"STATIC": "yes"},
        inputs={
            "rate": _LiteralInput(),
            "data": SimpleNamespace(
                type="uri_folder",
                path="azureml:training-data:4",
                mode="ro_mount",
                description=None,
                value=None,
            ),
        },
        outputs={
            "results": SimpleNamespace(
                type="uri_folder",
                path=None,
                mode="rw_mount",
                description="results",
            )
        },
        resources=SimpleNamespace(
            instance_count=2,
            shm_size="4g",
            docker_args=None,
            locations=None,
        ),
        limits=SimpleNamespace(timeout=3600),
        distribution=SimpleNamespace(
            type="pytorch",
            process_count_per_instance=1,
            worker_count=None,
            parameter_server_count=None,
            chief_count=None,
            ray_port=None,
            include_dashboard=None,
        ),
        queue_settings=SimpleNamespace(job_tier="standard"),
        job_tier=None,
        priority="high",
        services={
            "debug": SimpleNamespace(type="ssh", port=22, properties=None, nodes=None)
        },
        tags={"customer": "fixture"},
    )

    materialized = materialize_aml_command_job(job)

    assert materialized["inputs"]["rate"] == {
        "type": "literal",
        "value": 0.25,
        "description": "learning rate",
    }
    assert materialized["inputs"]["data"] == {
        "type": "uri_folder",
        "path": "azureml:training-data:4",
        "mode": "ro_mount",
    }
    assert materialized["outputs"]["results"] == {
        "type": "uri_folder",
        "mode": "rw_mount",
        "description": "results",
    }
    assert materialized["resources"] == {
        "instance_count": 2,
        "shm_size": "4g",
    }
    assert materialized["limits"] == {"timeout": 3600}
    assert materialized["distribution"] == {
        "type": "pytorch",
        "process_count_per_instance": 1,
    }
    assert materialized["queue_settings"] == {"job_tier": "standard"}
    assert materialized["services"]["debug"] == {"type": "ssh", "port": 22}


def test_materialize_preserves_current_ray_and_compatibility_fields():
    job = SimpleNamespace(
        name="ray-job",
        type="command",
        command="python train.py",
        inputs={},
        outputs={},
        environment_variables={},
        tags={},
        properties={"customer.setting": "enabled"},
        identity=SimpleNamespace(type="user_identity"),
        is_deterministic=False,
        parent_job_name="parent-job",
        notification_setting=SimpleNamespace(),
        intellectual_property=None,
        parameters={"legacy": "value"},
        resources=SimpleNamespace(
            instance_count=2,
            instance_type="Standard_NC24ads_A100_v4",
            shm_size="8g",
            docker_args=None,
            locations=None,
        ),
        limits=None,
        distribution=SimpleNamespace(
            type="ray",
            process_count_per_instance=None,
            worker_count=None,
            parameter_server_count=None,
            chief_count=None,
            ray_port=None,
            port=6379,
            address="auto",
            include_dashboard=True,
            dashboard_port=8265,
            head_node_additional_args="--block",
            worker_node_additional_args="--block",
        ),
        queue_settings=None,
        job_tier=None,
        priority=None,
        services=None,
        display_name="Ray job",
        description=None,
        experiment_name="migration",
        code=None,
        environment="mcr.microsoft.com/example:1",
        compute="cpu",
    )

    materialized = materialize_aml_command_job(job)

    assert materialized["properties"] == {"customer.setting": "enabled"}
    assert materialized["identity"] == "user_identity"
    assert materialized["is_deterministic"] is False
    assert materialized["parent_job_name"] == "parent-job"
    assert materialized["notification_setting"] is True
    assert materialized["parameters"] == {"legacy": "value"}
    assert materialized["resources"]["instance_type"] == ("Standard_NC24ads_A100_v4")
    assert materialized["distribution"] == {
        "type": "ray",
        "port": 6379,
        "address": "auto",
        "include_dashboard": True,
        "dashboard_port": 8265,
        "head_node_additional_args": "--block",
        "worker_node_additional_args": "--block",
    }


def test_materialize_mapping_like_sdk_command_uses_public_attributes():
    materialized = materialize_aml_command_job(_MappingCommand())

    assert materialized["name"] == "mapping-command"
    assert materialized["type"] == "command"
    assert materialized["command"] == "python train.py"
    assert "component" not in materialized


def test_materialize_ignores_sdk_accessor_errors_for_literal_inputs():
    job = _MappingCommand()
    job.inputs = {"epochs": _SdkLiteralInput()}

    materialized = materialize_aml_command_job(job)

    assert materialized["inputs"]["epochs"] == {
        "type": "integer",
        "value": 3,
    }


def test_materialize_reads_literal_value_from_sdk_private_data():
    job = _MappingCommand()
    job.inputs = {"epochs": _LiveSdkLiteralInput()}

    materialized = materialize_aml_command_job(job)

    assert materialized["inputs"]["epochs"] == {
        "type": "string",
        "value": 3,
    }


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        ("alpha", "string"),
        (7, "integer"),
        (0.25, "number"),
        (True, "boolean"),
    ],
)
def test_materialize_serializes_raw_primitive_inputs(value, expected_type):
    job = _MappingCommand()
    job.inputs = {"value": value}

    materialized = materialize_aml_command_job(job)

    assert materialized["inputs"]["value"] == {
        "type": expected_type,
        "value": value,
    }


def test_download_code_copies_an_explicit_local_snapshot(tmp_path):
    source_code = tmp_path / "original-code"
    source_code.mkdir()
    (source_code / "train.py").write_text("print('source')", encoding="utf-8")
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        source_code_path=source_code,
        work_dir=tmp_path / "work",
        poll_interval_seconds=1,
        timeout_seconds=10,
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )

    copied = migrator._download_code("ignored", "source-job")

    assert copied == tmp_path / "work" / "source-code"
    assert (copied / "train.py").read_text(encoding="utf-8") == "print('source')"


def test_resolve_environment_reuses_direct_acr_image(tmp_path):
    image = "private.azurecr.io/training/image@sha256:" + "a" * 64
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu-cluster"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
    )
    client = SimpleNamespace(
        environments=SimpleNamespace(
            get=lambda *args, **kwargs: SimpleNamespace(
                image=image,
                build=None,
                conda_file=None,
            )
        )
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=client,
        emit=lambda message: None,
    )

    assert migrator._resolve_environment_image("azureml:training-env:7") == image


def test_resolve_environment_rejects_conda_overlay(tmp_path):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu-cluster"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
    )
    client = SimpleNamespace(
        environments=SimpleNamespace(
            get=lambda *args, **kwargs: SimpleNamespace(
                image="private.azurecr.io/training/image:7",
                build=None,
                conda_file={"dependencies": ["python=3.11"]},
            )
        )
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=client,
        emit=lambda message: None,
    )

    with pytest.raises(ValueError, match="Conda overlays are not evaluated"):
        migrator._resolve_environment_image("azureml:training-env:7")


def test_prepare_foundry_attempt_archives_failed_job(tmp_path):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        poll_interval_seconds=1,
        timeout_seconds=10,
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )
    migrator.journal.data["foundryJob"] = {
        "name": "failed-job",
        "status": "Failed",
        "apimRequestId": "apim-id",
    }
    migrator.journal.data["requestBodyPath"] = "first-request.json"

    attempt_number, state = migrator._prepare_foundry_attempt()

    assert attempt_number == 2
    assert state == {}
    assert migrator.journal.data["foundryJobAttempts"] == [
        {
            "name": "failed-job",
            "status": "Failed",
            "apimRequestId": "apim-id",
            "requestBodyPath": "first-request.json",
        }
    ]


def test_manifest_rejects_changed_migration_invocation(tmp_path):
    source = AmlWorkspace("sub", "rg", "ws", "cpu")
    target = FoundryTarget(
        project_endpoint="https://example.test",
        project_name="project",
        storage_connection_name="storage",
        compute_id="/compute",
        instance_type="Singularity.D4_v3",
        api_version="2026-01-15-preview",
    )
    request = MigrationRequest(
        source=source,
        target=target,
        source_job_name="source-job",
        work_dir=tmp_path,
        poll_interval_seconds=1,
        timeout_seconds=10,
    )
    AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )
    changed_request = MigrationRequest(
        source=source,
        target=FoundryTarget(
            project_endpoint=target.project_endpoint,
            project_name=target.project_name,
            storage_connection_name=target.storage_connection_name,
            compute_id=target.compute_id,
            instance_type="Singularity.NC24ads_A100_v4",
            api_version=target.api_version,
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        poll_interval_seconds=2,
        timeout_seconds=20,
    )

    with pytest.raises(ValueError, match="different migration invocation"):
        AmlCommandJobMigrator(
            changed_request,
            credential=object(),
            ml_client=SimpleNamespace(),
            emit=lambda message: None,
        )


def test_manifest_rejects_changed_source_job_definition(tmp_path):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        poll_interval_seconds=1,
        timeout_seconds=10,
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )
    migrator._record_source_job(
        {"name": "source-job", "type": "command", "command": "python a.py"}
    )

    with pytest.raises(ValueError, match="source AML command-job definition changed"):
        migrator._record_source_job(
            {"name": "source-job", "type": "command", "command": "python b.py"}
        )


def test_manifest_rejects_request_drift_for_existing_foundry_job(tmp_path):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        poll_interval_seconds=1,
        timeout_seconds=10,
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )
    foundry_state: dict[str, object] = {}
    original_body = {"properties": {"command": "python a.py"}}
    migrator._record_foundry_request(foundry_state, original_body)
    foundry_state["name"] = "submitted-job"

    with pytest.raises(ValueError, match="submitted with a different request body"):
        migrator._record_foundry_request(
            foundry_state,
            {"properties": {"command": "python b.py"}},
        )

    assert migrator.journal.data["requestBody"] == original_body


def test_manifest_redacts_sensitive_values_without_weakening_resume(tmp_path):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        poll_interval_seconds=1,
        timeout_seconds=10,
    )
    source_job = {
        "name": "source-job",
        "type": "command",
        "command": "python train.py --token=source-secret",  # pragma: allowlist secret
        "environment_variables": {
            "API_TOKEN": "environment-secret"  # pragma: allowlist secret
        },
    }
    request_body = {
        "properties": {
            "command": "python train.py --password=target-secret",  # pragma: allowlist secret
            "environmentVariables": {
                "CLIENT_SECRET": "request-secret"  # pragma: allowlist secret
            },
        }
    }
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )
    migrator._record_source_job(source_job)
    migrator._record_foundry_request({}, request_body)

    manifest_text = (tmp_path / "migration-manifest.json").read_text(encoding="utf-8")
    request_text = (tmp_path / "foundry-job-request.json").read_text(encoding="utf-8")
    for secret in (
        "source-secret",
        "environment-secret",
        "target-secret",
        "request-secret",
    ):
        assert secret not in manifest_text
        assert secret not in request_text
    assert "<redacted>" in manifest_text
    assert "<redacted>" in request_text

    resumed = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )
    resumed._record_source_job(source_job)
    persisted = json.loads(manifest_text)
    assert persisted["sourceJobSha256"] == resumed.journal.data["sourceJobSha256"]


def test_redacted_source_uri_resumes_by_fingerprint(tmp_path):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        poll_interval_seconds=1,
        timeout_seconds=10,
    )
    source_uri = "https://source.blob.core.windows.net/data/input?sig=source-secret"
    foundry_id = "azureai://accounts/a/projects/p/data/input/versions/1"
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )
    migrator.journal.data["inputs"] = {
        "data": {
            "sourceUri": source_uri,
            "sourceUriSha256": migrator_module._sha256_json(source_uri),
            "inputType": "uri_folder",
            "foundryAssetId": foundry_id,
        }
    }
    migrator._save()
    manifest_text = (tmp_path / "migration-manifest.json").read_text(encoding="utf-8")
    assert "source-secret" not in manifest_text
    assert "sig=%3Credacted%3E" in manifest_text

    resumed = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
    )
    mappings = resumed._prepare_inputs(
        {"data": {"type": "uri_folder", "path": source_uri}},
        "source-job",
    )

    assert mappings == {source_uri: foundry_id}


def test_wait_for_foundry_job_reuses_cached_token(tmp_path, monkeypatch):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        poll_interval_seconds=0.01,
        timeout_seconds=10,
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
        sleeper=lambda seconds: None,
    )
    token_calls = []
    statuses = iter(("Queued", "Running", "Completed"))
    monkeypatch.setattr(
        migrator_module,
        "get_foundry_access_token",
        lambda **kwargs: (
            token_calls.append(True)
            or SimpleNamespace(token="cached-token", expires_on=int(time.time()) + 3600)
        ),
    )
    monkeypatch.setattr(
        migrator_module,
        "get_job_status_by_name",
        lambda *args, **kwargs: SimpleNamespace(
            summary=lambda: {
                "status": next(statuses),
                "apimRequestId": "apim-id",
            }
        ),
    )

    status = migrator._wait_for_foundry_job("target-job")

    assert status == "Completed"
    assert len(token_calls) == 1


def test_wait_for_foundry_job_continues_after_paused_status(tmp_path, monkeypatch):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        poll_interval_seconds=0.01,
        timeout_seconds=10,
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
        sleeper=lambda seconds: None,
    )
    statuses = iter(("Paused", "Running", "Completed"))
    monkeypatch.setattr(
        migrator_module,
        "get_foundry_access_token",
        lambda **kwargs: SimpleNamespace(
            token="cached-token",
            expires_on=int(time.time()) + 3600,
        ),
    )
    monkeypatch.setattr(
        migrator_module,
        "get_job_status_by_name",
        lambda *args, **kwargs: SimpleNamespace(
            summary=lambda: {
                "status": next(statuses),
                "apimRequestId": "apim-id",
            }
        ),
    )

    status = migrator._wait_for_foundry_job("target-job")

    assert status == "Completed"
    assert migrator.journal.data["foundryJob"]["status"] == "Completed"


@pytest.mark.parametrize("terminal_status", ("Failed", "Canceled", "Cancelled"))
def test_wait_for_foundry_job_rejects_non_success_terminal_status(
    tmp_path,
    monkeypatch,
    terminal_status,
):
    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id="/compute",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path,
        poll_interval_seconds=0.01,
        timeout_seconds=10,
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=SimpleNamespace(),
        emit=lambda message: None,
        sleeper=lambda seconds: None,
    )
    monkeypatch.setattr(
        migrator_module,
        "get_foundry_access_token",
        lambda **kwargs: SimpleNamespace(
            token="cached-token",
            expires_on=int(time.time()) + 3600,
        ),
    )
    monkeypatch.setattr(
        migrator_module,
        "get_job_status_by_name",
        lambda *args, **kwargs: SimpleNamespace(
            summary=lambda: {
                "status": terminal_status,
                "apimRequestId": "apim-id",
            }
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=f"reached terminal status {terminal_status}",
    ):
        migrator._wait_for_foundry_job("target-job")

    assert migrator.journal.data["foundryJob"]["status"] == terminal_status


def test_download_asset_path_resolves_aml_datastore_uri(tmp_path, monkeypatch):
    captured = {}

    def fake_download(uri, target_dir, *, credential):
        captured["uri"] = uri
        captured["target_dir"] = target_dir
        (target_dir / "nested").mkdir(parents=True)
        (target_dir / "nested" / "data.jsonl").write_text("{}\n", encoding="utf-8")
        return ("nested\\data.jsonl",)

    monkeypatch.setattr(
        migrator_module,
        "_download_azure_blob_path_with_identity",
        fake_download,
    )
    client = SimpleNamespace(
        datastores=SimpleNamespace(
            get=lambda name: SimpleNamespace(
                account_name="account",
                container_name="container",
                endpoint="core.windows.net",
            )
        )
    )

    files = migrator_module._download_asset_path_with_identity(
        client,
        object(),
        uri=(
            "azureml://datastores/foundrymigrationidentityblob/paths/" "migration/table"
        ),
        target_dir=tmp_path,
    )

    assert captured["uri"] == (
        "https://account.blob.core.windows.net/container/migration/table"
    )
    assert files == ("nested\\data.jsonl",)
    assert (tmp_path / "nested" / "data.jsonl").exists()


def test_download_asset_path_resolves_fully_qualified_datastore_uri(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_download(uri, target_dir, *, credential):
        captured["uri"] = uri
        return ("MLTable",)

    monkeypatch.setattr(
        migrator_module,
        "_download_azure_blob_path_with_identity",
        fake_download,
    )
    client = SimpleNamespace(
        datastores=SimpleNamespace(
            get=lambda name: SimpleNamespace(
                account_name="account",
                container_name="container",
                endpoint="core.windows.net",
            )
        )
    )

    files = migrator_module._download_asset_path_with_identity(
        client,
        object(),
        uri=(
            "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/"
            "datastores/identityblob/paths/migration/table/"
        ),
        target_dir=tmp_path,
    )

    assert captured["uri"] == (
        "https://account.blob.core.windows.net/container/migration/table/"
    )
    assert files == ("MLTable",)


def test_migrator_downloads_uploads_translates_and_submits(tmp_path, monkeypatch):
    data_id = (
        "azureml:/subscriptions/sub/resourceGroups/rg/providers/"
        "Microsoft.MachineLearningServices/workspaces/ws/data/train/versions/1"
    )
    model_id = (
        "azureml:/subscriptions/sub/resourceGroups/rg/providers/"
        "Microsoft.MachineLearningServices/workspaces/ws/models/seed/versions/1"
    )
    code_id = (
        "azureml:/subscriptions/sub/resourceGroups/rg/providers/"
        "Microsoft.MachineLearningServices/workspaces/ws/codes/code/versions/1"
    )
    environment_id = (
        "azureml:/subscriptions/sub/resourceGroups/rg/providers/"
        "Microsoft.MachineLearningServices/workspaces/ws/environments/env/versions/1"
    )
    source_job = SimpleNamespace(
        name="source-job",
        type="command",
        display_name="Source job",
        description="fixture",
        experiment_name="migration",
        command=(
            "python train.py --data ${{inputs.data}} --model ${{inputs.model}} "
            "--output ${{outputs.results}}"
        ),
        code=code_id,
        environment=environment_id,
        compute="azureml:cpu-cluster",
        environment_variables={"DATA_PATH": "${{inputs.data}}"},
        inputs={
            "literal": {"type": "integer", "value": 3},
            "data": {"type": "uri_folder", "path": data_id, "mode": "ro_mount"},
            "model": {"type": "custom_model", "path": model_id, "mode": "download"},
        },
        outputs={"results": {"type": "uri_folder", "mode": "rw_mount"}},
        resources=SimpleNamespace(
            instance_count=1, shm_size=None, docker_args=None, locations=None
        ),
        limits=SimpleNamespace(timeout=600),
        distribution=None,
        queue_settings=SimpleNamespace(job_tier="standard"),
        job_tier=None,
        priority="high",
        services=None,
        tags={"fixture": "true"},
        status="Completed",
    )

    class FakeCodeOperations:
        def download(self, name, version, download_path):
            destination = tmp_path / "work" / "source-code" / name
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "train.py").write_text("print('ok')", encoding="utf-8")

    class FakeModelOperations:
        def get(self, name, version=None, label=None):
            return SimpleNamespace(
                name=name,
                version=version or "1",
                path=(
                    "azureml://datastores/workspaceblobstore/paths/"
                    "registered-model/model"
                ),
            )

        def download(self, name, version, download_path):
            destination = (
                tmp_path / "work" / "inputs" / "model" / "model" / name / "model"
            )
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "weights.bin").write_bytes(b"model")

    class FakeJobs:
        def __init__(self):
            self.created = None

        def get(self, name):
            if name == "source-job":
                return source_job
            return SimpleNamespace(name=name, status="Completed")

        def create_or_update(self, job):
            self.created = job
            return SimpleNamespace(name=job.name, status="Completed")

        def download(self, name, *, download_path, all):
            destination = (
                tmp_path / "work" / "export-download" / "named-outputs" / "asset_1"
            )
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "records.jsonl").write_text("{}\n", encoding="utf-8")

    fake_jobs = FakeJobs()
    fake_client = SimpleNamespace(
        jobs=fake_jobs,
        environments=SimpleNamespace(
            get=lambda *args, **kwargs: SimpleNamespace(
                image="mcr.microsoft.com/example:1",
                conda_file=None,
                build=None,
            )
        ),
        models=FakeModelOperations(),
        _code=FakeCodeOperations(),
    )
    uploaded_datasets = []
    uploaded_models = []

    def fake_upload_dataset(local_path, **kwargs):
        uploaded_datasets.append((local_path, kwargs))
        return DatasetUploadResult(
            dataset_id=(
                f"azureai://accounts/a/projects/p/data/{kwargs['dataset_name']}/"
                f"versions/{kwargs['dataset_version']}"
            ),
            name=kwargs["dataset_name"],
            version=kwargs["dataset_version"],
            dataset_type="uri_folder",
            data_uri="https://storage.test/container/path",
            connection_name="storage",
            local_path=str(local_path),
        )

    def fake_upload_model(local_path, **kwargs):
        uploaded_models.append((local_path, kwargs))
        return ModelAssetUploadResult(
            asset_id=(
                f"azureai://accounts/a/projects/p/models/{kwargs['name']}/"
                f"versions/{kwargs['version']}"
            ),
            name=kwargs["name"],
            version=kwargs["version"],
            provisioning_status="Succeeded",
            blob_uri="https://storage.test/model",
            project_endpoint="https://example.test/api/projects/p",
            project_name="p",
            uploaded_files=("weights.bin",),
            total_bytes=5,
        )

    def fake_download_workspace_blob_prefix(
        ml_client,
        credential,
        *,
        prefix,
        target_dir,
    ):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "records.jsonl").write_text("{}\n", encoding="utf-8")
        return ("records.jsonl",)

    fake_response = SimpleNamespace(
        status_code=201,
        text="",
        apim_request_id="apim-id",
    )
    monkeypatch.setattr(migrator_module, "upload_dataset", fake_upload_dataset)
    monkeypatch.setattr(migrator_module, "upload_and_register_model", fake_upload_model)
    monkeypatch.setattr(
        migrator_module,
        "_download_workspace_blob_prefix",
        fake_download_workspace_blob_prefix,
    )
    monkeypatch.setattr(
        migrator_module,
        "get_foundry_access_token",
        lambda **kwargs: SimpleNamespace(token="token"),
    )
    monkeypatch.setattr(
        migrator_module,
        "submit_job",
        lambda request_body, **kwargs: SimpleNamespace(
            job_name="migrated-source-job-abcd",
            request_id="request-id",
            response=fake_response,
        ),
    )
    monkeypatch.setattr(
        migrator_module,
        "get_job_status_by_name",
        lambda *args, **kwargs: SimpleNamespace(
            summary=lambda: {"status": "Completed", "apimRequestId": "apim-id"}
        ),
    )

    request = MigrationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu-cluster"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="p",
            storage_connection_name="storage",
            compute_id="/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/a/computes/cpu",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_job_name="source-job",
        work_dir=tmp_path / "work",
        asset_version="20260728123456789",
        poll_interval_seconds=0.01,
        timeout_seconds=60,
    )
    migrator = AmlCommandJobMigrator(
        request,
        credential=object(),
        ml_client=fake_client,
        emit=lambda message: None,
        sleeper=lambda seconds: None,
    )

    result = migrator.migrate()

    assert result.target_job_name == "migrated-source-job-abcd"
    assert result.target_status == "Completed"
    assert result.request_body["properties"]["environmentImageReference"] == (
        "mcr.microsoft.com/example:1"
    )
    assert result.request_body["properties"]["codeId"].startswith("azureai://")
    assert result.request_body["properties"]["inputs"]["literal"] == {
        "jobInputType": "literal",
        "value": "3",
    }
    assert result.request_body["properties"]["inputs"]["data"]["uri"].startswith(
        "azureai://"
    )
    assert result.request_body["properties"]["inputs"]["model"]["uri"].startswith(
        "azureai://"
    )
    assert result.request_body["properties"]["resources"]["properties"] == {
        "AISuperComputer": {
            "slaTier": "Premium",
            "priority": "high",
            "imageVersion": "",
        }
    }
    assert len(uploaded_datasets) == 2
    assert len(uploaded_models) == 1
    assert uploaded_models[0][0] == (
        tmp_path / "work" / "inputs" / "model" / "model" / "seed" / "model"
    )
    assert uploaded_models[0][1]["blob_prefix"] == "model"
    assert (
        "--model ${{inputs.model}}/model"
        in result.request_body["properties"]["command"]
    )
    assert fake_jobs.created is not None
    assert type(fake_jobs.created.identity).__name__ == "UserIdentityConfiguration"
    assert "base64.b64decode" in fake_jobs.created.command
    assert not fake_jobs.created.code
    assert str(fake_jobs.created.outputs["exported_asset_1"].path).endswith(
        "/export/exported_asset_1"
    )
    assert (tmp_path / "work" / "migration-manifest.json").exists()
    assert (tmp_path / "work" / "foundry-job-request.json").exists()
