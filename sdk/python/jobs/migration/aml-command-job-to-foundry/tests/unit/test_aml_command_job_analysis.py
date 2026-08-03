from __future__ import annotations

from collections import Counter

import pytest

from foundrytrainingjob.aml_command_job_analysis import (
    EnvironmentInspection,
    _value,
    analyze_materialized_aml_command_job,
)
from foundrytrainingjob.aml_command_job_permissions import (
    PermissionCheck,
    RuntimePermissionInspection,
)


_DIGEST_IMAGE = "private.azurecr.io/training/image@sha256:" + "a" * 64


class _MappingLikeEntity(dict):
    environment = "azureml:environment:7"


def _source_job() -> dict:
    return {
        "name": "source-job",
        "type": "command",
        "command": "python train.py --data ${{inputs.data}} --model ${{inputs.model}}",
        "code": "azureml:code:1",
        "environment": "azureml:environment:1",
        "compute": "azureml:cpu-cluster",
        "identity": "user_identity",
        "inputs": {
            "epochs": {"type": "integer", "value": 3},
            "data": {
                "type": "uri_folder",
                "path": "azureml:data:1",
                "mode": "download",
            },
            "model": {
                "type": "custom_model",
                "path": "azureml:model:1",
                "mode": "download",
            },
        },
        "outputs": {
            "results": {
                "type": "uri_folder",
                "mode": "upload",
                "path": "azureml://datastores/source/paths/results",
            }
        },
        "environment_variables": {"STATIC": "preserved"},
        "resources": {"instance_count": 1, "shm_size": "2g"},
        "limits": {"timeout": 3600},
        "tags": {"scenario": "analysis"},
    }


def _environment() -> EnvironmentInspection:
    return EnvironmentInspection(
        source_reference="azureml:environment:1",
        image_reference=_DIGEST_IMAGE,
    )


def test_value_uses_public_attribute_when_mapping_key_is_absent():
    entity = _MappingLikeEntity({"component": "internal"})

    assert _value(entity, "environment") == "azureml:environment:7"
    assert _value(entity, "component") == "internal"


def test_migratable_policy_allows_copy_but_reports_loss_and_reference_gaps():
    report = analyze_materialized_aml_command_job(
        _source_job(),
        policy="migratable",
        dataset_transfer_mode="upload",
        environment=_environment(),
        user_assigned_identity_id="/subscriptions/sub/uai/target",
        data_reference_uris={
            "data": "https://source.blob.core.windows.net/container/data/"
        },
        source_storage_connection_available=True,
        source_storage_connection_name="source-storage",
        source_storage_connection_detail="https://source.blob.core.windows.net/",
    )

    summary = report.summary
    assert summary["policyPassed"] is True
    assert summary["canMigrateCurrentInvocation"] is True
    assert summary["canMigrateDefinitionLosslessly"] is False
    assert summary["canReferenceAllDependencies"] is False
    assert "output.results" in summary["lossyOrUnknownCapabilityIds"]
    assert "asset.code" in summary["nonReferenceableDependencyIds"]
    assert "input.model" in summary["nonReferenceableDependencyIds"]
    assert summary["countsBySemanticFidelity"] == dict(
        sorted(Counter(item.semantic_fidelity for item in report.capabilities).items())
    )


def test_reference_only_policy_names_copy_required_code_and_model():
    report = analyze_materialized_aml_command_job(
        _source_job(),
        policy="reference-only",
        dataset_transfer_mode="reference",
        environment=_environment(),
        user_assigned_identity_id="/subscriptions/sub/uai/target",
        data_reference_uris={
            "data": "https://source.blob.core.windows.net/container/data/"
        },
        source_storage_connection_available=True,
        source_storage_connection_name="source-storage",
        source_storage_connection_detail="https://source.blob.core.windows.net/",
    )

    assert report.policy_passed is False
    assert report.summary["blockingCapabilityIds"] == []
    assert set(report.summary["nonReferenceableDependencyIds"]) == {
        "asset.code",
        "input.model",
    }
    actions = {item.capability_id: item.selected_action for item in report.capabilities}
    assert actions["input.data"] == "reference"
    assert actions["asset.environment"] == "reference"
    assert actions["asset.code"] == "copy"
    assert actions["input.model"] == "copy"


def test_reference_only_passes_when_every_external_dependency_is_referenceable():
    source_job = _source_job()
    source_job["code"] = None
    source_job["inputs"].pop("model")
    report = analyze_materialized_aml_command_job(
        source_job,
        policy="reference-only",
        dataset_transfer_mode="reference",
        environment=_environment(),
        user_assigned_identity_id="/subscriptions/sub/uai/target",
        data_reference_uris={
            "data": "https://source.blob.core.windows.net/container/data/"
        },
        source_storage_connection_available=True,
        source_storage_connection_name="source-storage",
        source_storage_connection_detail="https://source.blob.core.windows.net/",
    )

    assert report.policy_passed is True
    assert report.summary["canReferenceAllDependencies"] is True


def test_unsupported_input_blocks_migratable_policy():
    source_job = _source_job()
    source_job["inputs"]["unsupported"] = {
        "type": "spark_dataframe",
        "path": "azureml:unsupported:1",
    }
    report = analyze_materialized_aml_command_job(
        source_job,
        policy="migratable",
        environment=_environment(),
        user_assigned_identity_id="/subscriptions/sub/uai/target",
    )

    assert report.policy_passed is False
    assert "input.unsupported" in report.summary["blockingCapabilityIds"]


def test_reference_policy_rejects_connection_for_different_storage_host():
    source_job = _source_job()
    source_job["code"] = None
    source_job["inputs"].pop("model")
    report = analyze_materialized_aml_command_job(
        source_job,
        policy="reference-only",
        dataset_transfer_mode="reference",
        environment=_environment(),
        user_assigned_identity_id="/subscriptions/sub/uai/target",
        data_reference_uris={
            "data": "https://source.blob.core.windows.net/container/data/"
        },
        source_storage_connection_available=True,
        source_storage_connection_name="wrong-storage",
        source_storage_connection_detail="https://other.blob.core.windows.net/",
    )

    assert report.policy_passed is False
    assert "input.data" in report.summary["blockingCapabilityIds"]
    finding = next(
        item for item in report.capabilities if item.capability_id == "input.data"
    )
    assert "matches source.blob.core.windows.net" in str(finding.remediation)


def test_environment_build_requirement_blocks_concrete_migration():
    report = analyze_materialized_aml_command_job(
        _source_job(),
        policy="migratable",
        environment=EnvironmentInspection(
            source_reference="azureml:environment:1",
            image_reference="private.azurecr.io/training/base:1",
            issue="AML environment has a Conda overlay.",
            has_conda_overlay=True,
        ),
        user_assigned_identity_id="/subscriptions/sub/uai/target",
    )

    assert report.policy_passed is False
    assert "asset.environment" in report.summary["blockingCapabilityIds"]


def _permission_inspection(status: str) -> RuntimePermissionInspection:
    return RuntimePermissionInspection(
        identity_resource_id="/subscriptions/sub/uai/target",
        principal_id="principal-id",
        client_id="client-id",
        checks=(
            PermissionCheck(
                requirement_id="rbac.target_storage",
                capability="Foundry output storage data-plane access",
                status=status,
                scope="/subscriptions/sub/storage",
                principal_id="principal-id",
                accepted_role_names=("Storage Blob Data Contributor",),
                reason="write outputs",
                message=f"permission is {status}",
            ),
        ),
    )


@pytest.mark.parametrize("status", ["missing", "unknown"])
def test_missing_or_unknown_rbac_blocks_migratable_policy(status):
    report = analyze_materialized_aml_command_job(
        _source_job(),
        policy="migratable",
        environment=_environment(),
        user_assigned_identity_id="/subscriptions/sub/uai/target",
        permission_inspection=_permission_inspection(status),
    )

    assert report.policy_passed is False
    assert "rbac.target_storage" in report.summary["blockingCapabilityIds"]
    assert report.summary["runtimePermissionsSatisfied"] is False
    assert report.summary[
        "missingPermissionIds" if status == "missing" else "unknownPermissionIds"
    ] == ["rbac.target_storage"]


def test_conditional_rbac_is_visible_without_being_falsely_satisfied():
    report = analyze_materialized_aml_command_job(
        _source_job(),
        policy="migratable",
        environment=_environment(),
        user_assigned_identity_id="/subscriptions/sub/uai/target",
        permission_inspection=_permission_inspection("conditional"),
    )

    assert report.policy_passed is True
    assert report.summary["runtimePermissionsSatisfied"] is False
    assert report.summary["conditionalPermissionIds"] == ["rbac.target_storage"]
