from __future__ import annotations

import pytest

from foundrytrainingjob.aml_command_job_migration import (
    audit_aml_command_job_compatibility,
    translate_aml_command_job,
)


AML_DATA_ID = (
    "azureml:/subscriptions/sub/resourceGroups/rg/providers/"
    "Microsoft.MachineLearningServices/workspaces/ws/data/train/versions/7"
)
AML_MODEL_ID = (
    "azureml:/subscriptions/sub/resourceGroups/rg/providers/"
    "Microsoft.MachineLearningServices/workspaces/ws/models/seed/versions/3"
)


def _source_job() -> dict:
    return {
        "name": "aml-source-job",
        "type": "command",
        "display_name": "AML source job",
        "description": "migration fixture",
        "experiment_name": "migration-e2e",
        "command": (
            "python train.py --literal ${{inputs.learning_rate}} "
            "--data ${{inputs.training_data}} --model ${{inputs.seed_model}} "
            "--output ${{outputs.scored_data}}"
        ),
        "environment_variables": {
            "STATIC_VALUE": "preserved",
            "AML_DATA_PATH": "${{inputs.training_data}}",
        },
        "inputs": {
            "learning_rate": {"type": "literal", "value": 0.01},
            "epochs": {"type": "integer", "value": 3},
            "training_data": {
                "type": "uri_folder",
                "path": AML_DATA_ID,
                "mode": "ro_mount",
            },
            "seed_model": {
                "type": "custom_model",
                "path": AML_MODEL_ID,
                "mode": "download",
            },
        },
        "outputs": {
            "scored_data": {
                "type": "uri_folder",
                "mode": "rw_mount",
                "path": "azureml://datastores/workspaceblobstore/paths/jobs/output",
            },
            "checkpoint": {"type": "custom_model", "mode": "upload"},
        },
        "resources": {"instance_count": 2, "shm_size": "4g"},
        "limits": {"timeout": 3600},
        "distribution": {
            "type": "pytorch",
            "process_count_per_instance": 1,
        },
        "tags": {"customer": "migration-test"},
        "job_tier": "standard",
        "priority": "high",
        "services": {
            "Studio": {"type": "Studio", "endpoint": "https://aml.example/run"},
            "debug": {"type": "SSH", "port": 22},
        },
    }


def test_translate_aml_command_job_rebinds_assets_and_primitives():
    result = translate_aml_command_job(
        _source_job(),
        foundry_compute_id="/subscriptions/sub/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/account/computes/cpu",
        foundry_instance_type="Singularity.D4_v3",
        environment_image_reference="mcr.microsoft.com/azureml/curated/minimal-ubuntu22.04-py39-cpu-inference:latest",
        migrated_asset_ids={
            AML_DATA_ID: "azureai://accounts/account/projects/project/data/train/versions/7",
            AML_MODEL_ID: "azureai://accounts/account/projects/project/models/seed/versions/3",
        },
        code_id="azureai://accounts/account/projects/project/data/code/versions/1",
        user_assigned_identity_id="/subscriptions/sub/resourceGroups/rg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/job-uai/",
        model_input_path_suffixes={"seed_model": "model"},
        input_path_suffixes={"training_data": "datasets/train"},
        output_asset_name_prefix="resurrected-job",
        output_asset_version="20260728123456789",
    )

    properties = result.request_body["properties"]
    assert properties["jobType"] == "Command"
    assert properties["environmentImageReference"].startswith("mcr.microsoft.com/")
    assert properties["computeId"].endswith("/computes/cpu")
    assert properties["codeId"].endswith("/data/code/versions/1")
    assert properties["userAssignedIdentityId"].endswith("/job-uai")
    assert properties["resources"] == {
        "instanceCount": 2,
        "instanceType": "Singularity.D4_v3",
        "shmSize": "4g",
    }
    assert properties["inputs"] == {
        "learning_rate": {"jobInputType": "literal", "value": "0.01"},
        "epochs": {"jobInputType": "literal", "value": "3"},
        "training_data": {
            "jobInputType": "uri_folder",
            "uri": "azureai://accounts/account/projects/project/data/train/versions/7",
            "mode": "ReadOnlyMount",
        },
        "seed_model": {
            "jobInputType": "custom_model",
            "uri": "azureai://accounts/account/projects/project/models/seed/versions/3",
            "mode": "Download",
        },
    }
    assert properties["outputs"]["scored_data"] == {
        "jobOutputType": "uri_folder",
        "mode": "ReadWriteMount",
        "assetName": "resurrected-job-scored_data",
        "assetVersion": "20260728123456789",
    }
    assert properties["outputs"]["checkpoint"]["jobOutputType"] == "custom_model"
    assert properties["outputs"]["checkpoint"]["mode"] == "ReadWriteMount"
    assert properties["environmentVariables"] == {"STATIC_VALUE": "preserved"}
    assert properties["command"].startswith(
        "export AML_DATA_PATH='${{inputs.training_data}}/datasets/train' && "
        "python train.py"
    )
    assert "--data ${{inputs.training_data}}/datasets/train" in properties["command"]
    assert "--model ${{inputs.seed_model}}/model" in properties["command"]
    assert properties["tags"]["migration.sourceJob"] == "aml-source-job"
    assert properties["limits"] == {
        "jobLimitsType": "Command",
        "timeout": "PT3600S",
    }
    assert properties["distribution"] == {
        "distributionType": "PyTorch",
        "processCountPerInstance": 1,
    }
    assert properties["queueSettings"] == {"jobTier": "Standard"}
    assert properties["priority"] == "High"
    assert properties["services"] == {"debug": {"jobServiceType": "SSH", "port": 22}}
    assert any("dropped AML datastore path" in warning for warning in result.warnings)
    assert any("moved into the command" in warning for warning in result.warnings)
    assert any("generated AML" in warning for warning in result.warnings)


def test_compatibility_audit_warns_for_unpreserved_source_semantics():
    source_job = _source_job()
    source_job.update(
        {
            "identity": "UserIdentityConfiguration",
            "parent_job_name": "pipeline-parent",
            "is_deterministic": False,
            "notification_setting": True,
            "parameters": {"legacy": "metadata"},
        }
    )
    source_job["resources"]["instance_type"] = "Standard_NC24ads_A100_v4"
    source_job["inputs"]["learning_rate"].update(
        {
            "min": 0.001,
            "max": 1.0,
            "optional": True,
            "path_on_compute": "/mnt/fixed",
        }
    )
    source_job["outputs"]["scored_data"]["early_available"] = True

    warnings = audit_aml_command_job_compatibility(
        source_job,
        user_assigned_identity_id="/subscriptions/sub/uai/target",
    )

    assert any("Job identity" in warning for warning in warnings)
    assert any("Parent job" in warning for warning in warnings)
    assert any("is_deterministic=False" in warning for warning in warnings)
    assert any("notification settings" in warning for warning in warnings)
    assert any("parameter metadata" in warning for warning in warnings)
    assert any("Source instance type" in warning for warning in warnings)
    assert any("minimum-value constraint" in warning for warning in warnings)
    assert any("maximum-value constraint" in warning for warning in warnings)
    assert any("optional-input semantics" in warning for warning in warnings)
    assert any("fixed compute path" in warning for warning in warnings)
    assert any("early-available output semantics" in warning for warning in warnings)


def test_translate_preserves_properties_and_warns_for_private_acr_tag():
    source_job = _source_job()
    source_job["properties"] = {
        "customer.setting": "enabled",
        "attempts": 3,
        "_azureml.ComputeTargetType": "amlctrain",
        "ContentSnapshotId": "source-only",
    }

    result = translate_aml_command_job(
        source_job,
        foundry_compute_id="/foundry/compute",
        foundry_instance_type="Singularity.D4_v3",
        environment_image_reference="private.azurecr.io/training/image:v7",
        migrated_asset_ids={
            AML_DATA_ID: "azureai://accounts/a/projects/p/data/train/versions/7",
            AML_MODEL_ID: "azureai://accounts/a/projects/p/models/seed/versions/3",
        },
        user_assigned_identity_id="/subscriptions/sub/uai/target",
    )

    assert result.request_body["properties"]["properties"] == {
        "customer.setting": "enabled",
        "attempts": "3",
    }
    assert any("tag-based rather than digest-pinned" in w for w in result.warnings)
    assert any("is reused in place, not copied" in w for w in result.warnings)
    assert any("AML runtime properties are not copied" in w for w in result.warnings)


def test_translate_aml_command_job_requires_migrated_aml_assets():
    with pytest.raises(ValueError, match="no migrated Foundry asset ID"):
        translate_aml_command_job(
            _source_job(),
            foundry_compute_id="/foundry/compute",
            foundry_instance_type="Singularity.D4_v3",
            environment_image_reference="mcr.microsoft.com/example:1",
        )


def test_translate_requires_mapping_for_bare_compact_aml_asset():
    source_job = _source_job()
    source_job["inputs"] = {
        "training_data": {
            "type": "uri_folder",
            "path": "training-data:7",
            "mode": "download",
        }
    }

    with pytest.raises(ValueError, match="no migrated Foundry asset ID"):
        translate_aml_command_job(
            source_job,
            foundry_compute_id="/foundry/compute",
            foundry_instance_type="Singularity.D4_v3",
            environment_image_reference="mcr.microsoft.com/example:1",
        )


def test_translate_aml_command_job_preserves_public_url_input():
    source_job = _source_job()
    source_job["inputs"] = {
        "public_file": {
            "type": "uri_file",
            "path": "https://example.test/data.csv",
            "mode": "direct",
        }
    }

    result = translate_aml_command_job(
        source_job,
        foundry_compute_id="/foundry/compute",
        foundry_instance_type="Singularity.D4_v3",
        environment_image_reference="mcr.microsoft.com/example:1",
        migrated_asset_ids={
            AML_DATA_ID: "azureai://accounts/a/projects/p/data/train/versions/7",
            AML_MODEL_ID: "azureai://accounts/a/projects/p/models/seed/versions/3",
        },
    )

    assert result.request_body["properties"]["inputs"]["public_file"] == {
        "jobInputType": "uri_file",
        "uri": "https://example.test/data.csv",
        "mode": "Direct",
    }


def test_translate_templated_env_when_source_command_already_expands_it():
    source_job = _source_job()
    source_job["command"] = (
        'export AML_DATA_PATH="${{inputs.training_data}}" && '
        "python train.py --data ${{inputs.training_data}}"
    )
    source_job["environment_variables"] = {
        "STATIC_VALUE": "preserved",
        "AML_DATA_PATH": "${{inputs.training_data}}",
    }

    result = translate_aml_command_job(
        source_job,
        foundry_compute_id="/foundry/compute",
        foundry_instance_type="Singularity.D4_v3",
        environment_image_reference="mcr.microsoft.com/example:1",
        migrated_asset_ids={
            AML_DATA_ID: "azureai://accounts/a/projects/p/data/train/versions/7",
            AML_MODEL_ID: "azureai://accounts/a/projects/p/models/seed/versions/3",
        },
    )

    properties = result.request_body["properties"]
    assert properties["environmentVariables"] == {"STATIC_VALUE": "preserved"}
    assert properties["command"].count("export AML_DATA_PATH=") == 2
    assert properties["command"].endswith(
        "python train.py --data ${{inputs.training_data}}"
    )


def test_translate_skips_unreferenced_generated_default_output():
    source_job = _source_job()
    source_job["outputs"]["default"] = {
        "type": "uri_folder",
        "mode": "rw_mount",
        "path": (
            "azureml://datastores/workspaceartifactstore/"
            "ExperimentRun/dcid.source-job"
        ),
    }

    result = translate_aml_command_job(
        source_job,
        foundry_compute_id="/foundry/compute",
        foundry_instance_type="Singularity.D4_v3",
        environment_image_reference="mcr.microsoft.com/example:1",
        migrated_asset_ids={
            AML_DATA_ID: "azureai://accounts/a/projects/p/data/train/versions/7",
            AML_MODEL_ID: "azureai://accounts/a/projects/p/models/seed/versions/3",
        },
    )

    assert "default" not in result.request_body["properties"]["outputs"]
    assert any("generated AML run-artifact" in warning for warning in result.warnings)


def test_translate_renames_referenced_default_output():
    source_job = _source_job()
    source_job["outputs"] = {"default": {"type": "uri_folder", "mode": "upload"}}
    source_job["command"] = "python train.py --out ${{outputs.default}}"
    source_job["environment_variables"] = {"OUTPUT_PATH": "${{outputs.default}}"}

    result = translate_aml_command_job(
        source_job,
        foundry_compute_id="/foundry/compute",
        foundry_instance_type="Singularity.D4_v3",
        environment_image_reference="mcr.microsoft.com/example:1",
        migrated_asset_ids={
            AML_DATA_ID: "azureai://accounts/a/projects/p/data/train/versions/7",
            AML_MODEL_ID: "azureai://accounts/a/projects/p/models/seed/versions/3",
        },
    )

    properties = result.request_body["properties"]
    assert set(properties["outputs"]) == {"aml_default"}
    assert "${{outputs.aml_default}}" in properties["command"]
    assert "${{outputs.default}}" not in properties["command"]
    assert any("renamed to 'aml_default'" in warning for warning in result.warnings)


def test_translate_omits_null_queue_tier_sentinel():
    source_job = _source_job()
    source_job["job_tier"] = "null"

    result = translate_aml_command_job(
        source_job,
        foundry_compute_id="/foundry/compute",
        foundry_instance_type="Singularity.D4_v3",
        environment_image_reference="mcr.microsoft.com/example:1",
        migrated_asset_ids={
            AML_DATA_ID: "azureai://accounts/a/projects/p/data/train/versions/7",
            AML_MODEL_ID: "azureai://accounts/a/projects/p/models/seed/versions/3",
        },
    )

    assert "queueSettings" not in result.request_body["properties"]


def test_translate_uri_file_output_appends_source_file_name():
    source_job = _source_job()
    source_job["inputs"] = {}
    source_job["outputs"] = {
        "summary": {
            "type": "uri_file",
            "mode": "upload",
            "path": ("azureml://datastores/identity/paths/run/summary/summary.json"),
        }
    }
    source_job["command"] = "python write.py --summary ${{outputs.summary}}"

    result = translate_aml_command_job(
        source_job,
        foundry_compute_id="/foundry/compute",
        foundry_instance_type="Singularity.D4_v3",
        environment_image_reference="mcr.microsoft.com/example:1",
    )

    assert result.request_body["properties"]["command"].endswith(
        "--summary ${{outputs.summary}}/summary.json"
    )
    assert any(
        "resolves uri_file output placeholders to directories" in warning
        for warning in result.warnings
    )
