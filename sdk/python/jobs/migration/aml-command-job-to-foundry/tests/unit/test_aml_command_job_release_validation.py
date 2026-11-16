from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from foundrytrainingjob.aml_command_job_release_validation import (
    ReleaseValidationRequest,
    _implementation_evidence,
    _write_json,
    run_aml_foundry_release_validation,
)
from foundrytrainingjob.aml_command_job_migrator import AmlWorkspace, FoundryTarget
import foundrytrainingjob.aml_command_job_release_validation as release_module


_SUMMARY = {
    "fixture": "aml-foundry-command-job-migration",
    "epochs": 3,
    "learningRate": 0.125,
}
_OUTPUT_DIGESTS = {
    "results": "results-digest",
    "summary": "summary-digest",
    "metrics_table": "metrics-digest",
    "trained_model": "model-digest",
}


def test_implementation_evidence_includes_migration_safety_surfaces():
    paths = {item["path"] for item in _implementation_evidence()}

    assert "aml_command_job_migration_cli.py" in paths
    assert "aml_command_job_migrator.py" in paths
    assert "e2e/sanitization.py" in paths


def test_release_json_writer_redacts_sensitive_values(tmp_path):
    report_path = tmp_path / "release.json"

    _write_json(
        report_path,
        {
            "sourceUri": (
                "https://source.blob.core.windows.net/data?sig=source-secret"
            ),
            "API_TOKEN": "token-secret",
        },
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert "source-secret" not in report_text
    assert "token-secret" not in report_text
    assert "<redacted>" in report_text


class _FakeResult:
    def __init__(self, request, *, source_version: str):
        mode = request.dataset_transfer_mode
        manifest_path = Path(request.work_dir) / "migration" / "migration-manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_inputs = {}
        for name, input_type in {
            "training_data": "uri_folder",
            "config_file": "uri_file",
            "training_table": "mltable",
        }.items():
            record = {
                "assetName": f"{mode}-{name}",
                "assetVersion": request.asset_version,
                "foundryAssetId": f"azureai://data/{mode}-{name}/{request.asset_version}",
                "inputType": input_type,
            }
            if mode == "reference":
                record.update(
                    {
                        "transferMode": "reference",
                        "referenceDataUri": (
                            f"https://source.blob.core.windows.net/data/{name}"
                        ),
                    }
                )
                if input_type != "uri_file":
                    record.update(
                        {
                            "registeredDataUri": (
                                "https://source.blob.core.windows.net/data"
                            ),
                            "foundryInputPathSuffix": name,
                        }
                    )
            else:
                record["localPath"] = str(Path(request.work_dir) / "inputs" / name)
            manifest_inputs[name] = record
        manifest_path.write_text(
            json.dumps({"inputs": manifest_inputs}),
            encoding="utf-8",
        )
        inputs = {
            "epochs": {"jobInputType": "literal", "value": "3"},
            "learning_rate": {"jobInputType": "literal", "value": "0.125"},
            "message": {
                "jobInputType": "literal",
                "value": "resurrected-command-job",
            },
            "enabled": {"jobInputType": "literal", "value": "True"},
            "training_data": {
                "jobInputType": "uri_folder",
                "uri": "azureai://data/training-data/1",
            },
            "config_file": {
                "jobInputType": "uri_file",
                "uri": "azureai://data/config/1",
            },
            "training_table": {
                "jobInputType": "uri_folder" if mode == "reference" else "mltable",
                "uri": "azureai://data/table/1",
            },
            "seed_model": {
                "jobInputType": "custom_model",
                "uri": "azureai://models/seed/1",
            },
        }
        outputs = {
            name: {
                "jobOutputType": output_type,
                "assetName": f"{mode}-{name}",
                "assetVersion": request.asset_version,
            }
            for name, output_type in {
                "results": "uri_folder",
                "summary": "uri_file",
                "metrics_table": "uri_folder",
                "trained_model": "custom_model",
            }.items()
        }
        evidence = SimpleNamespace(
            output_values={name: _SUMMARY for name in outputs},
        )
        self.fixture = SimpleNamespace(
            job_name="aml-source-job",
            status="Completed",
            asset_version=source_version,
        )
        self.migration = SimpleNamespace(
            source_job_name="aml-source-job",
            target_job_name=request.target_job_name,
            target_status="Completed",
            manifest_path=str(manifest_path),
            request_body={
                "properties": {
                    "inputs": inputs,
                    "outputs": outputs,
                    "codeId": "azureai://data/code/1",
                    "environmentImageReference": "mcr.microsoft.com/image:1",
                    "computeId": "/computes/cpu",
                    "userAssignedIdentityId": "/identities/uai",
                    "environmentVariables": {"STATIC_SETTING": "preserved"},
                    "command": "export DATA_FROM_ENV='${{inputs.training_data}}'",
                    "resources": {"instanceCount": 1, "shmSize": "2g"},
                    "limits": {"timeout": "PT3600S"},
                }
            },
        )
        self.validation = SimpleNamespace(
            metrics_table_dataset_id="azureai://data/metrics/1"
        )
        self.source_evidence = evidence
        self.target_evidence = evidence
        self.equivalence = SimpleNamespace(
            equivalent=True,
            logs_equivalent=True,
            outputs_equivalent=True,
            source_output_digests=dict(_OUTPUT_DIGESTS),
            target_output_digests=dict(_OUTPUT_DIGESTS),
        )

    def to_dict(self):
        return {"fixture": {"jobName": self.fixture.job_name}}


class _FakeAnalysis:
    policy = "migratable"
    policy_passed = True
    summary = {
        "runtimePermissionsSatisfied": True,
        "missingPermissionIds": [],
        "unknownPermissionIds": [],
        "familiesWithoutLiveEvidence": [],
        "countsBySupport": {"supported": 1},
        "countsBySemanticFidelity": {"equivalent": 1},
    }

    def to_dict(self):
        return {"policy": self.policy, "summary": self.summary}


def test_release_validation_reuses_one_source_and_isolates_target_versions(
    tmp_path,
    monkeypatch,
):
    versions = iter(("20260731120000001", "20260731120000002"))
    monkeypatch.setattr(
        release_module,
        "create_foundry_asset_version",
        lambda: next(versions),
    )
    requests = []
    created_fixtures = []

    def fake_create_fixture(request, **kwargs):
        created_fixtures.append(request)
        return SimpleNamespace(
            job_name="aml-source-job",
            status="Completed",
            asset_version=request.asset_version,
        )

    monkeypatch.setattr(
        release_module,
        "create_aml_migration_fixture",
        fake_create_fixture,
    )

    def fake_run(request, **kwargs):
        requests.append(request)
        result = _FakeResult(request, source_version="20260731120000001")
        report_path = Path(request.work_dir) / "equivalence-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
        return result

    monkeypatch.setattr(
        release_module,
        "run_aml_foundry_equivalence_exercise",
        fake_run,
    )
    monkeypatch.setattr(
        release_module,
        "analyze_aml_command_job",
        lambda **kwargs: _FakeAnalysis(),
    )
    monkeypatch.setattr(
        release_module,
        "_inspect_dataset_bindings",
        lambda result, **kwargs: {
            name: {"connectionName": kwargs["expected_connection_name"]}
            for name in ("training_data", "config_file", "training_table")
        },
    )
    request = ReleaseValidationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://account.services.ai.azure.com",
            project_name="project",
            storage_connection_name="project-storage",
            compute_id="/computes/cpu",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
            user_assigned_identity_id="/identities/uai",
        ),
        source_storage_connection_name="aml-source-storage",
        work_dir=tmp_path,
    )

    result = run_aml_foundry_release_validation(
        request,
        credential=object(),
        ml_client=object(),
        emit=lambda message: None,
        sleeper=lambda seconds: None,
    )

    assert len(requests) == 2
    assert len(created_fixtures) == 1
    assert requests[0].dataset_transfer_mode == "upload"
    assert requests[0].existing_source_job_name == "aml-source-job"
    assert requests[0].existing_source_asset_version == "20260731120000001"
    assert requests[1].dataset_transfer_mode == "reference"
    assert requests[1].existing_source_job_name == "aml-source-job"
    assert requests[1].existing_source_asset_version == "20260731120000001"
    assert requests[1].asset_version == "20260731120000002"
    assert requests[0].asset_version != requests[1].asset_version
    assert result.report["status"] == "passed"
    assert result.report["releaseDecision"]["qualifiedScopeReady"] is True
    assert result.report["releaseDecision"]["unrestrictedCustomerReleaseReady"] is False
    assert Path(result.report_path).is_file()
    assert result.report["evidenceFiles"]


def test_release_validation_persists_failure_decision(tmp_path):
    request = ReleaseValidationRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://account.services.ai.azure.com",
            project_name="project",
            storage_connection_name="project-storage",
            compute_id="/computes/cpu",
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        source_storage_connection_name="",
        work_dir=tmp_path,
    )

    try:
        run_aml_foundry_release_validation(
            request, credential=object(), ml_client=object()
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected missing source connection to fail.")

    report = json.loads(
        (tmp_path / "release-validation-report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert report["releaseDecision"]["qualifiedScopeReady"] is False
