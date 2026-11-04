from __future__ import annotations

from types import SimpleNamespace

import foundrytrainingjob.aml_command_job_migration_cli as cli_module
from foundrytrainingjob.aml_command_job_equivalence import (
    JobEquivalenceReport,
    JobEvidence,
    MigrationEquivalenceResult,
)
from foundrytrainingjob.aml_command_job_fixture import (
    FixtureResult,
    FixtureValidationResult,
)
from foundrytrainingjob.aml_command_job_migrator import MigrationResult
from foundrytrainingjob.aml_command_job_release_validation import (
    ReleaseValidationResult,
)


def test_exercise_cli_runs_fixture_migration_and_validation(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli_module, "AzureCliCredential", lambda **kwargs: object())
    fixture = FixtureResult(
        job_name="source-job",
        status="Completed",
        asset_version="20260728123456789",
        data_asset_id="azureml:data:1",
        file_asset_id="azureml:file:1",
        mltable_asset_id="azureml:table:1",
        model_asset_id="azureml:model:1",
        work_dir=str(tmp_path / "source-fixture"),
        expected_summary={
            "fixture": "aml-foundry-command-job-migration",
            "trainingDataDigest": "data",
            "trainingTableDigest": "table",
            "seedModelDigest": "model",
        },
    )
    migration = MigrationResult(
        manifest_path=str(tmp_path / "manifest.json"),
        source_job_name="source-job",
        target_job_name="target-job",
        target_status="Completed",
        request_body={"properties": {"outputs": {}}},
        asset_mappings={"azureml:data:1": "azureai://data/1"},
        warnings=("translated",),
    )

    source_evidence = JobEvidence(
        service="AML",
        job_name="source-job",
        status="Completed",
        log_root=str(tmp_path / "aml-logs"),
        output_root=str(tmp_path / "aml-outputs"),
        log_files=("stream.log",),
        output_files=("result.json", "summary.json", "model.json"),
        application_records=({"fixture": "aml-foundry-command-job-migration"},),
        output_values={
            "results": {"fixture": "aml-foundry-command-job-migration"},
            "summary": {"fixture": "aml-foundry-command-job-migration"},
            "trained_model": {"source": "model", "epochs": 3},
        },
    )
    target_evidence = JobEvidence(
        **{
            **source_evidence.__dict__,
            "service": "Foundry",
            "job_name": "target-job",
            "log_root": str(tmp_path / "foundry-logs"),
            "output_root": str(tmp_path / "validation"),
        }
    )
    validation = FixtureValidationResult(
        summary={"fixture": "aml-foundry-command-job-migration"},
        results_dataset_id="azureai://results/1",
        summary_dataset_id="azureai://summary/1",
        metrics_table_dataset_id="azureai://metrics/1",
        model_asset_id="azureai://model-output/1",
        model_downloaded_files=("model.json",),
        model_total_bytes=10,
        validation_dir=str(tmp_path / "validation"),
    )
    equivalence = JobEquivalenceReport(
        source_job_name="source-job",
        target_job_name="target-job",
        logs_equivalent=True,
        outputs_equivalent=True,
        equivalent=True,
        source_log_record_digests=("record",),
        target_log_record_digests=("record",),
        source_output_digests={"results": "digest"},
        target_output_digests={"results": "digest"},
    )
    monkeypatch.setattr(
        cli_module,
        "run_aml_foundry_equivalence_exercise",
        lambda *args, **kwargs: MigrationEquivalenceResult(
            fixture=fixture,
            migration=migration,
            validation=validation,
            source_evidence=source_evidence,
            target_evidence=target_evidence,
            equivalence=equivalence,
        ),
    )

    exit_code = cli_module.main(
        [
            "exercise",
            "--source-resource-group",
            "rg",
            "--source-workspace",
            "ws",
            "--source-export-compute",
            "cpu",
            "--work-dir",
            str(tmp_path),
            "--poll-interval",
            "1",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"sourceJobName": "source-job"' in output
    assert '"targetJobName": "target-job"' in output
    assert '"model_asset_id": "azureai://model-output/1"' in output
    assert '"equivalent": true' in output


def test_migrate_parser_requires_source_job():
    parser = cli_module.build_parser()
    args = parser.parse_args(
        [
            "migrate",
            "--source-resource-group",
            "rg",
            "--source-workspace",
            "ws",
            "--source-export-compute",
            "cpu",
            "--source-job",
            "job",
            "--dataset-transfer-mode",
            "reference",
            "--source-storage-connection",
            "aml-source-storage",
        ]
    )

    assert args.source_job == "job"
    assert args.foundry_api_version == "2026-01-15-preview"
    assert args.target_job_tier == "Premium"
    assert args.source_identity_datastore == "foundrymigrationidentityblob"
    assert args.dataset_transfer_mode == "reference"
    assert args.source_storage_connection == "aml-source-storage"
    assert args.grant_reference_storage_access is False


def test_migrate_reference_access_grant_rechecks_preflight(
    tmp_path,
    monkeypatch,
    capsys,
):
    inspection = SimpleNamespace(
        checks=(
            SimpleNamespace(
                requirement_id="rbac.source_storage.source",
                status="missing",
            ),
        )
    )

    class FakeReport:
        def __init__(self, *, passed):
            self.policy_passed = passed
            self.permission_inspection = inspection
            self.capabilities = (
                SimpleNamespace(
                    capability_id="rbac.source_storage.source",
                    blocking=not passed,
                ),
            )

        def to_dict(self):
            return {
                "summary": {"policy": "migratable", "policyPassed": self.policy_passed}
            }

    reports = iter((FakeReport(passed=False), FakeReport(passed=True)))
    analysis_calls = []
    monkeypatch.setattr(cli_module, "AzureCliCredential", lambda **kwargs: object())
    monkeypatch.setattr(
        cli_module,
        "_run_analyze",
        lambda args, credential, *, policy=None: (
            analysis_calls.append(policy) or next(reports)
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "grant_missing_reference_storage_access",
        lambda value, **kwargs: (
            SimpleNamespace(
                to_dict=lambda: {
                    "roleName": "Storage Blob Data Reader",
                    "scope": "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts/source",
                }
            ),
        ),
    )
    migration_calls = []
    monkeypatch.setattr(
        cli_module,
        "_run_migrate",
        lambda args, credential: (
            migration_calls.append(True)
            or {"migration": {"targetJobName": "target-job"}}
        ),
    )

    exit_code = cli_module.main(
        [
            "migrate",
            "--source-resource-group",
            "rg",
            "--source-workspace",
            "ws",
            "--source-export-compute",
            "cpu",
            "--source-job",
            "job",
            "--dataset-transfer-mode",
            "reference",
            "--source-storage-connection",
            "source-connection",
            "--user-assigned-identity-id",
            "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/uai",
            "--grant-reference-storage-access",
            "--work-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert analysis_calls == ["migratable", "migratable"]
    assert migration_calls == [True]
    output = capsys.readouterr().out
    assert '"roleName": "Storage Blob Data Reader"' in output
    assert '"targetJobName": "target-job"' in output
    evidence = __import__("json").loads(
        (tmp_path / "reference-storage-role-assignments.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["assignments"][0]["roleName"] == "Storage Blob Data Reader"


def test_migrate_reference_access_grant_does_not_mutate_with_other_blockers(
    monkeypatch,
    capsys,
):
    class FakeReport:
        policy_passed = False
        permission_inspection = SimpleNamespace(
            checks=(
                SimpleNamespace(
                    requirement_id="rbac.source_storage.source",
                    status="missing",
                ),
            )
        )
        capabilities = (
            SimpleNamespace(
                capability_id="rbac.source_storage.source",
                blocking=True,
            ),
            SimpleNamespace(capability_id="input.unsupported", blocking=True),
        )

        def to_dict(self):
            return {"summary": {"policy": "migratable", "policyPassed": False}}

    monkeypatch.setattr(cli_module, "AzureCliCredential", lambda **kwargs: object())
    monkeypatch.setattr(
        cli_module,
        "_run_analyze",
        lambda *args, **kwargs: FakeReport(),
    )
    monkeypatch.setattr(
        cli_module,
        "grant_missing_reference_storage_access",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("RBAC must not change for an unsupported migration")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_run_migrate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Migration must not start")
        ),
    )

    exit_code = cli_module.main(
        [
            "migrate",
            "--source-resource-group",
            "rg",
            "--source-workspace",
            "ws",
            "--source-export-compute",
            "cpu",
            "--source-job",
            "job",
            "--dataset-transfer-mode",
            "reference",
            "--source-storage-connection",
            "source-connection",
            "--user-assigned-identity-id",
            "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/uai",
            "--grant-reference-storage-access",
        ]
    )

    assert exit_code == 2
    assert '"referenceStorageRoleAssignments": []' in capsys.readouterr().out


def test_default_work_dir_uses_configured_external_runs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AML_MIGRATION_RUNS_DIR", str(tmp_path))

    work_dir = cli_module._default_work_dir("job/name")

    assert work_dir == tmp_path / "job-name"


def test_qualify_release_cli_writes_cost_bounded_result(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(cli_module, "AzureCliCredential", lambda **kwargs: object())
    captured = {}

    def fake_qualify(request, **kwargs):
        captured["request"] = request
        return ReleaseValidationResult(
            report_path=str(tmp_path / "release-validation-report.json"),
            report={
                "status": "passed",
                "releaseDecision": {
                    "qualifiedScopeReady": True,
                    "unrestrictedCustomerReleaseReady": False,
                },
            },
        )

    monkeypatch.setattr(
        cli_module,
        "run_aml_foundry_release_validation",
        fake_qualify,
    )

    exit_code = cli_module.main(
        [
            "qualify-release",
            "--source-resource-group",
            "rg",
            "--source-workspace",
            "ws",
            "--source-export-compute",
            "cpu",
            "--source-storage-connection",
            "aml-source-storage",
            "--work-dir",
            str(tmp_path),
            "--reuse-completed-upload",
            "--poll-interval",
            "1",
        ]
    )

    assert exit_code == 0
    assert captured["request"].source_storage_connection_name == ("aml-source-storage")
    assert captured["request"].work_dir == tmp_path
    assert captured["request"].reuse_completed_upload is True
    output = capsys.readouterr().out
    assert '"qualifiedScopeReady": true' in output
    assert '"unrestrictedCustomerReleaseReady": false' in output


def test_analyze_parser_does_not_require_export_compute():
    args = cli_module.build_parser().parse_args(
        [
            "analyze",
            "--source-resource-group",
            "rg",
            "--source-workspace",
            "ws",
            "--source-job",
            "job",
            "--analysis-policy",
            "reference-only",
        ]
    )

    assert args.source_export_compute == ""
    assert args.analysis_policy == "reference-only"


def test_analyze_cli_returns_two_when_policy_fails(
    tmp_path,
    monkeypatch,
    capsys,
):
    captured = {}

    class FakeReport:
        policy_passed = False

        def to_dict(self):
            return {
                "summary": {
                    "policy": "reference-only",
                    "policyPassed": False,
                    "nonReferenceableDependencyIds": ["asset.code"],
                }
            }

    monkeypatch.setattr(cli_module, "AzureCliCredential", lambda **kwargs: object())

    def fake_analyze(**kwargs):
        captured.update(kwargs)
        return FakeReport()

    monkeypatch.setattr(cli_module, "analyze_aml_command_job", fake_analyze)
    report_path = tmp_path / "analysis.json"

    exit_code = cli_module.main(
        [
            "analyze",
            "--source-resource-group",
            "rg",
            "--source-workspace",
            "ws",
            "--source-job",
            "job",
            "--analysis-policy",
            "reference-only",
            "--report-file",
            str(report_path),
        ]
    )

    assert exit_code == 2
    assert captured["source_job_name"] == "job"
    assert captured["policy"] == "reference-only"
    assert captured["source"].export_compute == ""
    assert report_path.is_file()
    report = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    assert report["analysis"]["summary"]["policyPassed"] is False
    assert '"policyPassed": false' in capsys.readouterr().out


def test_migrate_preflight_failure_creates_nothing(monkeypatch, capsys):
    class FakeReport:
        policy_passed = False

        def to_dict(self):
            return {
                "summary": {
                    "policy": "reference-only",
                    "policyPassed": False,
                    "nonReferenceableDependencyIds": ["asset.code"],
                }
            }

    monkeypatch.setattr(cli_module, "AzureCliCredential", lambda **kwargs: object())
    monkeypatch.setattr(
        cli_module,
        "analyze_aml_command_job",
        lambda **kwargs: FakeReport(),
    )
    monkeypatch.setattr(
        cli_module,
        "AmlCommandJobMigrator",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("migration must not start after a failed preflight")
        ),
    )

    exit_code = cli_module.main(
        [
            "migrate",
            "--source-resource-group",
            "rg",
            "--source-workspace",
            "ws",
            "--source-export-compute",
            "cpu",
            "--source-job",
            "job",
            "--preflight-policy",
            "reference-only",
        ]
    )

    assert exit_code == 2
    assert '"policyPassed": false' in capsys.readouterr().out
