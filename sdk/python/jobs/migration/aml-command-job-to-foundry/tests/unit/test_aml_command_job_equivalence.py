from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from foundrytrainingjob.aml_command_job_equivalence import (
    JobEvidence,
    JobEquivalenceReport,
    MigrationEquivalenceRequest,
    compare_job_evidence,
    extract_application_records,
    run_aml_foundry_equivalence_exercise,
    snapshot_fixture_outputs,
)
from foundrytrainingjob.aml_command_job_fixture import (
    FixtureResult,
    FixtureValidationResult,
)
from foundrytrainingjob.aml_command_job_migrator import (
    AmlWorkspace,
    FoundryTarget,
    MigrationResult,
)
import foundrytrainingjob.aml_command_job_equivalence as equivalence_module


SUMMARY = {
    "fixture": "aml-foundry-command-job-migration",
    "epochs": 3,
    "learningRate": 0.125,
    "message": "resurrected-command-job",
    "enabled": True,
    "trainingDataDigest": "data-digest",
    "trainingTableDigest": "table-digest",
    "seedModelDigest": "model-digest",
}
MODEL = {"source": "model-digest", "epochs": 3}


def _evidence(
    service: str,
    *,
    records: tuple[dict, ...] = (SUMMARY,),
    outputs: dict | None = None,
) -> JobEvidence:
    output_values = outputs or {
        "results": SUMMARY,
        "summary": SUMMARY,
        "metrics_table": SUMMARY,
        "trained_model": MODEL,
    }
    return JobEvidence(
        service=service,
        job_name=f"{service.lower()}-job",
        status="Completed",
        log_root=f"/{service}/logs",
        output_root=f"/{service}/outputs",
        log_files=("raw.log",),
        output_files=("result.json", "summary.json", "metrics.jsonl", "model.json"),
        application_records=records,
        output_values=output_values,
        log_digests={"raw.log": f"raw-{service}"},
        output_digests={name: f"digest-{name}" for name in output_values},
    )


def test_extract_application_records_ignores_platform_wrappers_and_duplicates():
    payload = json.dumps(SUMMARY, sort_keys=True)
    aml_log = (
        "2026-07-28T00:00:00Z platform setup\n"
        "MIGRATION_FIXTURE_COMPLETED\n"
        f"MIGRATION_FIXTURE_RECORD:{payload}\n"
    )
    foundry_log = (
        "[rank=0] runtime prefix\n"
        f"stdout | MIGRATION_FIXTURE_RECORD:{payload}\n"
        f"stdout | MIGRATION_FIXTURE_RECORD:{payload}\n"
    )

    assert extract_application_records((aml_log,)) == (SUMMARY,)
    assert extract_application_records((foundry_log,)) == (SUMMARY,)


def test_extract_application_records_rejects_malformed_json():
    with pytest.raises(ValueError, match="Malformed MIGRATION_FIXTURE_RECORD"):
        extract_application_records(("MIGRATION_FIXTURE_RECORD:{bad-json}",))


def test_snapshot_fixture_outputs_normalizes_service_path_layout(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "summary").mkdir()
    (tmp_path / "metrics_table").mkdir()
    (tmp_path / "trained_model").mkdir()
    (tmp_path / "results" / "result.json").write_text(
        json.dumps(SUMMARY, indent=2), encoding="utf-8"
    )
    (tmp_path / "summary" / "summary.json").write_text(
        json.dumps(SUMMARY, separators=(",", ":")), encoding="utf-8"
    )
    (tmp_path / "metrics_table" / "metrics.jsonl").write_text(
        json.dumps(SUMMARY) + "\n", encoding="utf-8"
    )
    (tmp_path / "trained_model" / "model.json").write_text(
        json.dumps(MODEL), encoding="utf-8"
    )

    values, files = snapshot_fixture_outputs(tmp_path)

    assert values == {
        "results": SUMMARY,
        "summary": SUMMARY,
        "metrics_table": SUMMARY,
        "trained_model": MODEL,
    }
    assert files == (
        "metrics_table\\metrics.jsonl",
        "results\\result.json",
        "summary\\summary.json",
        "trained_model\\model.json",
    )


def test_compare_job_evidence_accepts_different_raw_logs_with_same_records():
    report = compare_job_evidence(_evidence("AML"), _evidence("Foundry"))

    assert report.logs_equivalent is True
    assert report.outputs_equivalent is True
    assert report.equivalent is True
    assert report.source_log_record_digests == report.target_log_record_digests
    report.assert_equivalent()


def test_compare_job_evidence_reports_log_and_output_mismatches():
    changed_summary = {**SUMMARY, "epochs": 4}
    target_outputs = {
        "results": changed_summary,
        "summary": SUMMARY,
        "metrics_table": SUMMARY,
        "trained_model": MODEL,
    }
    report = compare_job_evidence(
        _evidence("AML"),
        _evidence("Foundry", records=(changed_summary,), outputs=target_outputs),
    )

    assert report.equivalent is False
    assert report.logs_equivalent is False
    assert report.outputs_equivalent is False
    assert "canonical application log records differ" in report.log_mismatches[0]
    assert "'results'" in report.output_mismatches[0]
    with pytest.raises(AssertionError, match="AML and Foundry job evidence differs"):
        report.assert_equivalent()


def test_shared_exercise_uses_observed_aml_output_as_foundry_oracle(
    tmp_path,
    monkeypatch,
):
    fixture = FixtureResult(
        job_name="source-job",
        status="Completed",
        asset_version="20260728123456789",
        data_asset_id="azureml:data:1",
        file_asset_id="azureml:file:1",
        mltable_asset_id="azureml:table:1",
        model_asset_id="azureml:model:1",
        work_dir=str(tmp_path / "fixture"),
        expected_summary=SUMMARY,
    )
    migration = MigrationResult(
        manifest_path=str(tmp_path / "manifest.json"),
        source_job_name="source-job",
        target_job_name="target-job",
        target_status="Completed",
        request_body={"properties": {"outputs": {}}},
        asset_mappings={
            "https://source.blob.core.windows.net/data?sig=source-secret": (
                "azureai://data/training/versions/1"
            )
        },
        warnings=("token=warning-secret",),
    )
    validation = FixtureValidationResult(
        summary=SUMMARY,
        results_dataset_id="azureai://results/1",
        summary_dataset_id="azureai://summary/1",
        metrics_table_dataset_id="azureai://metrics/1",
        model_asset_id="azureai://model/1",
        model_downloaded_files=("model.json",),
        model_total_bytes=10,
        validation_dir=str(tmp_path / "validation"),
    )
    source_evidence = _evidence("AML")
    target_evidence = _evidence("Foundry")
    report = compare_job_evidence(source_evidence, target_evidence)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        equivalence_module,
        "_reuse_completed_fixture",
        lambda *args, **kwargs: fixture,
    )
    monkeypatch.setattr(
        equivalence_module,
        "collect_aml_fixture_evidence",
        lambda *args, **kwargs: source_evidence,
    )

    class FakeMigrator:
        def __init__(self, request, **kwargs):
            captured["migration_request"] = request

        def migrate(self):
            return migration

    monkeypatch.setattr(equivalence_module, "AmlCommandJobMigrator", FakeMigrator)

    def fake_validate(*args, **kwargs):
        captured["expected_summary"] = kwargs["expected_summary"]
        return validation

    monkeypatch.setattr(
        equivalence_module,
        "validate_migrated_fixture",
        fake_validate,
    )
    monkeypatch.setattr(
        equivalence_module,
        "collect_foundry_fixture_evidence",
        lambda *args, **kwargs: target_evidence,
    )
    monkeypatch.setattr(
        equivalence_module,
        "compare_job_evidence",
        lambda *args, **kwargs: report,
    )
    request = MigrationEquivalenceRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu"),
        target=FoundryTarget(
            project_endpoint="https://example.test",
            project_name="project",
            storage_connection_name="storage",
            compute_id=(
                "/subscriptions/sub/resourceGroups/rg/providers/"
                "Microsoft.CognitiveServices/accounts/a/computes/cpu"
            ),
            instance_type="Singularity.D4_v3",
            api_version="2026-01-15-preview",
        ),
        work_dir=tmp_path,
        asset_version="20260729123456789",
        dataset_transfer_mode="reference",
        source_storage_connection_name="aml-source-storage",
        existing_source_job_name="source-job",
        existing_source_asset_version="20260728123456789",
    )

    result = run_aml_foundry_equivalence_exercise(
        request,
        credential=object(),
        ml_client=object(),
        emit=lambda message: None,
        sleeper=lambda seconds: None,
    )

    assert result.equivalence.equivalent is True
    assert captured["expected_summary"] == SUMMARY
    assert captured["migration_request"].source_job_name == "source-job"
    assert captured["migration_request"].asset_version == "20260729123456789"
    assert captured["migration_request"].dataset_transfer_mode == "reference"
    assert (
        captured["migration_request"].source_storage_connection_name
        == "aml-source-storage"
    )
    report = json.loads(
        (tmp_path / "equivalence-report.json").read_text(encoding="utf-8")
    )
    report_text = (tmp_path / "equivalence-report.json").read_text(encoding="utf-8")
    assert "source-secret" not in report_text
    assert "warning-secret" not in report_text
    assert "<redacted>" in report_text
    assert report["equivalence"]["equivalent"] is True
    assert report["sourceEvidence"]["log_root"] == source_evidence.log_root
    assert report["targetEvidence"]["log_root"] == target_evidence.log_root
    assert (
        report["equivalence"]["source_log_record_digests"]
        == report["equivalence"]["target_log_record_digests"]
    )


def test_collect_aml_evidence_reads_user_logs_from_artifact_store(
    tmp_path,
    monkeypatch,
):
    fixture = FixtureResult(
        job_name="source-job",
        status="Completed",
        asset_version="20260728123456789",
        data_asset_id="azureml:data:1",
        file_asset_id="azureml:file:1",
        mltable_asset_id="azureml:table:1",
        model_asset_id="azureml:model:1",
        work_dir=str(tmp_path / "fixture"),
        expected_summary=SUMMARY,
    )

    class FakeJobs:
        def stream(self, name):
            print(f"RunId: {name}\nExecution Summary: Completed")

    datastores = {
        "workspaceartifactstore": SimpleNamespace(
            account_name="account",
            container_name="artifacts",
            endpoint="core.windows.net",
        ),
        "workspaceblobstore": SimpleNamespace(
            account_name="account",
            container_name="outputs",
            endpoint="core.windows.net",
        ),
    }
    client = SimpleNamespace(
        jobs=FakeJobs(),
        datastores=SimpleNamespace(get=lambda name: datastores[name]),
    )

    def fake_download(*, prefix, target_dir, **kwargs):
        target_dir.mkdir(parents=True, exist_ok=True)
        if prefix.endswith("/user_logs"):
            payload = json.dumps(SUMMARY, sort_keys=True)
            (target_dir / "std_log.txt").write_text(
                "MIGRATION_FIXTURE_COMPLETED\n" f"MIGRATION_FIXTURE_RECORD:{payload}\n",
                encoding="utf-8",
            )
            return ("std_log.txt",)
        (target_dir / "results").mkdir()
        (target_dir / "summary").mkdir()
        (target_dir / "metrics_table").mkdir()
        (target_dir / "trained_model").mkdir()
        (target_dir / "results" / "result.json").write_text(
            json.dumps(SUMMARY), encoding="utf-8"
        )
        (target_dir / "summary" / "summary.json").write_text(
            json.dumps(SUMMARY), encoding="utf-8"
        )
        (target_dir / "metrics_table" / "metrics.jsonl").write_text(
            json.dumps(SUMMARY) + "\n", encoding="utf-8"
        )
        (target_dir / "trained_model" / "model.json").write_text(
            json.dumps(MODEL), encoding="utf-8"
        )
        return (
            "results\\result.json",
            "summary\\summary.json",
            "metrics_table\\metrics.jsonl",
            "trained_model\\model.json",
        )

    monkeypatch.setattr(
        equivalence_module,
        "_download_blob_prefix_with_identity",
        fake_download,
    )

    evidence = equivalence_module.collect_aml_fixture_evidence(
        fixture,
        ml_client=client,
        credential=object(),
        work_dir=tmp_path / "evidence",
    )

    assert evidence.application_records == (SUMMARY,)
    assert "user_logs/std_log.txt" in evidence.log_files
    assert evidence.output_values["trained_model"] == MODEL
