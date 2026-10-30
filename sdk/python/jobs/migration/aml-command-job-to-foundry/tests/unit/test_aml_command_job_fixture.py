from __future__ import annotations

from types import SimpleNamespace

from foundrytrainingjob.dataset import DatasetDownloadResult
from foundrytrainingjob.model_asset import ModelAssetDownloadResult
from foundrytrainingjob.aml_command_job_fixture import (
    FixtureRequest,
    create_aml_migration_fixture,
    validate_migrated_fixture,
)
from foundrytrainingjob.aml_command_job_migrator import (
    AmlWorkspace,
    FoundryTarget,
    MigrationResult,
)
import foundrytrainingjob.aml_command_job_fixture as fixture_module


def test_create_fixture_registers_assets_and_submits_capability_job(
    tmp_path, monkeypatch
):
    created_data = []
    created_models = []
    created_codes = []

    class FakeDataOperations:
        def create_or_update(self, asset):
            created_data.append(asset)
            asset._id = f"azureml:{asset.name}:{asset.version}"
            return asset

    class FakeModelOperations:
        def create_or_update(self, asset):
            created_models.append(asset)
            asset._id = f"azureml:{asset.name}:{asset.version}"
            return asset

    class FakeCodeOperations:
        def create_or_update(self, asset):
            created_codes.append(asset)
            asset._id = f"azureml:{asset.name}:{asset.version}"
            return asset

    class FakeJobOperations:
        def __init__(self):
            self.created = None

        def create_or_update(self, job):
            self.created = job
            return SimpleNamespace(name=job.name, status="Completed")

        def get(self, name):
            return SimpleNamespace(name=name, status="Completed")

    jobs = FakeJobOperations()
    client = SimpleNamespace(
        data=FakeDataOperations(),
        models=FakeModelOperations(),
        _code=FakeCodeOperations(),
        jobs=jobs,
    )
    uploaded_paths = {
        "code": "https://account.blob.core.windows.net/container/fixture/code",
        "data": "azureml://datastores/workspaceblobstore/paths/fixture/data",
        "file": "azureml://datastores/workspaceblobstore/paths/fixture/file/config.json",
        "table": "azureml://datastores/workspaceblobstore/paths/fixture/table",
        "model": "azureml://datastores/workspaceblobstore/paths/fixture/model",
    }
    monkeypatch.setattr(
        fixture_module,
        "_upload_fixture_paths_with_identity",
        lambda *args, **kwargs: uploaded_paths,
    )
    monkeypatch.setattr(
        fixture_module,
        "_ensure_identity_datastore",
        lambda *args, **kwargs: SimpleNamespace(name="foundrymigrationidentityblob"),
    )
    monkeypatch.setattr(
        fixture_module,
        "_create_protected_aml_code_asset",
        lambda local_path, **kwargs: FakeCodeOperations().create_or_update(
            SimpleNamespace(
                id="azureml:foundry-migration-code:1",
                name="foundry-migration-code",
                path=str(local_path),
                version="1",
            )
        ),
    )
    request = FixtureRequest(
        source=AmlWorkspace("sub", "rg", "ws", "cpu-cluster"),
        work_dir=tmp_path,
        job_name="fixture-job",
        asset_version="20260728123456789",
        poll_interval_seconds=0.01,
        timeout_seconds=60,
    )

    result = create_aml_migration_fixture(
        request,
        credential=object(),
        ml_client=client,
        emit=lambda message: None,
        sleeper=lambda seconds: None,
    )

    assert result.job_name == "fixture-job"
    assert result.status == "Completed"
    assert result.expected_summary["fixture"] == "aml-foundry-command-job-migration"
    assert result.expected_summary["learningRate"] == 0.125
    assert len(result.expected_summary["trainingDataDigest"]) == 64
    assert result.expected_summary["templatedEnvironmentDigest"] == (
        result.expected_summary["trainingDataDigest"]
    )
    assert len(result.expected_summary["trainingTableDigest"]) == 64
    assert len(result.expected_summary["seedModelDigest"]) == 64
    assert len(created_data) == 3
    assert len(created_models) == 1
    assert len(created_codes) == 1
    assert jobs.created is not None
    assert set(jobs.created.inputs) == {
        "epochs",
        "learning_rate",
        "message",
        "enabled",
        "training_data",
        "config_file",
        "training_table",
        "seed_model",
    }
    assert set(jobs.created.outputs) == {
        "results",
        "summary",
        "metrics_table",
        "trained_model",
    }
    assert type(jobs.created.identity).__name__ == "UserIdentityConfiguration"
    assert str(jobs.created.inputs["training_data"].mode) == "download"
    assert str(jobs.created.outputs["results"].mode) == "upload"
    assert str(jobs.created.outputs["results"].path).endswith("/outputs/results")
    assert str(jobs.created.outputs["summary"].path).endswith(
        "/outputs/summary/summary.json"
    )
    assert str(jobs.created.outputs["metrics_table"].path).endswith(
        "/outputs/metrics_table"
    )
    assert str(jobs.created.outputs["trained_model"].path).endswith(
        "/outputs/trained_model"
    )
    assert jobs.created.environment_variables == {
        "STATIC_SETTING": "preserved",
        "DATA_FROM_ENV": "${{inputs.training_data}}",
    }
    assert jobs.created.command.startswith(
        'export DATA_FROM_ENV="${{inputs.training_data}}" && python exercise_job.py'
    )
    assert not jobs.created.queue_settings
    assert jobs.created.code == created_codes[0].id
    assert created_data[0].path == uploaded_paths["data"]
    assert created_data[1].path == uploaded_paths["file"]
    assert created_data[2].path == uploaded_paths["table"]
    assert created_models[0].path == uploaded_paths["model"]
    assert created_models[0].version == "1"
    assert created_codes[0].path == str(tmp_path / "code")
    assert created_codes[0].version == "1"
    assert (tmp_path / "code" / "exercise_job.py").exists()
    exercise_script = (tmp_path / "code" / "exercise_job.py").read_text(
        encoding="utf-8"
    )
    assert "MIGRATION_FIXTURE_RECORD:" in exercise_script
    assert "env_data_digest != argument_data_digest" in exercise_script
    assert "resolve() !=" not in exercise_script
    assert (tmp_path / "training-table" / "MLTable").exists()


def test_identity_uploader_preserves_relative_paths(tmp_path, monkeypatch):
    paths = fixture_module._write_fixture_files(tmp_path)
    uploaded: dict[str, bytes] = {}

    class FakeContainerClient:
        def upload_blob(self, *, name, data, overwrite):
            assert overwrite is True
            uploaded[name] = data.read()

        def close(self):
            pass

    class FakeServiceClient:
        def __init__(self, account_url, credential):
            assert account_url == "https://account.blob.core.windows.net"

        def get_container_client(self, name):
            assert name == "container"
            return FakeContainerClient()

        def close(self):
            pass

    monkeypatch.setattr(
        "azure.storage.blob.BlobServiceClient",
        FakeServiceClient,
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

    uris = fixture_module._upload_fixture_paths_with_identity(
        paths,
        version="20260728123456789",
        ml_client=client,
        credential=object(),
    )

    prefix = (
        "aml-foundry-command-migration/20260728123456789"  # pragma: allowlist secret
    )
    assert f"{prefix}/data/nested/part-001.jsonl" in uploaded
    assert f"{prefix}/table/MLTable" in uploaded
    assert f"{prefix}/model/weights.json" in uploaded
    assert uris["file"].endswith(f"{prefix}/file/config.json")
    assert uris["code"] == (
        f"https://account.blob.core.windows.net/container/{prefix}/code"
    )


def test_ensure_identity_datastore_creates_credentialless_alias():
    source = SimpleNamespace(
        account_name="account",
        container_name="container",
        endpoint="core.windows.net",
        protocol="https",
    )
    created = []

    class FakeDatastores:
        def get(self, name):
            if name == "workspaceblobstore":
                return source
            from azure.core.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError("missing")

        def create_or_update(self, datastore):
            created.append(datastore)
            return datastore

    result = fixture_module._ensure_identity_datastore(
        SimpleNamespace(datastores=FakeDatastores()),
        name="identityblob",
    )

    assert result.name == "identityblob"
    assert result.account_name == "account"
    assert result.container_name == "container"
    assert type(result.credentials).__name__ == "NoneCredentialConfiguration"
    assert len(created) == 1


def test_ensure_identity_datastore_reuses_matching_keyless_alias():
    source = SimpleNamespace(
        account_name="account",
        container_name="container",
    )
    existing = SimpleNamespace(
        name="identityblob",
        account_name="account",
        container_name="container",
        credentials=None,
    )
    datastores = SimpleNamespace(
        get=lambda name: source if name == "workspaceblobstore" else existing
    )

    result = fixture_module._ensure_identity_datastore(
        SimpleNamespace(datastores=datastores),
        name="identityblob",
    )

    assert result is existing


def test_ensure_identity_datastore_rejects_key_backed_alias():
    source = SimpleNamespace(
        account_name="account",
        container_name="container",
    )

    class AccountKeyConfiguration:
        pass

    existing = SimpleNamespace(
        name="identityblob",
        account_name="account",
        container_name="container",
        credentials=AccountKeyConfiguration(),
    )
    datastores = SimpleNamespace(
        get=lambda name: source if name == "workspaceblobstore" else existing
    )

    with __import__("pytest").raises(ValueError, match="not a credentialless alias"):
        fixture_module._ensure_identity_datastore(
            SimpleNamespace(datastores=datastores),
            name="identityblob",
        )


def test_protected_code_upload_sets_confirmation_metadata_and_hash(
    tmp_path,
    monkeypatch,
):
    code_dir = tmp_path / "code"
    nested = code_dir / "nested"
    nested.mkdir(parents=True)
    (code_dir / "exercise.py").write_text("print('ok')", encoding="utf-8")
    (nested / "helper.py").write_text("VALUE = 1", encoding="utf-8")
    uploaded: dict[str, bytes] = {}
    metadata: dict[str, dict[str, str]] = {}

    class FakeBlobClient:
        def __init__(self, name):
            self.name = name

        def set_blob_metadata(self, value):
            metadata[self.name] = dict(value)

    class FakeContainerClient:
        @classmethod
        def from_container_url(cls, *, container_url, credential):
            assert container_url == "https://account.blob.core.windows.net/pending"
            return cls()

        def upload_blob(self, *, name, data, overwrite):
            assert overwrite is True
            uploaded[str(name)] = data.read()

        def get_blob_client(self, name):
            return FakeBlobClient(name)

        def close(self):
            pass

    monkeypatch.setattr(
        "azure.storage.blob.ContainerClient",
        FakeContainerClient,
    )

    class FakeCodeOperations:
        _resource_group_name = "rg"
        _workspace_name = "ws"

        def __init__(self):
            self.created = None
            response = SimpleNamespace(
                blob_reference_for_consumption=SimpleNamespace(
                    blob_uri="https://account.blob.core.windows.net/pending"
                )
            )
            self._service_client = SimpleNamespace(
                code_versions=SimpleNamespace(
                    create_or_get_start_pending_upload=lambda **kwargs: response
                )
            )

        def create_or_update(self, code):
            self.created = code
            code._id = f"azureml:{code.name}:{code.version}"
            return code

    operations = FakeCodeOperations()
    client = SimpleNamespace(_code=operations)

    asset = fixture_module._create_protected_aml_code_asset(
        code_dir,
        name="protected-code",
        version="1",
        description="fixture",
        ml_client=client,
        credential=object(),
    )

    assert set(uploaded) == {"code/exercise.py", "code/nested/helper.py"}
    assert metadata == {
        "code/exercise.py": {
            "upload_status": "completed",
            "name": "protected-code",
            "version": "1",
        }
    }
    assert asset.path == "https://account.blob.core.windows.net/pending/code"
    assert len(asset._hash_sha256) == 64
    rest_properties = asset._to_rest_object().properties.properties
    assert rest_properties["hash_sha256"] == asset._hash_sha256
    assert rest_properties["hash_version"] == "202208"


def test_capture_aml_job_stream_persists_stdout_and_collector_error(tmp_path):
    class FakeJobs:
        def stream(self, name):
            print(f"job={name} user-log")
            raise RuntimeError("stream reached failed terminal state")

    target = fixture_module._capture_aml_job_stream(
        SimpleNamespace(jobs=FakeJobs()),
        "failed-job",
        tmp_path / "failed.log",
    )

    content = target.read_text(encoding="utf-8")
    assert "job=failed-job user-log" in content
    assert "stream collector raised RuntimeError" in content
    assert "stream reached failed terminal state" not in content


def test_validate_migrated_fixture_checks_dataset_content_and_model(
    tmp_path,
    monkeypatch,
):
    summary = {
        "fixture": "aml-foundry-command-job-migration",
        "epochs": 3,
        "learningRate": 0.125,
        "message": "resurrected-command-job",
        "enabled": True,
        "config": {"batch_size": 4},
    }

    def fake_download(target_dir, *, dataset_name, dataset_version, **kwargs):
        output_name = (
            "results"
            if dataset_name.endswith("results")
            else "metrics_table"
            if dataset_name.endswith("metrics-table")
            else "summary"
        )
        target = tmp_path / "validation" / output_name
        target.mkdir(parents=True, exist_ok=True)
        file_name = (
            "result.json"
            if output_name == "results"
            else "metrics.jsonl"
            if output_name == "metrics_table"
            else "summary.json"
        )
        (target / file_name).write_text(
            __import__("json").dumps(summary),
            encoding="utf-8",
        )
        downloaded_files = [file_name]
        if output_name == "metrics_table":
            (target / "MLTable").write_text(
                "paths:\n  - file: ./metrics.jsonl\n",
                encoding="utf-8",
            )
            downloaded_files.append("MLTable")
        return DatasetDownloadResult(
            dataset_id=f"azureai://data/{dataset_name}/{dataset_version}",
            name=dataset_name,
            version=dataset_version,
            dataset_type={
                "results": "uri_folder",
                "summary": "uri_file",
                "metrics_table": "uri_folder",
            }[output_name],
            data_uri="https://storage.test/output",
            target_dir=str(target),
            downloaded_files=tuple(downloaded_files),
            total_bytes=1,
        )

    monkeypatch.setattr(fixture_module, "download_dataset", fake_download)

    def fake_download_model(target_dir, **kwargs):
        target = tmp_path / "validation" / "trained_model"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.json").write_text(
            __import__("json").dumps(
                {"source": summary.get("seedModelDigest"), "epochs": 3}
            ),
            encoding="utf-8",
        )
        return ModelAssetDownloadResult(
            asset_id="azureai://models/trained/1",
            name=kwargs["name"],
            version=kwargs["version"],
            blob_uri="https://storage.test/model",
            target_dir=str(target),
            downloaded_files=("model.json",),
            total_bytes=10,
        )

    monkeypatch.setattr(fixture_module, "download_model_asset", fake_download_model)
    outputs = {
        "results": {
            "jobOutputType": "uri_folder",
            "assetName": "fixture-results",
            "assetVersion": "1",
        },
        "summary": {
            "jobOutputType": "uri_file",
            "assetName": "fixture-summary",
            "assetVersion": "1",
        },
        "metrics_table": {
            "jobOutputType": "uri_folder",
            "assetName": "fixture-metrics-table",
            "assetVersion": "1",
        },
        "trained_model": {
            "jobOutputType": "custom_model",
            "assetName": "fixture-model",
            "assetVersion": "1",
        },
    }
    migration = MigrationResult(
        manifest_path=str(tmp_path / "manifest.json"),
        source_job_name="source",
        target_job_name="target",
        target_status="Completed",
        request_body={"properties": {"outputs": outputs}},
        asset_mappings={},
        warnings=(),
    )
    target = FoundryTarget(
        project_endpoint="https://example.test",
        project_name="project",
        storage_connection_name="storage",
        compute_id="/compute",
        instance_type="Singularity.D4_v3",
        api_version="2026-01-15-preview",
    )

    validation = validate_migrated_fixture(
        migration,
        target=target,
        work_dir=tmp_path / "validation",
        credential=object(),
        timeout_seconds=10,
        poll_interval_seconds=0.01,
        emit=lambda message: None,
        sleeper=lambda seconds: None,
        expected_summary=summary,
    )

    assert validation.summary == summary
    assert validation.metrics_table_dataset_id == (
        "azureai://data/fixture-metrics-table/1"
    )
    assert validation.model_asset_id == "azureai://models/trained/1"
    assert validation.model_downloaded_files == ("model.json",)
