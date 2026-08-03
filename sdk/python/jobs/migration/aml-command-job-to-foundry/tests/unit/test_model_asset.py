from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from foundrytrainingjob import model_asset
from foundrytrainingjob.rest import FoundryRestResponse


def _response(
    status_code: int,
    body: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> FoundryRestResponse:
    return FoundryRestResponse(
        status_code=status_code,
        headers=headers or {},
        text=json.dumps(body) if body is not None else "",
    )


class _TrackingContainerClient:
    def __init__(self) -> None:
        self.active_uploads = 0
        self.max_active_uploads = 0
        self.closed = False
        self.uploaded_names: list[str] = []
        self._parallel_upload_started = threading.Event()
        self._lock = threading.Lock()

    def upload_blob(self, *, name: str, data: Any, overwrite: bool) -> None:
        assert overwrite is True
        with self._lock:
            self.active_uploads += 1
            self.max_active_uploads = max(self.max_active_uploads, self.active_uploads)
            if self.active_uploads >= 2:
                self._parallel_upload_started.set()
        self._parallel_upload_started.wait(timeout=1)
        data.read()
        with self._lock:
            self.uploaded_names.append(name)
            self.active_uploads -= 1

    def close(self) -> None:
        self.closed = True


class _DownloadStream:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def readinto(self, file_handle: Any) -> int:
        return file_handle.write(self.content)


class _DownloadBlobClient:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def download_blob(self) -> _DownloadStream:
        return _DownloadStream(self.content)


class _DownloadContainerClient:
    def __init__(self) -> None:
        self.closed = False
        self.blobs = {
            "models/run/model.json": b'{"epochs": 3}',
            "models/run/subdir/weights.bin": b"weights",
        }

    def list_blob_names(self, *, name_starts_with: str | None = None):
        return [
            name
            for name in self.blobs
            if not name_starts_with or name.startswith(name_starts_with)
        ]

    def get_blob_client(self, name: str) -> _DownloadBlobClient:
        return _DownloadBlobClient(self.blobs[name])

    def close(self) -> None:
        self.closed = True


class ModelAssetUploadTests(unittest.TestCase):
    def test_upload_and_register_model_uses_async_create_and_final_get(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)
            expected_total_bytes = 0
            for index in range(10):
                content = b"x" * (index + 1)
                expected_total_bytes += len(content)
                (local_path / f"shard-{index}.bin").write_bytes(content)

            container_client = _TrackingContainerClient()
            calls: list[tuple[str, str, dict[str, Any]]] = []
            operation_polls = iter(
                [
                    _response(200, {"status": "Running"}),
                    _response(200, {"status": "Succeeded"}),
                ]
            )
            model_get_count = 0

            def request(method: str, url: str, **kwargs: Any) -> FoundryRestResponse:
                nonlocal model_get_count
                calls.append((method, url, kwargs))
                if url.endswith("/startPendingUpload?api-version=2026-01-15-preview"):
                    return _response(
                        200,
                        {
                            "blobReferenceForConsumption": {
                                "blobUri": "https://storage.example/models/upload",
                                "credential": {
                                    "sasUri": "https://storage.example/models?sig=secret"
                                },
                            }
                        },
                    )
                if url.endswith("/createAsync?api-version=2026-01-15-preview"):
                    return _response(
                        202,
                        headers={"Operation-Location": "/operations/model-create-1"},
                    )
                if (
                    url
                    == "https://build-26-demo.services.ai.azure.com/operations/model-create-1"
                ):
                    return next(operation_polls)
                if url.endswith("/versions/1?api-version=2026-01-15-preview"):
                    model_get_count += 1
                    if model_get_count == 1:
                        return _response(404, {"error": {"code": "NotFound"}})
                    return _response(
                        200,
                        {
                            "id": "azureai://accounts/build-26-demo/projects/Build-26-Demo-Project/models/github-qwen3-8b-rle-demo/versions/1",
                            "properties": {"provisioningState": "Succeeded"},
                        },
                    )
                raise AssertionError(f"Unexpected request: {method} {url}")

            with (
                patch.object(
                    model_asset,
                    "_container_client_from_sas",
                    return_value=container_client,
                ),
                patch.object(
                    model_asset,
                    "get_foundry_access_token",
                    return_value=SimpleNamespace(token="token"),
                ),
                patch.object(model_asset.time, "sleep"),
                patch.object(model_asset, "_request_foundry", side_effect=request),
            ):
                result = model_asset.upload_and_register_model(
                    local_path,
                    name="github-qwen3-8b-rle-demo",
                    version="1",
                    project_endpoint="https://build-26-demo.services.ai.azure.com",
                    project_name="Build-26-Demo-Project",
                    api_version="2026-01-15-preview",
                    description="Local Qwen snapshot",
                    tags={"source": "Qwen/Qwen3-8B", "revision": "b968826d"},
                    blob_prefix="model",
                )

            self.assertTrue(
                result.asset_id.endswith("/models/github-qwen3-8b-rle-demo/versions/1")
            )
            self.assertEqual(result.provisioning_status, "Succeeded")
            self.assertEqual(result.total_bytes, expected_total_bytes)
            self.assertEqual(len(result.uploaded_files), 10)
            self.assertTrue(
                all(name.startswith("model/") for name in result.uploaded_files)
            )
            self.assertTrue(container_client.closed)
            self.assertGreater(container_client.max_active_uploads, 1)
            self.assertLessEqual(
                container_client.max_active_uploads,
                model_asset._MAX_PARALLEL_UPLOADS,
            )

            create_calls = [
                call
                for call in calls
                if call[1].endswith("/createAsync?api-version=2026-01-15-preview")
            ]
            self.assertEqual(len(create_calls), 1)
            method, _, kwargs = create_calls[0]
            self.assertEqual(method, "POST")
            self.assertEqual(
                kwargs["payload"],
                {
                    "blobUri": "https://storage.example/models/upload",
                    "description": "Local Qwen snapshot",
                    "tags": {"source": "Qwen/Qwen3-8B", "revision": "b968826d"},
                },
            )
            self.assertFalse(any(method == "PUT" for method, _, _ in calls))

    def test_upload_and_register_model_reuses_existing_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)
            (local_path / "model.json").write_text("{}", encoding="utf-8")

            with (
                patch.object(
                    model_asset,
                    "get_foundry_access_token",
                    return_value=SimpleNamespace(token="token"),
                ),
                patch.object(
                    model_asset,
                    "_request_foundry",
                    return_value=_response(
                        200,
                        {
                            "id": "azureai://accounts/a/projects/p/models/existing/versions/1",
                            "name": "existing",
                            "version": "1",
                            "blobUri": "https://storage.example/container/model",
                        },
                    ),
                ) as request,
            ):
                result = model_asset.upload_and_register_model(
                    local_path,
                    name="existing",
                    version="1",
                    project_endpoint="https://example.services.ai.azure.com",
                    project_name="p",
                )

            self.assertEqual(result.provisioning_status, "Succeeded")
            self.assertEqual(result.uploaded_files, ())
            self.assertEqual(request.call_count, 1)

    def test_upload_and_register_model_requires_final_get_200(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)
            (local_path / "config.json").write_text("{}", encoding="utf-8")
            container_client = _TrackingContainerClient()

            def request(method: str, url: str, **_: Any) -> FoundryRestResponse:
                if url.endswith("/startPendingUpload?api-version=2026-01-15-preview"):
                    return _response(
                        200,
                        {
                            "blobReferenceForConsumption": {
                                "blobUri": "https://storage.example/models/upload",
                                "credential": {
                                    "sasUri": "https://storage.example/models?sig=secret"
                                },
                            }
                        },
                    )
                if url.endswith("/createAsync?api-version=2026-01-15-preview"):
                    return _response(
                        202,
                        headers={"Location": "/operations/model-create-2"},
                    )
                if url.endswith("/operations/model-create-2"):
                    return _response(
                        200,
                        {"properties": {"provisioningState": "Succeeded"}},
                    )
                if url.endswith("/versions/1?api-version=2026-01-15-preview"):
                    return _response(404, {"error": {"code": "NotFound"}})
                raise AssertionError(f"Unexpected request: {method} {url}")

            with (
                patch.object(
                    model_asset,
                    "_container_client_from_sas",
                    return_value=container_client,
                ),
                patch.object(
                    model_asset,
                    "get_foundry_access_token",
                    return_value=SimpleNamespace(token="token"),
                ),
                patch.object(model_asset, "_request_foundry", side_effect=request),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "after createAsync failed: 404",
                ):
                    model_asset.upload_and_register_model(
                        local_path,
                        name="github-qwen3-8b-rle-demo",
                        version="1",
                        project_endpoint="https://build-26-demo.services.ai.azure.com",
                        project_name="Build-26-Demo-Project",
                        api_version="2026-01-15-preview",
                    )

    def test_download_model_asset_uses_credentials_and_preserves_relative_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            container_client = _DownloadContainerClient()
            calls: list[tuple[str, str, dict[str, Any]]] = []

            def request(method: str, url: str, **kwargs: Any) -> FoundryRestResponse:
                calls.append((method, url, kwargs))
                if url.endswith("/versions/1?api-version=2026-01-15-preview"):
                    return _response(
                        200,
                        {
                            "id": "azureai://accounts/a/projects/p/models/trained/versions/1",
                            "blobUri": "https://storage.example/container/models/run",
                        },
                    )
                if url.endswith(
                    "/versions/1/credentials?api-version=2026-01-15-preview"
                ):
                    return _response(
                        200,
                        {
                            "blobReferenceForConsumption": {
                                "blobUri": "https://storage.example/container/models/run",
                                "credential": {
                                    "sasUri": "https://storage.example/container"
                                },
                            }
                        },
                    )
                raise AssertionError(f"Unexpected request: {method} {url}")

            with (
                patch.object(
                    model_asset,
                    "_container_client_from_sas",
                    return_value=container_client,
                ),
                patch.object(
                    model_asset,
                    "get_foundry_access_token",
                    return_value=SimpleNamespace(token="token"),
                ),
                patch.object(model_asset, "_request_foundry", side_effect=request),
            ):
                result = model_asset.download_model_asset(
                    temp_dir,
                    name="trained",
                    version="1",
                    project_endpoint="https://example.services.ai.azure.com",
                    project_name="p",
                    api_version="2026-01-15-preview",
                )

            self.assertEqual(
                result.downloaded_files,
                ("model.json", "subdir\\weights.bin")
                if __import__("os").name == "nt"
                else ("model.json", "subdir/weights.bin"),
            )
            self.assertEqual(
                (Path(temp_dir) / "model.json").read_bytes(), b'{"epochs": 3}'
            )
            self.assertTrue((Path(temp_dir) / "subdir" / "weights.bin").exists())
            self.assertTrue(container_client.closed)
            credential_call = next(call for call in calls if "/credentials?" in call[1])
            self.assertEqual(credential_call[0], "POST")
            self.assertEqual(
                credential_call[2]["payload"],
                {"blobUri": "https://storage.example/container/models/run"},
            )


if __name__ == "__main__":
    unittest.main()
