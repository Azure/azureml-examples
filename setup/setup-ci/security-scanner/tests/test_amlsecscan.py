import json
import os
import subprocess
import sys
import pytest
import requests
from unittest.mock import call, patch

# Mock environment variables
os.environ["MSI_ENDPOINT"] = "http://127.0.0.1:46808/MSI/auth"
os.environ["MSI_SECRET"] = "1234"
os.environ["CI_NAME"] = "mock-host"
os.environ[
    "MLFLOW_TRACKING_URI"
] = "azureml://westus2.api.azureml.ms/mlflow/v1.0/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock"

# Load test code to test
test_folder = os.path.dirname(__file__)
sys.path.append(os.path.dirname(test_folder))
import amlsecscan


class RestResponse:
    def __init__(self, status_code, reason=None, json=None):
        self.status_code = status_code
        self.reason = reason
        self._json = json

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(self.status_code + " " + self.reason)


def mock_requests_get(url, headers=None):
    if (
        url
        == "https://management.azure.com/subscriptions/mock-s/resourceGroups/mock-rg/providers/Microsoft.OperationalInsights/workspaces/mock-w?api-version=2021-06-01"
    ):
        return RestResponse(200, json={"properties": {"customerId": "mock-cid"}})
    if (
        url
        == "http://127.0.0.1:46808/MSI/auth?resource=https://management.azure.com&api-version=2017-09-01"
    ):
        return RestResponse(200, json={"access_token": "1234"})
    if (
        url
        == "https://management.azure.com/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock0/providers/microsoft.insights/diagnosticSettings?api-version=2021-05-01-preview"
    ):
        return RestResponse(200, json={"value": []})
    if (
        url
        == "https://management.azure.com/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock1/providers/microsoft.insights/diagnosticSettings?api-version=2021-05-01-preview"
    ):
        return RestResponse(200, json={"value": [{"properties": {}}]})
    if (
        url
        == "https://management.azure.com/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock2/providers/microsoft.insights/diagnosticSettings?api-version=2021-05-01-preview"
    ):
        return RestResponse(
            200,
            json={
                "value": [
                    {
                        "properties": {
                            "workspaceId": "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourcegroups/defaultresourcegroup-wus2/providers/microsoft.operationalinsights/workspaces/mock"
                        }
                    }
                ]
            },
        )
    raise ValueError(f"No mock for GET {url}")


def mock_requests_post(url, headers=None, data=None):
    if (
        url
        == "https://management.azure.com/subscriptions/mock-s/resourceGroups/mock-rg/providers/Microsoft.OperationalInsights/workspaces/mock-w/sharedKeys?api-version=2020-08-01"
    ):
        return RestResponse(200, json={"primarySharedKey": "1234"})
    elif (
        url
        == "https://mock-cid.ods.opinsights.azure.com/api/logs?api-version=2016-04-01"
    ):
        return RestResponse(200)

    raise ValueError(f"No mock for POST {url}")


def test_send_assessment():

    with patch.object(requests, "get", side_effect=mock_requests_get) as mock_get:
        with patch.object(
            requests, "post", side_effect=mock_requests_post
        ) as mock_post:
            telemetry = amlsecscan.LogAnalyticsTelemetry(
                "/subscriptions/mock-s/resourceGroups/mock-rg/providers/Microsoft.OperationalInsights/workspaces/mock-w"
            )
            amlsecscan._send_assessment(
                telemetry, "Malware", 3, {"elapsedTimeInS": 3.4}
            )

    assert mock_get.call_count == 2
    assert mock_post.call_count == 2


def test_send_health():

    with patch.object(requests, "get", side_effect=mock_requests_get) as mock_get:
        with patch.object(
            requests, "post", side_effect=mock_requests_post
        ) as mock_post:
            telemetry = amlsecscan.LogAnalyticsTelemetry(
                "/subscriptions/mock-s/resourceGroups/mock-rg/providers/Microsoft.OperationalInsights/workspaces/mock-w"
            )
            amlsecscan._send_health(telemetry, "Heartbeat")

    assert mock_get.call_count == 2
    assert mock_post.call_count == 2


def test_parse_clamav_stdout_without_malware():
    stdout = """
----------- SCAN SUMMARY -----------
Known viruses: 8572941
Engine version: 0.103.5
Scanned directories: 1
Scanned files: 150
Infected files: 0
Data scanned: 15.04 MB
Data read: 14.47 MB (ratio 1.04:1)
Time: 19.138 sec (0 m 19 s)
Start Date: 2022:04:26 21:55:34
End Date:   2022:04:26 21:55:53
"""

    findings, details = amlsecscan._parse_clamav_stdout(stdout)

    assert findings == 0
    assert details == {
        "engineVersion": "0.103.5",
        "knownViruses": 8572941,
        "scannedDirectories": 1,
        "scannedFiles": 150,
    }


def test_parse_clamav_stdout_with_malware():
    stdout = """
/root/eicar.com.txt: Win.Test.EICAR_HDB-1 FOUND

----------- SCAN SUMMARY -----------
Known viruses: 8572941
Engine version: 0.103.5
Scanned directories: 23
Scanned files: 25
Infected files: 1
Data scanned: 3.66 MB
Data read: 13.51 MB (ratio 0.27:1)
Time: 18.308 sec (0 m 18 s)
Start Date: 2022:04:26 21:49:47
End Date:   2022:04:26 21:50:05
"""

    findings, details = amlsecscan._parse_clamav_stdout(stdout)

    assert findings == 1
    assert details == {
        "engineVersion": "0.103.5",
        "files": [
            {"malwareType": "Win.Test.EICAR_HDB-1", "path": "/root/eicar.com.txt"}
        ],
        "knownViruses": 8572941,
        "scannedDirectories": 23,
        "scannedFiles": 25,
    }


def test_sanitize_log_analytics_resource_id():
    assert (
        amlsecscan._sanitize_log_analytics_resource_id(
            " /subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/WUS2/providers/Microsoft.OperationalInsights/workspaces/w7ed9d00ebb32 "
        )
        == "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/WUS2/providers/Microsoft.OperationalInsights/workspaces/w7ed9d00ebb32"
    )

    with pytest.raises(ValueError):
        amlsecscan._sanitize_log_analytics_resource_id(
            "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/WUS2/providers/Microsoft.OperationalInsights/workspaces"
        )


def test_default_install_path_is_root_controlled():
    assert amlsecscan._config_folder_path == "/opt/amlsecscan"
    assert amlsecscan._installed_scanner_path == "/opt/amlsecscan/amlsecscan.py"
    assert not amlsecscan._config_folder_path.startswith("/home/azureuser/")


def test_install_writes_root_owned_entrypoint(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    state_dir = tmp_path / "state"
    cron_path = tmp_path / "cron.d" / "amlsecscan"
    cron_path.parent.mkdir()

    global_config_path = config_dir / "config.json"
    installed_scanner_path = config_dir / "amlsecscan.py"
    run_script_path = config_dir / "run.sh"
    config_dir.mkdir()
    state_dir.mkdir()
    state_marker = state_dir / "existing-marker"
    global_config_path.write_text("existing config")
    installed_scanner_path.write_text("existing scanner")
    run_script_path.write_text("existing entrypoint")
    state_marker.write_text("keep")

    monkeypatch.setattr(amlsecscan, "_config_folder_path", config_dir.as_posix())
    monkeypatch.setattr(
        amlsecscan, "_global_config_path", global_config_path.as_posix()
    )
    monkeypatch.setattr(
        amlsecscan, "_installed_scanner_path", installed_scanner_path.as_posix()
    )
    monkeypatch.setattr(amlsecscan, "_state_folder_path", state_dir.as_posix())
    monkeypatch.setattr(amlsecscan, "_cron_path", cron_path.as_posix())
    monkeypatch.setattr(
        amlsecscan, "_local_config_path", (tmp_path / "missing.json").as_posix()
    )
    monkeypatch.setattr(amlsecscan.os, "geteuid", lambda: 0, raising=False)
    commands = []
    monkeypatch.setattr(
        amlsecscan, "_run", lambda command, check=True: commands.append(command)
    )
    monkeypatch.setattr(
        amlsecscan,
        "_is_trusted_root_directory",
        lambda path, allowed_entries=None: True,
    )

    log_analytics_resource_id = "/subscriptions/mock-s/resourceGroups/mock-rg/providers/Microsoft.OperationalInsights/workspaces/mock-w"

    with patch.object(amlsecscan.shutil, "chown") as mock_chown:
        amlsecscan._install(log_analytics_resource_id)

    assert json.loads(global_config_path.read_text()) == {
        "logAnalyticsResourceId": log_analytics_resource_id
    }
    assert installed_scanner_path.exists()
    assert state_dir.is_dir()
    assert state_marker.read_text() == "keep"
    assert commands[0] == "apt-get update"

    run_script = run_script_path.read_text()
    assert f'python3 {installed_scanner_path.as_posix()} "$@"' in run_script
    assert os.path.abspath(amlsecscan.__file__) not in run_script
    assert "cgroup.controllers" in run_script
    assert "cpu.max" in run_script
    assert "cpu.cfs_quota_us" in run_script
    assert "configure_cgroup || true" in run_script

    cron = cron_path.read_text()
    assert f"root {run_script_path.as_posix()} heartbeat" in cron

    mock_chown.assert_has_calls(
        [
            call(config_dir.as_posix(), "root", "root"),
            call(state_dir.as_posix(), "root", "root"),
        ],
        any_order=True,
    )
    assert len(mock_chown.call_args_list) == 6
    for chown_call in mock_chown.call_args_list:
        assert chown_call.args[1:] == ("root", "root")
    assert not list(config_dir.glob(".amlsecscan-*"))
    assert not list(cron_path.parent.glob(".amlsecscan-*"))


def test_untrusted_install_directory_is_recreated(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    unexpected_module = config_dir / "requests.py"
    unexpected_module.write_text("malicious")

    with patch.object(
        amlsecscan, "_is_trusted_root_directory", return_value=False
    ), patch.object(amlsecscan.shutil, "chown"):
        amlsecscan._ensure_root_owned_directory(
            config_dir.as_posix(),
            0o0755,
            {"config.json", "amlsecscan.py", "run.sh"},
        )

    assert config_dir.is_dir()
    assert not unexpected_module.exists()


def test_atomic_file_failure_restores_previous_files(tmp_path):
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.write_text("old first")
    second_path.write_text("old second")
    real_replace = os.replace

    def fail_second_replace(source, destination):
        if destination == second_path.as_posix():
            raise OSError("mock replacement failure")
        real_replace(source, destination)

    with patch.object(amlsecscan.shutil, "chown"), patch.object(
        amlsecscan.os, "replace", side_effect=fail_second_replace
    ):
        with pytest.raises(OSError, match="mock replacement failure"):
            amlsecscan._write_files_atomically(
                [
                    (first_path.as_posix(), "new first", 0o0644),
                    (second_path.as_posix(), "new second", 0o0644),
                ]
            )

    assert first_path.read_text() == "old first"
    assert second_path.read_text() == "old second"
    assert not list(tmp_path.glob(".amlsecscan-*"))


def test_install_failure_preserves_existing_entrypoint(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    state_dir = tmp_path / "state"
    cron_path = tmp_path / "cron.d" / "amlsecscan"
    config_dir.mkdir()
    state_dir.mkdir()
    cron_path.parent.mkdir()

    global_config_path = config_dir / "config.json"
    installed_scanner_path = config_dir / "amlsecscan.py"
    run_script_path = config_dir / "run.sh"
    global_config_path.write_text("existing config")
    installed_scanner_path.write_text("existing scanner")
    run_script_path.write_text("existing entrypoint")
    state_marker = state_dir / "existing-state"
    state_marker.write_text("existing state")
    cron_path.write_text("existing cron")

    monkeypatch.setattr(amlsecscan, "_config_folder_path", config_dir.as_posix())
    monkeypatch.setattr(
        amlsecscan, "_global_config_path", global_config_path.as_posix()
    )
    monkeypatch.setattr(
        amlsecscan, "_installed_scanner_path", installed_scanner_path.as_posix()
    )
    monkeypatch.setattr(amlsecscan, "_state_folder_path", state_dir.as_posix())
    monkeypatch.setattr(amlsecscan, "_cron_path", cron_path.as_posix())
    monkeypatch.setattr(
        amlsecscan, "_local_config_path", (tmp_path / "missing.json").as_posix()
    )
    monkeypatch.setattr(amlsecscan.os, "geteuid", lambda: 0, raising=False)

    def fail_package_install(command, check=True):
        raise subprocess.CalledProcessError(100, command)

    monkeypatch.setattr(amlsecscan, "_run", fail_package_install)

    log_analytics_resource_id = "/subscriptions/mock-s/resourceGroups/mock-rg/providers/Microsoft.OperationalInsights/workspaces/mock-w"

    with patch.object(amlsecscan.shutil, "chown"):
        with pytest.raises(subprocess.CalledProcessError):
            amlsecscan._install(log_analytics_resource_id)

    assert global_config_path.read_text() == "existing config"
    assert installed_scanner_path.read_text() == "existing scanner"
    assert run_script_path.read_text() == "existing entrypoint"
    assert state_marker.read_text() == "existing state"
    assert cron_path.read_text() == "existing cron"


def test_parse_trivy_results_1():
    findings_os, findings_python = amlsecscan._parse_trivy_results(
        test_folder + "/test_trivy_1.json"
    )
    assert findings_os == [
        {
            "CVE": "CVE-2022-21499",
            "packageName": "linux-headers-5.4.0-113",
            "packageVersion": "5.4.0-113.127",
            "severity": "HIGH",
            "title": "kernel: possible to use the debugger to write zero into a location of choice",
        },
        {
            "CVE": "CVE-2022-21499",
            "packageName": "linux-headers-5.4.0-113-generic",
            "packageVersion": "5.4.0-113.127",
            "severity": "HIGH",
            "title": "kernel: possible to use the debugger to write zero into a location of choice",
        },
        {
            "CVE": "CVE-2022-21499",
            "packageName": "linux-libc-dev",
            "packageVersion": "5.4.0-113.127",
            "severity": "HIGH",
            "title": "kernel: possible to use the debugger to write zero into a location of choice",
        },
        {
            "CVE": "CVE-2022-21499",
            "packageName": "linux-tools-common",
            "packageVersion": "5.4.0-113.127",
            "severity": "HIGH",
            "title": "kernel: possible to use the debugger to write zero into a location of choice",
        },
    ]
    assert findings_python == [
        {
            "CVE": "CVE-2018-1000656",
            "file": "databricks/conda/pkgs/conda-4.8.2-py38_0/info/test/tests/conda_env/support/requirements.txt",
            "packageName": "flask",
            "packageVersion": "0.10.1",
            "severity": "HIGH",
            "title": "python-flask: Denial of Service via crafted JSON file",
        },
        {
            "CVE": "CVE-2019-1010083",
            "file": "databricks/conda/pkgs/conda-4.8.2-py38_0/info/test/tests/conda_env/support/requirements.txt",
            "packageName": "flask",
            "packageVersion": "0.10.1",
            "severity": "HIGH",
            "title": "python-flask: unexpected memory usage can lead to denial of service via crafted encoded JSON data",
        },
    ]


def test_parse_trivy_results_2():
    findings_os, findings_python = amlsecscan._parse_trivy_results(
        test_folder + "/test_trivy_2.json"
    )
    assert len(findings_os) == 0
    assert len(findings_python) == 12


def test_parse_trivy_results_3():
    findings_os, findings_python = amlsecscan._parse_trivy_results(
        test_folder + "/test_trivy_3.json"
    )
    assert len(findings_os) == 1
    assert len(findings_python) == 0


def test_filter_trivy_results():
    findings = [{"severity": "HIGH"} for n in range(100)]
    findings.append({"severity": "CRITICAL"})

    assert len(findings) == 101
    assert findings[0]["severity"] == "HIGH"
    assert findings[49]["severity"] == "HIGH"
    assert findings[99]["severity"] == "HIGH"
    assert findings[100]["severity"] == "CRITICAL"

    findings = amlsecscan._filter_trivy_results(findings)

    assert len(findings) == 50
    assert findings[0]["severity"] == "CRITICAL"
    assert findings[49]["severity"] == "HIGH"


def test_get_log_analytics_from_diagnostic_settings():

    with patch.object(requests, "get", side_effect=mock_requests_get) as mock_get:
        with patch.object(
            amlsecscan,
            "_azure_ml_resource_id",
            "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock0",
        ):
            assert amlsecscan._get_log_analytics_from_diagnostic_settings() is None
            assert mock_get.call_count == 2
        with patch.object(
            amlsecscan,
            "_azure_ml_resource_id",
            "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock1",
        ):
            assert amlsecscan._get_log_analytics_from_diagnostic_settings() is None
            assert mock_get.call_count == 4
        with patch.object(
            amlsecscan,
            "_azure_ml_resource_id",
            "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock2",
        ):
            assert (
                amlsecscan._get_log_analytics_from_diagnostic_settings()
                == "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourcegroups/defaultresourcegroup-wus2/providers/microsoft.operationalinsights/workspaces/mock"
            )
            assert mock_get.call_count == 6


@pytest.mark.skip(reason="requires sudo")
def test_install():
    amlsecscan._install(
        "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/WUS2/providers/Microsoft.OperationalInsights/workspaces/w7ed9d00ebb32"
    )
    amlsecscan._uninstall()


@pytest.mark.skip(reason="only runs on AML CI")
def test_get_auth_environment_variables():
    env_var = amlsecscan._get_auth_environment_variables()
    assert len(env_var["MSI_ENDPOINT"]) > 0
    assert len(env_var["MSI_SECRET"]) > 0
