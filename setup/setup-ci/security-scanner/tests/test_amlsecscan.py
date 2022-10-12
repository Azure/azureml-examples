import os
import sys
import pytest
import requests
from unittest.mock import patch

# Mock environment variables
os.environ["MSI_ENDPOINT"] = "http://127.0.0.1:46808/MSI/auth"
os.environ["MSI_SECRET"] = "1234"
os.environ["CI_NAME"] = "mock-host"
os.environ["MLFLOW_TRACKING_URI"] = "azureml://westus2.api.azureml.ms/mlflow/v1.0/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock"

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
    if url == "https://management.azure.com/subscriptions/mock-s/resourceGroups/mock-rg/providers/Microsoft.OperationalInsights/workspaces/mock-w?api-version=2021-06-01":
        return RestResponse(200, json={"properties": {"customerId": "mock-cid"}})
    if url == "http://127.0.0.1:46808/MSI/auth?resource=https://management.azure.com&api-version=2017-09-01":
        return RestResponse(200, json={"access_token": "1234"})
    if url == "https://management.azure.com/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock0/providers/microsoft.insights/diagnosticSettings?api-version=2021-05-01-preview":
        return RestResponse(200, json={"value": []})
    if url == "https://management.azure.com/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock1/providers/microsoft.insights/diagnosticSettings?api-version=2021-05-01-preview":
        return RestResponse(200, json={"value": [{"properties": {}}]})
    if url == "https://management.azure.com/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock2/providers/microsoft.insights/diagnosticSettings?api-version=2021-05-01-preview":
        return RestResponse(200, json={"value": [{"properties": {"workspaceId": "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourcegroups/defaultresourcegroup-wus2/providers/microsoft.operationalinsights/workspaces/mock"}}]})
    raise ValueError(f"No mock for GET {url}")


def mock_requests_post(url, headers=None, data=None):
    if url == "https://management.azure.com/subscriptions/mock-s/resourceGroups/mock-rg/providers/Microsoft.OperationalInsights/workspaces/mock-w/sharedKeys?api-version=2020-08-01":
        return RestResponse(200, json={"primarySharedKey": "1234"})
    elif url == "https://mock-cid.ods.opinsights.azure.com/api/logs?api-version=2016-04-01":
        return RestResponse(200)

    raise ValueError(f"No mock for POST {url}")


def test_send_assessment():

    with patch.object(requests, "get", side_effect=mock_requests_get) as mock_get:
        with patch.object(requests, "post", side_effect=mock_requests_post) as mock_post:
            telemetry = amlsecscan.LogAnalyticsTelemetry("/subscriptions/mock-s/resourceGroups/mock-rg/providers/Microsoft.OperationalInsights/workspaces/mock-w")
            amlsecscan._send_assessment(telemetry, "Malware", 3, {"elapsedTimeInS": 3.4})

    assert mock_get.call_count == 2
    assert mock_post.call_count == 2


def test_send_health():

    with patch.object(requests, "get", side_effect=mock_requests_get) as mock_get:
        with patch.object(requests, "post", side_effect=mock_requests_post) as mock_post:
            telemetry = amlsecscan.LogAnalyticsTelemetry("/subscriptions/mock-s/resourceGroups/mock-rg/providers/Microsoft.OperationalInsights/workspaces/mock-w")
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
        'engineVersion': '0.103.5',
        'knownViruses': 8572941,
        'scannedDirectories': 1,
        'scannedFiles': 150
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
        'engineVersion': '0.103.5',
        'files': [{'malwareType': 'Win.Test.EICAR_HDB-1', 'path': '/root/eicar.com.txt'}],
        'knownViruses': 8572941,
        'scannedDirectories': 23,
        'scannedFiles': 25
    }


def test_sanitize_log_analytics_resource_id():
    assert amlsecscan._sanitize_log_analytics_resource_id(" /subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/WUS2/providers/Microsoft.OperationalInsights/workspaces/w7ed9d00ebb32 ")\
        == "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/WUS2/providers/Microsoft.OperationalInsights/workspaces/w7ed9d00ebb32"

    with pytest.raises(ValueError):
        amlsecscan._sanitize_log_analytics_resource_id("/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/WUS2/providers/Microsoft.OperationalInsights/workspaces")


def test_parse_trivy_results_1():
    findings_os, findings_python = amlsecscan._parse_trivy_results(test_folder + "/test_trivy_1.json")
    assert findings_os == [
        {'CVE': 'CVE-2022-21499', 'packageName': 'linux-headers-5.4.0-113', 'packageVersion': '5.4.0-113.127', 'severity': 'HIGH', 'title': 'kernel: possible to use the debugger to write zero into a location of choice'},
        {'CVE': 'CVE-2022-21499', 'packageName': 'linux-headers-5.4.0-113-generic', 'packageVersion': '5.4.0-113.127', 'severity': 'HIGH', 'title': 'kernel: possible to use the debugger to write zero into a location of choice'},
        {'CVE': 'CVE-2022-21499', 'packageName': 'linux-libc-dev', 'packageVersion': '5.4.0-113.127', 'severity': 'HIGH', 'title': 'kernel: possible to use the debugger to write zero into a location of choice'},
        {'CVE': 'CVE-2022-21499', 'packageName': 'linux-tools-common', 'packageVersion': '5.4.0-113.127', 'severity': 'HIGH', 'title': 'kernel: possible to use the debugger to write zero into a location of choice'}
    ]
    assert findings_python == [
        {'CVE': 'CVE-2018-1000656', 'file': 'databricks/conda/pkgs/conda-4.8.2-py38_0/info/test/tests/conda_env/support/requirements.txt', 'packageName': 'flask', 'packageVersion': '0.10.1', 'severity': 'HIGH', 'title': 'python-flask: Denial of Service via crafted JSON file'},
        {'CVE': 'CVE-2019-1010083', 'file': 'databricks/conda/pkgs/conda-4.8.2-py38_0/info/test/tests/conda_env/support/requirements.txt', 'packageName': 'flask', 'packageVersion': '0.10.1', 'severity': 'HIGH', 'title': 'python-flask: unexpected memory usage can lead to denial of service via crafted encoded JSON data'}
    ]


def test_parse_trivy_results_2():
    findings_os, findings_python = amlsecscan._parse_trivy_results(test_folder + "/test_trivy_2.json")
    assert len(findings_os) == 0
    assert len(findings_python) == 12


def test_parse_trivy_results_3():
    findings_os, findings_python = amlsecscan._parse_trivy_results(test_folder + "/test_trivy_3.json")
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
        with patch.object(amlsecscan, "_azure_ml_resource_id", "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock0"):
            assert amlsecscan._get_log_analytics_from_diagnostic_settings() is None
            assert mock_get.call_count == 2
        with patch.object(amlsecscan, "_azure_ml_resource_id", "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock1"):
            assert amlsecscan._get_log_analytics_from_diagnostic_settings() is None
            assert mock_get.call_count == 4
        with patch.object(amlsecscan, "_azure_ml_resource_id", "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/targetresources/providers/Microsoft.MachineLearningServices/workspaces/mock2"):
            assert amlsecscan._get_log_analytics_from_diagnostic_settings() == "/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourcegroups/defaultresourcegroup-wus2/providers/microsoft.operationalinsights/workspaces/mock"
            assert mock_get.call_count == 6


@pytest.mark.skip(reason="requires sudo")
def test_install():
    amlsecscan._install("/subscriptions/d94a7037-ed50-426f-8a48-03035940fc7a/resourceGroups/WUS2/providers/Microsoft.OperationalInsights/workspaces/w7ed9d00ebb32")
    amlsecscan._uninstall()


@pytest.mark.skip(reason="only runs on AML CI")
def test_get_auth_environment_variables():
    env_var = amlsecscan._get_auth_environment_variables()
    assert len(env_var["MSI_ENDPOINT"]) > 0
    assert len(env_var["MSI_SECRET"]) > 0
