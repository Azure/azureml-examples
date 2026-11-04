#!/usr/bin/env python3

import argparse
import base64
import datetime
import hashlib
import hmac
import json
import logging
import logging.handlers
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import requests
from urllib.parse import urlparse

# User accounts running this script:
# _azbatch  - user account running scripts during Azure ML Compute Instance creation - can sudo
# azureuser - user account running scripts after Azure ML Compute Instance creation - can sudo on Compute Instances created with rootAccess = true (default)
# root      - does not need introduction...

_logger = logging.getLogger("amlsecscan")
_computer = os.environ["CI_NAME"]
_azure_ml_resource_id = (
    "/" + urlparse(os.environ["MLFLOW_TRACKING_URI"]).path.split("/", 3)[3]
)  # Get the ARM Resource ID of the Azure ML Workspace we are running on

# Configuration priority: 1) command-line parameters, 2) local config file, 3) global config file
_config_folder_path = "/opt/amlsecscan"
_global_config_path = _config_folder_path + "/config.json"
_installed_scanner_path = _config_folder_path + "/amlsecscan.py"
_state_folder_path = "/var/lib/amlsecscan"
_cron_path = "/etc/cron.d/amlsecscan"
_local_config_path = os.path.abspath(os.path.splitext(__file__)[0] + ".json")


# Replacement for azure.identity.DefaultAzureCredential().get_token since azure.identity is not available in the conda base environment and does not handle Azure ML's MSI
def _get_access_token(resource):
    # Ensure the MSI environment variables are set (by default, they are set in shells when running in AML Studio Terminal but not when running in CRON)
    if "MSI_ENDPOINT" not in os.environ or "MSI_SECRET" not in os.environ:
        env_var = _get_auth_environment_variables()
        os.environ["MSI_ENDPOINT"] = env_var["MSI_ENDPOINT"]
        os.environ["MSI_SECRET"] = env_var["MSI_SECRET"]

    url = f"{os.environ['MSI_ENDPOINT']}?resource={resource}&api-version=2017-09-01"
    client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID", None)
    if (
        client_id is not None
        and re.match(
            "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            client_id,
            re.IGNORECASE,
        )
        is not None
    ):
        url = f"{url}&clientid={client_id}"
    resp = requests.get(url, headers={"Secret": os.environ["MSI_SECRET"]})
    resp.raise_for_status()
    return resp.json()["access_token"]


def _run(command, check=True):
    # To be compatible with Python 3.6 (default python for root user), 'text' and 'capture_output' cannot be used
    try:
        return subprocess.run(
            command,
            shell=True,
            check=check,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        _logger.exception(
            f"Error: {e}\n    stdout:\n{e.stdout}\n    stderr:\n{e.stderr}"
        )
        raise


def _is_trusted_root_directory(path, allowed_entries=None):
    try:
        directory_stat = os.lstat(path)
    except FileNotFoundError:
        return False

    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_uid != 0
        or directory_stat.st_gid != 0
        or stat.S_IMODE(directory_stat.st_mode) & 0o022
    ):
        return False

    if allowed_entries is None:
        return True

    for entry in os.scandir(path):
        if entry.name not in allowed_entries:
            return False
        entry_stat = entry.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(entry_stat.st_mode)
            or entry_stat.st_uid != 0
            or entry_stat.st_gid != 0
            or stat.S_IMODE(entry_stat.st_mode) & 0o022
        ):
            return False

    return True


def _ensure_root_owned_directory(path, mode, allowed_entries=None):
    if _is_trusted_root_directory(path, allowed_entries):
        shutil.chown(path, "root", "root")
        os.chmod(path, mode)
        return

    if os.path.lexists(path):
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.unlink(path)

    os.makedirs(path, mode=mode, exist_ok=False)
    shutil.chown(path, "root", "root")
    os.chmod(path, mode)


def _stage_root_owned_file(path, content, mode):
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".amlsecscan-", dir=os.path.dirname(path)
    )
    try:
        if isinstance(content, bytes):
            with os.fdopen(descriptor, "wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())

        shutil.chown(temporary_path, "root", "root")
        os.chmod(temporary_path, mode)
        return temporary_path
    except Exception:
        if os.path.lexists(temporary_path):
            os.unlink(temporary_path)
        raise


def _create_backup(path):
    descriptor, backup_path = tempfile.mkstemp(
        prefix=".amlsecscan-backup-", dir=os.path.dirname(path)
    )
    os.close(descriptor)
    os.unlink(backup_path)
    try:
        os.link(path, backup_path, follow_symlinks=False)
        return backup_path
    except Exception:
        if os.path.lexists(backup_path):
            os.unlink(backup_path)
        raise


def _write_files_atomically(files):
    staged_files = []
    backups = {}
    replaced_paths = []
    try:
        for path, content, mode in files:
            staged_files.append((path, _stage_root_owned_file(path, content, mode)))

        for path, _ in staged_files:
            backups[path] = _create_backup(path) if os.path.lexists(path) else None

        for path, staged_path in staged_files:
            os.replace(staged_path, path)
            replaced_paths.append(path)
    except Exception:
        for path in reversed(replaced_paths):
            backup_path = backups[path]
            if backup_path is None:
                os.unlink(path)
            else:
                os.replace(backup_path, path)
        raise
    finally:
        for _, staged_path in staged_files:
            if os.path.lexists(staged_path):
                os.unlink(staged_path)
        for backup_path in backups.values():
            if backup_path is not None and os.path.lexists(backup_path):
                os.unlink(backup_path)


class StdOutTelemetry:
    def send(self, log_type, data):
        print(json.dumps({"table": log_type, "rows": data}))


class LogAnalyticsTelemetry:
    def __init__(self, log_analytics_resource_id):

        # Get the ARM Resource ID of the Log Analytics Workspace
        if log_analytics_resource_id is None:
            config_path = (
                _local_config_path
                if os.path.exists(_local_config_path)
                else _global_config_path
                if os.path.exists(_global_config_path)
                else None
            )
            if config_path is not None:
                _logger.debug(f"Loading configuration from {config_path}")
                with open(config_path, "rt") as file:
                    config = json.load(file)
                log_analytics_resource_id = config["logAnalyticsResourceId"]
        self.log_analytics_resource_id = _sanitize_log_analytics_resource_id(
            log_analytics_resource_id
        )

        # Get an AAD access token for ARM
        access_token = _get_access_token("https://management.azure.com")
        headers = {
            "Authorization": "Bearer " + access_token
        }  # [SuppressMessage("Microsoft.Security", "CS001:SecretInline", Justification="No secret")]

        # Get the Log Analytics Customer ID from ARM
        response = requests.get(
            "https://management.azure.com"
            + self.log_analytics_resource_id
            + "?api-version=2021-06-01",
            headers=headers,
        )
        response.raise_for_status()
        self.log_analytics_customer_id = response.json()["properties"]["customerId"]

        # Get the Log Analytics Shared Key from ARM
        response = requests.post(
            "https://management.azure.com"
            + self.log_analytics_resource_id
            + "/sharedKeys?api-version=2020-08-01",
            headers=headers,
        )
        response.raise_for_status()
        self.log_analytics_shared_key = response.json()["primarySharedKey"]

        _logger.debug(f"Azure ML Workspace ARM Resource ID: {_azure_ml_resource_id}")
        _logger.debug(
            f"Log Analytics Workspace ARM Resource ID: {self.log_analytics_resource_id}"
        )
        _logger.debug(f"Log Analytics Customer ID: {self.log_analytics_customer_id}")

    # From: https://docs.microsoft.com/en-us/azure/azure-monitor/logs/data-collector-api#python-sample
    def _build_signature(self, date, content_length, method, content_type, resource):
        x_headers = "x-ms-date:" + date
        string_to_hash = (
            method
            + "\n"
            + str(content_length)
            + "\n"
            + content_type
            + "\n"
            + x_headers
            + "\n"
            + resource
        )
        bytes_to_hash = bytes(string_to_hash, encoding="utf-8")
        decoded_key = base64.b64decode(self.log_analytics_shared_key)
        encoded_hash = base64.b64encode(
            hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()
        ).decode()
        authorization = "SharedKey {}:{}".format(
            self.log_analytics_customer_id, encoded_hash
        )
        return authorization

    def send(self, log_type, data):
        body = json.dumps(data)
        method = "POST"
        content_type = "application/json"
        resource = "/api/logs"
        rfc1123date = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
        content_length = len(body)
        signature = self._build_signature(
            rfc1123date, content_length, method, content_type, resource
        )

        headers = {
            "content-type": content_type,
            "Authorization": signature,
            "Log-Type": log_type,
            "x-ms-date": rfc1123date,
        }

        response = requests.post(
            "https://"
            + self.log_analytics_customer_id
            + ".ods.opinsights.azure.com"
            + resource
            + "?api-version=2016-04-01",
            data=body,
            headers=headers,
        )
        response.raise_for_status()
        _logger.info(
            f"Sent {len(data)} telemetry row(s) to table {log_type} in Log Analytics workspace {self.log_analytics_resource_id}"
        )
        _logger.debug(f"Telemetry rows: {data}")


def _send_health(telemetry, type_, status=None, details=None):
    telemetry.send(
        "AmlSecurityComputeHealth",
        [
            {
                "WorkspaceId": _azure_ml_resource_id,
                "Computer": _computer,
                "Type": type_,  # Enum: Heartbeat, ScanMalware, ScanOsVulnerabilities, ScanPythonVulnerabilities
                "Status": status
                if status is not None
                else "",  # Enum: Started, Succeeded, Failed, ''
                "Details": json.dumps(details) if details is not None else "",
            }
        ],
    )


def _send_assessment(telemetry, type_, findings, details=None):
    telemetry.send(
        "AmlSecurityComputeAssessments",
        [
            {
                "WorkspaceId": _azure_ml_resource_id,
                "Computer": _computer,
                "Type": type_,  # Enum: Malware, OsVulnerabilities, PythonVulnerabilities
                "Status": "Healthy" if findings == 0 else "Unhealthy",
                "Findings": findings,
                "Details": json.dumps(details) if details is not None else "",
            }
        ],
    )


def _get_log_analytics_from_diagnostic_settings():
    # Get an AAD access token for ARM
    access_token = _get_access_token("https://management.azure.com")
    headers = {
        "Authorization": "Bearer " + access_token
    }  # [SuppressMessage("Microsoft.Security", "CS001:SecretInline", Justification="No secret")]

    # List diagnostic settings on the Azure ML workspace
    response = requests.get(
        "https://management.azure.com"
        + _azure_ml_resource_id
        + "/providers/microsoft.insights/diagnosticSettings?api-version=2021-05-01-preview",
        headers=headers,
    )
    response.raise_for_status()

    # Select the first Log Analytics workspace
    for settings in response.json()["value"]:
        if "workspaceId" in settings["properties"]:
            return settings["properties"]["workspaceId"]
    return None


def _install(log_analytics_resource_id):
    if os.geteuid() != 0:
        raise Exception(
            "Installation must be performed by the root user. Please run again using sudo."
        )

    with open(os.path.abspath(__file__), "rb") as file:
        scanner_source = file.read()

    config = {"logAnalyticsResourceId": None}

    # Load config file if present
    if os.path.exists(_local_config_path):
        _logger.debug(f"Loading configuration from {_local_config_path}")
        with open(_local_config_path, "rt") as file:
            config.update(json.load(file))
        _logger.debug(
            f"logAnalyticsResourceId after loading config file: {config['logAnalyticsResourceId']}"
        )

    # Set Log Analytics workspace ARM Resource ID if passed via command-line parameter
    if log_analytics_resource_id is not None:
        config["logAnalyticsResourceId"] = log_analytics_resource_id
        _logger.debug(
            f"logAnalyticsResourceId after setting command-line parameter: {config['logAnalyticsResourceId']}"
        )

    # Retrieve Log Analytics workspace ARM Resource ID from Azure ML diagnostic settings if
    # provided neither via local config file nor command-line parameter
    if config.get("logAnalyticsResourceId", None) is None:
        config["logAnalyticsResourceId"] = _get_log_analytics_from_diagnostic_settings()
        _logger.debug(
            f"logAnalyticsResourceId after querying Azure ML diagnostic settings: {config['logAnalyticsResourceId']}"
        )

    # Sanitize the Log Analytics workspace ARM Resource ID
    config["logAnalyticsResourceId"] = _sanitize_log_analytics_resource_id(
        config["logAnalyticsResourceId"]
    )

    _logger.debug(f"Configuration: {config}")

    _logger.info("Installing Trivy")
    _run("apt-get update")
    _run(
        "apt-get install -y --no-install-recommends --quiet wget apt-transport-https gnupg lsb-release"
    )
    _run(
        "wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add -"
    )
    _run(
        "echo deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main | tee -a /etc/apt/sources.list.d/trivy.list"
    )
    _run("apt-get update")
    _run("apt-get install -y --no-install-recommends --quiet trivy")

    _logger.debug(f"Ensuring folder {_config_folder_path}")
    _ensure_root_owned_directory(
        _config_folder_path,
        0o0755,
        {"config.json", "amlsecscan.py", "run.sh"},
    )

    _logger.debug(f"Ensuring state folder {_state_folder_path}")
    _ensure_root_owned_directory(_state_folder_path, 0o0755)

    script_path = _config_folder_path + "/run.sh"
    script = f"""#!/bin/bash
set -e
exec 1> >(logger -s -t AMLSECSCAN) 2>&1

# Limit CPU usage to 20% and reduce priority (note: the configuration is not persisted during reboot)
configure_cgroup() {{
    if [ -f /sys/fs/cgroup/cgroup.controllers ]
    then
        cgroup_path=/sys/fs/cgroup/amlsecscan
        mkdir -p "$cgroup_path" || return 0
        [ -w "$cgroup_path/cpu.max" ] && echo "20000 100000" > "$cgroup_path/cpu.max"
        [ -w "$cgroup_path/cpu.weight" ] && echo 5 > "$cgroup_path/cpu.weight"
        [ -w "$cgroup_path/cgroup.procs" ] && echo $$ > "$cgroup_path/cgroup.procs"
    elif [ -d /sys/fs/cgroup/cpu ]
    then
        cgroup_path=/sys/fs/cgroup/cpu/amlsecscan
        mkdir -p "$cgroup_path" || return 0
        [ -w "$cgroup_path/cpu.cfs_period_us" ] && echo 100000 > "$cgroup_path/cpu.cfs_period_us"
        [ -w "$cgroup_path/cpu.cfs_quota_us" ] && echo 20000 > "$cgroup_path/cpu.cfs_quota_us"
        [ -w "$cgroup_path/cpu.shares" ] && echo 5 > "$cgroup_path/cpu.shares"
        [ -w "$cgroup_path/tasks" ] && echo $$ > "$cgroup_path/tasks"
    fi
}}
configure_cgroup || true

nice -n 19 python3 {_installed_scanner_path} "$@"
"""

    cron = f"""*/10 * * * * root {script_path} heartbeat
37 5 * * * root {script_path} scan all
@reboot root sleep 600 && {script_path} scan all
"""

    _logger.info("Writing scanner files and CRON schedule")
    _write_files_atomically(
        [
            (_global_config_path, json.dumps(config, indent=2), 0o0644),
            (_installed_scanner_path, scanner_source, 0o0755),
            (script_path, script, 0o0755),
            (_cron_path, cron, 0o0644),
        ]
    )


def _uninstall():
    if os.geteuid() != 0:
        raise Exception(
            "Uninstallation must be performed by the root user. Please run again using sudo."
        )

    _logger.info(f"Deleting crontab file {_cron_path}")
    _run(f"rm -f {_cron_path}")

    _logger.info(f"Deleting folder {_config_folder_path}")
    shutil.rmtree(_config_folder_path, ignore_errors=True)

    _logger.info(f"Deleting state folder {_state_folder_path}")
    shutil.rmtree(_state_folder_path, ignore_errors=True)


def _sanitize_log_analytics_resource_id(log_analytics_resource_id):
    if log_analytics_resource_id is None:
        raise ValueError(
            "Log Analytics Workspace ARM Resource ID missing. Please provide it either via config file, command-line parameter, or Azure ML diagnostic settings."
        )

    log_analytics_resource_id = log_analytics_resource_id.strip()

    if len(log_analytics_resource_id.split("/")) != 9:
        raise ValueError(
            "Log Analytics Workspace ARM Resource ID format should be /subscriptions/{subscription}/resourceGroups/{resource_group}/providers/Microsoft.OperationalInsights/workspaces/{workspace} instead of '"
            + log_analytics_resource_id
            + "'"
        )

    return log_analytics_resource_id


def _get_auth_environment_variables():
    out = _run("cat /etc/environment.sso")
    return {
        pair[0]: pair[1]
        for pair in [line.split("=", 2) for line in out.stdout.splitlines()]
    }


def _parse_clamav_stdout(stdout):

    files = []
    details = {}
    findings = 0

    for line in stdout.splitlines():

        match = re.match(r"^(.+?):\s*(.+?)\s+FOUND", line)
        if match is not None:
            files.append({"path": match.group(1), "malwareType": match.group(2)})
            continue

        match = re.match(r"Infected files:\s*(\d+)", line)
        if match is not None:
            findings = int(match.group(1))
            continue

        match = re.match(r"Known viruses:\s*(\d+)", line)
        if match is not None:
            details["knownViruses"] = int(match.group(1))
            continue

        match = re.match(r"Engine version:\s*(.+)", line)
        if match is not None:
            details["engineVersion"] = match.group(1)
            continue

        match = re.match(r"Scanned files:\s*(\d+)", line)
        if match is not None:
            details["scannedFiles"] = int(match.group(1))
            continue

        match = re.match(r"Scanned directories:\s*(\d+)", line)
        if match is not None:
            details["scannedDirectories"] = int(match.group(1))
            continue

    if findings != len(files):
        raise Exception(
            f"Failed to parse ClamAV stdout (findings: {findings}, files: {len(files)})"
        )

    if len(files) > 0:
        details["files"] = files

    return (findings, details)


def _parse_trivy_results(trivy_scan_path):

    findings_os = []
    findings_python = []
    with open(trivy_scan_path, "rt") as file:
        data = json.load(file)

        for result in data["Results"]:
            if result["Class"] == "os-pkgs":
                for vulnerability in result.get("Vulnerabilities", []):
                    findings_os.append(
                        {
                            "title": vulnerability.get(
                                "Title",
                                vulnerability["PkgName"]
                                + " "
                                + vulnerability["VulnerabilityID"],
                            ),
                            "packageName": vulnerability["PkgName"],
                            "packageVersion": vulnerability["InstalledVersion"],
                            "CVE": vulnerability["VulnerabilityID"],
                            "severity": vulnerability["Severity"],
                        }
                    )
            elif result["Class"] == "lang-pkgs" and result["Type"] == "pip":
                for vulnerability in result.get("Vulnerabilities", []):
                    findings_python.append(
                        {
                            "title": vulnerability.get(
                                "Title",
                                vulnerability["PkgName"]
                                + " "
                                + vulnerability["VulnerabilityID"],
                            ),
                            "packageName": vulnerability["PkgName"],
                            "packageVersion": vulnerability["InstalledVersion"],
                            "file": result["Target"],
                            "CVE": vulnerability["VulnerabilityID"],
                            "severity": vulnerability["Severity"],
                        }
                    )
            else:
                _logger.warning(
                    f"Skipping unhandled vulnerability of class {result['Class']} and type {result['Type']} for file {result['Target']}. "
                )

    return (findings_os, findings_python)


# Limit the finding list to top 50 by severity so that the Log Analytics limit of 32K string length is not hit (which truncates JSON strings and makes them invalid)
def _filter_trivy_results(findings):
    return sorted(
        findings,
        key=lambda x: 0
        if x["severity"] == "CRITICAL"
        else 1
        if x["severity"] == "HIGH"
        else 2,
    )[:50]


def _scan_vulnerabilities(telemetry):

    start_time = time.time()
    _send_health(telemetry, "ScanVulnerabilities", "Started")

    try:
        os.makedirs(_state_folder_path, exist_ok=True)
        shutil.rmtree(f"{_state_folder_path}/anaconda", ignore_errors=True)
        for env_name in (
            entry.name for entry in os.scandir("/anaconda/envs") if entry.is_dir()
        ):
            requirements_path = (
                f"{_state_folder_path}/anaconda/{env_name}/requirements.txt"
            )
            _logger.info(
                f"Saving pip freeze of conda environment {env_name} to {requirements_path}"
            )
            os.makedirs(os.path.dirname(requirements_path), exist_ok=True)
            _run(
                f"{shlex.quote(f'/anaconda/envs/{env_name}/bin/python3')} -m pip freeze > {shlex.quote(requirements_path)}"
            )

        _logger.info("Running Trivy scan")
        _run(
            f"/usr/local/bin/trivy filesystem --format json --output {_state_folder_path}/trivy.json --security-checks vuln --severity HIGH,CRITICAL --ignore-unfixed /"
        )

        findings_os, findings_python = _parse_trivy_results(
            f"{_state_folder_path}/trivy.json"
        )

        _send_assessment(
            telemetry,
            "OsVulnerabilities",
            len(findings_os),
            {"findings": _filter_trivy_results(findings_os)}
            if len(findings_os) > 0
            else None,
        )
        _send_assessment(
            telemetry,
            "PythonVulnerabilities",
            len(findings_python),
            {"findings": _filter_trivy_results(findings_python)}
            if len(findings_python) > 0
            else None,
        )
        _send_health(
            telemetry,
            "ScanVulnerabilities",
            "Succeeded",
            {"elapsedTimeInS": time.time() - start_time},
        )
        return True

    except subprocess.CalledProcessError as e:
        _send_health(
            telemetry,
            "ScanVulnerabilities",
            "Failed",
            {
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "elapsedTimeInS": time.time() - start_time,
            },
        )
        return False
    except Exception as e:
        _logger.exception(f"Error: {e}")
        _send_health(
            telemetry,
            "ScanVulnerabilities",
            "Failed",
            {"error": str(e), "elapsedTimeInS": time.time() - start_time},
        )
        return False


def _scan_malware(telemetry):

    start_time = time.time()
    _send_health(telemetry, "ScanMalware", "Started")

    try:

        # Run ClamAV (with AzSecPack malware definitions if present)
        database_option = (
            "-d /var/lib/azsec-clamav"
            if os.path.exists("/var/lib/azsec-clamav")
            else ""
        )
        command = (
            f"clamscan {database_option} -r -i --exclude-dir=^/sys/ /bin /boot /home /lib /lib64 /opt /root /sbin /anaconda",
        )
        _logger.info(f"Running: {command}")
        out = _run(command, check=False)

        # returncode:
        # == 0 -> clamscan completed scan without finding malware
        # == 1 -> clamscan completed scan with malware found
        # >= 2 -> clamscan failed to scan
        if out.returncode >= 2:
            raise Exception(f"Scan failed with exit code {out.returncode}")

        findings, details = _parse_clamav_stdout(out.stdout)

        if findings == 0 and out.returncode != 0:
            raise Exception(
                f"Failed to parse ClamAV stdout (findings: {findings}, exit code: {out.returncode})"
            )

        _send_assessment(telemetry, "Malware", findings, details)
        _send_health(
            telemetry,
            "ScanMalware",
            "Succeeded",
            {"elapsedTimeInS": time.time() - start_time},
        )
        return True

    except subprocess.CalledProcessError as e:
        _send_health(
            telemetry,
            "ScanMalware",
            "Failed",
            {
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "elapsedTimeInS": time.time() - start_time,
            },
        )
        return False
    except Exception as e:
        _logger.exception(e)
        _send_health(
            telemetry,
            "ScanMalware",
            "Failed",
            {"error": str(e), "elapsedTimeInS": time.time() - start_time},
        )
        return False


def _add_common_arguments(parser):
    parser.add_argument(
        "-la",
        "--log-analytics-resource-id",
        help="ARM Resource ID of the Log Analytics workspace to log telemetry to",
        dest="log_analytics_resource_id",
    )
    parser.add_argument(
        "-ll",
        "--log-level",
        help="level of log messages to display (default: INFO)",
        dest="log_level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output (default: log-analytics)",
        dest="output",
        choices=["log-analytics", "stdout"],
    )


if __name__ == "__main__":
    # Logging to stdout (forwarded to syslog in run.sh)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    try:
        # Command-line parser
        parser = argparse.ArgumentParser(
            description="Azure ML Compute Security Scanner"
        )
        subparsers = parser.add_subparsers(dest="command")

        # Command: "install"
        parser_install = subparsers.add_parser(
            "install",
            help="Install dependencies and start scheduled scans. Must be run as root (use sudo).",
        )
        _add_common_arguments(parser_install)

        # Command: "uninstall"
        parser_uninstall = subparsers.add_parser(
            "uninstall", help="Remove scheduled scans. Must be run as root (use sudo)."
        )
        _add_common_arguments(parser_uninstall)

        # Command: "heartbeat"
        parser_heartbeat = subparsers.add_parser(
            "heartbeat", help="Emit a telemetry heartbeat"
        )
        _add_common_arguments(parser_heartbeat)

        # Command: "scan"
        parser_scan = subparsers.add_parser("scan", help="Run security scans")
        subparsers_scan = parser_scan.add_subparsers(dest="scan_type")

        # Command: "scan all"
        parser_scan_all = subparsers_scan.add_parser(
            "all", help="Run all security scans"
        )
        _add_common_arguments(parser_scan_all)

        # Command: "scan malware"
        parser_scan_malware = subparsers_scan.add_parser(
            "malware", help="Scan for malware"
        )
        _add_common_arguments(parser_scan_malware)

        # Command: "scan vulnerabilities"
        parser_scan_vulnerabilities = subparsers_scan.add_parser(
            "vulnerabilities", help="Scan for OS and Python vulnerabilities"
        )
        _add_common_arguments(parser_scan_vulnerabilities)

        args = parser.parse_args()

        if args.command is None:
            parser.print_help()
            exit(1)

        if "log_level" in args and args.log_level is not None:
            _logger.setLevel(getattr(logging, args.log_level))

        if args.command == "install":
            _install(args.log_analytics_resource_id)
        elif args.command == "uninstall":
            _uninstall()
        elif args.command == "heartbeat":
            telemetry = (
                StdOutTelemetry()
                if args.output == "stdout"
                else LogAnalyticsTelemetry(args.log_analytics_resource_id)
            )
            _send_health(telemetry, "Heartbeat")
        elif args.command == "scan":
            if args.scan_type is None:
                parser.print_help()
                exit(1)
            telemetry = (
                StdOutTelemetry()
                if args.output == "stdout"
                else LogAnalyticsTelemetry(args.log_analytics_resource_id)
            )
            if args.scan_type == "all":
                success0 = _scan_vulnerabilities(telemetry)
                success1 = _scan_malware(telemetry)
                exit(0 if success0 and success1 else 2)
                # TODO: Python vulns
            elif args.scan_type == "vulnerabilities":
                success = _scan_vulnerabilities(telemetry)
                exit(0 if success else 2)
            elif args.scan_type == "malware":
                success = _scan_malware(telemetry)
                exit(0 if success else 2)
            else:
                raise ValueError(f"Insupported scan type '{args.scan_type}'")
        else:
            raise ValueError(f"Insupported command '{args.command}'")
    except Exception as e:
        _logger.critical(f"Unhandled exception: {e}")
        raise
