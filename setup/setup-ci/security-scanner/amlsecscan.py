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
import shutil
import subprocess
import sys
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
_config_folder_path = "/home/azureuser/.amlsecscan"
_global_config_path = _config_folder_path + "/config.json"
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

    _logger.debug(f"Creating folder {_config_folder_path}")
    os.makedirs(_config_folder_path, exist_ok=True)
    shutil.chown(_config_folder_path, "azureuser", "azureuser")

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

    _logger.info(f"Writing configuration file {_global_config_path}")
    with open(_global_config_path, "wt") as file:
        json.dump(config, file, indent=2)
    shutil.chown(_global_config_path, "azureuser", "azureuser")

    _logger.info("Installing Trivy")
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

    script_path = _config_folder_path + "/run.sh"
    _logger.info(f"Writing script file {script_path}")
    with open(script_path, "wt") as file:
        file.write(
            f"""#!/bin/bash
set -e
exec 1> >(logger -s -t AMLSECSCAN) 2>&1

# Limit CPU usage to 20% and reduce priority (note: the configuration is not persisted during reboot)
if [ ! -d /sys/fs/cgroup/cpu/amlsecscan ]
then
    mkdir -p /sys/fs/cgroup/cpu/amlsecscan
    echo 100000 | tee /sys/fs/cgroup/cpu/amlsecscan/cpu.cfs_period_us > /dev/null
    echo 20000 | tee /sys/fs/cgroup/cpu/amlsecscan/cpu.cfs_quota_us > /dev/null
    echo 5 | tee /sys/fs/cgroup/cpu/amlsecscan/cpu.shares > /dev/null
fi
echo $$ | tee /sys/fs/cgroup/cpu/amlsecscan/tasks > /dev/null

nice -n 19 python3 {os.path.abspath(__file__)} $1 $2 $3 $4 $5
"""
        )
    os.chmod(script_path, 0o0755)

    _logger.info(f"Writing crontab file /etc/cron.d/amlsecscan")
    with open("/etc/cron.d/amlsecscan", "wt") as file:
        file.write(
            f"""*/10 * * * * root {script_path} heartbeat
37 5 * * * root {script_path} scan all
@reboot root sleep 600 && {script_path} scan all
"""
        )
    os.chmod("/etc/cron.d/amlsecscan", 0o0644)


def _uninstall():
    if os.geteuid() != 0:
        raise Exception(
            "Uninstallation must be performed by the root user. Please run again using sudo."
        )

    _logger.info(f"Deleting crontab file /etc/cron.d/amlsecscan")
    _run("rm -f /etc/cron.d/amlsecscan")

    _logger.info(f"Deleting folder {_config_folder_path}")
    shutil.rmtree(_config_folder_path, ignore_errors=True)


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
        shutil.rmtree(f"{_config_folder_path}/anaconda", ignore_errors=True)
        for env_name in (
            entry.name for entry in os.scandir("/anaconda/envs") if entry.is_dir()
        ):
            _logger.info(
                f"Saving pip freeze of conda environment {env_name} to {_config_folder_path}/anaconda/{env_name}/requirements.txt"
            )
            os.makedirs(f"{_config_folder_path}/anaconda/{env_name}", exist_ok=True)
            _run(
                f"/anaconda/envs/{env_name}/bin/python3 -m pip freeze > {_config_folder_path}/anaconda/{env_name}/requirements.txt"
            )

        _logger.info("Running Trivy scan")
        _run(
            f"/usr/local/bin/trivy filesystem --format json --output {_config_folder_path}/trivy.json --security-checks vuln --severity HIGH,CRITICAL --ignore-unfixed /"
        )

        findings_os, findings_python = _parse_trivy_results(
            f"{_config_folder_path}/trivy.json"
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
