---
page_type: sample
languages:
- bash
- python
products:
- azure-machine-learning
description: Sample setup script to scan Compute Instances for malware and security vulnerabilities
---

# Compute Instance Security Scanner

[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../../../LICENSE)

A security scanner for Azure ML [Compute Instances](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-instance) reporting malware and vulnerabilities in OS and Python packages to [Azure Log Analytics](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/log-analytics-overview). For details on the vulnerability management process for the Azure Machine Learning service, see [Vulnerability Management](https://learn.microsoft.com/azure/machine-learning/concept-vulnerability-management).

## Getting Started

> Prerequisite: an Azure ML workspace with a Compute Instance running and [diagnostic logs](https://learn.microsoft.com/en-us/azure/machine-learning/monitor-azure-machine-learning) streaming to Log Analytics. See further down for alternative setups.

1. **Upload the scanner to Azure ML**:
   1. Download [`amlsecscan.py`](amlsecscan.py)
   2. Open [Azure ML Studio](https://ml.azure.com/)
   3. Go to the Notebooks tab
   4. Upload the file into your user folder `/Users/{user_name}` (replacing `{user_name}` with your user alias)
2. **Install the scanner**: open a terminal in Azure ML Notebooks and run `sudo ./amlsecscan.py install`
3. **Run a scan**: in the terminal, run `sudo ./amlsecscan.py scan all` (this takes a few minutes)

## Assessments

The security scanner installs [ClamAV](https://www.clamav.net/) to report malware and [Trivy](https://github.com/aquasecurity/trivy) to report OS and Python vulnerabilities.

Security scans are scheduled via CRON jobs to run either daily around 5AM or 10 minutes after OS startup. A CRON job also emits heartbeats every 10 minutes. Scans have their CPU usage limited to 20% and are deprioritized by running at priority 19.

Trivy is configured to report vulnerabilities of severity either `HIGH` or `CRITICAL` for which a fix is available. The ClamAV realtime scanning is not enabled.

## Telemetry

In Log Analytics, the scanner reports hearbeats to table `AmlSecurityComputeHealth_CL` and assessment results to `AmlSecurityComputeAssessments_CL`.

Examples of Log Analytics [KQL](https://docs.microsoft.com/en-us/azure/data-explorer/kql-quick-reference) queries:
- Recent heartbeats and scan status: `AmlSecurityComputeHealth_CL | top 100 by TimeGenerated desc`
- Recent assessments: `AmlSecurityComputeAssessments_CL | top 100 by TimeGenerated desc`

## Installation

Irrespective of how the scanner is installed, the scanner script must first be copied to the Azure ML workspace:
1. Download [amlsecscan.py](amlsecscan.py)
2. Open [Azure ML Studio](https://ml.azure.com/)
3. Go to the Notebooks tab
4. Upload the file into your user folder `/Users/{user_name}` (replacing `{user_name}` with your user alias)

The scanner can be installed on both existing and new Compute Instances.

### Existing Compute Instances

To install the scanner using default settings, run the `install` command without parameters:
```bash
sudo ./amlsecscan.py install
```
The scanner will use the first Log Analytics workspace to which the Azure ML workspace streams [diagnostic logs](https://learn.microsoft.com/en-us/azure/machine-learning/monitor-azure-machine-learning).

If diagnostic logs are not enabled or an alternative Log Analytics workspace needs to be used, its ARM Resource ID can be specified on the command line using the `-la` parameter:
```bash
sudo ./amlsecscan.py install -la /subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.OperationalInsights/workspaces/{workspace_name}
```

Another option, in case this configuration is reused multiple times, is to store the ARM Resource ID of the Log Analytics workspace in a JSON configuration file called
`amlsecscan.json` in the same folder as `amlsecscan.py`:
```json
{
    "logAnalyticsResourceId": "/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/providers/Microsoft.OperationalInsights/workspaces/{workspace_name}"
}
```

> The ARM Resource ID of a Log Analytics workspace can be obtained by opening the [Azure Portal](https://portal.azure.com), navigating to the Log Analytics workspace, and copying this substring from the browser URL.

### New Compute Instances

#### Using Azure ML Studio

1. Create a file called `amlsecscan.sh` with content `sudo python3 amlsecscan.py install` .
2. Open the [Compute Instance list](https://ml.azure.com/compute/list) in [Azure ML Studio](https://ml.azure.com)
3. Click on the `+ New` button
4. In the pop-up, select the machine name and size then click `Next: Advanced Settings`
5. Toggle `Provision with setup script`, select `Local file`, and pick `amlsecscan.sh`
6. Click on the `Create` button

#### Using an ARM Template

For automated deployments, the scanner can be installed as part of the ARM templates deploying the Compute Instances.

Start by creating an ARM template, say `deploy.json`, with a `setupScripts` section. In the example below replace `{azure_ml_workspace_name}`, `{azure_ml_compute_name}`, `{user_name}` with appropriate values. You may also want to adjust `location`/`VMSize`, add `schedules`, etc.
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "resources": [
    {
      "type": "Microsoft.MachineLearningServices/workspaces/computes",
      "name": "{azure_ml_workspace_name}/{azure_ml_compute_name}",
      "location": "westus2",
      "apiVersion": "2021-07-01",
      "properties": {
        "computeType": "ComputeInstance",
        "disableLocalAuth": true,
        "properties": {
          "VMSize": "Standard_F2s_v2",
          "applicationSharingPolicy": "Personal",
          "sshSettings": { "sshPublicAccess": "Disabled" },
          "setupScripts": {
            "scripts": {
              "creationScript": {
                "scriptSource":"inline",
                "scriptData":"[base64('sudo python3 {user_name}/amlsecscan.py install')]",
                "timeout": "10m"
              }
            }
          }
        }
      }
    }
  ]
}
```

Deploy the ARM template using the [Az CLI](https://docs.microsoft.com/en-us/cli/azure/):
```bash
az login
az account set --subscription {subscription_id}
az deployment group create --resource-group {resource_group_name} --template-file deploy.json
```

## Clean Up

To stop scan scheduling and remove the scanner, run `sudo ./amlsecscan.py uninstall` .

## Troubleshooting

### Ensure that telemetry is emitted

Check the scanner health in Log Analytics: `AmlSecurityComputeHealth_CL | top 100 by TimeGenerated desc` .

It should show heartbeats with `Type_s == 'Heartbeat'` every 10 minutes.

Scan status (`Type_s == 'ScanMalware'` or `Type_s == 'ScanVulnerabilities'`) should appear as pairs of log entries,
one with `Status_s = 'Started'` followed by one with `Status_s = 'Succeeded'`. If `Status_s` is `Failed`,  the `Details_s` field includes the error message.

If heartbeats are not present in Log Analytics, verify whether heartbeats can be emitted by running `./amlsecscan.py heartbeat` in a terminal on the Compute Instance.

### Ensure that the scanner is running

If logs are missing in Log Analytics, scans may not be running. Local logs are available in `syslog` for investigation:
- Check `cron` logs: `sudo cat /var/log/syslog | grep -i cron`
- Check scanner logs: `sudo cat /var/log/syslog | grep -i amlsecscan`

The CRON configuration is located at `/etc/cron.d/amlsecscan` .

Scans can be run manually with higher verbosity to get more details: `sudo /home/azureuser/.amlsecscan/run.sh scan all -ll DEBUG` .

### Investigate Compute Instance deployment failures

Compute Instance creation logs are stored under `/Logs/{azure_ml_compute_name}/creation`.
They can also be found by selecting the Compute Instance in Azure ML Studio, clicking on the `Logs` tab, and opening the file `Setup > stdout`.

### Verify that malware gets reported

Malware detection can be verified by downloading a simulated malware file: `wget -O ~/eicar.com.txt https://secure.eicar.org/eicar.com.txt` .

The malware should be reported in Log Analytics: `AmlSecurityComputeAssessments_CL | where Type_s == 'Malware' | top 100 by TimeGenerated desc` .

### Verify that the scanner files are present

After installation, the following files should be present on the Compute Instance:

File|Description
--|--
`/home/azureuser/.amlsecscan/config.json`|Scanner configuration
`/home/azureuser/.amlsecscan/run.sh`|Scanner CRON entry point
`/etc/cron.d/amlsecscan`|Scanner CRON schedule

### Verify that resource-usage limits are in place

When running through CRON schedule, scans have their CPU usage limited to 20% and are deprioritized by running at priority 19.
When running manually, CPU usage is not limited and priority is left as default.

After a scan is run, its `cgroups` configuration limiting resource usage can be found under `/sys/fs/cgroup/cpu/amlsecscan` .
