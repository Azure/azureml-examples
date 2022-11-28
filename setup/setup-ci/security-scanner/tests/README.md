# Unit tests

To get started, create a conda environment:
```bash
conda create --yes --name test-amlsecscan python=3.8
conda activate test-amlsecscan
pip install requests~=2.28 pytest~=7.1
```

Then run the unit tests:
```bash
pytest
```

# Integration tests

## Test on an existing Compute Instance

1. Open a terminal on the Compute Instance (with `root` user enabled) and run:
```bash
sudo ./amlsecscan.py install
./amlsecscan.py heartbeat
sudo ./amlsecscan.py scan all
```
2. Check [Log Analytics](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/log-analytics-overview) for hearbeats and scan status in table `AmlSecurityComputeHealth_CL` and assessment results in `AmlSecurityComputeAssessments_CL`.

## Test on a new Compute Instance

### Using Azure ML Studio

1. Open the [Compute Instance list](https://ml.azure.com/compute/list) in [Azure ML Studio](https://ml.azure.com)
2. Click the `+ New` button
3. In the pop-up, select a small `Standard_F2s_v2` as machine size
4. Click `Next: Advanced Settings`
5. Set a schedule to shutdown daily
6. Toggle `Provision with setup script`, select `Local file`, and pick `amlsecscan.sh`
7. After the Compute Instance is successfully created, check the Compute Instance logs under `Setup > stdout`. There should be no error.
8. Check [Log Analytics](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/log-analytics-overview) for hearbeats using query `AmlSecurityComputeHealth_CL | top 100 by TimeGenerated desc`.
9. Delete the Compute Instance.

### Using an ARM template

1. Update `name` along with the user name in `scriptData` in `deploy.json` as needed
2. Deploy by running:
```bash
az deployment group create --resource-group {name} --template-file deploy.json
```
3. Check the Compute Instance logs under `Setup > stdout`. There should be no error.
4. Check [Log Analytics](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/log-analytics-overview) for hearbeats using query `AmlSecurityComputeHealth_CL | top 100 by TimeGenerated desc`.
5. Delete the Compute Instance using Azure ML Studio.

## Test on a local Docker container

1. Update `logAnalyticsResourceId` in `amlsecscan.json`
2. Build the Docker image and start the container:
```bash
docker build --tag test-amlsecscan-with-root --file with-root.dockerfile ..
docker run -it test-amlsecscan-with-root
```
3. In the container, start the mock MSI server (prompts for AAD device-code authentication):
```bash
/home/mock-msi.py &
```
4. Install the security scanner, emit a heartbeat, and run a scan:
```bash
apt-get update
freshclam

/home/amlsecscan.py install -ll DEBUG
/home/amlsecscan.py heartbeat -ll DEBUG
/home/amlsecscan.py scan all -ll DEBUG
```
5. Check [Log Analytics](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/log-analytics-overview) for hearbeats and scan status in table `AmlSecurityComputeHealth_CL` and assessment results in `AmlSecurityComputeAssessments_CL`.
The `Computer` name is `mock-host-with-root`. Assessments should report `/home/eicar.com.txt` as malware and `/home/requirements.txt` as having a critical Python vulnerability.
