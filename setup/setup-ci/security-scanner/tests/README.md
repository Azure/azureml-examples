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

Open a terminal on the Compute Instance (with `root` user enabled) and run:
```bash
sudo ./amlsecscan.py install
./amlsecscan.py heartbeat
sudo ./amlsecscan.py scan all
```

Check Log Analytics for hearbeats and scan status in table `AmlSecurityComputeHealth_CL` and assessment results in `AmlSecurityComputeAssessments_CL`.

## Test on a new Compute Instance

### Using Azure ML Studio

Open the [Compute Instance list](https://ml.azure.com/compute/list) in [Azure ML Studio](https://ml.azure.com), then click the `+ New` button.
In the pop-up, select a small `Standard_F2s_v2` as machine size, then click `Next: Advanced Settings`.
Set a schedule to shutdown daily. Then toggle `Provision with setup script`, select `Local file`, and pick `amlsecscan.sh`.

After the Compute Instance is successfully created, check the Compute Instance logs under `Setup > stdout`. There should be no error.

Then check Log Analytics for hearbeats using query `AmlSecurityComputeHealth_CL | top 100 by TimeGenerated desc`.

Finally, delete the Compute Instance.

### Using an ARM template

Update `name` along with the user name in `scriptData` in `deploy.json` as needed, then deploy by running:
```bash
az deployment group create --resource-group {name} --template-file deploy.json
```

Check the Compute Instance logs under `Setup > stdout`. There should be no error.

Then check Log Analytics for hearbeats using query `AmlSecurityComputeHealth_CL | top 100 by TimeGenerated desc`.

Finally, delete the Compute Instance using Azure ML Studio.

## Test on a local Docker container

Update `logAnalyticsResourceId` in `amlsecscan.json`.

Build the Docker image and start the container:
```bash
docker build --tag test-amlsecscan-with-root --file with-root.dockerfile ..
docker run -it test-amlsecscan-with-root
```

In the container, start the mock MSI server (prompts for AAD device-code authentication):
```bash
/home/mock-msi.py &
```

Install the security scanner, emit a heartbeat, and run a scan:
```bash
apt-get update
freshclam

/home/amlsecscan.py install -ll DEBUG
/home/amlsecscan.py heartbeat -ll DEBUG
/home/amlsecscan.py scan all -ll DEBUG
```

Check Log Analytics for hearbeats and scan status in table `AmlSecurityComputeHealth_CL` and assessment results in `AmlSecurityComputeAssessments_CL`.
The `Computer` name is `mock-host-with-root`. Assessments should report `/home/eicar.com.txt` as malware and `/home/requirements.txt` as having a critical Python vulnerability.
