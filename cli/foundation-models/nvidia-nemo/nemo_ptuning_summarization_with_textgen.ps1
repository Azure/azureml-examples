$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi; Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'; Remove-Item .\AzureCLI.msi

az extension add -n ml

az login

az ml job create -g <RESOURCE_GROUP> -w <WORKSPACE> --subscription <SUBSCRIPTION> -f nemo_pipeline_job.yml