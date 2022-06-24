Write-Host "#### Installing RStudio-pro-2022 #########"

Invoke-WebRequest -uri https://download1.rstudio.org/desktop/windows/RStudio-pro-2022.02.3-492.pro3.exe -OutFile RStudio-pro-2022.exe

Start-Process .\RStudio-pro-2022.exe -ArgumentList " /S" -Wait -NoNewWindow

Write-Host "#### Completed RStudio-pro-2022 Installtion #########"