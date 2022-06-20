	Write-Host "#### Installing Chocolatey #########"
	$chocoExePath = 'C:\ProgramData\Chocolatey\bin'

    if ($($env:Path).ToLower().Contains($($chocoExePath).ToLower())) {
        Write-Log "Chocolatey found in PATH, skipping install..."
    } else {
        Write-Log "Installing Chocolatey..."
        # Add to system PATH
        $systemPath = [environment]::GetEnvironmentVariable('Path',[System.EnvironmentVariableTarget]::Machine)
        $systemPath += ';' + $chocoExePath
        [environment]::SetEnvironmentVariable("PATH",$systemPath,[System.EnvironmentVariableTarget]::Machine)

        # Update local process' path
        $userPath = [environment]::GetEnvironmentVariable('Path',[System.EnvironmentVariableTarget]::User)
        if ($userPath) {
            $env:Path = $systemPath + ";" + $userPath
        } else {
            $env:Path = $systemPath
        }

        # Run the installer
        Invoke-Expression ((New-Object net.webclient).DownloadString('https://chocolatey.org/install.ps1'))
    }
    choco feature enable --name allowGlobalConfirmation # stop the -y flag being needed for all "choco install"s
    choco feature disable --name checksumFiles # lots of packages have no checksums, e.g. WinSDK, so allow them
    choco install -y vcredist140
	Write-Host "#### Installing Rstudio #########"
	C:\ProgramData\chocolatey\choco install r.studio -y
	Write-Host "#### Completed Rstudio installtion with choco #########"
