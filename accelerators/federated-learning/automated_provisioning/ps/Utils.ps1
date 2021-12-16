function Wait-For-Successful-Registration(){
    param (
        $ProviderName
    )
    Write-Output "Waiting for successful registration of the $ProviderName provider."
    $Provider = az provider show -n $ProviderName | ConvertFrom-Json
    while ($Provider.RegistrationState -ne "Registered"){
        Start-Sleep -Seconds 60
        $Provider = az provider show -n $ProviderName | ConvertFrom-Json
    }
    Write-Output "The $ProviderName provider has been successfuly registered."
}

function Deploy-RG-If-Not-Exists {
    Param (
        $RGName,
        $RGLocation,
        $Purpose
    )
    Write-Output "Name of the $Purpose resource group to create: $RGName, in $RGLocation location."
    if ( $(az group exists --name $RGName) -eq $true ){
        Write-Output "The resource group '$RGName' already exists."
    } else {
        Write-Output "Creating the resource group..."
        az deployment sub create --location $RGLocation --template-file $PSScriptRoot/../bicep/ResourceGroup.bicep --parameters rgname=$RGName rglocation=$RGLocation
    }
}