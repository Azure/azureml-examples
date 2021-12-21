function Wait-SuccessfulRegistration {
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

function Deploy-RGIfInexistent {
    Param (
        $RGName,
        $RGLocation,
        $Purpose
    )
    Write-Output "Name of the $Purpose resource group to create: $RGName, in $RGLocation location."
    if ( $(az group exists --name $RGName) -eq $true ){
        Write-Output "The resource group '$RGName' already exists."
    } else {
        Write-Error "Creating the resource group..."
        az deployment sub create --location $RGLocation --template-file $PSScriptRoot/../bicep/ResourceGroup.bicep --parameters rgname=$RGName rglocation=$RGLocation
    }
}

function Confirm-ComputeName {
    Param(
        $ComputeName
    )
    Write-Output "Validating requested compute name..."
    $RegEx = '^[a-z0-9-]{2,16}$'
    if ($ComputeName -match $RegEx){
        Write-Output "Compute name $ComputeName is valid."
    } else{
        Write-Output "Compute name $ComputeName is invalid. It can include letters, digits and dashes. It must start with a letter, end with a letter or digit, and be between 2 and 16 characters in length."
        exit
    }
}