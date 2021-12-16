function Wait-For-Successful-Registration {
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

function Create-RG-If-Not-Exists{
    param (
        $rgname,
        $rglocation,
        $purpose
    )
    Write-Output "Name of the $purpose resource group to create: $rgname, in $rglocation location."
    if ( $(az group exists --name $rgname) ){
        Write-Output "The resource group '$rgname' already exists."
    } else {
        Write-Output "Creating the resource group..."
        az deployment sub create --location $rglocation --template-file $PSScriptRoot/../bicep/ResourceGroup.bicep --parameters rgname=$rgname rglocation=$rglocation
    }
}