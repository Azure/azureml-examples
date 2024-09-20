$validation_list = "pipelineComponent,workspace,environment,dataset,model"
$validation_entities = $validation_list.split(",");
foreach ($validation_entity in $validation_entities) {
    Write-Output $validation_entity
    $uri = "https://azuremlschemas.azureedge.net/latest/$validation_entity.schema.json"
    Write-Output "validating $uri"
    $response = Invoke-RestMethod -Uri $uri -Method Get
    if ([bool]($response.PSobject.Properties.name -match "definitions")){
        Write-Output "success $validation_entity"
    }
    else{
        Write-Output "fail $validation_entity"
        throw [System.IO.FileNotFoundException] "$validation_entity not found."
    }
}