from urllib.request import urlopen
import json

# url = 'pipelineComponent.schema.json'
#'#/definitions/PipelineComponentSchema'
validation_required = {
    "pipelineComponent": "PipelineComponentSchema",
    "workspace": "WorkspaceSchema",
    "environment": "EnvironmentSchema",
    "dataset": "DatasetSchema",
    "model": "ModelSchema",
}


def validation(entity_name: str, validation_value: str) -> None:
    response = urlopen(f"https://azuremlschemas.azureedge.net/latest/{entity_name}.schema.json")
    string = response.read().decode("utf-8")
    json_obj = json.loads(string)
    actual_val = json_obj["$ref"]
    if validation_value in actual_val:
        print(f"successfully validated schema : {entity_name}")
    else:
        print(f"unable to validate schema for : {entity_name}")
        raise Exception(f"unable to validate schema for : {entity_name}")


for validate in validation_required:
    validation(validate, validation_required[validate])
