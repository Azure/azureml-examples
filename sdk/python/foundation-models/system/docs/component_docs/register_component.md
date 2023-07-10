# Deploy Component
This component helps user to register the finetuned model.

The components can be seen in your workspace components page ![as shown in the figure](../images/register_component.png)

> **Note**: The compute on which register component is running, the Managed Identity must be set with Contributor access to the whole resource group as scope.

# Inputs

1. _model_checkpoint_dir_ (URI_FOLDER, required):

    Path to folder containing finetuned model. Can be pytorch/mlflow model.

2. _name_for_registered_model_ (string, required):

    This is the name with which model gets registered in workspace.

3. _model_type_ (string, required):

    Type of the model should be specified. Supporting custom_model and mlflow_model as types.

4. _registry_name_ (string, optional):

    Provide registry name if the model is to be registerd into a registry.


# Outputs

There is no output for this component. The model details are available in Models section of AzureML Studio.

# Examples

Sample experiment using the register component : https://ml.azure.com/experiments/id/cefb600a-f64a-4d12-85b8-53819f1f7eea/runs/shy_heart_yxj0p19bvw?wsid=/subscriptions/381b38e9-9840-4719-a5a0-61d9585e1e91/resourcegroups/sasik_rg/providers/Microsoft.MachineLearningServices/workspaces/aml-finetuning-ws&tid=72f988bf-86f1-41af-91ab-2d7cd011db47#

# Caveats, Recommendations and Known Issues
- Can add other types of model in future.
