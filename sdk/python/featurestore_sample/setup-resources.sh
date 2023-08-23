pip install --upgrade jupytext

# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION=$(az ml workspace show --query location -o tsv)
RESOURCE_GROUP=$(az group show --query name -o tsv)
AML_WORKSPACE_NAME=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
OUTPUT_COMMAND="print"
# </create_variables>

# <convert_notebook_to_py>
jupytext --to py "notebooks/sdk_only/1. Develop a feature set and register with managed feature store.ipynb"
jupytext --to py "notebooks/sdk_only/2. Enable materialization and backfill feature data.ipynb"
jupytext --to py "notebooks/sdk_only/3. Experiment and train models using features.ipynb"
jupytext --to py "notebooks/sdk_only/4. Enable recurrent materialization and run batch inference.ipynb"

#jupytext --to py "notebooks/sdk_and_cli/1. Develop a feature set and register with managed feature store.ipynb"
#jupytext --to py "notebooks/sdk_and_cli/2. Enable materialization and backfill feature data.ipynb"
#jupytext --to py "notebooks/sdk_and_cli/3. Experiment and train models using features.ipynb"
#jupytext --to py "notebooks/sdk_and_cli/4. Enable recurrent materialization and run batch inference.ipynb"
# <convert_notebook_to_py>

#<replace_template_values>
sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
    s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
    s/<AML_WORKSPACE_NAME>/$AML_WORKSPACE_NAME/g;
    s/display/$OUTPUT_COMMAND/g;" $1