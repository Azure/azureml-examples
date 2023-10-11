pip install --upgrade jupytext

# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
LOCATION=$(az ml workspace show --query location -o tsv)
RESOURCE_GROUP=$(az group show --query name -o tsv)
AML_WORKSPACE_NAME=$(az configure -l --query "[?name=='workspace'].value" -o tsv)
OUTPUT_COMMAND="print"
FEATURE_STORAGE_ACCOUNT_NAME=${RESOURCE_GROUP}fs
USER_ID="36b5b70a-a2b2-45e6-a496-df3c2ffde085"
RAND_NUM=$RANDOM
UAI_NAME=fstoreuai${RAND_NUM}
REDIS_NAME=${RESOURCE_GROUP}rds
VERSION=$(((RANDOM%1000)+1))
# </create_variables>

# <convert_notebook_to_py>
NOTEBOOK_1="notebooks/sdk_only/1. Develop a feature set and register with managed feature store"
NOTEBOOK_2="notebooks/sdk_only/2. Experiment and train models using features"
NOTEBOOK_3="notebooks/sdk_only/3. Enable recurrent materialization and run batch inference"
NOTEBOOK_4="notebooks/sdk_only/4. Enable online store and run online inference"
jupytext --to py "${NOTEBOOK_1}.ipynb"
jupytext --to py "${NOTEBOOK_2}.ipynb"
jupytext --to py "${NOTEBOOK_3}.ipynb"
jupytext --to py "${NOTEBOOK_4}.ipynb"
# <convert_notebook_to_py>

#<replace_template_values>
sed -i "s/<SUBSCRIPTION_ID>/$SUBSCRIPTION_ID/g;
    s/<RESOURCE_GROUP>/$RESOURCE_GROUP/g;
    s/<AML_WORKSPACE_NAME>/$AML_WORKSPACE_NAME/g;" $1

#<replace_template_values>
sed -i "s/display/$OUTPUT_COMMAND/g;s/.\/Users\/<your_user_alias>\/featurestore_sample/.\//g;
    s/<VERSION>/$VERSION/g;" "${NOTEBOOK_1}.py"
sed -i "s/display/$OUTPUT_COMMAND/g;s/.\/Users\/<your_user_alias>\/featurestore_sample/.\//g;
    s/<VERSION>/$VERSION/g;" "${NOTEBOOK_2}.py"
sed -i "s/display/$OUTPUT_COMMAND/g;s/.\/Users\/<your_user_alias>\/featurestore_sample/.\//g;
    s/<VERSION>/$VERSION/g;" "${NOTEBOOK_3}.py"
sed -i "s/display/$OUTPUT_COMMAND/g;s/.\/Users\/<your_user_alias>\/featurestore_sample/.\//g;
    s/<REDIS_NAME>/$REDIS_NAME/g;
    s/<VERSION>/$VERSION/g;" "${NOTEBOOK_4}.py"