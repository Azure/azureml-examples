#/bin/bash

source ../set-environment-vars.sh

az ml data create -f data-asset.yml \
--workspace-name $aml_ws \
--resource-group $rg_name \
--subscription $subscription_id