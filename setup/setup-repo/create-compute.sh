# <create_computes>

az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 8

az ml compute create -n gpu-cluster --type amlcompute --min-instances 0 --max-instances 4 --size Standard_NC12

az ml compute create -n cpu-cluster-hri --type amlcompute --min-instances 0 --max-instances 8 --identity-type UserAssigned --user-assigned-identities /subscriptions/72c03bf3-4e69-41af-9532-dfcdc3eefef4/resourcegroups/static-test-resources/providers/Microsoft.ManagedIdentity/userAssignedIdentities/aml-training-build-managed-identity
# </create_computes>



az ml compute update -n cpu-cluster --max-instances 200

az ml compute update -n gpu-cluster --max-instances 40

