####################################################################################################
# All the post cleanup scripts are called from this file                                           #
####################################################################################################

./remove_role_assignments.ps1 -ResourceScope "/subscriptions/$env:SUBSCRIPTION_ID/resourceGroups/$env:RESOURCE_GROUP_NAME"
