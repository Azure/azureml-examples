####################################################################################################
# All the post cleanup scripts are called from this file                                           #
####################################################################################################

scripts/remove_role_assignments.ps1 "/subscriptions/$env:SUBSCRIPTION_ID/resourceGroups/$env:RESOURCE_GROUP_NAME"
