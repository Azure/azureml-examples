####################################################################################################
# All the post cleanup scripts are called from this file                                           #
####################################################################################################

&"$PSScriptroot\remove_role_assignments.ps1" -ResourceGroupName "$env:RESOURCE_GROUP_NAME"
