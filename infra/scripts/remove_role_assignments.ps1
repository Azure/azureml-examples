####################################################################################################
# This script allows you to remove stale role assignments listed as 'Identity not found'           #
####################################################################################################
[CmdletBinding()]
param (
    [Parameter(Mandatory=$true)]
    [ValidateNotNullOrEmpty()]
    [string]$ResourceGroupName
)
function Get-RoleAssignmentCount {
    $OBJTYPE = "Unknown"
    Write-Output "Determing unknown Role Assignment count in the ResourceGroup:$ResourceGroupName..."
    $stale = Get-AzRoleAssignment -ResourceGroupName "$ResourceGroupName" | Where-Object { $_.ObjectType -eq $OBJTYPE}
    $unknownRoleAssignmentCount = $stale.Count
    Write-Output "Total Unknown Role Assignment Count: $unknownRoleAssignmentCount in the ResourceGroup:$ResourceGroupName..."

    return $unknownRoleAssignmentCount
}

try
{
    $OBJTYPE = "Unknown"
    Write-Output "Pre-checking the RoleAssignment count..."
    Get-RoleAssignmentCount
    # Remove only limited RoleDefinitions
    $staleRoleAssignments = Get-AzRoleAssignment -ResourceGroupName "$ResourceGroupName" | Where-Object {($_.ObjectType -eq $OBJTYPE) -and ($_.RoleDefinitionName -match "Storage Blob Data Reader|AzureML Metrics Writer (preview)|AcrPull")}
    $unknownRoleAssignmentCount = $staleRoleAssignments.Count
    Write-Output "Initiating the cleanup of unknownRole in the ResourceGroup:$ResourceGroupName having count as $unknownRoleAssignmentCount..."
    $staleRoleAssignments | Remove-AzRoleAssignment
    Write-Output "Check the Role Assignment count after cleanup"
    Get-RoleAssignmentCount
    Write-Output "Role Assignment clean-up complete."
}
catch
{
    Write-Error "There was an issue in cleaning-up the Role Assignment. See details: $($_.Exception.Message)"
    Exit 1
}
