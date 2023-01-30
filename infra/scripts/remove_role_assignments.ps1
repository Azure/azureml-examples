####################################################################################################
# This script allows you to remove stale role assignments listed as 'Identity not found'           #
####################################################################################################
[CmdletBinding()]
param (
    [Parameter(Mandatory=$true)]
    [string]
    $ResourceScope
)
function Get-RoleAssignmentCount {
    Write-Output "Determing unknown Role Assignment count..."
    $stale = Get-AzRoleAssignment -Scope "$ResourceScope" | Where-Object { $_.ObjectType -eq 'Unknown'}
    $unknownRoleAssignmentCount = $stale.Count
    Write-Output "Total Unknown Role Assignment Count: $unknownRoleAssignmentCount..."

    return $unknownRoleAssignmentCount
}

try
{
    Write-Output "Pre-checking the RoleAssignment count..."
    Get-RoleAssignmentCount
    Get-AzRoleAssignment -Scope "$ResourceScope" | Where-Object { $_.ObjectType -eq 'Unknown'} | Remove-AzRoleAssignment
    Write-Output "Check the Role Assignment count after cleanup"
    Get-RoleAssignmentCount
    Write-Output "Role Assignment clean-up complete."
}
catch
{
    Write-Error "There was an issue in cleaning-up the Role Assignment. See details: $($_.Exception.Message)"
    Exit 1
}
