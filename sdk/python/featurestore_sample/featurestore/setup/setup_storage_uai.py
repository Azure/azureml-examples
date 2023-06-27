## This utility is created for ease of use in the docs tutorials. It might not follow all best practices. Do not use it for production use.

## This setup script will do:
##  - create a gen2 storage account and a container (or use existing one)
##  - create a user-assigned managed identity (or using existing one)
##  - grant RBAC permissions to the user-assigned managed identity

from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.storage.models import (
    StorageAccountCreateParameters,
    Sku,
    Kind,
    AccessTier,
)
from azure.storage.filedatalake import DataLakeServiceClient
from azure.mgmt.storage import StorageManagementClient
from azure.core.exceptions import ResourceExistsError
from azure.mgmt.msi import ManagedServiceIdentityClient
from azure.mgmt.msi.models import Identity
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.authorization.models import RoleAssignmentCreateParameters
from uuid import uuid4
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential


def create_gen2_storage_container(
    credential,
    storage_subscription_id,
    storage_resource_group_name,
    storage_account_name,
    storage_location,
    storage_file_system_name,
):
    print(
        "creating storage account {account} and container {container}".format(
            account=storage_account_name, container=storage_file_system_name
        )
    )
    # Create a storage management client
    storage_client = StorageManagementClient(credential, storage_subscription_id)

    # Set up the storage account creation parameters
    storage_account_params = StorageAccountCreateParameters(
        sku=Sku(name="Standard_LRS"),  # Use locally redundant storage
        kind=Kind.STORAGE_V2,  # Use Gen2 storage
        location=storage_location,
        access_tier=AccessTier.HOT,
        enable_https_traffic_only=True,
        is_hns_enabled=True,  # Enable the hierarchical namespace feature
    )

    try:
        result = storage_client.storage_accounts.begin_create(
            storage_resource_group_name, storage_account_name, storage_account_params
        )

        result.wait()
        print(f"storage '{storage_account_name}' created successfully.")
    except ResourceExistsError:
        print(f"storage '{storage_account_name}' already exists.")

    # Create a storage management client (if you haven't done it before)
    storage_client = StorageManagementClient(credential, storage_subscription_id)

    # Get the storage account properties
    storage_account_properties = storage_client.storage_accounts.get_properties(
        storage_resource_group_name, storage_account_name
    )

    # Get the primary dfs service endpoint
    primary_endpoint = storage_account_properties.primary_endpoints.dfs

    # Create a Data Lake service client
    data_lake_service_client = DataLakeServiceClient(
        account_url=primary_endpoint, credential=credential
    )

    # Create the file system
    try:
        file_system_client = data_lake_service_client.create_file_system(
            storage_file_system_name
        )
        print(f"File system '{storage_file_system_name}' created successfully.")
    except ResourceExistsError:
        print(f"File system '{storage_file_system_name}' already exists.")

    gen2_container_arm_id = "/subscriptions/{sub_id}/resourceGroups/{rg}/providers/Microsoft.Storage/storageAccounts/{account}/blobServices/default/containers/{container}".format(
        sub_id=storage_subscription_id,
        rg=storage_resource_group_name,
        account=storage_account_name,
        container=storage_file_system_name,
    )

    return gen2_container_arm_id


def create_user_assigned_managed_identity(
    credential, uai_subscription_id, uai_resource_group_name, uai_name, uai_location
):
    # provision UAI
    msi_client = ManagedServiceIdentityClient(credential, uai_subscription_id)
    print("creating new user assigned managed identity")

    # Set up the managed identity creation parameters
    managed_identity_params = Identity(location=uai_location)

    # Create the managed identity
    try:
        managed_identity = msi_client.user_assigned_identities.create_or_update(
            uai_resource_group_name, uai_name, managed_identity_params
        )
        print(f"managed identity '{uai_name}' created successfully.")
    except ResourceExistsError:
        print(f"managed identity '{uai_name}' already exists.")

    uai_principal_id = managed_identity.principal_id
    uai_client_id = managed_identity.client_id
    uai_arm_id = managed_identity.id

    return uai_principal_id, uai_client_id, uai_arm_id


def grant_rbac_permissions(
    credential,
    uai_principal_id,
    storage_subscription_id,
    storage_resource_group_name,
    storage_account_name,
    featurestore_subscription_id,
    featurestore_resource_group_name,
    featurestore_name,
):

    # Grant RBAC
    # Create an authorization management client
    auth_client = AuthorizationManagementClient(credential, storage_subscription_id)

    # Construct the scope, which is the storage account's resource ID
    scope = f"/subscriptions/{storage_subscription_id}/resourceGroups/{storage_resource_group_name}/providers/Microsoft.Storage/storageAccounts/{storage_account_name}"

    # The role definition ID for the "Storage Blob Data Contributor" role
    # You can find other built-in role definition IDs in the Azure documentation
    role_definition_id = f"/subscriptions/{storage_subscription_id}/providers/Microsoft.Authorization/roleDefinitions/ba92f5b4-2d11-453d-a403-e96b0029c9fe"

    # Generate a random UUID for the role assignment name
    role_assignment_name = str(uuid4())

    # Set up the role assignment creation parameters
    role_assignment_params = RoleAssignmentCreateParameters(
        principal_id=uai_principal_id,
        role_definition_id=role_definition_id,
        principal_type="ServicePrincipal",
    )

    try:
        # Create the role assignment
        result = auth_client.role_assignments.create(
            scope, role_assignment_name, role_assignment_params
        )
        print(f"Storage RBAC granted to managed identity '{uai_principal_id}'.")
    except ResourceExistsError:
        print(f"Storage RBAC already exists for managed identity '{uai_principal_id}'.")

    auth_client = AuthorizationManagementClient(
        credential, featurestore_subscription_id
    )

    scope = f"/subscriptions/{featurestore_subscription_id}/resourceGroups/{featurestore_resource_group_name}/providers/Microsoft.MachineLearningServices/workspaces/{featurestore_name}"

    # The role definition ID for the "AzureML Data Scientist" role
    # You can find other built-in role definition IDs in the Azure documentation
    role_definition_id = f"/subscriptions/{featurestore_subscription_id}/providers/Microsoft.Authorization/roleDefinitions/f6c7c914-8db3-469d-8ca1-694a8f32e121"

    # Generate a random UUID for the role assignment name
    role_assignment_name = str(uuid4())

    # Set up the role assignment creation parameters
    role_assignment_params = RoleAssignmentCreateParameters(
        principal_id=uai_principal_id,
        role_definition_id=role_definition_id,
        principal_type="ServicePrincipal",
    )

    # Create the role assignment
    try:
        # Create the role assignment
        result = auth_client.role_assignments.create(
            scope, role_assignment_name, role_assignment_params
        )
        print(f"feature store RBAC granted to managed identity '{uai_principal_id}'.")
    except ResourceExistsError:
        print(
            f"feature store RBAC already exists for managed identity '{uai_principal_id}'."
        )


def grant_user_aad_storage_data_reader_role(
    credential,
    user_aad_objectId,
    storage_subscription_id,
    storage_resource_group_name,
    storage_account_name,
):
    from azure.mgmt.storage import StorageManagementClient
    from azure.mgmt.authorization import AuthorizationManagementClient
    from azure.mgmt.authorization.models import RoleAssignmentCreateParameters
    import uuid

    # Initialize the Storage Management and Authorization Management clients
    storage_client = StorageManagementClient(credential, storage_subscription_id)
    authorization_client = AuthorizationManagementClient(
        credential, storage_subscription_id
    )

    # Get the storage account
    storage_account = storage_client.storage_accounts.get_properties(
        storage_resource_group_name, storage_account_name
    )
    storage_account_id = storage_account.id

    # Get the "Blob Data Reader" role definition
    role_definitions = authorization_client.role_definitions.list(
        storage_account_id, filter="roleName eq 'Storage Blob Data Reader'"
    )
    role_definition = next(iter(role_definitions), None)

    if role_definition is None:
        print("Role definition not found")
    else:
        # Create a new role assignment
        role_assignment_guid = str(uuid.uuid4())
        role_assignment = authorization_client.role_assignments.create(
            storage_account_id,
            role_assignment_guid,
            RoleAssignmentCreateParameters(
                principal_id=user_aad_objectId, role_definition_id=role_definition.id
            ),
        )
        print("Role assignment created:", role_assignment.id)
