using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.Resources;
using System;
using System.Threading.Tasks;
using ManagedServiceIdentity = Azure.ResourceManager.Models.ManagedServiceIdentity;

namespace Azure.MachineLearning.Samples.Resources.Workspace
{
    internal class WorkspaceOperations
    {

        /// <summary>
        /// If the specified Workspace exists, get that Workspace.
        /// If it does not exist, creates a new Workspace. 
        /// </summary>
        /// <param name="resourceGroup">The resource group within the Azure subscription.</param>
        /// <param name="workspaceName">The name of the Workspace.</param>
        /// <param name="location">Location of the resource</param>
        /// <returns></returns>
        // <GetOrCreateWorkspaceAsync>
        public static async Task<MachineLearningWorkspaceResource> GetOrCreateWorkspaceAsync(
            ResourceGroupResource resourceGroup,
            string workspaceName,
            string location,
            string applicationInsightsId,
            string storageAccountId,
            string keyVaultId)
        {
            Console.WriteLine("Creating a Workspace...");
            MachineLearningWorkspaceResource workspace;
            if (await resourceGroup.GetMachineLearningWorkspaces().ExistsAsync(workspaceName)) // if (workspace != null)
            {
                Console.WriteLine($"workspace {workspaceName} already exists.");
                // We can get the workspace now that we know it exists.
                workspace = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
                Console.WriteLine($"Workspace details: {workspace.Data.Id}");
            }
            else
            {
                Console.WriteLine($"workspace {workspaceName} does not exist.");
                // Create Workspace data
                MachineLearningWorkspaceData data = new MachineLearningWorkspaceData()
                {
                    Location = location,
                    ApplicationInsights = applicationInsightsId,
                    ContainerRegistry = null,
                    StorageAccount = storageAccountId,
                    KeyVault = keyVaultId,
                    Identity = new ManagedServiceIdentity(ResourceManager.Models.ManagedServiceIdentityType.SystemAssigned),
                };
                data.Tags.Add("for", "test");
                ArmOperation<MachineLearningWorkspaceResource> lro = await resourceGroup.GetMachineLearningWorkspaces().CreateOrUpdateAsync(WaitUntil.Completed, workspaceName, data);
                workspace = lro.Value;
                Console.WriteLine($"Workspace {workspace} created.");
            }

            return workspace;
        }
        // </GetOrCreateWorkspaceAsync>


        /// <summary>
        /// List all the workspaces in the resource group
        /// </summary>
        /// <param name="resourceGroup"></param>
        /// <returns></returns>
        public static async Task ListAllWorkspaceAsync(
            ResourceGroupResource resourceGroup)
        {
            Console.WriteLine("Listing Workspaces in the Resource group...");
            MachineLearningWorkspaceCollection machineLearningWorkspaceCollection = resourceGroup.GetMachineLearningWorkspaces();

            AsyncPageable<MachineLearningWorkspaceResource> response = machineLearningWorkspaceCollection.GetAllAsync();
            await foreach (MachineLearningWorkspaceResource workspace in response)
            {
                Console.WriteLine(workspace.Data.Name);
            }

        }
        // </ListAllWorkspaceAsync>
    }
}
