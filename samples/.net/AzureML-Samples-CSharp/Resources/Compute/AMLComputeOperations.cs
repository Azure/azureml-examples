using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;

namespace Azure.MachineLearning.Samples.Resources.Compute;

internal class AMLComputeOperations
{

    /// <summary>
    /// If the specified MachineLearningComputeResource exists, get that MachineLearningComputeResource.
    /// If it does not exist, creates a new MachineLearningComputeResource. 
    /// </summary>
    /// <param name="resourceGroup">The name of the resource group within the Azure subscription.</param>
    /// <param name="workspaceName">The name of the Workspace.</param>
    /// <param name="computeName">The name of the AmlCompute.</param>
    /// <param name="location">Location of the resource</param>
    /// <returns></returns>
    // <GetOrCreateAMLComputeAsync>
    public static async Task<MachineLearningComputeResource> GetOrCreateAMLComputeAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string computeName,
        string location)
    {
        Console.WriteLine("Creating an amlCompute...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        MachineLearningComputeResource machineLearningComputeResource;
        // GetMachineLearningComputes

        if (await ws.GetMachineLearningComputes().ExistsAsync(computeName))
        {
            Console.WriteLine($"MachineLearningComputeResource {computeName} already exists.");
            machineLearningComputeResource = await ws.GetMachineLearningComputes().GetAsync(computeName);
            Console.WriteLine($"MachineLearningComputeResource details: {machineLearningComputeResource.Data.Id}");
        }
        else
        {
            Console.WriteLine($"MachineLearningComputeResource {computeName} does not exist.");
            // Create MachineLearningComputeData data
            MachineLearningComputeData data = new MachineLearningComputeData
            {
                Location = location,
                Properties = new AmlCompute
                {
                    Properties = new AmlComputeProperties
                    {
                        ScaleSettings = new ScaleSettings(4),
                        VmSize = "Standard_DS3_v2",
                        // VmSize = "STANDARD_NC6",
                    }
                },
            };

            data.Tags.Add("for", "test");

            ArmOperation<MachineLearningComputeResource> MachineLearningComputeResourceOperation = await ws.GetMachineLearningComputes().CreateOrUpdateAsync(WaitUntil.Completed, computeName, data);
            machineLearningComputeResource = MachineLearningComputeResourceOperation.Value;
            Console.WriteLine($"aml compute {machineLearningComputeResource} created.");
        }

        return machineLearningComputeResource;
    }
    // </GetOrCreateAMLComputeAsync>

    /// <summary>
    /// List AMLComputeProperties
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// /// <param name="computeName"></param>
    /// <returns></returns>
    public static async Task ListComputePropertiesAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string computeName)
    {
        Console.WriteLine("Listing AMLCompute in the workspace...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        MachineLearningComputeResource amlCompute = await ws.GetMachineLearningComputes().GetAsync(computeName);
        AmlCompute amlComputeProperties = (AmlCompute)amlCompute.Data.Properties;
        Console.WriteLine($"SKU from amlComputeProperties: {amlComputeProperties.Properties.VmSize}");
    }
    // </ListComputePropertiesAsync>
}
