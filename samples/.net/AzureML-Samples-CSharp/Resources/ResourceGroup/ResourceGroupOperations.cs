using Azure;
using Azure.Core;
using Azure.ResourceManager;
using Azure.ResourceManager.Resources;
using System;
using System.Threading.Tasks;

namespace Azure.MachineLearning.Samples.Resources.ResourceGroup;

internal class ResourceGroupOperations
{

    /// <summary>
    /// Creates a new resource group with the specified name
    /// If one already exists then it gets updated
    /// </summary>
    /// <param name="armClient"></param>
    /// <param name="resourceGroupName"></param>
    /// <returns></returns>
    public static async Task<ResourceGroupResource> CreateResourceGroup(ArmClient armClient, string resourceGroupName)
    {
        ResourceGroupResource resourceGroup;
        Console.WriteLine("Creating a resource group...");

        // Now we get a ResourceGroup container for that subscription
        SubscriptionResource subscription = await armClient.GetDefaultSubscriptionAsync();
        // This is a scoped operations object, and any operations you perform will be done under that subscription. From this object, you have access to all children via collection objects. Or you can access individual children by ID.
        // With the collection, we can create a new resource group with an specific name
        ResourceGroupCollection resourceGroups = subscription.GetResourceGroups();

        if (await resourceGroups.ExistsAsync(resourceGroupName))
        {
            Console.WriteLine($"Resource Group {resourceGroupName} exists.");
            // We can get the resource group now that we know it exists.
            // This does introduce a small race condition where resource group could have been deleted between the check and the get.
            resourceGroup = await resourceGroups.GetAsync(resourceGroupName);
            resourceGroup = await resourceGroup.AddTagAsync("key1", "value1");
        }
        else
        {
            Console.WriteLine($"Resource Group {resourceGroupName} does not exist.");
            AzureLocation location = AzureLocation.WestUS2;
            // resourceGroup = await rgCollection.CreateOrUpdate(WaitUntil.Completed, resourceGroupName, new ResourceGroupData(location)).WaitForCompletionAsync();
            ArmOperation<ResourceGroupResource> lro = await resourceGroups.CreateOrUpdateAsync(WaitUntil.Completed, resourceGroupName, new ResourceGroupData(location));
            resourceGroup = lro.Value;
            Console.WriteLine($"Resource Group {resourceGroup.Data.Name} created.");
        }
        return resourceGroup;
    }


    /// <summary>
    /// Delete the resource group
    /// </summary>
    /// <param name="armClient"></param>
    /// <param name="resourceGroupName"></param>
    /// <returns></returns>
    public static async Task DeleteResourceGroup(ArmClient armClient, string resourceGroupName)
    {
        SubscriptionResource subscription = await armClient.GetDefaultSubscriptionAsync().ConfigureAwait(false);
        if (await subscription.GetResourceGroups().ExistsAsync(resourceGroupName))
        {
            Console.WriteLine($"Deleting the Resource Group {resourceGroupName}");
            ResourceGroupResource resourceGroup = await subscription.GetResourceGroups().GetAsync(resourceGroupName);
            await resourceGroup.DeleteAsync(WaitUntil.Completed);
        }
    }
}
