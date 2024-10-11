using Azure.Core;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;

namespace Azure.MachineLearning.Samples.Assets.Model;

class ModelOperations
{

    /// <summary>
    /// If the specified ModelSpecification exists, get that ModelSpecification.
    /// If it does not exist, creates a new ModelSpecification. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="modelName"></param>
    /// <param name="version"></param>
    /// <returns></returns>
    // <GetOrCreateModelVersionAsync>
    public static async Task<MachineLearningModelVersionResource> GetOrCreateModelVersionAsync(
        ArmClient armClient,
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string modelName,
        string version,
        string modelUri)
    {
        Console.WriteLine("Creating a ModelVersion Versions...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        string resourceId = $"{ws.Id}/models/{modelName}";
        var id = new ResourceIdentifier(resourceId);
        MachineLearningModelContainerResource modelContainerResource = armClient.GetMachineLearningModelContainerResource(id);

        MachineLearningModelVersionProperties properties = new MachineLearningModelVersionProperties
        {
            JobName = "TestJob",
            Description = "Test Description for ModelContainer",
            Tags = new Dictionary<string, string> { { "tag-name-1", "tag-value-1" } },
            IsAnonymous = false,
            Properties = new Dictionary<string, string> { { "property-name-1", "property-value-1" } },
            Flavors = new Dictionary<string, MachineLearningFlavorData>() { { "python_function", new MachineLearningFlavorData { Data = new Dictionary<string, string>() { { "loader_module", "test" } } } } },
            IsArchived = false,
            ModelType = "CustomModel",
            ModelUri = new Uri(modelUri),
        };

        MachineLearningModelVersionData data = new MachineLearningModelVersionData(properties);

        ArmOperation<MachineLearningModelVersionResource> ModelVersionResourceOperation = await modelContainerResource.GetMachineLearningModelVersions().CreateOrUpdateAsync(WaitUntil.Completed, version, data);
        MachineLearningModelVersionResource modelVersionResource = ModelVersionResourceOperation.Value;
        Console.WriteLine($"ModelVersionResource {modelVersionResource.Data.Id} created.");

        return modelVersionResource;
    }
    // </GetOrCreateModelVersionAsync>

}