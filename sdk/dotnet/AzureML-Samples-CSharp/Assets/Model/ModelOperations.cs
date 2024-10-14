using Azure.Core;
using Azure.Identity;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;
using Azure.Storage.Blobs;

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

    /// <summary>
    /// Download the model artifact from the workspace datastore.
    /// </summary>
    // <DownloadLatestModelVersion>
    public static async Task DownloadModelVersion(
        string subscriptionId,
        string resourceGroupName,
        string workspaceName,
        string modelName,
        string version,
        string downloadToPath)
    {
        var cred = new DefaultAzureCredential();
        var armClient = new ArmClient(cred);

        Console.WriteLine("Getting model version data ...");
        var modelId = MachineLearningModelVersionResource.CreateResourceIdentifier(subscriptionId, resourceGroupName, workspaceName, modelName, version);
        var modelResult = await armClient.GetMachineLearningModelVersionResource(modelId).GetAsync();
        var modelData = modelResult.Value.Data;
        Console.WriteLine($"Succeeded on id: {modelData.Id}");

        Console.WriteLine("Getting workspace datastore ...");
        var datastoreName = "workspaceblobstore";
        var datastoreId = MachineLearningDatastoreResource.CreateResourceIdentifier(subscriptionId, resourceGroupName, workspaceName, datastoreName);
        var datastoreResult = await armClient.GetMachineLearningDatastoreResource(datastoreId).GetAsync();
        var datastoreData = datastoreResult.Value.Data;
        Console.WriteLine($"Succeeded on id: {datastoreData.Id}");
        Console.WriteLine(datastoreData);

        var blobName = modelData.Properties.ModelUri.AbsolutePath.Split("/paths/").Last();
        Console.WriteLine($"Model blob name: {blobName}");

        var datastoreProperties = (MachineLearningAzureBlobDatastore)datastoreData.Properties;
        var storageEndpoint = $"https://{datastoreProperties.AccountName}.blob.core.windows.net/{datastoreProperties.ContainerName}";
        Console.WriteLine($"Storage endpoint: {storageEndpoint}");

        var modelUri = new Uri($"{storageEndpoint}/{blobName}");
        Console.WriteLine($"Downloading model from {modelUri} ...");

        var blobClient = new BlobClient(modelUri, cred);
        blobClient.DownloadTo(downloadToPath);
        Console.WriteLine($"Succeded on downloading model to {downloadToPath}");
    }
    // </DownloadLatestModelVersion>
}