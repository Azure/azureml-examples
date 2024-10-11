using Azure.Core;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;

namespace Azure.MachineLearning.Samples.Assets.Data;

internal class DataOperations
{

    /// <summary>
    /// If the specified Data exists, get that Data.
    /// If it does not exist, creates a new Data. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="dataName"></param>
    /// <param name="version"></param>
    /// <returns></returns>
    // <GetOrCreateDataVersionAsync>
    public static async Task<MachineLearningDataVersionResource> GetOrCreateDataVersionAsync(
        ArmClient armClient,
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string dataName,
        string version)
    {
        Console.WriteLine("Creating a DataVersion Resource...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        string resourceId = $"{ws.Id}/data/{dataName}";
        var id = new ResourceIdentifier(resourceId);
        MachineLearningDataContainerResource dataContainerResource = armClient.GetMachineLearningDataContainerResource(id);

        bool exists = await dataContainerResource.GetMachineLearningDataVersions().ExistsAsync(version);


        MachineLearningDataVersionResource dataVersionBaseResource;
        if (exists)
        {
            Console.WriteLine($"DataVersionBaseResource {dataName} exists.");
            dataVersionBaseResource = await dataContainerResource.GetMachineLearningDataVersions().GetAsync(version);
            Console.WriteLine($"DataVersionBaseResource details: {dataVersionBaseResource.Data.Id}");
        }
        else
        {

            Console.WriteLine($"Creating DataVersionBaseResource {dataName}");
            // UriFolderDataVersion, or UriFileDataVersion or MLTableData
            MachineLearningDataVersionProperties properties = new MachineLearningUriFileDataVersion(new Uri("https://pipelinedata.blob.core.windows.net/sampledata/nytaxi/"))
            {
                Description = "Test description",
                Tags = new Dictionary<string, string> { { "tag-name-1", "tag-value-1" } },
                Properties = new Dictionary<string, string> { { "property-name-1", "property-value-1" } },
                IsAnonymous = false,
                IsArchived = false,
            };

            MachineLearningDataVersionData data = new MachineLearningDataVersionData(properties);

            ArmOperation<MachineLearningDataVersionResource> dataVersionBaseResourceOperation = await dataContainerResource.GetMachineLearningDataVersions().CreateOrUpdateAsync(WaitUntil.Completed, version, data);
            dataVersionBaseResource = dataVersionBaseResourceOperation.Value;
            Console.WriteLine($"DataVersionBaseResource {dataVersionBaseResource.Data.Id} created.");
        }
        return dataVersionBaseResource;
    }

    /// <summary>
    /// List all Data in the workspace
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <returns></returns>
    public static async Task ListDataAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName)
    {
        Console.WriteLine("Listing Datasets in the workspace...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        MachineLearningDataContainerCollection dataContainerCollection = ws.GetMachineLearningDataContainers();
        AsyncPageable<MachineLearningDataContainerResource> response = dataContainerCollection.GetAllAsync();
        await foreach (MachineLearningDataContainerResource dataContainerResource in response)
        {
            Console.WriteLine(dataContainerResource.Data.Name);
        }
    }
    // </ListDataAsync>


    /// <summary>
    /// List all Datastore in the workspace
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <returns></returns>
    public static async Task ListDatastoreAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName)
    {
        Console.WriteLine("Listing Datastore in the workspace...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        MachineLearningDatastoreCollection datastoreCollection = ws.GetMachineLearningDatastores();
        AsyncPageable<MachineLearningDatastoreResource> response = datastoreCollection.GetAllAsync();
        await foreach (MachineLearningDatastoreResource datastoreResource in response)
        {
            MachineLearningDatastoreProperties properties = datastoreResource.Data.Properties;
            switch (properties)
            {
                case MachineLearningAzureFileDatastore:
                    MachineLearningAzureFileDatastore azureFileDatastore = (MachineLearningAzureFileDatastore)datastoreResource.Data.Properties;
                    Console.WriteLine($"AccountName {azureFileDatastore.AccountName}");
                    Console.WriteLine($"FileShareName {azureFileDatastore.FileShareName}");
                    Console.WriteLine($"Endpoint {azureFileDatastore.Endpoint}");
                    break;

                case MachineLearningAzureBlobDatastore:
                    MachineLearningAzureBlobDatastore azureBlobDatastore = (MachineLearningAzureBlobDatastore)datastoreResource.Data.Properties;
                    Console.WriteLine($"AccountName {azureBlobDatastore.AccountName}");
                    Console.WriteLine($"ContainerName {azureBlobDatastore.ContainerName}");
                    Console.WriteLine($"Endpoint {azureBlobDatastore.Endpoint}");
                    break;

                case MachineLearningAzureDataLakeGen1Datastore:
                    MachineLearningAzureDataLakeGen1Datastore azureDataLakeGen1Datastore = (MachineLearningAzureDataLakeGen1Datastore)datastoreResource.Data.Properties;
                    Console.WriteLine($"StoreName {azureDataLakeGen1Datastore.StoreName}");
                    break;

                case MachineLearningAzureDataLakeGen2Datastore:
                    MachineLearningAzureDataLakeGen2Datastore azureDataLakeGen2Datastore = (MachineLearningAzureDataLakeGen2Datastore)datastoreResource.Data.Properties;
                    Console.WriteLine($"AccountName {azureDataLakeGen2Datastore.AccountName}");
                    Console.WriteLine($"Filesystem {azureDataLakeGen2Datastore.Filesystem}");
                    Console.WriteLine($"Endpoint {azureDataLakeGen2Datastore.Endpoint}");
                    break;

                default:
                    Console.WriteLine("Unknown datastoreResource");
                    break;
            }
        }
    }
    // </ListDatastoreAsync>
}
