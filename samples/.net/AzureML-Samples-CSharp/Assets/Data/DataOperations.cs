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
    public static async Task<DataVersionBaseResource> GetOrCreateDataVersionAsync(
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
        DataContainerResource dataContainerResource = armClient.GetDataContainerResource(id);

        bool exists = await dataContainerResource.GetDataVersionBases().ExistsAsync(version);


        DataVersionBaseResource dataVersionBaseResource;
        if (exists)
        {
            Console.WriteLine($"DataVersionBaseResource {dataName} exists.");
            dataVersionBaseResource = await dataContainerResource.GetDataVersionBases().GetAsync(version);
            Console.WriteLine($"DataVersionBaseResource details: {dataVersionBaseResource.Data.Id}");
        }
        else
        {

            Console.WriteLine($"Creating DataVersionBaseResource {dataName}");
            // UriFolderDataVersion, or UriFileDataVersion or MLTableData
            DataVersionBaseProperties properties = new UriFileDataVersion(new Uri("https://pipelinedata.blob.core.windows.net/sampledata/nytaxi/"))
            {
                Description = "Test description",
                Tags = new Dictionary<string, string> { { "tag-name-1", "tag-value-1" } },
                Properties = new Dictionary<string, string> { { "property-name-1", "property-value-1" } },
                IsAnonymous = false,
                IsArchived = false,
            };

            DataVersionBaseData data = new DataVersionBaseData(properties);

            ArmOperation<DataVersionBaseResource> dataVersionBaseResourceOperation = await dataContainerResource.GetDataVersionBases().CreateOrUpdateAsync(WaitUntil.Completed, version, data);
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
        DataContainerCollection dataContainerCollection = ws.GetDataContainers();
        AsyncPageable<DataContainerResource> response = dataContainerCollection.GetAllAsync();
        await foreach (DataContainerResource dataContainerResource in response)
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
        DatastoreCollection datastoreCollection = ws.GetDatastores();
        AsyncPageable<DatastoreResource> response = datastoreCollection.GetAllAsync();
        await foreach (DatastoreResource datastoreResource in response)
        {
            DatastoreProperties properties = datastoreResource.Data.Properties;
            switch (properties)
            {
                case AzureFileDatastore:
                    AzureFileDatastore azureFileDatastore = (AzureFileDatastore)datastoreResource.Data.Properties;
                    Console.WriteLine($"AccountName {azureFileDatastore.AccountName}");
                    Console.WriteLine($"FileShareName {azureFileDatastore.FileShareName}");
                    Console.WriteLine($"Endpoint {azureFileDatastore.Endpoint}");
                    break;

                case AzureBlobDatastore:
                    AzureBlobDatastore azureBlobDatastore = (AzureBlobDatastore)datastoreResource.Data.Properties;
                    Console.WriteLine($"AccountName {azureBlobDatastore.AccountName}");
                    Console.WriteLine($"ContainerName {azureBlobDatastore.ContainerName}");
                    Console.WriteLine($"Endpoint {azureBlobDatastore.Endpoint}");
                    break;

                case AzureDataLakeGen1Datastore:
                    AzureDataLakeGen1Datastore azureDataLakeGen1Datastore = (AzureDataLakeGen1Datastore)datastoreResource.Data.Properties;
                    Console.WriteLine($"StoreName {azureDataLakeGen1Datastore.StoreName}");
                    break;

                case AzureDataLakeGen2Datastore:
                    AzureDataLakeGen2Datastore azureDataLakeGen2Datastore = (AzureDataLakeGen2Datastore)datastoreResource.Data.Properties;
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
