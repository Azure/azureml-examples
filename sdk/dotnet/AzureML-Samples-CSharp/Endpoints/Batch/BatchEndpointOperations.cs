using Azure.Core;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;
using ManagedServiceIdentity = Azure.ResourceManager.Models.ManagedServiceIdentity;

namespace Azure.MachineLearning.Samples.Endpoints.Batch;

internal class BatchEndpointOperations
{

    /// <summary>
    /// If the specified BatchEndpoint exists, get that BatchEndpoint.
    /// If it does not exist, creates a new BatchEndpoint. 
    /// </summary>
    /// <param name="resourceGroup">The name of the resource group within the Azure subscription.</param>
    /// <param name="workspaceName">The name of the Workspace.</param>
    /// <param name="endpointName">The name of the BatchEndpoint.</param>
    /// <param name="location">Location.</param>
    /// <returns></returns>
    // <GetOrCreateBatchEndpointAsync>
    public static async Task<MachineLearningBatchEndpointResource> GetOrCreateBatchEndpointAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string endpointName,
        string location)
    {
        Console.WriteLine("Creating a BatchEndpoint...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        bool exists = await ws.GetMachineLearningBatchEndpoints().ExistsAsync(endpointName);

        MachineLearningBatchEndpointResource endpointResource;
        if (exists)
        {
            Console.WriteLine($"BatchEndpoint {endpointName} exists.");
            endpointResource = await ws.GetMachineLearningBatchEndpoints().GetAsync(endpointName);
            Console.WriteLine($"BatchEndpointResource details: {endpointResource.Data.Id}");
        }
        else
        {
            Console.WriteLine($"BatchEndpoint {endpointName} does not exist.");

            MachineLearningBatchEndpointProperties properties = new MachineLearningBatchEndpointProperties(MachineLearningEndpointAuthMode.AadToken)
            {
                Description = "test batch endpoint",
                Properties = { { "additionalProp1", "value1" } },
            };

            MachineLearningBatchEndpointData data = new MachineLearningBatchEndpointData(location, properties)
            {
                Kind = "BatchSample",
                Sku = new MachineLearningSku("Default")
                {
                    Tier = MachineLearningSkuTier.Standard,
                    Capacity = 2,
                    Family = "familyA",
                    Size = "Standard_F2s_v2",
                },
                Identity = new ManagedServiceIdentity(ResourceManager.Models.ManagedServiceIdentityType.SystemAssigned),
            };

            ArmOperation<MachineLearningBatchEndpointResource> endpointResourceOperation = await ws.GetMachineLearningBatchEndpoints().CreateOrUpdateAsync(WaitUntil.Completed, endpointName, data);
            endpointResource = endpointResourceOperation.Value;
            Console.WriteLine($"BatchEndpointResource {endpointResource.Data.Id} created.");
        }

        return endpointResource;
    }
    // </GetOrCreateBatchEndpointAsync>



    /// <summary>
    /// If the specified BatchEndpoint exists, get that BatchEndpoint.
    /// If it does not exist, creates a new BatchEndpoint.
    /// </summary>
    /// <param name="resourceGroup">The name of the resource group within the Azure subscription.</param>
    /// <param name="workspaceName">The name of the Workspace.</param>
    /// <param name="endpointName">The name of the BatchEndpoint.</param>
    /// <param name="deploymentName">The name of the deploymentName.</param>
    /// <param name="modelId"></param>
    /// <param name="environmentId"></param>
    /// <param name="codeArtifactId"></param>
    /// <param name="computeId"></param>
    /// <param name="location"></param>
    /// <returns></returns>
    // <GetOrCreateBatchDeploymentAsync>
    public static async Task<MachineLearningBatchDeploymentResource> GetOrCreateBatchDeploymentAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string endpointName,
        string deploymentName,
        string modelId,
        string environmentId,
        string codeArtifactId,
        string computeId,
        string location)
    {
        Console.WriteLine("Creating a BatchDeployment...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        MachineLearningBatchEndpointResource endpointResource = await ws.GetMachineLearningBatchEndpoints().GetAsync(endpointName);
        Console.WriteLine(endpointResource.Data.Id);

        bool exists = await endpointResource.GetMachineLearningBatchDeployments().ExistsAsync(deploymentName);


        MachineLearningBatchDeploymentResource deploymentResource;
        if (exists)
        {
            Console.WriteLine($"BatchDeployment {deploymentName} exists.");
            deploymentResource = await endpointResource.GetMachineLearningBatchDeployments().GetAsync(deploymentName);
            Console.WriteLine($"BatchDeploymentResource details: {deploymentResource.Data.Id}");
        }
        else
        {
            Console.WriteLine($"BatchDeployment {deploymentName} does not exist.");

            MachineLearningBatchDeploymentProperties properties = new MachineLearningBatchDeploymentProperties
            {
                Description = "This is a batch deployment",
                ErrorThreshold = 10,
                MaxConcurrencyPerInstance = 5,
                LoggingLevel = MachineLearningBatchLoggingLevel.Info,
                MiniBatchSize = 10,
                OutputFileName = "mypredictions.csv",
                OutputAction = MachineLearningBatchOutputAction.AppendRow,
                Properties = { { "additionalProp1", "value1" } },
                EnvironmentId = environmentId,
                Compute = computeId,
                Resources = new MachineLearningDeploymentResourceConfiguration { InstanceCount = 1, },
                EnvironmentVariables = new Dictionary<string, string>
                {
                    { "TestVariable", "TestValue" },
                },
                RetrySettings = new MachineLearningBatchRetrySettings
                {
                    MaxRetries = 4,
                    Timeout = new TimeSpan(0, 3, 0),
                },
                CodeConfiguration = new MachineLearningCodeConfiguration("main.py")
                {
                    CodeId = new ResourceIdentifier(codeArtifactId),
                },
                Model = new MachineLearningIdAssetReference(new ResourceIdentifier(modelId)),
            };

            MachineLearningBatchDeploymentData data = new MachineLearningBatchDeploymentData(location, properties)
            {
                Kind = "SampleBatchDeployment",
                Sku = new MachineLearningSku("Default")
                {
                    Tier = MachineLearningSkuTier.Standard,
                    Capacity = 2,
                    Family = "familyA",
                    Size = "Standard_F2s_v2",
                },
            };

            ArmOperation<MachineLearningBatchDeploymentResource> endpointResourceOperation = await endpointResource.GetMachineLearningBatchDeployments().CreateOrUpdateAsync(WaitUntil.Completed, deploymentName, data);
            deploymentResource = endpointResourceOperation.Value;
            Console.WriteLine($"BatchDeploymentResource {deploymentResource.Data.Id} created.");
        }

        return deploymentResource;
    }
    // </GetOrCreateBatchDeploymentAsync>


    /// <summary>
    /// List all Batch deployments in the workspace
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <returns></returns>
    public static async Task ListBatchDeploymentsAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName)
    {
        Console.WriteLine("Listing all Batch deployments in the workspace...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        foreach (var edp in ws.GetMachineLearningBatchEndpoints().GetAll())
        {
            foreach (var dep in edp.GetMachineLearningBatchDeployments().GetAll())
            {
                Console.WriteLine(dep.Data.Name);
            }
        }
    }
    // </ListBatchDeploymentsAsync>

}
