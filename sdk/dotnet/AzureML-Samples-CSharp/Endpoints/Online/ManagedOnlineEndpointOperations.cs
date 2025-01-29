using Azure.Core;
using Azure.MachineLearning.Samples.Shared;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;
using Newtonsoft.Json.Linq;
using ManagedServiceIdentity = Azure.ResourceManager.Models.ManagedServiceIdentity;

namespace Azure.MachineLearning.Samples.Endpoints.Online;

internal class ManagedOnlineEndpointOperations
{
    /// <summary>
    /// If the specified OnlineEndpoint exists, get that OnlineEndpoint.
    /// If it does not exist, creates a new OnlineEndpoint. 
    /// </summary>
    /// <param name="resourceGroup">The name of the resource group within the Azure subscription</param>
    /// <param name="workspaceName">The name of the Workspace</param>
    /// <param name="endpointName">The name of the OnlineEndpoint</param>
    /// <param name="location"></param>
    /// <returns></returns>
    // <GetOrCreateOnlineEndpointAsync>
    public static async Task<MachineLearningOnlineEndpointResource> GetOrCreateOnlineEndpointAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string endpointName,
        string location)
    {
        Console.WriteLine("Creating an OnlineEndpoint...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        bool exists = await ws.GetMachineLearningOnlineEndpoints().ExistsAsync(endpointName);

        MachineLearningOnlineEndpointResource endpointResource;
        if (exists)
        {
            Console.WriteLine($"OnlineEndpoint {endpointName} exists.");
            endpointResource = await ws.GetMachineLearningOnlineEndpoints().GetAsync(endpointName);
            Console.WriteLine($"OnlineEndpointResource details: {endpointResource.Data.Id}");
        }
        else
        {
            Console.WriteLine($"OnlineEndpoint {endpointName} does not exist.");
            MachineLearningOnlineEndpointProperties properties = new MachineLearningOnlineEndpointProperties(MachineLearningEndpointAuthMode.AmlToken)
            {
                //ARM resource ID of the compute if it exists.
                //Compute = "",
                Description = "test online endpoint",
                Properties = { { "additionalProp1", "value1" } },
                // Percentage of traffic from endpoint to divert to each deployment.Traffic values need to sum to 100.
                Traffic = { { "deployment1", 100 } },
                // PublicNetworkAccess = PublicNetworkAccessType.Enabled,
                // Percentage of traffic to be mirrored to each deployment without using returned scoring. Traffic values need to sum to utmost 50.
                MirrorTraffic = { { "deployment1", 30 } },
            };

            // ManagedServiceIdentity Identity = new ManagedServiceIdentity(ManagedServiceIdentityType.SystemAssigned);
            MachineLearningOnlineEndpointData OnlineEndpointData = new MachineLearningOnlineEndpointData(location, properties)
            {
                Kind = "SampleKind",
                // Identity = ManagedServiceIdentity(Azure.ResourceManager.MachineLearningServices.Models.ManagedServiceIdentityType.SystemAssigned),
                // Identity = new ManagedServiceIdentity(ManagedServiceIdentityType.SystemAssigned),
                Sku = new MachineLearningSku("Default")
                {
                    Tier = MachineLearningSkuTier.Standard,
                    Capacity = 2,
                    Family = "familyA",
                    Size = "Standard_F2s_v2",
                },
                Identity = new ManagedServiceIdentity(ResourceManager.Models.ManagedServiceIdentityType.SystemAssigned),
            };
            // new OnlineEndpointTrackedResourceData(Location.WestUS2, properties) { Kind = "SampleKind", Identity = identity };

            ArmOperation<MachineLearningOnlineEndpointResource> endpointResourceOperation = await ws.GetMachineLearningOnlineEndpoints().CreateOrUpdateAsync(WaitUntil.Completed, endpointName, OnlineEndpointData);
            endpointResource = endpointResourceOperation.Value;
            Console.WriteLine($"OnlineEndpointResource {endpointResource.Data.Id} created.");
        }

        return endpointResource;
    }
    // </GetOrCreateOnlineEndpointAsync>


    /// <summary>
    /// If the specified OnlineDeployment exists, get that OnlineDeployment.
    /// If it does not exist, creates a new BatchDeployment. 
    /// </summary>
    /// <param name="resourceGroup">The name of the resource group within the Azure subscription.</param>
    /// <param name="workspaceName">The name of the Workspace.</param>
    /// <param name="endpointName">The name of the Online Endpoint.</param>
    /// <param name="deploymentName">The name of the deployment.</param>
    /// <param name="modelId"></param>
    /// <param name="environmentId"></param>
    /// <param name="codeArtifactId"></param>
    /// <param name="location"></param>
    /// <returns></returns>
    // <GetOrCreateOnlineDeploymentAsync>
    public static async Task<MachineLearningOnlineDeploymentResource> GetOrCreateOnlineDeploymentAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string endpointName,
        string deploymentName,
        string modelId,
        string environmentId,
        string codeArtifactId,
        string location)
    {
        Console.WriteLine("Creating a OnlineDeployment...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        MachineLearningOnlineEndpointResource endpointResource = await ws.GetMachineLearningOnlineEndpoints().GetAsync(endpointName);
        Console.WriteLine(endpointResource.Data.Id);

        bool exists = await endpointResource.GetMachineLearningOnlineDeployments().ExistsAsync(deploymentName);


        // https://docs.microsoft.com/azure/machine-learning/how-to-troubleshoot-online-endpoints

        MachineLearningOnlineDeploymentResource deploymentResource;
        if (exists)
        {
            Console.WriteLine($"OnlineDeployment {deploymentName} exists.");
            deploymentResource = await endpointResource.GetMachineLearningOnlineDeployments().GetAsync(deploymentName);
            Console.WriteLine($"OnlineDeploymentResource details: {deploymentResource.Data.Id}");
        }
        else
        {
            Console.WriteLine($"OnlineDeployment {deploymentName} does not exist.");

            // scaleSettings for Kubernetes
            //var scaleSettings = new TargetUtilizationScaleSettings
            //{
            //    PollingInterval = TimeSpan.FromSeconds(1),
            //    TargetUtilizationPercentage = 50,
            //    MinInstances = 1,
            //    MaxInstances = 1,
            //};

            //scaleSettings = new DefaultScaleSettings();

            var managedOnlineDeploymentDetails = new MachineLearningManagedOnlineDeployment
            {
                Description = "This is a test online deployment",
                // EgressPublicNetworkAccess=EgressPublicNetworkAccessType.Disabled,
                // The path to mount the model in custom container.
                // Custom model mount path for curated environments is not supported
                // ModelMountPath = "/var/mountpath",
                EgressPublicNetworkAccess = MachineLearningEgressPublicNetworkAccessType.Disabled,
                Properties = { { "additionalProp1", "value1" } },
                EnvironmentId = environmentId,
                EnvironmentVariables = new Dictionary<string, string>
                {
                    { "TestVariable", "TestValue" }
                },
                RequestSettings = new MachineLearningOnlineRequestSettings
                {
                    MaxQueueWait = TimeSpan.FromMilliseconds(30),
                    RequestTimeout = TimeSpan.FromMilliseconds(60),
                    MaxConcurrentRequestsPerInstance = 3,
                },
                LivenessProbe = new MachineLearningProbeSettings
                {
                    FailureThreshold = 10,
                    SuccessThreshold = 1,
                    InitialDelay = TimeSpan.FromSeconds(10),
                    Timeout = TimeSpan.FromSeconds(10),
                    Period = TimeSpan.FromSeconds(2),
                },
                // Only for ManagedOnlineDeployment
                ReadinessProbe = new MachineLearningProbeSettings
                {
                    FailureThreshold = 10,
                    SuccessThreshold = 1,
                    InitialDelay = TimeSpan.FromSeconds(10),
                    Timeout = TimeSpan.FromSeconds(10),
                    Period = TimeSpan.FromSeconds(2),
                },
                AppInsightsEnabled = false,
                CodeConfiguration = new MachineLearningCodeConfiguration("main.py")
                {
                    CodeId = new ResourceIdentifier(codeArtifactId),
                },
                InstanceType = "Standard_F2s_v2",
                Model = modelId,
                // ScaleSettings = new DefaultScaleSettings(),
            };

            MachineLearningOnlineDeploymentData data = new MachineLearningOnlineDeploymentData(location, managedOnlineDeploymentDetails)
            {
                Kind = "SampleKindDeployment",
                Sku = new MachineLearningSku("Default")
                {
                    Tier = MachineLearningSkuTier.Standard,
                    Capacity = 2,
                    Family = "familyA",
                    Size = "Standard_F2s_v2",
                },
            };

            ArmOperation<MachineLearningOnlineDeploymentResource> deploymentResourceOperation = await endpointResource.GetMachineLearningOnlineDeployments().CreateOrUpdateAsync(WaitUntil.Completed, deploymentName, data);
            deploymentResource = deploymentResourceOperation.Value;
            Console.WriteLine($"OnlineDeploymentResource {deploymentResource.Data.Id} created.");
        }

        return deploymentResource;
    }
    // </GetOrCreateOnlineDeploymentAsync>


    /// <summary>
    /// List all Online deployments in the workspace
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <returns></returns>
    public static async Task ListOnlineDeploymentsAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName)
    {
        Console.WriteLine("Listing all Online deployments in the workspace...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        foreach (var edp in ws.GetMachineLearningOnlineEndpoints().GetAll())
        {
            foreach (var dep in edp.GetMachineLearningOnlineDeployments().GetAll())
            {
                Console.WriteLine(dep.Data.Name);
            }
        }
    }
    // </ListOnlineDeploymentsAsync>


    /// <summary>
    /// Invoke ScoringUri for OnlineEndpoint
    /// </summary>
    /// <param name="resourceGroup">The name of the resource group within the Azure subscription.</param>
    /// <param name="workspaceName">The name of the Workspace.</param>
    /// <param name="endpointName">The name of the Onlinendpoint.</param>
    /// <returns></returns>
    // <InvokeOnlineEndpoint>
    private static async Task InvokeOnlineEndpoint(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string endpointName)
    {
        Console.WriteLine("Invoking an OnlineEndpoint...");

        // var uriString = "https://mysdkonlineend001.westus2.inference.ml.azure.com/score";
        JObject jsonObject = JObject.Parse(@"{
'data': [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
]
}");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        bool exists = await ws.GetMachineLearningOnlineEndpoints().ExistsAsync(endpointName);

        if (exists)
        {
            Console.WriteLine($"OnlineEndpoint {endpointName} exists.");
            MachineLearningOnlineEndpointResource endpointResource = await ws.GetMachineLearningOnlineEndpoints().GetAsync(endpointName);
            Console.WriteLine($"OnlineEndpointResource details: {endpointResource.Data.Id}");

            var scoringUri = endpointResource.Data.Properties.ScoringUri;
            Console.WriteLine($"Using ScoringUri: {scoringUri} to invoke call for OnlineEndpoint");
            await Common.InvokeRequestResponseService(scoringUri, jsonObject);
        }
        else
        {
            Console.WriteLine($"OnlineEndpoint {endpointName} does not exist.");
        }
    }
    // </InvokeOnlineEndpoint>

}
