using Azure;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;

namespace Azure.MachineLearning.Samples.Resources.Compute;

internal class AKSComputeOperations
{

    /// <summary>
    /// If the specified MachineLearningComputeResource exists, get that MachineLearningComputeResource.
    /// If it does not exist, creates a new MachineLearningComputeResource. 
    /// </summary>
    /// <param name="resourceGroup">The name of the resource group within the Azure subscription.</param>
    /// <param name="workspaceName">The name of the Workspace.</param>
    /// <param name="computeName">The name of the aks Compute.</param>
    /// <param name="location">Location of the resource</param>
    /// <returns></returns>
    // <GetOrCreateAKSComputeAsync>
    public static async Task<MachineLearningComputeResource> GetOrCreateAKSComputeAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string computeName,
        string location,
        string clusterFqdn)
    {
        Console.WriteLine("Creating an AKSCompute...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        MachineLearningComputeResource machineLearningComputeResource;
        if (await ws.GetMachineLearningComputes().ExistsAsync(computeName))
        {
            Console.WriteLine($"MachineLearningComputeResource {computeName} already exists.");
            machineLearningComputeResource = await ws.GetMachineLearningComputes().GetAsync(computeName);
            Console.WriteLine($"AKS MachineLearningComputeResource details: {machineLearningComputeResource.Data.Id}");
        }
        else
        {
            Console.WriteLine($"MachineLearningComputeResource {computeName} does not exist.");
            AksSchemaProperties aksProperties = new AksSchemaProperties
            {
                ClusterFqdn = clusterFqdn,
                ClusterPurpose = ClusterPurpose.DevTest,
                AgentCount = 2,
                AgentVmSize = "Standard_D3_v2",
                LoadBalancerType = LoadBalancerType.InternalLoadBalancer,
                //LoadBalancerSubnet = "default",
                //AksNetworkingConfiguration = new AksNetworkingConfiguration
                //{
                //    SubnetId = subnetId,
                //    ServiceCidr = "192.168.2.0/24",
                //    DnsServiceIP = "192.168.2.100",
                //    DockerBridgeCidr = "192.168.3.1/24",
                //},
                // Auto SSL certificate cannot be used with a private IP address. Use auto SSL certificate with a public IP address or provide your own SSL certfiicate with a private IP address.
                //SslConfiguration = new SslConfiguration
                //{
                //    Status = SslConfigurationStatus.Auto,
                //    LeafDomainLabel = "myakstst",
                //    OverwriteExistingDomain = true,
                //},
            };
            // Create MachineLearningComputeData data
            MachineLearningComputeData data = new MachineLearningComputeData
            {
                Location = location,
                Properties = new AksCompute
                {
                    Properties = aksProperties,
                    Description = "AKS test description",
                }
            };

            data.Tags.Add("for", "test");

            ArmOperation<MachineLearningComputeResource> machineLearningComputeResourceOperation = await ws.GetMachineLearningComputes().CreateOrUpdateAsync(WaitUntil.Completed, computeName, data);
            machineLearningComputeResource = machineLearningComputeResourceOperation.Value;
            Console.WriteLine($"aks compute {machineLearningComputeResource} created.");
        }

        return machineLearningComputeResource;
    }
    // </GetOrCreateAKSComputeAsync>

    /// <summary>
    /// List AksComputeProperties
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
        Console.WriteLine("Listing aksComputeProperties from the workspace...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        MachineLearningComputeResource aksCompute = await ws.GetMachineLearningComputes().GetAsync(computeName);
        AksCompute aksComputeProperties = (AksCompute)aksCompute.Data.Properties;
        Console.WriteLine($"AgentVmSize from aksComputeProperties: {aksComputeProperties.Properties.AgentVmSize}");
        Console.WriteLine($"AgentCount from aksComputeProperties: {aksComputeProperties.Properties.AgentCount}");
        Console.WriteLine($"ClusterFqdn from aksComputeProperties: {aksComputeProperties.Properties.ClusterFqdn}");
        Console.WriteLine($"ClusterPurpose from aksComputeProperties: {aksComputeProperties.Properties.ClusterPurpose}");
    }
    // </ListComputePropertiesAsync>
}
