using Azure.Core;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;
using Newtonsoft.Json.Linq;

namespace Azure.MachineLearning.Samples.Assets.Environment;

internal class EnvironmentOperations
{

    /// <summary>
    /// If the specified EnvironmentSpecification exists, get that EnvironmentSpecification.
    /// If it does not exist, creates a new EnvironmentSpecification. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="environmentName"></param>
    /// <param name="version"></param>
    /// <returns></returns>
    // <GetOrCreateEnvironmentSpecificationVersionAsync>
    public static async Task<EnvironmentVersionResource> GetOrCreateEnvironmentVersionAsync(
        ArmClient armClient,
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string environmentName,
        string version)
    {
        Console.WriteLine("Creating an EnvironmentVersion Resource...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        string resourceId = $"{ws.Id}/environments/{environmentName}";
        var id = new ResourceIdentifier(resourceId);
        EnvironmentContainerResource environmentContainerResource = armClient.GetEnvironmentContainerResource(id);

        var condaDependences = new JObject();
        condaDependences["channels"] = new JArray() { "conda-forge" };
        var dependencies = new JArray
        {
            "python=3.7.10",
            "numpy",
            "pip",
            "scikit-learn==0.19.1",
            "scipy",
            new JObject
            {
                ["pip"] = new JArray(new string[]
                {
                    "azureml-defaults",
                    "inference-schema[numpy-support]",
                    "joblib",
                    "numpy",
                    "scikit-learn==0.19.1",
                    "scipy"
                })
            }
        };

        condaDependences["dependencies"] = dependencies;
        Console.WriteLine($"condaDependences: {condaDependences}");

        EnvironmentVersionProperties properties = new EnvironmentVersionProperties
        {
            Description = "Test",
            CondaFile = condaDependences.ToString(),
            Tags = { { "key1", "value1" }, { "key2", "value2" } },
            OSType = OperatingSystemType.Linux,
            IsAnonymous = false,
            Image = "mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
        };

        EnvironmentVersionData data = new EnvironmentVersionData(properties);

        ArmOperation<EnvironmentVersionResource> environmentVersionResourceOperation = await environmentContainerResource.GetEnvironmentVersions().CreateOrUpdateAsync(WaitUntil.Completed, version, data);
        EnvironmentVersionResource environmentVersionResource = environmentVersionResourceOperation.Value;
        Console.WriteLine($"EnvironmentVersionResource {environmentVersionResource.Data.Id} created.");

        return environmentVersionResource;
    }
    // </GetOrCreateEnvironmentVersionAsync>

}
