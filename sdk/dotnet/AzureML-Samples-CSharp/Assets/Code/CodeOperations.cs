using Azure.Core;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;

namespace Azure.MachineLearning.Samples.Assets.Code;

internal class CodeOperations
{

    /// <summary>
    /// If the specified CodeVersion exists, get that CodeVersion.
    /// If it does not exist, creates a new CodeVersion. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="codeName"></param>
    /// <param name="version"></param>
    /// <returns></returns>
    // <GetOrCreateCodeVersionAsync>
    public static async Task<MachineLearningCodeVersionResource> GetOrCreateCodeVersionAsync(
        ArmClient armClient,
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string codeName,
        string version,
        string codeUri)
    {
        Console.WriteLine("Creating a CodeVersionResource...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        string resourceId = $"{ws.Id}/codes/{codeName}";
        var id = new ResourceIdentifier(resourceId);
        MachineLearningCodeContainerResource codeContainerDataResource = armClient.GetMachineLearningCodeContainerResource(id);

        MachineLearningCodeVersionProperties properties = new MachineLearningCodeVersionProperties { CodeUri = new Uri(codeUri) };
        MachineLearningCodeVersionData data = new MachineLearningCodeVersionData(properties);

        ArmOperation<MachineLearningCodeVersionResource> CodeVersionResourceOperation = await codeContainerDataResource.GetMachineLearningCodeVersions().CreateOrUpdateAsync(WaitUntil.Completed, version, data);
        MachineLearningCodeVersionResource codeVersionResource = CodeVersionResourceOperation.Value;
        Console.WriteLine($"codeVersionResource {codeVersionResource.Data.Id} created.");

        return codeVersionResource;
    }
    // </GetOrCreateCodeVersionAsync>

}
