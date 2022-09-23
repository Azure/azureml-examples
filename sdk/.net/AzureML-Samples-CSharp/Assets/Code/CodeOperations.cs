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
    public static async Task<CodeVersionResource> GetOrCreateCodeVersionAsync(
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
        CodeContainerResource codeContainerDataResource = armClient.GetCodeContainerResource(id);

        CodeVersionProperties properties = new CodeVersionProperties { CodeUri = new Uri(codeUri) };
        CodeVersionData data = new CodeVersionData(properties);

        ArmOperation<CodeVersionResource> CodeVersionResourceOperation = await codeContainerDataResource.GetCodeVersions().CreateOrUpdateAsync(WaitUntil.Completed, version, data);
        CodeVersionResource codeVersionResource = CodeVersionResourceOperation.Value;
        Console.WriteLine($"codeVersionResource {codeVersionResource.Data.Id} created.");

        return codeVersionResource;
    }
    // </GetOrCreateCodeVersionAsync>

}
