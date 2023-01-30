using Azure.Core;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;
using Newtonsoft.Json.Linq;

namespace Azure.MachineLearning.Samples.Assets.Component;

internal class ComponentOperations
{

    /// <summary>
    /// If the specified ComponentVersion exists, get that ComponentVersion.
    /// If it does not exist, creates a new ComponentVersion. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="componentName"></param>
    /// <param name="version"></param>
    /// <returns></returns>
    // <GetOrCreateComponentVersionAsync>
    public static async Task<ComponentVersionResource> GetOrCreateComponentVersionAsync(
        ArmClient armClient,
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string componentName,
        string version,
        string environmentId,
        string codeArtifactId)
    {
        Console.WriteLine("Creating a ComponentVersion Resource...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        string resourceId = $"{ws.Id}/components/{componentName}";
        var id = new ResourceIdentifier(resourceId);
        ComponentContainerResource componentContainerResource = armClient.GetComponentContainerResource(id);

        JObject jsonObject = JObject.Parse(@"{
  '$schema': 'https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json',
  'name': '" + componentName + @"',
  'type': 'command',
  'version': '" + version + @"',
  'code': 'azureml:/" + codeArtifactId + @"',
  'command': 'echo Hello World & echo [${{inputs.component_in_number}}] & echo ${{inputs.component_in_path}} & echo ${{outputs.component_out_path}}',
  'description': 'This is the basic command component',
  'display_name': 'A basic Command Component',
  'environment': 'azureml:" + environmentId + @"',
  'inputs': {
    'component_in_number': {
      'default': '10.99',
      'description': 'A number',
      'optional': true,
      'type': 'number'
    },
    'component_in_path': {
      'description': 'A path',
      'optional': false,
      'type': 'path'
    }
  },
'outputs': {
    'component_out_path': {
        'name': 'component_out_path',
        'type': 'path'
    }
  },
  'is_deterministic': true,
  'outputs': {
    'component_out_path': {
      'type': 'path'
    }
  },
  'resources': {
    'instance_count': 1
  },
  'tags': {
    'owner': 'sdkteam',
    'tag': 'tagvalue'
  },
}");

        ComponentVersionProperties properties = new ComponentVersionProperties { ComponentSpec = new BinaryData(jsonObject.ToString()) };
        ComponentVersionData data = new ComponentVersionData(properties);

        ArmOperation<ComponentVersionResource> componentVersionResourceOperation = await componentContainerResource.GetComponentVersions().CreateOrUpdateAsync(WaitUntil.Completed, version, data);
        ComponentVersionResource componentVersionResource = componentVersionResourceOperation.Value;
        Console.WriteLine($"ComponentVersionResource {componentVersionResource.Id} created.");
        return componentVersionResource;
    }
    // </GetOrCreateComponentVersionAsync>



    /// <summary>
    /// If the specified ComponentVersion exists, get that ComponentVersion.
    /// If it does not exist, creates a new ComponentVersion. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="componentName"></param>
    /// <param name="version"></param>
    /// <returns></returns>
    // <GetOrCreateComponentVersion_Pipeline_Async>
    public static async Task<ComponentVersionResource> GetOrCreateComponentVersion_Pipeline_Async(
        ArmClient armClient,
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string componentName,
        string version,
        string environmentId,
        string codeArtifactId)
    {
        Console.WriteLine("Creating a ComponentVersion Resource...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        string resourceId = $"{ws.Id}/components/{componentName}";
        var id = new ResourceIdentifier(resourceId);
        ComponentContainerResource componentContainerResource = armClient.GetComponentContainerResource(id);

        JObject jsonObject = JObject.Parse(@"{
  '$schema': 'https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json',
  'name': '" + componentName + @"',
  'type': 'command',
  'version': '" + version + @"',
  'code': 'azureml:/" + codeArtifactId + @"',
  'command': 'echo Hello World & echo [${{inputs.component_in_number}}] ',
  'description': 'This is the basic command component',
  'display_name': 'A basic Command Component',
  'environment': 'azureml:" + environmentId + @"',
  'inputs': {
    'component_in_number': {
      'default': '10.99',
      'description': 'A number',
      'optional': true,
      'type': 'number'
    },
  },
'outputs': {
  },
  'is_deterministic': true,
  'outputs': {
  },
  'resources': {
    'instance_count': 1
  },
  'tags': {
    'owner': 'sdkteam',
    'tag': 'tagvalue'
  },
}");

        ComponentVersionProperties properties = new ComponentVersionProperties { ComponentSpec = new BinaryData(jsonObject.ToString()) };
        ComponentVersionData data = new ComponentVersionData(properties);

        ArmOperation<ComponentVersionResource> componentVersionResourceOperation = await componentContainerResource.GetComponentVersions().CreateOrUpdateAsync(WaitUntil.Completed, version, data);
        ComponentVersionResource componentVersionResource = componentVersionResourceOperation.Value;
        Console.WriteLine($"ComponentVersionResource {componentVersionResource.Id} created.");
        return componentVersionResource;
    }
    // </GetOrCreateComponentVersion_Pipeline_Async>

}
