using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;
using Newtonsoft.Json.Linq;

namespace Azure.MachineLearning.Samples.Jobs.PipelineJob;

internal class PipelineJobOperations
{

    /// <summary>
    /// If the specified Pipeline exists, get that Pipeline.
    /// If it does not exist, creates a new Pipeline. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="id"></param>
    /// <param name="experimentName"></param>
    /// <param name="computeId"></param>
    /// <param name="componentId"></param>
    /// <param name="datastore"></param>
    /// <returns></returns>
    // </SubmitPipelineJobAsync>
    public static async Task<MachineLearningJobResource> SubmitPipelineJobAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string id,
        string experimentName,
        // string environmentId,
        string computeId,
        string componentId,
        string datastore)
    {
        Console.WriteLine("Creating a PipelineJob...");
        MachineLearningJobResource MachineLearningJobResource;
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        var jsonObject = new JObject();
        jsonObject.Add("ContinueRunOnStepFailure", false);
        jsonObject.Add("DefaultDatastoreName", datastore);
        jsonObject.Add("Datastore", datastore);

        BinaryData binaryDataSetting = new BinaryData(jsonObject.ToString());

        var step01 = JObject.FromObject(new
        {
            ComputeId = computeId,
            ComponentId = componentId,
            Inputs = new Dictionary<string, JobInput> {
                { "component_in_number", new LiteralJobInput("21")},
            },
        });

        var inputDict = new Dictionary<string, JobInput>
        {
            { "component_in_number", new LiteralJobInput("22")},
            };
        Dictionary<string, BinaryData> jobsDictionary = new Dictionary<string, BinaryData>();
        jobsDictionary.Add("job01", new BinaryData(step01.ToString()));
        var pipelineJob = new Azure.ResourceManager.MachineLearning.Models.PipelineJob()
        {
            ExperimentName = experimentName,
            Description = "This is a description of test pipeline Job",
            DisplayName = "Pipeline display name-" + Guid.NewGuid().ToString("n").Substring(0, 6),
            ComputeId = computeId,
            Inputs = inputDict,
            Jobs = jobsDictionary,
            Outputs = null,
            Properties = new Dictionary<string, string>
            {
                { "property-name", "property-value" },
            },
            Tags = new Dictionary<string, string>
            {
                { "tag-name", "tag-value" },
            },
            IsArchived = false,
            Settings = binaryDataSetting,
        };
        Console.WriteLine($"Pipeline Job: {pipelineJob.Jobs.FirstOrDefault().Value}");
        MachineLearningJobData MachineLearningJobData = new MachineLearningJobData(pipelineJob);
        ArmOperation<MachineLearningJobResource> jobOperation = await ws.GetMachineLearningJobs().CreateOrUpdateAsync(WaitUntil.Completed, id, MachineLearningJobData);
        MachineLearningJobResource = jobOperation.Value;
        Console.WriteLine($"Pipeline Job: {MachineLearningJobResource.Data.Id} created.");
        return MachineLearningJobResource;
    }
    // </SubmitPipelineJobAsync>

}
