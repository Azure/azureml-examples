using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;

namespace Azure.MachineLearning.Samples.Jobs.CommandJob;

internal class CommandJobOperations
{

    /// <summary>
    /// If the specified CommandJob exists, get that CommandJob.
    /// If it does not exist, creates a new CommandJob. 
    /// </summary>
    /// <param name="resourceGroupName"></param>
    /// <param name="workspaceName"></param>
    /// <param name="id"></param>
    /// <param name="experimentName"></param>
    /// <returns></returns>
    // <SubmitCommandJobAsync>
    public static async Task<MachineLearningJobResource> SubmitCommandJobAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string id,
        string experimentName,
        string environmentId,
        string computeId)
    {
        Console.WriteLine("Creating a CommandJob...");
        // The command to execute on startup of the job. eg. "python train.py"
        string Command = "echo \"hello world\"";
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        var commandJob = new Azure.ResourceManager.MachineLearning.Models.CommandJob(
            Command, environmentId)
        {
            ExperimentName = experimentName,
            DisplayName = "Display name-" + Guid.NewGuid().ToString("n").Substring(0, 6),
            Limits = new CommandJobLimits()
            {
                // The max run duration in ISO 8601 format, after which the job will be cancelled.
                Timeout = TimeSpan.FromSeconds(120),
            },
            EnvironmentId = environmentId,
            IsArchived = false,
            Inputs = new Dictionary<string, JobInput> {
                { "int_a", new LiteralJobInput("11")},
                { "int_b", new LiteralJobInput("21")},
            },
            ComputeId = computeId,
            Resources = new ResourceConfiguration
            {
                InstanceCount = 1,
            },
            Outputs = new Dictionary<string, JobOutput> {
                { "out_sum", new UriFolderJobOutput(){Mode = OutputDeliveryMode.ReadWriteMount, Description = null} },
                { "out_prod", new UriFolderJobOutput(){Mode = OutputDeliveryMode.ReadWriteMount, Description = null} },
            },

            Properties = new Dictionary<string, string>
                {
                    { "property-name", "property-value" },
                },
            Tags = new Dictionary<string, string>
                {
                    { "tag-name", "tag-value" },
                },

            // Environment variables included in the job.
            EnvironmentVariables = new Dictionary<string, string>()
                {
                    { "env-var", "env-var-value" }
                },
            Description = "This is a description of test Command Job",

        };

        MachineLearningJobData data = new MachineLearningJobData(commandJob);
        ArmOperation<MachineLearningJobResource> jobOperation = await ws.GetMachineLearningJobs().CreateOrUpdateAsync(WaitUntil.Completed, id, data);
        MachineLearningJobResource jobResource = jobOperation.Value;
        Console.WriteLine($"JobCreateOrUpdateOperation {jobResource.Data.Id} created.");
        return jobResource;
    }
    // </SubmitCommandJobAsync>
}
