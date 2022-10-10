using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;
using Newtonsoft.Json.Linq;

namespace Azure.MachineLearning.Samples.Jobs.SweepJob;

internal class SweepJobOperations
{

    /// <summary>
    /// If the specified SweepJob exists, get that SweepJob.
    /// If it does not exist, creates a new SweepJob. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="id"></param>
    /// <param name="experimentName"></param>
    /// <param name="environmentId"></param>
    /// <param name="codeArtifactId"></param>
    /// <param name="computeId"></param>
    /// <returns></returns>
    // </SubmitSweepJobAsync>
    public static async Task<MachineLearningJobResource> SubmitSweepJobAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string id,
        string experimentName,
        string environmentId,
        string codeArtifactId,
        string computeId)
    {
        Console.WriteLine("Creating a SweepJob...");
        // The command to execute on startup of the job. eg. "python train.py"
        // string Command = "echo \"hello world\"";
        string Command = "python ./hello.py";
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        JObject parameterSpace = JObject.FromObject(new Dictionary<string, List<object>>()
        {
            {
                "parameter1",
                new List<object>()
                {
                    "uniform",
                    new List<double>()
                    {
                        0.0,
                        1.0
                    }
                }
            }
        });

        BinaryData searchSpace = new BinaryData(parameterSpace.ToString());

        TrialComponent trialComponent = new TrialComponent(Command, environmentId)
        {
            Resources = new ResourceConfiguration
            {
                InstanceCount = 1,
            },
            CodeId = codeArtifactId,
            EnvironmentVariables = new Dictionary<string, string>()
                    {
                        { "env-var", "env-var-value" }
                    }
        };
        SamplingAlgorithm samplingAlgorithm = new RandomSamplingAlgorithm
        {
            Rule = RandomSamplingAlgorithmRule.Random,
            Seed = 2,
        };

        Objective objective = new Objective(Goal.Maximize, "primary-metric-name");

        var sweepJob = new Azure.ResourceManager.MachineLearning.Models.SweepJob(objective, samplingAlgorithm, searchSpace, trialComponent)
        {
            ExperimentName = experimentName,
            DisplayName = "Sweep display name-" + Guid.NewGuid().ToString("n").Substring(0, 6),
            Limits = new SweepJobLimits()
            {
                // The max run duration in ISO 8601 format, after which the job will be cancelled.
                Timeout = TimeSpan.FromSeconds(1800),
                MaxConcurrentTrials = 2,
                MaxTotalTrials = 10,
                TrialTimeout = TimeSpan.FromMinutes(1200),
            },
            ComputeId = computeId,
            EarlyTermination = new TruncationSelectionPolicy()
            {
                EvaluationInterval = 10,
                DelayEvaluation = 200,
                TruncationPercentage = 50,
            },
            //     Mapping of input data bindings used in the job.
            Inputs = new Dictionary<string, JobInput> {
                { "int_a", new LiteralJobInput("11")},
                { "int_b", new LiteralJobInput("21")},
            },
            //     Mapping of output data bindings used in the job.
            Outputs = new Dictionary<string, JobOutput> {
                { "Dataset", new UriFolderJobOutput() { Mode = OutputDeliveryMode.ReadWriteMount, Description = "Output Description" } },

            },
            Properties = new Dictionary<string, string>
                {
                    { "property-name", "property-value" },
                },
            Tags = new Dictionary<string, string>
                {
                    { "tag-name", "tag-value" },
                },
            Description = "This is a description of test Sweep Job",
            IsArchived = false,
        };

        MachineLearningJobData data = new MachineLearningJobData(sweepJob);
        ArmOperation<MachineLearningJobResource> jobOperation = await ws.GetMachineLearningJobs().CreateOrUpdateAsync(WaitUntil.Completed, id, data);
        MachineLearningJobResource jobResource = jobOperation.Value;
        Console.WriteLine($"JobCreateOrUpdateOperation {jobResource.Data.Id} created.");
        return jobResource;
    }
    // </SubmitSweepJobAsync>

}
