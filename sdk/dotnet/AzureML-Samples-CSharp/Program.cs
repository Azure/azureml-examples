using Azure.Core;
using Azure.Identity;
using Azure.MachineLearning.Samples.Assets.Code;
using Azure.MachineLearning.Samples.Assets.Component;
using Azure.MachineLearning.Samples.Assets.Data;
using Azure.MachineLearning.Samples.Assets.Environment;
using Azure.MachineLearning.Samples.Assets.Model;
using Azure.MachineLearning.Samples.Endpoints.Batch;
using Azure.MachineLearning.Samples.Endpoints.Online;
using Azure.MachineLearning.Samples.Jobs.AutomlJob;
using Azure.MachineLearning.Samples.Jobs.CommandJob;
using Azure.MachineLearning.Samples.Jobs.PipelineJob;
using Azure.MachineLearning.Samples.Jobs.SweepJob;
using Azure.MachineLearning.Samples.Resources.Compute;
using Azure.MachineLearning.Samples.Resources.ResourceGroup;
using Azure.MachineLearning.Samples.Resources.Workspace;
using Azure.MachineLearning.Samples.Shared;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;
using Constants = Azure.MachineLearning.Samples.Shared.Constants;

namespace Azure.MachineLearning.Samples;

public class Program
{
    public static async Task Main(string[] args)
    {
        try
        {
            await PerformMachineLearningOperations().ConfigureAwait(false);
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(Common.FlattenException(exception));
        }

        Console.WriteLine("Press Enter to continue.");
        Console.ReadLine();
    }

    /// <summary>
    /// Perform ML operations
    /// </summary>
    /// <returns></returns>
    static async Task PerformMachineLearningOperations()
    {
        // First we construct our armClient
        ArmClient armClient = new ArmClient(new DefaultAzureCredential());
        SubscriptionResource subscription = await armClient.GetDefaultSubscriptionAsync();

        // Create ResourceGroup
        var resourceGroup = await ResourceGroupOperations.CreateResourceGroup(armClient, Constants.ResourceGroupName);
        Console.WriteLine($"resourceGroup is {resourceGroup.Data.Id}...");
        AzureLocation location = AzureLocation.WestUS2;

        // Create workspace, if not exists. Note, you need to have the dependent resources created by yourself.
        // Dependent resources: ApplicationInsights, KeyVault, Storage account
        MachineLearningWorkspaceResource workspace = await WorkspaceOperations.GetOrCreateWorkspaceAsync(resourceGroup, Constants.WorkspaceName, location, Constants.ApplicationInsightsId, Constants.StorageAccountId, Constants.KeyVaultId);
        Console.WriteLine($"Workspace created: {workspace.Id}...");

        //Get a list of Workspace within a specific resource group
        await WorkspaceOperations.ListAllWorkspaceAsync(resourceGroup);

        // Create AmlCompute
        var amlCompute = await AMLComputeOperations.GetOrCreateAMLComputeAsync(resourceGroup, Constants.WorkspaceName, Constants.AMLComputeName, location);
        await AMLComputeOperations.ListComputePropertiesAsync(resourceGroup, Constants.WorkspaceName, amlCompute.Data.Name);

        // Create AKSCompute
        var aksCompute = await AKSComputeOperations.GetOrCreateAKSComputeAsync(resourceGroup, Constants.WorkspaceName, Constants.AKSComputeName, location, Constants.ClusterFqdn);
        Console.WriteLine($"Aks Compute created: {aksCompute.Id}...");

        // Create CodeVersionResource
        var codeResource = await CodeOperations.GetOrCreateCodeVersionAsync(armClient, resourceGroup, Constants.WorkspaceName, Constants.CodeName, Constants.CodeVersion, Constants.CodeUri);
        Console.WriteLine($"CodeVersionResource created: {codeResource.Id}...");

        // Create EnvironmentVersionResource
        var environmentResource = await EnvironmentOperations.GetOrCreateEnvironmentVersionAsync(armClient, resourceGroup, Constants.WorkspaceName, Constants.EnvironmentName, Constants.EnvironmentVersion);
        Console.WriteLine($"EnvironmentVersionResource created: {environmentResource.Id}...");

        //// Create ComponentVersionResource
        var componentResource = await ComponentOperations.GetOrCreateComponentVersionAsync(armClient, resourceGroup, Constants.WorkspaceName, Constants.ComponentName, Constants.ComponentVersion, environmentResource.Id, codeArtifactId: codeResource.Id);
        Console.WriteLine($"ComponentVersionResource created: {componentResource.Id}...");

        //// Create ComponentVersionResource
        componentResource = await ComponentOperations.GetOrCreateComponentVersion_Pipeline_Async(armClient, resourceGroup, Constants.WorkspaceName, Constants.ComponentNameForPipeline, Constants.ComponentVersionForPipeline, environmentResource.Id, codeArtifactId: codeResource.Id);
        Console.WriteLine($"ComponentVersionResource created: {componentResource.Id}...");

        // Create DatasetVersionResource
        var dataResource = await DataOperations.GetOrCreateDataVersionAsync(armClient, resourceGroup, Constants.WorkspaceName, Constants.DataName, Constants.DataVersion);
        Console.WriteLine($"Dataset created: {dataResource.Id}...");

        await DataOperations.ListDataAsync(resourceGroup, Constants.WorkspaceName);
        await DataOperations.ListDatastoreAsync(resourceGroup, Constants.WorkspaceName);
        
        // Create ModelVersionResource
        var modelResource = await ModelOperations.GetOrCreateModelVersionAsync(armClient, resourceGroup, Constants.WorkspaceName, Constants.ModelName, Constants.ModelVersion, Constants.ModelUri);
        Console.WriteLine($"ModelVersionResource created: {modelResource.Id}...");

        //// Create OnlineEndpoint
        var OnlineEndpointResource = await ManagedOnlineEndpointOperations.GetOrCreateOnlineEndpointAsync(resourceGroup, Constants.WorkspaceName, Constants.OnlineEndpointName, location);
        Console.WriteLine($"OnlineEndpointResource created: {OnlineEndpointResource.Id}...");

        // Create OnlineDeployment
        var OnlineDeploymentResource = await ManagedOnlineEndpointOperations.GetOrCreateOnlineDeploymentAsync(resourceGroup, Constants.WorkspaceName, Constants.OnlineEndpointName, Constants.OnlineDeploymentName, modelResource.Id, environmentResource.Id, codeResource.Id, location);
        Console.WriteLine($"OnlineDeploymentResource created: {OnlineDeploymentResource.Id}...");

        // Create BatchEndpoint
        var BatchEndpointResource = await BatchEndpointOperations.GetOrCreateBatchEndpointAsync(resourceGroup, Constants.WorkspaceName, Constants.BatchEndpointName, location);
        Console.WriteLine($"BatchEndpointResource created: {BatchEndpointResource.Id}...");

        // Create BatchDeployment
        var BatchDeploymentResource = await BatchEndpointOperations.GetOrCreateBatchDeploymentAsync(resourceGroup, Constants.WorkspaceName, Constants.BatchEndpointName, Constants.BatchDeploymentName, modelResource.Id, environmentResource.Id, codeResource.Id, Constants.ComputeId, location);
        Console.WriteLine($"BatchDeploymentResource created: {BatchDeploymentResource.Id}...");

        // Submit a Command Job.
        string jobId = Common.RandomString(15, true);
        var commandJob = await CommandJobOperations.SubmitCommandJobAsync(resourceGroup, Constants.WorkspaceName, jobId, Constants.ExperimentName, environmentResource.Id, Constants.ComputeId);
        Console.WriteLine($"Experiment: {commandJob.Data.Properties.ExperimentName} returned status: {commandJob.Data.Properties.Status}...");

        commandJob = await Common.WaitForJobToFinishAsync(resourceGroup, Constants.WorkspaceName, jobId);
        if (commandJob.Data.Properties.Status == JobStatus.Completed)
        {
            Console.WriteLine($"CommandJob Experiment: {commandJob.Data.Properties.ExperimentName} ended with status: {commandJob.Data.Properties.Status}...");
        }

        // Submit a sweepJob Job.
        jobId = Common.RandomString(15, true);
        var sweepJob = await SweepJobOperations.SubmitSweepJobAsync(resourceGroup, Constants.WorkspaceName, jobId, Constants.SweepJobExperimentName, environmentResource.Id, codeResource.Id, Constants.ComputeId);
        Console.WriteLine($"Experiment: {sweepJob.Data.Properties.ExperimentName} returned status: {sweepJob.Data.Properties.Status}...");

        sweepJob = await Common.WaitForJobToFinishAsync(resourceGroup, Constants.WorkspaceName, jobId);
        if (sweepJob.Data.Properties.Status == JobStatus.Completed)
        {
            Console.WriteLine($"SweepJob Experiment: {sweepJob.Data.Properties.ExperimentName} ended with status: {sweepJob.Data.Properties.Status}...");
        }

        // Submit a pipelineJob Job.
        jobId = Common.RandomString(15, true);
        var pipelineJob = await PipelineJobOperations.SubmitPipelineJobAsync(resourceGroup, Constants.WorkspaceName, jobId, Constants.PipelineJobExperimentName, Constants.ComputeId, componentResource.Id, Constants.Datastore);
        Console.WriteLine($"PipelineJob Experiment: {pipelineJob.Data.Properties.ExperimentName} returned status: {pipelineJob.Data.Properties.Status}...");

        pipelineJob = await Common.WaitForJobToFinishAsync(resourceGroup, Constants.WorkspaceName, jobId);
        if (pipelineJob.Data.Properties.Status == JobStatus.Completed)
        {
            Console.WriteLine($"PipelineJob Experiment: {pipelineJob.Data.Properties.ExperimentName} ended with status: {pipelineJob.Data.Properties.Status}...");
        }

        jobId = Common.RandomString(15, true);
        string uniqueName = Guid.NewGuid().ToString("n").Substring(0, 6);
        // Submit an AutoML TextNer Job
        MachineLearningJobResource autoMLTextNerJob = await AutoMLJobOperations.SubmitAutoMLTextNerAsync(resourceGroup, Constants.WorkspaceName, jobId, "AutoMLTextNerJob" + uniqueName, environmentResource.Id, Constants.ComputeId);
        Console.WriteLine($"Experiment: {autoMLTextNerJob.Data.Properties.ExperimentName} returned status: {autoMLTextNerJob.Data.Properties.Status}...");

        autoMLTextNerJob = await Common.WaitForJobToFinishAsync(resourceGroup, Constants.WorkspaceName, jobId);
        if (autoMLTextNerJob.Data.Properties.Status == JobStatus.Completed)
        {
            Console.WriteLine($"AutoMLTextNerJob Experiment: {autoMLTextNerJob.Data.Properties.ExperimentName} ended with status: {autoMLTextNerJob.Data.Properties.Status}...");
        }

        jobId = Common.RandomString(15, true);
        // Submit an AutoML Forecast Job
        MachineLearningJobResource autoMLForecastJob = await AutoMLJobOperations.SubmitAutoMLForecastAsync(resourceGroup, Constants.WorkspaceName, jobId, "AutoMLForecastingJob" + uniqueName, environmentResource.Id, Constants.ComputeId);
        Console.WriteLine($"Experiment: {autoMLForecastJob.Data.Properties.ExperimentName} returned status: {autoMLForecastJob.Data.Properties.Status}...");

        autoMLForecastJob = await Common.WaitForJobToFinishAsync(resourceGroup, Constants.WorkspaceName, jobId);
        if (autoMLForecastJob.Data.Properties.Status == JobStatus.Completed)
        {
            Console.WriteLine($"AutoMLForecastJob Experiment: {autoMLForecastJob.Data.Properties.ExperimentName} ended with status: {autoMLForecastJob.Data.Properties.Status}...");
        }

        jobId = Common.RandomString(15, true);
        // Submit an AutoML TextClassification job
        MachineLearningJobResource autoMLTextClassificationJob = await AutoMLJobOperations.SubmitAutoMLTextClassificationAsync(resourceGroup, Constants.WorkspaceName, jobId, "AutoMLTextClassificationJob" + uniqueName, environmentResource.Id, Constants.ComputeId);
        Console.WriteLine($"Experiment: {autoMLTextClassificationJob.Data.Properties.ExperimentName} returned status: {autoMLTextClassificationJob.Data.Properties.Status}...");

        // autoMLTextClassificationJob = await WaitForJobToFinishAsync(resourceGroup, Constants.WorkspaceName, jobId);
        if (autoMLTextClassificationJob.Data.Properties.Status == JobStatus.Completed)
        {
            Console.WriteLine($"AutoMLForecastJob Experiment: {autoMLTextClassificationJob.Data.Properties.ExperimentName} ended with status: {autoMLTextClassificationJob.Data.Properties.Status}...");
        }

        jobId = Common.RandomString(15, true);
        // Submit an AutoML ImageClassification job
        var autoMLImageClassificationJob = await AutoMLJobOperations.SubmitAutoMLImageClassificationAsync(resourceGroup, Constants.WorkspaceName, jobId, "AutoMLImageClassificationJob" + uniqueName, environmentResource.Id, Constants.ComputeId);
        Console.WriteLine($"Experiment: {autoMLImageClassificationJob.Data.Properties.ExperimentName} returned status: {autoMLImageClassificationJob.Data.Properties.Status}...");

        // autoMLImageClassificationJob = await WaitForJobToFinishAsync(resourceGroup, Constants.WorkspaceName, jobId);
        if (autoMLImageClassificationJob.Data.Properties.Status == JobStatus.Completed)
        {
            Console.WriteLine($"AutoMLImageClassificationJob Experiment: {autoMLImageClassificationJob.Data.Properties.ExperimentName} ended with status: {autoMLImageClassificationJob.Data.Properties.Status}...");
        }

        jobId = Common.RandomString(15, true);
        // Submit an AutoML ImageObjectDetection job
        var autoMLImageObjectDetectionJob = await AutoMLJobOperations.SubmitAutoMLImageObjectDetectionAsync(resourceGroup, Constants.WorkspaceName, jobId, "AutoMLImageObjectDetectionJob" + uniqueName, environmentId: environmentResource.Id, computeId: Constants.ComputeId);
        Console.WriteLine($"Experiment: {autoMLImageObjectDetectionJob.Data.Properties.ExperimentName} returned status: {autoMLImageObjectDetectionJob.Data.Properties.Status}...");

        autoMLImageObjectDetectionJob = await Common.WaitForJobToFinishAsync(resourceGroup, Constants.WorkspaceName, jobId);
        if (autoMLImageObjectDetectionJob.Data.Properties.Status == JobStatus.Completed)
        {
            Console.WriteLine($"AutoMLImageObjectDetectionJob Experiment: {autoMLImageObjectDetectionJob.Data.Properties.ExperimentName} ended with status: {autoMLImageObjectDetectionJob.Data.Properties.Status}...");
        }
        
    }
}