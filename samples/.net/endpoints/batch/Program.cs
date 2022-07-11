// <copyright file="Program.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using Azure.Core;
using Azure.Identity;
using Azure.ResourceManager.MachineLearning;

namespace Microsoft.Azure.MachineLearning.Samples.BatchInferencing
{
    internal class Program
    {
        public static async Task Main(string[] args)
        {
            TokenCredential credentials = new DefaultAzureCredential();
            
            Guid subscriptionId = Guid.Parse("<your_subscription_id>");
            string resourceGroupName = "<your_resource_group_name>";
            string workspaceName = "<your_workspace_name>";
            string batchEndpointName = "<your_endpoint_name>";

            // Get the batch endpoint to get its scoring URI.
            BatchEndpointData batchEndpointData = await BatchEndpointClient
                .GetAsync(subscriptionId, resourceGroupName, workspaceName, batchEndpointName, credentials)
                .ConfigureAwait(false);

            Uri scoringUri = batchEndpointData.Properties.ScoringUri;
            Console.WriteLine($"Scoring URI: {scoringUri}");

            // Leveraging the mnist open dataset.
            Uri inputFolderUri = new Uri($"https://pipelinedata.blob.core.windows.net/sampledata/mnist");
            
            // storing the inference result in the workspace blobstore.
            // Filepath needs to be unique or else the 
            string outputFileName = "myData.csv";
            Uri outputFileUri = new Uri($"azureml://datastores/workspaceblobstore/paths/{batchEndpointName}/mnistOutput/{outputFileName}");

            // Invoke the batch endpoint.
            await BatchEndpointClient.InvokeAsync(scoringUri, inputFolderUri, outputFileUri, credentials).ConfigureAwait(false);
        }
    }
}