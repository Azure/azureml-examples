// <copyright file="Program.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System.Text;
using System.Text.Json;
using Azure.Core;
using Azure.Identity;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using batch_inferencing_samples;

namespace BatchInferencingSamples
{
    internal class Program
    {
        public static async Task Main(string[] args)
        {
            TokenCredential credentials = new DefaultAzureCredential();
            
            Guid subscriptionId = Guid.Parse("f375b912-331c-4fc5-8e9f-2d7205e3e036");
            string resourceGroupName = "batch-inferencing";
            string workspaceName = "batch-inferencing-ws-eastus";
            string batchEndpointName = "klod1";

            // Get the batch endpoint to get its scoring URI.
            BatchEndpointData batchEndpointData = await BatchEndpointActions
                .Get(subscriptionId, resourceGroupName, workspaceName, batchEndpointName, credentials)
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
            await BatchEndpointActions.Invoke(scoringUri, inputFolderUri, outputFileUri, credentials).ConfigureAwait(false);
        }
    }
}