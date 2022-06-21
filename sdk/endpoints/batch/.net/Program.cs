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

            BatchEndpointData batchEndpointData = await BatchEndpointActions
                .Get(subscriptionId, resourceGroupName, workspaceName, batchEndpointName, credentials)
                .ConfigureAwait(false);

            Uri scoringUri = batchEndpointData.Properties.ScoringUri;
            Console.WriteLine($"Scoring URI: {scoringUri}");
            
            await BatchEndpointActions.Invoke(scoringUri, credentials).ConfigureAwait(false);
        }
    }
}