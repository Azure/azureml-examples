// <copyright file="Program.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System.Text;
using System.Text.Json;
using Azure.Core;
using Azure.Identity;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearningServices;

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
            string batchEndpointName = "mnist-3";

            Uri scoringUri = await GetScoringUri(subscriptionId, resourceGroupName, workspaceName, batchEndpointName, credentials).ConfigureAwait(false);
            await InvokeEndpoint(scoringUri, credentials).ConfigureAwait(false);
        }

        public static async Task<Uri> GetScoringUri(Guid subscriptionId, string resourceGroup, string workspaceName, string batchEndpointName, TokenCredential tokenCredential)
        {
            Azure.ResourceManager.ResourceIdentifier batchEndpointResourceId =
                new Azure.ResourceManager.ResourceIdentifier($"/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/batchEndpoints/{batchEndpointName}");

            ArmClient myArmClient = new ArmClient(tokenCredential);
            var batchEndpoint = myArmClient.GetBatchEndpointTrackedResource(batchEndpointResourceId);
            var data = await batchEndpoint.GetAsync().ConfigureAwait(false);
            return new Uri(data.Value.Data.Properties.ScoringUri);
        }

        private static async Task InvokeEndpoint(Uri scoringUri, TokenCredential credentials)
        {
            var client = new HttpClient();

            BatchScoringPayload payload = CreatePayload(scoringUri);

            var stringPayload = JsonSerializer.Serialize(payload);
            var httpContent = new StringContent(stringPayload, Encoding.UTF8, "application/json");

            var request = new HttpRequestMessage()
            {
                RequestUri = scoringUri,
                Method = HttpMethod.Post,
                Content = httpContent
            };
            string accessToken = await GetAccessToken(credentials).ConfigureAwait(false);
            request.Headers.Add("Authorization", $"Bearer {accessToken}");

            var response = await client.SendAsync(request).ConfigureAwait(false);
            Console.WriteLine($"Status Code: {(int)response.StatusCode} - {response.ReasonPhrase}");
        }

        private static async Task<string> GetAccessToken(TokenCredential credentials)
        {
            string[] scopes = new string[]
            {
                "https://ml.azure.com/.default"
            };
            TokenRequestContext ctx = new TokenRequestContext(scopes);
            var accessToken = await credentials.GetTokenAsync(ctx, CancellationToken.None).ConfigureAwait(false);

            return accessToken.Token;
        }

        private static BatchScoringPayload CreatePayload(Uri scoringUri)
            => new BatchScoringPayload
            {
                Properties = new BatchScoringInputProperties
                {
                    InputData = new Dictionary<string, BatchScoringInput>
                    {
                        {
                            "mnistinput",
                            new BatchScoringInput
                            {
                                JobInputType = JobInputType.UriFolder,
                                Uri = scoringUri,
                            }
                        },
                    },
                },
            };
    }
}