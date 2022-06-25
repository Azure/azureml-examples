// <copyright file="EndpointClient.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using Azure.Core;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using BatchInferencingSamples;
using System.Text;
using System.Text.Json;

namespace BatchInferencingSamples
{
    internal class BatchEndpointActions
    {
        public static async Task<BatchEndpointData> Get(Guid subscriptionId, string resourceGroup, string workspaceName, string batchEndpointName, TokenCredential tokenCredential)
        {
            ResourceIdentifier batchEndpointResourceId =
                new ResourceIdentifier($"/subscriptions/{subscriptionId}/resourceGroups/{resourceGroup}/providers/Microsoft.MachineLearningServices/workspaces/{workspaceName}/batchEndpoints/{batchEndpointName}");

            ArmClient armClient = new ArmClient(tokenCredential);
            var batchEndpoint = armClient.GetBatchEndpointResource(batchEndpointResourceId);
            var data = await batchEndpoint.GetAsync().ConfigureAwait(false);
            return data.Value.Data;
        }

        public static async Task Invoke(Uri scoringUri,Uri inputFolderUri, Uri outputFileUri, TokenCredential credentials)
        {
            var client = new HttpClient();

            BatchScoringPayload payload = new BatchScoringPayload
            {
                Properties = new BatchScoringProperties
                {
                    InputData = new Dictionary<string, BatchScoringInput>
                    {
                        {
                            "myInput",
                            new BatchScoringInput
                            {
                                JobInputType = JobInputType.UriFolder,
                                Uri = inputFolderUri,
                            }
                        },
                    },
                    OutputData = new Dictionary<string, BatchScoringOutput>
                    {
                        {
                            "myOutput",
                            new BatchScoringOutput
                            {
                                JobOutputType = JobOutputType.UriFile,
                                Uri = outputFileUri
                            }
                        }
                    }
                },
            };

            var stringPayload = JsonSerializer.Serialize(payload);
            var httpContent = new StringContent(stringPayload, Encoding.UTF8, "application/json");

            var request = new HttpRequestMessage()
            {
                RequestUri = scoringUri,
                Method = HttpMethod.Post,
                Content = httpContent
            };

            AuthenticationClient authClient = new AuthenticationClient();

            string accessToken = await authClient.GetAccessToken(credentials).ConfigureAwait(false);
            request.Headers.Add("Authorization", $"Bearer {accessToken}");

            var response = await client.SendAsync(request).ConfigureAwait(false);
            Console.WriteLine($"Status Code: {(int)response.StatusCode} - {response.ReasonPhrase}");
        }
    }
}
