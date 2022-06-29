// <copyright file="EndpointClient.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using Azure;
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
            
            Response<BatchEndpointResource> batchEndpointData = await armClient
                .GetBatchEndpointResource(batchEndpointResourceId)
                .GetAsync()
                .ConfigureAwait(false);
            
            return batchEndpointData.Value.Data;
        }

        public static async Task Invoke(Uri scoringUri,Uri inputFolderUri, Uri outputFileUri, TokenCredential credentials)
        {
            HttpClient client = new HttpClient();

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

            string stringPayload = JsonSerializer.Serialize(payload);
            StringContent httpContent = new StringContent(stringPayload, Encoding.UTF8, "application/json");

            HttpRequestMessage request = new HttpRequestMessage()
            {
                RequestUri = scoringUri,
                Method = HttpMethod.Post,
                Content = httpContent
            };

            AuthenticationClient authClient = new AuthenticationClient();

            string accessToken = await authClient.GetAccessToken(credentials).ConfigureAwait(false);
            request.Headers.Add("Authorization", $"Bearer {accessToken}");

            HttpResponseMessage response = await client.SendAsync(request).ConfigureAwait(false);
            Console.WriteLine($"Status Code: {(int)response.StatusCode} - {response.ReasonPhrase}");
        }
    }
}
