// <copyright file="BatchEndpointClient.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using Azure;
using Azure.Core;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using System.Text;
using System.Text.Json;

namespace Microsoft.Azure.MachineLearning.Samples.BatchInferencing
{
    internal class BatchEndpointClient
    {
        /// <summary>
        /// Gets batch endpoint information.
        /// </summary>
        /// <param name="subscriptionId">The subcription ID holding the batch endpoint.</param>
        /// <param name="resourceGroup">The name of the resource group holding the batch endpoint.</param>
        /// <param name="workspaceName">The name of the workspace holding the batch endpoint.</param>
        /// <param name="batchEndpointName">The batch endpoint name.</param>
        /// <param name="tokenCredential">The credentials leveraged to access the batch endpoint data.</param>
        /// <returns></returns>
        public static async Task<BatchEndpointData> GetAsync(Guid subscriptionId, string resourceGroup, string workspaceName, string batchEndpointName, TokenCredential tokenCredential)
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

        /// <summary>
        /// Invokes a batch endpoint which triggers a batch inferencing job.
        /// </summary>
        /// <param name="scoringUri">The batch endpoint scoring URI.</param>
        /// <param name="inputFolderUri">The URI pointing to the location of the input folder containing the data to be inferenced.</param>
        /// <param name="outputFileUri">The URI pointing to the location where the inferenced output should be written to.</param>
        /// <param name="tokenCredential">The credentials leveraged to score.</param>
        /// <returns></returns>
        public static async Task InvokeAsync(Uri scoringUri,Uri inputFolderUri, Uri outputFileUri, TokenCredential credentials)
        {
            HttpClient client = new HttpClient();

            BatchScoringPayload payload = new BatchScoringPayload
            {
                Properties = new BatchScoringProperties
                {
                    InputData = new Dictionary<string, BatchScoringInput>
                    {
                        ["myInput"] = new BatchScoringInput
                        {
                            JobInputType = JobInputType.UriFolder,
                            Uri = inputFolderUri,
                        }
                    },
                    OutputData = new Dictionary<string, BatchScoringOutput>
                    {
                        ["myOutput"] = new BatchScoringOutput
                        {
                            JobOutputType = JobOutputType.UriFile,
                            Uri = outputFileUri
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

            string accessToken = await authClient.GetAccessTokenAsync(credentials).ConfigureAwait(false);
            request.Headers.Add("Authorization", $"Bearer {accessToken}");

            HttpResponseMessage response = await client.SendAsync(request).ConfigureAwait(false);
            Console.WriteLine($"Status Code: {(int)response.StatusCode} - {response.ReasonPhrase}");
        }
    }
}
