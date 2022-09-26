// <copyright file="AuthenticationClient.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using Azure.Core;

namespace Microsoft.Azure.MachineLearning.Samples.BatchInferencing
{
    internal class AuthenticationClient
    {
        /// <summary>
        /// Gets an access token to score.
        /// </summary>
        /// <param name="credentials">The user's credentials.</param>
        /// <returns>A string representation of the access token.</returns>
        public async Task<string> GetAccessTokenAsync(TokenCredential credentials)
        {
            // Given this is a dataplane operation we need the authentication scope to come from Azure Machine Learning
            string[] scopes = new string[]
            {
                "https://ml.azure.com/.default"
            };
            TokenRequestContext ctx = new TokenRequestContext(scopes);
            AccessToken accessToken = await credentials.GetTokenAsync(ctx, CancellationToken.None).ConfigureAwait(false);

            return accessToken.Token;
        }
    }
}
