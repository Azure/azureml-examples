// <copyright file="AuthenticationClient.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using Azure.Core;

namespace BatchInferencingSamples
{
    internal class AuthenticationClient
    {
        public async Task<string> GetAccessToken(TokenCredential credentials)
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
