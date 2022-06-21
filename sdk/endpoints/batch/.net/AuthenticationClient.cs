using Azure.Core;

namespace batch_inferencing_samples
{
    internal class AuthenticationClient
    {
        public async Task<string> GetAccessToken(TokenCredential credentials)
        {
            TokenCredential credentials = new DefaultAzureCredential();

            string[] scopes = new string[]
            {
                "https://ml.azure.com/.default"
            };
            TokenRequestContext ctx = new TokenRequestContext(scopes);
            var accessToken = await credentials.GetTokenAsync(ctx, CancellationToken.None).ConfigureAwait(false);

            return accessToken.Token;
        }
    }
}
