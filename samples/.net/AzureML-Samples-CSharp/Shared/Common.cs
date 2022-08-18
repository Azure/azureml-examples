using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Net.Http.Headers;
using System.Text;

namespace Azure.MachineLearning.Samples.Shared;

internal class Common
{

    /// <summary>
    /// Poll and Wait for ML CommandJob to finish
    /// </summary>
    /// <param name="workspaceClient"></param>
    /// <param name="resourceGroupName"></param>
    /// <param name="workspaceName"></param>
    /// <param name="id"></param>
    /// <returns></returns>
    // <WaitForJobToFinishAsync>
    public static async Task<MachineLearningJobResource> WaitForJobToFinishAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string id)
    {
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);
        // delay between each retry (in milliseconds)
        const int SleepIntervalMs = 20 * 1000;
        Console.WriteLine($"Starting to poll the status of Job Id: {id}");
        MachineLearningJobResource jobResource;
        do
        {
            jobResource = await ws.GetMachineLearningJobs().GetAsync(id);
            Console.WriteLine($"DateTime: {DateTime.Now}, Experiment Name:'{jobResource.Data.Properties.ExperimentName}' status returned: '{jobResource.Data.Properties.Status}'.");

            if (jobResource.Data.Properties.Status != JobStatus.Completed && jobResource.Data.Properties.Status != JobStatus.Failed && jobResource.Data.Properties.Status != JobStatus.Canceled)
            {
                await Task
                    .Delay(SleepIntervalMs)
                    .ConfigureAwait(false);
            }
        }
        while (jobResource.Data.Properties.Status != JobStatus.Completed && jobResource.Data.Properties.Status != JobStatus.Failed && jobResource.Data.Properties.Status != JobStatus.Canceled);

        return jobResource;
    }
    // </WaitForJobToFinishAsync>

    /// <summary>
    /// Format the exception
    /// </summary>
    /// <param name="exception"></param>
    /// <returns></returns>
    public static string FlattenException(Exception exception)
    {
        var stringBuilder = new StringBuilder();

        while (exception != null)
        {
            stringBuilder.AppendLine(exception.Message);
            stringBuilder.AppendLine(exception.StackTrace);

            exception = exception.InnerException;
        }
        return stringBuilder.ToString();
    }


    /// <summary>
    /// Generates a random string with the given length
    /// </summary>
    /// <param name="size">Size of the string</param>
    /// <param name="lowerCase">If true, generate lowercase string</param>
    /// <returns>Random string</returns>
    public static string RandomString(int size, bool lowerCase)
    {
        StringBuilder builder = new StringBuilder();
        Random random = new Random();
        char ch;
        for (int i = 0; i < size; i++)
        {
            ch = Convert.ToChar(Convert.ToInt32(Math.Floor(26 * random.NextDouble() + 65)));
            builder.Append(ch);
        }
        if (lowerCase)
            return builder.ToString().ToLower();
        return builder.ToString();
    }

    /// <summary>
    /// Invoke BatchDeployment
    /// </summary>
    /// <param name="uri"></param>
    /// <param name="jsonObject"></param>
    /// <param name="deploymentName"></param>
    /// <returns></returns>
    // <BatchInvokeRequestResponseService>
    public static async Task BatchInvokeRequestResponseService1(Uri uri, JObject jsonObject, string deploymentName)
    {
        var handler = new HttpClientHandler()
        {
            ClientCertificateOptions = ClientCertificateOption.Manual,
            ServerCertificateCustomValidationCallback =
                    (httpRequestMessage, cert, cetChain, policyErrors) => { return true; }
        };
        using (var client = new HttpClient(handler))
        {
            var scoreRequest = ToCollections(jsonObject);
            /*
                When you enable token authentication for a real-time endpoint, a user must provide an Azure Machine Learning JWT token to the web service to access it.

                You can use the following Azure Machine Learning CLI command to retrieve the access token:

                az ml online-endpoint get-credentials -n my-endpoint
            */

            // Replace this with the API key for the web service
            const string apiKey = "<< ApiKey TO BE UPDATED >>";
            if (string.IsNullOrEmpty(apiKey))
            {
                Console.WriteLine("apiKey is Null or empty. Please update the apiKey in the code");
            }
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
            client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
            client.DefaultRequestHeaders.Add("azureml-model-deployment", deploymentName);
            client.BaseAddress = uri;

            var requestString = JsonConvert.SerializeObject(scoreRequest);
            var content = new StringContent(requestString);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            HttpResponseMessage response = await client.PostAsync("", content);

            if (response.IsSuccessStatusCode)
            {
                string result = await response.Content.ReadAsStringAsync();
                Console.WriteLine("Result: {0}", result);
            }
            else
            {
                Console.WriteLine(string.Format("The request failed with status code: {0}", response.StatusCode));

                // Print the headers-they include the requert ID and the timestamp,
                // which are useful for debugging the failure
                Console.WriteLine(string.Format("Headers: {0}", response.Headers.ToString()));

                string responseContent = await response.Content.ReadAsStringAsync();
                Console.WriteLine(string.Format("responseContent: {0}", responseContent));
            }
        }
    }
    // </BatchInvokeRequestResponseService>

    /// <summary>
    /// 
    /// </summary>
    /// <param name="uri"></param>
    /// <param name="jsonObject"></param>
    /// <returns></returns>
    // <InvokeRequestResponseService>
    public static async Task InvokeRequestResponseService(Uri uri, JObject jsonObject)
    {
        var handler = new HttpClientHandler()
        {
            ClientCertificateOptions = ClientCertificateOption.Manual,
            ServerCertificateCustomValidationCallback =
                    (httpRequestMessage, cert, cetChain, policyErrors) => { return true; }
        };
        using (var client = new HttpClient(handler))
        {
            var scoreRequest = ToCollections(jsonObject);
            /*
                When you enable token authentication for a real-time endpoint, a user must provide an Azure Machine Learning JWT token to the web service to access it.

                You can use the following Azure Machine Learning CLI command to retrieve the access token:

                az ml online-endpoint get-credentials -n my-endpoint
            */

            // Replace this with the API key for the web service

            const string apiKey = "<< ApiKey TO BE UPDATED >>";

            if (string.IsNullOrEmpty(apiKey))
            {
                Console.WriteLine("apiKey is Null or empty. Please update the apiKey in the code");
            }
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
            client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
            client.BaseAddress = uri;

            var requestString = JsonConvert.SerializeObject(scoreRequest);
            var content = new StringContent(requestString);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            HttpResponseMessage response = await client.PostAsync("", content);

            if (response.IsSuccessStatusCode)
            {
                string result = await response.Content.ReadAsStringAsync();
                Console.WriteLine("Result: {0}", result);
            }
            else
            {
                Console.WriteLine(string.Format("The request failed with status code: {0}", response.StatusCode));

                // Print the headers - they include the requert ID and the timestamp,
                // which are useful for debugging the failure
                Console.WriteLine(string.Format("Headers: {0}", response.Headers.ToString()));

                string responseContent = await response.Content.ReadAsStringAsync();
                Console.WriteLine(string.Format("responseContent: {0}", responseContent));
            }
        }
    }
    // </InvokeRequestResponseService>

    /// <summary>
    /// Convert JObject into Dictionary<string, object>
    /// </summary>
    /// <param name="o"></param>
    /// <returns></returns>
    // <ToCollections>
    public static object ToCollections(object o)
    {
        if (o is JObject jo) return jo.ToObject<IDictionary<string, object>>().ToDictionary(k => k.Key, v => ToCollections(v.Value));
        if (o is JArray ja) return ja.ToObject<List<object>>().Select(ToCollections).ToList();
        return o;
    }
    // </ToCollections>
}
