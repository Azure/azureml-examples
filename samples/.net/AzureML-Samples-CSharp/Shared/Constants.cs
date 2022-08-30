namespace Azure.MachineLearning.Samples.Shared;

public static class Constants
{
    #region Subscription
    public const string SubscriptionId = "<SubscriptionId GUID to be updated>";
    #endregion

    #region Workspace
    public const string ResourceGroupName = "ml-test-client";
    public const string WorkspaceName = "test-ws2";
    public const string StorageAccountName = "<StorageAccountName to be updated>";
    public const string KeyVaultName = "<KeyVaultName to be updated>";
    public const string AppInsightsName = "<AppInsightsName to be updated>";
    public const string StorageAccountId = $"/subscriptions/{SubscriptionId}/resourceGroups/{ResourceGroupName}/providers/Microsoft.Storage/storageAccounts/{StorageAccountName}";
    public const string KeyVaultId = $"/subscriptions/{SubscriptionId}/resourceGroups/{ResourceGroupName}/providers/Microsoft.KeyVault/vaults/{StorageAccountName}";
    public const string ApplicationInsightsId = $"/subscriptions/{SubscriptionId}/resourceGroups/{ResourceGroupName}/providers/Microsoft.Insights/components/{AppInsightsName}";
    #endregion

    #region AKSCompute
    public const string AKSComputeName = "myakscomp01";
    public const string ClusterFqdn = "testaks0101";
    #endregion

    #region AMLCompute
    public const string AMLComputeName = "test-cpu-cluster";
    public const string ComputeId = $"/subscriptions/{SubscriptionId}/resourceGroups/{ResourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{WorkspaceName}/computes/{AMLComputeName}";
    #endregion

    #region Code
    public const string CodeName = "mytestcode";
    public const string CodeVersion = "1";
    // URI where the code artifact has been uploaded
    public const string CodeUri = "https://testes0174657236803.blob.core.windows.net/azureml-blobstore-de12e8fd-da9a-43b4-ae81-c972bd73beae/";
    #endregion

    #region Component
    public const string ComponentName = "mycomponent1";
    public const string ComponentVersion = "0.0.1";
    public const string ComponentNameForPipeline = "mypipelinecomponent1";
    public const string ComponentVersionForPipeline = "0.0.1";
    #endregion

    #region Data
    public const string DataName = "mydata01";
    public const string DataVersion = "1";
    public const string Datastore = $"/subscriptions/{SubscriptionId}/resourceGroups/{ResourceGroupName}/providers/Microsoft.MachineLearningServices/workspaces/{WorkspaceName}/datastores/workspaceblobstore";
    #endregion

    #region Endpoint
    public const string OnlineEndpointName = "mysdkonlineend002";
    public const string BatchEndpointName = "mysdkbatchend001";
    public const string OnlineDeploymentName = "mysdkonlinedep002";
    public const string BatchDeploymentName = "mysdkbatchdep00101";
    #endregion

    #region Environment
    public const string EnvironmentName = "mytestenvspec1";
    public const string EnvironmentVersion = "1";
    #endregion

    #region Job
    public const string ExperimentName = "TestCommandJob";
    public const string SweepJobExperimentName = "TestSweepJob";
    public const string PipelineJobExperimentName = "TestPipelineJob";
    #endregion

    #region Model
    public const string ModelName = "testmodel";
    public const string ModelVersion = "1";
    public const string ModelUri = $"azureml://subscriptions/{SubscriptionId}/resourceGroups/{ResourceGroupName}/workspaces/{WorkspaceName}/datastores/workspaceblobstore/paths/WebUpload/220609095255-3418361437/sklearn_regression_model.pkl";
    #endregion
}
