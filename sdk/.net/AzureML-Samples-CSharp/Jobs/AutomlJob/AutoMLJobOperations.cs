using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;

namespace Azure.MachineLearning.Samples.Jobs.AutomlJob;

internal class AutoMLJobOperations
{

    /// <summary>
    /// If the specified CommandJob exists, get that CommandJob.
    /// If it does not exist, creates a new CommandJob. 
    /// </summary>
    /// <param name="resourceGroupName"></param>
    /// <param name="workspaceName"></param>
    /// <param name="id"></param>
    /// <param name="experimentName"></param>
    /// <returns></returns>
    // <SubmitAutoMLTextNerAsync>
    public static async Task<MachineLearningJobResource> SubmitAutoMLTextNerAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string id,
        string experimentName,
        string environmentId,
        string computeId)
    {
        Console.WriteLine("Creating an AutoML TextNer...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        var trainData = new MLTableJobInput(new Uri("https://raw.githubusercontent.com/Azure/azureml-examples/main/cli/jobs/automl-standalone-jobs/cli-automl-text-ner-conll/training-mltable-folder"))
        {
            Mode = InputDeliveryMode.ReadOnlyMount,
            Description = "Train data",
        };

        //var trainData1 = new MLTableJobInput(new Uri("azureml:mydata01:1"))
        //{
        //    Mode = InputDeliveryMode.ReadOnlyMount,
        //    Description = "Train data",
        //};

        var validationData = new MLTableJobInput(new Uri("https://raw.githubusercontent.com/Azure/azureml-examples/main/cli/jobs/automl-standalone-jobs/cli-automl-text-ner-conll/validation-mltable-folder"))
        {
            Mode = InputDeliveryMode.ReadOnlyMount,
            Description = "Validation data",
        };

        var trainingDataSettings = new TrainingDataSettings(trainData);

        AutoMLVertical taskDetails = new TextNer
        {
            LogVerbosity = LogVerbosity.Debug,
            DataSettings = new NlpVerticalDataSettings("label", trainingDataSettings)
            {
                // The combined validation and test sizes must be between 0 and 0.99 inclusive when specified. Test split is not supported for task type: text-ner.
                ValidationData = new NlpVerticalValidationDataSettings()
                {
                    Data = validationData,
                    // Validation size must be between 0.01 and 0.99 inclusive when specified. Test size must be between 0 and 0.99 inclusive when specified. Test split is not supported for task type: text-ner
                    ValidationDataSize = 0.20,
                },
                // Test split is not supported for task type: text-ner.
                //TestData = new TestDataSettings()
                //{
                //    Data = testData,
                //    // Test size must be between 0 and 0.99 inclusive when specified. Test split is not supported for task type: text-ner
                //    TestDataSize = 0.38,
                //},
            },
            FeaturizationDatasetLanguage = "US",
            LimitSettings = new NlpVerticalLimitSettings
            {
                MaxTrials = 2,
                Timeout = TimeSpan.FromMinutes(1800),
                MaxConcurrentTrials = 2
            },

        };
        var autoMLJob = new AutoMLJob(taskDetails)
        {
            ExperimentName = experimentName,
            DisplayName = "AutoMLJobTextNer-" + Guid.NewGuid().ToString("n").Substring(0, 6),
            EnvironmentId = environmentId,
            IsArchived = false,
            ComputeId = computeId,
            Resources = new ResourceConfiguration
            {
                InstanceCount = 2,
            },
            Properties = new Dictionary<string, string>
                {
                    { "property-name", "property-value" },
                },
            Tags = new Dictionary<string, string>
                {
                    { "tag-name", "tag-value" },
                },

            //Environment variables included in the job.
            EnvironmentVariables = new Dictionary<string, string>()
                {
                    { "env-var", "env-var-value" }
                },
            Description = "This is a description of test AutoMLJob for TextNer",

        };
        MachineLearningJobData data = new MachineLearningJobData(autoMLJob);
        ArmOperation<MachineLearningJobResource> jobOperation = await ws.GetMachineLearningJobs().CreateOrUpdateAsync(WaitUntil.Completed, id, data);
        MachineLearningJobResource jobResource = jobOperation.Value;
        Console.WriteLine($"JobCreateOrUpdateOperation {jobResource.Data.Id} created.");
        return jobResource;
    }
    // </SubmitAutoMLTextNerAsync>


    /// <summary>
    /// If the specified AutoMLForecast job exists, get that AutoMLForecast job.
    /// If it does not exist, creates a new AutoMLForecast job. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="id"></param>
    /// <param name="experimentName"></param>
    /// <param name="environmentId"></param>
    /// <param name="computeId"></param>
    /// <returns></returns>
    // <SubmitAutoMLForecastAsync>
    public static async Task<MachineLearningJobResource> SubmitAutoMLForecastAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string id,
        string experimentName,
        string environmentId,
        string computeId)
    {
        Console.WriteLine("Creating an AutoML Forecast job...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        var trainData = new MLTableJobInput(new Uri("https://raw.githubusercontent.com/Azure/azureml-examples/main/cli/jobs/automl-standalone-jobs/cli-automl-forecasting-task-energy-demand/training-mltable-folder"))
        {
            Mode = InputDeliveryMode.ReadOnlyMount,
            Description = "Train data",
        };
        var validationData = new MLTableJobInput(new Uri("https://raw.githubusercontent.com/Azure/azureml-examples/main/cli/jobs/automl-standalone-jobs/cli-automl-forecasting-task-energy-demand/validation-mltable-folder"))
        {
            Mode = InputDeliveryMode.ReadOnlyMount,
            Description = "Validation data",
        };
        var trainingDataSettings = new TrainingDataSettings(trainData);

        AutoMLVertical taskDetails = new Forecasting
        {
            LogVerbosity = LogVerbosity.Debug,
            PrimaryMetric = ForecastingPrimaryMetrics.NormalizedRootMeanSquaredError,
            AllowedModels = new List<ForecastingModels>() { ForecastingModels.ExponentialSmoothing, ForecastingModels.GradientBoosting },
            BlockedModels = new List<ForecastingModels>() { ForecastingModels.Average },
            FeaturizationSettings = new TableVerticalFeaturizationSettings
            {
                EnableDnnFeaturization = false,
                Mode = FeaturizationMode.Auto,
            },
            ForecastingSettings = new ForecastingSettings
            {
                CountryOrRegionForHolidays = "US",
                TimeColumnName = "timeStamp",
                ShortSeriesHandlingConfig = ShortSeriesHandlingConfiguration.Auto,
                // Frequency = "1",
                FeatureLags = FeatureLags.Auto,
                TargetAggregateFunction = TargetAggregationFunction.Mean,
                //// Time column name is present in the grain columns. Please remove it from grain list.
                //TimeSeriesIdColumnNames = new List<string>() { "temp" },
                UseStl = UseStl.Season,
                // Number of periods between the origin time of one CV fold and the next fold.
                CvStepSize = 1,
                Seasonality = new AutoSeasonality(),
                ForecastHorizon = new CustomForecastHorizon(2),
                TargetLags = new CustomTargetLags(new List<int> { 1 }),
                TargetRollingWindowSize = new AutoTargetRollingWindowSize(),

            },
            DataSettings = new TableVerticalDataSettings("precip", trainingDataSettings)
            {
                ValidationData = new TableVerticalValidationDataSettings()
                {
                    Data = validationData,
                    // ValidationDataSize = .05,
                    NCrossValidations = new CustomNCrossValidations(2),
                },
                //// Test split is not supported for task type: forecasting. 
                //TestData = new TestDataSettings()
                //{
                //    Data = testData,
                //    TestDataSize = .20,
                //},
            },

            TrainingSettings = new TrainingSettings
            {
                EnableDnnTraining = false,
                EnableStackEnsemble = false,
                EnableVoteEnsemble = true,
                EnsembleModelDownloadTimeout = TimeSpan.FromSeconds(250),
                StackEnsembleSettings = new StackEnsembleSettings()
                {
                    StackMetaLearnerTrainPercentage = 0.12,
                    StackMetaLearnerType = StackMetaLearnerType.LightGBMRegressor
                },
                EnableModelExplainability = false,
                EnableOnnxCompatibleModels = false,
            },
            LimitSettings = new TableVerticalLimitSettings
            {
                MaxTrials = 5,
                Timeout = TimeSpan.FromMinutes(1800),
                MaxConcurrentTrials = 2,
                EnableEarlyTermination = true,
                ExitScore = 0.90,
                MaxCoresPerTrial = -1,
                TrialTimeout = TimeSpan.FromMinutes(1200),
            },
        };
        // AutoMLVertical
        var autoMLJob = new AutoMLJob(taskDetails)
        {
            ExperimentName = experimentName,
            DisplayName = "AutoMLJob forecasting-" + Guid.NewGuid().ToString("n").Substring(0, 6),
            EnvironmentId = environmentId,
            IsArchived = false,
            ComputeId = computeId,
            Resources = new ResourceConfiguration
            {
                InstanceCount = 2,
            },
            Properties = new Dictionary<string, string>
                {
                    { "property-name", "property-value" },
                },
            Tags = new Dictionary<string, string>
                {
                    { "tag-name", "tag-value" },
                },

            // Environment variables included in the job.
            EnvironmentVariables = new Dictionary<string, string>()
                {
                    { "env-var", "env-var-value" }
                },
            Description = "This is a description of test AutoMLJob for forecast",
        };

        MachineLearningJobData MachineLearningJobData = new MachineLearningJobData(autoMLJob);
        ArmOperation<MachineLearningJobResource> jobOperation = await ws.GetMachineLearningJobs().CreateOrUpdateAsync(WaitUntil.Completed, id, MachineLearningJobData);
        MachineLearningJobResource jobResource = jobOperation.Value;
        Console.WriteLine($"JobCreateOrUpdateOperation {jobResource.Data.Id} created.");
        return jobResource;
    }
    // </SubmitAutoMLForecastAsync>


    /// <summary>
    /// If the specified AutoMLTextClassification exists, get that AutoMLTextClassification job.
    /// If it does not exist, creates a new AutoMLTextClassification job. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="id"></param>
    /// <param name="experimentName"></param>
    /// <param name="environmentId"></param>
    /// <param name="computeId"></param>
    /// <returns></returns>
    // <SubmitAutoMLTextClassificationAsync>
    public static async Task<MachineLearningJobResource> SubmitAutoMLTextClassificationAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string id,
        string experimentName,
        string environmentId,
        string computeId)
    {
        Console.WriteLine("Creating an AutoML TextClassification job...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        var trainData = new MLTableJobInput(new Uri("https://raw.githubusercontent.com/Azure/azureml-examples/main/cli/jobs/automl-standalone-jobs/cli-automl-text-classification-newsgroup/training-mltable-folder"))
        {
            Mode = InputDeliveryMode.ReadOnlyMount,
            Description = "Train data",
        };
        var validationData = new MLTableJobInput(new Uri("https://raw.githubusercontent.com/Azure/azureml-examples/main/cli/jobs/automl-standalone-jobs/cli-automl-text-classification-newsgroup/validation-mltable-folder"))
        {
            Mode = InputDeliveryMode.ReadOnlyMount,
            Description = "Validation data",
        };
        var trainingData = new TrainingDataSettings(trainData);

        AutoMLVertical taskDetails = new TextClassification()
        {
            LogVerbosity = LogVerbosity.Info,
            LimitSettings = new NlpVerticalLimitSettings()
            {
                MaxConcurrentTrials = 2,
                MaxTrials = 10,
                // The max run duration in ISO 8601 format, after which the job will be cancelled.
                // Experiment timeout needs to be between 00:15:00 and 7.00:00:00 if set.
                //Timeout = TimeSpan.FromSeconds(1000),
            },
            PrimaryMetric = ClassificationPrimaryMetrics.AUCWeighted,
            DataSettings = new NlpVerticalDataSettings("y", trainingData)
            {
                // Validation size must be between 0.01 and 0.99 inclusive when specified.
                ValidationData = new NlpVerticalValidationDataSettings()
                {
                    Data = validationData,
                    ValidationDataSize = .45,
                },
                //// Test size must be between 0 and 0.99 inclusive when specified.
                //TestData = new TestDataSettings()
                //{
                //    Data = testData,
                //    TestDataSize = .20,
                //}
            },
            FeaturizationDatasetLanguage = "en",
        };

        var autoMLJob = new AutoMLJob(taskDetails)
        {
            ExperimentName = experimentName,
            DisplayName = "AutoMLJob TextClassification-" + Guid.NewGuid().ToString("n").Substring(0, 6),
            EnvironmentId = environmentId,
            IsArchived = false,
            ComputeId = computeId,
            Resources = new ResourceConfiguration
            {
                InstanceCount = 3,
            },
            Properties = new Dictionary<string, string>
                {
                    { "property-name", "property-value" },
                },
            Tags = new Dictionary<string, string>
                {
                    { "tag-name", "tag-value" },
                },
            EnvironmentVariables = new Dictionary<string, string>()
                {
                    { "env-var", "env-var-value" }
                },
            Description = "This is a description of test AutoMLJob for TextClassification",
        };

        MachineLearningJobData MachineLearningJobData = new MachineLearningJobData(autoMLJob);
        ArmOperation<MachineLearningJobResource> jobOperation = await ws.GetMachineLearningJobs().CreateOrUpdateAsync(WaitUntil.Completed, id, MachineLearningJobData);
        MachineLearningJobResource jobResource = jobOperation.Value;
        Console.WriteLine($"JobCreateOrUpdateOperation {jobResource.Data.Id} created.");
        return jobResource;
    }
    // </SubmitAutoMLTextClassificationAsync>


    /// <summary>
    /// If the specified AutoMLImageClassification exists, get that ImageClassification job.
    /// If it does not exist, creates a new AutoMLImageClassification job. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="id"></param>
    /// <param name="experimentName"></param>
    /// <param name="environmentId"></param>
    /// <param name="computeId"></param>
    /// <returns></returns>
    // <SubmitAutoMLImageClassificationAsync>
    public static async Task<MachineLearningJobResource> SubmitAutoMLImageClassificationAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string id,
        string experimentName,
        string environmentId,
        string computeId)
    {
        Console.WriteLine("Creating an AutoML ImageClassification job...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        // Upload the MLTable in the default workspaceblobstore.
        var trainData = new MLTableJobInput(new Uri("azureml://datastores/workspaceblobstore/paths/training-mltable-folder"))
        {
            Mode = InputDeliveryMode.EvalMount,
            Description = "Train data",
        };
        var validationData = new MLTableJobInput(new Uri("azureml://datastores/workspaceblobstore/paths/validation-mltable-folder"))
        {
            Mode = InputDeliveryMode.EvalDownload,
            Description = "Validation data",
        };

        var trainingData = new TrainingDataSettings(trainData);

        ImageVerticalDataSettings dataSettings = new ImageVerticalDataSettings("label", trainingData)
        {
            // TargetColumnName = "label",
            //TestData = new TestDataSettings()
            //{
            //    Data = testData,
            //    TestDataSize = .20,
            //},
            ValidationData = new ImageVerticalValidationDataSettings()
            {
                Data = validationData,
                // Validation size must be between 0.01 and 0.99 inclusive when specified. Test size must be between 0 and 0.99 inclusive when specified. Test split is not supported for task type: text-ner
                ValidationDataSize = 0.20,
            },
        };
        ImageLimitSettings limitSettings = new ImageLimitSettings()
        {
            MaxConcurrentTrials = 2,
            MaxTrials = 10,
            Timeout = TimeSpan.FromHours(2)
        };

        ImageSweepLimitSettings sweepLimits = new ImageSweepLimitSettings() { MaxConcurrentTrials = 4, MaxTrials = 20 };
        SamplingAlgorithmType samplingAlgorithm = SamplingAlgorithmType.Random;
        List<ImageModelDistributionSettingsClassification> searchSpaceList = new List<ImageModelDistributionSettingsClassification>()
            {
                new ImageModelDistributionSettingsClassification()
                {
                    ModelName = "choice('vitb16r224', 'vits16r224')",
                    LearningRate = "uniform(0.001, 0.01)",
                    NumberOfEpochs = "choice(15, 30)",
                },
                new ImageModelDistributionSettingsClassification()
                {
                    ModelName = "choice('seresnext', 'resnet50')",
                    LearningRate = "uniform(0.001, 0.01)",
                    NumberOfEpochs = "choice(0, 2)",
                }
            };

        AutoMLVertical taskDetails = new ImageClassification(dataSettings, limitSettings)
        {
            LogVerbosity = LogVerbosity.Info,
            PrimaryMetric = ClassificationPrimaryMetrics.Accuracy,
            SweepSettings = new ImageSweepSettings(sweepLimits, samplingAlgorithm)
            {
                EarlyTermination = new BanditPolicy() { SlackFactor = 0.2f, EvaluationInterval = 3 },
            },
            SearchSpace = searchSpaceList,
        };

        var autoMLJob = new AutoMLJob(taskDetails)
        {
            ExperimentName = experimentName,
            DisplayName = "AutoMLJob ImageClassification-" + Guid.NewGuid().ToString("n").Substring(0, 6),
            EnvironmentId = environmentId,
            IsArchived = false,
            ComputeId = computeId,
            Resources = new ResourceConfiguration
            {
                InstanceCount = 3,
            },
            Properties = new Dictionary<string, string>
                {
                    { "property-name", "property-value" },
                },
            Tags = new Dictionary<string, string>
                {
                    { "tag-name", "tag-value" },
                },
            EnvironmentVariables = new Dictionary<string, string>()
                {
                    { "env-var", "env-var-value" }
                },
            Description = "This is a description of test AutoMLJob for multi-class Image classification job using fridge items dataset",
        };

        MachineLearningJobData MachineLearningJobData = new MachineLearningJobData(autoMLJob);
        ArmOperation<MachineLearningJobResource> jobOperation = await ws.GetMachineLearningJobs().CreateOrUpdateAsync(WaitUntil.Completed, id, MachineLearningJobData);
        MachineLearningJobResource jobResource = jobOperation.Value;
        Console.WriteLine($"JobCreateOrUpdateOperation {jobResource.Data.Id} created.");
        return jobResource;
    }
    // </SubmitAutoMLImageClassificationAsync>



    /// <summary>
    /// If the specified AutoMLImageObjectDetection exists, get that AutoMLImageObjectDetection job.
    /// If it does not exist, creates a new ImageObjectDetection job. 
    /// </summary>
    /// <param name="resourceGroup"></param>
    /// <param name="workspaceName"></param>
    /// <param name="id"></param>
    /// <param name="experimentName"></param>
    /// <param name="environmentId"></param>
    /// <param name="computeId"></param>
    /// <returns></returns>
    // <SubmitAutoMLImageObjectDetectionAsync>
    public static async Task<MachineLearningJobResource> SubmitAutoMLImageObjectDetectionAsync(
        ResourceGroupResource resourceGroup,
        string workspaceName,
        string id,
        string experimentName,
        string environmentId,
        string computeId)
    {
        Console.WriteLine("Creating an AutoML ImageObjectDetection job...");
        MachineLearningWorkspaceResource ws = await resourceGroup.GetMachineLearningWorkspaces().GetAsync(workspaceName);

        // Upload the MLTable in the default workspaceblobstore.
        var trainData = new MLTableJobInput(new Uri("azureml://datastores/workspaceblobstore/paths/training-mltable-folder-od"))
        {
            Mode = InputDeliveryMode.ReadOnlyMount,
            Description = "Train data",
        };
        var validationData = new MLTableJobInput(new Uri("azureml://datastores/workspaceblobstore/paths/validation-mltable-folder-od"))
        {
            Mode = InputDeliveryMode.ReadOnlyMount,
            Description = "Validation data",
        };
        var trainingData = new TrainingDataSettings(trainData);

        ImageVerticalDataSettings dataSettings = new ImageVerticalDataSettings("label", trainingData)
        {
            // TargetColumnName = "Label",
            //TestData = new TestDataSettings()
            //{
            //    Data = testData,
            //    TestDataSize = .20,
            //},
            ValidationData = new ImageVerticalValidationDataSettings()
            {
                Data = validationData,
                // Validation size must be between 0.01 and 0.99 inclusive when specified. Test size must be between 0 and 0.99 inclusive when specified. Test split is not supported for task type: text-ner
                ValidationDataSize = 0.20,
            },
        };
        ImageLimitSettings limitSettings = new ImageLimitSettings()
        {
            MaxConcurrentTrials = 1,
            MaxTrials = 2,
            Timeout = TimeSpan.FromHours(2)
        };

        ImageSweepLimitSettings sweepLimits = new ImageSweepLimitSettings() { MaxConcurrentTrials = 2, MaxTrials = 10 };
        SamplingAlgorithmType samplingAlgorithm = SamplingAlgorithmType.Random;
        List<ImageModelDistributionSettingsObjectDetection> searchSpaceList = new List<ImageModelDistributionSettingsObjectDetection>()
            {
                new ImageModelDistributionSettingsObjectDetection()
                {
                    ModelName = "yolov5",
                    EarlyStopping = "true",
                    LearningRate = "uniform(0.0001, 0.01)",
                    ModelSize = "choice('small', 'medium')",

                },
                new ImageModelDistributionSettingsObjectDetection()
                {
                    ModelName = "fasterrcnn_resnet50_fpn",
                    LearningRate = "uniform(0.0001, 0.001)",
                    Optimizer = "choice('sgd', 'adam', 'adamw')",
                    ModelSize = "choice('small', 'medium')",
                    MinSize = "choice(600, 800)",

                },
            };

        AutoMLVertical taskDetails = new ImageObjectDetection(dataSettings, limitSettings)
        {
            LogVerbosity = LogVerbosity.Info,
            PrimaryMetric = ObjectDetectionPrimaryMetrics.MeanAveragePrecision,
            SweepSettings = new ImageSweepSettings(sweepLimits, samplingAlgorithm)
            {
                EarlyTermination = new BanditPolicy() { SlackFactor = 0.2f, EvaluationInterval = 2, DelayEvaluation = 6 },
            },
            SearchSpace = searchSpaceList,
            // ModelSettings = modelSettings,
        };

        var autoMLJob = new AutoMLJob(taskDetails)
        {
            ExperimentName = experimentName,
            DisplayName = "AutoMLJob ImageObjectDetection-" + Guid.NewGuid().ToString("n").Substring(0, 6),
            EnvironmentId = environmentId,
            IsArchived = false,
            ComputeId = computeId,
            Resources = new ResourceConfiguration
            {
                InstanceCount = 3,
            },
            Properties = new Dictionary<string, string>
                {
                    { "property-name", "property-value" },
                },
            Tags = new Dictionary<string, string>
                {
                    { "tag-name", "tag-value" },
                },
            EnvironmentVariables = new Dictionary<string, string>()
                {
                    { "env-var", "env-var-value" }
                },
            Description = "This is a description of test AutoMLJob for ImageObjectDetection",
        };

        MachineLearningJobData MachineLearningJobData = new MachineLearningJobData(autoMLJob);
        ArmOperation<MachineLearningJobResource> jobOperation = await ws.GetMachineLearningJobs().CreateOrUpdateAsync(WaitUntil.Completed, id, MachineLearningJobData);
        MachineLearningJobResource jobResource = jobOperation.Value;
        Console.WriteLine($"JobCreateOrUpdateOperation {jobResource.Data.Id} created.");
        return jobResource;
    }
    // </SubmitAutoMLImageObjectDetectionAsync>

}
