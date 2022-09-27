// <copyright file="BatchScoringOutput.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace Microsoft.Azure.MachineLearning.Samples.BatchInferencing
{
    public class BatchScoringOutput
    {
        /// <summary>
        /// Gets or sets the type of output created by the inferencing job.
        /// </summary>
        public JobOutputType JobOutputType { get; set; }
        
        /// <summary>
        /// Gets or sets the URI pointing to the output location. Supports AML Datastore.
        /// </summary>
        public Uri? Uri { get; set; }
    }
}
