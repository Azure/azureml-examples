// <copyright file="BatchScoringPayload.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace Microsoft.Azure.MachineLearning.Samples.BatchInferencing
{
    internal class BatchScoringPayload
    {
        /// <summary>
        /// Gets or sets the scoring properties leveraged by the inferencing job.
        /// </summary>
        public BatchScoringProperties Properties { get; set; } = new BatchScoringProperties();
    }
}
