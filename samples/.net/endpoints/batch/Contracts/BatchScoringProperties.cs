// <copyright file="BatchScoringProperties.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace Microsoft.Azure.MachineLearning.Samples.BatchInferencing
{
    internal class BatchScoringProperties
    {
        /// <summary>
        /// Gets or sets the settings of the source data.
        /// </summary>
        public IDictionary<string, BatchScoringInput> InputData { get; set; } = new Dictionary<string, BatchScoringInput>();
        
        /// <summary>
        /// Gets or sets the settings of the output data.
        /// </summary>
        public IDictionary<string, BatchScoringOutput> OutputData { get; set; } = new Dictionary<string, BatchScoringOutput>();
    }
}
