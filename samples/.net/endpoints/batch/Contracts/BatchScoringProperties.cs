// <copyright file="BatchScoringProperties.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace BatchInferencingSamples
{
    internal class BatchScoringProperties
    {
        /// <summary>
        /// Settings of the source data.
        /// </summary>
        public Dictionary<string, BatchScoringInput> InputData { get; set; } = new Dictionary<string, BatchScoringInput>();
        
        /// <summary>
        /// Settings of the output data.
        /// </summary>
        public Dictionary<string, BatchScoringOutput> OutputData { get; set; } = new Dictionary<string, BatchScoringOutput>();
    }
}
