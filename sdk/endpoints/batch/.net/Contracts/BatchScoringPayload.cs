// <copyright file="BatchScoringPayload.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace BatchInferencingSamples
{
    internal class BatchScoringPayload
    {
        /// <summary>
        /// Scoring properties leveraged by the inferencing job.
        /// </summary>
        public BatchScoringProperties Properties { get; set; } = new BatchScoringProperties();
    }
}
