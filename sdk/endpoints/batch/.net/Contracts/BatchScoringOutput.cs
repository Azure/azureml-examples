// <copyright file="BatchScoringOutput.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace BatchInferencingSamples
{
    public class BatchScoringOutput
    {
        /// <summary>
        /// The type of output created by the inferencing job.
        /// </summary>
        public JobOutputType JobOutputType { get; set; }
        
        /// <summary>
        /// The URI pointing to the output location. Supports AML Datastore.
        /// </summary>
        public Uri? Uri { get; set; }
    }
}
