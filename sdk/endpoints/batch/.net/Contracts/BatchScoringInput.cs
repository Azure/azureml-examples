// <copyright file="BatchScoringInput.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace BatchInferencingSamples
{
    public class BatchScoringInput
    {
        /// <summary>
        /// The type of input, either or file or a folder.
        /// </summary>
        public JobInputType JobInputType { get; set; }

        /// <summary>
        /// The URI, either an AML datastore, asset or open dataset.
        /// </summary>
        public Uri? Uri { get; set; }
    }
}
