// <copyright file="BatchScoringInput.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace Microsoft.Azure.MachineLearning.Samples.BatchInferencing
{
    public class BatchScoringInput
    {
        /// <summary>
        /// Gets or sets the type of input, either or file or a folder.
        /// </summary>
        public JobInputType JobInputType { get; set; }

        /// <summary>
        /// Gets or sets the URI to the input data. Either an AML datastore, asset or open dataset.
        /// </summary>
        public Uri? Uri { get; set; }
    }
}
