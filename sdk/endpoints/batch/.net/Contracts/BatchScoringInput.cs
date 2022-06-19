// <copyright file="BatchScoringInput.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace BatchInferencingSamples
{
    public class BatchScoringInput
    {
        public JobInputType JobInputType { get; set; }
        public Uri? Uri { get; set; }
    }
}
