// <copyright file="BatchScoringOutput.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace BatchInferencingSamples
{
    public class BatchScoringOutput
    {
        public JobOutputType JobOutputType { get; set; }
        public Uri? Uri { get; set; }
    }
}
