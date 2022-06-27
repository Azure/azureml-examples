// <copyright file="JobOutputType.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

using System.Text.Json.Serialization;

namespace BatchInferencingSamples
{
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum JobOutputType
    {
        UriFile
    }
}
