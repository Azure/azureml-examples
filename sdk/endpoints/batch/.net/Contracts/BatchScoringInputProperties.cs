namespace BatchInferencingSamples
{
    internal class BatchScoringInputProperties
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
