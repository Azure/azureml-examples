from shrike.pipeline import FederatedPipelineBase, StepOutput
from subgraph import FederatedSubgraph


class DemoFederatedLearning(FederatedPipelineBase):
    @classmethod
    def required_subgraphs(cls):
        return {"demo-subgraph": FederatedSubgraph}

    def preprocess(self, config):
        preprocess_func = self.component_load("preprocessing")
        pipeline_input_dataset = self.dataset_load(
            name=config.democomponent.input_data,
            version=config.democomponent.input_data_version,
        )
        preprocess_step = preprocess_func(
            input_data=pipeline_input_dataset,
            message=config.federated_config.params.msg,
        )
        return StepOutput(preprocess_step, ["results"])

    def train(self, config, input, silo):
        # input = model weights
        train_func = self.component_load("traininsilo")
        input_data = self.dataset_load(silo.params.dataset)
        train_step = train_func(
            input_01=input_data, input_02=input[0], message=silo.params.msg
        )
        return StepOutput(train_step, ["results"])

    def midprocess(self, config, input):
        demo_subgraph = self.subgraph_load("demo-subgraph")
        midprocess_step = demo_subgraph(
            input_data_01=input[0], input_data_02=input[1], input_data_03=input[2]
        )
        return StepOutput(midprocess_step)

    def postprocess(self, config, input):
        postprocess_func = self.component_load("postprocessing")
        postprocess_step = postprocess_func(input_data=input[0])
        return StepOutput(postprocess_step, ["results"])


if __name__ == "__main__":
    DemoFederatedLearning.main()
