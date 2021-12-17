"""
TODO
"""

from azure.ml.component import dsl
from shrike.pipeline.pipeline_helper import AMLPipelineHelper


class FederatedSubgraph(AMLPipelineHelper):
    def build(self, config):
        component = self.component_load("aggregatemodelweights")

        @dsl.pipeline(
            name="federated-subgraph",
            default_datastore=config.compute.compliant_datastore,
        )
        def demosubgraph_pipeline_function(
            input_data_01,
            input_data_02,
            input_data_03,
        ):
            component_step_1 = component(
                input_data_01=input_data_01,
                input_data_02=input_data_02,
            )

            self.apply_smart_runsettings(component_step_1, gpu=True)

            component_step_2 = component(
                input_data_01=component_step_1.outputs.results,
                input_data_02=input_data_03,
            )

            self.apply_smart_runsettings(component_step_2, gpu=True)

            return {"subgraph_results": component_step_2.outputs.results}

        # finally return the function itself to be built by helper code
        return demosubgraph_pipeline_function
