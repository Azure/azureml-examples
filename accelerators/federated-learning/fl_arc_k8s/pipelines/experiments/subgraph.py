"""
The AML subgraph for demo graph
"""
# pylint: disable=no-member
# NOTE: because it raises 'dict' has no 'outputs' member in dsl.pipeline construction
import os
import sys

from azure.ml.component import dsl
from shrike.pipeline.pipeline_helper import AMLPipelineHelper

# NOTE: if you need to import from pipelines.*
ACCELERATOR_ROOT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if ACCELERATOR_ROOT_PATH not in sys.path:
    print(f"Adding to path: {ACCELERATOR_ROOT_PATH}")
    sys.path.append(str(ACCELERATOR_ROOT_PATH))


class DemoSubgraph(AMLPipelineHelper):
    """Runnable/reusable pipeline helper class

    This class inherits from AMLPipelineHelper which provides
    helper functions to create reusable production pipelines.
    """

    def build(self, config):
        component = self.component_load("AggregateModelWeights")

        @dsl.pipeline(
            name="demo-subgraph",
            description="The AML pipeline for the demo subgraph",
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

            self.apply_recommended_runsettings(
                "AggregateModelWeights", component_step_1, gpu=True
            )

            component_step_2 = component(
                input_data_01=component_step_1.outputs.results,
                input_data_02=input_data_03,
            )

            self.apply_recommended_runsettings(
                "AggregateModelWeights", component_step_2, gpu=True
            )

            # return {key: output}
            return {"subgraph_results": component_step_2.outputs.results}

        # finally return the function itself to be built by helper code
        return demosubgraph_pipeline_function
