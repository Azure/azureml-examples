import os
import shutil
from mldesigner import command_component, Input, Output


@command_component()
def aggregated_train(
    model_1: Input, model_2: Input, agg_output: Output
) -> Output(type="boolean", is_control=True):
    print("Aggregate the train models.")
    shutil.copytree(model_1, os.path.join(agg_output, "model_1"))
    shutil.copytree(model_2, os.path.join(agg_output, "model_2"))
    return True
