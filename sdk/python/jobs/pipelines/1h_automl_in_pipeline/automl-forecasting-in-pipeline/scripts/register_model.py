import argparse
import os
import uuid
import shutil
from azureml.core.model import Model
from azureml.core.run import Run, _OfflineRun
from azureml.core import Workspace


def get_model_path(model_artifact_path):
    return model_artifact_path.split("/")[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base_name", help="Name of the registered model")
    parser.add_argument("--model_path", help="Path to input model")
    args = parser.parse_args()

    print("Argument 1(model_name): %s" % args.model_base_name)
    print("Argument 2(model_path): %s" % args.model_path)

    run = Run.get_context()
    ws = None
    if type(run) == _OfflineRun:
        ws = Workspace.from_config()
    else:
        ws = run.experiment.workspace

    model = Model.register(
        ws, model_path=args.model_path, model_name=args.model_base_name,
    )
    print("Registered version {0} of model {1}".format(model.version, model.name))
