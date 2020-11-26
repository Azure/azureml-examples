from azureml.core import Workspace, Experiment, 
from azureml.core import ScriptRunConfig, Environment

# get workspace
ws = Workspace.from_config()

# create experiment
exp = Experiment(
    workspace=ws, 
    name="an-introduction-hello-world-tutorial"
)

# get a curated environment
env = Environment.get(ws, "AzureML-Tutorial")

# define script run config
src = ScriptRunConfig(
    source_directory="src",
    script="hello.py", 
    compute_target="cpu-cluster",
    environment=env
)

# submit job
run = exp.submit(src)
run.wait_for_completion(show_output=True)
