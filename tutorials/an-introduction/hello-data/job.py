from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset

# get workspace
ws = Workspace.from_config()

# create experiment
exp = Experiment(
    workspace=ws, 
    name="an-introduction-hello-world-tutorial"
)

# get a curated environment
env = Environment.get(ws, "AzureML-Tutorial")

# define dataset
ds = Dataset.File.from_files("https://azuremlexamples.blob.core.windows.net/datasets/iris.csv")
# add data location to arguments for script
arguments = ["--data-path", ds.as_mount()]

# define script run config
src = ScriptRunConfig(
    source_directory="src",
    script="hello.py",
    arguments=arguments,
    compute_target="cpu-cluster",
    environment=env
)

# submit job
run = exp.submit(src)
run.wait_for_completion(show_output=True)
