from azureml.core import Workspace, Environment, ScriptRunConfig, Experiment

ws = Workspace.from_config()

env = Environment.from_conda_specification(name = 'myenv', file_path = 'environment.yml')

src = ScriptRunConfig(source_directory='.', script='train.py')
src.run_config.environment = env
src.run_config.target = 'local'

run = Experiment(ws, "sklearn-diabetes").submit(src)
run.wait_for_completion(show_output=True)
