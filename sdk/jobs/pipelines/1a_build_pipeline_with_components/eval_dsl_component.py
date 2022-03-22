
from azure.ml import dsl
from azure.ml.entities import Environment
from azure.ml.dsl._types import DataInput, DataOutput

conda_env = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04"
)

@dsl.command_component(
    name="Eval",
    display_name="Eval",
    description="A dummy eval component defined by dsl component.",
    version="0.0.1",
    environment=conda_env,
    # specify your code folder, default code folder is current file's parent
    # code='.'    
)
def eval_func(
    scoring_result: DataInput,
    eval_output: DataOutput,
):
    from pathlib import Path
    from datetime import datetime
    print ("hello evaluation world...")

    lines = [
        f'Scoring result path: {scoring_result}',
        f'Evaluation output path: {eval_output}',
    ]

    for line in lines:
        print(line)

    # Evaluate the incoming scoring result and output evaluation result.
    # Here only output a dummy file for demo.
    curtime = datetime.now().strftime("%b-%d-%Y %H:%M:%S")
    eval_msg = f"Eval done at {curtime}\n"
    (Path(eval_output) / 'eval_result.txt').write_text(eval_msg)
