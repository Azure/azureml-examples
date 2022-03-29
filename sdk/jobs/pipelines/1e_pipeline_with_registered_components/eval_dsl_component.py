
from azure.ml import dsl, ArtifactInput, ArtifactOutput

@dsl.command_component(
    name="eval_model",
    display_name="Eval Model",
    description="A dummy eval component defined by dsl component.",
    version="0.0.3",
)
def eval_model(
    scoring_result: ArtifactInput,
    eval_output: ArtifactOutput,
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
