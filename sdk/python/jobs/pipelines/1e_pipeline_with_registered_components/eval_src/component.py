from mldesigner import command_component, Input, Output


@command_component(display_name="Eval Model", version="0.0.9")
def eval_model(
    scoring_result: Input(type="uri_folder"), eval_output: Output(type="uri_folder")
):
    """A dummy eval component defined by dsl component."""

    from pathlib import Path
    from datetime import datetime

    print("hello evaluation world...")

    lines = [
        f"Scoring result path: {scoring_result}",
        f"Evaluation output path: {eval_output}",
    ]

    for line in lines:
        print(line)

    # Evaluate the incoming scoring result and output evaluation result.
    # Here only output a dummy file for demo.
    curtime = datetime.now().strftime("%b-%d-%Y %H:%M:%S")
    eval_msg = f"Eval done at {curtime}\n"
    (Path(eval_output) / "eval_result.txt").write_text(eval_msg)
