from mldesigner import Output, Input, command_component
from pathlib import Path

@command_component()
def plus(
    number_input_folder: Input(type="uri_folder"),
    number_output: Output(type="uri_folder")
    )-> Output(type="boolean", is_control=True):
    """module run logic goes here"""
    number_input = int((Path(number_input_folder) / 'number').read_text())
    if number_input>=10:
        return False    
    number = number_input + 2
    text = str(number)
    (Path(number_output) / 'number').write_text(text)

    return True