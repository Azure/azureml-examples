from pathlib import Path
from mldesigner import command_component, Input, Output

@command_component
def write_output_folder(my_output: Output(type="uri_folder")):
    """Test write output"""
    res_file = Path(my_output) / "result.txt"
    Path(res_file).write_text("Hello world!")


@command_component
def write_output_file(my_output: Output(type="uri_file")):
    """Test write output"""
    Path(my_output).write_text("Hello world!")


@command_component
def write_output_annotation() -> Output(type="uri_folder"):
    """Test write output"""
    res_folder = Path("my_output")
    res_folder.mkdir()
    Path(res_folder / "result.txt").write_text("Hello world!")
    return str(res_folder)

@command_component
def write_str_output() -> Output(type="string", is_control=True):
    return "Hello World"