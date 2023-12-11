# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from typing_extensions import Annotated
from mldesigner import Input, Meta, Output, command_component
from mldesigner.dsl import group


@group
class GroupOutputs:
    output_int: Annotated[int, Meta(description="test annotation int")] = None
    output_float: Annotated[float, Meta(description="test annotation float")] = None
    output_str: Annotated[str, Meta(description="test annotation str")] = None
    output_bool: Annotated[bool, Meta(description="test annotation bool")] = None

    output_int2: int = None
    output_float2: float = None
    output_str2: str = None
    output_bool2: bool = None

    output_int3: Output(type="integer")
    output_float3: Output(type="number")
    output_str3: Output(type="string")
    output_bool3: Output(type="boolean")


@command_component()
def component_return_annotated_group_outputs(
        input_int: int = None, input_float: float = None, input_str: str = None, input_bool: bool = None
) -> GroupOutputs:
    return GroupOutputs(
        output_int=input_int,
        output_float=input_float,
        output_str=input_str,
        output_bool=input_bool,
        output_int2=input_int,
        output_float2=input_float,
        output_str2=input_str,
        output_bool2=input_bool,
        output_int3=input_int,
        output_float3=input_float,
        output_str3=input_str,
        output_bool3=input_bool,
    )


@command_component()
def component_return_annotated_output(input_int: int) -> Annotated[int, Meta(description="test annotation int")]:
    return 1


@command_component()
def component_return_int_output(input_int: int) -> int:
    return input_int


@command_component()
def component_return_integer_output(input_int: int) -> Output(type="integer"):
    return input_int