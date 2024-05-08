from promptflow import tool
import json


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need

@tool
def my_python_tool(json_body: object) -> str:
    suggested_action = json_body["suggested_action"]
    return suggested_action
