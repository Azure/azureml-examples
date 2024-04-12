from promptflow import tool

# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type hints to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def get_url(input: str) -> str:
  """Returns the input URL."""
  return input
