from promptflow import tool

@tool
def get_url(input: str) -> str:
  """Returns the input URL."""
  return input
