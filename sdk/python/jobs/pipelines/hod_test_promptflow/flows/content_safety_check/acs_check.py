from promptflow import tool
import requests
import bs4


@tool
def acs_check(check_result: object) -> bool:
  if check_result["suggested_action"] == "Accept":
    return True
  else:
    return False
