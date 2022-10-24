from mldesigner import Output, command_component

@command_component()
def condition_func() -> Output(type="boolean", is_control=True):
    """module run logic goes here"""
    return False
