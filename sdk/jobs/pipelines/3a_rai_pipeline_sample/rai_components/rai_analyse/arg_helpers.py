# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import json
import logging

from typing import Any, Dict, List, Union

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def get_from_args(args, arg_name: str, custom_parser, allow_none: bool) -> Any:
    _logger.info("Looking for command line argument '{0}'".format(arg_name))
    result = None

    extracted = getattr(args, arg_name)
    if extracted is None and not allow_none:
        raise ValueError("Required argument {0} missing".format(arg_name))

    if custom_parser:
        if extracted is not None:
            result = custom_parser(extracted)
    else:
        result = extracted

    _logger.info("{0}: {1}".format(arg_name, result))

    return result


def boolean_parser(target: str) -> bool:
    true_values = ["True", "true"]
    false_values = ["False", "false"]
    if target in true_values:
        return True
    if target in false_values:
        return False
    raise ValueError("Failed to parse to boolean: {target}")


def float_or_json_parser(target: str) -> Union[float, Any]:
    try:
        return json.loads(target)
    except json.JSONDecodeError:
        return float(target.strip('"').strip("'"))


def str_or_int_parser(target: str) -> Union[str, int]:
    try:
        return int(target.strip('"').strip("'"))
    except ValueError:
        return target


def str_or_list_parser(target: str) -> Union[str, list]:
    try:
        decoded = json.loads(target)
        if not isinstance(decoded, list):
            raise ValueError("Supplied JSON string not list: {0}".format(target))
        return decoded
    except json.JSONDecodeError:
        # String, but need to get rid of quotes
        return target.strip('"').strip("'")


def int_or_none_parser(target: str) -> Union[None, int]:
    try:
        return int(target.strip('"').strip("'"))
    except ValueError:
        if "None" in target:
            return None
        raise ValueError("int_or_none_parser failed on: {0}".format(target))


def json_empty_is_none_parser(target: str) -> Union[Dict, List]:
    parsed = json.loads(target)
    if len(parsed) == 0:
        return None
    else:
        return parsed
