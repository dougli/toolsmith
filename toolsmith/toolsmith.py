import inspect
from typing import Callable, get_type_hints


def wrap_tool(tool: Callable[..., str]):
    """Wraps a Python function to be compatible with OpenAI's function calling API.

    Args:
        tool: The Python function to wrap

    Returns:
        dict: Function schema compatible with OpenAI's API
    """

    # Get function signature info
    sig = inspect.signature(tool)
    type_hints = get_type_hints(tool)
    doc = inspect.getdoc(tool) or ""

    # Build parameters schema
    parameters = {"type": "object", "properties": {}, "required": []}

    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, type(None))
        param_schema = {"type": _python_type_to_json_type(param_type)}

        parameters["properties"][param_name] = param_schema
        if param.default == param.empty:
            parameters["required"].append(param_name)

    # Build function schema
    schema = {"name": tool.__name__, "description": doc, "parameters": parameters}

    return schema


def _python_type_to_json_type(py_type: type) -> str:
    """Convert Python type to JSON schema type"""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return type_map.get(py_type, "string")
