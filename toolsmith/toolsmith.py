import inspect
from typing import Any, Callable, Union, get_type_hints

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel


def func_to_schema(
    tool: Callable[..., Any], strict: bool = False
) -> ChatCompletionToolParam:
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
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, type(None))
        param_schema = _python_type_to_json_type(param_type)

        parameters["properties"][param_name] = param_schema
        if param.default == param.empty:
            parameters["required"].append(param_name)

    # Build function schema
    schema: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": tool.__name__,
            "description": doc,
            "parameters": parameters,
        },
    }
    if strict:
        parameters["additionalProperties"] = False
        schema["function"]["strict"] = True

    return schema


def _python_type_to_json_type(py_type: type) -> Union[str, dict[str, Any]]:
    """Convert Python type to JSON schema type"""
    # Handle Pydantic models
    if issubclass(py_type, BaseModel):
        result = py_type.model_json_schema()
        # Recursively strip out the "title" field since it's redundant from the name
        result = _strip_title(result)
        return result

    # Basic type mapping
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }
    if py_type in type_map:
        return type_map[py_type]
    else:
        raise ValueError(f"Unsupported type: {py_type}")


def _strip_title(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip out the "title" field since it's redundant from the name"""
    if "title" in schema:
        del schema["title"]
    for key, value in schema.items():
        if isinstance(value, dict):
            _strip_title(value)
    return schema
