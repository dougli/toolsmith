import inspect
from typing import Any, Callable, Union, get_type_hints

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, create_model


def func_to_schema(
    fn: Callable[..., Any], strict: bool = False
) -> ChatCompletionToolParam:
    """Wraps a Python function to be compatible with OpenAI's function calling API.

    Args:
        tool: The Python function to wrap

    Returns:
        dict: Function schema compatible with OpenAI's API
    """

    # Build parameters schema
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    # Convert function args to a Pydantic model first
    args_model = func_to_pydantic(fn)

    # Get the JSON schema from the Pydantic model
    model_schema = args_model.model_json_schema()
    model_schema = _strip_title(model_schema)

    # Build function schema
    schema: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": model_schema,
        },
    }
    if strict:
        model_schema["additionalProperties"] = False
        schema["function"]["strict"] = True

    return schema


def func_to_pydantic(func: Callable[..., Any]) -> type[BaseModel]:
    """Convert a function's arguments to a Pydantic model. Used for input validation."""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Create field definitions for the model
    fields = {}
    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, type(None))

        # Handle default values
        if param.default != param.empty:
            fields[param_name] = (param_type, param.default)
        else:
            fields[param_name] = (param_type, ...)

    # Create a new Pydantic model class dynamically
    return create_model(f"{func.__name__}Args", **fields)


def _strip_title(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip out the "title" field since it's redundant from the name"""
    if "title" in schema:
        del schema["title"]
    for key, value in schema.items():
        if isinstance(value, dict):
            _strip_title(value)
    return schema
