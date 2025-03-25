import inspect
import logging
from typing import Any, Callable, get_type_hints

from openai import pydantic_function_tool
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)


def func_to_schema(fn: Callable[..., Any]) -> ChatCompletionToolParam:
    """Wraps a Python function to be compatible with OpenAI's function calling API.

    Args:
        fn: The Python function to wrap

    Returns:
        dict: Function schema compatible with OpenAI's API
    """
    args_model = func_to_pydantic(fn)
    schema = pydantic_function_tool(
        args_model, name=fn.__name__, description=inspect.getdoc(fn) or ""
    )
    model_schema = _strip_title(schema["function"].get("parameters", {}))
    _validate(fn.__name__, "parameters", model_schema)

    return schema


def func_to_pydantic(func: Callable[..., Any]) -> type[BaseModel]:
    """Convert a function's arguments to a Pydantic model. Used for input validation."""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Create field definitions for the model
    fields = {}
    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name)

        if not param_type:
            raise ValueError(
                f"Parameter `{param_name}` in `{func.__name__}` is not typed. Add a type hint."
            )

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


def _validate(fn_name: str, parent_key: str, schema: dict[str, Any]) -> None:
    if schema.get("type") == "object":
        additional_properties = schema.get("additionalProperties", False)
        if "properties" not in schema or additional_properties is not False:
            raise ValueError(
                f"`{fn_name}`: `{parent_key}` is a dict, which is not allowed. Convert to Pydantic instead."
            )

    if schema.get("type") == "array" and schema.get("items") == {}:
        raise ValueError(
            f"`{fn_name}`: `{parent_key}` is a list with untyped items. Please type the items."
        )

    for key, value in schema.items():
        if isinstance(value, dict):
            _validate(fn_name, key, value)

        if (
            key == "enum"
            and isinstance(value, list)
            and not all(isinstance(v, str) for v in value)
        ):
            logger.warning(
                f"`{fn_name}`: `{parent_key}` is an enum with non-string values. Note that the model won't see enum keys and this may cause issues."
            )
