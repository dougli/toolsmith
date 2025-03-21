import json
from typing import Any, Callable, Sequence, Union

from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel

from toolsmith.toolsmith import func_to_pydantic, func_to_schema


class Invocation(BaseModel):
    id: str
    func: Callable[..., str]
    args: dict[str, Any]

    def execute(self) -> str:
        return self.func(**self.args)


class Toolbox(BaseModel):
    functions: dict[str, Callable[..., str]]

    _schema_cache: Union[list[ChatCompletionToolParam], None] = None
    _func_arg_models_cache: Union[dict[str, type[BaseModel]], None] = None

    model_config = {"frozen": True}

    def __init__(self, functions: Sequence[Callable[..., str]]):
        super().__init__(functions={f.__name__: f for f in functions})

    def get_schema(self) -> Sequence[ChatCompletionToolParam]:
        if self._schema_cache is None:
            self._schema_cache = [func_to_schema(f) for f in self.functions.values()]
        return self._schema_cache

    def get_func_arg_models(self) -> dict[str, type[BaseModel]]:
        if self._func_arg_models_cache is None:
            self._func_arg_models_cache = {
                name: func_to_pydantic(f) for name, f in self.functions.items()
            }
        return self._func_arg_models_cache

    def _parse_args(self, func_name: str, args_json: str) -> dict[str, Any]:
        func_args_model = self.get_func_arg_models()[func_name]
        return dict(func_args_model(**json.loads(args_json)))

    def parse_invocations(
        self, tool_calls: list[ChatCompletionMessageToolCall]
    ) -> list[Invocation]:
        # Convert the args to actual types
        result: list[Invocation] = []
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            if func_name not in self.functions:
                raise ValueError(f"Function {func_name} not found in toolbox")

            result.append(
                Invocation(
                    id=tool_call.id,
                    func=self.functions[func_name],
                    args=self._parse_args(func_name, tool_call.function.arguments),
                )
            )

        return result

    def execute_function_calls(
        self, invocations: list[Invocation]
    ) -> list[ChatCompletionToolMessageParam]:
        results: list[ChatCompletionToolMessageParam] = []
        for invocation in invocations:
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": invocation.id,
                    "content": invocation.execute(),
                }
            )

        return results
