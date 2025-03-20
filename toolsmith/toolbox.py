import json
from typing import Any, Callable, Sequence

from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionToolParam
from pydantic import BaseModel

from toolsmith import func_to_schema


class Invocation(BaseModel):
    id: str
    func: Callable[..., str]
    args: dict[str, Any]

    def execute(self) -> str:
        return self.func(**self.args)


class Toolbox(BaseModel):
    functions: dict[str, Callable[..., str]]

    def __init__(self, functions: Sequence[Callable[..., str]]):
        super().__init__(functions={f.__name__: f for f in functions})

    def get_schema(self) -> Sequence[ChatCompletionToolParam]:
        return [func_to_schema(f) for f in self.functions.values()]

    def parse_invocations(
        self, tool_calls: list[ChatCompletionMessageToolCall]
    ) -> list[Invocation]:
        # Convert the args to actual types
        result: list[Invocation] = []
        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name in self.functions:
                result.append(
                    Invocation(
                        id=tool_call.id,
                        func=self.functions[tool_call.function.name],
                        args=json.loads(tool_call.function.arguments),
                    )
                )

        return result

    def execute_function_calls(self, invocations: list[Invocation]) -> list[str]:
        return [invocation.execute() for invocation in invocations]
