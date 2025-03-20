import json
from typing import Any, Callable, Sequence

from pydantic import BaseModel

from toolsmith import func_to_schema


class Invocation(BaseModel):
    func: Callable[..., str]
    args: dict[str, Any]


class Toolbox(BaseModel):
    functions: dict[str, Callable[..., str]]

    def get_schema(self) -> list[dict[str, Any]]:
        return [func_to_schema(f) for f in self.functions.values()]

    def parse_function_call(self, args: Any) -> list[Invocation]:
        obj = json.loads(args)

        # Convert the args to actual types
        for key, value in obj.items():
            if key in self.functions:
                obj[key] = self.functions[key](value)
        return obj

    def execute_function_calls(self, invocations: list[Invocation]) -> list[str]:
        pass

    async def execute_function_calls_async(
        self, invocations: list[Invocation]
    ) -> list[str]:
        pass
