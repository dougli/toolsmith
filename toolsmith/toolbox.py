import asyncio
import json
from typing import Any, Awaitable, Callable, Generic, Sequence, TypeVar, Union

from openai.types.chat import (
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel
from typing_extensions import Self

from toolsmith.toolsmith import func_to_pydantic, func_to_schema

T = TypeVar("T")


class Invocation(BaseModel, Generic[T]):
    """A single tool call to be executed by the toolbox.

    Args:
        id: The ID of the tool call
        func: The function to call
        args: The arguments to pass to the function
    """

    id: str
    func: Callable[..., T]
    args: dict[str, Any]

    def execute(self) -> T:
        return self.func(**self.args)


class BaseToolbox(BaseModel, Generic[T]):
    functions: dict[str, Callable[..., T]] = {}

    _schema_cache: Union[list[ChatCompletionToolParam], None] = None
    _func_arg_models_cache: Union[dict[str, type[BaseModel]], None] = None

    model_config = {"frozen": True}

    @classmethod
    def create(cls, functions: Sequence[Callable[..., T]]) -> Self:
        return cls(functions={f.__name__: f for f in functions})

    def get_schema(self) -> Sequence[ChatCompletionToolParam]:
        """Get OpenAI function schemas for all functions in the toolbox.

        Returns:
            Sequence[ChatCompletionToolParam]: List of function schemas compatible with OpenAI's API.
            Each schema describes the name, description and parameters of a function.
        """
        if self._schema_cache is None:
            self._schema_cache = [func_to_schema(f) for f in self.functions.values()]
        return self._schema_cache

    def get_func_arg_models(self) -> dict[str, type[BaseModel]]:
        """Get Pydantic models for validating arguments of all functions in the toolbox.

        Returns:
            dict[str, type[BaseModel]]: Mapping of function names to their corresponding
            Pydantic models. Each model validates the arguments for that function.
        """
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
    ) -> list[Invocation[T]]:
        """Parse a list of tool calls into a list of Invocation objects. Invocations
        can be executed by calling the `execute` method on them, giving you more fine-grained
        control over execution than the `execute_tool_calls` method.

        Args:
            tool_calls: List of tool calls from the OpenAI API to execute

        Returns:
            List of Invocation objects, one for each tool call
        """
        result = []
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


class Toolbox(BaseToolbox[Union[str, dict[str, Any]]]):
    def execute_tool_calls(
        self, tool_calls: list[ChatCompletionMessageToolCall]
    ) -> list[ChatCompletionToolMessageParam]:
        """Execute tool calls and returns a result that can be used to respond to the OpenAI API.
        Note that this method is synchronous and will block until all tool calls are executed.
        For asynchronous execution, use the `AsyncToolbox` class.

        Args:
            tool_calls: List of tool calls from the OpenAI API to execute

        Returns:
            List of tool messages containing the results of executing each tool call

        Warning:
            Tool calls are executed in the order they are given. If a tool call raises an
            uncaught exception, the remaining tool calls will not be executed. To prevent
            this, catch exceptions within the tool handler functions themselves
            and return error messages as part of the normal return value.
        """
        invocations = self.parse_invocations(tool_calls)

        results: list[ChatCompletionToolMessageParam] = []
        for invocation in invocations:
            execution_result = invocation.execute()
            if isinstance(execution_result, dict):
                execution_result = json.dumps(execution_result)
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": invocation.id,
                    "content": execution_result,
                }
            )

        return results


class AsyncToolbox(BaseToolbox[Awaitable[Union[str, dict[str, Any]]]]):
    async def _execute_single_invocation(
        self, invocation: Invocation[Awaitable[Union[str, dict[str, Any]]]]
    ) -> ChatCompletionToolMessageParam:
        execution_result = await invocation.execute()
        if isinstance(execution_result, dict):
            execution_result = json.dumps(execution_result)
        return {
            "role": "tool",
            "tool_call_id": invocation.id,
            "content": execution_result,
        }

    async def execute_tool_calls(
        self, tool_calls: list[ChatCompletionMessageToolCall]
    ) -> list[ChatCompletionToolMessageParam]:
        """Execute multiple tool calls asynchronously and returns a result that can be
        used to respond to the OpenAI API.

        Args:
            tool_calls: List of tool calls from the OpenAI API to execute

        Returns:
            List of tool messages containing the results of executing each tool call

        Warning:
            If any individual tool call raises an uncaught exception, other pending tool calls
            will continue to run but may be left in an indeterminate state.
            To prevent this, catch exceptions within the tool handler functions themselves
            and return error messages as part of the normal return value.
        """
        invocations = self.parse_invocations(tool_calls)

        return await asyncio.gather(
            *[self._execute_single_invocation(inv) for inv in invocations],
        )
