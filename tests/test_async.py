from typing import Awaitable, Callable

import openai
from pydantic import BaseModel

from toolsmith import AsyncToolbox


async def _run_test(prompt: str, fn: Callable[..., Awaitable[str]]):
    toolbox = AsyncToolbox.create([fn])

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        tools=toolbox.get_schema(),
    )

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None

    results = await toolbox.execute_tool_calls(tool_calls)
    assert len(results) == 1
    return results[0]["content"]


async def test_str_and_int():
    async def create_user(name: str, age: int) -> str:
        return f"User {name} created with age {age}"

    result = await _run_test("Create a user named John with age 30", create_user)
    assert result == "User John created with age 30"


async def test_boolean():
    async def toggle_light(on: bool) -> str:
        return f"The light is now {'on' if on else 'off'}"

    result = await _run_test("Turn the light on", toggle_light)
    assert result == "The light is now on"


async def test_pydantic_nested():
    class Address(BaseModel):
        city: str
        country: str

    class User(BaseModel):
        name: str
        age: int
        address: Address

    async def create_user(user: User) -> str:
        return f"User {user.name} created with age {user.age} at {user.address.city}, {user.address.country}"

    result = await _run_test(
        "Create a user named John with age 30 in New York, USA", create_user
    )
    assert result == "User John created with age 30 at New York, USA"


async def test_parallel():
    async def create_user(name: str, age: int) -> str:
        return f"User {name} created with age {age}"

    toolbox = AsyncToolbox.create([create_user])

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Create two users: John aged 30 and Jane aged 25",
            }
        ],
        tools=toolbox.get_schema(),
    )

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None

    results = await toolbox.execute_tool_calls(tool_calls)
    assert len(results) == 2
    results.sort(key=lambda x: str(x["content"]))
    assert results[0]["content"] == "User Jane created with age 25"
    assert results[1]["content"] == "User John created with age 30"
