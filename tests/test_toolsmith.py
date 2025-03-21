from enum import Enum
from typing import Callable

import openai
import pytest

from toolsmith import Toolbox, func_to_schema


def test_basic_schema_generation():
    # Wrap the weather function
    def get_weather(city: str, country: str, hour: int) -> str:
        """Country should be a two-letter code. Hour should be in 24-hour format."""
        return "Sunny and 72°F"

    schema = func_to_schema(get_weather)

    # Verify schema structure
    assert schema["function"] == {
        "name": "get_weather",
        "description": "Country should be a two-letter code. Hour should be in 24-hour format.",
        "parameters": {
            "type": "object",
            "required": ["city", "country", "hour"],
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"},
                "hour": {"type": "integer"},
            },
        },
    }


def _run_test(prompt: str, fn: Callable[..., str]):
    toolbox = Toolbox(functions=[fn])

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        tools=toolbox.get_schema(),
    )

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None

    invocations = toolbox.parse_invocations(tool_calls)
    assert len(invocations) == 1

    results = toolbox.execute_function_calls(invocations)
    return results[0]["content"]


def test_str_and_int():
    def create_user(name: str, age: int) -> str:
        return f"User {name} created with age {age}"

    result = _run_test("Create a user named John with age 30", create_user)
    assert result == "User John created with age 30"


def test_str_enum():
    class Room(Enum):
        KITCHEN = "kitchen"
        BEDROOM = "bedroom"

    def get_room_temperature(room: Room) -> str:
        return f"The temperature in the {room.value} is 72°F"

    result = _run_test("What's the temperature in the kitchen?", get_room_temperature)
    assert result == "The temperature in the kitchen is 72°F"


def test_int_enum_raises():
    class Room(Enum):
        KITCHEN = 1
        BEDROOM = 2

    def get_room_temperature(room: Room) -> str:
        return f"The temperature in the {room.name.lower()} is 72°F"

    with pytest.raises(ValueError, match='Enum "Room" has non-string values'):
        _run_test("What's the temperature in the bedroom?", get_room_temperature)


def test_boolean():
    def toggle_light(on: bool) -> str:
        return f"The light is now {'on' if on else 'off'}"

    result = _run_test("Turn the light on", toggle_light)
    assert result == "The light is now on"


def test_list():
    def add_to_zoo(animal_types: list[str]) -> str:
        return f"Added {', '.join(animal_types)} to the zoo"

    result = _run_test("Add a cat and a dog to the zoo", add_to_zoo)
    assert result == "Added cat, dog to the zoo"


def test_list_without_keys():
    def add_to_zoo(animal_types: list) -> str:
        return f"Added {', '.join(animal_types)} to the zoo"

    with pytest.raises(
        ValueError,
        match="`animal_types` is a list with untyped items. Please type the items.",
    ):
        _run_test("Add a cat and a dog to the zoo", add_to_zoo)


def test_dict():
    def set_key_value(settings: dict) -> str:
        return ""

    with pytest.raises(ValueError, match="`settings` is a dict, which is not allowed"):
        _run_test("Set the volume to 50", set_key_value)


def test_dict_with_keys():
    def set_key_value(settings: dict[str, str]) -> str:
        return ""

    with pytest.raises(ValueError, match="`settings` is a dict, which is not allowed"):
        _run_test("Set the volume to 50", set_key_value)
