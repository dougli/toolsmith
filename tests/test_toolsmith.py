import openai
import pytest
from pydantic import BaseModel, Field

from toolsmith import func_to_schema
from toolsmith.toolbox import Toolbox


def get_weather(city: str, country: str, hour: int) -> str:
    """Country should be a two-letter code. Hour should be in 24-hour format."""
    return "Sunny and 72°F"


class UserArgs(BaseModel):
    name: str = Field(..., description="Name of the user to create.")
    email: str = Field(..., description="Email of the user to create.")
    skills: list[str] = Field(
        [], description="Skills to add to the user. Can be empty."
    )


def create_user(args: UserArgs) -> str:
    """Create a new user with the given name and email."""
    return f"User {args.name} created with email {args.email}"


def test_basic_schema_generation():
    # Wrap the weather function
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


def test_pydantic_schema_generation():
    # Wrap the weather function
    schema = func_to_schema(create_user)

    # Verify schema structure
    assert schema["function"] == {
        "name": "create_user",
        "description": "Create a new user with the given name and email.",
        "parameters": {
            "type": "object",
            "required": ["args"],
            "properties": {
                "args": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the user to create.",
                        },
                        "email": {
                            "type": "string",
                            "description": "Email of the user to create.",
                        },
                        "skills": {
                            "default": [],
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Skills to add to the user. Can be empty.",
                        },
                    },
                    "required": ["name", "email"],
                },
            },
        },
    }


def test_pydantic_schema_with_openai():
    # Wrap the weather function
    toolbox = Toolbox([create_user])

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Create a new user with the name John Doe and email john.doe@example.com.",
            }
        ],
        tools=toolbox.get_schema(),
    )

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None
    invocations = toolbox.parse_invocations(tool_calls)

    assert len(invocations) == 1
    assert invocations[0].func == create_user
    assert invocations[0].args == {
        "args": {"name": "John Doe", "email": "john.doe@example.com"}
    }


@pytest.mark.skip("Skipping weather function calling test")
def test_weather_function_calling():
    toolbox = Toolbox(functions=[get_weather])

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What's the weather like in Seattle at 12pm?"}
        ],
        tools=toolbox.get_schema(),
    )

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None

    invocations = toolbox.parse_invocations(tool_calls)
    assert len(invocations) == 1
    assert invocations[0].func == get_weather
    assert invocations[0].args == {"city": "Seattle", "country": "US", "hour": 12}

    results = toolbox.execute_function_calls(invocations)
    assert results == ["Sunny and 72°F"]
