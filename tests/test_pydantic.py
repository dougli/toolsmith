from enum import Enum

import openai
from pydantic import BaseModel, Field

from toolsmith import Toolbox, func_to_schema


class User(BaseModel):
    name: str = Field(..., description="Name of the user to create.")
    email: str = Field(..., description="Email of the user to create.")
    skills: list[str] = Field(
        ..., description="Skills to add to the user. Can be empty."
    )


class AnimalType(Enum):
    DOG = "dog"
    CAT = "cat"
    BIRD = "bird"


class Animal(BaseModel):
    name: str
    type: AnimalType


def create_user(user: User) -> str:
    """Create a new user with the given name and email."""
    return f"User {user.name} created with email {user.email}"


def add_to_zoo(animal: Animal) -> str:
    """Add an animal to the zoo"""
    return f"Added {animal.name}, a {animal.type.value} to the zoo"


def add_many_to_zoo(animals: list[Animal]) -> str:
    """Add many animals to the zoo"""
    return f"Added {', '.join([a.name for a in animals])} to the zoo"


def test_pydantic_schema_generation():
    # Wrap the weather function
    schema = func_to_schema(create_user)

    # Verify schema structure
    assert schema["function"] == {
        "name": "create_user",
        "description": "Create a new user with the given name and email.",
        "parameters": {
            "type": "object",
            "required": ["user"],
            "properties": {
                "user": {
                    "$ref": "#/$defs/User",
                },
            },
            "$defs": {
                "User": {
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
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Skills to add to the user. Can be empty.",
                        },
                    },
                    "required": ["name", "email", "skills"],
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "strict": True,
    }


def test_pydantic_enum():
    schema = func_to_schema(add_to_zoo)
    assert schema["function"] == {
        "name": "add_to_zoo",
        "description": "Add an animal to the zoo",
        "parameters": {
            "type": "object",
            "properties": {
                "animal": {"$ref": "#/$defs/Animal"},
            },
            "$defs": {
                "Animal": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"$ref": "#/$defs/AnimalType"},
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False,
                },
                "AnimalType": {
                    "type": "string",
                    "enum": ["dog", "cat", "bird"],
                },
            },
            "required": ["animal"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def test_pydantic_schema_with_openai():
    # Wrap the weather function
    toolbox = Toolbox.create([create_user])

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
        "user": User(name="John Doe", email="john.doe@example.com", skills=[])
    }
    assert (
        invocations[0].execute()
        == "User John Doe created with email john.doe@example.com"
    )


def test_pydantic_with_enums():
    # Wrap the add_animal function
    toolbox = Toolbox.create([add_to_zoo])

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Add a cat named Whiskers to the zoo."}],
        tools=toolbox.get_schema(),
    )

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None
    invocations = toolbox.parse_invocations(tool_calls)

    assert len(invocations) == 1
    assert invocations[0].func == add_to_zoo
    assert invocations[0].args == {
        "animal": Animal(name="Whiskers", type=AnimalType.CAT)
    }
    assert invocations[0].execute() == "Added Whiskers, a cat to the zoo"
