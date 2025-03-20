import openai

from toolsmith import Toolbox, func_to_schema


def get_weather(city: str, country: str, hour: int) -> str:
    """Country should be a two-letter code. Hour should be in 24-hour format."""
    return "Sunny and 72°F"


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
