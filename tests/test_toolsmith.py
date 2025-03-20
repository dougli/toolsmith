import openai
import pytest

from toolsmith import func_to_schema


def get_weather(city: str, country: str) -> str:
    """Get the current weather for a city

    Args:
        city: Name of the city
        country: Country code (e.g. US, UK)

    Returns:
        str: Weather description
    """
    # Mock weather response
    return "Sunny and 72°F"


def test_weather_function_calling():
    # Wrap the weather function
    weather_schema = func_to_schema(get_weather)

    # Verify schema structure
    assert weather_schema["name"] == "get_weather"
    assert "Get the current weather for a city" in weather_schema["description"]

    # Verify parameters
    params = weather_schema["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"city", "country"}

    properties = params["properties"]
    assert properties["city"]["type"] == "string"
    assert properties["country"]["type"] == "string"

    # Test with real OpenAI API
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather like in Seattle?"}],
        functions=[weather_schema],
        function_call={"name": "get_weather"},
    )

    function_call = response.choices[0].message.function_call
    assert function_call.name == "get_weather"
    args = function_call.arguments


def test_weather_invalid_args():
    weather_schema = func_to_schema(get_weather)

    # Verify required parameters are enforced
    with pytest.raises(TypeError):
        get_weather("Seattle")  # Missing country argument
