# toolsmith

Toolsmith turns your Python functions into structured, AI-callable tools, using type hints and docstrings. Integrates with any LLM provider that is compatible with the OpenAI API.

- **Very easy to use:** Just write normal Python functions with typehints.
- **Pydantic support:** Pydantic types get automatically serialized and deserialized.
- **Unopinionated:** Toolsmith gets out of the way in terms of how you want to wire up the LLM loop.
- **Fast:** Highly performant based on Pydantic's speed.

## Installation

```sh
$ pip install toolsmith
```

## Usage

Define your functions directly

```py
import openai
import toolsmith

# Define your functions normally
def create_user(name: str, age: int) -> str:
    """Saves a user to the DB"""
    return f"Created user {name}, age {age}"

toolbox = toolsmith.create([create_user])

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Make a 33 year old user called Bob"}]
    tools=toolbox.get_schema(),
)

invocations = toolbox.parse_invocations(response.choices[0].message.tool_calls)
results = toolbox.execute_function_calls(invocations)
```
