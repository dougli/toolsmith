# toolsmith

Toolsmith turns your Python functions into structured, AI-callable tools, using type hints and docstrings. Integrates with any LLM provider that is compatible with the OpenAI API.

- **Very easy to use:** Just write normal Python functions with typehints.
- **Pydantic support:** Pydantic types get automatically serialized and deserialized.
- **Unopinionated:** Toolsmith gets out of the way in terms of how you want to wire up the LLM loop.
- **Fast:** Highly performant based on Pydantic's speed.
- **Async support:** Built-in support for async functions with parallel execution of multiple tool calls.

## Installation

```sh
$ pip install toolsmith
```

## Usage

Simply define any functions you may have normally:

```py
def create_user(name: str, age: int) -> str:
    """Saves a user to the DB"""
    return f"Created user {name}, age {age}"

def search_users(query: str, regex: bool) -> str:
    return "Found some users"
```

Put it all together and call the OpenAI API:

```py
import openai
from toolsmith import AsyncToolbox

toolbox = AsyncToolbox.create([create_user, search_users])

client = openai.AsyncOpenAI()
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Make a 33 year old user called Alice"}]
    tools=toolbox.get_schema(),  # toolsmith handles the schema
)

# toolsmith will automatically call your functions here
results = await toolbox.execute_tool_calls(response.choices[0].message.tool_calls)
```

## Learn more

https://dougli.github.io/toolsmith/
