# Toolsmith

Toolsmith transforms regular Python functions into structured tools that can be called by AI models through the OpenAI API. It handles all schema generation and argument deserialization automatically, letting you focus on writing normal Python code.

## Key Features

- **Simple Integration**: Just write regular Python functions with type hints - Toolsmith handles the rest
- **Type Safety**: Full type checking and validation using Pydantic
- **Async Support**: Built-in support for async functions with parallel execution
- **Pydantic Models**: First-class support for complex data structures using Pydantic models
- **Unopinionated**: Integrates with any OpenAI API-compatible LLM provider

## Quick Start

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

from toolsmith import Toolbox

toolbox = Toolbox.create([create_user, search_users])

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Make a 33 year old user called Alice"}]
    tools=toolbox.get_schema(),
)

tool_calls = response.choices[0].message.tool_calls
results = toolbox.execute_function_calls(tool_calls)
```

## How Toolbox.create() Works

`Toolbox.create()` is the core function that transforms your Python functions into AI-callable tools. Here's what happens when you call it:

1. **Schema Generation**: For each function you provide, Toolsmith makes the function schema available to the model based on its:

   - Function name and docstring (for descriptions)
   - Parameter types (using Python type hints)

2. **Type Conversion**: Toolsmith converts Python types to JSON Schema:

   - Basic types (str, int, bool) map directly
   - Pydantic models are converted to nested JSON schemas
   - Enums definitions are passed to the model
   - Lists are properly typed based on their item types

3. **Validation Setup**: Toolsmith creates validators for each function to ensure:
   - All required parameters are provided
   - Parameters match their expected types
   - Complex objects (like Pydantic models) are properly deserialized

When the LLM makes a tool call, Toolsmith handles all the parsing and type conversion, so your function receives properly typed Python objects.

### Pydantic support

Toolsmith supports Pydantic objects (even nested ones), lists, and enums. For example, if you want to create a list of users:

```py
class User(BaseModel):
    name: str = Field(..., description="The name of the user")
    age: int = Field(..., description="The user's age")

def create_users(users: list[User]) -> str:
    """Creates multiple users"""
    return f"Created {len(users)} users!"
```

Any description strings for Pydantic fields will also be made visible to the LLM.

### Optional arguments

To specify optional arguments, add `None` as a union type:

```py
def create_user(name: str | None) -> str:
    return "User created"
```

## Execution

Toolsmith also makes it easy to execute your functions; it will automatically handle argument deserialization, function execution, and return type formatting so you don't have to.

### Pydantic argument deserialization

When a function accepts a Pydantic model as an argument, Toolsmith will:

1. Parse the LLM's JSON response into the correct Pydantic object
2. Validate all fields according to Pydantic's validation rules

### Return values

You can return a string or JSON in the form of a serializable `dict[str, Any]`. If a `dict` is provided, it will be serialized as JSON.

### Returning function call results

Calling `toolbox.execute_function_calls(...)` will return results in a format that can be appended as tool message responses. Simply append this as an additional message in the chat message api.

```py
tool_calls = response.choices[0].message.tool_calls
results = toolbox.execute_function_calls(tool_calls)
messages.extend(results)
```

### Async support

Toolsmith supports `async` functions out of the box. Simply use `AsyncToolbox`. If there are multiple tool calls in one response, all calls run **in parallel** and are returned to the assistant.

### Exception handling

Toolsmith doesn't handle exceptions. Uncaught exceptions in tool handlers will bubble up through the `toolbox.execute()` calls.
