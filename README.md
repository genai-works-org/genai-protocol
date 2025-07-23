# GenAI Agent Protocol

## GenAI Agent Protocol is an async‑first Python framework for building WebSocket‑based AI agents that let you: 

* Connect an agent to the GenAI.works ecosystem 
* Process messages via registered handler functions
* Upload and retrieve files with contextual metadata (`agent_context`)
* Log messages with contextual metadata (`agent_context`)

## ✨ Features

🧠 Agent Binding: Decorator-based agent registration\
🪝 WebSocket Communication: Bidirectional messaging with a central server\
📁 File Manager: Async file upload/download & metadata fetch\
🪵 Context Logger: Structured, contextual WebSocket-based logging\
🔎 OpenAI Schema Conversion: Automatically converts Pydantic-type function signatures to OpenAI-compatible schemas\
📞 Agent‑to‑agent calls: invoke another registered agent from within your handler

## 📚 Core Concepts

## ⚙️ Environment Variables Setup

Before you run the **GenAI Agent Protocol**, make sure to configure the necessary environment variables.

You can do this by creating a `.env` file in your project root or exporting them directly in your terminal session.

### Required:

```bash
AGENT_JWT_TOKEN="<your JWT token from GenAI CLI or UI>"
```

### Defaults (you can override if needed):

```bash
ROUTER_WS_URL=ws://localhost:8080/ws           # WebSocket router URL
BACKEND_API_BASE_URL=http://localhost:8000     # Backend API URL
IS_LOCAL_SETUP=true                            # Flag to indicate local development
```

If your agent logic requires additional environment variables, just add them to the `.env` file or terminal session the same way.

**GenAISession**

A central controller that registers agents and manages the event lifecycle. 
```python
from genai_session.session import GenAISession

genai_session = GenAISession()
```

**@bind(...)**

Registers a handler function with the session and make them visible to GenAI infrastructure.
```python
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

genai_session = GenAISession()

@genai_session.bind(name="test_name", description="Test Description")
async def message_handler(agent_context: GenAIContext, parameter: str) -> str:
    ...
```

**GenAIContext**

Provides contextual info (agent_uuid, request_id, etc.), a logger, and access to the FileManager.
```python
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

genai_session = GenAISession()

@genai_session.bind(name="test_name", description="Test Description")
async def message_handler(agent_context: GenAIContext, parameter: str) -> str:
    request_id = agent_context.request_id
```

**Files**

Handles file uploads (save) and retrievals (get_by_id, get_metadata_by_id).
```python
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

genai_session = GenAISession()

@genai_session.bind(name="txt_content_reader_agent", description="Agent returns txt file content")
async def get_file_content(agent_context: GenAIContext, file_id: str) -> str:
    file = await agent_context.files.get_by_id(file_id)
    file_metadata = await agent_context.files.get_metadata_by_id(file_id)
    ...
```

**Logger**

Sends JSON logs through WebSocket with severity levels (debug, info, warning, error, critical).
```python
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

genai_session = GenAISession()

@genai_session.bind()
async def reverse_name(agent_context: GenAIContext, name: str) -> str:
    """Agent reverses the name"""
    agent_context.logger.info("Inside the reverse_name function")
    agent_context.logger.debug(f"name: {name}")
    ...
```

**Invoke Agent from Agent**

You can invoke another agent from within an agent using the `genai_session.send` method (this method is working ONLY in IS_LOCAL_SETUP=true).\
This method takes the `agent_uuid` and `params` as arguments.
```python
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
from genai_session.utils.agents import AgentResponse

genai_session = GenAISession()

@genai_session.bind()
async def invoke_another_agent(agent_context: GenAIContext, name: dict) -> str:
    """Agent invokes another registered agent"""
    agent_response: AgentResponse = await genai_session.send(
        agent_uuid="agent_uuid", # you can get UUID from - await agent_context.get_agents()
        params={
            "username": name,
            "interests": ["python", "genai"],
            "age": 30,
        } # key is a parameter name, value is the value you want to pass
    )
    response = agent_response.response
    is_success = agent_response.is_success
    ...
```

**External environment variables example**
```python
import asyncio
import os
from typing import Any, Annotated

import requests

from genai_session.session import GenAISession

session = GenAISession()

BASE_URL = os.environ.get("BASE_WEATHER_API_URL")
API_KEY = os.environ.get("WEATHER_API_KEY")

@session.bind(name="get_weather_agent", description="Get weather forecast data")
async def get_weather(
        agent_context, city_name: Annotated[str, "City name to get weather forecast for"],
        date: Annotated[str, "Date to get forecast for in yyyy-MM-dd format"]
) -> dict[str, Any]:

    agent_context.logger.info("Inside get_translation")
    params = {"q": city_name, "dt": date, "key": API_KEY}
    response = requests.get(BASE_URL, params=params)

    return {"weather_forecast": response.json()["forecast"]["forecastday"][0]["day"]}

async def main():
    await session.process_events()


if __name__ == "__main__":
    asyncio.run(main())
```


## 📝 Function annotation examples

**No parameters**
```python
@genai_session.bind(name="get_current_date", description="Return current date")
async def get_current_date(agent_context: GenAIContext):
    ...
```

**Built-in types**
```python
@genai_session.bind(name="file_saver", description="Saves file")
async def file_saver(
    agent_context: GenAIContext,
    filename: str,
    file_content: str, 
    page_count: int, 
    images_names: list[str]
) -> dict:
    ...
```

**Pydantic models**
```python
from pydantic import BaseModel, Field
from typing import List, Any


class TranslationInput(BaseModel):
    text: str = Field(..., description="Text to translate")
    language: str = Field(..., description="Code of the language to translate to (e.g. 'fr', 'es')")
    banned_words: List[str] = Field(..., description="List of words to be banned from translation")

@genai_session.bind(name="translation_agent", description="Translate the text into specified language")
async def get_translation(
    agent_context: GenAIContext,
    params: TranslationInput
) -> dict[str, Any]:
    text = params.text
    language = params.language
    banned_words = params.banned_words
    ...
```

**`typing` Annotations**
```python
from typing import Any, Annotated

@genai_session.bind(name="translation_agent", description="Translate the text into specified language")
async def get_translation(
    agent_context: GenAIContext, 
    text: Annotated[str, "Text to translate"],
    language: Annotated[str, "Code of the language to translate to (e.g. 'fr', 'es')"],
    banned_words: Annotated[list[str], "List of words to be banned from translation"],
) -> dict[str, Any]:
    ...
```

## 🚀 Running the Event Loop

Start your agent's event loop:

```python
async def main():
    await genai_session.process_events()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## 📂 Example Agents
You can find fully working agent examples in the [GenAI Agentos GitHub Repository](https://github.com/genai-works-org/genai-agentos/tree/main/genai_agents_example).
