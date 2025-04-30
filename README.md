# GenAI Agent Protocol

## This is an async-first Python framework for building WebSocket-based AI agents that can:

* Connect Agent to GenAI ecosystem
* Process messages via registered handler functions
* Upload and retrieve files with contextual metadata (`agent_context`)
* Log messages with contextual metadata (`agent_context`)

## ✨ Features

* 🧠 Agent Binding: Decorator-based agent registration
* 🪝 WebSocket Communication: Bidirectional messaging with a central server
* 📁 File Manager: Async file upload/download & metadata fetch
* 🪵 Context Logger: Structured, contextual WebSocket-based logging
* 🔎 OpenAI Schema Conversion: Automatically converts Pydantic-type function signatures to OpenAI-compatible schemas
* 📞 Agents invoke from Agent: Call other agents from within an agent

## 📚 Core Concepts

**GenAISession**

Central controller that registers agents and manages event lifecycle.
```python
from genai_session.session import GenAISession

genai_session = GenAISession(
    jwt_token="<jwt received from GenAI CLI>",
    ws_url="<Router Container Public URL>", # Note if exposing infrastructure with ngrok
    api_base_url="<Backend Container Public URL>", # Note if exposing infrastructure with ngrok
)
```

**@bind(...)**

Registers a handler function with the session and make them visible to GenAI infrastructure.
```python
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

genai_session = GenAISession(jwt_token="<jwt received from GenAI CLI>")

@genai_session.bind(name="Test Name", description="Test Description")
async def message_handler(agent_context: GenAIContext, parameter: str) -> str:
    ...
```

**GenAIContext**

Provides contextual info (agent_uuid, request_id, etc.), a logger, and access to the FileManager.
```python
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

genai_session = GenAISession(jwt_token="<jwt received from GenAI CLI>")

@genai_session.bind(name="Test Name", description="Test Description")
async def message_handler(agent_context: GenAIContext, parameter: str) -> str:
    request_id = agent_context.request_id
```

**Files**

Handles file uploads (save) and retrievals (get_by_id, get_metadata_by_id).
```python
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

genai_session = GenAISession(jwt_token="<jwt received from GenAI CLI>")

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

genai_session = GenAISession(jwt_token="<jwt received from GenAI CLI>")

@genai_session.bind()
async def reverse_name(agent_context: GenAIContext, name: str) -> str:
    """Agent reverses the name"""
    agent_context.logger.info("Inside the reverse_name function")
    agent_context.logger.debug(f"name: {name}")
    ...
```

**Invoke Agent from Agent**

You can invoke another agent from within an agent using the `genai_session.send` method.\
This method takes the `agent_uuid` and `params` as arguments.
```python
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
from genai_session.utils.agents import AgentResponse

genai_session = GenAISession(jwt_token="<jwt received from GenAI CLI>")

@genai_session.bind()
async def invoke_another_agent(agent_context: GenAIContext, name: dict) -> str:
    """Agent invokes another registered agent"""
    agent_response: AgentResponse = await genai_session.send(
        agent_uuid="agent_uuid", # you can get UUID from - await agent_context.get_my_agents()
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

## 📝 Function annotation examples

**No parameters**
```python
@genai_session.bind(name="GetCurrentDate", description="Return current date")
async def get_current_date(agent_context: GenAIContext):
    ...
```

**Built-in types**
```python
@genai_session.bind(name="Saver", description="Saves file")
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

@genai_session.bind(name="TranslationAgent", description="Translate the text into specified language")
async def get_translation(
    agent_context: GenAIContext,
    params: TranslationInput
) -> dict[str, Any]:
    text = params.text
    language = params.language
    banned_words = params.banned_words
    ...
```

**typing Annotations**
```python
from typing import Any, Annotated

@genai_session.bind(name="TranslationAgent", description="Translate the text into specified language")
async def get_translation(
    agent_context: GenAIContext, 
    text: Annotated[str, "Text to translate"],
    language: Annotated[str, "Code of the language to translate to (e.g. 'fr', 'es')"],
    banned_words: Annotated[list[str], "List of words to be banned from translation"],
) -> dict[str, Any]:
    ...
```