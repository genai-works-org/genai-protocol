import asyncio
import contextlib
import inspect
import json
import logging
import os
import time
import traceback
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

import aiohttp
import jwt
import pydantic
import websockets
from dotenv import load_dotenv
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import WebSocketException, ConnectionClosedError, ConnectionClosedOK

from genai_session.utils.agents import Agent, AgentResponse
from genai_session.utils.context import GenAIContext
from genai_session.utils.exceptions import BaseAIAgentException, RouterInaccessibleException
from genai_session.utils.function_annotation import convert_to_openai_schema
from genai_session.utils.logging_manager import ContextLogger
from genai_session.utils.naming_enums import WSMessageType, ERROR_TYPE_EXCEPTIONS_MAPPING


class GenAISession:
    """
    Manages WebSocket communication and agent lifecycle for GenAI-based functions.

    This class handles:
    - Agent registration
    - Function binding with OpenAI-compatible schemas
    - Message sending and receiving via WebSocket
    - Synchronous and asynchronous function invocation
    - Session and request metadata tracking
    """

    def __init__(self, log_level: int = logging.INFO) -> None:

        caller_frame = inspect.stack()[1]
        caller_path = Path(caller_frame.filename).resolve()
        env_path = caller_path.parent / ".env"
        load_dotenv(dotenv_path=env_path)

        self.is_local_setup = os.environ.get("IS_LOCAL_SETUP", "true").lower() == "true"

        self.agent: Optional[Agent] = None
        self.jwt_token = os.environ.get("AGENT_JWT_TOKEN", "")
        self.ws_url = os.environ.get("ROUTER_WS_URL", "ws://localhost:8080/ws")
        self.api_base_url = os.environ.get("BACKEND_API_BASE_URL", "http://localhost:8000").rstrip("/")

        self._session_id: str = ""
        self._request_id: str = ""
        self._send_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False

    @property
    def headers(self) -> dict:
        """
        Constructs authorization headers based on the JWT token.

        Returns:
            dict: A dictionary containing authorization headers.
        """
        headers = {}
        if self.jwt_token:
            headers["X-Custom-Authorization"] = self.jwt_token
        return headers

    @property
    def request_id(self) -> str:
        """
        Returns the current request ID.

        Returns:
            str: The request identifier.
        """
        return self._request_id

    @request_id.setter
    def request_id(self, value: str) -> None:
        self._request_id = value

    @property
    def session_id(self) -> str:
        """
        Returns the current session ID.

        Returns:
            str: The session identifier.
        """
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    @property
    def agent_uuid(self) -> str:
        """
        Extracts and returns the agent UUID from the JWT token.

        Returns:
            str: The agent UUID or empty string if decoding fails.
        """
        try:
            decoded = jwt.decode(
                self.jwt_token,
                options={"verify_signature": False},
                algorithms=["HS256"]
            )
            return decoded.get("sub")
        except jwt.exceptions.DecodeError:
            return ""

    def bind(self, name: Optional[str] = None, description: Optional[str] = None) -> Callable:
        """
        Binds a Python function to a GenAI agent with OpenAI-compatible schema.

        Args:
            name (Optional[str]): Custom agent name (defaults to function name).
            description (Optional[str]): Custom agent description (defaults to function docstring).

        Returns:
            Callable: The wrapped function, ready to be executed by the agent.
        """

        def decorator(func: Callable) -> Callable:
            function_schema = convert_to_openai_schema(func)
            function_description = function_schema.get("function", {}).get("description", "")
            function_name = function_schema.get("function", {}).get("name", "")

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            if description:
                function_schema["function"]["description"] = description

            function_schema["function"]["name"] = self.agent_uuid if self.is_local_setup else (name or function_name)

            self.agent = Agent(
                handler=func,
                description=description or function_description,
                name=name or function_name,
                input_schema=function_schema
            )

            self.logger.info(f"Agent name: {self.agent.name}")
            self.logger.info(f"Agent description: {self.agent.description}")

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def get_agent_metadata(self) -> dict:
        """
        Returns metadata about the currently registered agent.

        Returns:
            dict: Dictionary containing name, description, and input schema.
        """
        return {
            "name": self.agent.name,
            "description": self.agent.description,
            "input_schema": self.agent.input_schema,
        }

    async def get_agents(self) -> list[dict]:
        """
        Retrieves all agents registered with the backend.

        Returns:
            list[dict]: A list of agent metadata dictionaries.
        """
        url = f"{self.api_base_url}/api/agents/frontend"
        headers = {
            "Authorization": f"Bearer {self.jwt_token}"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()

    async def get_active_agents(self) -> list[dict]:
        """
        Retrieves only the active agents from the backend.

        Returns:
            list[dict]: A list of active agent metadata dictionaries.
        """
        url = f"{self.api_base_url}/api/agents/frontend"
        headers = {
            "Authorization": f"Bearer {self.jwt_token}"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, params={"is_active": True}) as response:
                response.raise_for_status()
                return await response.json()

    async def send(
        self,
        message: dict,
        agent_uuid: str,
        close_timeout: int = None
    ) -> AgentResponse:
        """
        Sends a message to a specific agent over WebSocket and waits for a response.

        Args:
            message (dict): Payload to be sent to the agent.
            agent_uuid (str): Target agent's UUID.
            close_timeout (int, optional): Timeout in seconds to wait for the response.

        Returns:
            AgentResponse: Contains success flag, execution time, and response/error.
        """
        if not self.is_local_setup:
            return AgentResponse(
                is_success=True,
                execution_time=0,
                response="'send' method is not available in remote setup."
            )

        headers = {"x-custom-invoke-key": f"{self.agent_uuid}:{agent_uuid}"}

        async with websockets.connect(self.ws_url, additional_headers=headers) as ws:
            init_message = json.dumps({
                "message_type": WSMessageType.AGENT_INVOKE.value,
                "agent_uuid": agent_uuid,
                "request_payload": {**message},
                "request_metadata": {
                    "request_id": self.request_id,
                    "session_id": self.session_id,
                }
            })

            self.logger.debug(f"Sending message to: {agent_uuid}")
            self.logger.debug(f"Message: {message}")
            await ws.send(init_message)

            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=close_timeout) if close_timeout else await ws.recv()  # noqa: E501
                except asyncio.TimeoutError:
                    return AgentResponse(is_success=False, execution_time=0, response="Request timed out")

                body = json.loads(msg)

                if message_type := body.get("message_type"):
                    if message_type == WSMessageType.AGENT_RESPONSE.value:
                        return AgentResponse(
                            is_success=True,
                            execution_time=body.get("execution_time", 0),
                            response=body.get("response", "")
                        )
                    elif message_type == WSMessageType.AGENT_ERROR.value:
                        return AgentResponse(
                            is_success=False,
                            execution_time=body.get("execution_time", 0),
                            response=body.get("error", {}).get("error_message", "")
                        )

    def sync_invoke(self, request_id="", session_id="", request_payload: dict = None) -> dict:
        """
        Synchronously invokes the bound agent function in a blocking context.

        Args:
            request_id (str): The request identifier.
            session_id (str): The session identifier.
            request_payload (dict): The input payload for the function.

        Returns:
            dict: The result of the function invocation.
        """
        request_payload = request_payload or {}

        agent_context = GenAIContext(
            agent_uuid=self.agent_uuid,
            api_base_url=self.api_base_url,
            jwt_token=self.jwt_token
        )

        agent_context.request_id = request_id
        agent_context.session_id = session_id

        agent_context.logger = ContextLogger(
            agent_uuid=agent_context.agent_uuid,
            request_id=agent_context.request_id,
            session_id=agent_context.session_id,
            internal_logger=self.logger
        )

        async def wrapper():
            return await self.agent.handler(agent_context=agent_context, **request_payload)

        result = asyncio.run(wrapper())

        logs = self.get_agent_logs(agent_context, request_id)

        return {
            "response": result,
            "logs": logs,
            "metadata": {
                "request_id": request_id,
                "session_id": session_id,
            }
        }

    async def process_events(self) -> None:
        """
        Starts an event loop that listens for WebSocket messages and routes them to the agent handler.
        """
        try:
            async with websockets.connect(self.ws_url, additional_headers=self.headers) as ws:
                agent_context = GenAIContext(
                    agent_uuid=self.agent_uuid,
                    websocket=ws,
                    api_base_url=self.api_base_url,
                    jwt_token=self.jwt_token
                )

                init_message = json.dumps({
                    "message_type": WSMessageType.AGENT_REGISTER.value,
                    "request_payload": {
                        "agent_name": self.agent.name,
                        "agent_description": self.agent.description,
                        "agent_input_schema": self.agent.input_schema,
                    }
                })

                await ws.send(init_message)

                async def receive_messages():
                    try:
                        while True:
                            msg = await ws.recv()
                            body = json.loads(msg)
                            task = asyncio.create_task(self._handle_agent_request(agent_context, ws, body))
                            task.add_done_callback(self._handle_task_result)
                    except (ConnectionClosedError, ConnectionClosedOK):
                        self.logger.error(f"WebSocket disconnected, the router service is not accessible")
                        self._shutdown_event.set()
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        self.logger.exception(f"Unexpected error occurred: {e}")
                        self._shutdown_event.set()

                self._shutdown_event.clear()
                listener_task = asyncio.create_task(receive_messages())

                await self._shutdown_event.wait()
                listener_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await listener_task

        except (WebSocketException, ConnectionRefusedError, OSError) as e:
            self.logger.error(f"Failed to connect to WebSocket server: {e}")
            self._shutdown_event.set()
            raise RouterInaccessibleException("Router service is not accessible.")

    def _handle_task_result(self, task: asyncio.Task):
        """
        Internal callback to handle exceptions raised in async agent request tasks.

        Args:
            task (asyncio.Task): The async task whose result is being handled.
        """
        try:
            task.result()
        except BaseAIAgentException as e:
            self.logger.error(f"Agent encountered an error and will exit: {e}")
            self._shutdown_event.set()

    async def _handle_agent_request(
        self,
        agent_context: GenAIContext,
        ws: ClientConnection,
        body: dict,
    ):
        """
        Internal method to process a single agent request message.

        Args:
            agent_context (GenAIContext): Context for current agent execution.
            ws (ClientConnection): Active WebSocket connection to the router.
            body (dict): Payload of the request message.
        """
        request_payload = body.get("request_payload", {})
        request_metadata = body.get("request_metadata", {})
        error = body.get("error", {})
        invoked_by = body.get("invoked_by", "")
        execution_time = 0

        # Raise known exception if agent error occurred
        if error:
            exception = ERROR_TYPE_EXCEPTIONS_MAPPING.get(error.get("error_type"))
            raise exception(error.get("error_message"))

        # Sync request/session IDs
        for attr in ("request_id", "session_id"):
            value = request_metadata.pop(attr, "")
            setattr(agent_context, attr, value)
            setattr(self, attr, value)

        agent_context.logger = ContextLogger(
            agent_uuid=agent_context.agent_uuid,
            request_id=agent_context.request_id,
            session_id=agent_context.session_id,
            websocket=ws,
            invoked_by=invoked_by,
            internal_logger=self.logger,
        )

        agent_context.invoked_by = invoked_by

        try:
            start_time = time.perf_counter()

            logging_data = {
                "request_payload": request_payload,
                "is_start_execution": True
            }
            agent_context.logger.info(logging_data)
            self.logger.debug(logging_data)

            # Call the bound function
            result = await self.agent.handler(agent_context=agent_context, **request_payload)

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            logging_data = {
                "execution_time": execution_time,
                "response": result,
                "is_start_execution": False
            }
            agent_context.logger.info(logging_data)
            self.logger.debug(logging_data)

            # Format result
            if isinstance(result, pydantic.BaseModel):
                response = result.model_dump()
            elif isinstance(result, BytesIO):
                response = {"response": result.getvalue().decode("utf-8")}
            elif isinstance(result, bytes):
                response = {"response": result.decode("utf-8")}
            elif not isinstance(result, dict):
                response = {"response": result}
            else:
                response = {"response": result}

        except Exception as e:
            agent_context.logger.critical(traceback.format_exc())
            response = {
                "message_type": WSMessageType.AGENT_ERROR.value,
                "error": {
                    "error_message": str(e)
                }
            }
        else:
            response["message_type"] = WSMessageType.AGENT_RESPONSE.value

        response["execution_time"] = execution_time
        response["invoked_by"] = invoked_by

        await agent_context.logger.flush_logs()

        # Send response back over WebSocket
        async with self._send_lock:
            try:
                is_error = response.get("message_type") == WSMessageType.AGENT_ERROR.value
                response = json.dumps(response)
            except (TypeError, ValueError) as e:
                response = json.dumps({
                    "message_type": WSMessageType.AGENT_ERROR.value,
                    "execution_time": execution_time,
                    "invoked_by": invoked_by,
                    "error": {
                        "error_message": str(e)
                    }
                })
                is_error = True

                agent_context.logger.critical(traceback.format_exc())
                self.logger.error("Failed to send response. Invalid data type.")
                self.logger.error(e)

            await ws.send(response)

            if is_error:
                self._shutdown_event.set()

    @staticmethod
    def get_agent_logs(agent_context: GenAIContext, request_id: str) -> list[dict]:
        return [
            {
                "level": data[0],
                "message": data[-1]
            }
            for data in agent_context.logger.pending_log_tasks.get(request_id, [])
        ]
