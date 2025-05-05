import asyncio
import contextlib
import json
import logging
import time
import traceback
from functools import wraps
from io import BytesIO
from typing import Callable, Optional

import aiohttp
import jwt
import pydantic
import websockets
from websockets.asyncio.client import ClientConnection

from genai_session.utils.agents import Agent, AgentResponse
from genai_session.utils.context import GenAIContext
from genai_session.utils.exceptions import BaseAIAgentException, RouterInaccessibleException
from genai_session.utils.function_annotation import convert_to_openai_schema
from genai_session.utils.naming_enums import WSMessageType, ERROR_TYPE_EXCEPTIONS_MAPPING


class GenAISession:
    """
    Manages WebSocket communication and agent lifecycle for GenAI-based functions.
    Handles sending and receiving messages, registering agents, and invoking functions.
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:8080/ws",
        api_base_url: str = "http://localhost:8000",
        jwt_token: str = "",
        api_key: str = "",
        log_level: int = logging.INFO
    ) -> None:
        """
        Initializes the GenAISession with WebSocket and API connection details.

        Args:
            ws_url: WebSocket server URL, main bus for Agents communication.
            api_base_url: REST API base URL, to interact with Backend API.
            jwt_token: Optional JWT token for authorization.
            api_key: Optional API key for authorization.
            log_level: Logging level (e.g., logging.DEBUG, logging.INFO).
        """
        self.ws_url = ws_url
        self.api_base_url = api_base_url
        self.jwt_token = jwt_token
        self.api_key = api_key
        self.agent: Optional[Agent] = None
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
        """Returns authorization headers based on JWT or API key."""
        headers = {}
        if self.jwt_token:
            headers["X-Custom-Authorization"] = self.jwt_token
        if self.api_key:
            headers["API-KEY"] = self.api_key
        return headers

    @property
    def request_id(self) -> str:
        return self._request_id

    @request_id.setter
    def request_id(self, value: str) -> None:
        self._request_id = value

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    @property
    def agent_uuid(self) -> str:
        """Returns the agent UUID."""
        try:
            decoded = jwt.decode(
                self.jwt_token,
                options={"verify_signature": False},
                algorithms=["HS256"]
            )
            return decoded.get("sub")
        except jwt.exceptions.DecodeError:
            return

    @property
    def agent_id(self) -> str:
        """Returns the current agent ID (JWT or API key)."""
        return self.agent_uuid or self.api_key

    def bind(self, name: Optional[str] = None, description: Optional[str] = None) -> Callable:
        """
        Decorator to bind a Python function to an AI agent with an OpenAI-compatible schema.

        Args:
            name: Optional custom agent name, default is function name.
            description: Optional custom agent description, default is function docstring.
        """

        def decorator(func: Callable) -> Callable:
            function_schema = convert_to_openai_schema(func)
            function_description = function_schema.get("function", {}).get("description", "")
            function_name = function_schema.get("function", {}).get("description", "")

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            if description:
                function_schema["function"]["description"] = description

            function_schema["function"]["name"] = self.agent_uuid  # Name it by agent ID

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

    async def get_my_agents(self) -> list[dict]:
        """
        Fetches the list of previously registered agents from the API.

        Returns:
            List of agent metadata dictionaries.
        """
        url = f"{self.api_base_url}/api/agents"
        headers = {
            "Authorization": f"Bearer {self.jwt_token}"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()

    async def get_my_active_agents(self) -> list[dict]:
        """
        Fetches the list of previously registered agents from the API.

        Returns:
            List of agent metadata dictionaries.
        """
        url = f"{self.api_base_url}/api/agents/active"
        headers = {
            "Authorization": f"Bearer {self.jwt_token}"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()

    async def send(
        self,
        message: dict,
        client_id: str,
        close_timeout: int = None
    ) -> AgentResponse:
        """
        Sends a request to a remote agent via WebSocket and waits for a response.

        Args:
            message: The message dictionary to send.
            client_id: The target agent/client UUID.
            headers: Optional HTTP headers.
            close_timeout: Optional timeout for waiting for a response.

        Returns:
            AgentResponse object containing the result or error.
        """
        headers = {"x-custom-invoke-key": f"{self.agent_id}:{client_id}"}

        async with websockets.connect(self.ws_url, additional_headers=headers) as ws:
            init_message = json.dumps({
                "message_type": WSMessageType.AGENT_INVOKE.value,
                "agent_uuid": client_id,
                "request_payload": {**message},
                "request_metadata": {
                    "request_id": self.request_id,
                    "session_id": self.session_id,
                }
            })

            self.logger.debug(f"Sending message to: {client_id}")
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

    async def process_events(self, send_logs: bool = True) -> None:
        """
        Starts a long-running process to receive and handle agent requests from the WebSocket server.

        Args:
            send_logs: Whether to log requests and responses.
        """
        try:
            async with websockets.connect(self.ws_url, additional_headers=self.headers) as ws:
                agent_context = GenAIContext(
                    agent_uuid=self.agent_id,
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
                            task = asyncio.create_task(self._handle_agent_request(agent_context, ws, body, send_logs))
                            task.add_done_callback(self._handle_task_result)
                    except asyncio.CancelledError:
                        pass
                    except websockets.exceptions.ConnectionClosedError:
                        raise RouterInaccessibleException("Router service has disconnected. Please make sure it is running and accepting websocket messages")  # noqa: E501

                self._shutdown_event.clear()
                listener_task = asyncio.create_task(receive_messages())

                await self._shutdown_event.wait()
                listener_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await listener_task
        except websockets.exceptions.ConnectionClosedError:
            raise RouterInaccessibleException("Router service is not accessible. Please make sure it is running and accepting websocket messages")  # noqa: E501
    
    def _handle_task_result(self, task: asyncio.Task):
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
        send_logs: bool = True
    ):
        """
        Internal method to process a single agent request message.

        Args:
            agent_context: The context object for the current agent session.
            ws: WebSocket connection object.
            body: Incoming WebSocket message body.
            send_logs: Whether to log the request and response.
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

        try:
            start_time = time.perf_counter()

            if send_logs:
                logging_data = {"invoked_by": invoked_by, "request_payload": request_payload}
                agent_context.logger.info(logging_data)
                self.logger.debug(logging_data)

            # Call the bound function
            result = await self.agent.handler(agent_context=agent_context, **request_payload)

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            if send_logs:
                logging_data = {"execution_time": execution_time, "response": result}
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
            if send_logs:
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

        # Send response back over WebSocket
        async with self._send_lock:
            try:
                response = json.dumps(response)
            except TypeError as e:
                response = json.dumps({
                    "message_type": WSMessageType.AGENT_ERROR.value,
                    "execution_time": execution_time,
                    "invoked_by": invoked_by,
                    "error": {
                        "error_message": str(e)
                    }
                })

                if send_logs:
                    agent_context.logger.critical(traceback.format_exc())
                    self.logger.error("Failed to send response. Invalid data type.")
                    self.logger.error(e)

            await ws.send(response)
