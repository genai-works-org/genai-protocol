import asyncio
import json
from collections import defaultdict
from io import BytesIO
from logging import Logger

from websockets.asyncio.client import ClientConnection

from genai_session.utils.naming_enums import WSMessageType


class ContextLogger:
    """
    A contextual async logger that sends structured log messages to a WebSocket client.

    Useful in distributed or remote environments where centralized, real-time logging
    is needed for debugging and tracing agent activity.
    """

    def __init__(
        self,
        agent_uuid: str,
        request_id: str,
        session_id: str,
        internal_logger: Logger,
        websocket: ClientConnection = None,
        invoked_by: str = "",
    ):
        """
        Initialize the logger with request/session context and WebSocket connection.

        Args:
            agent_uuid (str): Unique ID of the AI agent performing the task.
            request_id (str): Unique identifier of the current request.
            session_id (str): Identifier of the session the request belongs to.
            websocket (ClientConnection): WebSocket used to send log messages.
            invoked_by (str, optional): ID of the invoking agent, if applicable.
        """
        self.agent_uuid = agent_uuid
        self.websocket = websocket
        self.request_id = request_id
        self.session_id = session_id
        self.invoked_by = invoked_by
        self.pending_log_tasks = defaultdict(list)
        self.internal_logger = internal_logger

    async def _message_logging(self, message: str | dict, log_level: str = "info"):
        """
        Internal coroutine to send a log message through the WebSocket.

        Args:
            message (str | dict): The message to log. If a dictionary, it will be serialized to JSON.
            log_level (str): The severity level of the log (e.g., "info", "error").
        """
        if not self.websocket:
            return

        await self.websocket.send(
            json.dumps({
                "message_type": WSMessageType.AGENT_LOG.value,
                "log_message": self._convert_message(message),
                "log_level": log_level,
                "agent_uuid": self.agent_uuid,
                "request_id": self.request_id,
                "session_id": self.session_id,
                "invoked_by": self.invoked_by,
            })
        )

    def __convert_value(self, obj):
        """
        Recursively converts complex objects into JSON-serializable types.

        Args:
            obj: Any Python object.

        Returns:
            A JSON-serializable version of the object.
        """
        if isinstance(obj, dict):
            return {k: self.__convert_value(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.__convert_value(i) for i in obj]
        elif isinstance(obj, BytesIO):
            return obj.getvalue().decode("utf-8")
        elif isinstance(obj, bytes):
            return obj.decode("utf-8")
        else:
            return obj

    def _convert_message(self, message: str | dict) -> str:
        """
        Converts a log message into a JSON string, handling dicts recursively.

        Args:
            message (str | dict): The message to convert.

        Returns:
            str: JSON-encoded message string.
        """
        if isinstance(message, dict):
            try:
                return json.dumps(self.__convert_value(message))
            except TypeError as e:
                return json.dumps({"error": str(e)})
        return message

    # The following methods create a task that logs a message with the appropriate log level

    def debug(self, message):
        """
        Log a debug-level message.

        Args:
            message (str | dict): The message to log.
        """
        self.internal_logger.debug(message)
        self.pending_log_tasks[self.request_id].append(("debug", message))

    def info(self, message):
        """
        Log an info-level message.

        Args:
            message (str | dict): The message to log.
        """
        self.internal_logger.info(message)
        self.pending_log_tasks[self.request_id].append(("info", message))

    def warning(self, message):
        """
        Log a warning-level message.

        Args:
            message (str | dict): The message to log.
        """
        self.internal_logger.warning(message)
        self.pending_log_tasks[self.request_id].append(("warning", message))

    def error(self, message):
        """
        Log an error-level message.

        Args:
            message (str | dict): The message to log.
        """
        self.internal_logger.error(message)
        self.pending_log_tasks[self.request_id].append(("error", message))

    def critical(self, message):
        """
        Log a critical-level message.

        Args:
            message (str | dict): The message to log.
        """
        self.internal_logger.critical(message)
        self.pending_log_tasks[self.request_id].append(("critical", message))

    async def flush_logs(self):
        if logs := self.pending_log_tasks.get(self.request_id, []):
            tasks = [
                self._message_logging(message=msg, log_level=level)
                for level, msg in logs
            ]
            await asyncio.gather(*tasks)
            self.pending_log_tasks.pop(self.request_id)
