import asyncio
import json
from io import BytesIO

from websockets.asyncio.client import ClientConnection

from genai_session.utils.naming_enums import WSMessageType


class ContextLogger:
    """
    A contextual async logger that sends structured log messages to a WebSocket client.
    Useful in distributed/remote environments where centralized logging is essential.
    """

    def __init__(
            self,
            agent_uuid: str,
            request_id: str,
            session_id: str,
            websocket: ClientConnection
    ):
        """
        Initialize the logger with context and WebSocket connection.

        Args:
            agent_uuid (str): Unique ID of the AI agent.
            request_id (str): Request identifier for tracking logs.
            session_id (str): Session identifier to group related logs.
            websocket (ClientConnection): WebSocket connection to send logs to.
        """
        self.agent_uuid = agent_uuid
        self.websocket = websocket
        self.request_id = request_id
        self.session_id = session_id

    async def _message_logging(self, message: str | dict, log_level: str = "info"):
        """
        Internal coroutine to send a log message through WebSocket.

        Args:
            message (str | dict): The log message content.
            log_level (str): Logging severity level.
        """
        await self.websocket.send(
            json.dumps({
                "message_type": WSMessageType.AGENT_LOG.value,
                "log_message": self._convert_message(message),
                "log_level": log_level,
                "agent_uuid": self.agent_uuid,
                "request_id": self.request_id,
                "session_id": self.session_id,
            })
        )

    def __convert_value(self, obj):
        """
        Recursively convert objects to serializable types for JSON logging.

        Args:
            obj: The object to convert.

        Returns:
            A JSON-serializable version of the input object.
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
        Convert a message (string or dictionary) to a JSON string.

        Args:
            message (str | dict): The log message to convert.

        Returns:
            str: JSON string representation of the message.
        """
        if isinstance(message, dict):
            try:
                return json.dumps(self.__convert_value(message))
            except TypeError as e:
                return json.dumps({"error": str(e)})
        return message

    # The following methods create a task that logs a message with the appropriate log level

    def debug(self, message):
        """Log a debug-level message."""
        asyncio.create_task(self._message_logging(message=message, log_level="debug"))

    def info(self, message):
        """Log an info-level message."""
        asyncio.create_task(self._message_logging(message=message, log_level="info"))

    def warning(self, message):
        """Log a warning-level message."""
        asyncio.create_task(self._message_logging(message=message, log_level="warning"))

    def error(self, message):
        """Log an error-level message."""
        asyncio.create_task(self._message_logging(message=message, log_level="error"))

    def critical(self, message):
        """Log a critical-level message."""
        asyncio.create_task(self._message_logging(message=message, log_level="critical"))
