from genai_session.utils.file_manager import FileManager
from genai_session.utils.logging_manager import ContextLogger
from websockets.asyncio.client import ClientConnection


class GenAIContext:
    """
    Encapsulates contextual metadata and utilities for a GenAI agent session.

    This class provides easy access to the agent's request/session context,
    logging capabilities, and file handling APIs.

    Attributes:
        agent_uuid (str): Unique identifier for the agent.
        websocket (ClientConnection): The WebSocket connection used for communication.
        api_base_url (str): Base URL of the backend API.
    """

    def __init__(self, agent_uuid: str, jwt_token: str, websocket: ClientConnection, api_base_url: str):
        self.agent_uuid = agent_uuid
        self.websocket = websocket
        self.api_base_url = api_base_url
        self.jwt_token = jwt_token
        self._request_id = ""
        self._session_id = ""

    @property
    def request_id(self) -> str:
        """Get the current request ID."""
        return self._request_id

    @request_id.setter
    def request_id(self, value: str) -> None:
        """Set the request ID."""
        self._request_id = value

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        """Set the session ID."""
        self._session_id = value

    @property
    def logger(self) -> ContextLogger:
        """
        Returns a context-aware logger that includes the agent UUID,
        request ID, session ID, and websocket reference for logging.
        """
        return ContextLogger(
            agent_uuid=self.agent_uuid,
            request_id=self.request_id,
            session_id=self.session_id,
            websocket=self.websocket,
        )

    @property
    def files(self) -> FileManager:
        """
        Returns a file manager instance scoped to the current request and session,
        allowing the agent to access uploaded or generated files.
        """
        return FileManager(
            api_base_url=self.api_base_url,
            request_id=self.request_id,
            session_id=self.session_id,
            jwt_token=self.jwt_token
        )
