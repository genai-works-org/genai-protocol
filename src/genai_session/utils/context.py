from websockets.asyncio.client import ClientConnection

from genai_session.utils.file_manager import FileManager
from genai_session.utils.logging_manager import ContextLogger


class GenAIContext:
    """
    Encapsulates contextual metadata and utilities for a GenAI agent session.

    Provides access to request/session-specific metadata, logging, and file handling APIs
    to support the execution of AI agents in a conversational environment.

    Attributes:
        agent_uuid (str): Unique identifier of the current agent.
        websocket (ClientConnection): WebSocket connection used for real-time communication.
        api_base_url (str): Base URL of the backend API for file or agent-related interactions.
        jwt_token (str): JWT token used for authenticated API calls.
    """

    def __init__(self, agent_uuid: str, jwt_token: str, api_base_url: str, websocket: ClientConnection = None):
        """
        Initializes the GenAIContext.

        Args:
            agent_uuid (str): The unique identifier for the agent.
            jwt_token (str): JWT used for authenticated requests to the API.
            api_base_url (str): The base URL for backend API access.
            websocket (ClientConnection, optional): The WebSocket connection used for the session.
        """
        self.agent_uuid = agent_uuid
        self.websocket = websocket
        self.api_base_url = api_base_url
        self.jwt_token = jwt_token
        self._request_id = ""
        self._session_id = ""
        self._invoked_by = ""

        self._logger = None
        self._file_manager = None

    @property
    def request_id(self) -> str:
        """Gets the current request ID."""
        return self._request_id

    @request_id.setter
    def request_id(self, value: str) -> None:
        """Sets the current request ID."""
        self._request_id = value

    @property
    def session_id(self) -> str:
        """Gets the current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        """Sets the current session ID."""
        self._session_id = value

    @property
    def invoked_by(self) -> str:
        """Gets the ID of the agent that invoked the current request."""
        return self._invoked_by

    @invoked_by.setter
    def invoked_by(self, value: str) -> None:
        """Sets the ID of the agent that invoked the current request."""
        self._invoked_by = value

    @property
    def logger(self) -> ContextLogger:
        """
        Returns:
            ContextLogger: A context-aware logger that includes agent/session/request metadata.
        """
        return self._logger

    @logger.setter
    def logger(self, value: ContextLogger) -> None:
        self._logger = value

    @property
    def files(self) -> FileManager:
        """
        Returns:
            FileManager: A file manager scoped to the current request and session for
            accessing uploaded or generated files.
        """
        return FileManager(
                api_base_url=self.api_base_url,
                request_id=self.request_id,
                session_id=self.session_id,
                jwt_token=self.jwt_token
            )
