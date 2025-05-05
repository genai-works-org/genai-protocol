class BaseAIAgentException(Exception):
    """Base class for all AI agent exceptions."""
    pass


class FailedFileUploadException(BaseAIAgentException):
    """Exception raised when file upload fails."""
    pass


class FileNotFoundException(BaseAIAgentException):
    """Exception raised when a file is not found."""
    pass


class MissingContextParameterException(BaseAIAgentException):
    """Exception raised when a required context parameter is missing."""
    pass


class InvalidAgentUUIDException(BaseAIAgentException):
    """Exception raised when an invalid agent UUID is provided."""
    pass


class InvalidJSONFormatException(BaseAIAgentException):
    """Exception raised when the JSON format is invalid."""
    pass


class NoRequestPayloadException(BaseAIAgentException):
    """Exception raised when no request payload is provided."""
    pass


class AgentNotActiveException(BaseAIAgentException):
    """Exception raised when the agent is not active."""
    pass


class IncorrectFileInputException(BaseAIAgentException):
    """Exception raised when the file input is incorrect."""
    pass


class RouterInaccessibleException(BaseAIAgentException):
    """Exception raised when `router` service has disconnected"""
    pass
