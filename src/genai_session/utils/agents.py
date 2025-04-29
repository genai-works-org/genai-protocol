from typing import Callable, Union


class Agent:
    """
    Represents an AI agent with metadata and a function handler.

    Attributes:
        handler (Callable): The function that the agent will execute when invoked.
        description (str): A human-readable description of the agent's purpose.
        name (str): The name used to identify the agent.
        input_schema (dict): The OpenAI-compatible schema describing the agent's input parameters.
    """

    def __init__(self, handler: Callable, description: str, name: str, input_schema: dict) -> None:
        self.handler = handler
        self.description = description
        self.name = name
        self.input_schema = input_schema


class AgentResponse:
    """
    Represents a response returned by an agent after invocation.

    Attributes:
        is_success (bool): Indicates whether the agent executed successfully.
        execution_time (float): The time (in seconds) taken to execute the agent function.
        response (Union[dict, str]): The result of the function execution or an error message.
    """

    def __init__(self, is_success: bool, execution_time: float, response: Union[dict, str]) -> None:
        self.is_success = is_success
        self.execution_time = execution_time
        self.response = response
