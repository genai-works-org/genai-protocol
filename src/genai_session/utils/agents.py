from typing import Callable, Union


class Agent:
    """
    Represents an AI agent with metadata and an executable handler function.

    Attributes:
        handler (Callable): The function that will be executed when the agent is invoked.
        description (str): A human-readable description of the agent's functionality.
        name (str): A unique name identifier for the agent.
        input_schema (dict): An OpenAI-compatible schema defining the agent's expected input parameters.
    """

    def __init__(self, handler: Callable, description: str, name: str, input_schema: dict) -> None:
        """
        Initializes an Agent instance.

        Args:
            handler (Callable): The function to be executed by the agent.
            description (str): Description of what the agent does.
            name (str): The agent's identifier name.
            input_schema (dict): Schema defining the expected inputs for the agent.
        """
        self.handler = handler
        self.description = description
        self.name = name
        self.input_schema = input_schema


class AgentResponse:
    """
    Represents the outcome of an agent's execution.

    Attributes:
        is_success (bool): Indicates whether the agent's execution was successful.
        execution_time (float): Time taken to execute the agent's function, in seconds.
        response (Union[dict, str]): The output of the agent's function or an error message.
    """

    def __init__(self, is_success: bool, execution_time: float, response: Union[dict, str]) -> None:
        """
        Initializes an AgentResponse instance.

        Args:
            is_success (bool): True if the agent executed successfully, False otherwise.
            execution_time (float): Duration of the agent's function execution in seconds.
            response (Union[dict, str]): The result of the execution or an error message.
        """
        self.is_success = is_success
        self.execution_time = execution_time
        self.response = response
