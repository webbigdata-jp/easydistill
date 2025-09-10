# functions/exceptions.py
"""Custom exceptions for the agent system."""

class AgentException(Exception):
    """Base exception class for agent-related errors."""
    def __init__(self, message, task_id=None, node_name=None):
        super().__init__(message)
        self.task_id = task_id
        self.node_name = node_name


class LLMExecutionError(AgentException):
    """Raised when there's an error executing an LLM call."""
    def __init__(self, message, task_id=None, node_name=None, model_name=None):
        super().__init__(message, task_id, node_name)
        self.model_name = model_name


class CodeExecutionError(AgentException):
    """Raised when there's an error executing Python code."""
    def __init__(self, message, task_id=None, node_name=None, code=None):
        super().__init__(message, task_id, node_name)
        self.code = code


class EvaluationError(AgentException):
    """Raised when there's an error during answer evaluation."""
    pass


class RepairError(AgentException):
    """Raised when there's an error during the repair process."""
    pass