import inspect
from types import UnionType, NoneType
from typing import Annotated, Any, Literal, Optional, get_origin, get_args, Union

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from genai_session.utils.exceptions import MissingContextParameterException


def convert_type(annotation) -> dict:
    """
    Converts a Python type annotation into a JSON Schema dictionary.
    Supports standard types, Pydantic models, typing constructs like Union, List, etc.

    Args:
        annotation: The type annotation to convert.

    Returns:
        dict: JSON Schema representation of the type.
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        return convert_type(args[0])

    if origin in (UnionType, Union):
        any_of_schemas = [convert_type(arg) for arg in args]
        return {"anyOf": any_of_schemas}

    if origin is Literal:
        return {"type": "string", "enum": list(args)}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is NoneType:
        return {"type": "null"}

    if origin is None and annotation in (list, tuple):
        return {"type": "array", "items": {}}

    if origin is None and annotation is dict:
        return {"type": "object", "additionalProperties": True}

    if origin is dict:
        _, val_type = args
        return {"type": "object", "additionalProperties": convert_type(val_type)}

    if origin is list:
        return {"type": "array", "items": convert_type(args[0])}

    if origin is tuple and args:
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": convert_type(args[0])}
        else:
            return {
                "type": "array",
                "items": [convert_type(arg) for arg in args],
                "minItems": len(args),
                "maxItems": len(args),
            }

    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return model_to_schema(annotation)

    return {}  # Fallback for unsupported types


def model_to_schema(model: type[BaseModel]) -> dict:
    """
    Converts a Pydantic model class to a JSON Schema dictionary.

    Args:
        model (type[BaseModel]): The Pydantic model class.

    Returns:
        dict: JSON Schema representation of the model.
    """
    properties = {}
    required = []

    for name, field in model.model_fields.items():
        schema = convert_type(field.annotation)
        if field.description:
            schema["description"] = field.description
        if field.default and field.default is not Ellipsis and field.default != PydanticUndefined:
            schema["default"] = field.default
        properties[name] = schema
        if field.is_required():
            required.append(name)

    return {
        "type": "object",
        "description": model.__doc__ or "",
        "properties": properties,
        "required": required,
    }


def extract_annotation_and_description(param) -> tuple[type, Optional[str], Optional[Any]]:
    """
    Extracts the annotation, description, and default value from a function parameter.

    Args:
        param: The function parameter object from `inspect`.

    Returns:
        tuple: (annotation, description, default value)
    """
    annotation = param.annotation
    description = None
    default = param.default

    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation, description = args[0], args[1]

    elif isinstance(default, FieldInfo):
        description = default.description
        if annotation is inspect.Parameter.empty:
            annotation = default.annotation
        default = None if default.default is Ellipsis else default.default

    return annotation, description, default


def convert_to_openai_schema(func) -> dict:
    """
    Converts a function's signature into an OpenAI-compatible function schema.

    Args:
        func: The function to convert.

    Returns:
        dict: OpenAI-style schema representing the function.

    Raises:
        MissingContextParameterException: If the function does not include 'agent_context'.
    """
    agent_context = "agent_context"
    sig = inspect.signature(func)

    if agent_context not in sig.parameters:
        raise MissingContextParameterException(
            f"GenAI Agent must contain a `{agent_context}` parameter. Please read the documentation for more details."
        )

    params_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    for name, param in sig.parameters.items():
        if name == agent_context:
            continue

        annotation, description, default = extract_annotation_and_description(param)
        param_schema = convert_type(annotation)

        if description:
            param_schema["description"] = description
        if default != inspect._empty:
            param_schema["default"] = default

        params_schema["properties"][name] = param_schema

        if default == inspect._empty:
            params_schema["required"].append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": params_schema,
        },
    }
