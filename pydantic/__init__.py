"""Lightweight stand-in for pydantic used in offline test environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

MISSING = object()


class ValidationError(Exception):
    """Simple validation error placeholder."""


@dataclass
class FieldInfo:
    default: Any = MISSING
    default_factory: Optional[Callable[[], Any]] = None
    description: Optional[str] = None


def Field(
    default: Any = MISSING,
    *,
    default_factory: Optional[Callable[[], Any]] = None,
    description: Optional[str] = None,
) -> FieldInfo:
    """Return metadata for field definitions."""

    return FieldInfo(default=default, default_factory=default_factory, description=description)


class BaseModelMeta(type):
    def __new__(mcls, name, bases, namespace):
        annotations: Dict[str, Any] = namespace.get("__annotations__", {})
        fields: Dict[str, FieldInfo] = {}
        for field_name in annotations:
            value = namespace.get(field_name, MISSING)
            if isinstance(value, FieldInfo):
                fields[field_name] = value
                if value.default is not MISSING:
                    namespace[field_name] = value.default
                elif value.default_factory is None and field_name in namespace:
                    namespace.pop(field_name, None)
            else:
                fields[field_name] = FieldInfo(default=value)
        namespace["__fields__"] = fields
        return super().__new__(mcls, name, bases, namespace)


class BaseModel(metaclass=BaseModelMeta):
    """Minimal BaseModel implementation compatible with tests."""

    model_config: Dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        extra_keys = set(data) - set(self.__fields__)
        if extra_keys and self.model_config.get("extra") == "forbid":
            unknown = ", ".join(sorted(extra_keys))
            raise TypeError(f"Unexpected fields: {unknown}")
        for name, info in self.__fields__.items():
            if name in data:
                value = data[name]
            elif info.default is not MISSING:
                value = info.default
            elif info.default_factory is not None:
                value = info.default_factory()
            else:
                raise ValidationError(f"Missing required field: {name}")
            setattr(self, name, value)

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name in self.__fields__:
            value = getattr(self, name)
            result[name] = self._serialize_value(value)
        return result

    def model_dump_json(self, **kwargs: Any) -> str:
        return json.dumps(self.model_dump(**kwargs))

    @classmethod
    def model_json_schema(cls) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for name, info in cls.__fields__.items():
            properties[name] = {"title": name}
            if info.description:
                properties[name]["description"] = info.description
            if info.default is MISSING and info.default_factory is None:
                required.append(name)
        schema = {
            "title": cls.__name__,
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, list):
            return [BaseModel._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: BaseModel._serialize_value(val) for key, val in value.items()}
        return value
