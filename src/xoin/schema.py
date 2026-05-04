from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def response_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for ``StructuredOutput`` / native provider formats.

    OpenAI Chat Completions ``response_format`` with ``strict: true`` requires every
    ``object`` to set ``additionalProperties`` to ``false``. Pydantic's default schema
    omits that key, which yields HTTP 400 from the API.
    """

    schema = model.model_json_schema()
    _openai_strict_object_shapes(schema)
    return schema


def _openai_strict_object_shapes(node: Any) -> None:
    """Recursively ensure object schemas satisfy OpenAI structured-output strict rules."""

    if isinstance(node, list):
        for item in node:
            _openai_strict_object_shapes(item)
        return
    if not isinstance(node, dict):
        return

    props = node.get("properties")
    if node.get("type") == "object" or (isinstance(props, dict) and props):
        node.setdefault("type", "object")
        ap = node.get("additionalProperties")
        if ap is True or ap is None:
            node["additionalProperties"] = False
        elif isinstance(ap, dict):
            _openai_strict_object_shapes(ap)

    for key, val in node.items():
        if key == "properties" and isinstance(val, dict):
            for child in val.values():
                _openai_strict_object_shapes(child)
        elif key == "$defs" and isinstance(val, dict):
            for child in val.values():
                _openai_strict_object_shapes(child)
        elif key in ("items", "prefixItems", "contains", "not") and isinstance(val, dict):
            _openai_strict_object_shapes(val)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(val, list):
            for child in val:
                _openai_strict_object_shapes(child)
        elif key == "if" or key == "then" or key == "else":
            if isinstance(val, dict):
                _openai_strict_object_shapes(val)
