from typing import Any

from pydantic import BaseModel

from xoin.errors import StructuredOutputError


def build_prompt_schema(name: str, schema: dict[str, Any] | None) -> str:
    base = f'Return ONLY valid JSON for "{name}". No markdown fences or extra text.'
    if not schema:
        return base
    import json

    return f"{base}\nJSON schema:\n{json.dumps(schema, indent=2)}"


def parse_json_like(text: str) -> Any:
    raw = text.strip()
    if not raw:
        raise StructuredOutputError("Structured output was empty.")
    import json
    import re

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw, re.I)
    if m:
        return parse_json_like(m.group(1))

    extracted = _extract_balanced_json(raw)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError as e:
            raise StructuredOutputError("Unable to parse structured output as JSON.") from e

    raise StructuredOutputError("Unable to locate valid JSON in the model response.")


def validate_response(model: type[BaseModel], raw_text: str, raw_obj: Any | None) -> BaseModel:
    candidate = raw_obj if raw_obj is not None else parse_json_like(raw_text)
    try:
        return model.model_validate(candidate)
    except Exception as e:
        raise StructuredOutputError("Structured output validation failed.") from e


def _extract_balanced_json(input: str) -> str | None:
    starts = ("{", "[")
    for i, ch in enumerate(input):
        if ch not in starts:
            continue
        stack: list[str] = []
        in_str = False
        esc = False
        for j in range(i, len(input)):
            c = input[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
                continue
            if c in "{[":
                stack.append(c)
            elif c in "}]":
                if not stack:
                    break
                op = stack.pop()
                if not _pairs(op, c):
                    break
                if not stack:
                    return input[i : j + 1]
    return None


def _pairs(opening: str, closing: str) -> bool:
    return (opening == "{" and closing == "}") or (opening == "[" and closing == "]")
