"""Inline named templates configured on the ``Xoin`` client.

This mirrors how JavaScript registers ``templates: { ... }`` on ``createXoin`` and later references
``templateId``.

Instead of reading from disk, you embed reusable prompt fragments directly in application config.

Environment variables
---------------------
``OPENAI_API_KEY``
    Required.

Run::

    python examples/named_templates_registry.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel, Field

from xoin import StructuredOutput, TemplateDefinition, Xoin
from xoin.providers import OpenAIProvider


class UserProfile(BaseModel):
    name: str = Field(description="Person name extracted from the text.")
    age: int = Field(description="Numeric age.")


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    async with Xoin(
        templates={
            "extract_user": TemplateDefinition(
                template='Extract a JSON object with keys name (string) and age (number) from: "{{text}}"',
                defaults={"text": ""},
                description="Registry-backed extraction prompt used via template_id.",
            ),
        },
        providers={
            "openai": OpenAIProvider(api_key=api_key, default_model=model),
        },
        default_provider="openai",
    ) as xoin:
        result = await xoin.generate(
            provider="openai",
            template_id="extract_user",
            variables={
                "text": "Ravi is 34 years old.",
            },
            structured=StructuredOutput(
                response_model=UserProfile,
                name="user_profile",
                mode="auto",
            ),
        )

    payload: dict[str, Any] = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
