"""Structured extraction with DeepSeek chat models.

Mirrors ``examples/native-js/deepseek-structured-output.js``.

DeepSeek uses OpenAI-compatible chat payloads; xoin-py maps JSON-schema style requests to
``response_format: {\"type\": \"json_object\"}`` under the hood.

Environment variables
---------------------
``DEEPSEEK_API_KEY``
    Required.
``DEEPSEEK_MODEL``
    Optional override.

Run::

    python examples/deepseek_structured_output.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel, Field

from xoin import StructuredSpec, Xoin
from xoin.providers import DeepSeekProvider


class UserProfile(BaseModel):
    name: str = Field(description="Person name extracted from the sentence.")
    age: int = Field(description="Numeric age.")


async def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY. Copy examples/.env.example to examples/.env.")

    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    async with Xoin(
        providers={
            "deepseek": DeepSeekProvider(api_key=api_key, default_model=model),
        },
        default_provider="deepseek",
    ) as xoin:
        result = await xoin.generate(
            provider="deepseek",
            prompt='Extract a JSON object from: "Kabir is 33 years old."',
            structured=StructuredSpec(
                response_model=UserProfile,
                name="user_profile",
                mode="auto",
            ),
        )

    payload: dict[str, Any] = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
