"""Structured extraction with Anthropic Claude.

This example mirrors ``examples/native-js/anthropic-structured-output.js``.

Environment variables
---------------------
``ANTHROPIC_API_KEY``
    Required.
``ANTHROPIC_MODEL``
    Optional override (defaults to a recent Sonnet snapshot).

Run::

    python examples/anthropic_structured_output.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel, Field

from xoin import StructuredOutput, Xoin
from xoin.providers import AnthropicProvider


class UserProfile(BaseModel):
    """Structured object validated after the model responds."""

    name: str = Field(description="Person name extracted from the sentence.")
    age: int = Field(description="Numeric age.")


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY. Copy examples/.env.example to examples/.env.")

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # ``async with`` closes the internally-managed ``httpx.AsyncClient`` when done.
    async with Xoin(
        providers={
            "anthropic": AnthropicProvider(api_key=api_key, default_model=model),
        },
        default_provider="anthropic",
    ) as xoin:
        result = await xoin.generate(
            provider="anthropic",
            prompt='Extract a JSON object from: "Ria is 25 years old."',
            structured=StructuredOutput(
                response_model=UserProfile,
                name="user_profile",
                description="Validated profile extracted from noisy natural language.",
                mode="auto",
            ),
        )

    payload: dict[str, Any] = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
