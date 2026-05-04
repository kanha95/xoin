"""Structured extraction with Mistral chat models.

Mirrors ``examples/native-js/mistral-structured-output.js``.

Environment variables
---------------------
``MISTRAL_API_KEY``
    Required.
``MISTRAL_MODEL``
    Optional override.

Run::

    python examples/mistral_structured_output.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel, Field

from xoin import StructuredOutput, Xoin
from xoin.providers import MistralProvider


class UserProfile(BaseModel):
    name: str = Field(description="Person name extracted from the sentence.")
    age: int = Field(description="Numeric age.")


async def main() -> None:
    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing MISTRAL_API_KEY. Copy examples/.env.example to examples/.env.")

    model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")

    async with Xoin(
        providers={
            "mistral": MistralProvider(api_key=api_key, default_model=model),
        },
        default_provider="mistral",
    ) as xoin:
        result = await xoin.generate(
            provider="mistral",
            prompt='Extract a JSON object from: "Nina is 28 years old."',
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
