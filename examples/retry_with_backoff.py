"""Demonstrate ``RetryCfg`` for transient ``ProviderExecutionError`` failures.

Retries fire **before** xoin-py advances to the next provider in a fallback chain.

Environment variables
---------------------
``OPENAI_API_KEY``
    Required.

Run::

    python examples/retry_with_backoff.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel, Field

from xoin import StructuredSpec, Xoin
from xoin.providers import OpenAIProvider
from xoin.types import RetryCfg


class UserProfile(BaseModel):
    name: str = Field(description="Person name extracted from the sentence.")
    age: int = Field(description="Numeric age.")


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    async with Xoin(
        providers={
            "openai": OpenAIProvider(api_key=api_key, default_model=model),
        },
        default_provider="openai",
        # Defaults apply when a call omits ``retry=``.
        retry=RetryCfg(retries=2, delay_ms=400, backoff_multiplier=2.0),
    ) as xoin:
        result = await xoin.generate(
            provider="openai",
            prompt='Extract a JSON object from: "Meera is 29 years old."',
            structured=StructuredSpec(
                response_model=UserProfile,
                name="user_profile",
                mode="auto",
            ),
            # Per-call overrides are allowed when you need tighter knobs for a sensitive query.
            retry=RetryCfg(retries=3, delay_ms=150, backoff_multiplier=1.8),
        )

    payload: dict[str, Any] = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
