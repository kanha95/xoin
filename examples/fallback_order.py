"""Sequential failover across providers using ``provider_order``.

Mirrors ``examples/native-js/fallback-order.js``.

``provider_order`` extends the chain **after** the explicit ``provider`` argument and before the
client-level ``default_provider`` / ``fallback_providers`` configuration.

Environment variables
---------------------
``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``
    Required.

Run::

    python examples/fallback_order.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel, Field

from xoin import StructuredSpec, Xoin
from xoin.providers import AnthropicProvider, OpenAIProvider


class UserProfile(BaseModel):
    name: str = Field(description="Person name extracted from the sentence.")
    age: int = Field(description="Numeric age.")


async def main() -> None:
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not openai_key or not anthropic_key:
        raise SystemExit("Requires OPENAI_API_KEY and ANTHROPIC_API_KEY.")

    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    async with Xoin(
        default_provider="openai",
        fallback_providers=["anthropic"],
        providers={
            "openai": OpenAIProvider(api_key=openai_key, default_model=openai_model),
            "anthropic": AnthropicProvider(api_key=anthropic_key, default_model=anthropic_model),
        },
    ) as xoin:
        prompt = 'Extract a JSON object from: "Ava is 31 years old."'
        result = await xoin.generate(
            provider="openai",
            provider_order=["anthropic"],
            prompt=prompt,
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
