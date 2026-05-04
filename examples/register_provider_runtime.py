"""Attach providers after ``Xoin`` is constructed.

JavaScript exposes ``xoin.registerProvider(name, factory)``. Python keeps things explicit:
``Xoin.register_provider`` accepts an already-built adapter instance.

This pattern helps when credentials arrive asynchronously (vault rotation, feature flags, etc.).

Environment variables
---------------------
``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``
    Required.

Run::

    python examples/register_provider_runtime.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel, Field

from xoin import StructuredOutput, Xoin
from xoin.providers import AnthropicProvider, OpenAIProvider


class Handshake(BaseModel):
    greeting: str = Field(description="Short polite greeting.")


async def main() -> None:
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not openai_key or not anthropic_key:
        raise SystemExit("Requires OPENAI_API_KEY and ANTHROPIC_API_KEY.")

    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    async with Xoin(
        providers={
            "openai": OpenAIProvider(api_key=openai_key, default_model=openai_model),
        },
        default_provider="openai",
    ) as xoin:
        # Imagine this arrives milliseconds later from your secret manager.
        xoin.register_provider(
            "anthropic",
            AnthropicProvider(api_key=anthropic_key, default_model=anthropic_model),
        )

        result = await xoin.generate(
            provider="openai",
            provider_order=["anthropic"],
            prompt="Say hello to the engineering team in one polished sentence.",
            structured=StructuredOutput(
                response_model=Handshake,
                name="handshake",
                mode="prompted",
            ),
        )

    payload: dict[str, Any] = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
