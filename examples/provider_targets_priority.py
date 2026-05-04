"""Explicit priority ordering with per-target models (``provider_targets``).

This mirrors the ``providerTargets`` pattern showcased throughout the xoin-js README:
each tuple carries a ``priority`` (lower runs earlier), optional ``model`` override, and the
provider registry key.

Retries apply **per generation attempt** before moving on to the next priority bucket.

Environment variables
---------------------
``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GROQ_API_KEY``
    Required for this sample.

Run::

    python examples/provider_targets_priority.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel, Field

from xoin import StructuredSpec, Xoin
from xoin.providers import AnthropicProvider, OpenAIProvider
from xoin.providers.base import Capabilities
from xoin.types import PriorityProviderTarget, RetryCfg


class OrderSummary(BaseModel):
    reference: str = Field(description="Short reference id invented if missing.")
    currency: str
    total: float


async def main() -> None:
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not openai_key or not anthropic_key or not groq_key:
        raise SystemExit("Requires OPENAI_API_KEY, ANTHROPIC_API_KEY, and GROQ_API_KEY.")

    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    groq = OpenAIProvider(
        api_key=groq_key,
        name="groq",
        base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        default_model=groq_model,
        default_embedding_model=None,
        capabilities=Capabilities(structured_outputs="json-object", embeddings=False),
    )

    async with Xoin(
        providers={
            "openai": OpenAIProvider(api_key=openai_key, default_model=openai_model),
            "anthropic": AnthropicProvider(api_key=anthropic_key, default_model=anthropic_model),
            "groq": groq,
        },
    ) as xoin:
        prompt = (
            'Extract an order summary as JSON with keys reference, currency, total from: '
            '"Order EU-998 paid EUR 129.90 yesterday."'
        )

        result = await xoin.generate(
            provider_targets=[
                PriorityProviderTarget(priority=1, provider="openai", model=openai_model),
                PriorityProviderTarget(priority=2, provider="anthropic", model=anthropic_model),
                PriorityProviderTarget(priority=3, provider="groq", model=groq_model),
            ],
            retry=RetryCfg(retries=1, delay_ms=250, backoff_multiplier=2.0),
            prompt=prompt,
            structured=StructuredSpec(
                response_model=OrderSummary,
                name="order_summary",
                mode="auto",
            ),
        )

    payload: dict[str, Any] = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
