"""Fan-out the same generation request to multiple providers concurrently.

This mirrors ``examples/native-js/generate-many.js``.

``generate_many`` schedules **independent** completions—there is **no** automatic failover between
targets. Use ``generate`` + ``provider_order`` / ``provider_targets`` when you need sequential
fallback semantics.

Environment variables
---------------------
``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``
    Required for this sample.

Run::

    python examples/generate_many.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from xoin import Xoin
from xoin.providers import AnthropicProvider, OpenAIProvider
from xoin.types import GenManyTarget


async def main() -> None:
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not openai_key or not anthropic_key:
        raise SystemExit("Requires OPENAI_API_KEY and ANTHROPIC_API_KEY.")

    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    prompt = "Summarize why structured outputs are useful in exactly 2 short bullets."

    async with Xoin(
        providers={
            "openai": OpenAIProvider(api_key=openai_key, default_model=openai_model),
            "anthropic": AnthropicProvider(api_key=anthropic_key, default_model=anthropic_model),
        },
    ) as xoin:
        # asyncio.gather happens internally—your code awaits once.
        results = await xoin.generate_many(
            targets=[
                GenManyTarget(provider="openai", model="gpt-4o-mini"),
                GenManyTarget(provider="anthropic", model="claude-sonnet-4-20250514"),
            ],
            prompt=prompt,
        )

    payload: list[dict[str, Any]] = [item.model_dump(mode="json") for item in results]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
