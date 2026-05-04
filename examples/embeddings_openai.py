"""Create OpenAI embeddings with xoin-py.

Mirrors ``examples/native-js/embeddings-openai.js``.

Environment variables
---------------------
``OPENAI_API_KEY``
    Required.
``OPENAI_EMBEDDING_MODEL``
    Optional embedding model id.

Run::

    python examples/embeddings_openai.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from xoin import Xoin
from xoin.providers import OpenAIProvider


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Copy examples/.env.example to examples/.env.")

    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    async with Xoin(
        providers={
            "openai": OpenAIProvider(
                api_key=api_key,
                default_embedding_model=embedding_model,
            ),
        },
        default_provider="openai",
    ) as xoin:
        result = await xoin.embed(
            provider="openai",
            model=embedding_model,
            input=[
                "semantic search",
                "vector database",
            ],
        )

    preview = [vector[:5] for vector in result.embeddings]
    payload: dict[str, Any] = {
        "provider": result.provider,
        "model": result.model,
        "embedding_count": len(result.embeddings),
        "dimensions": len(result.embeddings[0]) if result.embeddings else 0,
        "preview": preview,
        "usage": result.usage.model_dump(mode="json") if result.usage else None,
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
