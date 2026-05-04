"""Groq via OpenAI-compatible HTTP surface.

Mirrors ``examples/native-js/groq-openai-compatible.js``.

Groq exposes an OpenAI-compatible API. In xoin-py you model that with ``OpenAIProvider`` plus:

* a custom ``name`` (how the adapter appears in logs/errors),
* ``capabilities=Capabilities(structured_outputs=\"json-object\", embeddings=False)``,
* and the Groq ``base_url``.

Environment variables
---------------------
``GROQ_API_KEY``
    Required.
``GROQ_BASE_URL``
    Optional (defaults to Groq's OpenAI-compatible endpoint).
``GROQ_MODEL``
    Optional model id.

Run::

    python examples/groq_openai_compatible.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel, Field

from xoin import StructuredSpec, Xoin
from xoin.providers.base import Capabilities
from xoin.providers.openai import OpenAIProvider


class UserProfile(BaseModel):
    name: str = Field(description="Person name extracted from the sentence.")
    age: int = Field(description="Numeric age.")


async def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GROQ_API_KEY. Copy examples/.env.example to examples/.env.")

    base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    groq = OpenAIProvider(
        api_key=api_key,
        name="groq",
        base_url=base_url,
        default_model=model,
        default_embedding_model=None,
        capabilities=Capabilities(structured_outputs="json-object", embeddings=False),
    )

    async with Xoin(providers={"groq": groq}, default_provider="groq") as xoin:
        result = await xoin.generate(
            provider="groq",
            prompt='Extract a JSON object from: "Aarav is 35 years old."',
            structured=StructuredSpec(
                response_model=UserProfile,
                name="user_profile",
                mode="native",
            ),
        )

    payload: dict[str, Any] = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
