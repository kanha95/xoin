"""Load a YAML template from disk and substitute ``{{variables}}``.

Mirrors ``examples/native-js/template-file.js``.

xoin-py resolves ``template_file`` using the same semantics as the JavaScript client:

* ``*.yaml`` / ``*.yml`` → dictionary with a ``template`` string + optional ``defaults``
* ``*.json`` → same shape
* anything else → treated as the raw template body

This requires **PyYAML** (installed via ``pip install -e ".[examples]"``).

Environment variables
---------------------
``OPENAI_API_KEY``
    Required.

Run::

    python examples/template_file.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from xoin import StructuredSpec, Xoin
from xoin.providers import OpenAIProvider


class UserProfile(BaseModel):
    name: str = Field(description="Person name extracted from the template body.")
    age: int = Field(description="Numeric age.")


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    template_path = Path(__file__).resolve().parent / "templates" / "extract_user.yaml"
    if not template_path.is_file():
        raise SystemExit(f"Missing template file at {template_path}")

    async with Xoin(
        providers={
            "openai": OpenAIProvider(api_key=api_key, default_model=model),
        },
        default_provider="openai",
    ) as xoin:
        result = await xoin.generate(
            provider="openai",
            template_file=template_path,
            variables={
                "user_query": "Nina is 28 years old and lives in Pune.",
            },
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
