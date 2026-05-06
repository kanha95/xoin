"""Gemini structured output using the built-in provider adapter.

Environment variables
---------------------
``GEMINI_API_KEY``
    Required.
``GEMINI_MODEL``
    Optional chat model id (defaults to ``gemini-2.5-flash``).
``GEMINI_BASE_URL``
    Optional override for OpenAI-compatible Gemini endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from xoin import StructuredOutput, Xoin
from xoin.providers import GeminiProvider


class CompanyProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(validation_alias=AliasChoices("name", "Name"))
    industry: str = Field(validation_alias=AliasChoices("industry", "Industry"))
    founded_year: int = Field(
        validation_alias=AliasChoices("founded_year", "foundedYear", "founded", "Founded"),
    )


PROMPT = """
Extract a structured company profile:
- Name: Acme Robotics
- Industry: Industrial automation
- Founded: 2018

Return JSON keys exactly as:
- name
- industry
- founded_year
"""


async def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY.")

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")

    async with Xoin(
        providers={
            "gemini": GeminiProvider(api_key=api_key, default_model=model, base_url=base_url),
        },
        default_provider="gemini",
    ) as xoin:
        result = await xoin.generate(
            provider="gemini",
            prompt=PROMPT.strip(),
            structured=StructuredOutput(response_model=CompanyProfile, name="company_profile"),
        )

    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
