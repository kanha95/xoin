import json
from typing import Any

import httpx

from xoin.errors import ProviderExecutionError


async def fetch_json(
    client: httpx.AsyncClient,
    provider: str,
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: httpx.Timeout | float | None,
) -> Any:
    try:
        r = await client.post(url, headers=headers, json=body, timeout=timeout)
    except httpx.RequestError as e:
        raise ProviderExecutionError(f"{provider} request failed: {e}", provider) from e

    try:
        data = r.json()
    except json.JSONDecodeError as e:
        raw = r.text
        snippet = raw if len(raw) <= 400 else raw[:400] + "..."
        raise ProviderExecutionError(f"{provider} returned invalid JSON: {snippet}", provider) from e

    if not r.is_success:
        raw = r.text
        snippet = raw if len(raw) <= 400 else raw[:400] + "..."
        raise ProviderExecutionError(f"{provider} HTTP {r.status_code}: {snippet}", provider)

    return data
