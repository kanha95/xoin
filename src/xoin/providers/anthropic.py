from typing import Any

import httpx

from xoin.http import fetch_json
from xoin.providers.base import (
    Capabilities,
    ChatCompletionParameters,
    EmbeddingParameters,
    JsonSchemaResponseFormat,
    ProviderChatResponse,
    ProviderEmbeddingResponse,
)
from xoin.types import ChatMessage, Usage
from xoin.errors import EmbeddingError, ProviderExecutionError


class AnthropicProvider:
    name = "anthropic"
    capabilities = Capabilities(structured_outputs="json-schema", embeddings=False)

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.anthropic.com/v1",
        default_model: str | None = "claude-sonnet-4-20250514",
        headers: dict[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self.default_model = default_model
        self.default_embedding_model = None
        self._headers_extra = headers or {}

    async def generate(
        self, client: httpx.AsyncClient, parameters: ChatCompletionParameters
    ) -> ProviderChatResponse:
        sys_parts = [m.content for m in parameters.messages if m.role == "system"]
        conv = [m for m in parameters.messages if m.role != "system"]

        body: dict[str, Any] = {
            "model": parameters.model,
            "max_tokens": parameters.max_tokens if parameters.max_tokens is not None else 1024,
            "messages": [_anth_msg(m) for m in conv],
            **parameters.provider_options,
        }
        if parameters.temperature is not None:
            body["temperature"] = parameters.temperature
        if sys_parts:
            body["system"] = "\n\n".join(sys_parts)
        body.update(_tool_payload(parameters))

        data = await fetch_json(
            client,
            self.name,
            f"{self._base}/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                **self._headers_extra,
            },
            body=body,
            timeout=parameters.timeout,
        )

        blocks = data.get("content") or []
        tool = next((b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"), None)
        texts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        if tool:
            import json

            text = json.dumps(tool.get("input"))
            structured = tool.get("input")
        else:
            text = "\n".join(texts).strip()
            structured = None

        if not text:
            raise ProviderExecutionError("anthropic returned an empty response.", self.name, parameters.model)

        return ProviderChatResponse(
            model=data.get("model") or parameters.model,
            text=text,
            structured_data=structured,
            usage=_usage(data.get("usage")),
            finish_reason=data.get("stop_reason"),
            raw=data,
        )

    async def embed(self, _client: httpx.AsyncClient, _parameters: EmbeddingParameters) -> ProviderEmbeddingResponse:
        raise EmbeddingError('Provider "anthropic" does not support embeddings in xoin defaults.')


def _anth_msg(m: ChatMessage) -> dict[str, str]:
    role = "assistant" if m.role == "tool" else m.role
    return {"role": role, "content": m.content}


def _tool_payload(parameters: ChatCompletionParameters) -> dict[str, Any]:
    rf = parameters.response_format
    if not isinstance(rf, JsonSchemaResponseFormat):
        return {}
    return {
        "tools": [
            {
                "name": rf.name,
                "description": rf.description or "Return structured JSON output.",
                "input_schema": rf.schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": rf.name},
    }


def _usage(u: dict[str, Any] | None) -> Usage | None:
    if not u:
        return None
    inp = u.get("input_tokens")
    out = u.get("output_tokens")
    tot = None
    if inp is not None or out is not None:
        tot = (inp or 0) + (out or 0)
    return Usage(input_tokens=inp, output_tokens=out, total_tokens=tot)
