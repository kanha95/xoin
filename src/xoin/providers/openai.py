from typing import Any

import httpx

from xoin.errors import EmbeddingError, ProviderExecutionError
from xoin.http import fetch_json
from xoin.providers.base import (
    Capabilities,
    ChatCompletionParameters,
    EmbeddingParameters,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    PlainTextResponseFormat,
    ProviderChatResponse,
    ProviderEmbeddingResponse,
)
from xoin.types import ChatMessage, Usage


class OpenAIProvider:
    """OpenAI Chat Completions + embeddings (or any OpenAI-compatible HTTP surface)."""

    def __init__(
        self,
        api_key: str,
        *,
        name: str = "openai",
        base_url: str = "https://api.openai.com/v1",
        default_model: str | None = "gpt-4o-mini",
        default_embedding_model: str | None = "text-embedding-3-small",
        capabilities: Capabilities | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.capabilities = capabilities or Capabilities(structured_outputs="json-schema", embeddings=True)
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self.default_model = default_model
        self.default_embedding_model = default_embedding_model
        self._headers_extra = headers or {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json", **self._headers_extra}

    async def generate(
        self, client: httpx.AsyncClient, parameters: ChatCompletionParameters
    ) -> ProviderChatResponse:
        payload: dict[str, Any] = {
            "model": parameters.model,
            "messages": [_msg(m) for m in parameters.messages],
            **parameters.provider_options,
        }
        if parameters.temperature is not None:
            payload["temperature"] = parameters.temperature
        if parameters.max_tokens is not None:
            payload["max_tokens"] = parameters.max_tokens
        rf = _response_format(parameters.response_format)
        if rf is not None:
            payload["response_format"] = rf

        data = await fetch_json(
            client,
            self.name,
            f"{self._base}/chat/completions",
            headers=self._headers(),
            body=payload,
            timeout=parameters.timeout,
        )
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = _normalize_content(msg.get("content"))
        if not text:
            raise ProviderExecutionError(f"{self.name} returned an empty response.", self.name, parameters.model)

        usage = _usage_oai(data.get("usage"))
        return ProviderChatResponse(
            model=data.get("model") or parameters.model,
            text=text,
            structured_data=None,
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def embed(self, client: httpx.AsyncClient, parameters: EmbeddingParameters) -> ProviderEmbeddingResponse:
        if not self.capabilities.embeddings:
            raise EmbeddingError(f'Provider "{self.name}" does not support embeddings.')

        payload: dict[str, Any] = {
            "model": parameters.model,
            "input": parameters.input,
            **parameters.provider_options,
        }
        data = await fetch_json(
            client,
            self.name,
            f"{self._base}/embeddings",
            headers=self._headers(),
            body=payload,
            timeout=parameters.timeout,
        )
        rows = data.get("data") or []
        vectors = [row["embedding"] for row in rows if isinstance(row, dict) and "embedding" in row]
        usage = _usage_oai(data.get("usage"))
        return ProviderEmbeddingResponse(
            model=data.get("model") or parameters.model, embeddings=vectors, usage=usage, raw=data
        )


def _msg(m: ChatMessage) -> dict[str, str]:
    return {"role": m.role, "content": m.content}


def _response_format(fmt: Any) -> dict[str, Any] | None:
    if fmt is None or isinstance(fmt, PlainTextResponseFormat):
        return None
    if isinstance(fmt, JsonObjectResponseFormat):
        return {"type": "json_object"}
    if isinstance(fmt, JsonSchemaResponseFormat):
        return {
            "type": "json_schema",
            "json_schema": {"name": fmt.name, "description": fmt.description, "schema": fmt.schema, "strict": True},
        }
    return None


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return ""


def _usage_oai(u: dict[str, Any] | None) -> Usage | None:
    if not u:
        return None
    return Usage(
        input_tokens=u.get("prompt_tokens"),
        output_tokens=u.get("completion_tokens"),
        total_tokens=u.get("total_tokens"),
    )
