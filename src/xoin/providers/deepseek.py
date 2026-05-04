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


class DeepSeekProvider:
    name = "deepseek"
    capabilities = Capabilities(structured_outputs="json-object", embeddings=False)

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        default_model: str | None = "deepseek-chat",
        headers: dict[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self.default_model = default_model
        self.default_embedding_model = None
        self._headers_extra = headers or {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json", **self._headers_extra}

    async def generate(
        self, client: httpx.AsyncClient, parameters: ChatCompletionParameters
    ) -> ProviderChatResponse:
        payload: dict[str, Any] = {
            "model": parameters.model,
            "messages": [{"role": m.role, "content": m.content} for m in parameters.messages],
            **parameters.provider_options,
        }
        if parameters.temperature is not None:
            payload["temperature"] = parameters.temperature
        if parameters.max_tokens is not None:
            payload["max_tokens"] = parameters.max_tokens
        rf = _ds_response_format(parameters.response_format)
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
        text = _norm(msg.get("content"))
        if not text:
            raise ProviderExecutionError(f"{self.name} returned an empty response.", self.name, parameters.model)

        return ProviderChatResponse(
            model=data.get("model") or parameters.model,
            text=text,
            structured_data=None,
            usage=_usage(data.get("usage")),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def embed(self, _client: httpx.AsyncClient, _parameters: EmbeddingParameters) -> ProviderEmbeddingResponse:
        raise EmbeddingError('Provider "deepseek" does not support embeddings.')


def _ds_response_format(fmt: Any) -> dict[str, Any] | None:
    if fmt is None or isinstance(fmt, PlainTextResponseFormat):
        return None
    if isinstance(fmt, JsonSchemaResponseFormat):
        return {"type": "json_object"}
    if isinstance(fmt, JsonObjectResponseFormat):
        return {"type": "json_object"}
    return None


def _norm(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    return ""


def _usage(u: dict[str, Any] | None) -> Usage | None:
    if not u:
        return None
    return Usage(
        input_tokens=u.get("prompt_tokens"),
        output_tokens=u.get("completion_tokens"),
        total_tokens=u.get("total_tokens"),
    )
