from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import httpx

from xoin.types import ChatMessage, Usage


@dataclass(frozen=True, slots=True)
class Capabilities:
    structured_outputs: Literal["json-schema", "json-object", "prompt-only"] = "json-object"
    embeddings: bool = True


@dataclass(frozen=True, slots=True)
class PlainTextResponseFormat:
    """Chat completion returns free-form text (no ``response_format`` constraint)."""

    kind: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class JsonObjectResponseFormat:
    """Provider must return a single JSON object (OpenAI-style ``json_object``)."""

    kind: Literal["json_object"] = "json_object"


@dataclass(frozen=True, slots=True)
class JsonSchemaResponseFormat:
    """Structured JSON constrained by a schema (native on OpenAI / Anthropic tools)."""

    name: str
    schema: dict[str, Any]
    description: str | None = None
    kind: Literal["json_schema"] = "json_schema"


ChatResponseFormat = PlainTextResponseFormat | JsonObjectResponseFormat | JsonSchemaResponseFormat


@dataclass(slots=True)
class ChatCompletionParameters:
    """Inputs for one provider chat completion call (after routing and message assembly)."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None
    max_tokens: int | None
    response_format: ChatResponseFormat | None
    provider_options: dict[str, Any]
    timeout: float | None


@dataclass(slots=True)
class EmbeddingParameters:
    """Inputs for one provider embeddings call."""

    model: str
    input: list[str]
    provider_options: dict[str, Any]
    timeout: float | None


@dataclass(slots=True)
class ProviderChatResponse:
    """Parsed chat completion from a vendor HTTP response (before Pydantic validation)."""

    model: str
    text: str
    structured_data: Any | None
    usage: Usage | None
    finish_reason: str | None
    raw: Any


@dataclass(slots=True)
class ProviderEmbeddingResponse:
    """Parsed embedding batch from a vendor HTTP response."""

    model: str
    embeddings: list[list[float]]
    usage: Usage | None
    raw: Any


@runtime_checkable
class Provider(Protocol):
    name: str
    capabilities: Capabilities
    default_model: str | None
    default_embedding_model: str | None

    async def generate(
        self, client: httpx.AsyncClient, parameters: ChatCompletionParameters
    ) -> ProviderChatResponse: ...

    async def embed(self, client: httpx.AsyncClient, parameters: EmbeddingParameters) -> ProviderEmbeddingResponse: ...
