from __future__ import annotations

from xoin.providers.base import Capabilities
from xoin.providers.openai import OpenAIProvider


class GeminiProvider(OpenAIProvider):
    """Google Gemini via the OpenAI-compatible chat endpoint.

    This keeps parity with other built-in providers by reusing the same request/response
    normalization path as :class:`~xoin.providers.openai.OpenAIProvider`.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        default_model: str | None = "gemini-2.5-flash",
        headers: dict[str, str] | None = None,
        capabilities: Capabilities | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            name="gemini",
            base_url=base_url,
            default_model=default_model,
            default_embedding_model=None,
            capabilities=capabilities or Capabilities(structured_outputs="json-object", embeddings=False),
            headers=headers,
        )
