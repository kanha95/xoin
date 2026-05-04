from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SerializeAsAny


class TemplateDefinition(BaseModel):
    """Named or file-backed prompt template (parity with xoin-js ``TemplateDefinition``)."""

    model_config = ConfigDict(frozen=True)

    template: str
    defaults: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class StructuredSpec(BaseModel):
    """Structured JSON matching ``response_model`` (a Pydantic ``BaseModel`` subclass)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, populate_by_name=True)

    response_model: type[BaseModel]
    mode: Literal["auto", "native", "prompted"] = "auto"
    name: str = "structured_response"
    description: str | None = None
    #: Optional provider-facing JSON Schema (xoin-js ``structured.jsonSchema``). Validation still uses ``response_model``.
    json_schema: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("json_schema", "jsonSchema"),
    )


class RetryCfg(BaseModel):
    model_config = ConfigDict(frozen=True)

    retries: int = Field(ge=0, default=0)
    delay_ms: int = Field(ge=0, default=0)
    backoff_multiplier: float = Field(ge=1.0, default=1.0)


class GenManyTarget(BaseModel):
    """Single provider/model pair for :meth:`~xoin.client.Xoin.generate_many`."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str | None = None


class PriorityProviderTarget(BaseModel):
    """Priority-ordered fallback step (parity with xoin-js ``providerTargets``)."""

    model_config = ConfigDict(frozen=True)

    priority: int
    provider: str
    model: str | None = None


class GenResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: str
    model: str
    text: str
    #: ``SerializeAsAny`` so ``model_dump`` includes subclass fields (plain ``BaseModel`` infers {}).
    data: Annotated[BaseModel | None, SerializeAsAny()] = None
    usage: Usage | None = None
    finish_reason: str | None = None
    raw: Any = None


class EmbedResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: str
    model: str
    embeddings: list[list[float]]
    usage: Usage | None = None
    raw: Any = None
