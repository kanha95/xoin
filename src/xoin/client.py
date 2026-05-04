from __future__ import annotations

"""``Xoin`` facade: template resolution, provider routing, retries, and structured-output wiring.

Execution path for chat:

1. Resolve prompt (inline ``prompt`` or named/file template).
2. ``_resolve_generation_route`` builds an ordered list of ``(provider, model_override)`` from
   ``provider`` / ``provider_order`` / ``provider_targets``, then ``default_provider`` /
   ``fallback_providers``.
3. ``run_fallback`` tries each entry until one succeeds.
4. ``_generate_provider`` builds chat messages, picks native vs prompted JSON schema from
   ``Capabilities``, calls ``Provider.generate``, then optionally validates with Pydantic.

Provider modules under ``xoin.providers`` stay thin (HTTP + vendor request/response shapes).
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import httpx

from xoin.errors import ProviderConfigurationError, ProviderExecutionError, StructuredOutputError
from xoin.fallback import run_fallback
from xoin.schema import response_json_schema
from xoin.structured import build_prompt_schema, validate_response
from xoin.templates import render_template, resolve_named_template
from xoin.types import (
    ChatMessage,
    EmbedResult,
    GenManyTarget,
    GenResult,
    PriorityProviderTarget,
    RetryCfg,
    StructuredSpec,
    TemplateDefinition,
)
from xoin.providers.base import (
    Capabilities,
    ChatCompletionParameters,
    EmbeddingParameters,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    PlainTextResponseFormat,
    Provider,
)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------


class Xoin:
    """Configure providers once; call ``generate``, ``generate_many``, or ``embed``."""

    def __init__(
        self,
        *,
        providers: dict[str, Provider],
        default_provider: str | None = None,
        fallback_providers: list[str] | None = None,
        retry: int | RetryCfg | None = None,
        templates: Mapping[str, TemplateDefinition] | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        if not providers:
            raise ProviderConfigurationError("No providers configured.")
        self._providers = providers
        self._default = default_provider
        self._fallback = list(fallback_providers or [])
        self._retry = retry
        self._templates = dict(templates or {})
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Xoin:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def generate(
        self,
        *,
        provider: str | None = None,
        provider_order: list[str] | None = None,
        provider_targets: Sequence[PriorityProviderTarget | Mapping[str, Any]] | None = None,
        model: str | None = None,
        prompt: str | None = None,
        template: str | None = None,
        template_id: str | None = None,
        template_file: str | Path | None = None,
        variables: Mapping[str, Any] | None = None,
        system: str | None = None,
        messages: Sequence[ChatMessage | Mapping[str, Any]] | None = None,
        structured: StructuredSpec | Mapping[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        signal: Any | None = None,
        retry: int | RetryCfg | None = None,
    ) -> GenResult:
        spec = _coerce_structured(structured)
        cfg = _retry_cfg(retry, self._retry)
        timeout = (timeout_ms / 1000.0) if timeout_ms is not None else None
        po = _merge_provider_kwargs(metadata, provider_options)
        prompt_text = self._compose_prompt(
            prompt=prompt,
            template=template,
            template_id=template_id,
            template_file=template_file,
            variables=variables,
        )
        route = self._resolve_generation_route(provider, provider_order, provider_targets, model)

        async def attempt() -> GenResult:
            _cancel_if_requested(signal)
            calls: list[Callable[[], Awaitable[GenResult]]] = []
            for pname, target_model in route:

                async def run(
                    provider_name: str = pname,
                    explicit_model: str | None = target_model,
                ) -> GenResult:
                    # Per-target model overrides the request-level ``model`` exactly like xoin-js.
                    requested_model = model if explicit_model is None else explicit_model
                    return await self._generate_provider(
                        provider_name,
                        requested_model,
                        prompt_text,
                        system,
                        messages,
                        spec,
                        temperature,
                        max_tokens,
                        timeout,
                        po,
                        signal,
                    )

                calls.append(run)
            return await run_fallback(calls)

        return await _with_retry(attempt, cfg)

    async def generate_many(
        self,
        *,
        targets: Sequence[GenManyTarget | Mapping[str, Any]],
        model: str | None = None,
        prompt: str | None = None,
        template: str | None = None,
        template_id: str | None = None,
        template_file: str | Path | None = None,
        variables: Mapping[str, Any] | None = None,
        system: str | None = None,
        messages: Sequence[ChatMessage | Mapping[str, Any]] | None = None,
        structured: StructuredSpec | Mapping[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        signal: Any | None = None,
    ) -> list[GenResult]:
        """Run the same request concurrently across ``targets`` (no cross-provider fallback)."""

        spec = _coerce_structured(structured)
        timeout = (timeout_ms / 1000.0) if timeout_ms is not None else None
        po = _merge_provider_kwargs(metadata, provider_options)
        prompt_text = self._compose_prompt(
            prompt=prompt,
            template=template,
            template_id=template_id,
            template_file=template_file,
            variables=variables,
        )

        cleaned = [_coerce_many_target(t) for t in targets]
        if not cleaned:
            raise ProviderConfigurationError("generate_many requires at least one target.")

        for t in cleaned:
            if t.provider not in self._providers:
                raise ProviderConfigurationError(f'Unknown provider "{t.provider}" in generate_many targets.')

        async def one(target: GenManyTarget) -> GenResult:
            _cancel_if_requested(signal)
            explicit = target.model
            requested_model = model if explicit is None else explicit
            return await self._generate_provider(
                target.provider,
                requested_model,
                prompt_text,
                system,
                messages,
                spec,
                temperature,
                max_tokens,
                timeout,
                po,
                signal,
            )

        return await asyncio.gather(*(one(t) for t in cleaned))

    async def embed(
        self,
        *,
        input: str | list[str],  # noqa: A003
        provider: str | None = None,
        model: str | None = None,
        timeout_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        signal: Any | None = None,
    ) -> EmbedResult:
        _cancel_if_requested(signal)
        name = provider or self._default or (self._fallback[0] if self._fallback else None)
        if not name:
            raise ProviderConfigurationError("No provider configured for embeddings.")
        p = self._providers.get(name)
        if p is None:
            raise ProviderConfigurationError(f'Provider "{name}" is not registered.')
        if not p.capabilities.embeddings:
            raise ProviderConfigurationError(f'Provider "{name}" does not support embeddings.')
        em = model or p.default_embedding_model
        if not em:
            raise ProviderConfigurationError(f'Provider "{name}" requires an embedding model.')
        texts = input if isinstance(input, list) else [input]
        timeout = (timeout_ms / 1000.0) if timeout_ms is not None else None
        embed_po = _merge_provider_kwargs(metadata, provider_options)
        raw = await p.embed(
            self._client,
            EmbeddingParameters(model=em, input=texts, provider_options=embed_po, timeout=timeout),
        )
        return EmbedResult(provider=p.name, model=raw.model, embeddings=raw.embeddings, usage=raw.usage, raw=raw.raw)

    def register_provider(self, name: str, provider: Provider) -> None:
        """Attach another provider at runtime (parity with xoin-js ``registerProvider``)."""

        self._providers[name] = provider

    # --- Prompt/templates and provider ordering ---

    def _compose_prompt(
        self,
        *,
        prompt: str | None,
        template: str | None,
        template_id: str | None,
        template_file: str | Path | None,
        variables: Mapping[str, Any] | None,
    ) -> str | None:
        definition = resolve_named_template(
            inline_template=template,
            template_id=template_id,
            template_file=template_file,
            templates=self._templates,
        )
        if definition is not None:
            return render_template(definition, variables)
        return prompt

    def _resolve_generation_route(
        self,
        provider: str | None,
        provider_order: list[str] | None,
        provider_targets: Sequence[PriorityProviderTarget | Mapping[str, Any]] | None,
        request_model: str | None,
    ) -> list[tuple[str, str | None]]:
        if provider_targets is not None and len(provider_targets) > 0:
            pts = [_coerce_priority_target(x) for x in provider_targets]
            pts.sort(key=lambda item: item.priority)
            seen: set[str] = set()
            route: list[tuple[str, str | None]] = []
            for t in pts:
                key = f"{t.provider}::{t.model or ''}"
                if key in seen:
                    continue
                seen.add(key)
                if t.provider not in self._providers:
                    raise ProviderConfigurationError(
                        f'provider_targets references unknown provider "{t.provider}".',
                    )
                route.append((t.provider, t.model))
            if not route:
                raise ProviderConfigurationError("provider_targets produced an empty route.")
            return route

        names = self._provider_order(provider, provider_order)
        return [(n, request_model) for n in names]

    def _provider_order(self, primary: str | None, extra: list[str] | None) -> list[str]:
        order = [
            *([primary] if primary else []),
            *(extra or []),
            *([self._default] if self._default else []),
            *self._fallback,
        ]
        seen: set[str] = set()
        out: list[str] = []
        for k in order:
            if k and k not in seen and k in self._providers:
                seen.add(k)
                out.append(k)
        if not out:
            raise ProviderConfigurationError("No configured providers match this request.")
        return out

    # --- Single provider call + structured validation ---

    async def _generate_provider(
        self,
        pname: str,
        requested_model: str | None,
        prompt: str | None,
        system: str | None,
        messages: Sequence[ChatMessage | Mapping[str, Any]] | None,
        structured: StructuredSpec | None,
        temperature: float | None,
        max_tokens: int | None,
        timeout: float | None,
        provider_options: dict[str, Any],
        signal: Any | None,
    ) -> GenResult:
        _cancel_if_requested(signal)
        prov = self._providers.get(pname)
        if prov is None:
            raise ProviderConfigurationError(f'Provider "{pname}" is not registered.')
        mdl = requested_model or prov.default_model
        if not mdl:
            raise ProviderExecutionError(f'Provider "{pname}" requires a model.', pname)

        native = _native_structured(structured, prov.capabilities)
        msgs = _build_messages(messages, system, structured, native, prompt)
        response_format = _provider_chat_response_format(structured, prov.capabilities, native)

        completion = ChatCompletionParameters(
            model=mdl,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            provider_options=provider_options,
            timeout=timeout,
        )
        try:
            raw = await prov.generate(self._client, completion)
        except ProviderExecutionError:
            raise
        except Exception as e:
            raise ProviderExecutionError(f"{pname} failed: {e}", pname, mdl) from e

        data = None
        if structured:
            try:
                data = validate_response(structured.response_model, raw.text, raw.structured_data)
            except StructuredOutputError as exc:
                raise ProviderExecutionError(
                    f"Structured output did not match schema: {exc}",
                    prov.name,
                    mdl,
                ) from exc

        return GenResult(
            provider=prov.name,
            model=raw.model,
            text=raw.text,
            data=data,
            usage=raw.usage,
            finish_reason=raw.finish_reason,
            raw=raw.raw,
        )


# ---------------------------------------------------------------------------
# Factory + request helpers (coercion, retry, message/response-format builders)
# ---------------------------------------------------------------------------


def create_xoin(**kwargs: Any) -> Xoin:
    return Xoin(**kwargs)


def _merge_provider_kwargs(
    metadata: Mapping[str, Any] | None,
    provider_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """``provider_options`` wins over ``metadata`` on key clashes (xoin-js parity)."""

    return {**dict(metadata or {}), **dict(provider_options or {})}


def _cancel_if_requested(signal: Any) -> None:
    """Cooperative cancellation before provider I/O (``asyncio.Event`` or ``.aborted`` flag)."""

    if signal is None:
        return
    if isinstance(signal, asyncio.Event) and signal.is_set():
        raise asyncio.CancelledError()
    if getattr(signal, "aborted", False):
        raise asyncio.CancelledError()


def _effective_json_schema(spec: StructuredSpec) -> dict[str, Any]:
    if spec.json_schema is not None:
        return spec.json_schema
    return response_json_schema(spec.response_model)


def _coerce_structured(s: StructuredSpec | Mapping[str, Any] | None) -> StructuredSpec | None:
    if s is None:
        return None
    if isinstance(s, StructuredSpec):
        return s
    return StructuredSpec.model_validate(s)


def _coerce_many_target(value: GenManyTarget | Mapping[str, Any]) -> GenManyTarget:
    if isinstance(value, GenManyTarget):
        return value
    return GenManyTarget.model_validate(value)


def _coerce_priority_target(value: PriorityProviderTarget | Mapping[str, Any]) -> PriorityProviderTarget:
    if isinstance(value, PriorityProviderTarget):
        return value
    return PriorityProviderTarget.model_validate(value)


def _retry_cfg(local: int | RetryCfg | None, default: int | RetryCfg | None) -> RetryCfg:
    src = local if local is not None else default
    if isinstance(src, int):
        return RetryCfg(retries=max(0, src))
    if src is None:
        return RetryCfg()
    return src


async def _with_retry(factory: Callable[[], Awaitable[T]], cfg: RetryCfg) -> T:
    attempt = 0
    while True:
        try:
            return await factory()
        except ProviderExecutionError:
            if attempt >= cfg.retries:
                raise
            attempt += 1
            wait = max(0.0, cfg.delay_ms / 1000.0 * (cfg.backoff_multiplier ** (attempt - 1)))
            if wait:
                await asyncio.sleep(wait)


def _native_structured(spec: StructuredSpec | None, caps: Capabilities) -> bool:
    if not spec:
        return False
    mode = spec.mode
    cap = caps.structured_outputs
    if mode == "prompted":
        return False
    if mode == "native":
        return cap != "prompt-only"
    if cap == "json-schema":
        return True
    return cap == "json-object"


def _provider_chat_response_format(
    spec: StructuredSpec | None,
    caps: Capabilities,
    native: bool,
) -> PlainTextResponseFormat | JsonObjectResponseFormat | JsonSchemaResponseFormat | None:
    if not spec:
        return None
    if not native:
        return PlainTextResponseFormat()
    if caps.structured_outputs == "json-schema":
        return JsonSchemaResponseFormat(
            name=spec.name,
            description=spec.description,
            schema=_effective_json_schema(spec),
        )
    if caps.structured_outputs == "json-object":
        return JsonObjectResponseFormat()
    return PlainTextResponseFormat()


def _build_messages(
    messages: Sequence[ChatMessage | Mapping[str, Any]] | None,
    system: str | None,
    structured: StructuredSpec | None,
    native: bool,
    prompt: str | None,
) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    if messages:
        for item in messages:
            out.append(item if isinstance(item, ChatMessage) else ChatMessage.model_validate(item))
    if system:
        out.insert(0, ChatMessage(role="system", content=system))
    if structured and not native:
        schema = _effective_json_schema(structured)
        out.insert(0, ChatMessage(role="system", content=build_prompt_schema(structured.name, schema)))
    if prompt:
        out.append(ChatMessage(role="user", content=prompt))
    if not out:
        raise ProviderConfigurationError("Provide prompt, messages, templates, or any combination with content.")
    return out
