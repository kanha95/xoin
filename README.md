# xoin-py

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/kanha95/xoin-js/main/assets/XOIN_LOGO_RACT.png" />
    <img src="https://raw.githubusercontent.com/kanha95/xoin-js/main/assets/XOIN_LOGO_LIGHT_BG.png" alt="xoin — Python LLM client: OpenAI, Anthropic Claude, Mistral, DeepSeek, Pydantic structured outputs, embeddings, provider fallback" width="240" />
  </picture>
</p>

<p align="center">
  <strong>Python LLM client for OpenAI, Claude, DeepSeek &amp; more</strong> — async chat completions, <strong>Pydantic</strong>-validated structured outputs, text embeddings, and provider fallback for production services.
</p>

<p align="center">
  <img alt="Open Source" src="https://img.shields.io/badge/open%20source-yes-22c55e">
  <img alt="Free to Use" src="https://img.shields.io/badge/free%20to%20use-yes-0ea5e9">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-3776ab">
  <img alt="Structured Output" src="https://img.shields.io/badge/structured%20output-pydantic%20validated-8b5cf6">
</p>

**xoin-py** is an open source **LLM API client** for **Python 3.10+** that connects to multiple AI providers — **OpenAI**, **Anthropic**, **Mistral**, **DeepSeek** — through one **consistent async API** built on **httpx**.

It helps you ship AI features with:

✅ **Chat completions** (OpenAI-style where applicable; Anthropic Messages API for Claude)  
✅ **Structured output** validated with **Pydantic** (`BaseModel`)  
✅ **Text embeddings** on providers that expose OpenAI-compatible `/embeddings` (OpenAI, Mistral)  
✅ **Automatic provider fallback** (`provider_order`, `default_provider`, `fallback_providers`)  
✅ **Retries** with backoff on transient **provider execution** failures  

Async-first, minimal dependencies (**httpx**, **pydantic**). Sister library to the JavaScript **[xoin-js](https://github.com/kanha95/xoin-js)** client (`@xoin/xoin-js`).

## Table of Contents

- [Why xoin-py](#why-xoin-py)
- [Installation](#installation)
- [Who It Is For](#who-it-is-for)
- [Works Well In](#works-well-in)
- [Quick Start](#quick-start)
- [Built-in Providers](#built-in-providers)
- [Parity with xoin-js](#parity-with-xoin-js)
- [Core Concepts](#core-concepts)
- [API Overview](#api-overview)
- [`Xoin` / `create_xoin` configuration](#xoin--create_xoin-configuration)
- [`generate` parameters](#generate-parameters)
- [`StructuredOutput` (structured output)](#structuredspec-structured-output)
- [Schema examples (Pydantic)](#schema-examples-pydantic)
- [Retry and fallback strategy](#retry-and-fallback-strategy)
- [`embed` parameters](#embed-parameters)
- [Provider constructors](#provider-constructors)
- [Parameter & types reference](#parameter--types-reference)
- [Examples by use case](#examples-by-use-case)
- [Framework snippets](#framework-snippets)
- [Custom providers](#custom-providers)
- [Error handling](#error-handling)
- [Development](#development)

## Why xoin-py

Production **Python** backends that call **LLM APIs** quickly outgrow one-off SDK calls and ad-hoc JSON parsing.

You usually want:

- one **multi-provider** surface for OpenAI, Anthropic, Mistral, DeepSeek  
- **structured outputs** validated with **Pydantic** before business logic runs  
- **provider fallback** when a vendor errors or rate-limits  
- **embeddings** on the same abstraction where supported  
- **async** integration with FastAPI, Starlette, workers, and scripts  

**xoin-py** targets that workflow in a **small** codebase: configure providers once, `await x.generate(...)` / `await x.embed(...)`.

## Installation

```bash
pip install xoin-py
```

Import the package as **`xoin`** (distribution name on PyPI is **`xoin-py`**):

```python
from xoin import Xoin, StructuredOutput
from xoin.providers import OpenAIProvider
```

Optional: load secrets from the environment (`os.environ`, `pydantic-settings`, etc.) — same idea as `dotenv` in Node.

## Who It Is For

**xoin-py** fits if you are building:

- FastAPI / Starlette routes that return structured model output  
- asyncio services and workers  
- extraction, classification, and summarization pipelines  
- internal tools that need validated JSON from LLMs  
- **RAG** or search flows that need embeddings (OpenAI / Mistral)

## Works Well In

Server-side Python where **API keys stay private**:

- FastAPI, Starlette, Django ASGI  
- asyncio scripts and CLIs  
- background workers (Celery with async bridge, arq, etc.)  

Do **not** embed provider API keys in browser-delivered code.

## Quick Start

Complete asyncio example:

```python
import asyncio
import os

from pydantic import BaseModel

from xoin import StructuredOutput, Xoin
from xoin.providers import OpenAIProvider


class UserProfile(BaseModel):
    name: str
    age: int


async def main() -> None:
    async with Xoin(
        providers={
            "openai": OpenAIProvider(
                api_key=os.environ["OPENAI_API_KEY"],
                default_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            ),
        },
        default_provider="openai",
    ) as xoin:
        result = await xoin.generate(
            provider="openai",
            prompt='Extract a JSON object from: "Ava is 31 years old."',
            structured=StructuredOutput(response_model=UserProfile, name="user_profile"),
        )

    print(result.data)


asyncio.run(main())
```

**Sample output (live APIs)** — captured by running the snippet above with keys from `examples/.env` (`set -a && source examples/.env && set +a`). Exact models, token counts, and response IDs vary per request; the `raw` field from `GenResult.model_dump()` is omitted here for readability.

OpenAI (`gpt-4o-mini`, structured native schema):

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini-2024-07-18",
  "text": "{\"name\":\"Ava\",\"age\":31}",
  "data": {
    "name": "Ava",
    "age": 31
  },
  "usage": {
    "input_tokens": 72,
    "output_tokens": 10,
    "total_tokens": 82
  },
  "finish_reason": "stop"
}
```

DeepSeek — same `UserProfile` pattern, from `python examples/deepseek_structured_output.py`:

```json
{
  "provider": "deepseek",
  "model": "deepseek-v4-flash",
  "text": "{\"name\": \"Kabir\", \"age\": 33}",
  "data": {
    "name": "Kabir",
    "age": 33
  },
  "usage": {
    "input_tokens": 40,
    "output_tokens": 13,
    "total_tokens": 53
  },
  "finish_reason": "stop"
}
```

Why this works:

- one prompt  
- native or prompt-based JSON depending on provider capability (`StructuredOutput.mode`)  
- **Pydantic** validates into `result.data`  
- failures surface as typed exceptions (`xoin.errors`)

## Built-in Providers

**xoin-py** ships concrete provider classes (each in its own module under `xoin.providers`):

| Provider class | Typical use |
|----------------|-------------|
| `OpenAIProvider` | OpenAI Chat Completions + embeddings |
| `AnthropicProvider` | Claude Messages API + native tool-use structured output |
| `MistralProvider` | Mistral chat (`json_object` structured mode) + embeddings |
| `DeepSeekProvider` | DeepSeek chat (`json_object`); **no embeddings** in defaults |

```python
import os

from xoin import Xoin
from xoin.providers import AnthropicProvider, DeepSeekProvider, MistralProvider, OpenAIProvider

xoin = Xoin(
    default_provider="openai",
    fallback_providers=["anthropic", "deepseek"],
    providers={
        "openai": OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            default_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            default_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        ),
        "anthropic": AnthropicProvider(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            default_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        ),
        "mistral": MistralProvider(
            api_key=os.environ["MISTRAL_API_KEY"],
            default_model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            default_embedding_model=os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed"),
        ),
        "deepseek": DeepSeekProvider(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            default_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        ),
    },
)
```

**OpenAI-compatible** backends (Groq, Azure OpenAI-style gateways, etc.) use the same `OpenAIProvider` class with `name="groq"`, a custom `base_url`, and `capabilities=Capabilities(structured_outputs="json-object", embeddings=False)` when embeddings are unavailable.

## Parity with xoin-js

[xoin-js](https://github.com/kanha95/xoin-js) is the reference JavaScript client (`npm install @xoin/xoin-js`). **xoin-py** follows the same ideas with Python idioms.

| Feature | xoin-js | xoin-py |
|--------|---------|---------|
| Structured schemas | Zod | **Pydantic** `BaseModel` |
| HTTP | `fetch` | **httpx.AsyncClient** (shared on `Xoin`) |
| Templates (`template`, `templateId`, `templateFile`, `variables`) | ✅ | ✅ (`variables` replaces JS `input`; YAML needs **PyYAML**) |
| `generateMany` | ✅ | ✅ (`await xoin.generate_many(...)`) |
| Priority `providerTargets` | ✅ | ✅ (`provider_targets=[PriorityProviderTarget(...)]`) |
| `registerProvider` | ✅ | ✅ (`xoin.register_provider(...)`) |
| Manual `jsonSchema` alongside schema | ✅ (`structured.jsonSchema`) | ✅ `StructuredOutput(json_schema=...)` accepts **`jsonSchema`** in dict input |
| `signal` / AbortSignal | ✅ | ✅ cooperative **`signal=`**: `asyncio.Event` (`is_set()`) or any object with **`.aborted` truthy** → raises `asyncio.CancelledError` before HTTP |
| `metadata` passthrough | ✅ | ✅ merged shallow into outbound bodies (**`provider_options` wins** on key clashes) |

**Fallback vs validation:** After a successful HTTP response, if **Pydantic** validation fails, xoin-py now wraps that failure as **`ProviderExecutionError`**, so the same **`provider_order`** / **`provider_targets`** fallback chain used for HTTP errors can try the **next** provider (matching xoin-js behavior). You can still catch the underlying cause via `exc.__cause__` when needed. Direct callers may also catch **`StructuredOutputError`** from **`validate_response`** in lower-level code paths.

## Core Concepts

### 1. One client, many providers

Register all vendor adapters on `Xoin(providers={...})`.

### 2. Structured output first

Define a **`BaseModel`** and pass `StructuredOutput(response_model=...)`. Parsed output is `GenResult.data`.

### 3. Fallback without glue

Pass `provider_order=[...]` on `generate`, or configure `default_provider` / `fallback_providers` on the client.

### 4. Embeddings where the API matches OpenAI

Use `await xoin.embed(input=[...])` with OpenAI or Mistral defaults.

### 5. Async context manager

Prefer `async with Xoin(...) as xoin:` — closes the internal **httpx** client when `Xoin` created it.

## API Overview

Main exports (`from xoin import ...`):

- `Xoin`, `create_xoin`
- `StructuredOutput`, `ChatMessage`, `GenResult`, `EmbedResult`, `Usage`, `RetryCfg`
- `GenManyTarget`, `PriorityProviderTarget`, `TemplateDefinition`
- `render_template`, `resolve_named_template`, `load_template_file`
- `errors` (module with exception classes)

Main methods:

- `await xoin.generate(...)`
- `await xoin.generate_many(...)`
- `await xoin.embed(...)`
- `xoin.register_provider(name, provider)`

For a **plain-language explanation of every argument and result field**, see **[Parameter & types reference](#parameter--types-reference)** below.

Provider classes (`from xoin.providers import ...`):

- `OpenAIProvider`, `AnthropicProvider`, `MistralProvider`, `DeepSeekProvider`

Protocol / internals (`xoin.providers.base`): `Provider`, `ChatCompletionParameters`, `EmbeddingParameters`, `Capabilities` — useful for **custom** adapters.

## `Xoin` / `create_xoin` configuration

`Xoin(...)` (and `create_xoin(**kwargs)` — identical) accepts:

| Parameter | Type | What it does |
|-----------|------|----------------|
| `providers` | `dict[str, Provider]` | Registered providers keyed by name (e.g. `"openai"`). **Required.** |
| `default_provider` | `str \| None` | Used when a request omits `provider` and as part of fallback ordering. |
| `fallback_providers` | `list[str] \| None` | Append-only fallback chain after primary / `provider_order`. |
| `templates` | `dict[str, TemplateDefinition] \| None` | Named templates referenced via `template_id`. |
| `retry` | `int \| RetryCfg \| None` | Default retry policy for `generate` (`ProviderExecutionError` only). |
| `client` | `httpx.AsyncClient \| None` | Inject a shared client (tests, custom timeouts). If omitted, `Xoin` owns one. |
| `timeout_s` | `float` | Default timeout when `Xoin` creates its own `AsyncClient`. |

Example:

```python
from xoin import RetryCfg, Xoin
from xoin.providers import AnthropicProvider, OpenAIProvider

xoin = Xoin(
    default_provider="openai",
    fallback_providers=["anthropic"],
    retry=RetryCfg(retries=2, delay_ms=300, backoff_multiplier=2.0),
    providers={
        "openai": OpenAIProvider(api_key="..."),
        "anthropic": AnthropicProvider(api_key="..."),
    },
)
```

## `generate` parameters

`await xoin.generate(**kwargs)` — async chat / structured generation.

| Parameter | Type | What it does |
|-----------|------|----------------|
| `provider` | `str \| None` | Primary provider key from `providers`. |
| `provider_order` | `list[str] \| None` | Extra ordering after `provider`, before `default_provider` / `fallback_providers`. |
| `provider_targets` | `list[PriorityProviderTarget \| dict] \| None` | Explicit priority plan (lower `priority` runs first). When set, replaces `provider` / `provider_order` routing. |
| `model` | `str \| None` | Overrides the provider’s `default_model`. |
| `prompt` | `str \| None` | Final user message appended after history / system / structured instructions. |
| `template` | `str \| None` | Inline template text containing `{{variables}}`. |
| `template_id` | `str \| None` | Lookup into `Xoin(..., templates={...})`. |
| `template_file` | `str \| Path \| None` | Load YAML/JSON/plain template definitions from disk. |
| `variables` | `Mapping[str, Any] \| None` | Values merged with template defaults (same role as JS `input`). |
| `system` | `str \| None` | System instruction (inserted before conversation messages). |
| `messages` | `Sequence[ChatMessage \| dict] \| None` | Chat history (`role`, `content`). |
| `structured` | `StructuredOutput \| dict \| None` | Enables parsing + **Pydantic** validation into `GenResult.data`. |
| `temperature` | `float \| None` | Sampling temperature. |
| `max_tokens` | `int \| None` | Max output tokens (Anthropic defaults internally if unset). |
| `timeout_ms` | `int \| None` | Per-request timeout override (converted to seconds for httpx). |
| `metadata` | `Mapping[str, Any] \| None` | Extra fields merged into the provider JSON body before `provider_options`. |
| `provider_options` | `Mapping[str, Any] \| None` | Vendor-specific fields merged **after** `metadata` (same keys override). |
| `signal` | `Any \| None` | Cooperative cancel check: `asyncio.Event` when **set**, or **truthy** `.aborted`. |
| `retry` | `int \| RetryCfg \| None` | Overrides client-level `retry` for this call. |

Plain text:

```python
result = await xoin.generate(
    provider="openai",
    prompt="Write a short welcome message for a new SaaS customer.",
    temperature=0.7,
    max_tokens=120,
)
print(result.text)
```

Chat-style:

```python
from xoin import ChatMessage

result = await xoin.generate(
    provider="openai",
    system="You are a concise support assistant.",
    messages=[
        ChatMessage(role="user", content="My payment failed yesterday."),
        ChatMessage(role="assistant", content="I can help with that."),
        ChatMessage(role="user", content="What should I check first?"),
    ],
)
```

Fallback chain:

```python
result = await xoin.generate(
    provider="openai",
    provider_order=["anthropic", "mistral"],
    prompt="Extract the order summary from the customer message.",
    structured=StructuredOutput(response_model=OrderSummary),
)
```

## `generate_many`

`await xoin.generate_many(**kwargs)` fans the **same** logical request out to multiple `(provider, model?)` targets **in parallel** via `asyncio.gather`. There is **no** shared fallback chain between targets—pair it with `generate` when you need resilience.

Shared parameters match `generate`, except `provider`, `provider_order`, `provider_targets`, and `retry` are replaced by:

| Parameter | Type | What it does |
|-----------|------|----------------|
| `targets` | `Sequence[GenManyTarget \| dict]` | Each entry names a provider key (and optional per-target `model`). |

All other shared knobs (`prompt`, templates, `structured`, `metadata`, `signal`, `temperature`, …) behave the same as `generate`. There is **no** `retry` wrapper around `generate_many` per target (matches xoin-js).

```python
from xoin.types import GenManyTarget

results = await xoin.generate_many(
    targets=[
        GenManyTarget(provider="openai"),
        GenManyTarget(provider="anthropic"),
    ],
    prompt="Summarize why structured outputs matter in two bullets.",
)

for item in results:
    print(item.provider, item.text[:120])
```

Runnable copies of these flows live under `examples/` (see `examples/README.md`).

## `StructuredOutput` (structured output)

Use `StructuredOutput` when you want **validated** JSON mapped to a **Pydantic** model (`GenResult.data`).

| Field | Type | What it does |
|-------|------|----------------|
| `response_model` | `type[BaseModel]` | **Required** — model used for **validation** (and default JSON Schema when native). |
| `json_schema` | `dict[str, Any] \| None` | Optional provider-facing schema override (JS `jsonSchema`). When set, native/prompt paths send this dict instead of `model_json_schema()`. |
| `mode` | `'auto' \| 'native' \| 'prompted'` | Same semantics as xoin-js (native vs prompt-only instructions). |
| `name` | `str` | Logical name / Anthropic tool name (default `"structured_response"`). |
| `description` | `str \| None` | Extra hint for providers that support descriptions. |

**Modes**

- `auto` — use native structured features when the provider supports JSON Schema / JSON object modes; otherwise prepend strict JSON instructions.  
- `native` — require native capability (`prompt-only` capabilities fall back to prompts).  
- `prompted` — always use prompt instructions + local parsing.

Dict shorthand works (`StructuredOutput.model_validate`):

```python
await xoin.generate(
    prompt="…",
    structured={
        "response_model": UserProfile,
        "name": "user_profile",
        "mode": "auto",
        # Optional camelCase parity:
        # "jsonSchema": {"type": "object", "properties": {...}, "required": [...]},
    },
)
```

Extraction example:

```python
from pydantic import BaseModel

from xoin import StructuredOutput


class ShippingAddress(BaseModel):
    line1: str
    city: str
    postal_code: str
    country: str


result = await xoin.generate(
    provider="anthropic",
    prompt='Extract the shipping address from: "Ship this to 10 Park Street, Pune 411001, India."',
    structured=StructuredOutput(
        response_model=ShippingAddress,
        name="shipping_address",
        description="Normalized shipping address extracted from user input",
        mode="auto",
    ),
)

print(result.data)
```

## Schema examples (Pydantic)

Below mirror common **[xoin-js](https://github.com/kanha95/xoin-js)** Zod patterns using **Pydantic v2**.

### 1. Basic object

```python
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


result = await xoin.generate(
    provider="openai",
    prompt='Extract a JSON object from: "Ava is 31 years old."',
    structured=StructuredOutput(response_model=User, name="user_profile"),
)
```

### 2. List of objects

Use a **`RootModel`** (or a small wrapper model) when the model must return a **top-level JSON array**.

```python
from pydantic import BaseModel, RootModel


class OrderLine(BaseModel):
    product: str
    quantity: int
    price: float


class OrderLines(RootModel[list[OrderLine]]):
    pass


result = await xoin.generate(
    provider="openai",
    prompt=(
        "Extract all purchased items:\n"
        '"2 wireless mice at 25 each, 1 keyboard at 70, and 3 mouse pads at 10 each."'
    ),
    structured=StructuredOutput(response_model=OrderLines, name="order_items"),
)
# Parsed payload is ``result.data.root``
```

### 3. Nested models

```python
from pydantic import BaseModel


class Customer(BaseModel):
    name: str
    email: str


class Address(BaseModel):
    line1: str
    city: str
    postal_code: str
    country: str


class Item(BaseModel):
    sku: str
    title: str
    quantity: int


class CustomerOrder(BaseModel):
    customer: Customer
    shipping_address: Address
    items: list[Item]


result = await xoin.generate(
    provider="anthropic",
    prompt=f"Extract order details from:\n{email_text}",
    structured=StructuredOutput(response_model=CustomerOrder, name="customer_order"),
)
```

### 4. Literal enums (strict categories)

```python
from typing import Literal

from pydantic import BaseModel


class Ticket(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str


result = await xoin.generate(
    provider="anthropic",
    prompt="My card was charged twice and I still cannot access premium features.",
    structured=StructuredOutput(response_model=Ticket, name="ticket_classification"),
)
```

### 5. Optional fields

```python
from pydantic import BaseModel


class Lead(BaseModel):
    name: str
    company: str
    email: str | None = None
    phone: str | None = None
    budget: str | None = None


result = await xoin.generate(
    provider="openai",
    prompt=f"Extract lead details from:\n{lead_message}",
    structured=StructuredOutput(response_model=Lead, name="lead_profile"),
)
```

### 6. Union / discriminated unions

**Plain union:**

```python
from typing import Literal, Union

from pydantic import BaseModel


class Refund(BaseModel):
    action: Literal["refund"]
    order_id: str
    reason: str


class Replace(BaseModel):
    action: Literal["replace"]
    order_id: str
    item: str


EmailAction = Union[Refund, Replace]


result = await xoin.generate(
    provider="openai",
    prompt=f"Determine the action:\n{support_message}",
    structured=StructuredOutput(response_model=EmailAction, name="email_action"),
)
```

**Discriminated union:**

```python
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class EmailNotif(BaseModel):
    channel: Literal["email"]
    subject: str
    body: str


class SmsNotif(BaseModel):
    channel: Literal["sms"]
    message: str


class NotificationEnvelope(BaseModel):
    notification: Annotated[
        Union[EmailNotif, SmsNotif],
        Field(discriminator="channel"),
    ]


result = await xoin.generate(
    provider="openai",
    prompt=f"Build notification payload from:\n{event_text}",
    structured=StructuredOutput(response_model=NotificationEnvelope, name="notification_payload"),
)
```

### 7. Choosing schema styles (rules of thumb)

- **`BaseModel` fields** for most business responses  
- **`list[T]`** when the model must return an array at the top level  
- **`Literal[...]`** when downstream code branches on fixed values  
- **`| None` optional fields** when keys may be absent  
- **Unions / discriminators** when multiple shapes are valid  

JSON Schema sent to providers is derived from **`model_json_schema()`** unless you implement a **custom provider** that overrides behavior.

## Retry and fallback strategy

### Retry the same provider

Retries apply when **`ProviderExecutionError`** is raised **inside** the generate attempt (HTTP errors, empty completions from the HTTP layer, etc.).

```python
from xoin import RetryCfg

result = await xoin.generate(
    provider="openai",
    retry=2,
    prompt="Extract the user profile from this message.",
    structured=StructuredOutput(response_model=UserProfile),
)
```

Object form:

```python
result = await xoin.generate(
    provider="openai",
    retry=RetryCfg(retries=2, delay_ms=500, backoff_multiplier=2.0),
    prompt="Extract the user profile from this message.",
    structured=StructuredOutput(response_model=UserProfile),
)
```

| `RetryCfg` field | Meaning |
|------------------|---------|
| `retries` | Extra attempts **before** giving up on that attempt bundle |
| `delay_ms` | Base delay between retries |
| `backoff_multiplier` | Multiplies delay each retry |

### Fallback across providers

Ordering is built from: **`provider`** (if set), then **`provider_order`**, then **`default_provider`**, then **`fallback_providers`** — deduplicated.

```python
result = await xoin.generate(
    provider="openai",
    provider_order=["anthropic", "mistral"],
    prompt="Summarize this incident for executives.",
)
```

If **every** provider in the chain raises **`ProviderExecutionError`**, the last failure may surface as **`AggregateProviderError`** when multiple providers were tried.

## `embed` parameters

`await xoin.embed(**kwargs)` — vector embeddings (OpenAI / Mistral defaults).

| Parameter | Type | What it does |
|-----------|------|----------------|
| `input` | `str \| list[str]` | Text(s) to embed (**keyword-only**; shadows builtin name intentionally). |
| `provider` | `str \| None` | Provider key; defaults to `default_provider` or first `fallback_providers` entry. |
| `model` | `str \| None` | Overrides provider `default_embedding_model`. |
| `timeout_ms` | `int \| None` | Per-request timeout override. |
| `metadata` | `Mapping \| None` | Merged into the embeddings JSON body before `provider_options`. |
| `provider_options` | `Mapping \| None` | Extra JSON fields for the embeddings request (overrides `metadata` keys). |
| `signal` | `Any \| None` | Same cooperative cancellation semantics as `generate`. |

Example:

```python
result = await xoin.embed(
    provider="openai",
    model="text-embedding-3-small",
    input=[
        "How do I reset my password?",
        "How do I update my billing card?",
    ],
)

print(len(result.embeddings))
print(len(result.embeddings[0]))
```

**DeepSeek** and **Anthropic** defaults **do not** expose embeddings in xoin-py — configure OpenAI or Mistral for vectors.

## Parameter & types reference

Short tables above are for scanning. This section explains **what each argument does**, in plain language.

### `create_xoin(...)`

Same keyword arguments as [`Xoin(...)`](#xoin--create_xoin-configuration). Returns a new client instance.

---

### `Xoin(...)` constructor

| Argument | What it is for |
|----------|----------------|
| **`providers`** | **Required.** Dict of `{name: Provider}`. Each **name** is the string you pass as `provider=` or list entries in `provider_order` / targets. |
| **`default_provider`** | Used when a call omits `provider` and when building the fallback list. Must be one of the keys in `providers`. |
| **`fallback_providers`** | Ordered list of provider **names** tried **after** explicit `provider` / `provider_order` entries (deduplicated). |
| **`retry`** | Default retry policy for **`generate` only** (`int` = retry count with zero delay, or a [`RetryCfg`](#retrycfg) object). Does **not** apply to `generate_many` or `embed`. |
| **`templates`** | Dict of `{id: TemplateDefinition}` for use with `generate(..., template_id="...")`. |
| **`client`** | Optional shared **`httpx.AsyncClient`**. If you omit it, `Xoin` creates one and owns it (closed by `aclose()` or `async with`). |
| **`timeout_s`** | Default socket/read timeout for the internally created client only (seconds). Per-call `timeout_ms` still overrides per request. |

Lifecycle: call **`await xoin.aclose()`** when you are done, or use **`async with Xoin(...) as xoin:`** so the owned client is closed automatically.

---

### `Xoin.generate(...)`

All arguments are **keyword-only**.

#### Routing & model

| Argument | What it is for |
|----------|----------------|
| **`provider`** | First provider **name** to try. |
| **`provider_order`** | Extra names appended after `provider`, before `default_provider` / `fallback_providers`. Duplicates and unknown names are skipped. |
| **`provider_targets`** | If non-empty, **replaces** the `provider` + `provider_order` logic. Each [`PriorityProviderTarget`](#priorityprovidertarget) has a **`priority`** (lower numbers run first), **`provider`**, and optional **`model`**. Duplicate `(provider, model)` pairs are deduplicated. |
| **`model`** | Chat model id for this request. If omitted, each provider’s **`default_model`** is used. When using `provider_targets`, a target’s **`model`** overrides this for that step only. |

#### Prompt content (pick one style or combine carefully)

| Argument | What it is for |
|----------|----------------|
| **`prompt`** | Plain user text appended as the last **user** message (after history and template-driven content). |
| **`template`** | Inline template string with `{{variable}}` placeholders. If set, it wins over `template_id` / `template_file` and the rendered string becomes the prompt body (via the template pipeline). |
| **`template_id`** | Looks up a **`TemplateDefinition`** in `Xoin(..., templates={...})`. |
| **`template_file`** | Path to `.yaml` / `.yml` / `.json` / plain text template file (JSON/YAML must include a `"template"` string field). Requires **PyYAML** for YAML. |
| **`variables`** | Dict merged **on top of** template **`defaults`** when rendering `{{...}}` placeholders. Missing keys raise **`TemplateError`**. Non-string values are JSON-encoded in the output. |
| **`system`** | System instruction inserted **before** conversational messages when building the provider payload. |
| **`messages`** | Prior turns: each [`ChatMessage`](#chatmessage) (or dict with `role` / `content`). Parsed and combined with `system`, structured instructions, and `prompt`. |

**Template precedence:** if **`template`** is set, it is used and **`template_id` / `template_file`** are ignored. Otherwise **`template_id`** is resolved from the client registry; otherwise **`template_file`** is loaded. Only when **none** of those three are set does xoin use a bare **`prompt`** string (plus optional **`messages`**).

You must end up with **at least one** message after composition—otherwise **`ProviderConfigurationError`**.

#### Structured output & sampling

| Argument | What it is for |
|----------|----------------|
| **`structured`** | Optional [`StructuredOutput`](#structuredspec-fields). When set, the client asks the model for JSON, parses it, and validates into **`GenResult.data`**. See also [modes](#structuredspec-structured-output). |
| **`temperature`** | Sampling temperature forwarded to the provider when not `None`. |
| **`max_tokens`** | Cap on completion tokens. Anthropic defaults this internally when unset. |

#### Timeouts, metadata, cancellation, retries

| Argument | What it is for |
|----------|----------------|
| **`timeout_ms`** | Overrides the httpx timeout for **this** HTTP call (milliseconds). Converted to seconds internally. |
| **`metadata`** | Shallow dict merged into the **JSON request body** first. Use for cross-cutting fields your vendor accepts. |
| **`provider_options`** | Second dict merged into the body; **wins on duplicate keys** over `metadata`. Use for vendor-specific flags (`response_format` extras, `top_p`, etc.—whatever the API allows next to `messages`). |
| **`signal`** | Cooperative cancel hook **before** network I/O: pass an **`asyncio.Event`** and call **`event.set()`** from another task, **or** any object with a **truthy** **`aborted`** attribute. Raises **`asyncio.CancelledError`**. |
| **`retry`** | Overrides the client’s default **`retry`** for this **`generate`** call only. Retries run the **whole** fallback chain again on **`ProviderExecutionError`** (including structured validation failures wrapped as that error). |

#### Result: [`GenResult`](#genresult)

---

### `Xoin.generate_many(...)`

Same keywords as **`generate`**, except **`provider`**, **`provider_order`**, **`provider_targets`**, and **`retry`** are **not** supported.

| Argument | What it is for |
|----------|----------------|
| **`targets`** | **Required.** Non-empty sequence of [`GenManyTarget`](#genmanytarget) (or dicts). Each item names a **`provider`** and optional **`model`**. Runs **in parallel** (`asyncio.gather`). |

There is **no** automatic fallback between targets: each target performs exactly **one** provider call. Combine with **`generate`** when you need retries or fallback.

Results appear in the **same order** as **`targets`** (after coercion).

---

### `Xoin.embed(...)`

| Argument | What it is for |
|----------|----------------|
| **`input`** | **Keyword-only** (`input=` avoids shadowing the builtin in signatures). One string or a **list of strings** to embed. |
| **`provider`** | Embedding provider **name**. Defaults to **`default_provider`**, else the **first** entry in **`fallback_providers`**. |
| **`model`** | Embedding model id; defaults to the provider’s **`default_embedding_model`**. |
| **`timeout_ms`** | Per-request timeout override (milliseconds). |
| **`metadata`** | Merged into the embeddings JSON body before **`provider_options`**. |
| **`provider_options`** | Vendor-specific body fields; overrides **`metadata`** on key clashes. |
| **`signal`** | Same cancellation semantics as **`generate`**. |

Returns **[`EmbedResult`](#embedresult)**. Providers without **`capabilities.embeddings`** cannot be used.

---

### `Xoin.register_provider(name, provider)`

| Argument | What it is for |
|----------|----------------|
| **`name`** | String key used in `provider=` / ordering / targets. |
| **`provider`** | Instance implementing the **`Provider`** protocol. |

Overwrites an existing entry if **`name`** collides.

---

### `StructuredOutput` fields

Used for validated JSON outputs (`GenResult.data`).

| Field | Type | What it is for |
|-------|------|----------------|
| **`response_model`** | `type[BaseModel]` | **Required.** Pydantic model used to **validate** the model output. |
| **`mode`** | `"auto"` \| `"native"` \| `"prompted"` | Chooses provider-native JSON/schema modes vs instructions-only (see [modes](#structuredspec-structured-output)). |
| **`name`** | `str` | Logical schema/tool name (default `"structured_response"`). |
| **`description`** | optional `str` | Hint for providers that accept a schema description. |
| **`json_schema`** | optional `dict` | Provider-facing JSON Schema. Accept **`json_schema`** or **`jsonSchema`** in dict input. If omitted, schema is derived from **`response_model.model_json_schema()`**. Validation always uses **`response_model`**, not this dict. |

---

### `RetryCfg`

| Field | Meaning |
|-------|---------|
| **`retries`** | Maximum **extra** attempts after the first failure (`>= 0`). |
| **`delay_ms`** | Base pause **before** each retry (`>= 0`), in milliseconds. |
| **`backoff_multiplier`** | Factor `>= 1.0`. Seconds slept before retry attempt *n* (1-based) equal `(delay_ms / 1000) * (backoff_multiplier ** (n - 1))`. |

Only **`ProviderExecutionError`** triggers retries (inside **`generate`**).

---

### `GenManyTarget`

| Field | Meaning |
|-------|---------|
| **`provider`** | Registered provider **name** (required). |
| **`model`** | Optional per-target chat model; overrides the request-level **`model`** for that parallel call. |

---

### `PriorityProviderTarget`

| Field | Meaning |
|-------|---------|
| **`priority`** | Integer sort key—**smaller** values are tried **earlier**. |
| **`provider`** | Registered provider **name**. |
| **`model`** | Optional model override for that step in the fallback chain. |

---

### `TemplateDefinition`

| Field | Meaning |
|-------|---------|
| **`template`** | String containing `{{placeholder}}` markers. |
| **`defaults`** | Default values for placeholders (merged under runtime **`variables`**). |
| **`description`** | Optional human-readable note (not sent to the LLM by xoin-py). |

---

### `ChatMessage`

| Field | Meaning |
|-------|---------|
| **`role`** | `"system"` \| `"user"` \| `"assistant"` \| `"tool"` |
| **`content`** | Message text for that turn. |

---

### `GenResult`

| Field | Meaning |
|-------|---------|
| **`provider`** | Provider **name** that produced the response. |
| **`model`** | Model id returned or requested. |
| **`text`** | Raw assistant text from the provider. |
| **`data`** | Parsed **`BaseModel`** when **`structured`** was set; otherwise **`None`**. |
| **`usage`** | Optional [`Usage`](#usage) token counts. |
| **`finish_reason`** | Provider-specific completion reason string when available. |
| **`raw`** | Decoded JSON dict (or similar) from the vendor for debugging. |

---

### `EmbedResult`

| Field | Meaning |
|-------|---------|
| **`provider`** | Provider **name**. |
| **`model`** | Embedding model id. |
| **`embeddings`** | List of float vectors (one per input string). |
| **`usage`** | Optional [`Usage`](#usage). |
| **`raw`** | Raw provider payload for debugging. |

---

### `Usage`

| Field | Meaning |
|-------|---------|
| **`input_tokens`** | Prompt tokens when the vendor reports them. |
| **`output_tokens`** | Completion tokens (chat). |
| **`total_tokens`** | Sum when reported. |

Any field may be **`None`** if the API did not return it.

---

### `Capabilities` (dataclass)

Used on **`OpenAIProvider(capabilities=...)`** or custom providers.

| Field | Values | Meaning |
|-------|--------|---------|
| **`structured_outputs`** | `"json-schema"` \| `"json-object"` \| `"prompt-only"` | What the adapter can express natively: JSON Schema response format, plain JSON mode, or prompts only. |
| **`embeddings`** | `bool` | Whether **`embed`** is allowed on this adapter. |

---

### Template helpers (`xoin.templates`)

| Function | What it does |
|----------|----------------|
| **`render_template(definition, variables=None)`** | Substitutes `{{keys}}` using **`defaults`** ∪ **`variables`**. |
| **`load_template_file(path)`** | Loads a **`TemplateDefinition`** from disk (YAML needs PyYAML). |
| **`resolve_named_template(...)`** | Low-level: chooses inline vs id vs file (used internally by **`Xoin`**). Rarely needed in application code. |

---

### Custom providers: `ChatCompletionParameters` / `EmbeddingParameters`

When implementing **[`Provider`](#custom-providers)**:

**`ChatCompletionParameters`**: `model`, `messages` (**list[`ChatMessage`]**), `temperature`, `max_tokens`, `response_format` (`PlainTextResponseFormat` / `JsonObjectResponseFormat` / `JsonSchemaResponseFormat`), `provider_options` (already merged metadata + options), `timeout` (`float | None`, seconds).

**`EmbeddingParameters`**: `model`, `input` (**list[str]**), `provider_options`, `timeout`.

## Provider constructors

### `OpenAIProvider`

| Parameter | Description |
|-----------|-------------|
| `api_key` | Bearer token (required). |
| `name` | Provider key used in logs/errors and when registering (`"openai"` by default). Use a different value when you register the same class twice (e.g. `"groq"`). |
| `base_url` | API root; default `https://api.openai.com/v1`. Point at any OpenAI-compatible server. |
| `default_model` | Chat model id when `generate(..., model=None)`. |
| `default_embedding_model` | Embedding model when `embed(..., model=None)`. |
| `capabilities` | Override structured-output and embedding support (see [Capabilities](#capabilities-dataclass)). Defaults to JSON Schema structured outputs + embeddings enabled. |
| `headers` | Extra HTTP headers merged into every request. |

### `AnthropicProvider`

| Parameter | Description |
|-----------|-------------|
| `api_key` | Anthropic API key (sent as `x-api-key`; required). |
| `base_url` | Messages API root; default `https://api.anthropic.com/v1`. |
| `default_model` | Claude model id when `generate(..., model=None)`. |
| `headers` | Extra HTTP headers. |

**Fixed on the class:** `name = "anthropic"`, structured outputs via JSON Schema path, **`embeddings=False`** (use another provider for vectors).

### `MistralProvider`

| Parameter | Description |
|-----------|-------------|
| `api_key` | Mistral API key (required). |
| `base_url` | Default `https://api.mistral.ai/v1`. |
| `default_model` | Chat model when `model` omitted. |
| `default_embedding_model` | Embedding model when `embed(..., model=None)`. |
| `headers` | Extra HTTP headers. |

**Fixed on the class:** `name = "mistral"`, structured outputs use **`json-object`** mode (not full JSON Schema passthrough).

### `DeepSeekProvider`

| Parameter | Description |
|-----------|-------------|
| `api_key` | DeepSeek API key (required). |
| `base_url` | Default `https://api.deepseek.com`. |
| `default_model` | Chat model when `model` omitted. |
| `headers` | Extra HTTP headers. |

**Fixed on the class:** `name = "deepseek"`, **`json-object`** structured mode, **`embeddings=False`**.

## Examples by use case

### Extract CRM-style fields

```python
class Lead(BaseModel):
    company: str
    contact_name: str
    email: str
    budget: str


result = await xoin.generate(
    provider="openai",
    prompt=(
        'Extract company, contact, email, and budget from: '
        '"Hi, this is Sarah from Northwind. Reach me at sarah@northwind.com. '
        'Our budget is around $15k."'
    ),
    structured=StructuredOutput(response_model=Lead, name="lead_info"),
)
```

### Classify support tickets

```python
from typing import Literal

from pydantic import BaseModel


class Ticket(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str


result = await xoin.generate(
    provider="anthropic",
    system="You classify support tickets.",
    prompt="My card was charged twice and I still cannot access premium features.",
    structured=StructuredOutput(response_model=Ticket, name="ticket_classification"),
)
```

### Summarize transcript

```python
result = await xoin.generate(
    provider="openai",
    prompt=f"Summarize this meeting transcript in 5 bullet points:\n\n{transcript}",
    temperature=0.2,
    max_tokens=250,
)
```

### Embedding documents for search / RAG

```python
vectors = await xoin.embed(input=[doc.content for doc in documents])
```

## Framework snippets

### asyncio CLI script

```python
import asyncio
import os
from typing import Literal

from pydantic import BaseModel

from xoin import StructuredOutput, Xoin
from xoin.providers import OpenAIProvider


class Sentiment(BaseModel):
    label: Literal["positive", "neutral", "negative"]


async def main() -> None:
    async with Xoin(
        providers={"openai": OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])},
        default_provider="openai",
    ) as xoin:
        result = await xoin.generate(
            prompt='Classify sentiment of: "The onboarding was surprisingly smooth."',
            structured=StructuredOutput(response_model=Sentiment),
        )
    print(result.data)


asyncio.run(main())
```

### FastAPI route

```python
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from xoin import StructuredOutput, Xoin
from xoin.errors import StructuredOutputError
from xoin.providers import OpenAIProvider


class Summary(BaseModel):
    summary: str


class Body(BaseModel):
    text: str


app = FastAPI()

_xoin = Xoin(
    providers={"openai": OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])},
    default_provider="openai",
)


@app.post("/summarize")
async def summarize(body: Body) -> Summary:
    try:
        result = await _xoin.generate(
            provider="openai",
            prompt=f"Summarize:\n{body.text}",
            structured=StructuredOutput(response_model=Summary),
        )
    except StructuredOutputError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    assert result.data is not None
    return result.data
```

> **Lifecycle:** in production, prefer a shared **`httpx.AsyncClient`** passed into **`Xoin(client=...)`** or manage startup/shutdown hooks instead of creating a new **`Xoin`** per request.

## Custom providers

Implement the **`Provider`** protocol from **`xoin.providers.base`**: supply **`name`**, **`capabilities`**, **`default_model`**, **`default_embedding_model`**, and async **`generate` / `embed`** methods accepting **`httpx.AsyncClient`** plus **`ChatCompletionParameters` / `EmbeddingParameters`**.

```python
import httpx

from xoin.providers.base import (
    Capabilities,
    ChatCompletionParameters,
    EmbeddingParameters,
    ProviderChatResponse,
    ProviderEmbeddingResponse,
)
from xoin.types import Usage


class GatewayProvider:
    name = "gateway"
    capabilities = Capabilities(structured_outputs="prompt-only", embeddings=True)
    default_model = "gateway-chat"
    default_embedding_model = "gateway-embed"

    async def generate(
        self, client: httpx.AsyncClient, parameters: ChatCompletionParameters
    ) -> ProviderChatResponse:
        response = await client.post(
            "https://my-gateway.example.com/chat",
            json={
                "model": parameters.model,
                "messages": [m.model_dump() for m in parameters.messages],
                "temperature": parameters.temperature,
                "max_tokens": parameters.max_tokens,
                **parameters.provider_options,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return ProviderChatResponse(
            model=payload.get("model", parameters.model),
            text=payload["text"],
            structured_data=None,
            usage=None,
            finish_reason=payload.get("finish_reason"),
            raw=payload,
        )

    async def embed(self, client: httpx.AsyncClient, parameters: EmbeddingParameters) -> ProviderEmbeddingResponse:
        response = await client.post(
            "https://my-gateway.example.com/embed",
            json={"model": parameters.model, "input": parameters.input, **parameters.provider_options},
        )
        response.raise_for_status()
        payload = response.json()
        return ProviderEmbeddingResponse(
            model=payload.get("model", parameters.model),
            embeddings=payload["embeddings"],
            usage=None,
            raw=payload,
        )


xoin = Xoin(providers={"gateway": GatewayProvider()})
```

## Error handling

Exceptions live under **`xoin.errors`**:

| Class | When |
|-------|------|
| `XoinError` | Base class (`code` attribute) |
| `TemplateError` | Missing variables / malformed template files |
| `StructuredOutputError` | JSON parse / **Pydantic** validation failure |
| `ProviderExecutionError` | Provider HTTP/runtime failures surfaced by xoin-py |
| `ProviderConfigurationError` | Missing provider, model, embedding capability, etc. |
| `EmbeddingError` | Embedding not supported on provider |
| `AggregateProviderError` | All fallback providers failed |

```python
from xoin.errors import (
    AggregateProviderError,
    ProviderConfigurationError,
    ProviderExecutionError,
    StructuredOutputError,
    TemplateError,
)

try:
    result = await xoin.generate(
        provider="openai",
        prompt="Extract a user.",
        structured=StructuredOutput(response_model=UserProfile),
    )
    print(result.data)
except TemplateError:
    print("Prompt template configuration failed.")
except StructuredOutputError:
    print("Model output did not match the schema.")
except ProviderConfigurationError:
    print("Misconfigured provider or missing default model.")
except ProviderExecutionError as exc:
    print(f"{exc.provider} failed:", exc)
except AggregateProviderError as exc:
    print("All providers failed:", exc.errors)
```

## Examples

See `examples/README.md` for fully commented scripts (structured outputs, embeddings, `generate_many`,
priority `provider_targets`, templates, retries, and runtime `register_provider`).

## Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests -v --tb=short
```

Tests use **`httpx.MockTransport`** — no real provider keys required. See **`TEST_REPORT.md`** for the latest local run summary.

---

**Related:** JavaScript / TypeScript client — **[xoin-js](https://github.com/kanha95/xoin-js)** (`@xoin/xoin-js`).
