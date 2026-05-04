# xoin-py examples

These scripts mirror the **standalone JavaScript samples** in
[`xoin-js/examples/native-js`](https://github.com/kanha95/xoin-js/tree/main/examples/native-js):

structured extraction across vendors, embeddings, parallel **`generate_many`** fan‑out,
priority‑sorted **`provider_targets`** failover, retries/backoff, template loading,
runtime **`register_provider`**, and an OpenAI‑compatible Groq endpoint.

Each file is **self-contained**, heavily commented, and intended as production-grade starter code.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[examples]"
```

Copy the environment template and add API keys:

```bash
cp examples/.env.example examples/.env
```

Load variables **without printing secrets**:

```bash
set -a
source examples/.env
set +a
```

## Running an example

Always execute modules **from the repository root** so paths resolve consistently:

```bash
python examples/anthropic_structured_output.py
```

Windows PowerShell users can mirror loading via `$Env:OPENAI_API_KEY="..."` or a `.env` loader.

## What lives here

| File | Mirrors JS |
|------|-----------|
| `openai_structured_output.py` | `openai-structured-output.js` (deep nested healthcare extraction) |
| `anthropic_structured_output.py` | `anthropic-structured-output.js` |
| `mistral_structured_output.py` | `mistral-structured-output.js` |
| `deepseek_structured_output.py` | `deepseek-structured-output.js` |
| `groq_openai_compatible.py` | `groq-openai-compatible.js` |
| `embeddings_openai.py` | `embeddings-openai.js` |
| `generate_many.py` | `generate-many.js` |
| `fallback_order.py` | `fallback-order.js` |
| `provider_targets_priority.py` | Priority semantics used throughout JS README (`providerTargets`) |
| `template_file.py` | `template-file.js` |
| `named_templates_registry.py` | `templateId` + `templates={}` registry usage |
| `retry_with_backoff.py` | `retry` object patterns |
| `register_provider_runtime.py` | parity with JS `registerProvider` |

Templates depend on **PyYAML** (included via `.[examples]`).
