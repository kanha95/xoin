"""xoin: structured LLM calls with Pydantic schemas across multiple providers."""

from xoin.client import Xoin, create_xoin
from xoin import errors
from xoin.templates import load_template_file, render_template, resolve_named_template
from xoin.types import (
    ChatMessage,
    EmbedResult,
    GenManyTarget,
    GenResult,
    PriorityProviderTarget,
    RetryCfg,
    StructuredSpec,
    TemplateDefinition,
    Usage,
)

__all__ = [
    "Xoin",
    "create_xoin",
    "errors",
    "ChatMessage",
    "StructuredSpec",
    "GenManyTarget",
    "PriorityProviderTarget",
    "GenResult",
    "EmbedResult",
    "Usage",
    "RetryCfg",
    "TemplateDefinition",
    "render_template",
    "resolve_named_template",
    "load_template_file",
]

__version__ = "0.1.0"
