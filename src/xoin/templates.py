"""Template loading and ``{{variable}}`` rendering (xoin-js parity)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from xoin.errors import TemplateError
from xoin.types import TemplateDefinition


def render_template(definition: TemplateDefinition, variables: Mapping[str, Any] | None = None) -> str:
    """Replace ``{{ keys }}`` using defaults plus runtime ``variables``."""

    values: dict[str, Any] = {**definition.defaults, **dict(variables or {})}

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise TemplateError(f'Missing template variable "{key}".')
        value = values[key]
        if value is None:
            return ""
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return re.sub(r"{{\s*([a-zA-Z0-9_.$-]+)\s*}}", replace, definition.template)


def load_template_file(path: str | Path) -> TemplateDefinition:
    """Load a YAML, JSON, or plain-text template definition from disk."""

    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise TemplateError(f'Unable to read template file "{path}".') from e

    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise TemplateError(
                'Reading YAML templates requires PyYAML. Install with: pip install "pyyaml>=6"'
            ) from e
        parsed = yaml.safe_load(raw)
        return _normalize_definition(parsed, label=str(p))

    if suffix == ".json":
        parsed = json.loads(raw)
        return _normalize_definition(parsed, label=str(p))

    return TemplateDefinition(template=raw)


def resolve_named_template(
    *,
    inline_template: str | None,
    template_id: str | None,
    template_file: str | Path | None,
    templates: Mapping[str, TemplateDefinition] | None,
) -> TemplateDefinition | None:
    """Pick inline text, registry id, or file — same precedence idea as xoin-js."""

    if inline_template is not None:
        return TemplateDefinition(template=inline_template)

    if template_id is not None:
        if templates is None or template_id not in templates:
            raise TemplateError(f'Unknown template id "{template_id}".')
        return templates[template_id]

    if template_file is not None:
        return load_template_file(template_file)

    return None


def _normalize_definition(value: Any, *, label: str) -> TemplateDefinition:
    if isinstance(value, str):
        return TemplateDefinition(template=value)

    if not isinstance(value, dict) or "template" not in value or not isinstance(value["template"], str):
        raise TemplateError(f'Template "{label}" must contain a string "template" field.')

    defaults = value.get("defaults")
    if defaults is not None and not isinstance(defaults, dict):
        raise TemplateError(f'Template "{label}" defaults must be a mapping.')

    description = value.get("description")
    if description is not None and not isinstance(description, str):
        raise TemplateError(f'Template "{label}" description must be a string.')

    return TemplateDefinition(
        template=value["template"],
        defaults=dict(defaults or {}),
        description=description,
    )
