class XoinError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class TemplateError(XoinError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "TEMPLATE_ERROR")


class StructuredOutputError(XoinError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "STRUCTURED_OUTPUT_ERROR")


class ProviderExecutionError(XoinError):
    def __init__(self, message: str, provider: str, model: str | None = None) -> None:
        super().__init__(message, "PROVIDER_EXECUTION_ERROR")
        self.provider = provider
        self.model = model


class ProviderConfigurationError(XoinError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "PROVIDER_CONFIGURATION_ERROR")


class EmbeddingError(XoinError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "EMBEDDING_ERROR")


class AggregateProviderError(XoinError):
    def __init__(self, errors: list[ProviderExecutionError]) -> None:
        detail = ", ".join(f"{e.provider}{f'/{e.model}' if e.model else ''}" for e in errors)
        super().__init__(f"All providers failed: {detail}", "ALL_PROVIDERS_FAILED")
        self.errors = errors
