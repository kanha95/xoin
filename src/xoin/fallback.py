from collections.abc import Awaitable, Callable
from typing import TypeVar

from xoin.errors import AggregateProviderError, ProviderExecutionError

T = TypeVar("T")


async def run_fallback(calls: list[Callable[[], Awaitable[T]]]) -> T:
    errs: list[ProviderExecutionError] = []
    for fn in calls:
        try:
            return await fn()
        except ProviderExecutionError as e:
            errs.append(e)
            continue
    if len(errs) == 1:
        raise errs[0]
    raise AggregateProviderError(errs)
