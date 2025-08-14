from __future__ import annotations

import sys
from typing import Protocol, TypeVar, runtime_checkable

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

T = TypeVar('T')


@runtime_checkable
class Disposable(Protocol):
    def dispose(self) -> None: ...


@runtime_checkable
class AsyncDisposable(Protocol):
    async def dispose_async(self) -> None: ...


@runtime_checkable
class Resizable(Protocol[T]):
    def resize(self, new_size: int) -> T: ...


@runtime_checkable
class Copyable(Protocol[T]):
    def copy_from(self, source: T) -> None: ...
    def copy_to(self, destination: T) -> None: ...


@runtime_checkable
class Serializable(Protocol):
    def serialize(self) -> bytes: ...
    def deserialize(self, data: bytes) -> Self: ...


@runtime_checkable
class Cacheable(Protocol):
    def cache_key(self) -> str: ...
    def is_cache_valid(self) -> bool: ...