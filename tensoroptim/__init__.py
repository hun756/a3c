from __future__ import annotations

__version__ = "1.0.0"
__author__ = "TensorOptim Team"
__license__ = "MIT"

from .types import (
    TensorID,
    MemoryOffset,
    ByteSize,
    MemoryBackendType,
    TensorLifecycleState,
    CompressionType,
    AllocationStrategy,
    IMemorySegment,
    IAllocator,
    ICodec,
    TensorDescriptor,
)

from .exceptions import (
    TensorOptimError,
    MemoryError,
    AllocationFailure,
    MemoryPoolExhausted,
    TensorError,
    TensorCorruption,
    TensorNotFound,
    CompressionError,
    BackendError,
)

__all__ = [
    "TensorID",
    "MemoryOffset",
    "ByteSize",
    "MemoryBackendType",
    "TensorLifecycleState",
    "CompressionType",
    "AllocationStrategy",
    "IMemorySegment",
    "IAllocator",
    "ICodec",
    "TensorDescriptor",
    "TensorOptimError",
    "MemoryError",
    "AllocationFailure",
    "MemoryPoolExhausted",
    "TensorError",
    "TensorCorruption",
    "TensorNotFound",
    "CompressionError",
    "BackendError",
]

VERSION_INFO = tuple(map(int, __version__.split('.')))

def get_version() -> str:
    return __version__

def get_version_info() -> tuple[int, ...]:
    return VERSION_INFO