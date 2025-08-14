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
    TensorMetrics,
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

from .core import (
    TensorReference,
    SharedTensor,
    TensorRegistry,
    TensorManager,
)

from .memory import (
    MemoryBackend,
    HugePagesBackend,
    PosixShmBackend,
    CudaIpcBackend,
    SlabAllocator,
)

from .codecs.codec import TensorCodec

from .factory import (
    get_default_manager,
    create_manager,
    create_optimized_manager_for_cuda,
    create_high_throughput_manager,
    create_memory_efficient_manager,
)

from .profiling import PerformanceProfiler

__all__ = [
    # Types
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
    "TensorMetrics",
    
    # Exceptions
    "TensorOptimError",
    "MemoryError",
    "AllocationFailure",
    "MemoryPoolExhausted",
    "TensorError",
    "TensorCorruption",
    "TensorNotFound",
    "CompressionError",
    "BackendError",
    
    # Core Components
    "TensorReference",
    "SharedTensor",
    "TensorRegistry",
    "TensorManager",
    
    # Memory Components
    "MemoryBackend",
    "HugePagesBackend",
    "PosixShmBackend",
    "CudaIpcBackend",
    "SlabAllocator",
    
    # Codecs
    "TensorCodec",
    
    # Factory Functions
    "get_default_manager",
    "create_manager",
    "create_optimized_manager_for_cuda",
    "create_high_throughput_manager",
    "create_memory_efficient_manager",
    
    # Profiling
    "PerformanceProfiler",
]

VERSION_INFO = tuple(map(int, __version__.split('.')))

def get_version() -> str:
    return __version__

def get_version_info() -> tuple[int, ...]:
    return VERSION_INFO