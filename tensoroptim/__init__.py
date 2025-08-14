"""
TensorOptim - Advanced Shared Tensor Management Library

A high-performance tensor optimization and memory management library designed for
distributed reinforcement learning applications, particularly A3C implementations.

Key Features:
- Ultra-fast shared tensor management
- NUMA-aware memory allocation
- Advanced compression and serialization
- Multi-backend memory support (mmap, shared memory, CUDA IPC)
- High-performance codec with parallel processing
- Comprehensive performance profiling
"""

from __future__ import annotations

__version__ = "3.0.0"
__author__ = "Tensor Memory Systems"
__email__ = "info@tensormemory.com"
__license__ = "MIT"

# Core components
from .core.manager import (
    SharedTensorManager,
    create_manager,
    create_optimized_manager_for_cuda,
    create_high_throughput_manager,
    create_memory_efficient_manager,
    get_default_manager
)
from .core.tensor import SharedTensor
from .core.pool import TensorPool
from .core.registry import TensorRegistry

# Memory backends
from .memory.backends import MemoryBackendType
from .memory.allocators import SlabAllocator
from .memory.segments import (
    HugePagesSegment,
    NumaAwareSharedMemory,
    OptimizedBuffer
)

# Codecs and compression
from .codecs.codec import TensorCodec
from .codecs.compression import CompressionType

# Performance and profiling
from .profiling.profiler import PerformanceProfiler
from .profiling.metrics import TensorMetrics

# Types and descriptors
from .types.descriptors import TensorDescriptor
from .types.enums import (
    TensorLifecycleState,
    AllocationStrategy
)
from .types.protocols import (
    IMemorySegment,
    IAllocator,
    ICodec
)

# Exceptions
from .exceptions import (
    TensorError,
    MemoryPoolExhausted,
    TensorCorruption,
    AllocationFailure
)

# Public API
__all__ = [
    # Core components
    "SharedTensorManager",
    "create_manager",
    "create_optimized_manager_for_cuda", 
    "create_high_throughput_manager",
    "create_memory_efficient_manager",
    "get_default_manager",
    "SharedTensor", 
    "TensorPool",
    "TensorRegistry",
    
    # Memory backends
    "MemoryBackendType",
    "SlabAllocator",
    "HugePagesSegment",
    "NumaAwareSharedMemory",
    "OptimizedBuffer",
    
    # Codecs
    "TensorCodec",
    "CompressionType",
    
    # Performance
    "PerformanceProfiler",
    "TensorMetrics",
    
    # Types
    "TensorDescriptor",
    "TensorLifecycleState",
    "AllocationStrategy",
    "IMemorySegment",
    "IAllocator", 
    "ICodec",
    
    # Exceptions
    "TensorError",
    "MemoryPoolExhausted",
    "TensorCorruption",
    "AllocationFailure",
]

# Version info
VERSION_INFO = tuple(map(int, __version__.split('.')))

def get_version() -> str:
    """Get the current version string."""
    return __version__

def get_version_info() -> tuple[int, ...]:
    """Get version as tuple of integers."""
    return VERSION_INFO