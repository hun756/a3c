"""
Enumeration types for TensorOptim library.

This module defines all enumeration types used throughout the library
for configuration, state management, and type safety.
"""

from enum import IntEnum, auto


class MemoryBackendType(IntEnum):
    """Memory backend types for tensor storage."""
    MMAP_PRIVATE = 1
    MMAP_SHARED = 2
    POSIX_SHM = 3
    SYSV_SHM = 4
    CUDA_IPC = 5
    HUGEPAGES = 6
    NUMA_AWARE = 7


class TensorLifecycleState(IntEnum):
    """Tensor lifecycle states for state management."""
    UNINITIALIZED = 0
    ALLOCATING = 1
    ALLOCATED = 2
    MATERIALIZING = 3
    ACTIVE = 4
    PERSISTING = 5
    CACHED = 6
    DETACHING = 7
    DETACHED = 8
    CORRUPTED = 9
    EXPIRED = 10


class CompressionType(IntEnum):
    """Compression algorithms for tensor data."""
    NONE = 0
    LZ4 = 1
    ZSTD = 2
    CUSTOM = 3


class AllocationStrategy(IntEnum):
    """Memory allocation strategies."""
    FIRST_FIT = 1
    BEST_FIT = 2
    WORST_FIT = 3
    BUDDY_SYSTEM = 4
    SLAB_ALLOCATOR = 5
    POOL_ALLOCATOR = 6