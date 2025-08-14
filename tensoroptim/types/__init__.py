"""
Type definitions and protocols for TensorOptim library.

This module provides type definitions, protocols, and data structures
used throughout the TensorOptim library for type safety and clarity.
"""

from .descriptors import TensorDescriptor, TensorMetrics
from .enums import (
    MemoryBackendType,
    TensorLifecycleState, 
    CompressionType,
    AllocationStrategy
)
from .protocols import (
    IMemorySegment,
    IAllocator,
    ICodec
)
from .aliases import (
    TensorID,
    MemoryOffset,
    ByteSize
)

__all__ = [
    # Descriptors
    "TensorDescriptor",
    "TensorMetrics",
    
    # Enums
    "MemoryBackendType",
    "TensorLifecycleState",
    "CompressionType", 
    "AllocationStrategy",
    
    # Protocols
    "IMemorySegment",
    "IAllocator",
    "ICodec",
    
    # Type aliases
    "TensorID",
    "MemoryOffset",
    "ByteSize",
]