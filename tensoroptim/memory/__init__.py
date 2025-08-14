"""
Memory management components for TensorOptim library.

This module provides various memory backends, allocators, and segments
for efficient tensor storage and management.
"""

from .backends import MemoryBackendType
from .allocators import SlabAllocator
from .segments import (
    HugePagesSegment,
    NumaAwareSharedMemory,
    OptimizedBuffer
)

__all__ = [
    "MemoryBackendType",
    "SlabAllocator", 
    "HugePagesSegment",
    "NumaAwareSharedMemory",
    "OptimizedBuffer",
]