"""
Data descriptors and metrics for TensorOptim library.

This module defines data classes and structures used to describe
tensor metadata, metrics, and configuration throughout the library.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property, reduce
from operator import mul
from typing import Tuple, Dict
import torch

from .aliases import TensorID, ByteSize
from .enums import CompressionType


@dataclass(frozen=True, slots=True, unsafe_hash=True)
class TensorMetrics:
    """Metrics and performance data for tensor operations."""
    
    creation_time: float
    access_count: int = 0
    last_access: float = 0.0
    serialization_time: float = 0.0
    deserialization_time: float = 0.0
    compression_ratio: float = 1.0
    memory_efficiency: float = 1.0
    
    def update_access(self, timestamp: float) -> TensorMetrics:
        """Create a new metrics instance with updated access information."""
        return TensorMetrics(
            creation_time=self.creation_time,
            access_count=self.access_count + 1,
            last_access=timestamp,
            serialization_time=self.serialization_time,
            deserialization_time=self.deserialization_time,
            compression_ratio=self.compression_ratio,
            memory_efficiency=self.memory_efficiency
        )


@dataclass(frozen=True, unsafe_hash=True)
class TensorDescriptor:
    """Advanced descriptor containing comprehensive tensor metadata."""
    
    tensor_id: TensorID
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    stride: Tuple[int, ...]
    storage_offset: int
    requires_grad: bool
    timestamp: float
    checksum: int
    compression_type: CompressionType
    metrics: TensorMetrics
    numa_node: int = -1
    alignment: int = 64
    
    @cached_property
    def element_count(self) -> int:
        """Calculate total number of elements in the tensor."""
        return reduce(mul, self.shape, 1) if self.shape else 0
    
    @cached_property
    def element_size(self) -> int:
        """Get the size of each element in bytes."""
        return torch.empty(0, dtype=self.dtype).element_size()
    
    @cached_property
    def raw_byte_size(self) -> ByteSize:
        """Calculate raw byte size without alignment."""
        return ByteSize(self.element_count * self.element_size)
    
    @cached_property
    def aligned_byte_size(self) -> ByteSize:
        """Calculate aligned byte size with padding."""
        size = self.raw_byte_size
        return ByteSize(((size + self.alignment - 1) // self.alignment) * self.alignment)
    
    @cached_property
    def cache_key(self) -> str:
        """Generate a unique cache key for this tensor descriptor."""
        return f"{self.tensor_id}_{hash((self.shape, self.dtype, self.device))}"
    
    def is_compatible_with(self, other: 'TensorDescriptor') -> bool:
        """Check if this descriptor is compatible with another for operations."""
        return (
            self.shape == other.shape and
            self.dtype == other.dtype and
            self.device == other.device
        )
    
    def get_memory_footprint(self) -> Dict[str, int]:
        """Get detailed memory footprint information."""
        return {
            'raw_bytes': self.raw_byte_size,
            'aligned_bytes': self.aligned_byte_size,
            'element_count': self.element_count,
            'element_size': self.element_size,
            'alignment_overhead': self.aligned_byte_size - self.raw_byte_size
        }