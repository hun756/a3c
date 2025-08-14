"""
Protocol definitions for TensorOptim library.

This module defines protocols (interfaces) that define the contracts
for various components in the TensorOptim library.
"""

from __future__ import annotations
from typing import Protocol, TypeVar, runtime_checkable, Tuple, Dict
import torch

from .aliases import MemoryOffset, ByteSize
from .enums import CompressionType, AllocationStrategy
from .descriptors import TensorDescriptor

TensorT = TypeVar('TensorT', bound=torch.Tensor)


@runtime_checkable
class IMemorySegment(Protocol):
    """Protocol for advanced memory segment implementations."""
    
    @property
    def size(self) -> ByteSize:
        """Get the size of the memory segment."""
        ...
    
    @property
    def virtual_address(self) -> int:
        """Get the virtual address of the memory segment."""
        ...
    
    @property
    def numa_node(self) -> int:
        """Get the NUMA node this segment is bound to."""
        ...
    
    def read_vectorized(self, offset: MemoryOffset, size: ByteSize) -> memoryview:
        """Read data from the segment using vectorized operations."""
        ...
    
    def write_vectorized(self, offset: MemoryOffset, data: bytes | memoryview) -> None:
        """Write data to the segment using vectorized operations."""
        ...
    
    def prefetch(self, offset: MemoryOffset, size: ByteSize) -> None:
        """Prefetch data into cache."""
        ...
    
    def madvise_sequential(self) -> None:
        """Advise kernel that access will be sequential."""
        ...
    
    def madvise_random(self) -> None:
        """Advise kernel that access will be random."""
        ...
    
    def madvise_willneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        """Advise kernel that data will be needed soon."""
        ...
    
    def madvise_dontneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        """Advise kernel that data is not needed."""
        ...
    
    def sync_async(self) -> None:
        """Asynchronously sync data to storage."""
        ...
    
    def sync_sync(self) -> None:
        """Synchronously sync data to storage."""
        ...
    
    def get_page_faults(self) -> int:
        """Get the number of page faults for this segment."""
        ...
    
    def close(self) -> None:
        """Close and cleanup the memory segment."""
        ...


@runtime_checkable
class IAllocator(Protocol):
    """Protocol for advanced memory allocator implementations."""
    
    def allocate_aligned(
        self, 
        size: ByteSize, 
        alignment: int = 64, 
        numa_node: int = -1
    ) -> Tuple[MemoryOffset, IMemorySegment]:
        """Allocate aligned memory."""
        ...
    
    def deallocate_fast(self, offset: MemoryOffset, segment: IMemorySegment) -> None:
        """Fast deallocation of memory."""
        ...
    
    def reallocate(
        self, 
        offset: MemoryOffset, 
        old_size: ByteSize, 
        new_size: ByteSize, 
        segment: IMemorySegment
    ) -> MemoryOffset:
        """Reallocate memory to a new size."""
        ...
    
    def defragment_concurrent(self) -> int:
        """Perform concurrent defragmentation."""
        ...
    
    def get_fragmentation_ratio(self) -> float:
        """Get the current fragmentation ratio."""
        ...
    
    def get_utilization_stats(self) -> Dict[str, float]:
        """Get memory utilization statistics."""
        ...
    
    def set_allocation_strategy(self, strategy: AllocationStrategy) -> None:
        """Set the allocation strategy."""
        ...
    
    def enable_compaction(self, enable: bool) -> None:
        """Enable or disable memory compaction."""
        ...


@runtime_checkable
class ICodec(Protocol[TensorT]):
    """Protocol for ultra-high performance tensor codec implementations."""
    
    def encode_parallel(
        self, 
        tensor: TensorT, 
        compression: CompressionType = CompressionType.NONE
    ) -> bytes:
        """Encode tensor to bytes using parallel processing."""
        ...
    
    def decode_parallel(
        self, 
        data: bytes | memoryview, 
        descriptor: TensorDescriptor
    ) -> TensorT:
        """Decode bytes to tensor using parallel processing."""
        ...
    
    def validate_integrity_fast(
        self, 
        data: bytes | memoryview, 
        expected_checksum: int
    ) -> bool:
        """Fast integrity validation of tensor data."""
        ...
    
    def compute_checksum_simd(self, tensor: TensorT) -> int:
        """Compute checksum using SIMD instructions."""
        ...
    
    def estimate_compression_ratio(self, tensor: TensorT) -> float:
        """Estimate compression ratio for the tensor."""
        ...
    
    def benchmark_compression_speed(self, tensor: TensorT) -> Dict[CompressionType, float]:
        """Benchmark compression speed for different algorithms."""
        ...