from __future__ import annotations
from typing import Protocol, TypeVar, runtime_checkable, Tuple, Dict
import torch

from .aliases import MemoryOffset, ByteSize
from .enums import CompressionType, AllocationStrategy
from .descriptors import TensorDescriptor

TensorT = TypeVar('TensorT', bound=torch.Tensor)


@runtime_checkable
class IMemorySegment(Protocol):
    @property
    def size(self) -> ByteSize:
        ...
    
    @property
    def virtual_address(self) -> int:
        ...
    
    @property
    def numa_node(self) -> int:
        ...
    
    def read_vectorized(self, offset: MemoryOffset, size: ByteSize) -> memoryview:
        ...
    
    def write_vectorized(self, offset: MemoryOffset, data: bytes | memoryview) -> None:
        ...
    
    def prefetch(self, offset: MemoryOffset, size: ByteSize) -> None:
        ...
    
    def madvise_sequential(self) -> None:
        ...
    
    def madvise_random(self) -> None:
        ...
    
    def madvise_willneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        ...
    
    def madvise_dontneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        ...
    
    def sync_async(self) -> None:
        ...
    
    def sync_sync(self) -> None:
        ...
    
    def get_page_faults(self) -> int:
        ...
    
    def close(self) -> None:
        ...


@runtime_checkable
class IAllocator(Protocol):
    def allocate_aligned(
        self, 
        size: ByteSize, 
        alignment: int = 64, 
        numa_node: int = -1
    ) -> Tuple[MemoryOffset, IMemorySegment]:
        ...
    
    def deallocate_fast(self, offset: MemoryOffset, segment: IMemorySegment) -> None:
        ...
    
    def reallocate(
        self, 
        offset: MemoryOffset, 
        old_size: ByteSize, 
        new_size: ByteSize, 
        segment: IMemorySegment
    ) -> MemoryOffset:
        ...
    
    def defragment_concurrent(self) -> int:
        ...
    
    def get_fragmentation_ratio(self) -> float:
        ...
    
    def get_utilization_stats(self) -> Dict[str, float]:
        ...
    
    def set_allocation_strategy(self, strategy: AllocationStrategy) -> None:
        ...
    
    def enable_compaction(self, enable: bool) -> None:
        ...


@runtime_checkable
class ICodec(Protocol[TensorT]):
    def encode_parallel(
        self, 
        tensor: TensorT, 
        compression: CompressionType = CompressionType.NONE
    ) -> bytes:
        ...
    
    def decode_parallel(
        self, 
        data: bytes | memoryview, 
        descriptor: TensorDescriptor
    ) -> TensorT:
        ...
    
    def validate_integrity_fast(
        self, 
        data: bytes | memoryview, 
        expected_checksum: int
    ) -> bool:
        ...
    
    def compute_checksum_simd(self, tensor: TensorT) -> int:
        ...
    
    def estimate_compression_ratio(self, tensor: TensorT) -> float:
        ...
    
    def benchmark_compression_speed(self, tensor: TensorT) -> Dict[CompressionType, float]:
        ...