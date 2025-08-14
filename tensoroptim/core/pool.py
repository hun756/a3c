"""
Advanced tensor pool implementation for TensorOptim library.

This module provides high-performance tensor pooling with NUMA awareness,
compression, and advanced allocation strategies.
"""

from __future__ import annotations
import asyncio
import gc
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace as dataclass_replace
from pathlib import Path
from threading import RLock
from typing import Optional, Any, Dict, List, Tuple
from uuid import uuid4

import torch

from ..types.aliases import TensorID, ByteSize
from ..types.descriptors import TensorDescriptor, TensorMetrics
from ..types.enums import (
    MemoryBackendType, 
    CompressionType, 
    AllocationStrategy,
    TensorLifecycleState
)
from ..types.protocols import IAllocator, IMemorySegment
from ..memory.segments import HugePagesSegment, NumaAwareSharedMemory
from ..memory.allocators import SlabAllocator
from ..codecs.codec import TensorCodec
from ..profiling.profiler import PerformanceProfiler
from .registry import TensorRegistry
from .tensor import TensorReference, SharedTensor

# Optional dependencies
try:
    import numa
    HAS_NUMA = True
except ImportError:
    HAS_NUMA = False

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


def ultra_timed_operation(profiler_getter, operation: str, bytes_processed: int = 0):
    """Decorator for timing operations with profiler."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            profiler = profiler_getter(self)
            start = time.perf_counter()
            error = False
            actual_bytes = bytes_processed
            try:
                result = func(self, *args, **kwargs)
                if hasattr(result, '__len__'):
                    actual_bytes = len(result)
                return result
            except Exception:
                error = True
                raise
            finally:
                duration = time.perf_counter() - start
                profiler.record_operation_detailed(operation, duration, actual_bytes, error)
        return wrapper
    return decorator


class TensorPool:
    """Advanced tensor pool with NUMA awareness and compression."""
    
    __slots__ = (
        '_allocator', '_codec', '_registry', '_profiler', '_backend_type',
        '_segments', '_lock', '_executor', '_cleanup_task', '_numa_nodes',
        '_allocation_strategy', '_compression_enabled'
    )
    
    def __init__(
        self,
        backend: MemoryBackendType = MemoryBackendType.HUGEPAGES,
        max_memory: ByteSize = ByteSize(4 * 1024**3),
        compression: bool = True,
        numa_aware: bool = True,
        allocation_strategy: AllocationStrategy = AllocationStrategy.SLAB_ALLOCATOR
    ):
        self._backend_type = backend
        self._compression_enabled = compression
        self._allocation_strategy = allocation_strategy
        self._numa_nodes = self._detect_numa_nodes() if numa_aware else [-1]
        
        self._codec = TensorCodec(use_parallel=True)
        self._registry = TensorRegistry()
        self._profiler = PerformanceProfiler()
        self._segments: Dict[str, IMemorySegment] = {}
        self._lock = RLock()
        
        max_workers = min(64, (os.cpu_count() or 1) * 4)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ultra_tensor_pool"
        )
        self._cleanup_task: Optional[asyncio.Task] = None
        
        self._allocator = self._create_ultra_allocator(backend, max_memory)
    
    def _detect_numa_nodes(self) -> List[int]:
        """Detect available NUMA nodes."""
        if HAS_NUMA:
            return list(range(numa.get_max_node() + 1))
        return [-1]
    
    def _create_ultra_allocator(
        self, 
        backend: MemoryBackendType, 
        max_memory: ByteSize
    ) -> IAllocator:
        """Create allocator based on backend type."""
        segment_name = f"ultra_tensor_pool_{os.getpid()}_{id(self)}"
        
        if backend == MemoryBackendType.HUGEPAGES:
            if os.name == 'nt':
                segment_path = Path(f"C:/temp/{segment_name}.huge")
                segment_path.parent.mkdir(exist_ok=True)
            else:
                segment_path = Path(f"/dev/hugepages/{segment_name}.huge")
            segment = HugePagesSegment(segment_path, max_memory, numa_node=self._numa_nodes[0])
        elif backend == MemoryBackendType.NUMA_AWARE:
            segment = NumaAwareSharedMemory(segment_name, max_memory, numa_node=self._numa_nodes[0])
        elif backend == MemoryBackendType.POSIX_SHM:
            segment = NumaAwareSharedMemory(segment_name, max_memory)
        else:
            if os.name == 'nt':
                segment = NumaAwareSharedMemory(segment_name, max_memory)
            else:
                segment_path = Path(f"/tmp/{segment_name}.mmap")
                segment = HugePagesSegment(segment_path, max_memory)
        
        self._segments[segment_name] = segment
        
        if self._allocation_strategy == AllocationStrategy.SLAB_ALLOCATOR:
            return SlabAllocator(segment, ByteSize(1024 * 1024))
        else:
            raise NotImplementedError(f"Allocation strategy {self._allocation_strategy} not implemented")
    
    @ultra_timed_operation(lambda self: self._profiler, 'share_tensor')
    def share_tensor(self, tensor: torch.Tensor) -> SharedTensor[torch.Tensor]:
        """Share a tensor with ultra-fast access."""
        with self._lock:
            descriptor = self._create_descriptor(tensor)
            
            compression_type = CompressionType.NONE
            if self._compression_enabled:
                ratio = self._codec.estimate_compression_ratio(tensor)
                compression_type = CompressionType.LZ4 if ratio < 0.8 and HAS_LZ4 else CompressionType.NONE
            
            descriptor = dataclass_replace(descriptor, compression_type=compression_type)
            
            optimal_numa = self._select_optimal_numa_node(descriptor.raw_byte_size)
            offset, segment = self._allocator.allocate_aligned(
                descriptor.aligned_byte_size, 
                descriptor.alignment,
                optimal_numa
            )
            
            self._profiler.record_allocation(descriptor.aligned_byte_size, "tensor_allocation")
            
            encoded = self._codec.encode_parallel(tensor, compression_type)
            segment.write_vectorized(offset, encoded)
            
            reference = TensorReference(descriptor, segment, offset, self._codec)
            
            reference._state = TensorLifecycleState.ACTIVE
            
            self._registry.register_optimized(reference)
            
            return SharedTensor(reference)
    
    @ultra_timed_operation(lambda self: self._profiler, 'get_tensor')
    def get_tensor(self, tensor_id: TensorID) -> Optional[SharedTensor[torch.Tensor]]:
        """Get a shared tensor with ultra-fast access."""
        reference = self._registry.get_optimized(tensor_id)
        if reference:
            shared_tensor = SharedTensor(reference)
            shared_tensor.prefetch_async()
            return shared_tensor
        return None
    
    async def share_tensor_async(self, tensor: torch.Tensor) -> SharedTensor[torch.Tensor]:
        """Asynchronously share a tensor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.share_tensor, tensor)
    
    async def get_tensor_async(self, tensor_id: TensorID) -> Optional[SharedTensor[torch.Tensor]]:
        """Asynchronously get a shared tensor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.get_tensor, tensor_id)
    
    def _create_descriptor(self, tensor: torch.Tensor) -> TensorDescriptor:
        """Create advanced descriptor for tensor."""
        metrics = TensorMetrics(
            creation_time=time.perf_counter(),
            access_count=0,
            last_access=time.perf_counter()
        )
        
        return TensorDescriptor(
            tensor_id=TensorID(uuid4()),
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            requires_grad=tensor.requires_grad,
            timestamp=time.time(),
            checksum=self._codec.compute_checksum_simd(tensor),
            compression_type=CompressionType.NONE,
            metrics=metrics,
            numa_node=-1,
            alignment=64
        )
    
    def _select_optimal_numa_node(self, size: ByteSize) -> int:
        """Select optimal NUMA node for allocation."""
        if len(self._numa_nodes) == 1:
            return self._numa_nodes[0]
        
        utilization_stats = self._allocator.get_utilization_stats()
        utilization = utilization_stats.get('utilization', 0.0)
        
        if utilization < 0.7:
            return self._numa_nodes[0]
        
        return self._numa_nodes[hash(threading.current_thread()) % len(self._numa_nodes)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'performance': self._profiler.get_summary(),
            'registry': self._registry.get_access_statistics(),
            'allocator': self._allocator.get_utilization_stats(),
            'backend': self._backend_type.name,
            'numa_nodes': self._numa_nodes,
            'compression_enabled': self._compression_enabled,
            'memory_usage': self._registry.get_memory_usage_estimate()
        }
    
    def optimize_memory_layout(self) -> Dict[str, int]:
        """Optimize memory layout and perform cleanup."""
        defragmented = self._allocator.defragment_concurrent()
        
        for segment in self._segments.values():
            segment.madvise_sequential()
        
        collected = gc.collect()
        
        return {
            'defragmented_bytes': defragmented,
            'gc_collected': collected
        }
    
    def cleanup_expired_tensors(self, max_age_seconds: float = 3600.0) -> int:
        """Cleanup expired tensors."""
        return self._registry.cleanup_expired(max_age_seconds)
    
    def get_tensors_by_criteria(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[str] = None,
        device: Optional[str] = None
    ) -> List[SharedTensor[torch.Tensor]]:
        """Get tensors matching specific criteria."""
        references = self._registry.get_tensor_by_criteria(shape, dtype, device)
        return [SharedTensor(ref) for ref in references]
    
    def cleanup_resources(self) -> None:
        """Cleanup all resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        self._executor.shutdown(wait=True)
        
        self._registry.shutdown()
        
        for segment in self._segments.values():
            segment.close()
        
        gc.collect()
    
    @property
    def backend_type(self) -> MemoryBackendType:
        """Get the backend type."""
        return self._backend_type
    
    @property
    def compression_enabled(self) -> bool:
        """Check if compression is enabled."""
        return self._compression_enabled
    
    @property
    def numa_nodes(self) -> List[int]:
        """Get available NUMA nodes."""
        return self._numa_nodes.copy()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup_resources()
        except Exception:
            pass