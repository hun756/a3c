"""
Ultra shared tensor manager implementation for TensorOptim library.

This module provides the main manager class for coordinating tensor operations
with background optimization and comprehensive resource management.
"""

from __future__ import annotations
import threading
import time
from contextlib import contextmanager, asynccontextmanager
from functools import lru_cache
from threading import RLock
from typing import Optional, Any, Dict, AsyncIterator, Iterator, Tuple

import torch

from ..types.aliases import TensorID, ByteSize
from ..types.enums import MemoryBackendType, AllocationStrategy
from ..exceptions import TensorError
from .pool import TensorPool
from .tensor import SharedTensor


class TensorManager:
    """Ultra-high performance shared tensor manager with background optimization."""
    
    __slots__ = ('_pool', '_lock', '_closed', '_background_optimizer', '_optimizer_interval')
    
    def __init__(
        self,
        backend: MemoryBackendType = MemoryBackendType.HUGEPAGES,
        max_memory: ByteSize = ByteSize(4 * 1024**3),
        enable_background_optimization: bool = True,
        optimizer_interval: float = 300.0,
        **kwargs
    ):
        self._pool = TensorPool(backend=backend, max_memory=max_memory, **kwargs)
        self._lock = RLock()
        self._closed = False
        self._optimizer_interval = optimizer_interval
        
        if enable_background_optimization:
            self._background_optimizer = threading.Thread(
                target=self._background_optimization_loop,
                daemon=True,
                name="UltraSharedTensorOptimizer"
            )
            self._background_optimizer.start()
        else:
            self._background_optimizer = None
    
    @contextmanager
    def share_tensor(self, tensor: torch.Tensor) -> Iterator[SharedTensor[torch.Tensor]]:
        """Context manager for sharing a tensor with automatic cleanup."""
        with self._lock:
            if self._closed:
                raise TensorError("Manager is closed")
            
            shared = self._pool.share_tensor(tensor)
            try:
                yield shared
            finally:
                shared.detach_optimized()
    
    @asynccontextmanager
    async def share_tensor_async(self, tensor: torch.Tensor) -> AsyncIterator[SharedTensor[torch.Tensor]]:
        """Async context manager for sharing a tensor with automatic cleanup."""
        if self._closed:
            raise TensorError("Manager is closed")
        
        shared = await self._pool.share_tensor_async(tensor)
        try:
            yield shared
        finally:
            shared.detach_optimized()
    
    def share_tensor_persistent(self, tensor: torch.Tensor) -> SharedTensor[torch.Tensor]:
        """Share a tensor persistently (manual cleanup required)."""
        with self._lock:
            if self._closed:
                raise TensorError("Manager is closed")
            return self._pool.share_tensor(tensor)
    
    async def share_tensor_persistent_async(self, tensor: torch.Tensor) -> SharedTensor[torch.Tensor]:
        """Asynchronously share a tensor persistently."""
        if self._closed:
            raise TensorError("Manager is closed")
        return await self._pool.share_tensor_async(tensor)
    
    def get_tensor(self, tensor_id: TensorID) -> Optional[SharedTensor[torch.Tensor]]:
        """Get a shared tensor by ID."""
        with self._lock:
            if self._closed:
                return None
            return self._pool.get_tensor(tensor_id)
    
    async def get_tensor_async(self, tensor_id: TensorID) -> Optional[SharedTensor[torch.Tensor]]:
        """Asynchronously get a shared tensor by ID."""
        if self._closed:
            return None
        return await self._pool.get_tensor_async(tensor_id)
    
    def get_tensors_by_criteria(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[str] = None,
        device: Optional[str] = None
    ) -> List[SharedTensor[torch.Tensor]]:
        """Get tensors matching specific criteria."""
        with self._lock:
            if self._closed:
                return []
            return self._pool.get_tensors_by_criteria(shape, dtype, device)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        with self._lock:
            if self._closed:
                return {}
            return self._pool.get_stats()
    
    def optimize_memory_now(self) -> Dict[str, int]:
        """Manually trigger memory optimization."""
        with self._lock:
            if self._closed:
                return {}
            return self._pool.optimize_memory_layout()
    
    def cleanup_expired_tensors(self, max_age_seconds: float = 3600.0) -> int:
        """Cleanup expired tensors."""
        with self._lock:
            if self._closed:
                return 0
            return self._pool.cleanup_expired_tensors(max_age_seconds)
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        with self._lock:
            if self._closed:
                return {}
            
            stats = self._pool.get_stats()
            return {
                'backend': stats.get('backend', 'unknown'),
                'numa_nodes': stats.get('numa_nodes', []),
                'compression_enabled': stats.get('compression_enabled', False),
                'memory_usage': stats.get('memory_usage', {}),
                'registry_stats': stats.get('registry', {}),
                'allocator_stats': stats.get('allocator', {})
            }
    
    def _background_optimization_loop(self) -> None:
        """Background optimization loop."""
        while not self._closed:
            try:
                time.sleep(self._optimizer_interval)
                if not self._closed:
                    self._pool.optimize_memory_layout()
                    
                    self._pool.cleanup_expired_tensors(3600.0)
                    
            except Exception:
                pass
    
    def set_optimizer_interval(self, interval_seconds: float) -> None:
        """Set the background optimizer interval."""
        if interval_seconds > 0:
            self._optimizer_interval = interval_seconds
    
    def is_closed(self) -> bool:
        """Check if the manager is closed."""
        return self._closed
    
    def close(self) -> None:
        """Close the manager and cleanup resources."""
        with self._lock:
            if not self._closed:
                self._closed = True
                
                if self._background_optimizer and self._background_optimizer.is_alive():
                    self._background_optimizer.join(timeout=1.0)
                
                self._pool.cleanup_resources()
    
    @property
    def backend_type(self) -> MemoryBackendType:
        """Get the backend type."""
        return self._pool.backend_type
    
    @property
    def compression_enabled(self) -> bool:
        """Check if compression is enabled."""
        return self._pool.compression_enabled
    
    @property
    def numa_nodes(self) -> List[int]:
        """Get available NUMA nodes."""
        return self._pool.numa_nodes
    
    def __enter__(self) -> TensorManager:
        """Context manager entry."""
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit with cleanup."""
        self.close()
    
    async def __aenter__(self) -> TensorManager:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, *args) -> None:
        """Async context manager exit with cleanup."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation of the manager."""
        return (
            f"TensorManager(backend={self.backend_type.name}, "
            f"compression={self.compression_enabled}, closed={self._closed})"
        )
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close()
        except Exception:
            pass



@lru_cache(maxsize=1)
def get_default_manager() -> TensorManager:
    """Get the default ultra shared tensor manager (singleton)."""
    return TensorManager()


def create_manager(**kwargs) -> TensorManager:
    """Create a new ultra shared tensor manager with custom configuration."""
    return TensorManager(**kwargs)


def create_optimized_manager_for_cuda() -> TensorManager:
    """Create a manager optimized for CUDA workloads."""
    return TensorManager(
        backend=MemoryBackendType.CUDA_IPC if torch.cuda.is_available() else MemoryBackendType.HUGEPAGES,
        compression=False,
        numa_aware=True,
        enable_background_optimization=True
    )


def create_high_throughput_manager() -> TensorManager:
    """Create a manager optimized for high throughput scenarios."""
    return TensorManager(
        backend=MemoryBackendType.HUGEPAGES,
        max_memory=ByteSize(16 * 1024**3),  # 16GB
        compression=True,
        numa_aware=True,
        allocation_strategy=AllocationStrategy.SLAB_ALLOCATOR,
        enable_background_optimization=True,
        optimizer_interval=180.0  # 3 minutes
    )


def create_memory_efficient_manager() -> TensorManager:
    """Create a manager optimized for memory efficiency."""
    return TensorManager(
        backend=MemoryBackendType.POSIX_SHM,
        max_memory=ByteSize(2 * 1024**3),  # 2GB
        compression=True,
        numa_aware=False,
        allocation_strategy=AllocationStrategy.SLAB_ALLOCATOR,
        enable_background_optimization=True,
        optimizer_interval=120.0  # 2 minutes
    )