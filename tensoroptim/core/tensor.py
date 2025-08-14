"""
Tensor reference and shared tensor implementations for TensorOptim library.

This module provides the core tensor abstraction with advanced lifecycle
management, weak references, and performance tracking.
"""

from __future__ import annotations
import time
import weakref
from dataclasses import replace as dataclass_replace
from threading import RLock
from typing import Generic, TypeVar, Optional, Any, cast, Tuple, Dict
from weakref import finalize

import torch

from ..types.aliases import TensorID, MemoryOffset, ByteSize
from ..types.descriptors import TensorDescriptor
from ..types.enums import TensorLifecycleState
from ..types.protocols import IMemorySegment, ICodec
from ..exceptions import TensorError, TensorCorruption

TensorT = TypeVar('TensorT', bound=torch.Tensor)


class TensorReference:
    """Advanced tensor reference with lifecycle management and weak references."""
    
    __slots__ = (
        '_descriptor', '_segment', '_offset', '_codec', '_state', '_lock',
        '_weak_tensor', '_access_count', '_last_access', '_finalizer', '__weakref__'
    )
    
    def __init__(
        self, 
        descriptor: TensorDescriptor, 
        segment: IMemorySegment, 
        offset: MemoryOffset, 
        codec: ICodec
    ):
        self._descriptor = descriptor
        self._segment = segment
        self._offset = offset
        self._codec = codec
        self._state = TensorLifecycleState.ALLOCATED
        self._lock = RLock()
        self._weak_tensor: Optional[weakref.ReferenceType] = None
        self._access_count = 0
        self._last_access = time.perf_counter()
        self._finalizer = finalize(self, self._cleanup_resources, segment, offset)
    
    @property
    def descriptor(self) -> TensorDescriptor:
        """Get the tensor descriptor."""
        return self._descriptor
    
    @property
    def state(self) -> TensorLifecycleState:
        """Get the current lifecycle state."""
        return self._state
    
    @property
    def is_valid(self) -> bool:
        """Check if the tensor reference is in a valid state."""
        return self._state in (TensorLifecycleState.ACTIVE, TensorLifecycleState.CACHED)
    
    @property
    def access_count(self) -> int:
        """Get the number of times this tensor has been accessed."""
        return self._access_count
    
    @property
    def last_access(self) -> float:
        """Get the timestamp of the last access."""
        return self._last_access
    
    def materialize_optimized(self) -> torch.Tensor:
        """Materialize the tensor from storage with optimizations."""
        with self._lock:
            self._access_count += 1
            self._last_access = time.perf_counter()
            
            if self._weak_tensor is not None:
                tensor = self._weak_tensor()
                if tensor is not None:
                    self._state = TensorLifecycleState.ACTIVE
                    return tensor
            
            if self._state == TensorLifecycleState.DETACHED:
                raise TensorError("Tensor has been detached", tensor_id=self._descriptor.tensor_id)
            
            self._state = TensorLifecycleState.MATERIALIZING
            
            self._segment.madvise_willneed(self._offset, self._descriptor.aligned_byte_size)
            
            data = self._segment.read_vectorized(self._offset, self._descriptor.raw_byte_size)
            
            if not self._codec.validate_integrity_fast(data, self._descriptor.checksum):
                self._state = TensorLifecycleState.CORRUPTED
                raise TensorCorruption("Tensor data integrity check failed", tensor_id=self._descriptor.tensor_id)
            
            tensor = self._codec.decode_parallel(data, self._descriptor)
            self._weak_tensor = weakref.ref(tensor, self._on_tensor_deleted)
            self._state = TensorLifecycleState.ACTIVE
            
            return tensor
    
    def persist_optimized(self, tensor: torch.Tensor) -> None:
        """Persist tensor to storage with optimizations."""
        with self._lock:
            if self._state == TensorLifecycleState.DETACHED:
                raise TensorError("Cannot persist to detached tensor", tensor_id=self._descriptor.tensor_id)
            
            self._state = TensorLifecycleState.PERSISTING
            
            new_checksum = self._codec.compute_checksum_simd(tensor)
            
            encoded = self._codec.encode_parallel(tensor, self._descriptor.compression_type)
            
            if len(encoded) > self._descriptor.aligned_byte_size:
                raise TensorError("Encoded tensor exceeds allocated space", tensor_id=self._descriptor.tensor_id)
            
            self._segment.write_vectorized(self._offset, encoded)
            
            self._descriptor = dataclass_replace(self._descriptor, checksum=new_checksum)
            
            self._weak_tensor = weakref.ref(tensor, self._on_tensor_deleted)
            self._state = TensorLifecycleState.ACTIVE
    
    def detach_gracefully(self) -> None:
        """Gracefully detach the tensor reference."""
        with self._lock:
            self._state = TensorLifecycleState.DETACHING
            self._weak_tensor = None
            self._finalizer.detach()
            self._state = TensorLifecycleState.DETACHED
    
    def _on_tensor_deleted(self, ref: weakref.ReferenceType) -> None:
        """Callback when the tensor is garbage collected."""
        with self._lock:
            if self._state == TensorLifecycleState.ACTIVE:
                self._state = TensorLifecycleState.CACHED
    
    @staticmethod
    def _cleanup_resources(segment: IMemorySegment, offset: MemoryOffset) -> None:
        """Cleanup resources when the reference is finalized."""
        try:
            segment.madvise_dontneed(offset, ByteSize(1))
        except Exception:
            pass


class SharedTensor(Generic[TensorT]):
    """High-performance shared tensor with advanced caching and metrics."""
    
    __slots__ = ('_reference', '_lock', '_performance_tracker')
    
    def __init__(self, reference: TensorReference):
        self._reference = reference
        self._lock = RLock()
        self._performance_tracker = {
            'get_calls': 0, 
            'set_calls': 0, 
            'total_time': 0.0
        }
    
    @property
    def tensor_id(self) -> TensorID:
        """Get the tensor ID."""
        return self._reference.descriptor.tensor_id
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the tensor shape."""
        return self._reference.descriptor.shape
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the tensor data type."""
        return self._reference.descriptor.dtype
    
    @property
    def device(self) -> torch.device:
        """Get the tensor device."""
        return self._reference.descriptor.device
    
    @property
    def is_valid(self) -> bool:
        """Check if the tensor is in a valid state."""
        return self._reference.is_valid
    
    @property
    def access_metrics(self) -> Dict[str, Any]:
        """Get access metrics for this tensor."""
        return {
            'access_count': self._reference.access_count,
            'last_access': self._reference.last_access,
            'performance': self._performance_tracker.copy()
        }
    
    def get(self) -> TensorT:
        """Get the tensor with ultra-fast access."""
        start = time.perf_counter()
        try:
            with self._lock:
                tensor = cast(TensorT, self._reference.materialize_optimized())
                self._performance_tracker['get_calls'] += 1
                return tensor
        finally:
            self._performance_tracker['total_time'] += time.perf_counter() - start
    
    def set(self, tensor: TensorT) -> None:
        """Set the tensor with ultra-fast persistence."""
        start = time.perf_counter()
        try:
            with self._lock:
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("Expected torch.Tensor")
                self._reference.persist_optimized(tensor)
                self._performance_tracker['set_calls'] += 1
        finally:
            self._performance_tracker['total_time'] += time.perf_counter() - start
    
    def detach_optimized(self) -> None:
        """Detach the tensor with cleanup."""
        with self._lock:
            self._reference.detach_gracefully()
    
    def prefetch_async(self) -> None:
        """Asynchronously prefetch tensor data."""
        descriptor = self._reference.descriptor
        self._reference._segment.prefetch(
            self._reference._offset, 
            descriptor.aligned_byte_size
        )
    
    def get_descriptor(self) -> TensorDescriptor:
        """Get the tensor descriptor."""
        return self._reference.descriptor
    
    def get_state(self) -> TensorLifecycleState:
        """Get the current lifecycle state."""
        return self._reference.state
    
    def __enter__(self) -> SharedTensor[TensorT]:
        """Context manager entry."""
        return self
    
    def __exit__(self, *args) -> None:
        """Context manager exit with cleanup."""
        self.detach_optimized()
    
    def __repr__(self) -> str:
        """String representation of the shared tensor."""
        return (
            f"SharedTensor(id={self.tensor_id}, shape={self.shape}, "
            f"dtype={self.dtype}, device={self.device}, valid={self.is_valid})"
        )