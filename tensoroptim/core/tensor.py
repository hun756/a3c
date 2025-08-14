from __future__ import annotations
import time
from threading import RLock
from typing import Generic, TypeVar, cast, Tuple, Dict, Any

import torch

from .reference import TensorReference
from ..types.aliases import TensorID
from ..types.enums import TensorLifecycleState
from ..types.descriptors import TensorDescriptor

TensorT = TypeVar('TensorT', bound=torch.Tensor)


class SharedTensor(Generic[TensorT]):
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
        return self._reference.descriptor.tensor_id
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._reference.descriptor.shape
    
    @property
    def dtype(self) -> torch.dtype:
        return self._reference.descriptor.dtype
    
    @property
    def device(self) -> torch.device:
        return self._reference.descriptor.device
    
    @property
    def is_valid(self) -> bool:
        return self._reference.is_valid
    
    @property
    def access_metrics(self) -> Dict[str, Any]:
        return {
            'access_count': self._reference.access_count,
            'last_access': self._reference.last_access,
            'performance': self._performance_tracker.copy()
        }
    
    def get(self) -> TensorT:
        start = time.perf_counter()
        try:
            with self._lock:
                tensor = cast(TensorT, self._reference.materialize())
                self._performance_tracker['get_calls'] += 1
                return tensor
        finally:
            self._performance_tracker['total_time'] += time.perf_counter() - start
    
    def set(self, tensor: TensorT) -> None:
        start = time.perf_counter()
        try:
            with self._lock:
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("Expected torch.Tensor")
                self._reference.persist(tensor)
                self._performance_tracker['set_calls'] += 1
        finally:
            self._performance_tracker['total_time'] += time.perf_counter() - start
    
    def detach(self) -> None:
        with self._lock:
            self._reference.detach()
    
    def prefetch_async(self) -> None:
        descriptor = self._reference.descriptor
        self._reference._segment.prefetch(
            self._reference._offset, 
            descriptor.aligned_byte_size
        )
    
    def get_descriptor(self) -> TensorDescriptor:
        return self._reference.descriptor
    
    def get_state(self) -> TensorLifecycleState:
        return self._reference.state
    
    def __enter__(self) -> SharedTensor[TensorT]:
        return self
    
    def __exit__(self, *args) -> None:
        self.detach()
    
    def __repr__(self) -> str:
        return (
            f"SharedTensor(id={self.tensor_id}, shape={self.shape}, "
            f"dtype={self.dtype}, device={self.device}, valid={self.is_valid})"
        )