from __future__ import annotations
import time
import weakref
from dataclasses import replace as dataclass_replace
from threading import RLock
from typing import Optional
from weakref import finalize

import torch

from ..types.aliases import TensorID, MemoryOffset, ByteSize
from ..types.descriptors import TensorDescriptor
from ..types.enums import TensorLifecycleState
from ..types.protocols import IMemorySegment, ICodec
from ..exceptions import TensorError, TensorCorruption


class TensorReference:
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
        return self._descriptor
    
    @property
    def state(self) -> TensorLifecycleState:
        return self._state
    
    @property
    def is_valid(self) -> bool:
        return self._state in (TensorLifecycleState.ACTIVE, TensorLifecycleState.CACHED)
    
    @property
    def access_count(self) -> int:
        return self._access_count
    
    @property
    def last_access(self) -> float:
        return self._last_access
    
    def materialize(self) -> torch.Tensor:
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
    
    def persist(self, tensor: torch.Tensor) -> None:
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
    
    def detach(self) -> None:
        with self._lock:
            self._state = TensorLifecycleState.DETACHING
            self._weak_tensor = None
            self._finalizer.detach()
            self._state = TensorLifecycleState.DETACHED
    
    def _on_tensor_deleted(self, ref: weakref.ReferenceType) -> None:
        with self._lock:
            if self._state == TensorLifecycleState.ACTIVE:
                self._state = TensorLifecycleState.CACHED
    
    @staticmethod
    def _cleanup_resources(segment: IMemorySegment, offset: MemoryOffset) -> None:
        try:
            segment.madvise_dontneed(offset, ByteSize(1))
        except Exception:
            pass