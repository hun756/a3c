from __future__ import annotations

from abc import ABC, abstractmethod
from threading import RLock
from typing import Generic, List, Optional, Tuple, TypeVar
from weakref import finalize

from ..types.aliases import DeviceID, MemoryPtr, StreamHandle
from ..types.enums import MemoryType
from .allocators import MemoryAllocator

T = TypeVar('T')


class AbstractMemory(ABC, Generic[T]):
    __slots__ = ('_size', '_memory_type', '_device_id', '_ptr', '_allocated', 
                 '_lock', '_alignment', '_ref_count', '_finalizer', '_allocator', '_access_pattern')
    
    def __init__(self, size: int, memory_type: MemoryType, device_id: Optional[DeviceID] = None, 
                 alignment: int = 256, allocator: Optional[MemoryAllocator] = None):
        self._size = size
        self._memory_type = memory_type
        self._device_id = device_id
        self._ptr: Optional[MemoryPtr] = None
        self._allocated = False
        self._lock = RLock()
        self._alignment = alignment
        self._ref_count = 0
        self._finalizer: Optional[finalize] = None
        self._allocator = allocator
        self._access_pattern: List[Tuple[int, int]] = []
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def effective_size(self) -> int:
        return ((self._size + self._alignment - 1) // self._alignment) * self._alignment
    
    @abstractmethod
    def _allocate_impl(self) -> MemoryPtr: ...
    
    @abstractmethod
    def _deallocate_impl(self, ptr: MemoryPtr) -> None: ...
    
    @abstractmethod
    def _prefetch_impl(self, stream_handle: Optional[StreamHandle]) -> None: ...
    
    @abstractmethod
    def _advise_impl(self, advice: str, stream_handle: Optional[StreamHandle]) -> None: ...
    
    def record_access(self, offset: int, size: int) -> None:
        self._access_pattern.append((offset, size))
        if len(self._access_pattern) > 1000:
            self._access_pattern = self._access_pattern[-500:]
    
    def get_access_pattern(self) -> List[Tuple[int, int]]:
        return self._access_pattern.copy()
    
    def prefetch(self, stream: Optional['AbstractStream'] = None) -> None:
        if self._allocated:
            stream_handle = stream.handle if stream else None
            self._prefetch_impl(stream_handle)
    
    def advise_read_mostly(self, stream: Optional['AbstractStream'] = None) -> None:
        stream_handle = stream.handle if stream else None
        self._advise_impl("READ_MOSTLY", stream_handle)
    
    def advise_preferred_location(self, device_id: DeviceID, stream: Optional['AbstractStream'] = None) -> None:
        stream_handle = stream.handle if stream else None
        self._advise_impl(f"PREFERRED_LOCATION_{device_id}", stream_handle)