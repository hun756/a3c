from __future__ import annotations

from abc import ABC, abstractmethod
from threading import RLock
from typing import Dict, List, Optional, Tuple


class MemoryAllocator(ABC):
    __slots__ = ('_total_size', '_allocated_size', '_free_blocks', '_allocated_blocks', '_lock')
    
    def __init__(self, total_size: int):
        self._total_size = total_size
        self._allocated_size = 0
        self._free_blocks: List[Tuple[int, int]] = [(0, total_size)]
        self._allocated_blocks: Dict[int, int] = {}
        self._lock = RLock()
    
    @abstractmethod
    def allocate(self, size: int, alignment: int = 256) -> Optional[int]: ...
    
    @abstractmethod
    def deallocate(self, ptr: int) -> bool: ...
    
    def get_fragmentation_ratio(self) -> float:
        with self._lock:
            if not self._free_blocks:
                return 0.0
            
            total_free = sum(size for _, size in self._free_blocks)
            largest_free = max(size for _, size in self._free_blocks) if self._free_blocks else 0
            
            return 1.0 - (largest_free / total_free) if total_free > 0 else 0.0