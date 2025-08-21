from __future__ import annotations

from typing import Dict, List, Optional

from .allocators import MemoryAllocator


class SlabAllocator(MemoryAllocator):
    __slots__ = ('_slab_size', '_slabs', '_free_slabs')
    
    def __init__(self, total_size: int, slab_size: int):
        super().__init__(total_size)
        self._slab_size = slab_size
        self._slabs: Dict[int, List[int]] = {}
        self._free_slabs: Dict[int, List[int]] = {}
        
        slab_count = total_size // slab_size
        for i in range(slab_count):
            offset = i * slab_size
            if slab_size not in self._free_slabs:
                self._free_slabs[slab_size] = []
            self._free_slabs[slab_size].append(offset)
    
    def allocate(self, size: int, alignment: int = 256) -> Optional[int]:
        slab_size = self._find_suitable_slab_size(size)
        
        with self._lock:
            if slab_size not in self._free_slabs or not self._free_slabs[slab_size]:
                return None
            
            offset = self._free_slabs[slab_size].pop()
            self._allocated_blocks[offset] = slab_size
            self._allocated_size += slab_size
            
            return offset
    
    def deallocate(self, ptr: int) -> bool:
        with self._lock:
            if ptr not in self._allocated_blocks:
                return False
            
            slab_size = self._allocated_blocks.pop(ptr)
            self._allocated_size -= slab_size
            
            if slab_size not in self._free_slabs:
                self._free_slabs[slab_size] = []
            self._free_slabs[slab_size].append(ptr)
            
            return True
    
    def _find_suitable_slab_size(self, size: int) -> int:
        return min(slab_size for slab_size in self._free_slabs.keys() if slab_size >= size)