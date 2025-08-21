from __future__ import annotations

from typing import Dict, Optional, Set

from .allocators import MemoryAllocator


class BuddyAllocator(MemoryAllocator):
    __slots__ = ('_min_size', '_max_size', '_free_lists')
    
    def __init__(self, total_size: int, min_size: int = 256):
        super().__init__(total_size)
        self._min_size = min_size
        self._max_size = total_size
        self._free_lists: Dict[int, Set[int]] = {}
        
        size = min_size
        while size <= total_size:
            self._free_lists[size] = set()
            size *= 2
        
        self._free_lists[total_size].add(0)
    
    def allocate(self, size: int, alignment: int = 256) -> Optional[int]:
        block_size = max(self._min_size, 1 << (size - 1).bit_length())
        
        with self._lock:
            for current_size in sorted(self._free_lists.keys()):
                if current_size >= block_size and self._free_lists[current_size]:
                    offset = self._free_lists[current_size].pop()
                    
                    while current_size > block_size:
                        current_size //= 2
                        buddy_offset = offset + current_size
                        self._free_lists[current_size].add(buddy_offset)
                    
                    self._allocated_blocks[offset] = block_size
                    self._allocated_size += block_size
                    return offset
            
            return None
    
    def deallocate(self, ptr: int) -> bool:
        with self._lock:
            if ptr not in self._allocated_blocks:
                return False
            
            block_size = self._allocated_blocks.pop(ptr)
            self._allocated_size -= block_size
            
            while block_size < self._max_size:
                buddy_offset = ptr ^ block_size
                if buddy_offset in self._free_lists[block_size]:
                    self._free_lists[block_size].remove(buddy_offset)
                    ptr = min(ptr, buddy_offset)
                    block_size *= 2
                else:
                    break
            
            self._free_lists[block_size].add(ptr)
            return True