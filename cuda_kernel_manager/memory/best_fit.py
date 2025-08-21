from __future__ import annotations

from typing import Optional

from .allocators import MemoryAllocator


class BestFitAllocator(MemoryAllocator):
    def allocate(self, size: int, alignment: int = 256) -> Optional[int]:
        aligned_size = ((size + alignment - 1) // alignment) * alignment
        
        with self._lock:
            best_block = None
            best_index = -1
            
            for i, (offset, block_size) in enumerate(self._free_blocks):
                if block_size >= aligned_size:
                    if best_block is None or block_size < best_block[1]:
                        best_block = (offset, block_size)
                        best_index = i
            
            if best_block is None:
                return None
            
            offset, block_size = best_block
            self._free_blocks.pop(best_index)
            
            if block_size > aligned_size:
                self._free_blocks.append((offset + aligned_size, block_size - aligned_size))
                self._free_blocks.sort()
            
            self._allocated_blocks[offset] = aligned_size
            self._allocated_size += aligned_size
            
            return offset
    
    def deallocate(self, ptr: int) -> bool:
        with self._lock:
            if ptr not in self._allocated_blocks:
                return False
            
            size = self._allocated_blocks.pop(ptr)
            self._allocated_size -= size
            
            self._free_blocks.append((ptr, size))
            self._free_blocks.sort()
            
            self._coalesce_free_blocks()
            return True
    
    def _coalesce_free_blocks(self) -> None:
        if len(self._free_blocks) <= 1:
            return
        
        coalesced = []
        current_offset, current_size = self._free_blocks[0]
        
        for offset, size in self._free_blocks[1:]:
            if current_offset + current_size == offset:
                current_size += size
            else:
                coalesced.append((current_offset, current_size))
                current_offset, current_size = offset, size
        
        coalesced.append((current_offset, current_size))
        self._free_blocks = coalesced