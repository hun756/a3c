from __future__ import annotations
import mmap
import os
from typing import Optional

from ...types.aliases import MemoryOffset, ByteSize
from ...exceptions import AllocationFailure


class NumaAwareSegment:
    def __init__(self, size: ByteSize, numa_node: int = -1):
        self._size = size
        self._numa_node = numa_node
        self._virtual_address = 0
        self._page_faults = 0
        self._mmap: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        
        self._create_segment()
    
    @property
    def size(self) -> ByteSize:
        return self._size
    
    @property
    def virtual_address(self) -> int:
        return self._virtual_address
    
    @property
    def numa_node(self) -> int:
        return self._numa_node
    
    def _create_segment(self) -> None:
        try:
            self._fd = os.open('/dev/zero', os.O_RDWR)
            
            self._mmap = mmap.mmap(
                self._fd,
                self._size,
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                mmap.PROT_READ | mmap.PROT_WRITE
            )
            
            self._virtual_address = id(self._mmap)
            
            if self._numa_node >= 0:
                self._bind_to_numa_node()
            
        except Exception as e:
            if self._fd:
                os.close(self._fd)
            raise AllocationFailure(f"Failed to create NUMA-aware segment: {e}")
    
    def _bind_to_numa_node(self) -> None:
        try:
            import ctypes
            import ctypes.util
            
            libc = ctypes.CDLL(ctypes.util.find_library('c'))
            
            MPOL_BIND = 2
            
            nodemask = ctypes.c_ulong(1 << self._numa_node)
            
            result = libc.mbind(
                ctypes.c_void_p(self._virtual_address),
                ctypes.c_size_t(self._size),
                ctypes.c_int(MPOL_BIND),
                ctypes.pointer(nodemask),
                ctypes.c_ulong(64),
                ctypes.c_int(0)
            )
            
            if result != 0:
                pass
                
        except Exception:
            pass
    
    def read_vectorized(self, offset: MemoryOffset, size: ByteSize) -> memoryview:
        if not self._mmap:
            raise RuntimeError("Segment not initialized")
        
        if offset + size > self._size:
            raise ValueError(f"Read beyond segment bounds: {offset + size} > {self._size}")
        
        return memoryview(self._mmap[offset:offset + size])
    
    def write_vectorized(self, offset: MemoryOffset, data: bytes | memoryview) -> None:
        if not self._mmap:
            raise RuntimeError("Segment not initialized")
        
        data_bytes = bytes(data) if isinstance(data, memoryview) else data
        
        if offset + len(data_bytes) > self._size:
            raise ValueError(f"Write beyond segment bounds: {offset + len(data_bytes)} > {self._size}")
        
        self._mmap[offset:offset + len(data_bytes)] = data_bytes
    
    def prefetch(self, offset: MemoryOffset, size: ByteSize) -> None:
        pass
    
    def madvise_sequential(self) -> None:
        if self._mmap and hasattr(mmap, 'MADV_SEQUENTIAL'):
            self._mmap.madvise(mmap.MADV_SEQUENTIAL)
    
    def madvise_random(self) -> None:
        if self._mmap and hasattr(mmap, 'MADV_RANDOM'):
            self._mmap.madvise(mmap.MADV_RANDOM)
    
    def madvise_willneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        if self._mmap and hasattr(mmap, 'MADV_WILLNEED'):
            self._mmap.madvise(mmap.MADV_WILLNEED, offset, size)
    
    def madvise_dontneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        if self._mmap and hasattr(mmap, 'MADV_DONTNEED'):
            self._mmap.madvise(mmap.MADV_DONTNEED, offset, size)
    
    def sync_async(self) -> None:
        if self._mmap:
            self._mmap.flush()
    
    def sync_sync(self) -> None:
        if self._mmap:
            self._mmap.flush()
    
    def get_page_faults(self) -> int:
        return self._page_faults
    
    def close(self) -> None:
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        
        if self._fd:
            os.close(self._fd)
            self._fd = None