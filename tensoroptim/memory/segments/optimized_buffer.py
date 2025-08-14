from __future__ import annotations
import ctypes
from typing import Optional

from ...types.aliases import MemoryOffset, ByteSize
from ...exceptions import AllocationFailure


class OptimizedBuffer:
    def __init__(self, size: ByteSize, numa_node: int = -1, alignment: int = 64):
        self._size = size
        self._numa_node = numa_node
        self._alignment = alignment
        self._virtual_address = 0
        self._page_faults = 0
        self._buffer: Optional[ctypes.Array] = None
        self._raw_buffer: Optional[ctypes.Array] = None
        
        self._create_buffer()
    
    @property
    def size(self) -> ByteSize:
        return self._size
    
    @property
    def virtual_address(self) -> int:
        return self._virtual_address
    
    @property
    def numa_node(self) -> int:
        return self._numa_node
    
    def _create_buffer(self) -> None:
        try:
            raw_size = self._size + self._alignment
            self._raw_buffer = (ctypes.c_ubyte * raw_size)()
            
            raw_addr = ctypes.addressof(self._raw_buffer)
            aligned_addr = (raw_addr + self._alignment - 1) & ~(self._alignment - 1)
            offset = aligned_addr - raw_addr
            
            self._buffer = (ctypes.c_ubyte * self._size).from_address(aligned_addr)
            self._virtual_address = aligned_addr
            
        except Exception as e:
            raise AllocationFailure(f"Failed to create optimized buffer: {e}")
    
    def read_vectorized(self, offset: MemoryOffset, size: ByteSize) -> memoryview:
        if not self._buffer:
            raise RuntimeError("Buffer not initialized")
        
        if offset + size > self._size:
            raise ValueError(f"Read beyond buffer bounds: {offset + size} > {self._size}")
        
        # Create a bytes object from the ctypes array slice
        data_slice = bytes(self._buffer[offset:offset + size])
        return memoryview(data_slice)
    
    def write_vectorized(self, offset: MemoryOffset, data: bytes | memoryview) -> None:
        if not self._buffer:
            raise RuntimeError("Buffer not initialized")
        
        data_bytes = bytes(data) if isinstance(data, memoryview) else data
        
        if offset + len(data_bytes) > self._size:
            raise ValueError(f"Write beyond buffer bounds: {offset + len(data_bytes)} > {self._size}")
        
        ctypes.memmove(
            ctypes.addressof(self._buffer) + offset,
            data_bytes,
            len(data_bytes)
        )
    
    def prefetch(self, offset: MemoryOffset, size: ByteSize) -> None:
        if not self._buffer:
            return
        
        try:
            import ctypes.util
            libc = ctypes.CDLL(ctypes.util.find_library('c'))
            
            addr = ctypes.addressof(self._buffer) + offset
            libc.__builtin_prefetch(ctypes.c_void_p(addr), 0, 3)
        except Exception:
            pass
    
    def madvise_sequential(self) -> None:
        pass
    
    def madvise_random(self) -> None:
        pass
    
    def madvise_willneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        pass
    
    def madvise_dontneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        pass
    
    def sync_async(self) -> None:
        pass
    
    def sync_sync(self) -> None:
        pass
    
    def get_page_faults(self) -> int:
        return self._page_faults
    
    def close(self) -> None:
        self._buffer = None
        self._raw_buffer = None