from __future__ import annotations
import mmap
import os
import tempfile
from typing import Optional

from ...types.aliases import MemoryOffset, ByteSize
from ...exceptions import AllocationFailure


class PosixShmSegment:
    def __init__(self, size: ByteSize, numa_node: int = -1):
        self._size = size
        self._numa_node = numa_node
        self._virtual_address = 0
        self._page_faults = 0
        self._mmap: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self._shm_name: Optional[str] = None
        
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
            self._shm_name = f"/tensoroptim_{os.getpid()}_{id(self)}"
            shm_path = f"/dev/shm{self._shm_name}"
            
            self._fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(self._fd, self._size)
            
            self._mmap = mmap.mmap(
                self._fd,
                self._size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE
            )
            
            self._virtual_address = id(self._mmap)
            
        except Exception as e:
            self._cleanup()
            raise AllocationFailure(f"Failed to create POSIX shared memory segment: {e}")
    
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
        self._cleanup()
    
    def _cleanup(self) -> None:
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        
        if self._fd:
            os.close(self._fd)
            self._fd = None
        
        if self._shm_name:
            try:
                os.unlink(f"/dev/shm{self._shm_name}")
            except OSError:
                pass
            self._shm_name = None