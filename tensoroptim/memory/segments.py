"""
Memory segment implementations for TensorOptim library.

This module provides various memory segment implementations including
hugepages, NUMA-aware shared memory, and optimized buffers.
"""

from __future__ import annotations
import ctypes
import mmap
import os
from pathlib import Path
from threading import RLock

from ..types.aliases import MemoryOffset, ByteSize
from ..types.protocols import IMemorySegment

# Optional dependencies
try:
    import numa
    HAS_NUMA = True
except ImportError:
    HAS_NUMA = False


class OptimizedBuffer:
    """High-performance aligned memory buffer with NUMA awareness."""
    
    __slots__ = ('_memory', '_capacity', '_size', '_alignment', '_numa_node', '_readonly', '_lock')
    
    def __init__(self, capacity: ByteSize, alignment: int = 64, numa_node: int = -1, readonly: bool = False):
        self._capacity = capacity
        self._alignment = alignment
        self._numa_node = numa_node
        self._readonly = readonly
        self._size = 0
        self._lock = RLock()
        
        if HAS_NUMA and numa_node >= 0:
            self._memory = self._allocate_numa_aligned(capacity, alignment, numa_node)
        else:
            self._memory = self._allocate_aligned(capacity, alignment)
    
    def _allocate_aligned(self, size: ByteSize, alignment: int) -> ctypes.Array:
        """Allocate aligned memory buffer."""
        aligned_size = ((size + alignment - 1) // alignment) * alignment
        buffer = (ctypes.c_uint8 * aligned_size)()
        
        addr = ctypes.addressof(buffer)
        if addr % alignment != 0:
            offset = alignment - (addr % alignment)
            return (ctypes.c_uint8 * (aligned_size - offset)).from_address(addr + offset)
        return buffer
    
    def _allocate_numa_aligned(self, size: ByteSize, alignment: int, numa_node: int) -> ctypes.Array:
        """Allocate NUMA-aware aligned memory buffer."""
        if not HAS_NUMA:
            return self._allocate_aligned(size, alignment)
        
        old_policy = numa.get_mempolicy()
        numa.set_preferred_node(numa_node)
        try:
            return self._allocate_aligned(size, alignment)
        finally:
            numa.set_mempolicy(old_policy)
    
    @property
    def capacity(self) -> ByteSize:
        """Get buffer capacity."""
        return self._capacity
    
    @property
    def size(self) -> ByteSize:
        """Get current buffer size."""
        return self._size
    
    @property
    def virtual_address(self) -> int:
        """Get virtual address of the buffer."""
        return ctypes.addressof(self._memory)
    
    @property
    def numa_node(self) -> int:
        """Get NUMA node this buffer is bound to."""
        return self._numa_node
    
    def write_at_vectorized(self, offset: MemoryOffset, data: bytes | memoryview) -> None:
        """Write data at specific offset using vectorized operations."""
        if self._readonly:
            raise ValueError("Buffer is read-only")
        
        with self._lock:
            if offset + len(data) > self._capacity:
                raise ValueError("Write exceeds buffer capacity")
            
            src_ptr = ctypes.c_char_p(bytes(data)) if isinstance(data, memoryview) else ctypes.c_char_p(data)
            dst_ptr = ctypes.addressof(self._memory) + offset
            
            ctypes.memmove(dst_ptr, src_ptr, len(data))
            self._size = max(self._size, offset + len(data))
    
    def read_at_vectorized(self, offset: MemoryOffset, size: ByteSize) -> memoryview:
        """Read data at specific offset using vectorized operations."""
        with self._lock:
            if offset + size > self._size:
                raise ValueError("Read exceeds buffer size")
            
            return memoryview((ctypes.c_uint8 * size).from_address(
                ctypes.addressof(self._memory) + offset
            ))
    
    def prefetch_range(self, offset: MemoryOffset, size: ByteSize) -> None:
        """Prefetch memory range into cache."""
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(0, offset, size, os.POSIX_FADV_WILLNEED)
    
    def as_memoryview_readonly(self) -> memoryview:
        """Get read-only memoryview of the buffer."""
        return memoryview(self._memory)[:self._size]


class HugePagesSegment:
    """Memory segment using hugepages for improved performance."""
    
    __slots__ = ('_fd', '_mmap', '_path', '_size', '_address', '_numa_node', '_page_faults')
    
    def __init__(self, path: Path, size: ByteSize, numa_node: int = -1, create: bool = True):
        self._path = path
        self._size = size
        self._numa_node = numa_node
        self._page_faults = 0
        
        if create:
            if os.name == 'nt':  # Win
                self._fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_BINARY, 0o600)
            else:
                self._fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(self._fd, size)
        else:
            if os.name == 'nt':
                self._fd = os.open(path, os.O_RDWR | os.O_BINARY)
            else:
                self._fd = os.open(path, os.O_RDWR)
        
        if os.name == 'nt':  # Win
            self._mmap = mmap.mmap(self._fd, size, access=mmap.ACCESS_WRITE)
        else:  # Unix-like systems
            mmap_flags = mmap.MAP_SHARED
            if hasattr(mmap, 'MAP_HUGETLB'):
                mmap_flags |= mmap.MAP_HUGETLB
            if hasattr(mmap, 'MAP_POPULATE'):
                mmap_flags |= mmap.MAP_POPULATE
            
            self._mmap = mmap.mmap(
                self._fd, size,
                flags=mmap_flags,
                prot=mmap.PROT_READ | mmap.PROT_WRITE
            )
        
        self._address = ctypes.addressof(ctypes.c_char.from_buffer(self._mmap))
        
        if HAS_NUMA and numa_node >= 0:
            self._bind_to_numa_node(numa_node)
    
    def _bind_to_numa_node(self, numa_node: int) -> None:
        """Bind memory pages to specific NUMA node."""
        if HAS_NUMA:
            numa.move_pages(os.getpid(), [self._address], [numa_node])
    
    @property
    def size(self) -> ByteSize:
        """Get segment size."""
        return self._size
    
    @property
    def virtual_address(self) -> int:
        """Get virtual address of the segment."""
        return self._address
    
    @property
    def numa_node(self) -> int:
        """Get NUMA node this segment is bound to."""
        return self._numa_node
    
    def read_vectorized(self, offset: MemoryOffset, size: ByteSize) -> memoryview:
        """Read data using vectorized operations."""
        return memoryview(self._mmap[offset:offset + size])
    
    def write_vectorized(self, offset: MemoryOffset, data: bytes | memoryview) -> None:
        """Write data using vectorized operations."""
        self._mmap[offset:offset + len(data)] = data
    
    def prefetch(self, offset: MemoryOffset, size: ByteSize) -> None:
        """Prefetch data into cache."""
        if hasattr(mmap, 'MADV_WILLNEED'):
            self._mmap.madvise(mmap.MADV_WILLNEED, offset, size)
    
    def madvise_sequential(self) -> None:
        """Advise kernel that access will be sequential."""
        if hasattr(mmap, 'MADV_SEQUENTIAL'):
            self._mmap.madvise(mmap.MADV_SEQUENTIAL)
    
    def madvise_random(self) -> None:
        """Advise kernel that access will be random."""
        if hasattr(mmap, 'MADV_RANDOM'):
            self._mmap.madvise(mmap.MADV_RANDOM)
    
    def madvise_willneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        """Advise kernel that data will be needed soon."""
        if hasattr(mmap, 'MADV_WILLNEED'):
            self._mmap.madvise(mmap.MADV_WILLNEED, offset, size)
    
    def madvise_dontneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        """Advise kernel that data is not needed."""
        if hasattr(mmap, 'MADV_DONTNEED'):
            self._mmap.madvise(mmap.MADV_DONTNEED, offset, size)
    
    def sync_async(self) -> None:
        """Asynchronously sync data to storage."""
        if hasattr(mmap, 'MS_ASYNC'):
            self._mmap.flush(0, 0)
    
    def sync_sync(self) -> None:
        """Synchronously sync data to storage."""
        if hasattr(mmap, 'MS_SYNC'):
            self._mmap.flush()
    
    def get_page_faults(self) -> int:
        """Get the number of page faults for this segment."""
        return self._page_faults
    
    def close(self) -> None:
        """Close and cleanup the memory segment."""
        try:
            if hasattr(self, '_mmap') and self._mmap:
                self._mmap.close()
                self._mmap = None
        except (BufferError, OSError):
            pass
        
        try:
            if hasattr(self, '_fd') and self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
        except OSError:
            pass


class NumaAwareSharedMemory:
    """NUMA-aware shared memory segment."""
    
    __slots__ = ('_shm', '_size', '_address', '_numa_node', '_name')
    
    def __init__(self, name: str, size: ByteSize, numa_node: int = -1, create: bool = True):
        import multiprocessing.shared_memory as shm
        self._name = name
        self._size = size
        self._numa_node = numa_node
        
        if HAS_NUMA and numa_node >= 0:
            old_policy = numa.get_mempolicy()
            numa.set_preferred_node(numa_node)
        
        try:
            self._shm = shm.SharedMemory(name=name, create=create, size=size)
        except FileExistsError:
            self._shm = shm.SharedMemory(name=name, create=False)
        finally:
            if HAS_NUMA and numa_node >= 0:
                numa.set_mempolicy(old_policy)
        
        self._address = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
    
    @property
    def size(self) -> ByteSize:
        """Get segment size."""
        return self._size
    
    @property
    def virtual_address(self) -> int:
        """Get virtual address of the segment."""
        return self._address
    
    @property
    def numa_node(self) -> int:
        """Get NUMA node this segment is bound to."""
        return self._numa_node
    
    def read_vectorized(self, offset: MemoryOffset, size: ByteSize) -> memoryview:
        """Read data using vectorized operations."""
        return memoryview(self._shm.buf[offset:offset + size])
    
    def write_vectorized(self, offset: MemoryOffset, data: bytes | memoryview) -> None:
        """Write data using vectorized operations."""
        self._shm.buf[offset:offset + len(data)] = data
    
    def prefetch(self, offset: MemoryOffset, size: ByteSize) -> None:
        """Prefetch data into cache (no-op for shared memory)."""
        pass
    
    def madvise_sequential(self) -> None:
        """Advise kernel that access will be sequential (no-op for shared memory)."""
        pass
    
    def madvise_random(self) -> None:
        """Advise kernel that access will be random (no-op for shared memory)."""
        pass
    
    def madvise_willneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        """Advise kernel that data will be needed soon (no-op for shared memory)."""
        pass
    
    def madvise_dontneed(self, offset: MemoryOffset, size: ByteSize) -> None:
        """Advise kernel that data is not needed (no-op for shared memory)."""
        pass
    
    def sync_async(self) -> None:
        """Asynchronously sync data to storage (no-op for shared memory)."""
        pass
    
    def sync_sync(self) -> None:
        """Synchronously sync data to storage (no-op for shared memory)."""
        pass
    
    def get_page_faults(self) -> int:
        """Get the number of page faults for this segment."""
        return 0
    
    def close(self) -> None:
        """Close the shared memory segment."""
        try:
            if hasattr(self, '_shm') and self._shm:
                self._shm.close()
                self._shm = None
        except (BufferError, OSError):
            pass
    
    def unlink(self) -> None:
        """Unlink the shared memory segment."""
        if hasattr(self, '_shm'):
            self._shm.unlink()