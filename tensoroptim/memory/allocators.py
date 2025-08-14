from __future__ import annotations
from collections import deque
from threading import RLock
from typing import Set, Tuple, Dict

from ..types.aliases import MemoryOffset, ByteSize
from ..types.protocols import IMemorySegment, IAllocator
from ..types.enums import AllocationStrategy
from ..exceptions import AllocationFailure


class SlabAllocator:
    """Slab allocator for efficient fixed-size object allocation."""
    
    __slots__ = (
        '_segment', '_slab_size', '_object_size', '_objects_per_slab',
        '_free_objects', '_partial_slabs', '_full_slabs', '_empty_slabs',
        '_lock', '_allocation_count', '_deallocation_count'
    )
    
    def __init__(
        self, 
        segment: IMemorySegment, 
        object_size: ByteSize, 
        slab_size: ByteSize = ByteSize(1024 * 1024)
    ):
        self._segment = segment
        self._slab_size = slab_size
        self._object_size = object_size
        self._objects_per_slab = max(1, slab_size // object_size)
        
        self._free_objects: deque[MemoryOffset] = deque()
        self._partial_slabs: Set[MemoryOffset] = set()
        self._full_slabs: Set[MemoryOffset] = set()
        self._empty_slabs: Set[MemoryOffset] = set()
        
        self._lock = RLock()
        self._allocation_count = 0
        self._deallocation_count = 0
        
        initial_slab = self._create_slab()
        self._partial_slabs.add(initial_slab)
    
    def _create_slab(self) -> MemoryOffset:
        """Create a new slab and populate it with free objects."""
        total_slabs = len(self._partial_slabs) + len(self._full_slabs) + len(self._empty_slabs)
        slab_offset = MemoryOffset(total_slabs * self._slab_size)
        
        if slab_offset + self._slab_size > self._segment.size:
            raise AllocationFailure("Cannot create new slab: would exceed segment size")
        
        for i in range(self._objects_per_slab):
            object_offset = MemoryOffset(slab_offset + i * self._object_size)
            self._free_objects.append(object_offset)
        
        return slab_offset
    
    def allocate_aligned(
        self, 
        size: ByteSize, 
        alignment: int = 64, 
        numa_node: int = -1
    ) -> Tuple[MemoryOffset, IMemorySegment]:
        """Allocate aligned memory from the slab."""
        if size > self._object_size:
            raise AllocationFailure(f"Size {size} exceeds slab object size {self._object_size}")
        
        with self._lock:
            if not self._free_objects:
                if self._empty_slabs:
                    slab_offset = self._empty_slabs.pop()
                    self._partial_slabs.add(slab_offset)
                    for i in range(self._objects_per_slab):
                        object_offset = MemoryOffset(slab_offset + i * self._object_size)
                        self._free_objects.append(object_offset)
                else:
                    slab_offset = self._create_slab()
                    self._partial_slabs.add(slab_offset)
            
            if not self._free_objects:
                raise AllocationFailure("No free objects available after slab creation")
            
            offset = self._free_objects.popleft()
            self._allocation_count += 1
            
            return offset, self._segment
    
    def deallocate_fast(self, offset: MemoryOffset, segment: IMemorySegment) -> None:
        """Fast deallocation of memory back to the slab."""
        with self._lock:
            self._free_objects.append(offset)
            self._deallocation_count += 1
            
            slab_offset = MemoryOffset((offset // self._slab_size) * self._slab_size)
            
            free_count_in_slab = sum(1 for obj in self._free_objects 
                                   if slab_offset <= obj < slab_offset + self._slab_size)
            
            if free_count_in_slab == self._objects_per_slab:
                if slab_offset in self._partial_slabs:
                    self._partial_slabs.remove(slab_offset)
                    self._empty_slabs.add(slab_offset)
                elif slab_offset in self._full_slabs:
                    self._full_slabs.remove(slab_offset)
                    self._empty_slabs.add(slab_offset)
    
    def reallocate(
        self, 
        offset: MemoryOffset, 
        old_size: ByteSize, 
        new_size: ByteSize, 
        segment: IMemorySegment
    ) -> MemoryOffset:
        """Reallocate memory to a new size."""
        if new_size <= self._object_size:
            return offset
        
        new_offset, _ = self.allocate_aligned(new_size)
        
        data = segment.read_vectorized(offset, min(old_size, new_size))
        segment.write_vectorized(new_offset, data)
        
        self.deallocate_fast(offset, segment)
        return new_offset
    
    def defragment_concurrent(self) -> int:
        """Perform concurrent defragmentation (no-op for slab allocator)."""
        return 0
    
    def get_fragmentation_ratio(self) -> float:
        """Get the current fragmentation ratio."""
        with self._lock:
            total_slabs = len(self._partial_slabs) + len(self._full_slabs) + len(self._empty_slabs)
            if total_slabs == 0:
                return 0.0
            return len(self._partial_slabs) / total_slabs
    
    def get_utilization_stats(self) -> Dict[str, float]:
        """Get memory utilization statistics."""
        with self._lock:
            total_objects = (len(self._partial_slabs) + len(self._full_slabs) + len(self._empty_slabs)) * self._objects_per_slab
            allocated_objects = self._allocation_count - self._deallocation_count
            
            return {
                'utilization': allocated_objects / total_objects if total_objects > 0 else 0.0,
                'fragmentation': self.get_fragmentation_ratio(),
                'allocation_count': float(self._allocation_count),
                'deallocation_count': float(self._deallocation_count),
                'total_slabs': float(len(self._partial_slabs) + len(self._full_slabs) + len(self._empty_slabs)),
                'partial_slabs': float(len(self._partial_slabs)),
                'full_slabs': float(len(self._full_slabs)),
                'empty_slabs': float(len(self._empty_slabs))
            }
    
    def set_allocation_strategy(self, strategy: AllocationStrategy) -> None:
        """Set the allocation strategy (no-op for slab allocator)."""
        pass
    
    def enable_compaction(self, enable: bool) -> None:
        """Enable or disable memory compaction (no-op for slab allocator)."""
        pass
    
    @property
    def object_size(self) -> ByteSize:
        """Get the object size for this slab allocator."""
        return self._object_size
    
    @property
    def slab_size(self) -> ByteSize:
        """Get the slab size for this allocator."""
        return self._slab_size
    
    @property
    def objects_per_slab(self) -> int:
        """Get the number of objects per slab."""
        return self._objects_per_slab