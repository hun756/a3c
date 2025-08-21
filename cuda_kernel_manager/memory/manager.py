from __future__ import annotations

import time
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple
from weakref import WeakSet

from ..types.aliases import DeviceID
from ..types.enums import MemoryType


class MemoryManager:
    __slots__ = ('_device_manager', '_allocators', '_memory_pools', '_allocated_memory',
                 '_lock', '_total_allocated', '_peak_allocation', '_allocation_stats',
                 '_garbage_collector', '_memory_advisor')
    
    def __init__(self, device_manager: 'DeviceManager'):
        self._device_manager = device_manager
        self._allocators: Dict[DeviceID, 'DeviceAllocator'] = {}
        self._memory_pools: Dict[Tuple[DeviceID, MemoryType], 'AdvancedMemoryPool'] = {}
        self._allocated_memory: WeakSet['CUDAMemory'] = WeakSet()
        self._lock = RLock()
        self._total_allocated = 0
        self._peak_allocation = 0
        self._allocation_stats: Dict[str, int] = {}
        self._garbage_collector = MemoryGarbageCollector(self)
        self._memory_advisor = MemoryAdvisor()
    
    def allocate_advanced(self, size: int, memory_type: MemoryType = MemoryType.DEVICE,
                         device_id: Optional[DeviceID] = None, alignment: int = 256,
                         use_hugepages: bool = False) -> 'CUDAMemory':
        if device_id is None:
            device_id = self._device_manager.get_optimal_device(size, size)
        
        pool_key = (device_id, memory_type)
        
        with self._lock:
            if pool_key not in self._memory_pools:
                self._memory_pools[pool_key] = AdvancedMemoryPool(memory_type, device_id)
            
            pool = self._memory_pools[pool_key]
            memory = pool.allocate_optimized(size, alignment, use_hugepages)
            
            if memory is None:
                if device_id not in self._allocators:
                    self._allocators[device_id] = DeviceAllocator(self._device_manager)
                
                allocator = self._allocators[device_id]
                memory = allocator.allocate(size, memory_type, device_id, alignment)
            
            if memory:
                self._allocated_memory.add(memory)
                self._total_allocated += size
                self._peak_allocation = max(self._peak_allocation, self._total_allocated)
                
                allocation_key = f"{memory_type.name}_{size//1024}KB"
                self._allocation_stats[allocation_key] = self._allocation_stats.get(allocation_key, 0) + 1
            
            return memory
    
    def trigger_garbage_collection(self) -> bool:
        return self._garbage_collector.collect()
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'total_allocated': self._total_allocated,
                'peak_allocation': self._peak_allocation,
                'active_allocations': len(self._allocated_memory),
                'allocation_stats': self._allocation_stats.copy(),
                'pool_stats': {
                    f"{key[0]}_{key[1].name}": pool.get_statistics()
                    for key, pool in self._memory_pools.items()
                }
            }


class AdvancedMemoryPool:
    __slots__ = ('_memory_type', '_device_id', '_size_classes', '_free_lists',
                 '_allocated_blocks', '_lock', '_statistics', '_peak_usage', '_fragmentation_ratio')
    
    def __init__(self, memory_type: MemoryType, device_id: DeviceID):
        self._memory_type = memory_type
        self._device_id = device_id
        self._size_classes = [256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
        self._free_lists: Dict[int, List['CUDAMemory']] = {size: [] for size in self._size_classes}
        self._allocated_blocks: WeakSet['CUDAMemory'] = WeakSet()
        self._lock = RLock()
        self._statistics = {'allocations': 0, 'deallocations': 0, 'cache_hits': 0, 'cache_misses': 0}
        self._peak_usage = 0
        self._fragmentation_ratio = 0.0

    def allocate_optimized(self, size: int, alignment: int = 256, use_hugepages: bool = False) -> Optional['CUDAMemory']:
        size_class = self._find_size_class(size)
        
        with self._lock:
            if size_class in self._free_lists and self._free_lists[size_class]:
                memory = self._free_lists[size_class].pop()
                self._allocated_blocks.add(memory)
                self._statistics['cache_hits'] += 1
                return memory
            
            memory = CUDAMemory(size_class, self._memory_type, self._device_id, alignment, use_hugepages=use_hugepages)
            memory.allocate()
            
            self._allocated_blocks.add(memory)
            self._statistics['allocations'] += 1
            return memory
    
    def deallocate_optimized(self, memory: 'CUDAMemory') -> None:
        size_class = self._find_size_class(memory.size)
        
        with self._lock:
            if len(self._free_lists[size_class]) < 64:
                self._free_lists[size_class].append(memory)
            else:
                memory.deallocate()
            
            self._allocated_blocks.discard(memory)
            self._statistics['deallocations'] += 1
    
    def _find_size_class(self, size: int) -> int:
        for size_class in self._size_classes:
            if size <= size_class:
                return size_class
        return max(self._size_classes)
    
    def get_statistics(self) -> Dict[str, int]:
        return self._statistics.copy()


class MemoryGarbageCollector:
    __slots__ = ('_memory_manager', '_collection_threshold', '_last_collection', '_forced_collections')
    
    def __init__(self, memory_manager: MemoryManager):
        self._memory_manager = memory_manager
        self._collection_threshold = 0.8
        self._last_collection = time.perf_counter()
        self._forced_collections = 0
    
    def collect(self) -> bool:
        current_time = time.perf_counter()
        
        if current_time - self._last_collection < 1.0:
            return False
        
        collected = 0
        
        for pool in self._memory_manager._memory_pools.values():
            for size_class, free_list in pool._free_lists.items():
                if len(free_list) > 32:
                    excess = len(free_list) - 16
                    for _ in range(excess):
                        if free_list:
                            memory = free_list.pop()
                            memory.deallocate()
                            collected += 1
        
        self._last_collection = current_time
        if collected > 0:
            self._forced_collections += 1
        
        return collected > 0


class MemoryAdvisor:
    __slots__ = ('_access_patterns', '_recommendations')
    
    def __init__(self):
        self._access_patterns: Dict[str, List[Tuple[int, int]]] = {}
        self._recommendations: Dict[str, str] = {}
    
    def analyze_access_pattern(self, memory_id: str, pattern: List[Tuple[int, int]]) -> str:
        self._access_patterns[memory_id] = pattern
        
        if len(pattern) < 10:
            return "INSUFFICIENT_DATA"
        
        sequential_accesses = 0
        random_accesses = 0
        
        for i in range(1, len(pattern)):
            prev_offset, prev_size = pattern[i-1]
            curr_offset, curr_size = pattern[i]
            
            if curr_offset == prev_offset + prev_size:
                sequential_accesses += 1
            else:
                random_accesses += 1
        
        if sequential_accesses > random_accesses * 2:
            recommendation = "PREFETCH_SEQUENTIAL"
        elif random_accesses > sequential_accesses * 2:
            recommendation = "CACHE_FRIENDLY"
        else:
            recommendation = "MIXED_ACCESS"
        
        self._recommendations[memory_id] = recommendation
        return recommendation