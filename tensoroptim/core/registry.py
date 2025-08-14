"""
Tensor registry implementation for TensorOptim library.

This module provides advanced tensor registry with LRU caching,
background cleanup, and comprehensive access statistics.
"""

from __future__ import annotations
import threading
import time
from collections import defaultdict
from heapq import heappop, heappush
from threading import RLock, Event
from typing import Optional, Any, Dict, List, Tuple

from ..types.aliases import TensorID
from ..types.descriptors import TensorDescriptor
from .tensor import TensorReference


class TensorRegistry:
    """Advanced tensor registry with LRU caching and background cleanup."""
    
    __slots__ = (
        '_tensors', '_read_lock', '_write_lock', '_lru_cache', '_access_stats',
        '_cleanup_queue', '_cleanup_threshold', '_cleanup_thread', '_shutdown_event'
    )
    
    def __init__(self, cleanup_threshold: int = 10000, max_cache_size: int = 1000):
        self._tensors: Dict[TensorID, TensorReference] = {}
        self._read_lock = RLock()
        self._write_lock = RLock()
        self._lru_cache: Dict[TensorID, float] = {}
        self._access_stats = defaultdict(int)
        self._cleanup_queue: List[tuple[float, TensorID]] = []
        self._cleanup_threshold = cleanup_threshold
        self._shutdown_event = Event()
        
        self._cleanup_thread = threading.Thread(
            target=self._background_cleanup,
            daemon=True,
            name="TensorRegistryCleanup"
        )
        self._cleanup_thread.start()
    
    def register_optimized(self, reference: TensorReference) -> None:
        """Register a tensor reference with optimized insertion."""
        tensor_id = reference.descriptor.tensor_id
        current_time = time.perf_counter()
        
        with self._write_lock:
            self._tensors[tensor_id] = reference
            self._lru_cache[tensor_id] = current_time
            
            heappush(self._cleanup_queue, (current_time, tensor_id))
            self._access_stats['register'] += 1
            
            if len(self._cleanup_queue) > self._cleanup_threshold:
                self._trigger_cleanup()
    
    def get_optimized(self, tensor_id: TensorID) -> Optional[TensorReference]:
        """Get a tensor reference with optimized lookup."""
        with self._read_lock:
            reference = self._tensors.get(tensor_id)
            if reference:
                self._lru_cache[tensor_id] = time.perf_counter()
                self._access_stats['get_hit'] += 1
            else:
                self._access_stats['get_miss'] += 1
            return reference
    
    def remove_fast(self, tensor_id: TensorID) -> bool:
        """Fast removal of a tensor reference."""
        with self._write_lock:
            removed = self._tensors.pop(tensor_id, None) is not None
            self._lru_cache.pop(tensor_id, None)
            if removed:
                self._access_stats['remove'] += 1
            return removed
    
    def list_active_optimized(self) -> List[TensorDescriptor]:
        """List all active tensor descriptors."""
        with self._read_lock:
            return [
                ref.descriptor for ref in self._tensors.values() 
                if ref.is_valid
            ]
    
    def get_access_statistics(self) -> Dict[str, Any]:
        """Get comprehensive access statistics."""
        with self._read_lock:
            total_requests = self._access_stats['get_hit'] + self._access_stats['get_miss']
            hit_ratio = (
                self._access_stats['get_hit'] / total_requests
                if total_requests > 0 else 0.0
            )
            
            return {
                'total_tensors': len(self._tensors),
                'access_stats': dict(self._access_stats),
                'cleanup_queue_size': len(self._cleanup_queue),
                'cache_hit_ratio': hit_ratio,
                'active_tensors': sum(1 for ref in self._tensors.values() if ref.is_valid),
                'cached_tensors': len(self._lru_cache)
            }
    
    def get_tensor_by_criteria(
        self, 
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[str] = None,
        device: Optional[str] = None
    ) -> List[TensorReference]:
        """Get tensors matching specific criteria."""
        with self._read_lock:
            results = []
            for ref in self._tensors.values():
                if not ref.is_valid:
                    continue
                
                desc = ref.descriptor
                if shape and desc.shape != shape:
                    continue
                if dtype and str(desc.dtype) != dtype:
                    continue
                if device and str(desc.device) != device:
                    continue
                
                results.append(ref)
            
            return results
    
    def cleanup_expired(self, max_age_seconds: float = 7200.0) -> int:
        """Manually cleanup expired tensors."""
        current_time = time.perf_counter()
        cutoff_time = current_time - max_age_seconds
        
        with self._write_lock:
            expired_ids = []
            
            for tensor_id, last_access in self._lru_cache.items():
                if last_access < cutoff_time:
                    ref = self._tensors.get(tensor_id)
                    if ref and (not ref.is_valid or ref.last_access < cutoff_time):
                        expired_ids.append(tensor_id)
            
            for tensor_id in expired_ids:
                self.remove_fast(tensor_id)
            
            self._access_stats['manual_cleanup'] += 1
            return len(expired_ids)
    
    def _trigger_cleanup(self) -> None:
        """Trigger automatic cleanup of old tensors."""
        current_time = time.perf_counter()
        max_age = 7200.0  # 2 hours
        cutoff_time = current_time - max_age
        
        expired_ids = []
        
        while self._cleanup_queue and self._cleanup_queue[0][0] < cutoff_time:
            _, tensor_id = heappop(self._cleanup_queue)
            
            if tensor_id in self._tensors:
                ref = self._tensors[tensor_id]
                if not ref.is_valid or ref.last_access < cutoff_time:
                    expired_ids.append(tensor_id)
        
        for tensor_id in expired_ids:
            self.remove_fast(tensor_id)
        
        self._access_stats['cleanup_runs'] += 1
    
    def _background_cleanup(self) -> None:
        """Background thread for periodic cleanup."""
        while not self._shutdown_event.wait(timeout=300):  # 5 minutes
            try:
                self._trigger_cleanup()
            except Exception:
                # Silently ignore cleanup errors to prevent thread death
                pass
    
    def get_memory_usage_estimate(self) -> Dict[str, int]:
        """Estimate memory usage of registered tensors."""
        with self._read_lock:
            total_bytes = 0
            active_bytes = 0
            tensor_count = 0
            
            for ref in self._tensors.values():
                desc = ref.descriptor
                tensor_bytes = desc.aligned_byte_size
                total_bytes += tensor_bytes
                
                if ref.is_valid:
                    active_bytes += tensor_bytes
                
                tensor_count += 1
            
            return {
                'total_bytes': total_bytes,
                'active_bytes': active_bytes,
                'cached_bytes': total_bytes - active_bytes,
                'tensor_count': tensor_count,
                'avg_tensor_size': total_bytes // tensor_count if tensor_count > 0 else 0
            }
    
    def shutdown(self) -> None:
        """Shutdown the registry and cleanup resources."""
        self._shutdown_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)
        
        # Cleanup all tensors
        with self._write_lock:
            for ref in self._tensors.values():
                try:
                    ref.detach_gracefully()
                except Exception:
                    pass
            
            self._tensors.clear()
            self._lru_cache.clear()
            self._cleanup_queue.clear()
    
    def __len__(self) -> int:
        """Get the number of registered tensors."""
        return len(self._tensors)
    
    def __contains__(self, tensor_id: TensorID) -> bool:
        """Check if a tensor ID is registered."""
        with self._read_lock:
            return tensor_id in self._tensors
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except Exception:
            pass