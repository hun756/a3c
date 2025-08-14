from __future__ import annotations
from collections import defaultdict
from threading import RLock
from typing import Dict, List, Optional, Set, Tuple, Any
import time

from .reference import TensorReference
from ..types.aliases import TensorID


class TensorRegistry:
    __slots__ = (
        '_tensors', '_by_shape', '_by_dtype', '_by_device', '_lock',
        '_access_counts', '_creation_times', '_last_access_times'
    )
    
    def __init__(self):
        self._tensors: Dict[TensorID, TensorReference] = {}
        self._by_shape: Dict[Tuple[int, ...], Set[TensorID]] = defaultdict(set)
        self._by_dtype: Dict[str, Set[TensorID]] = defaultdict(set)
        self._by_device: Dict[str, Set[TensorID]] = defaultdict(set)
        self._lock = RLock()
        
        self._access_counts: Dict[TensorID, int] = defaultdict(int)
        self._creation_times: Dict[TensorID, float] = {}
        self._last_access_times: Dict[TensorID, float] = {}
    
    def register_tensor(self, tensor_id: TensorID, reference: TensorReference) -> None:
        with self._lock:
            if tensor_id in self._tensors:
                raise ValueError(f"Tensor {tensor_id} already registered")
            
            descriptor = reference.descriptor
            
            self._tensors[tensor_id] = reference
            self._by_shape[descriptor.shape].add(tensor_id)
            self._by_dtype[str(descriptor.dtype)].add(tensor_id)
            self._by_device[str(descriptor.device)].add(tensor_id)
            
            current_time = time.perf_counter()
            self._creation_times[tensor_id] = current_time
            self._last_access_times[tensor_id] = current_time
    
    def unregister_tensor(self, tensor_id: TensorID) -> bool:
        with self._lock:
            if tensor_id not in self._tensors:
                return False
            
            reference = self._tensors[tensor_id]
            descriptor = reference.descriptor
            
            self._by_shape[descriptor.shape].discard(tensor_id)
            self._by_dtype[str(descriptor.dtype)].discard(tensor_id)
            self._by_device[str(descriptor.device)].discard(tensor_id)
            
            if not self._by_shape[descriptor.shape]:
                del self._by_shape[descriptor.shape]
            if not self._by_dtype[str(descriptor.dtype)]:
                del self._by_dtype[str(descriptor.dtype)]
            if not self._by_device[str(descriptor.device)]:
                del self._by_device[str(descriptor.device)]
            
            del self._tensors[tensor_id]
            self._access_counts.pop(tensor_id, None)
            self._creation_times.pop(tensor_id, None)
            self._last_access_times.pop(tensor_id, None)
            
            return True
    
    def get_tensor(self, tensor_id: TensorID) -> Optional[TensorReference]:
        with self._lock:
            reference = self._tensors.get(tensor_id)
            if reference:
                self._access_counts[tensor_id] += 1
                self._last_access_times[tensor_id] = time.perf_counter()
            return reference
    
    def find_tensors_by_criteria(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[str] = None,
        device: Optional[str] = None
    ) -> List[TensorReference]:
        with self._lock:
            candidate_ids = set(self._tensors.keys())
            
            if shape is not None:
                candidate_ids &= self._by_shape.get(shape, set())
            
            if dtype is not None:
                candidate_ids &= self._by_dtype.get(dtype, set())
            
            if device is not None:
                candidate_ids &= self._by_device.get(device, set())
            
            return [self._tensors[tid] for tid in candidate_ids if tid in self._tensors]
    
    def get_all_tensor_ids(self) -> List[TensorID]:
        with self._lock:
            return list(self._tensors.keys())
    
    def get_tensor_count(self) -> int:
        with self._lock:
            return len(self._tensors)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_tensors = len(self._tensors)
            if total_tensors == 0:
                return {
                    'total_tensors': 0,
                    'shapes': {},
                    'dtypes': {},
                    'devices': {},
                    'access_stats': {}
                }
            
            total_accesses = sum(self._access_counts.values())
            avg_accesses = total_accesses / total_tensors if total_tensors > 0 else 0
            
            most_accessed = sorted(
                self._access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            recently_created = sorted(
                self._creation_times.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                'total_tensors': total_tensors,
                'shapes': {str(k): len(v) for k, v in self._by_shape.items()},
                'dtypes': {k: len(v) for k, v in self._by_dtype.items()},
                'devices': {k: len(v) for k, v in self._by_device.items()},
                'access_stats': {
                    'total_accesses': total_accesses,
                    'avg_accesses_per_tensor': avg_accesses,
                    'most_accessed': [
                        {'tensor_id': str(tid), 'access_count': count}
                        for tid, count in most_accessed
                    ],
                    'recently_created': [
                        {'tensor_id': str(tid), 'creation_time': time_val}
                        for tid, time_val in recently_created
                    ]
                }
            }
    
    def cleanup_expired_tensors(self, max_age_seconds: float) -> int:
        with self._lock:
            current_time = time.perf_counter()
            expired_ids = []
            
            for tensor_id, last_access in self._last_access_times.items():
                if current_time - last_access > max_age_seconds:
                    expired_ids.append(tensor_id)
            
            for tensor_id in expired_ids:
                self.unregister_tensor(tensor_id)
            
            return len(expired_ids)
    
    def clear_all(self) -> None:
        with self._lock:
            self._tensors.clear()
            self._by_shape.clear()
            self._by_dtype.clear()
            self._by_device.clear()
            self._access_counts.clear()
            self._creation_times.clear()
            self._last_access_times.clear()
    
    def __len__(self) -> int:
        return len(self._tensors)
    
    def __contains__(self, tensor_id: TensorID) -> bool:
        return tensor_id in self._tensors
    
    def __repr__(self) -> str:
        with self._lock:
            return f"TensorRegistry(tensors={len(self._tensors)})"