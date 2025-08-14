"""
Performance profiler implementation for TensorOptim library.

This module provides comprehensive performance profiling capabilities
for monitoring tensor operations and memory usage.
"""

from __future__ import annotations
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional, Any, Iterator

# Optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class OperationProfile:
    """Profile data for a single operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: int = 0
    memory_after: int = 0
    memory_peak: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_delta(self) -> int:
        """Get memory usage change."""
        return self.memory_after - self.memory_before


@dataclass
class AggregatedProfile:
    """Aggregated profile statistics for an operation type."""
    operation_name: str
    call_count: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    total_memory_delta: int = 0
    peak_memory_usage: int = 0
    
    def update(self, profile: OperationProfile) -> None:
        """Update aggregated statistics with a new profile."""
        self.call_count += 1
        self.total_duration += profile.duration
        self.min_duration = min(self.min_duration, profile.duration)
        self.max_duration = max(self.max_duration, profile.duration)
        self.avg_duration = self.total_duration / self.call_count
        self.total_memory_delta += profile.memory_delta
        self.peak_memory_usage = max(self.peak_memory_usage, profile.memory_peak)


class PerformanceProfiler:
    """Ultra-high performance profiler for tensor operations."""
    
    def __init__(self, max_profiles: int = 10000, enable_memory_tracking: bool = True):
        self._max_profiles = max_profiles
        self._enable_memory_tracking = enable_memory_tracking and HAS_PSUTIL
        
        self._profiles: deque[OperationProfile] = deque(maxlen=max_profiles)
        self._aggregated: Dict[str, AggregatedProfile] = defaultdict(
            lambda: AggregatedProfile("")
        )
        self._active_operations: Dict[str, float] = {}
        self._lock = RLock()
        
        if self._enable_memory_tracking:
            self._process = psutil.Process()
    
    @contextmanager
    def profile_operation(
        self, 
        operation_name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Iterator[OperationProfile]:
        """Context manager for profiling operations."""
        start_time = time.perf_counter()
        memory_before = self._get_memory_usage() if self._enable_memory_tracking else 0
        
        profile = OperationProfile(
            operation_name=operation_name,
            start_time=start_time,
            end_time=0.0,
            duration=0.0,
            memory_before=memory_before,
            metadata=metadata or {}
        )
        
        try:
            yield profile
        finally:
            end_time = time.perf_counter()
            memory_after = self._get_memory_usage() if self._enable_memory_tracking else 0
            
            profile.end_time = end_time
            profile.duration = end_time - start_time
            profile.memory_after = memory_after
            profile.memory_peak = max(memory_before, memory_after)
            
            with self._lock:
                self._profiles.append(profile)
                
                if operation_name not in self._aggregated:
                    self._aggregated[operation_name] = AggregatedProfile(operation_name)
                self._aggregated[operation_name].update(profile)
    
    def start_operation(self, operation_name: str) -> None:
        """Start timing an operation."""
        with self._lock:
            self._active_operations[operation_name] = time.perf_counter()
    
    def end_operation(
        self, 
        operation_name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[OperationProfile]:
        """End timing an operation and record profile."""
        with self._lock:
            if operation_name not in self._active_operations:
                return None
            
            start_time = self._active_operations.pop(operation_name)
            end_time = time.perf_counter()
            
            profile = OperationProfile(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                metadata=metadata or {}
            )
            
            self._profiles.append(profile)
            
            if operation_name not in self._aggregated:
                self._aggregated[operation_name] = AggregatedProfile(operation_name)
            self._aggregated[operation_name].update(profile)
            
            return profile
    
    def get_profiles(
        self, 
        operation_name: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> List[OperationProfile]:
        """Get recorded profiles, optionally filtered by operation name."""
        with self._lock:
            profiles = list(self._profiles)
            
            if operation_name:
                profiles = [p for p in profiles if p.operation_name == operation_name]
            
            if limit:
                profiles = profiles[-limit:]
            
            return profiles
    
    def get_aggregated_stats(
        self, 
        operation_name: Optional[str] = None
    ) -> Dict[str, AggregatedProfile]:
        """Get aggregated statistics for operations."""
        with self._lock:
            if operation_name:
                return {operation_name: self._aggregated.get(operation_name)}
            return dict(self._aggregated)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of profiling data."""
        with self._lock:
            total_profiles = len(self._profiles)
            total_operations = len(self._aggregated)
            
            if total_profiles == 0:
                return {
                    'total_profiles': 0,
                    'total_operations': 0,
                    'summary': {}
                }
            
            total_duration = sum(p.duration for p in self._profiles)
            avg_duration = total_duration / total_profiles
            
            slowest_ops = sorted(
                self._aggregated.values(),
                key=lambda x: x.avg_duration,
                reverse=True
            )[:5]
            
            frequent_ops = sorted(
                self._aggregated.values(),
                key=lambda x: x.call_count,
                reverse=True
            )[:5]
            
            return {
                'total_profiles': total_profiles,
                'total_operations': total_operations,
                'total_duration': total_duration,
                'avg_duration': avg_duration,
                'slowest_operations': [
                    {
                        'name': op.operation_name,
                        'avg_duration': op.avg_duration,
                        'call_count': op.call_count
                    }
                    for op in slowest_ops
                ],
                'most_frequent_operations': [
                    {
                        'name': op.operation_name,
                        'call_count': op.call_count,
                        'total_duration': op.total_duration
                    }
                    for op in frequent_ops
                ]
            }
    
    def clear_profiles(self) -> None:
        """Clear all recorded profiles."""
        with self._lock:
            self._profiles.clear()
            self._aggregated.clear()
            self._active_operations.clear()
    
    def export_profiles(self, format: str = 'dict') -> Any:
        """Export profiles in specified format."""
        with self._lock:
            profiles_data = [
                {
                    'operation_name': p.operation_name,
                    'start_time': p.start_time,
                    'end_time': p.end_time,
                    'duration': p.duration,
                    'memory_before': p.memory_before,
                    'memory_after': p.memory_after,
                    'memory_delta': p.memory_delta,
                    'metadata': p.metadata
                }
                for p in self._profiles
            ]
            
            if format == 'dict':
                return profiles_data
            elif format == 'json':
                import json
                return json.dumps(profiles_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def record_operation_detailed(
        self, 
        operation: str, 
        duration: float, 
        bytes_processed: int = 0,
        error: bool = False
    ) -> None:
        """Record detailed operation metrics."""
        profile = OperationProfile(
            operation_name=operation,
            start_time=0.0,
            end_time=duration,
            duration=duration,
            metadata={'bytes_processed': bytes_processed, 'error': error}
        )
        
        with self._lock:
            self._profiles.append(profile)
            
            if operation not in self._aggregated:
                self._aggregated[operation] = AggregatedProfile(operation)
            self._aggregated[operation].update(profile)
    
    def record_allocation(self, size: int, allocation_type: str) -> None:
        """Record memory allocation."""
        pass
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        if not self._enable_memory_tracking:
            return 0
        
        try:
            return self._process.memory_info().rss
        except Exception:
            return 0
    
    @property
    def is_memory_tracking_enabled(self) -> bool:
        """Check if memory tracking is enabled."""
        return self._enable_memory_tracking
    
    @property
    def profile_count(self) -> int:
        """Get the number of recorded profiles."""
        return len(self._profiles)
    
    @property
    def operation_count(self) -> int:
        """Get the number of unique operations."""
        return len(self._aggregated)