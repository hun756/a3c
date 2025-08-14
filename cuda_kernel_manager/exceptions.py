from __future__ import annotations

from typing import Any, Dict, Optional

from .types.aliases import DeviceID


class CUDAError(Exception):
    __slots__ = ('error_code', 'device_id', 'context_info', '_cached_str', 'recovery_hint')
    
    def __init__(self, message: str, error_code: Optional[int] = None, 
                 device_id: Optional[DeviceID] = None, context_info: Optional[Dict[str, Any]] = None,
                 recovery_hint: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.device_id = device_id
        self.context_info = context_info or {}
        self.recovery_hint = recovery_hint
        self._cached_str: Optional[str] = None


class DeviceError(CUDAError): pass
class MemoryError(CUDAError): pass
class KernelError(CUDAError): pass
class CompileError(CUDAError): pass
class StreamError(CUDAError): pass
class ExecutionError(CUDAError): pass
class AllocationError(MemoryError): pass
class SynchronizationError(CUDAError): pass
class TopologyError(CUDAError): pass
class ProfilerError(CUDAError): pass
class GraphError(CUDAError): pass
class OptimizationError(CUDAError): pass


__all__ = [
    'CUDAError',
    'DeviceError',
    'MemoryError',
    'KernelError',
    'CompileError',
    'StreamError',
    'ExecutionError',
    'AllocationError',
    'SynchronizationError',
    'TopologyError',
    'ProfilerError',
    'GraphError',
    'OptimizationError',
]