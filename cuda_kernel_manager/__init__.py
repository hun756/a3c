from __future__ import annotations

from .exceptions import (
    AllocationError,
    CompileError,
    CUDAError,
    DeviceError,
    ExecutionError,
    GraphError,
    KernelError,
    MemoryError,
    OptimizationError,
    ProfilerError,
    StreamError,
    SynchronizationError,
    TopologyError,
)

__version__ = "3.0.0"

__all__ = [
    "AllocationError",
    "CompileError", 
    "CUDAError",
    "DeviceError",
    "ExecutionError",
    "GraphError",
    "KernelError",
    "MemoryError",
    "OptimizationError",
    "ProfilerError",
    "StreamError",
    "SynchronizationError",
    "TopologyError",
]