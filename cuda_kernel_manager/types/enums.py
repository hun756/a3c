from __future__ import annotations

from enum import IntEnum


class DeviceType(IntEnum):
    GPU = 0
    CPU = 1
    INTEGRATED = 2
    DPU = 3
    NPU = 4


class MemoryType(IntEnum):
    DEVICE = 0
    HOST = 1
    UNIFIED = 2
    PINNED = 3
    MAPPED = 4
    ZERO_COPY = 5
    WRITE_COMBINED = 6
    CACHED = 7


class StreamType(IntEnum):
    DEFAULT = 0
    NON_BLOCKING = 1
    BLOCKING = 2
    HIGH_PRIORITY = 3
    LOW_PRIORITY = 4
    GRAPHICS = 5
    COMPUTE = 6


class CompileStatus(IntEnum):
    PENDING = 0
    COMPILING = 1
    COMPILED = 2
    FAILED = 3
    CACHED = 4
    OPTIMIZING = 5
    JIT_COMPILING = 6


class ExecutionState(IntEnum):
    IDLE = 0
    RUNNING = 1
    COMPLETED = 2
    ERROR = 3
    QUEUED = 4
    SUSPENDED = 5


class AllocationStrategy(IntEnum):
    BEST_FIT = 0
    FIRST_FIT = 1
    WORST_FIT = 2
    BUDDY_SYSTEM = 3
    SLAB = 4
    POOL = 5


class SchedulingPolicy(IntEnum):
    FIFO = 0
    LIFO = 1
    PRIORITY = 2
    ROUND_ROBIN = 3
    SHORTEST_JOB_FIRST = 4
    WORK_STEALING = 5