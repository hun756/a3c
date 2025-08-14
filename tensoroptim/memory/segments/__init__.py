from .hugepages import HugePagesSegment
from .posix_shm import PosixShmSegment
from .numa_aware import NumaAwareSegment
from .optimized_buffer import OptimizedBuffer

# Alias for backward compatibility
NumaAwareSharedMemory = NumaAwareSegment

__all__ = [
    "HugePagesSegment",
    "PosixShmSegment", 
    "NumaAwareSegment",
    "OptimizedBuffer",
    "NumaAwareSharedMemory",
]