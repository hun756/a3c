from __future__ import annotations
from functools import lru_cache
from typing import Optional

import torch

from .core.manager import TensorManager
from .types.aliases import ByteSize
from .types.enums import MemoryBackendType, AllocationStrategy


@lru_cache(maxsize=1)
def get_default_manager() -> TensorManager:
    return TensorManager()


def create_manager(**kwargs) -> TensorManager:
    return TensorManager(**kwargs)


def create_optimized_manager_for_cuda() -> TensorManager:
    return TensorManager(
        backend=MemoryBackendType.CUDA_IPC if torch.cuda.is_available() else MemoryBackendType.HUGEPAGES,
        enable_background_optimization=True
    )


def create_high_throughput_manager() -> TensorManager:
    return TensorManager(
        backend=MemoryBackendType.HUGEPAGES,
        max_memory=ByteSize(16 * 1024**3),  # 16GB
        enable_background_optimization=True,
        optimizer_interval=180.0  # 3 minutes
    )


def create_memory_efficient_manager() -> TensorManager:
    return TensorManager(
        backend=MemoryBackendType.POSIX_SHM,
        max_memory=ByteSize(2 * 1024**3),  # 2GB
        enable_background_optimization=True,
        optimizer_interval=120.0  # 2 minutes
    )