from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional

import torch

from .aliases import TensorID, ByteSize
from .enums import CompressionType


@dataclass(frozen=True)
class TensorDescriptor:
    tensor_id: TensorID
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool = False
    compression_type: CompressionType = CompressionType.NONE
    checksum: int = 0
    created_at: float = field(default_factory=time.perf_counter)
    last_accessed: float = field(default_factory=time.perf_counter)
    numa_node: int = -1
    alignment: int = 64
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError(f"Invalid tensor shape: {self.shape}")
        
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise ValueError(f"Alignment must be a positive power of 2: {self.alignment}")
    
    @property
    def numel(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result
    
    @property
    def element_size(self) -> int:
        dtype_sizes = {
            torch.bool: 1,
            torch.uint8: 1,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 4,
            torch.int64: 8,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
            torch.float64: 8,
            torch.complex64: 8,
            torch.complex128: 16,
        }
        return dtype_sizes.get(self.dtype, 4)
    
    @property
    def raw_byte_size(self) -> ByteSize:
        return ByteSize(self.numel * self.element_size)
    
    @property
    def aligned_byte_size(self) -> ByteSize:
        raw_size = self.raw_byte_size
        return ByteSize((raw_size + self.alignment - 1) & ~(self.alignment - 1))
    
    @property
    def is_cuda(self) -> bool:
        return self.device.type == 'cuda'
    
    @property
    def is_cpu(self) -> bool:
        return self.device.type == 'cpu'
    
    def with_checksum(self, checksum: int) -> TensorDescriptor:
        return self.__class__(
            tensor_id=self.tensor_id,
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            compression_type=self.compression_type,
            checksum=checksum,
            created_at=self.created_at,
            last_accessed=time.perf_counter(),
            numa_node=self.numa_node,
            alignment=self.alignment,
            metadata=self.metadata
        )
    
    def with_compression(self, compression_type: CompressionType) -> TensorDescriptor:
        return self.__class__(
            tensor_id=self.tensor_id,
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            compression_type=compression_type,
            checksum=self.checksum,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            numa_node=self.numa_node,
            alignment=self.alignment,
            metadata=self.metadata
        )
    
    def estimate_memory_usage(self) -> dict:
        return {
            'raw_bytes': self.raw_byte_size,
            'aligned_bytes': self.aligned_byte_size,
            'overhead_bytes': self.aligned_byte_size - self.raw_byte_size,
            'compression_type': self.compression_type.name,
            'numa_node': self.numa_node
        }
    
    def __str__(self) -> str:
        return (
            f"TensorDescriptor(id={self.tensor_id}, shape={self.shape}, "
            f"dtype={self.dtype}, device={self.device}, "
            f"size={self.aligned_byte_size} bytes)"
        )